from __future__ import annotations

import os
import threading
import time
from collections import deque
from collections.abc import Callable
from concurrent.futures import Future, InvalidStateError, wait
from dataclasses import dataclass
from typing import Any


def _positive_int_env(name: str, default: int) -> int:
    try:
        return max(1, int(os.getenv(name, str(default))))
    except ValueError:
        return default


CLOUDFLARE_ASR_MAX_IN_FLIGHT = _positive_int_env(
    "CLOUDFLARE_ASR_MAX_IN_FLIGHT",
    50,
)
CLOUDFLARE_ASR_RATE_LIMIT_PER_MINUTE = _positive_int_env(
    "CLOUDFLARE_ASR_RATE_LIMIT_PER_MINUTE",
    600,
)
CLOUDFLARE_ASR_QUEUE_MAX_SIZE = _positive_int_env(
    "CLOUDFLARE_ASR_QUEUE_MAX_SIZE",
    1000,
)


class CloudflareAsrRequestCancelled(Exception):
    """Raised when a queued Cloudflare ASR request is cancelled."""


@dataclass(frozen=True)
class CloudflareAsrResult:
    value: Any
    queue_wait_ms: float


@dataclass
class _CloudflareAsrRequest:
    job_id: str
    invoke: Callable[[], Any]
    cancel_check: Callable[[], bool] | None
    future: Future[CloudflareAsrResult]
    queued_at: float
    dispatched: bool = False
    dispatched_at: float | None = None


class CloudflareAsrScheduler:
    """Process-wide fair scheduler for synchronous Workers AI requests."""

    def __init__(
        self,
        *,
        max_in_flight: int = CLOUDFLARE_ASR_MAX_IN_FLIGHT,
        requests_per_window: int = CLOUDFLARE_ASR_RATE_LIMIT_PER_MINUTE,
        window_seconds: float = 60.0,
        max_queue_size: int = CLOUDFLARE_ASR_QUEUE_MAX_SIZE,
    ) -> None:
        self.max_in_flight = max(1, int(max_in_flight))
        self.requests_per_window = max(1, int(requests_per_window))
        self.window_seconds = max(0.01, float(window_seconds))
        self.max_queue_size = max(1, int(max_queue_size))
        self._condition = threading.Condition()
        self._job_queues: dict[str, deque[_CloudflareAsrRequest]] = {}
        self._job_order: deque[str] = deque()
        self._request_times: deque[float] = deque()
        self._in_flight = 0
        self._running = False
        self._dispatcher: threading.Thread | None = None
        self._completed = 0
        self._failed = 0
        self._cancelled = 0

    def execute(
        self,
        job_id: str,
        invoke: Callable[[], Any],
        cancel_check: Callable[[], bool] | None = None,
    ) -> CloudflareAsrResult:
        normalized_job_id = str(job_id or "anonymous")
        if cancel_check is not None and cancel_check():
            raise CloudflareAsrRequestCancelled("Cloudflare ASR request cancelled")

        request = _CloudflareAsrRequest(
            job_id=normalized_job_id,
            invoke=invoke,
            cancel_check=cancel_check,
            future=Future(),
            queued_at=time.perf_counter(),
        )
        with self._condition:
            self._ensure_started_locked()
            if self._pending_count_locked() >= self.max_queue_size:
                raise RuntimeError("Cloudflare ASR queue is full")
            queue = self._job_queues.get(normalized_job_id)
            if queue is None:
                queue = deque()
                self._job_queues[normalized_job_id] = queue
                self._job_order.append(normalized_job_id)
            queue.append(request)
            self._condition.notify_all()

        while True:
            done, _ = wait((request.future,), timeout=0.25)
            if done:
                return request.future.result(timeout=0.25)
            if cancel_check is not None and cancel_check():
                self._cancel_request(request)
                raise CloudflareAsrRequestCancelled(
                    "Cloudflare ASR request cancelled"
                )

    def cancel_job(self, job_id: str) -> int:
        cancelled: list[_CloudflareAsrRequest] = []
        normalized_job_id = str(job_id or "anonymous")
        with self._condition:
            queue = self._job_queues.pop(normalized_job_id, None)
            if queue:
                cancelled.extend(queue)
            self._remove_job_from_order_locked(normalized_job_id)
            self._cancelled += len(cancelled)
            self._condition.notify_all()
        for request in cancelled:
            self._set_exception(
                request.future,
                CloudflareAsrRequestCancelled("Cloudflare ASR request cancelled"),
            )
        return len(cancelled)

    def snapshot(self) -> dict[str, int]:
        with self._condition:
            now = time.perf_counter()
            self._prune_request_times_locked(now)
            return {
                "pending_requests": self._pending_count_locked(),
                "pending_jobs": len(self._job_queues),
                "in_flight": self._in_flight,
                "requests_in_window": len(self._request_times),
                "max_in_flight": self.max_in_flight,
                "rate_limit_per_minute": self.requests_per_window,
                "queue_capacity": self.max_queue_size,
                "completed_requests": self._completed,
                "failed_requests": self._failed,
                "cancelled_requests": self._cancelled,
            }

    def shutdown(self, *, wait: bool = False) -> None:
        pending: list[_CloudflareAsrRequest] = []
        with self._condition:
            self._running = False
            for queue in self._job_queues.values():
                pending.extend(queue)
            self._job_queues.clear()
            self._job_order.clear()
            self._cancelled += len(pending)
            dispatcher = self._dispatcher
            self._dispatcher = None
            self._condition.notify_all()
        for request in pending:
            self._set_exception(
                request.future,
                CloudflareAsrRequestCancelled("Cloudflare ASR scheduler stopped"),
            )
        if wait and dispatcher is not None:
            dispatcher.join(timeout=2.0)

    def _ensure_started_locked(self) -> None:
        if self._running and self._dispatcher is not None:
            return
        self._running = True
        self._dispatcher = threading.Thread(
            target=self._dispatch_loop,
            name="cloudflare-asr-dispatcher",
            daemon=True,
        )
        self._dispatcher.start()

    def _dispatch_loop(self) -> None:
        while True:
            with self._condition:
                request: _CloudflareAsrRequest | None = None
                while request is None:
                    if not self._running:
                        return
                    self._drop_cancelled_requests_locked()
                    now = time.perf_counter()
                    self._prune_request_times_locked(now)
                    if self._in_flight >= self.max_in_flight or not self._job_order:
                        self._condition.wait(timeout=0.25)
                        continue
                    rate_wait = self._rate_wait_seconds_locked(now)
                    if rate_wait > 0:
                        self._condition.wait(timeout=min(rate_wait, 0.25))
                        continue
                    request = self._pop_next_request_locked()
                    if request is None:
                        continue
                    if (
                        request.cancel_check is not None
                        and request.cancel_check()
                    ):
                        self._cancelled += 1
                        self._set_exception(
                            request.future,
                            CloudflareAsrRequestCancelled(
                                "Cloudflare ASR request cancelled"
                            ),
                        )
                        request = None
                        continue
                    request.dispatched = True
                    request.dispatched_at = now
                    self._in_flight += 1
                    self._request_times.append(now)

            worker = threading.Thread(
                target=self._run_request,
                args=(request,),
                name=f"cloudflare-asr-{request.job_id[:12]}",
                daemon=True,
            )
            worker.start()

    def _run_request(self, request: _CloudflareAsrRequest) -> None:
        try:
            value = request.invoke()
            result = CloudflareAsrResult(
                value=value,
                queue_wait_ms=(
                    (request.dispatched_at or request.queued_at) - request.queued_at
                )
                * 1000,
            )
        except BaseException as exc:
            with self._condition:
                self._failed += 1
            self._set_exception(request.future, exc)
        else:
            with self._condition:
                self._completed += 1
            self._set_result(request.future, result)
        finally:
            with self._condition:
                self._in_flight = max(0, self._in_flight - 1)
                self._condition.notify_all()

    def _cancel_request(self, request: _CloudflareAsrRequest) -> None:
        with self._condition:
            if request.future.done():
                return
            if not request.dispatched:
                queue = self._job_queues.get(request.job_id)
                if queue is not None:
                    try:
                        queue.remove(request)
                    except ValueError:
                        pass
                    if not queue:
                        self._job_queues.pop(request.job_id, None)
                        self._remove_job_from_order_locked(request.job_id)
            self._cancelled += 1
            self._condition.notify_all()
        self._set_exception(
            request.future,
            CloudflareAsrRequestCancelled("Cloudflare ASR request cancelled"),
        )

    def _drop_cancelled_requests_locked(self) -> None:
        cancelled: list[_CloudflareAsrRequest] = []
        for job_id in list(self._job_order):
            queue = self._job_queues.get(job_id)
            if queue is None:
                self._remove_job_from_order_locked(job_id)
                continue
            retained = deque(
                request
                for request in queue
                if not (
                    request.cancel_check is not None
                    and request.cancel_check()
                )
            )
            if len(retained) != len(queue):
                cancelled.extend(request for request in queue if request not in retained)
            if retained:
                self._job_queues[job_id] = retained
            else:
                self._job_queues.pop(job_id, None)
                self._remove_job_from_order_locked(job_id)
        self._cancelled += len(cancelled)
        for request in cancelled:
            self._set_exception(
                request.future,
                CloudflareAsrRequestCancelled("Cloudflare ASR request cancelled"),
            )

    def _pop_next_request_locked(self) -> _CloudflareAsrRequest | None:
        while self._job_order:
            job_id = self._job_order.popleft()
            queue = self._job_queues.get(job_id)
            if not queue:
                self._job_queues.pop(job_id, None)
                continue
            request = queue.popleft()
            if queue:
                self._job_order.append(job_id)
            else:
                self._job_queues.pop(job_id, None)
            return request
        return None

    def _pending_count_locked(self) -> int:
        return sum(len(queue) for queue in self._job_queues.values())

    def _remove_job_from_order_locked(self, job_id: str) -> None:
        self._job_order = deque(
            queued_job_id
            for queued_job_id in self._job_order
            if queued_job_id != job_id
        )

    def _prune_request_times_locked(self, now: float) -> None:
        cutoff = now - self.window_seconds
        while self._request_times and self._request_times[0] <= cutoff:
            self._request_times.popleft()

    def _rate_wait_seconds_locked(self, now: float) -> float:
        if len(self._request_times) < self.requests_per_window:
            return 0.0
        return max(0.0, self._request_times[0] + self.window_seconds - now)

    @staticmethod
    def _set_result(future: Future[CloudflareAsrResult], result: CloudflareAsrResult) -> None:
        if future.done():
            return
        try:
            future.set_result(result)
        except InvalidStateError:
            pass

    @staticmethod
    def _set_exception(future: Future[Any], exc: BaseException) -> None:
        if future.done():
            return
        try:
            future.set_exception(exc)
        except InvalidStateError:
            pass


cloudflare_asr_scheduler = CloudflareAsrScheduler()


def get_cloudflare_asr_queue_status() -> dict[str, int]:
    return cloudflare_asr_scheduler.snapshot()
