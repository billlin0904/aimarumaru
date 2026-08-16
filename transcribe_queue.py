import asyncio
import inspect
import logging
import os
import threading
import time
from collections import deque
from collections.abc import Awaitable, Callable
from typing import Any, Optional


logger = logging.getLogger(__name__)


try:
    TRANSCRIBE_QUEUE_MAX_SIZE = max(
        1, int(os.getenv("TRANSCRIBE_QUEUE_MAX_SIZE", "200"))
    )
except ValueError:
    TRANSCRIBE_QUEUE_MAX_SIZE = 200
TRANSCRIBE_CLEANUP_INTERVAL_SECONDS = 300
try:
    TRANSCRIBE_WORKER_CONCURRENCY = max(
        1, int(os.getenv("TRANSCRIBE_WORKER_CONCURRENCY", "2"))
    )
except ValueError:
    TRANSCRIBE_WORKER_CONCURRENCY = 2
try:
    TRANSCRIBE_REMOTE_WORKER_CONCURRENCY = max(
        1, int(os.getenv("TRANSCRIBE_REMOTE_WORKER_CONCURRENCY", "50"))
    )
except ValueError:
    TRANSCRIBE_REMOTE_WORKER_CONCURRENCY = 50

TranscribeTask = dict[str, Any]
TranscribeTaskHandler = Callable[[TranscribeTask], Awaitable[None] | None]
TranscribeCleanupHandler = Callable[[], Awaitable[None] | None]

transcribe_queue: Optional[asyncio.Queue[TranscribeTask]] = None
transcribe_worker_task: Optional[asyncio.Task[Any]] = None
transcribe_cleanup_task: Optional[asyncio.Task[Any]] = None
transcribe_active_count = 0
transcribe_waiting_count = 0
transcribe_active_by_group = {"default": 0, "remote_asr": 0}
transcribe_worker_semaphores: dict[str, asyncio.Semaphore] = {}
task_handlers: dict[str, TranscribeTaskHandler] = {}
cleanup_handlers: list[TranscribeCleanupHandler] = []


class TranscriptionCancelled(Exception):
    """Raised when a queued transcription has been cancelled by its owner."""


class FairTranscriptionScheduler:
    """Round-robin scheduler for the GPU-bound ASR turns of each job.

    Queue workers may run multiple jobs concurrently, but only one local ASR
    inference turn is admitted at a time.  Each completed audio chunk goes to
    the back of the queue, allowing short jobs to make progress while a long
    job is still decoding.
    """

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._order: deque[str] = deque()
        self._active: str | None = None

    def acquire(
        self,
        job_id: str,
        cancel_check: Callable[[], bool] | None = None,
    ) -> float:
        started = time.perf_counter()
        with self._condition:
            if job_id not in self._order and self._active != job_id:
                self._order.append(job_id)
                self._condition.notify_all()
            while self._active is not None or not self._order or self._order[0] != job_id:
                if cancel_check is not None and cancel_check():
                    try:
                        self._order.remove(job_id)
                    except ValueError:
                        pass
                    self._condition.notify_all()
                    raise TranscriptionCancelled("轉譯已取消")
                self._condition.wait(timeout=0.25)
            self._active = job_id
        return (time.perf_counter() - started) * 1000

    def release(self, job_id: str, *, completed: bool = False) -> None:
        with self._condition:
            if self._active == job_id:
                self._active = None
            try:
                self._order.remove(job_id)
            except ValueError:
                pass
            if not completed:
                self._order.append(job_id)
            self._condition.notify_all()

    def abort(self, job_id: str) -> None:
        with self._condition:
            if self._active == job_id:
                self._active = None
            try:
                self._order.remove(job_id)
            except ValueError:
                pass
            self._condition.notify_all()


fair_transcription_scheduler = FairTranscriptionScheduler()


async def maybe_await(result: Awaitable[Any] | Any) -> Any:
    if inspect.isawaitable(result):
        return await result
    return result


def register_transcribe_handler(kind: str, handler: TranscribeTaskHandler) -> None:
    task_handlers[kind] = handler


def register_transcribe_cleanup(handler: TranscribeCleanupHandler) -> None:
    if handler not in cleanup_handlers:
        cleanup_handlers.append(handler)


def is_transcribe_queue_started() -> bool:
    return transcribe_queue is not None


def get_transcribe_queue_size() -> int:
    if transcribe_queue is None:
        return 0
    return transcribe_queue.qsize()


def get_transcribe_queue_counts() -> dict[str, int]:
    queue_size = get_transcribe_queue_size()
    waiting_count = queue_size + transcribe_waiting_count
    return {
        "queue_size": queue_size,
        "waiting_count": waiting_count,
        "transcribing_count": transcribe_active_count,
        "queue_capacity": transcribe_queue.maxsize if transcribe_queue else 0,
        "local_transcribing_count": transcribe_active_by_group["default"],
        "remote_transcribing_count": transcribe_active_by_group["remote_asr"],
        "local_worker_concurrency": TRANSCRIBE_WORKER_CONCURRENCY,
        "remote_worker_concurrency": TRANSCRIBE_REMOTE_WORKER_CONCURRENCY,
    }


def enqueue_transcribe_task(task: TranscribeTask) -> None:
    if transcribe_queue is None:
        raise RuntimeError("Transcribe queue is not started")
    if transcribe_queue.qsize() + transcribe_waiting_count >= transcribe_queue.maxsize:
        logger.warning(
            "Transcribe queue is full: limit=%d kind=%s id=%s",
            transcribe_queue.maxsize,
            task.get("kind", ""),
            task.get("id", ""),
        )
        raise asyncio.QueueFull
    try:
        transcribe_queue.put_nowait(task)
    except asyncio.QueueFull:
        logger.warning(
            "Transcribe queue is full: limit=%d kind=%s id=%s",
            transcribe_queue.maxsize,
            task.get("kind", ""),
            task.get("id", ""),
        )
        raise


def cancel_queued_transcribe_task(kind: str, task_id: str) -> bool:
    """Remove a matching task while preserving the order of all other tasks."""
    if transcribe_queue is None:
        return False

    retained_tasks: list[TranscribeTask] = []
    removed = False
    while True:
        try:
            task = transcribe_queue.get_nowait()
        except asyncio.QueueEmpty:
            break

        transcribe_queue.task_done()
        if (
            not removed
            and str(task.get("kind", "")) == kind
            and str(task.get("id", "")) == task_id
        ):
            removed = True
            continue
        retained_tasks.append(task)

    for task in retained_tasks:
        transcribe_queue.put_nowait(task)
    return removed


async def start_transcribe_queue(max_size: int = TRANSCRIBE_QUEUE_MAX_SIZE) -> None:
    global transcribe_queue, transcribe_worker_task, transcribe_cleanup_task
    global transcribe_active_count, transcribe_waiting_count
    global transcribe_active_by_group, transcribe_worker_semaphores
    if transcribe_queue is not None:
        return

    transcribe_active_count = 0
    transcribe_waiting_count = 0
    transcribe_active_by_group = {"default": 0, "remote_asr": 0}
    transcribe_worker_semaphores = {
        "default": asyncio.Semaphore(TRANSCRIBE_WORKER_CONCURRENCY),
        "remote_asr": asyncio.Semaphore(TRANSCRIBE_REMOTE_WORKER_CONCURRENCY),
    }
    transcribe_queue = asyncio.Queue(maxsize=max_size)
    transcribe_worker_task = asyncio.create_task(transcribe_queue_worker())
    transcribe_cleanup_task = asyncio.create_task(transcribe_cleanup_worker())
    logger.info(
        "Transcribe queue started: max_size=%d local_worker_concurrency=%d "
        "remote_worker_concurrency=%d",
        max_size,
        TRANSCRIBE_WORKER_CONCURRENCY,
        TRANSCRIBE_REMOTE_WORKER_CONCURRENCY,
    )


async def stop_transcribe_queue() -> None:
    global transcribe_queue, transcribe_worker_task, transcribe_cleanup_task
    global transcribe_active_count, transcribe_waiting_count
    global transcribe_active_by_group, transcribe_worker_semaphores

    tasks = [transcribe_worker_task, transcribe_cleanup_task]
    for task in tasks:
        if task:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    transcribe_queue = None
    transcribe_worker_task = None
    transcribe_cleanup_task = None
    transcribe_active_count = 0
    transcribe_waiting_count = 0
    transcribe_active_by_group = {"default": 0, "remote_asr": 0}
    transcribe_worker_semaphores = {}


async def transcribe_queue_worker() -> None:
    global transcribe_active_count, transcribe_waiting_count
    active: set[asyncio.Task[Any]] = set()

    async def run_task(task: TranscribeTask) -> None:
        global transcribe_active_count, transcribe_waiting_count
        worker_group = (
            "remote_asr"
            if str(task.get("worker_group") or "") == "remote_asr"
            else "default"
        )
        semaphore = transcribe_worker_semaphores[worker_group]
        acquired = False
        waiting = True
        try:
            await semaphore.acquire()
            acquired = True
            transcribe_waiting_count = max(0, transcribe_waiting_count - 1)
            waiting = False
            transcribe_active_count += 1
            transcribe_active_by_group[worker_group] += 1
            kind = str(task.get("kind", ""))
            handler = task_handlers.get(kind)
            if handler is None:
                logger.warning(
                    "No transcribe task handler registered for kind: %s",
                    kind,
                )
                return
            await maybe_await(handler(task))
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception(
                "Transcribe task handler failed: kind=%s id=%s",
                task.get("kind", ""),
                task.get("id", ""),
            )
        finally:
            if waiting:
                transcribe_waiting_count = max(0, transcribe_waiting_count - 1)
            if acquired:
                transcribe_active_count = max(0, transcribe_active_count - 1)
                transcribe_active_by_group[worker_group] = max(
                    0,
                    transcribe_active_by_group[worker_group] - 1,
                )
                semaphore.release()
            if transcribe_queue is not None:
                transcribe_queue.task_done()

    try:
        while True:
            assert transcribe_queue is not None
            dispatch_concurrency = max(
                transcribe_queue.maxsize,
                TRANSCRIBE_WORKER_CONCURRENCY
                + TRANSCRIBE_REMOTE_WORKER_CONCURRENCY,
            )
            while len(active) >= dispatch_concurrency:
                done, pending = await asyncio.wait(
                    active,
                    return_when=asyncio.FIRST_COMPLETED,
                )
                active = set(pending)
                for completed in done:
                    try:
                        completed.result()
                    except asyncio.CancelledError:
                        logger.warning("Transcribe task runner was cancelled")
                    except Exception:
                        # A handler failure must never terminate the queue
                        # worker; its job has already been marked failed by
                        # the handler owner and task_done() ran in finally.
                        logger.exception("Transcribe task runner failed")
            task = await transcribe_queue.get()
            transcribe_waiting_count += 1
            active.add(asyncio.create_task(run_task(task)))
    finally:
        for running in active:
            running.cancel()
        if active:
            await asyncio.gather(*active, return_exceptions=True)


async def transcribe_cleanup_worker() -> None:
    while True:
        for handler in list(cleanup_handlers):
            try:
                await maybe_await(handler())
            except Exception as exc:
                logger.exception("Transcribe cleanup handler failed: %s", exc)
        await asyncio.sleep(TRANSCRIBE_CLEANUP_INTERVAL_SECONDS)
