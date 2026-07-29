import asyncio
import inspect
import logging
from collections.abc import Awaitable, Callable
from typing import Any, Optional


logger = logging.getLogger(__name__)


TRANSCRIBE_QUEUE_MAX_SIZE = 100
TRANSCRIBE_CLEANUP_INTERVAL_SECONDS = 300

TranscribeTask = dict[str, Any]
TranscribeTaskHandler = Callable[[TranscribeTask], Awaitable[None] | None]
TranscribeCleanupHandler = Callable[[], Awaitable[None] | None]

transcribe_queue: Optional[asyncio.Queue[TranscribeTask]] = None
transcribe_worker_task: Optional[asyncio.Task[Any]] = None
transcribe_cleanup_task: Optional[asyncio.Task[Any]] = None
transcribe_active_count = 0
task_handlers: dict[str, TranscribeTaskHandler] = {}
cleanup_handlers: list[TranscribeCleanupHandler] = []


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
    waiting_count = get_transcribe_queue_size()
    return {
        "queue_size": waiting_count,
        "waiting_count": waiting_count,
        "transcribing_count": transcribe_active_count,
    }


def enqueue_transcribe_task(task: TranscribeTask) -> None:
    if transcribe_queue is None:
        raise RuntimeError("Transcribe queue is not started")
    transcribe_queue.put_nowait(task)


async def start_transcribe_queue(max_size: int = TRANSCRIBE_QUEUE_MAX_SIZE) -> None:
    global transcribe_queue, transcribe_worker_task, transcribe_cleanup_task, transcribe_active_count
    if transcribe_queue is not None:
        return

    transcribe_active_count = 0
    transcribe_queue = asyncio.Queue(maxsize=max_size)
    transcribe_worker_task = asyncio.create_task(transcribe_queue_worker())
    transcribe_cleanup_task = asyncio.create_task(transcribe_cleanup_worker())


async def stop_transcribe_queue() -> None:
    global transcribe_queue, transcribe_worker_task, transcribe_cleanup_task, transcribe_active_count

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


async def transcribe_queue_worker() -> None:
    global transcribe_active_count
    while True:
        assert transcribe_queue is not None
        task = await transcribe_queue.get()
        transcribe_active_count += 1
        try:
            kind = str(task.get("kind", ""))
            handler = task_handlers.get(kind)
            if handler is None:
                logger.warning(
                    "No transcribe task handler registered for kind: %s",
                    kind,
                )
                continue
            await maybe_await(handler(task))
        finally:
            transcribe_active_count = max(0, transcribe_active_count - 1)
            transcribe_queue.task_done()


async def transcribe_cleanup_worker() -> None:
    while True:
        for handler in list(cleanup_handlers):
            try:
                await maybe_await(handler())
            except Exception as exc:
                logger.exception("Transcribe cleanup handler failed: %s", exc)
        await asyncio.sleep(TRANSCRIBE_CLEANUP_INTERVAL_SECONDS)
