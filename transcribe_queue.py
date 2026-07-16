import asyncio
import inspect
from collections.abc import Awaitable, Callable
from typing import Any, Optional


TRANSCRIBE_QUEUE_MAX_SIZE = 100
TRANSCRIBE_CLEANUP_INTERVAL_SECONDS = 300

TranscribeTask = dict[str, Any]
TranscribeTaskHandler = Callable[[TranscribeTask], Awaitable[None] | None]
TranscribeCleanupHandler = Callable[[], Awaitable[None] | None]

transcribe_queue: Optional[asyncio.Queue[TranscribeTask]] = None
transcribe_worker_task: Optional[asyncio.Task[Any]] = None
transcribe_cleanup_task: Optional[asyncio.Task[Any]] = None
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


def enqueue_transcribe_task(task: TranscribeTask) -> None:
    if transcribe_queue is None:
        raise RuntimeError("Transcribe queue is not started")
    transcribe_queue.put_nowait(task)


async def start_transcribe_queue(max_size: int = TRANSCRIBE_QUEUE_MAX_SIZE) -> None:
    global transcribe_queue, transcribe_worker_task, transcribe_cleanup_task
    if transcribe_queue is not None:
        return

    transcribe_queue = asyncio.Queue(maxsize=max_size)
    transcribe_worker_task = asyncio.create_task(transcribe_queue_worker())
    transcribe_cleanup_task = asyncio.create_task(transcribe_cleanup_worker())


async def stop_transcribe_queue() -> None:
    global transcribe_queue, transcribe_worker_task, transcribe_cleanup_task

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


async def transcribe_queue_worker() -> None:
    while True:
        assert transcribe_queue is not None
        task = await transcribe_queue.get()
        try:
            kind = str(task.get("kind", ""))
            handler = task_handlers.get(kind)
            if handler is None:
                print(f"No transcribe task handler registered for kind: {kind}")
                continue
            await maybe_await(handler(task))
        finally:
            transcribe_queue.task_done()


async def transcribe_cleanup_worker() -> None:
    while True:
        for handler in list(cleanup_handlers):
            try:
                await maybe_await(handler())
            except Exception as exc:
                print(f"Transcribe cleanup handler failed: {exc}")
        await asyncio.sleep(TRANSCRIBE_CLEANUP_INTERVAL_SECONDS)
