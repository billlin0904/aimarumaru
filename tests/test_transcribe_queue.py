import asyncio
import threading
import time
import unittest

import transcribe_queue as queue_module
from transcribe_queue import (
    FairTranscriptionScheduler,
    TranscriptionCancelled,
    enqueue_transcribe_task,
    register_transcribe_handler,
    start_transcribe_queue,
    stop_transcribe_queue,
)


class FairTranscriptionSchedulerTests(unittest.TestCase):
    def test_turn_is_rotated_to_allow_waiting_job(self) -> None:
        scheduler = FairTranscriptionScheduler()
        scheduler.acquire("long")
        scheduler.release("long")

        short_acquired = threading.Event()
        short_finished = threading.Event()

        def short_job() -> None:
            scheduler.acquire("short")
            short_acquired.set()
            scheduler.release("short", completed=True)
            short_finished.set()

        thread = threading.Thread(target=short_job)
        thread.start()
        time.sleep(0.03)

        # The long job is currently at the head.  Once it yields its next
        # chunk, the waiting short job must become the next turn.
        scheduler.acquire("long")
        scheduler.release("long")

        self.assertTrue(short_acquired.wait(timeout=1.0))
        self.assertTrue(short_finished.wait(timeout=1.0))
        thread.join(timeout=1.0)
        self.assertFalse(thread.is_alive())

    def test_cancelled_wait_does_not_claim_turn(self) -> None:
        scheduler = FairTranscriptionScheduler()
        scheduler.acquire("long")

        with self.assertRaises(TranscriptionCancelled):
            scheduler.acquire("blocked", lambda: True)

        scheduler.abort("blocked")
        scheduler.abort("long")
        scheduler.abort("short")


class TranscribeQueueConcurrencyTests(unittest.IsolatedAsyncioTestCase):
    async def test_remote_jobs_use_separate_worker_slots(self) -> None:
        local_started = 0
        all_local_started = asyncio.Event()
        remote_started = asyncio.Event()
        release = asyncio.Event()

        async def handler(task: dict) -> None:
            nonlocal local_started
            if task.get("worker_group") == "remote_asr":
                remote_started.set()
            else:
                local_started += 1
                if local_started == queue_module.TRANSCRIBE_WORKER_CONCURRENCY:
                    all_local_started.set()
            await release.wait()

        kind = "test-separate-remote-workers"
        register_transcribe_handler(kind, handler)
        await start_transcribe_queue(
            max_size=queue_module.TRANSCRIBE_WORKER_CONCURRENCY + 2
        )
        try:
            for index in range(queue_module.TRANSCRIBE_WORKER_CONCURRENCY):
                enqueue_transcribe_task(
                    {"kind": kind, "id": f"local-{index}"}
                )
            enqueue_transcribe_task(
                {
                    "kind": kind,
                    "id": "remote",
                    "worker_group": "remote_asr",
                }
            )
            await asyncio.wait_for(all_local_started.wait(), timeout=1.0)
            await asyncio.wait_for(remote_started.wait(), timeout=1.0)
        finally:
            release.set()
            assert queue_module.transcribe_queue is not None
            await asyncio.wait_for(queue_module.transcribe_queue.join(), timeout=1.0)
            await stop_transcribe_queue()

    async def test_worker_starts_two_jobs_concurrently(self) -> None:
        if queue_module.TRANSCRIBE_WORKER_CONCURRENCY < 2:
            self.skipTest("requires worker concurrency >= 2")

        started: list[str] = []
        both_started = asyncio.Event()
        release = asyncio.Event()

        async def handler(task: dict) -> None:
            started.append(str(task["id"]))
            if len(started) == 2:
                both_started.set()
            await release.wait()

        kind = "test-concurrent-transcription"
        register_transcribe_handler(kind, handler)
        await start_transcribe_queue(max_size=4)
        try:
            enqueue_transcribe_task({"kind": kind, "id": "long"})
            enqueue_transcribe_task({"kind": kind, "id": "short"})
            await asyncio.wait_for(both_started.wait(), timeout=1.0)
            self.assertCountEqual(started, ["long", "short"])
            release.set()
            assert queue_module.transcribe_queue is not None
            await asyncio.wait_for(queue_module.transcribe_queue.join(), timeout=1.0)
        finally:
            release.set()
            await stop_transcribe_queue()

    async def test_failed_job_does_not_stop_queue_worker(self) -> None:
        completed: list[str] = []

        async def handler(task: dict) -> None:
            if task["id"] == "failed":
                raise RuntimeError("expected test failure")
            completed.append(str(task["id"]))

        kind = "test-failure-does-not-stop-worker"
        register_transcribe_handler(kind, handler)
        await start_transcribe_queue(max_size=4)
        try:
            enqueue_transcribe_task({"kind": kind, "id": "failed"})
            enqueue_transcribe_task({"kind": kind, "id": "next"})
            assert queue_module.transcribe_queue is not None
            await asyncio.wait_for(queue_module.transcribe_queue.join(), timeout=1.0)
            self.assertEqual(completed, ["next"])
            self.assertIsNotNone(queue_module.transcribe_worker_task)
            self.assertFalse(queue_module.transcribe_worker_task.done())
        finally:
            await stop_transcribe_queue()

    async def test_queue_rejects_jobs_above_capacity(self) -> None:
        kind = "test-queue-capacity"

        async def handler(task: dict) -> None:
            await asyncio.sleep(1)

        register_transcribe_handler(kind, handler)
        await start_transcribe_queue(max_size=1)
        try:
            enqueue_transcribe_task({"kind": kind, "id": "first"})
            with self.assertRaises(asyncio.QueueFull):
                enqueue_transcribe_task({"kind": kind, "id": "rejected"})
        finally:
            await stop_transcribe_queue()


if __name__ == "__main__":
    unittest.main()
