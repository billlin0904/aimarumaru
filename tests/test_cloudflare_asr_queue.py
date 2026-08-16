import threading
import time
import unittest

from cloudflare_asr_queue import (
    CloudflareAsrRequestCancelled,
    CloudflareAsrScheduler,
)


class CloudflareAsrSchedulerTests(unittest.TestCase):
    def test_jobs_take_turns_between_chunks(self) -> None:
        scheduler = CloudflareAsrScheduler(
            max_in_flight=1,
            requests_per_window=100,
            window_seconds=1.0,
        )
        first_started = threading.Event()
        release_first = threading.Event()
        invocation_order: list[str] = []

        def first_request() -> str:
            invocation_order.append("a1")
            first_started.set()
            release_first.wait(timeout=1.0)
            return "a1"

        def run_job_a() -> None:
            scheduler.execute("a", first_request)
            scheduler.execute("a", lambda: invocation_order.append("a2"))

        def run_job_b() -> None:
            scheduler.execute("b", lambda: invocation_order.append("b1"))

        thread_a = threading.Thread(target=run_job_a)
        thread_b = threading.Thread(target=run_job_b)
        thread_a.start()
        self.assertTrue(first_started.wait(timeout=1.0))
        thread_b.start()
        deadline = time.monotonic() + 1.0
        while scheduler.snapshot()["pending_requests"] < 1:
            if time.monotonic() >= deadline:
                self.fail("job B did not enter the scheduler queue")
            time.sleep(0.01)
        release_first.set()
        thread_a.join(timeout=1.0)
        thread_b.join(timeout=1.0)
        scheduler.shutdown(wait=True)

        self.assertFalse(thread_a.is_alive())
        self.assertFalse(thread_b.is_alive())
        self.assertEqual(invocation_order, ["a1", "b1", "a2"])

    def test_in_flight_limit_is_shared_by_all_jobs(self) -> None:
        scheduler = CloudflareAsrScheduler(
            max_in_flight=2,
            requests_per_window=100,
            window_seconds=1.0,
        )
        release = threading.Event()
        lock = threading.Lock()
        active = 0
        maximum_active = 0

        def invoke() -> None:
            nonlocal active, maximum_active
            with lock:
                active += 1
                maximum_active = max(maximum_active, active)
            release.wait(timeout=1.0)
            with lock:
                active -= 1

        threads = [
            threading.Thread(target=scheduler.execute, args=(str(index), invoke))
            for index in range(4)
        ]
        for thread in threads:
            thread.start()
        deadline = time.monotonic() + 1.0
        while scheduler.snapshot()["in_flight"] < 2:
            if time.monotonic() >= deadline:
                self.fail("scheduler did not fill its in-flight slots")
            time.sleep(0.01)
        self.assertEqual(scheduler.snapshot()["in_flight"], 2)
        release.set()
        for thread in threads:
            thread.join(timeout=1.0)
        scheduler.shutdown(wait=True)

        self.assertEqual(maximum_active, 2)
        self.assertTrue(all(not thread.is_alive() for thread in threads))

    def test_sliding_window_rate_limit_delays_next_request(self) -> None:
        scheduler = CloudflareAsrScheduler(
            max_in_flight=2,
            requests_per_window=1,
            window_seconds=0.12,
        )
        scheduler.execute("a", lambda: "first")
        started = time.perf_counter()
        scheduler.execute("b", lambda: "second")
        elapsed = time.perf_counter() - started
        scheduler.shutdown(wait=True)

        self.assertGreaterEqual(elapsed, 0.09)

    def test_provider_timeout_is_propagated(self) -> None:
        scheduler = CloudflareAsrScheduler(
            max_in_flight=1,
            requests_per_window=100,
            window_seconds=1.0,
        )

        def timeout() -> None:
            raise TimeoutError("provider timed out")

        started = time.perf_counter()
        with self.assertRaisesRegex(TimeoutError, "provider timed out"):
            scheduler.execute("timeout", timeout)
        scheduler.shutdown(wait=True)

        self.assertLess(time.perf_counter() - started, 0.5)

    def test_cancelled_pending_request_never_runs(self) -> None:
        scheduler = CloudflareAsrScheduler(
            max_in_flight=1,
            requests_per_window=100,
            window_seconds=1.0,
        )
        release = threading.Event()
        active_started = threading.Event()
        cancelled = threading.Event()
        pending_ran = threading.Event()

        def active_request() -> None:
            active_started.set()
            release.wait(timeout=1.0)

        active_thread = threading.Thread(
            target=scheduler.execute,
            args=("active", active_request),
        )
        active_thread.start()
        self.assertTrue(active_started.wait(timeout=1.0))

        errors: list[BaseException] = []

        def pending_job() -> None:
            try:
                scheduler.execute(
                    "pending",
                    pending_ran.set,
                    cancelled.is_set,
                )
            except BaseException as exc:
                errors.append(exc)

        pending_thread = threading.Thread(target=pending_job)
        pending_thread.start()
        deadline = time.monotonic() + 1.0
        while scheduler.snapshot()["pending_requests"] < 1:
            if time.monotonic() >= deadline:
                self.fail("pending request did not enter the queue")
            time.sleep(0.01)
        cancelled.set()
        pending_thread.join(timeout=1.0)
        release.set()
        active_thread.join(timeout=1.0)
        scheduler.shutdown(wait=True)

        self.assertFalse(pending_ran.is_set())
        self.assertEqual(len(errors), 1)
        self.assertIsInstance(errors[0], CloudflareAsrRequestCancelled)


if __name__ == "__main__":
    unittest.main()
