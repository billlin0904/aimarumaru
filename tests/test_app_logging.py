import logging
import tempfile
import unittest
from datetime import date
from pathlib import Path

from app_logging import (
    JsonLineFormatter,
    MultiprocessSafeDailyFileHandler,
    UvicornAccessPathFilter,
)


class MultiprocessSafeDailyFileHandlerTests(unittest.TestCase):
    def test_two_handlers_append_and_roll_without_renaming_shared_file(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            base_path = Path(temporary_directory) / "audioio-access.jsonl"
            current_date = [date(2026, 8, 8)]
            handlers = [
                MultiprocessSafeDailyFileHandler(
                    base_path,
                    retention_days=7,
                    timezone_name="Asia/Taipei",
                    date_provider=lambda: current_date[0],
                )
                for _ in range(2)
            ]
            for handler in handlers:
                handler.setFormatter(JsonLineFormatter("Asia/Taipei"))

            try:
                handlers[0].emit(logging.LogRecord(
                    "audioio.access", logging.INFO, __file__, 1,
                    "first-worker", (), None,
                ))
                handlers[1].emit(logging.LogRecord(
                    "audioio.access", logging.INFO, __file__, 1,
                    "second-worker", (), None,
                ))
                current_date[0] = date(2026, 8, 9)
                handlers[0].emit(logging.LogRecord(
                    "audioio.access", logging.INFO, __file__, 1,
                    "new-day", (), None,
                ))
            finally:
                for handler in handlers:
                    handler.close()

            first_day = base_path.with_name(
                "audioio-access.2026-08-08.jsonl"
            ).read_text(encoding="utf-8")
            second_day = base_path.with_name(
                "audioio-access.2026-08-09.jsonl"
            ).read_text(encoding="utf-8")
            self.assertIn("first-worker", first_day)
            self.assertIn("second-worker", first_day)
            self.assertIn("new-day", second_day)
            self.assertFalse(base_path.exists())

    def test_cleanup_keeps_only_configured_calendar_days(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            base_path = Path(temporary_directory) / "audioio-access.jsonl"
            expired = base_path.with_name("audioio-access.2026-08-02.jsonl")
            legacy_expired = base_path.with_name(
                "audioio-access.jsonl.2026-08-01"
            )
            retained = base_path.with_name("audioio-access.2026-08-03.jsonl")
            expired.write_text("expired", encoding="utf-8")
            legacy_expired.write_text("legacy expired", encoding="utf-8")
            retained.write_text("retained", encoding="utf-8")
            handler = MultiprocessSafeDailyFileHandler(
                base_path,
                retention_days=7,
                timezone_name="Asia/Taipei",
                date_provider=lambda: date(2026, 8, 9),
            )

            try:
                handler.emit(logging.LogRecord(
                    "audioio.access", logging.INFO, __file__, 1,
                    "cleanup", (), None,
                ))
            finally:
                handler.close()

            self.assertFalse(expired.exists())
            self.assertFalse(legacy_expired.exists())
            self.assertTrue(retained.exists())


class UvicornAccessPathFilterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.log_filter = UvicornAccessPathFilter({"/api/nvidia-smi"})

    @staticmethod
    def access_record(path: str) -> logging.LogRecord:
        return logging.LogRecord(
            "uvicorn.access",
            logging.INFO,
            __file__,
            1,
            '%s - "%s %s HTTP/%s" %d',
            ("127.0.0.1:1234", "GET", path, "1.1", 200),
            None,
        )

    def test_ignores_nvidia_polling_with_optional_query(self) -> None:
        self.assertFalse(
            self.log_filter.filter(
                self.access_record("/api/nvidia-smi?refresh=1")
            )
        )

    def test_keeps_other_access_logs(self) -> None:
        self.assertTrue(
            self.log_filter.filter(
                self.access_record("/api/transcribe-queue/status")
            )
        )


if __name__ == "__main__":
    unittest.main()
