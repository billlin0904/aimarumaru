import logging
import os
import sys
from datetime import datetime, timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


DEFAULT_LOG_TIMEZONE = "Asia/Taipei"
DEFAULT_LOG_LEVEL = "INFO"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S%z"
LOG_FORMAT = "[%(asctime)s] %(levelname)s %(name)s: %(message)s"


class TimezoneFormatter(logging.Formatter):
    def __init__(self, timezone_name: str):
        super().__init__(fmt=LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
        try:
            self.timezone = ZoneInfo(timezone_name)
        except ZoneInfoNotFoundError:
            self.timezone = timezone.utc

    def formatTime(self, record, datefmt=None):
        created_at = datetime.fromtimestamp(record.created, self.timezone)
        return created_at.strftime(datefmt or LOG_DATE_FORMAT)

    def format(self, record):
        formatted = super().format(record)
        if "\n" not in formatted:
            return formatted
        continuation_prefix = (
            f"[{self.formatTime(record, self.datefmt)}] "
            f"{record.levelname} {record.name}: "
        )
        lines = formatted.splitlines()
        return "\n".join(
            [lines[0], *(f"{continuation_prefix}{line}" for line in lines[1:])]
        )


def configure_logging() -> None:
    timezone_name = os.getenv(
        "AUDIOIO_LOG_TIMEZONE",
        DEFAULT_LOG_TIMEZONE,
    ).strip() or DEFAULT_LOG_TIMEZONE
    level_name = os.getenv(
        "AUDIOIO_LOG_LEVEL",
        DEFAULT_LOG_LEVEL,
    ).strip().upper()
    level = getattr(logging, level_name, logging.INFO)
    formatter = TimezoneFormatter(timezone_name)

    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    if not root_logger.handlers:
        root_logger.addHandler(logging.StreamHandler(sys.stdout))
    for handler in root_logger.handlers:
        handler.setFormatter(formatter)

    for logger_name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        logger = logging.getLogger(logger_name)
        for handler in logger.handlers:
            handler.setFormatter(formatter)
