import json
import logging
import os
import sys
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, TextIO
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from fastapi import FastAPI, Request


DEFAULT_LOG_TIMEZONE = "Asia/Taipei"
DEFAULT_LOG_LEVEL = "INFO"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S%z"
LOG_FORMAT = "[%(asctime)s] %(levelname)s %(name)s: %(message)s"
ACCESS_LOG_ENABLED_ENV = "AUDIOIO_ACCESS_LOG_ENABLED"
ACCESS_LOG_FILE_ENV = "AUDIOIO_ACCESS_LOG_FILE"
ACCESS_LOG_RETENTION_DAYS_ENV = "AUDIOIO_ACCESS_LOG_RETENTION_DAYS"
ACCESS_LOG_IGNORED_PATHS_ENV = "AUDIOIO_ACCESS_LOG_IGNORED_PATHS"
DEFAULT_ACCESS_LOG_IGNORED_PATHS = "/api/nvidia-smi"

access_logger = logging.getLogger("audioio.access")
access_logger.setLevel(logging.INFO)
access_logger.propagate = False

LOG_RECORD_FIELDS = set(logging.makeLogRecord({}).__dict__) | {
    "asctime",
    "message",
}


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


class JsonLineFormatter(TimezoneFormatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "event": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key in LOG_RECORD_FIELDS or key.startswith("_"):
                continue
            payload[key] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


class UvicornAccessPathFilter(logging.Filter):
    """Hide noisy polling endpoints from Uvicorn's console access log."""

    def __init__(self, ignored_paths: set[str]) -> None:
        super().__init__()
        self.ignored_paths = {
            path.rstrip("/") or "/"
            for path in ignored_paths
            if path.strip()
        }

    def filter(self, record: logging.LogRecord) -> bool:
        request_path = None
        if isinstance(record.args, tuple) and len(record.args) >= 3:
            request_path = str(record.args[2]).split("?", 1)[0]
        if request_path is None:
            message = record.getMessage()
            request_path = next(
                (
                    path
                    for path in self.ignored_paths
                    if f" {path} " in message or f" {path}?" in message
                ),
                None,
            )
        if request_path is None:
            return True
        normalized_path = request_path.rstrip("/") or "/"
        return normalized_path not in self.ignored_paths


def environment_flag(name: str, default: bool = True) -> bool:
    fallback = "true" if default else "false"
    return os.getenv(name, fallback).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def configured_ignored_access_paths() -> set[str]:
    return {
        path.strip()
        for path in os.getenv(
            ACCESS_LOG_IGNORED_PATHS_ENV,
            DEFAULT_ACCESS_LOG_IGNORED_PATHS,
        ).split(",")
        if path.strip()
    }


class MultiprocessSafeDailyFileHandler(logging.Handler):
    """Write directly to a dated file instead of renaming an open Windows file."""

    terminator = "\n"

    def __init__(
        self,
        base_path: Path,
        *,
        retention_days: int,
        timezone_name: str,
        date_provider: Callable[[], date] | None = None,
    ) -> None:
        super().__init__()
        self.base_path = base_path.resolve()
        self.baseFilename = str(self.base_path)
        self.retention_days = max(1, retention_days)
        try:
            self.timezone = ZoneInfo(timezone_name)
        except ZoneInfoNotFoundError:
            self.timezone = timezone.utc
        self._date_provider = date_provider or (
            lambda: datetime.now(self.timezone).date()
        )
        self._current_date: date | None = None
        self.stream: TextIO | None = None

    def dated_path(self, log_date: date) -> Path:
        suffix = self.base_path.suffix
        stem = self.base_path.name[:-len(suffix)] if suffix else self.base_path.name
        return self.base_path.with_name(
            f"{stem}.{log_date.isoformat()}{suffix}"
        )

    def _close_stream(self) -> None:
        if self.stream is None:
            return
        try:
            self.stream.flush()
            self.stream.close()
        finally:
            self.stream = None

    def _delete_expired_files(self, current_date: date) -> None:
        oldest_retained_date = current_date - timedelta(
            days=self.retention_days - 1
        )
        suffix = self.base_path.suffix
        stem = self.base_path.name[:-len(suffix)] if suffix else self.base_path.name
        prefix = f"{stem}."
        legacy_prefix = f"{self.base_path.name}."
        candidates = set(self.base_path.parent.glob(f"{stem}.*{suffix}"))
        candidates.update(self.base_path.parent.glob(f"{self.base_path.name}.*"))
        for candidate in candidates:
            name = candidate.name
            if name.startswith(legacy_prefix):
                date_text = name[len(legacy_prefix):]
            else:
                date_text = name[len(prefix):]
                if suffix and date_text.endswith(suffix):
                    date_text = date_text[:-len(suffix)]
            try:
                candidate_date = date.fromisoformat(date_text)
            except ValueError:
                continue
            if candidate_date >= oldest_retained_date:
                continue
            try:
                candidate.unlink()
            except OSError:
                # Another process may still have yesterday's file open on Windows.
                # A later rollover/startup will retry the cleanup.
                continue

    def _ensure_stream(self, current_date: date) -> None:
        if self.stream is not None and self._current_date == current_date:
            return
        self._close_stream()
        self.base_path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = self.dated_path(current_date).open(
            "a",
            encoding="utf-8",
        )
        self._current_date = current_date
        self._delete_expired_files(current_date)

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self._ensure_stream(self._date_provider())
            if self.stream is None:  # pragma: no cover - guarded above
                return
            self.stream.write(self.format(record) + self.terminator)
            self.stream.flush()
        except Exception:
            self.handleError(record)

    def close(self) -> None:
        self.acquire()
        try:
            self._close_stream()
        finally:
            self.release()
        super().close()


def configure_daily_access_file_logging() -> MultiprocessSafeDailyFileHandler | None:
    if not environment_flag(ACCESS_LOG_ENABLED_ENV):
        return None

    configured_path = os.getenv(
        ACCESS_LOG_FILE_ENV,
        "logs/audioio-access.jsonl",
    ).strip()
    if not configured_path:
        return None

    log_path = Path(configured_path)
    if not log_path.is_absolute():
        log_path = Path.cwd() / log_path
    log_path = log_path.resolve()

    for handler in access_logger.handlers:
        if isinstance(handler, MultiprocessSafeDailyFileHandler):
            if Path(handler.baseFilename).resolve() == log_path:
                return handler

    try:
        retention_days = max(
            1,
            int(os.getenv(ACCESS_LOG_RETENTION_DAYS_ENV, "7")),
        )
        log_path.parent.mkdir(parents=True, exist_ok=True)
        timezone_name = os.getenv(
            "AUDIOIO_LOG_TIMEZONE",
            DEFAULT_LOG_TIMEZONE,
        ).strip() or DEFAULT_LOG_TIMEZONE
        handler = MultiprocessSafeDailyFileHandler(
            log_path,
            retention_days=retention_days,
            timezone_name=timezone_name,
        )
    except (OSError, ValueError) as exc:
        logging.getLogger(__name__).warning("無法建立 access log 檔案：%s", exc)
        return None

    handler.setLevel(logging.INFO)
    handler.setFormatter(JsonLineFormatter(timezone_name))
    access_logger.addHandler(handler)
    return handler


def set_request_log_metadata(request: Request, **metadata: Any) -> None:
    current = getattr(request.state, "access_log_meta", None)
    if not isinstance(current, dict):
        current = {}
        request.state.access_log_meta = current
    current.update(
        {key: value for key, value in metadata.items() if value is not None}
    )


def log_structured_event(event: str, **metadata: Any) -> None:
    access_logger.info(
        event,
        extra={key: value for key, value in metadata.items() if value is not None},
    )


def configure_access_logging(app: FastAPI) -> None:
    if configure_daily_access_file_logging() is None:
        return

    @app.middleware("http")
    async def daily_access_log_middleware(request: Request, call_next):
        if request.url.path in configured_ignored_access_paths():
            return await call_next(request)
        started_at = datetime.now(timezone.utc)
        started = time.perf_counter()
        response = None
        exception_type = None
        try:
            response = await call_next(request)
            return response
        except Exception as exc:
            exception_type = type(exc).__name__
            raise
        finally:
            status_code = response.status_code if response is not None else 500
            meta = getattr(request.state, "access_log_meta", {})
            client = request.client
            client_ip = request.headers.get("cf-connecting-ip")
            if not client_ip:
                forwarded_for = request.headers.get("x-forwarded-for", "")
                client_ip = forwarded_for.split(",", 1)[0].strip() or None

            log_structured_event(
                "http_request",
                request_started_at=started_at.isoformat().replace("+00:00", "Z"),
                request_id=meta.get("request_id") or request.headers.get("x-request-id"),
                job_id=meta.get("job_id"),
                operation=meta.get("operation"),
                method=request.method,
                path=request.url.path,
                query_keys=sorted(set(request.query_params.keys())),
                status_code=status_code,
                outcome="success" if status_code < 400 else "error",
                latency_ms=round((time.perf_counter() - started) * 1000),
                exception_type=exception_type,
                client_ip=client_ip,
                client_host=client.host if client else None,
                user_agent=request.headers.get("user-agent"),
                cf_ray=request.headers.get("cf-ray"),
                cf_country=request.headers.get("cf-ipcountry"),
                request_content_type=request.headers.get("content-type"),
                request_content_encoding=request.headers.get("content-encoding"),
                request_content_length=request.headers.get("content-length"),
                response_content_encoding=(
                    response.headers.get("content-encoding")
                    if response is not None
                    else None
                ),
                response_content_length=(
                    response.headers.get("content-length")
                    if response is not None
                    else None
                ),
                job_status=meta.get("job_status"),
                media_kind=meta.get("media_kind"),
                output_format=meta.get("output_format"),
                source_language=meta.get("source_language"),
                target_language=meta.get("target_language"),
                detected_language=meta.get("detected_language"),
                language_probability=meta.get("language_probability"),
                subtitle_source=meta.get("subtitle_source"),
                segments=meta.get("segments"),
                groups=meta.get("groups"),
                source_ids=meta.get("source_ids"),
                preceding_source_groups=meta.get("preceding_source_groups"),
                context_segments=meta.get("context_segments"),
                preceding_context_segments=meta.get("preceding_context_segments"),
                following_context_segments=meta.get("following_context_segments"),
                on_screen_terms=meta.get("on_screen_terms"),
                low_confidence_spans=meta.get("low_confidence_spans"),
                characters=meta.get("characters"),
                source_kind=meta.get("source_kind"),
                ignore_subtitles=meta.get("ignore_subtitles"),
                include_word_timestamps=meta.get("include_word_timestamps"),
                vocal_separation=meta.get("vocal_separation"),
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

    ignored_access_paths = configured_ignored_access_paths()
    uvicorn_access_logger = logging.getLogger("uvicorn.access")
    if not any(
        isinstance(log_filter, UvicornAccessPathFilter)
        for log_filter in uvicorn_access_logger.filters
    ):
        uvicorn_access_logger.addFilter(
            UvicornAccessPathFilter(ignored_access_paths)
        )
