import argparse
import json
import logging
import os
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, RedirectResponse


logger = logging.getLogger(__name__)

DEFAULT_RETRY_AFTER_SECONDS = 1800
DEFAULT_MESSAGE = "Service is under maintenance. Please try again later."
BLOCKED_POST_PATHS = {
    "/api/transcribe-audio",
    "/api/youtube-live/jobs",
    "/youtube/srt",
}
MAINTENANCE_PAGE_PATHS = {"", "/youtube-live"}


@dataclass(frozen=True)
class MaintenanceState:
    enabled: bool = False
    message: str = DEFAULT_MESSAGE
    retry_after: int = DEFAULT_RETRY_AFTER_SECONDS


def get_default_maintenance_path(project_root: Optional[Path] = None) -> Path:
    configured_path = os.getenv("AUDIOIO_MAINTENANCE_FILE", "").strip()
    if configured_path:
        return Path(configured_path).expanduser()

    vault_root = Path("/vault")
    if os.name != "nt" and vault_root.is_dir():
        return vault_root / "config" / "audioio-maintenance.json"

    return (project_root or Path(__file__).resolve().parent) / "maintenance.json"


class MaintenanceManager:
    def __init__(self, path: Path, cache_seconds: float = 1.0):
        self.path = path
        self.cache_seconds = max(0.0, cache_seconds)
        self._state = MaintenanceState()
        self._next_refresh = 0.0
        self._lock = threading.Lock()

    def get_state(self, refresh: bool = False) -> MaintenanceState:
        now = time.monotonic()
        if not refresh and now < self._next_refresh:
            return self._state

        with self._lock:
            now = time.monotonic()
            if not refresh and now < self._next_refresh:
                return self._state
            self._state = self._read_state()
            self._next_refresh = now + self.cache_seconds
            return self._state

    def _read_state(self) -> MaintenanceState:
        if not self.path.is_file():
            return MaintenanceState()

        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            message = str(payload.get("message") or DEFAULT_MESSAGE).strip()
            retry_after = int(payload.get("retry_after", DEFAULT_RETRY_AFTER_SECONDS))
            return MaintenanceState(
                enabled=payload.get("enabled") is True,
                message=message[:300] or DEFAULT_MESSAGE,
                retry_after=max(60, min(retry_after, 86400)),
            )
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            logger.error("Unable to read maintenance config %s: %s", self.path, exc)
            return MaintenanceState()

    def write_state(self, state: MaintenanceState) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = self.path.with_suffix(f"{self.path.suffix}.tmp")
        temporary_path.write_text(
            json.dumps(asdict(state), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary_path.replace(self.path)
        self.get_state(refresh=True)


class MaintenanceMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, manager: MaintenanceManager):
        super().__init__(app)
        self.manager = manager

    async def dispatch(self, request: Request, call_next):
        state = self.manager.get_state()
        if not state.enabled:
            return await call_next(request)

        path = request.url.path.rstrip("/")
        headers = {
            "Cache-Control": "no-store",
            "Retry-After": str(state.retry_after),
        }
        if request.method == "GET" and path in MAINTENANCE_PAGE_PATHS:
            return RedirectResponse("/maintenance", status_code=307, headers=headers)

        if request.method == "POST" and path in BLOCKED_POST_PATHS:
            return JSONResponse(
                {
                    "detail": state.message,
                    "code": "maintenance",
                    "retry_after": state.retry_after,
                },
                status_code=503,
                headers=headers,
            )

        return await call_next(request)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Manage Audio IO maintenance mode.")
    parser.add_argument("action", choices=("on", "off", "status"))
    parser.add_argument("--message", default=DEFAULT_MESSAGE)
    parser.add_argument(
        "--retry-after",
        type=int,
        default=DEFAULT_RETRY_AFTER_SECONDS,
        help="Suggested retry delay in seconds.",
    )
    parser.add_argument("--file", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path = args.file or get_default_maintenance_path()
    manager = MaintenanceManager(path, cache_seconds=0)

    if args.action == "on":
        manager.write_state(
            MaintenanceState(
                enabled=True,
                message=args.message,
                retry_after=args.retry_after,
            )
        )
    elif args.action == "off":
        current = manager.get_state(refresh=True)
        manager.write_state(
            MaintenanceState(
                enabled=False,
                message=current.message,
                retry_after=current.retry_after,
            )
        )

    print(f"Config: {path}")
    print(json.dumps(asdict(manager.get_state(refresh=True)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
