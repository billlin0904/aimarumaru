from __future__ import annotations

import asyncio
import base64
import json
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

import aiohttp


TOGETHER_REALTIME_INPUT_FORMAT = "pcm_s16le_16000"


class TogetherRealtimeError(RuntimeError):
    pass


class TogetherRealtimeConnectionError(TogetherRealtimeError):
    pass


class TogetherRealtimeCancelled(TogetherRealtimeError):
    pass


@dataclass(frozen=True)
class TogetherRealtimeSettings:
    api_key: str
    websocket_url: str = "wss://api.together.ai/v1/realtime"
    model: str = "openai/whisper-large-v3"
    timeout_seconds: float = 120.0
    max_retries: int = 1
    frame_bytes: int = 4096


@dataclass(frozen=True)
class TogetherRealtimeResult:
    text: str
    delta_count: int


RealtimeConnector = Callable[
    [str, dict[str, str], dict[str, str]],
    Awaitable[Any],
]


class TogetherRealtimeTranscriber:
    """Persistent Together WebSocket session exposed through a sync facade."""

    def __init__(
        self,
        settings: TogetherRealtimeSettings,
        *,
        language: str | None = None,
        connector: RealtimeConnector | None = None,
    ) -> None:
        if not settings.api_key:
            raise RuntimeError("Together Realtime 尚未設定 TOGETHER_API_KEY")
        self.settings = settings
        self.language = str(language or "auto").strip().lower() or "auto"
        self._connector = connector
        self._loop = asyncio.new_event_loop()
        self._session: aiohttp.ClientSession | None = None
        self._websocket: Any = None
        self._closed = False
        self._completed_history = ""

    @property
    def model_name(self) -> str:
        return self.settings.model

    def __enter__(self) -> "TogetherRealtimeTranscriber":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    def transcribe(
        self,
        pcm_audio: bytes,
        *,
        on_delta: Callable[[str], None] | None = None,
        cancel_check: Callable[[], bool] | None = None,
    ) -> TogetherRealtimeResult:
        if self._closed:
            raise TogetherRealtimeError("Together Realtime session 已關閉")
        if len(pcm_audio) % 2:
            pcm_audio = pcm_audio[:-1]
        if not pcm_audio:
            return TogetherRealtimeResult(text="", delta_count=0)
        return self._loop.run_until_complete(
            self._transcribe_with_retry(pcm_audio, on_delta, cancel_check)
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._loop.run_until_complete(self._close_connection())
        finally:
            self._loop.close()

    async def _connect(self) -> Any:
        websocket = self._websocket
        if websocket is not None and not bool(getattr(websocket, "closed", False)):
            return websocket

        headers = {
            "Authorization": f"Bearer {self.settings.api_key}",
            "OpenAI-Beta": "realtime=v1",
        }
        params = {
            "intent": "transcription",
            "model": self.settings.model,
            "input_audio_format": TOGETHER_REALTIME_INPUT_FORMAT,
            "turn_detection": "none",
            "language": self.language,
        }
        if self._connector is not None:
            self._websocket = await self._connector(
                self.settings.websocket_url,
                headers,
                params,
            )
            return self._websocket

        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(
                total=None,
                connect=self.settings.timeout_seconds,
                sock_connect=self.settings.timeout_seconds,
            )
            self._session = aiohttp.ClientSession(timeout=timeout)
        try:
            self._websocket = await self._session.ws_connect(
                self.settings.websocket_url,
                headers=headers,
                params=params,
                heartbeat=30,
                max_msg_size=4 * 1024 * 1024,
            )
        except Exception:
            await self._close_connection()
            raise
        return self._websocket

    async def _transcribe_with_retry(
        self,
        pcm_audio: bytes,
        on_delta: Callable[[str], None] | None,
        cancel_check: Callable[[], bool] | None,
    ) -> TogetherRealtimeResult:
        for attempt in range(self.settings.max_retries + 1):
            if cancel_check is not None and cancel_check():
                raise TogetherRealtimeCancelled("轉譯已取消")
            try:
                websocket = await self._connect()
                return await self._transcribe_once(
                    websocket,
                    pcm_audio,
                    on_delta,
                    cancel_check,
                )
            except TogetherRealtimeCancelled:
                raise
            except TogetherRealtimeConnectionError as exc:
                await self._close_websocket()
                if attempt >= self.settings.max_retries:
                    raise TogetherRealtimeConnectionError(
                        f"Together Realtime 連線失敗: {exc}"
                    ) from exc
            except TogetherRealtimeError:
                raise
            except (aiohttp.ClientError, asyncio.TimeoutError, ConnectionError) as exc:
                await self._close_websocket()
                if attempt >= self.settings.max_retries:
                    raise TogetherRealtimeConnectionError(
                        f"Together Realtime 連線失敗: {exc}"
                    ) from exc
        raise AssertionError("Together Realtime retry loop did not return")

    async def _transcribe_once(
        self,
        websocket: Any,
        pcm_audio: bytes,
        on_delta: Callable[[str], None] | None,
        cancel_check: Callable[[], bool] | None,
    ) -> TogetherRealtimeResult:
        frame_bytes = max(2, int(self.settings.frame_bytes))
        frame_bytes -= frame_bytes % 2
        for offset in range(0, len(pcm_audio), frame_bytes):
            if cancel_check is not None and cancel_check():
                raise TogetherRealtimeCancelled("轉譯已取消")
            await websocket.send_json(
                {
                    "type": "input_audio_buffer.append",
                    "audio": base64.b64encode(
                        pcm_audio[offset : offset + frame_bytes]
                    ).decode("ascii"),
                }
            )
        await websocket.send_json({"type": "input_audio_buffer.commit"})

        deadline = self._loop.time() + self.settings.timeout_seconds
        delta_count = 0
        while True:
            if cancel_check is not None and cancel_check():
                raise TogetherRealtimeCancelled("轉譯已取消")
            remaining = deadline - self._loop.time()
            if remaining <= 0:
                raise asyncio.TimeoutError("Together Realtime 回應逾時")
            try:
                message = await asyncio.wait_for(
                    websocket.receive(),
                    timeout=min(1.0, remaining),
                )
            except asyncio.TimeoutError:
                continue
            if message.type == aiohttp.WSMsgType.TEXT:
                try:
                    payload = json.loads(message.data)
                except (TypeError, ValueError) as exc:
                    raise TogetherRealtimeError(
                        "Together Realtime 回傳了無效 JSON"
                    ) from exc
                event_type = str(payload.get("type") or "")
                if event_type == "conversation.item.input_audio_transcription.delta":
                    delta = self._strip_completed_prefix(
                        str(payload.get("delta") or "").strip()
                    )
                    if delta:
                        delta_count += 1
                        if on_delta is not None:
                            on_delta(delta)
                    continue
                if event_type == "conversation.item.input_audio_transcription.completed":
                    transcript = str(payload.get("transcript") or "").strip()
                    text = self._strip_completed_prefix(transcript)
                    if text:
                        self._completed_history = " ".join(
                            part
                            for part in (self._completed_history, text)
                            if part
                        )
                    return TogetherRealtimeResult(
                        text=text,
                        delta_count=delta_count,
                    )
                if event_type in {
                    "conversation.item.input_audio_transcription.failed",
                    "error",
                }:
                    error = payload.get("error")
                    detail = (
                        str(error.get("message") or error.get("code") or "")
                        if isinstance(error, dict)
                        else str(error or "")
                    )
                    raise TogetherRealtimeError(
                        f"Together Realtime 轉譯失敗: {detail or event_type}"
                    )
                continue
            if message.type == aiohttp.WSMsgType.ERROR:
                error = websocket.exception()
                raise TogetherRealtimeConnectionError(
                    f"Together Realtime WebSocket 錯誤: {error or 'unknown error'}"
                )
            if message.type in {
                aiohttp.WSMsgType.CLOSE,
                aiohttp.WSMsgType.CLOSED,
                aiohttp.WSMsgType.CLOSING,
            }:
                raise TogetherRealtimeConnectionError(
                    "Together Realtime WebSocket 已中斷"
                )

    def _strip_completed_prefix(self, text: str) -> str:
        history = self._completed_history.strip()
        value = text.strip()
        if history and value.startswith(history):
            return value[len(history) :].strip()
        return value

    async def _close_websocket(self) -> None:
        websocket = self._websocket
        self._websocket = None
        if websocket is not None and not bool(getattr(websocket, "closed", False)):
            await websocket.close()

    async def _close_connection(self) -> None:
        await self._close_websocket()
        session = self._session
        self._session = None
        if session is not None and not session.closed:
            await session.close()
