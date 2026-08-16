import json
import logging
import queue
import shutil
import subprocess
import tempfile
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterator, Optional, Union


logger = logging.getLogger(__name__)


PCM_SAMPLE_RATE = 16000
PCM_CHANNELS = 1
PCM_SAMPLE_WIDTH_BYTES = 2
PCM_BYTES_PER_SECOND = PCM_SAMPLE_RATE * PCM_CHANNELS * PCM_SAMPLE_WIDTH_BYTES


@dataclass(frozen=True)
class PcmAudioChunk:
    data: bytes
    offset_seconds: float
    duration_seconds: float
    is_final: bool


@dataclass(frozen=True)
class MediaAudioSource:
    location: str
    headers: dict[str, str] = field(default_factory=dict)
    label: str = "media"


MediaSourceInput = Union[Path, str, MediaAudioSource]


@dataclass(frozen=True)
class _StreamFailure:
    message: str


_STREAM_END = object()


def _required_executable(name: str) -> str:
    executable = shutil.which(name)
    if executable:
        return executable
    raise RuntimeError(f"找不到 {name}，請先安裝 FFmpeg 並確認它已加入 PATH")


def _normalize_source(source: MediaSourceInput) -> MediaAudioSource:
    if isinstance(source, MediaAudioSource):
        return source
    location = str(source)
    return MediaAudioSource(
        location=location,
        label=Path(location).stem or "media",
    )


def _ffmpeg_input_arguments(source: MediaAudioSource) -> list[str]:
    arguments: list[str] = []
    sanitized_headers: dict[str, str] = {}
    for raw_name, raw_value in source.headers.items():
        name = str(raw_name).replace("\r", "").replace("\n", "").strip()
        value = str(raw_value).replace("\r", "").replace("\n", "").strip()
        if name and value:
            sanitized_headers[name] = value

    user_agent = sanitized_headers.get("User-Agent") or sanitized_headers.get(
        "user-agent"
    )
    if user_agent:
        arguments.extend(["-user_agent", user_agent])

    custom_headers = [
        f"{name}: {value}\r\n"
        for name, value in sanitized_headers.items()
        if name.lower() != "user-agent" and value
    ]
    if custom_headers:
        arguments.extend(["-headers", "".join(custom_headers)])

    if source.location.lower().startswith(("http://", "https://")):
        arguments.extend(
            [
                "-reconnect",
                "1",
                "-reconnect_streamed",
                "1",
                "-reconnect_delay_max",
                "5",
            ]
        )
    arguments.extend(["-i", source.location])
    return arguments


def probe_media_duration(media_path: MediaSourceInput) -> float:
    source = _normalize_source(media_path)
    if source.location.lower().startswith(("http://", "https://")):
        return 0.0
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        logger.warning("ffprobe is unavailable; media duration cannot be detected")
        return 0.0

    try:
        completed = subprocess.run(
            [
                ffprobe,
                "-v",
                "error",
                "-show_entries",
                "format=duration:stream=duration",
                "-of",
                "json",
                source.location,
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        payload = json.loads(completed.stdout or "{}")
        durations: list[float] = []
        format_duration = (payload.get("format") or {}).get("duration")
        if format_duration is not None:
            try:
                durations.append(float(format_duration))
            except (TypeError, ValueError):
                pass
        for stream in payload.get("streams") or []:
            stream_duration = stream.get("duration")
            if stream_duration is not None:
                try:
                    durations.append(float(stream_duration))
                except (TypeError, ValueError):
                    pass
        return max((value for value in durations if value > 0), default=0.0)
    except Exception as exc:
        logger.warning("Could not probe media duration for %s: %s", source.label, exc)
        return 0.0


def decode_media_prefix(
    media_path: MediaSourceInput,
    duration_seconds: float,
    cancel_check: Optional[Callable[[], bool]] = None,
) -> bytes:
    ffmpeg = _required_executable("ffmpeg")
    source = _normalize_source(media_path)
    maximum_bytes = max(
        PCM_SAMPLE_WIDTH_BYTES,
        int(max(1.0, duration_seconds) * PCM_BYTES_PER_SECOND),
    )
    command = [
        ffmpeg,
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        *_ffmpeg_input_arguments(source),
        "-map",
        "0:a:0",
        "-vn",
        "-t",
        str(max(1.0, duration_seconds)),
        "-ac",
        str(PCM_CHANNELS),
        "-ar",
        str(PCM_SAMPLE_RATE),
        "-acodec",
        "pcm_s16le",
        "-f",
        "s16le",
        "pipe:1",
    ]

    with tempfile.TemporaryFile() as stderr_file:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=stderr_file,
        )
        data = bytearray()
        try:
            if process.stdout is None:
                raise RuntimeError("FFmpeg 無法建立音訊輸出管線")
            while True:
                if cancel_check is not None and cancel_check():
                    raise InterruptedError("轉譯已取消")
                block = process.stdout.read(64 * 1024)
                if not block:
                    break
                remaining = maximum_bytes - len(data)
                if remaining > 0:
                    data.extend(block[:remaining])
            return_code = process.wait(timeout=10)
            process.stdout.close()
            if return_code != 0:
                stderr_file.seek(0)
                detail = stderr_file.read().decode("utf-8", errors="replace").strip()
                raise RuntimeError(f"FFmpeg 讀取音訊失敗: {detail or f'exit code {return_code}'}")
        except Exception:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=3)
            raise

    if len(data) < PCM_SAMPLE_WIDTH_BYTES:
        raise RuntimeError("媒體檔案沒有可轉譯的音軌")
    return bytes(data[: len(data) - (len(data) % PCM_SAMPLE_WIDTH_BYTES)])


class FFmpegPcmChunkStream:
    def __init__(
        self,
        media_path: MediaSourceInput,
        chunk_seconds: float = 60.0,
        initial_chunk_seconds: Optional[float] = None,
        overlap_seconds: float = 2.0,
        queue_size: int = 2,
        prefetch_chunks: int = 1,
        cancel_check: Optional[Callable[[], bool]] = None,
    ) -> None:
        self.source = _normalize_source(media_path)
        self.chunk_seconds = max(10.0, float(chunk_seconds))
        self.initial_chunk_seconds = min(
            self.chunk_seconds,
            max(
                5.0,
                float(initial_chunk_seconds or self.chunk_seconds),
            ),
        )
        self.overlap_seconds = min(
            max(0.0, float(overlap_seconds)),
            self.chunk_seconds / 2,
        )
        self.cancel_check = cancel_check
        queue_capacity = max(1, int(queue_size))
        self.prefetch_chunks = max(1, min(queue_capacity, int(prefetch_chunks)))
        self._queue: queue.Queue[object] = queue.Queue(maxsize=queue_capacity)
        self._stop_event = threading.Event()
        self._process_lock = threading.Lock()
        self._process: Optional[subprocess.Popen[bytes]] = None
        self._thread = threading.Thread(
            target=self._produce,
            name=f"ffmpeg-pcm-{self.source.label[:24]}",
            daemon=True,
        )
        self._started = False

    def __enter__(self) -> "FFmpegPcmChunkStream":
        self.start()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    def start(self) -> None:
        if not self._started:
            self._started = True
            self._thread.start()

    def __iter__(self) -> Iterator[PcmAudioChunk]:
        self.start()
        buffered_chunks: list[PcmAudioChunk] = []
        reached_end = False
        while len(buffered_chunks) < self.prefetch_chunks:
            item = self._queue.get()
            if item is _STREAM_END:
                reached_end = True
                break
            if isinstance(item, _StreamFailure):
                raise RuntimeError(item.message)
            if isinstance(item, PcmAudioChunk):
                buffered_chunks.append(item)

        yield from buffered_chunks
        if reached_end:
            return

        while True:
            item = self._queue.get()
            if item is _STREAM_END:
                return
            if isinstance(item, _StreamFailure):
                raise RuntimeError(item.message)
            if isinstance(item, PcmAudioChunk):
                yield item

    def close(self) -> None:
        self._stop_event.set()
        with self._process_lock:
            process = self._process
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=3)
        if self._started and self._thread.is_alive():
            self._thread.join(timeout=5)

    def _cancelled(self) -> bool:
        return self._stop_event.is_set() or (
            self.cancel_check is not None and self.cancel_check()
        )

    def _put(self, item: object) -> bool:
        while not self._stop_event.is_set():
            try:
                self._queue.put(item, timeout=0.2)
                return True
            except queue.Full:
                continue
        return False

    def _read_bytes(self, stream, size: int) -> bytes:
        data = bytearray()
        while len(data) < size and not self._cancelled():
            block = stream.read(min(256 * 1024, size - len(data)))
            if not block:
                break
            data.extend(block)
        return bytes(data[: len(data) - (len(data) % PCM_SAMPLE_WIDTH_BYTES)])

    def _produce(self) -> None:
        ffmpeg = ""
        process: Optional[subprocess.Popen[bytes]] = None
        try:
            ffmpeg = _required_executable("ffmpeg")
            command = [
                ffmpeg,
                "-nostdin",
                "-hide_banner",
                "-loglevel",
                "error",
                *_ffmpeg_input_arguments(self.source),
                "-map",
                "0:a:0",
                "-vn",
                "-ac",
                str(PCM_CHANNELS),
                "-ar",
                str(PCM_SAMPLE_RATE),
                "-acodec",
                "pcm_s16le",
                "-f",
                "s16le",
                "pipe:1",
            ]
            chunk_bytes = int(self.chunk_seconds * PCM_BYTES_PER_SECOND)
            chunk_bytes -= chunk_bytes % PCM_SAMPLE_WIDTH_BYTES
            initial_chunk_bytes = int(
                self.initial_chunk_seconds * PCM_BYTES_PER_SECOND
            )
            initial_chunk_bytes -= initial_chunk_bytes % PCM_SAMPLE_WIDTH_BYTES
            overlap_bytes = int(self.overlap_seconds * PCM_BYTES_PER_SECOND)
            overlap_bytes -= overlap_bytes % PCM_SAMPLE_WIDTH_BYTES

            with tempfile.TemporaryFile() as stderr_file:
                process = subprocess.Popen(
                    command,
                    stdout=subprocess.PIPE,
                    stderr=stderr_file,
                )
                with self._process_lock:
                    self._process = process
                if process.stdout is None:
                    raise RuntimeError("FFmpeg 無法建立音訊輸出管線")

                current_target_bytes = initial_chunk_bytes
                current = self._read_bytes(process.stdout, current_target_bytes)
                offset_seconds = 0.0
                emitted_chunks = 0
                while current and not self._cancelled():
                    current_duration = len(current) / PCM_BYTES_PER_SECOND
                    if len(current) < current_target_bytes:
                        self._put(
                            PcmAudioChunk(
                                data=current,
                                offset_seconds=offset_seconds,
                                duration_seconds=current_duration,
                                is_final=True,
                            )
                        )
                        emitted_chunks += 1
                        current = b""
                        break

                    if not self._put(
                        PcmAudioChunk(
                            data=current,
                            offset_seconds=offset_seconds,
                            duration_seconds=current_duration,
                            is_final=False,
                        )
                    ):
                        break
                    emitted_chunks += 1

                    tail = current[-overlap_bytes:] if overlap_bytes else b""
                    next_fresh_bytes = chunk_bytes - overlap_bytes
                    fresh = self._read_bytes(process.stdout, next_fresh_bytes)
                    offset_seconds += (
                        current_target_bytes - overlap_bytes
                    ) / PCM_BYTES_PER_SECOND
                    current_target_bytes = chunk_bytes
                    if not fresh:
                        if tail and not self._cancelled():
                            self._put(
                                PcmAudioChunk(
                                    data=tail,
                                    offset_seconds=offset_seconds,
                                    duration_seconds=len(tail) / PCM_BYTES_PER_SECOND,
                                    is_final=True,
                                )
                            )
                            emitted_chunks += 1
                        current = b""
                        break
                    current = tail + fresh

                if process.stdout is not None:
                    process.stdout.close()
                if self._cancelled():
                    if process.poll() is None:
                        process.terminate()
                    return
                return_code = process.wait(timeout=10)
                if return_code != 0:
                    stderr_file.seek(0)
                    detail = stderr_file.read().decode("utf-8", errors="replace").strip()
                    raise RuntimeError(
                        f"FFmpeg 讀取音訊失敗: {detail or f'exit code {return_code}'}"
                    )
                if emitted_chunks == 0:
                    raise RuntimeError("媒體檔案沒有可轉譯的音軌")
        except Exception as exc:
            if not self._cancelled():
                self._put(_StreamFailure(str(exc)))
        finally:
            if process is not None and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=3)
            with self._process_lock:
                self._process = None
            self._put(_STREAM_END)
