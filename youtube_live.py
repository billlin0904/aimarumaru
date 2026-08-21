import asyncio
import json
import logging
import math
import os
import re
import secrets
import shutil
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any, Optional

import aiohttp
import numpy as np
import requests
from fastapi import APIRouter, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, Field

from app_logging import log_structured_event, set_request_log_metadata
from cloudflare_asr_queue import (
    CloudflareAsrRequestCancelled,
    cloudflare_asr_scheduler,
)
from gpu_info import get_pynvml_gpu_info
from media_audio_stream import (
    FFmpegPcmChunkStream,
    MediaSourceInput,
    decode_media_prefix,
    probe_media_duration,
)
from provider_profiles import (
    CloudflareWhisperClient,
    CloudflareWhisperSettings,
    ElevenLabsWhisperClient,
    ElevenLabsWhisperSettings,
    GroqWhisperClient,
    GroqWhisperSettings,
    REMOTE_ASR_PROVIDERS,
    TogetherWhisperClient,
    TogetherWhisperSettings,
    WhisperAsrClient,
    asr_provider_for_profile,
    normalize_processing_profile,
    retry_after_seconds,
    route_translation_workflow_payload,
    translation_type_for_profile,
)
from together_realtime import (
    TogetherRealtimeCancelled,
    TogetherRealtimeResult,
    TogetherRealtimeSettings,
    TogetherRealtimeTranscriber,
)
from text_converter import to_traditional_chinese
from transcribe_queue import (
    TranscriptionCancelled,
    cancel_queued_transcribe_task,
    enqueue_transcribe_task,
    fair_transcription_scheduler,
    get_transcribe_queue_counts,
    register_transcribe_cleanup,
    register_transcribe_handler,
)
from youtube_srt import (
    choose_subtitle_track,
    download_youtube_audio,
    download_subtitle_content,
    get_youtube_audio_stream_source,
    get_youtube_playlist_preview,
    get_youtube_video_info,
    is_youtube_rate_limit_error,
    parse_subtitle_content,
    YOUTUBE_RATE_LIMIT_MESSAGE,
)


logger = logging.getLogger(__name__)

YOUTUBE_LIVE_JOB_TTL_SECONDS = 3600
YOUTUBE_LIVE_EVENT_TIMEOUT_SECONDS = 30
ETA_MIN_ELAPSED_SECONDS = 10
ETA_MIN_PROCESSED_SECONDS = 30
AUDIOIO_DASHBOARD_TOKEN = os.getenv("AUDIOIO_DASHBOARD_TOKEN", "").strip()
AUDIOIO_DASHBOARD_JOB_LIMIT = max(
    1,
    min(100, int(os.getenv("AUDIOIO_DASHBOARD_JOB_LIMIT", "20"))),
)
TRANSLATE_API_BASE = os.getenv(
    "TRANSLATE_API_BASE",
    "https://translate.audio-io.com",
).rstrip("/")
TRANSLATE_API_TIMEOUT_SECONDS = float(
    os.getenv("TRANSLATE_API_TIMEOUT_SECONDS", "150")
)
TRANSLATE_PROXY_MAX_BODY_BYTES = 128 * 1024
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
GROQ_ASR_BASE_URL = os.getenv(
    "GROQ_ASR_BASE_URL",
    os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1"),
).rstrip("/")
GROQ_ASR_MODEL = os.getenv("GROQ_ASR_MODEL", "whisper-large-v3").strip()
GROQ_ASR_TIMEOUT_SECONDS = float(os.getenv("GROQ_ASR_TIMEOUT_SECONDS", "120"))
GROQ_ASR_MAX_RETRIES = max(0, int(os.getenv("GROQ_ASR_MAX_RETRIES", "2")))
GROQ_ASR_FALLBACK_WAIT_SECONDS = max(
    0.0, float(os.getenv("GROQ_ASR_FALLBACK_WAIT_SECONDS", "10"))
)
GROQ_ASR_MAX_WAIT_SECONDS = max(
    0.0, float(os.getenv("GROQ_ASR_MAX_WAIT_SECONDS", "30"))
)
GROQ_ASR_MIN_REQUEST_INTERVAL_SECONDS = max(
    0.0, float(os.getenv("GROQ_ASR_MIN_REQUEST_INTERVAL_SECONDS", "3.1"))
)
CLOUDFLARE_ACCOUNT_ID = os.getenv("CLOUDFLARE_ACCOUNT_ID", "").strip()
CLOUDFLARE_API_TOKEN = os.getenv("CLOUDFLARE_API_TOKEN", "").strip()
CLOUDFLARE_ASR_BASE_URL = os.getenv(
    "CLOUDFLARE_ASR_BASE_URL",
    "https://api.cloudflare.com/client/v4",
).rstrip("/")
CLOUDFLARE_ASR_MODEL = os.getenv(
    "CLOUDFLARE_ASR_MODEL",
    "@cf/openai/whisper-large-v3-turbo",
).strip()
CLOUDFLARE_ASR_TIMEOUT_SECONDS = float(
    os.getenv("CLOUDFLARE_ASR_TIMEOUT_SECONDS", "120")
)
CLOUDFLARE_ASR_MAX_RETRIES = max(
    0, int(os.getenv("CLOUDFLARE_ASR_MAX_RETRIES", "2"))
)
CLOUDFLARE_ASR_FALLBACK_WAIT_SECONDS = max(
    0.0, float(os.getenv("CLOUDFLARE_ASR_FALLBACK_WAIT_SECONDS", "5"))
)
CLOUDFLARE_ASR_MAX_WAIT_SECONDS = max(
    0.0, float(os.getenv("CLOUDFLARE_ASR_MAX_WAIT_SECONDS", "30"))
)
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY", "").strip()
TOGETHER_ASR_BASE_URL = os.getenv(
    "TOGETHER_ASR_BASE_URL",
    "https://api.together.ai/v1",
).rstrip("/")
TOGETHER_ASR_MODEL = os.getenv(
    "TOGETHER_ASR_MODEL",
    "openai/whisper-large-v3",
).strip()
TOGETHER_REALTIME_MODEL = TOGETHER_ASR_MODEL
TOGETHER_REALTIME_URL = os.getenv(
    "TOGETHER_REALTIME_URL",
    "wss://api.together.ai/v1/realtime",
).strip()
TOGETHER_ASR_TIMEOUT_SECONDS = float(
    os.getenv("TOGETHER_ASR_TIMEOUT_SECONDS", "120")
)
TOGETHER_ASR_MAX_RETRIES = max(
    0, int(os.getenv("TOGETHER_ASR_MAX_RETRIES", "2"))
)
TOGETHER_ASR_FALLBACK_WAIT_SECONDS = max(
    0.0, float(os.getenv("TOGETHER_ASR_FALLBACK_WAIT_SECONDS", "5"))
)
TOGETHER_ASR_MAX_WAIT_SECONDS = max(
    0.0, float(os.getenv("TOGETHER_ASR_MAX_WAIT_SECONDS", "30"))
)
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "").strip()
ELEVENLABS_ASR_BASE_URL = os.getenv(
    "ELEVENLABS_ASR_BASE_URL",
    "https://api.elevenlabs.io",
).rstrip("/")
ELEVENLABS_ASR_MODEL = os.getenv(
    "ELEVENLABS_ASR_MODEL",
    "scribe_v2",
).strip()
ELEVENLABS_ASR_TIMEOUT_SECONDS = float(
    os.getenv("ELEVENLABS_ASR_TIMEOUT_SECONDS", "180")
)
ELEVENLABS_ASR_MAX_RETRIES = max(
    0, int(os.getenv("ELEVENLABS_ASR_MAX_RETRIES", "2"))
)
ELEVENLABS_ASR_FALLBACK_WAIT_SECONDS = max(
    0.0, float(os.getenv("ELEVENLABS_ASR_FALLBACK_WAIT_SECONDS", "5"))
)
ELEVENLABS_ASR_MAX_WAIT_SECONDS = max(
    0.0, float(os.getenv("ELEVENLABS_ASR_MAX_WAIT_SECONDS", "30"))
)
TOGETHER_REALTIME_MAX_RETRIES = max(
    0, int(os.getenv("TOGETHER_REALTIME_MAX_RETRIES", "1"))
)
TOGETHER_REALTIME_CHUNK_SECONDS = max(
    10.0,
    float(os.getenv("TOGETHER_REALTIME_CHUNK_SECONDS", "10")),
)
TOGETHER_REALTIME_FRAME_BYTES = max(
    1024,
    int(os.getenv("TOGETHER_REALTIME_FRAME_BYTES", "4096")),
)
TOGETHER_BATCH_CHUNK_SECONDS = max(
    30.0,
    float(os.getenv("TOGETHER_BATCH_CHUNK_SECONDS", "600")),
)
TOGETHER_MULTILINGUAL_SOURCE_CHUNK_SECONDS = max(
    10.0,
    min(
        60.0,
        float(os.getenv("TOGETHER_MULTILINGUAL_SOURCE_CHUNK_SECONDS", "30")),
    ),
)
TOGETHER_DIARIZATION_MIN_SPEAKERS = max(
    1,
    int(os.getenv("TOGETHER_DIARIZATION_MIN_SPEAKERS", "1")),
)
TOGETHER_DIARIZATION_MAX_SPEAKERS = max(
    TOGETHER_DIARIZATION_MIN_SPEAKERS,
    int(os.getenv("TOGETHER_DIARIZATION_MAX_SPEAKERS", "5")),
)
YOUTUBE_WHISPER_VAD_FILTER = os.getenv(
    "YOUTUBE_WHISPER_VAD_FILTER",
    "true",
).strip().lower() in {"1", "true", "yes", "on"}
YOUTUBE_WHISPER_HALLUCINATION_SILENCE_SECONDS = float(
    os.getenv("YOUTUBE_WHISPER_HALLUCINATION_SILENCE_SECONDS", "1.0")
)
YOUTUBE_WHISPER_LOW_CONFIDENCE_THRESHOLD = float(
    os.getenv("YOUTUBE_WHISPER_LOW_CONFIDENCE_THRESHOLD", "0.35")
)
YOUTUBE_WHISPER_VAD_PARAMETERS = {
    "min_silence_duration_ms": 2000,
    "speech_pad_ms": 400,
}
YOUTUBE_WHISPER_STREAM_CHUNK_SECONDS = max(
    10.0,
    float(os.getenv("YOUTUBE_WHISPER_STREAM_CHUNK_SECONDS", "30")),
)
YOUTUBE_WHISPER_INITIAL_CHUNK_SECONDS = min(
    YOUTUBE_WHISPER_STREAM_CHUNK_SECONDS,
    max(5.0, float(os.getenv("YOUTUBE_WHISPER_INITIAL_CHUNK_SECONDS", "15"))),
)
YOUTUBE_WHISPER_STREAM_OVERLAP_SECONDS = min(
    YOUTUBE_WHISPER_STREAM_CHUNK_SECONDS / 2,
    max(0.0, float(os.getenv("YOUTUBE_WHISPER_STREAM_OVERLAP_SECONDS", "2"))),
)
YOUTUBE_WHISPER_STREAM_QUEUE_SIZE = max(
    1,
    int(os.getenv("YOUTUBE_WHISPER_STREAM_QUEUE_SIZE", "2")),
)
YOUTUBE_WHISPER_STREAM_PREFETCH_CHUNKS = max(
    1,
    min(
        YOUTUBE_WHISPER_STREAM_QUEUE_SIZE,
        int(os.getenv("YOUTUBE_WHISPER_STREAM_PREFETCH_CHUNKS", "2")),
    ),
)
YOUTUBE_WHISPER_LANGUAGE_DETECT_SECONDS = max(
    10.0,
    float(os.getenv("YOUTUBE_WHISPER_LANGUAGE_DETECT_SECONDS", "30")),
)


def dashboard_percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    rank = max(0, min(len(ordered) - 1, math.ceil(percentile * len(ordered)) - 1))
    return round(ordered[rank], 3)


_DASHBOARD_STATUS_PRIORITY = {
    "running": 4,
    "queued_for_transcription": 3,
    "awaiting_language_confirmation": 2,
    "queued": 2,
    "failed": 1,
    "cancelled": 0,
    "done": 0,
}


def dashboard_job_sort_key(item: tuple[str, dict[str, Any]]) -> tuple[int, float]:
    """Prioritize active jobs while keeping newest jobs first per state."""
    _job_id, job = item
    status = str(job.get("status") or "").strip().lower()
    return (
        _DASHBOARD_STATUS_PRIORITY.get(status, 0),
        float(job.get("created_at") or 0.0),
    )


def dashboard_request_authorized(
    configured_token: str,
    authorization: str,
    client_host: str,
) -> bool:
    scheme, _, supplied_token = authorization.partition(" ")
    if configured_token:
        return scheme.lower() == "bearer" and secrets.compare_digest(
            supplied_token,
            configured_token,
        )
    return client_host in {"127.0.0.1", "::1", "localhost"}


def new_job_telemetry() -> dict[str, Any]:
    now = time.time()
    return {
        "phase": "queued",
        "last_activity_at": now,
        "transcription": {
            "chunk_count": 0,
            "processed_seconds": 0.0,
            "progress_percent": 0.0,
            "segments_emitted": 0,
            "last_chunk_ms": None,
            "average_chunk_ms": None,
            "max_chunk_ms": None,
            "last_input_wait_ms": None,
            "average_input_wait_ms": None,
            "max_input_wait_ms": None,
            "last_scheduler_wait_ms": None,
            "average_scheduler_wait_ms": None,
            "max_scheduler_wait_ms": None,
            "last_inference_ms": None,
            "average_inference_ms": None,
            "max_inference_ms": None,
            "last_event_emit_ms": None,
            "average_event_emit_ms": None,
            "max_event_emit_ms": None,
            "processing_speed_x": None,
            "last_chunk_at": None,
        },
        "translation": {
            "active_requests": 0,
            "requests_total": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "grouping_requests": 0,
            "translation_requests": 0,
            "source_ids_seen": set(),
            "source_ids_succeeded": set(),
            "latencies_ms": [],
            "last_latency_ms": None,
            "last_status_code": None,
            "last_provider": None,
            "last_request_at": None,
            "prompt_tokens": 0,
            "output_tokens": 0,
            "estimated_cost_usd": 0.0,
            "estimated_cost_twd": 0.0,
        },
    }
VIDEO_UPLOAD_MAX_BYTES = int(
    os.getenv("VIDEO_UPLOAD_MAX_BYTES", str(2 * 1024 * 1024 * 1024))
)
VIDEO_UPLOAD_CHUNK_BYTES = max(
    5 * 1024 * 1024,
    min(
        50 * 1024 * 1024,
        int(os.getenv("VIDEO_UPLOAD_CHUNK_BYTES", str(20 * 1024 * 1024))),
    ),
)
VIDEO_UPLOAD_SESSION_TTL_SECONDS = max(
    600,
    int(os.getenv("VIDEO_UPLOAD_SESSION_TTL_SECONDS", "7200")),
)
VIDEO_UPLOAD_BATCH_MAX_FILES = max(
    1,
    int(os.getenv("VIDEO_UPLOAD_BATCH_MAX_FILES", "10")),
)
VIDEO_UPLOAD_EXTENSIONS = {
    ".avi",
    ".m4v",
    ".mkv",
    ".mov",
    ".mp4",
    ".mpeg",
    ".mpg",
    ".ts",
    ".webm",
    ".wmv",
}
TRANSCRIPTION_MODE_BEAM_SIZES = {
    "accurate": 5,
    "fast": 1,
}
ELEVENLABS_MAX_KEYTERMS = 1000
ELEVENLABS_MAX_KEYTERM_CHARACTERS = 50


def normalize_transcription_mode(value: Optional[str]) -> str:
    mode = str(value or "accurate").strip().lower()
    if mode not in TRANSCRIPTION_MODE_BEAM_SIZES:
        raise HTTPException(status_code=422, detail="不支援的轉譯模式")
    return mode


def validate_speaker_diarization(
    processing_profile: str,
    enabled: bool,
) -> None:
    del processing_profile
    if enabled:
        raise HTTPException(
            status_code=422,
            detail="講者辨識暫時停用；目前先測試 Together 逐字時間",
        )


class YoutubeLiveRequest(BaseModel):
    url: str
    language: str = ""
    captcha_token: str = ""
    ignore_subtitles: bool = False
    include_word_timestamps: bool = False
    speaker_diarization: bool = False
    transcription_mode: str = "accurate"
    processing_profile: str = "standard"
    asr_provider: str = ""
    elevenlabs_mode: str = "chunks"
    elevenlabs_keyterms: list[str] = Field(default_factory=list)


class YoutubeLanguageSelection(BaseModel):
    language: str


class YoutubeCancelRequest(BaseModel):
    cancel_token: str


class VideoUploadFileSessionRequest(BaseModel):
    filename: str
    size_bytes: int
    content_type: str = ""
    last_modified: Optional[int] = None


class VideoUploadBatchSessionRequest(BaseModel):
    files: list[VideoUploadFileSessionRequest]
    captcha_token: str = ""


class VideoUploadCompleteRequest(BaseModel):
    language: str = ""
    include_word_timestamps: bool = True
    speaker_diarization: bool = False
    transcription_mode: str = "accurate"
    processing_profile: str = "standard"
    asr_provider: str = ""
    elevenlabs_mode: str = "chunks"
    elevenlabs_keyterms: list[str] = Field(default_factory=list)


def video_upload_chunk_count(size_bytes: int, chunk_bytes: int) -> int:
    if size_bytes <= 0 or chunk_bytes <= 0:
        return 0
    return math.ceil(size_bytes / chunk_bytes)


def expected_video_upload_chunk_bytes(
    size_bytes: int,
    chunk_bytes: int,
    chunk_index: int,
) -> int:
    chunk_count = video_upload_chunk_count(size_bytes, chunk_bytes)
    if chunk_index < 0 or chunk_index >= chunk_count:
        raise ValueError("invalid upload chunk index")
    return min(chunk_bytes, size_bytes - (chunk_index * chunk_bytes))


def completed_video_upload_chunks(
    chunks_dir: Path,
    size_bytes: int,
    chunk_bytes: int,
) -> tuple[list[int], int]:
    completed: list[int] = []
    uploaded_bytes = 0
    for chunk_index in range(video_upload_chunk_count(size_bytes, chunk_bytes)):
        part_path = chunks_dir / f"{chunk_index:08d}.part"
        expected_bytes = expected_video_upload_chunk_bytes(
            size_bytes,
            chunk_bytes,
            chunk_index,
        )
        if part_path.is_file() and part_path.stat().st_size == expected_bytes:
            completed.append(chunk_index)
            uploaded_bytes += expected_bytes
    return completed, uploaded_bytes


def assemble_video_upload_chunks(
    chunks_dir: Path,
    output_path: Path,
    size_bytes: int,
    chunk_bytes: int,
) -> None:
    temporary_path = output_path.with_suffix(output_path.suffix + ".assembling")
    try:
        with temporary_path.open("wb") as output:
            for chunk_index in range(
                video_upload_chunk_count(size_bytes, chunk_bytes)
            ):
                part_path = chunks_dir / f"{chunk_index:08d}.part"
                expected_bytes = expected_video_upload_chunk_bytes(
                    size_bytes,
                    chunk_bytes,
                    chunk_index,
                )
                if not part_path.is_file() or part_path.stat().st_size != expected_bytes:
                    raise RuntimeError(f"缺少上傳分塊 {chunk_index}")
                with part_path.open("rb") as part:
                    shutil.copyfileobj(part, output, length=1024 * 1024)
        if temporary_path.stat().st_size != size_bytes:
            raise RuntimeError("合併後的影片大小不正確")
        temporary_path.replace(output_path)
    finally:
        temporary_path.unlink(missing_ok=True)


def sse_message(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def segment_payload(
    index: int,
    start: Optional[float],
    end: Optional[float],
    text: str,
    language: Optional[str] = None,
    low_confidence_spans: Optional[list[str]] = None,
    words: Optional[list[dict[str, Any]]] = None,
    speaker_id: Optional[str] = None,
) -> dict[str, Any]:
    payload = {
        "index": index,
        "start": start,
        "end": end,
        "text": text.strip(),
        "language": language,
    }
    if low_confidence_spans:
        payload["low_confidence_spans"] = low_confidence_spans
    if words is not None:
        payload["words"] = words
    if speaker_id:
        payload["speaker_id"] = speaker_id
    return payload


def whisper_low_confidence_spans(
    segment: Any,
    text: str,
    language: Optional[str],
    words: Optional[list[Any]] = None,
) -> list[str]:
    if str(language or "").lower().split("-", 1)[0] != "ko":
        return []

    spans: list[str] = []
    source_words = words if words is not None else getattr(segment, "words", None) or []
    for word in source_words:
        value = str(getattr(word, "word", "") or "").strip()
        probability = getattr(word, "probability", None)
        if (
            value
            and value in text
            and probability is not None
            and float(probability) < YOUTUBE_WHISPER_LOW_CONFIDENCE_THRESHOLD
            and value not in spans
        ):
            spans.append(value)
    return spans[:20]


def whisper_word_payloads(
    segment: Any,
    language: Optional[str],
    time_offset: float = 0.0,
    words: Optional[list[Any]] = None,
) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    previous_start = float("-inf")
    source_words = words if words is not None else getattr(segment, "words", None) or []
    for word in source_words:
        raw_value = str(getattr(word, "word", "") or "")
        value = to_traditional_chinese(
            raw_value,
            language,
        )
        if not value.strip():
            continue
        try:
            start = float(getattr(word, "start", None))
            end = float(getattr(word, "end", None))
        except (TypeError, ValueError):
            continue
        if not math.isfinite(start) or not math.isfinite(end) or end < start:
            continue
        start = max(start, previous_start)
        end = max(end, start + 0.001)
        previous_start = start
        payload: dict[str, Any] = {
            "word": value,
            "start": round(start + time_offset, 3),
            "end": round(end + time_offset, 3),
        }
        probability = getattr(word, "probability", None)
        if probability is not None:
            try:
                probability_value = float(probability)
            except (TypeError, ValueError):
                probability_value = None
            if probability_value is not None and math.isfinite(probability_value):
                payload["probability"] = round(probability_value, 6)
        speaker_id = str(getattr(word, "speaker_id", "") or "").strip()
        if speaker_id:
            payload["speaker_id"] = speaker_id
        payloads.append(payload)
    return payloads


def owned_whisper_segment(
    segment: Any,
    ownership_start: float,
    ownership_end: float,
    is_final_chunk: bool,
) -> Optional[tuple[str, float, float, list[Any]]]:
    """Return only the portion of a Whisper segment owned by this audio chunk.

    Adjacent chunks overlap. Assigning ownership to a complete Whisper segment can
    discard speech whenever that segment crosses the right boundary. Word
    midpoints provide a stable, lossless boundary: the left chunk owns words
    before the boundary and the right chunk owns words at or after it.

    If Whisper did not provide a complete set of usable word timestamps, the
    segment midpoint is used as a conservative fallback so the whole segment is
    emitted by exactly one chunk.
    """

    segment_start = float(getattr(segment, "start", 0.0) or 0.0)
    segment_end = float(getattr(segment, "end", 0.0) or 0.0)
    raw_words = [
        word
        for word in (getattr(segment, "words", None) or [])
        if str(getattr(word, "word", "") or "").strip()
    ]
    timed_words: list[tuple[Any, float, float]] = []
    for word in raw_words:
        try:
            word_start = float(getattr(word, "start", None))
            word_end = float(getattr(word, "end", None))
        except (TypeError, ValueError):
            break
        if (
            not math.isfinite(word_start)
            or not math.isfinite(word_end)
            or word_end < word_start
        ):
            break
        timed_words.append((word, word_start, word_end))

    if raw_words and len(timed_words) == len(raw_words):
        owned_words = [
            (word, word_start, word_end)
            for word, word_start, word_end in timed_words
            if (word_start + word_end) / 2 >= ownership_start
            and (
                is_final_chunk
                or (word_start + word_end) / 2 < ownership_end
            )
        ]
        if not owned_words:
            return None
        if len(owned_words) == len(timed_words):
            text = str(getattr(segment, "text", "") or "").strip()
        else:
            text = "".join(
                str(getattr(word, "word", "") or "")
                for word, _, _ in owned_words
            ).strip()
        if not text:
            return None
        return (
            text,
            owned_words[0][1],
            max(word_end for _, _, word_end in owned_words),
            [word for word, _, _ in owned_words],
        )

    segment_midpoint = (segment_start + segment_end) / 2
    if segment_midpoint < ownership_start:
        return None
    if not is_final_chunk and segment_midpoint >= ownership_end:
        return None
    text = str(getattr(segment, "text", "") or "").strip()
    if not text:
        return None
    return text, segment_start, segment_end, []


def transcription_progress_payload(
    audio_duration: float,
    processed_seconds: float,
    elapsed_seconds: float,
    current_timestamp: Optional[float] = None,
) -> dict[str, Any]:
    duration = max(0.0, float(audio_duration or 0.0))
    processed = max(0.0, float(processed_seconds or 0.0))
    elapsed = max(0.0, float(elapsed_seconds or 0.0))
    progress_percent = (
        min(100.0, processed / duration * 100)
        if duration > 0
        else 0.0
    )
    processing_speed = processed / elapsed if elapsed > 0 and processed > 0 else None
    remaining_seconds: Optional[float] = None
    completion_at: Optional[str] = None

    if (
        duration > 0
        and processing_speed is not None
        and elapsed >= ETA_MIN_ELAPSED_SECONDS
        and processed >= ETA_MIN_PROCESSED_SECONDS
    ):
        remaining_seconds = max(0.0, duration - processed) / processing_speed
        completion_timestamp = (
            current_timestamp if current_timestamp is not None else time.time()
        ) + remaining_seconds
        completion_at = (
            datetime.fromtimestamp(completion_timestamp, timezone.utc)
            .isoformat(timespec="seconds")
            .replace("+00:00", "Z")
        )

    return {
        "progress_percent": round(progress_percent, 2),
        "elapsed_seconds": round(elapsed, 3),
        "processing_speed_x": (
            round(processing_speed, 3)
            if processing_speed is not None
            else None
        ),
        "estimated_remaining_seconds": (
            round(remaining_seconds, 1)
            if remaining_seconds is not None
            else None
        ),
        "estimated_completion_at": completion_at,
    }


def chapter_payloads(video_info: dict[str, Any]) -> list[dict[str, Any]]:
    chapters = video_info.get("chapters") or []
    payloads: list[dict[str, Any]] = []
    for index, chapter in enumerate(chapters, start=1):
        title = str(chapter.get("title") or f"Chapter {index}").strip()
        start = chapter.get("start_time")
        end = chapter.get("end_time")
        payloads.append(
            {
                "index": index,
                "title": title,
                "start": float(start) if start is not None else None,
                "end": float(end) if end is not None else None,
            }
        )
    return payloads


def cleanup_youtube_live_jobs(jobs: dict[str, dict[str, Any]]) -> None:
    now = time.time()
    expired_ids = [
        job_id
        for job_id, job in jobs.items()
        if job.get("expires_at", 0) <= now
        and job.get("status") not in {"running", "queued", "queued_for_transcription"}
    ]
    for job_id in expired_ids:
        job = jobs.pop(job_id, None)
        if job:
            if job.get("status") == "awaiting_language_confirmation":
                queue = job.get("queue")
                if queue is not None:
                    queue.put_nowait(
                        {
                            "event": "failed",
                            "data": {"message": "語言確認已逾時，請重新建立轉譯任務"},
                        }
                    )
                    queue.put_nowait({"event": "close", "data": {}})
            cleanup_youtube_live_job_artifacts(job)


def cleanup_youtube_live_job_artifacts(job: dict[str, Any]) -> None:
    work_dir = job.pop("work_dir", None)
    job.pop("audio_path", None)
    job.pop("audio_source", None)
    job.pop("subtitle_segments", None)
    if work_dir:
        shutil.rmtree(work_dir, ignore_errors=True)


def readable_exception_message(exc: Exception) -> str:
    if isinstance(exc, HTTPException):
        return str(exc.detail)
    if is_youtube_rate_limit_error(exc):
        return YOUTUBE_RATE_LIMIT_MESSAGE
    return str(exc)


def put_thread_event(loop: asyncio.AbstractEventLoop, queue: asyncio.Queue[dict[str, Any]], event: dict[str, Any]) -> None:
    asyncio.run_coroutine_threadsafe(queue.put(event), loop).result()


def create_groq_whisper_client() -> GroqWhisperClient:
    return GroqWhisperClient(
        GroqWhisperSettings(
            api_key=GROQ_API_KEY,
            base_url=GROQ_ASR_BASE_URL,
            model=GROQ_ASR_MODEL,
            timeout_seconds=GROQ_ASR_TIMEOUT_SECONDS,
            max_retries=GROQ_ASR_MAX_RETRIES,
            fallback_wait_seconds=GROQ_ASR_FALLBACK_WAIT_SECONDS,
            max_wait_seconds=GROQ_ASR_MAX_WAIT_SECONDS,
            min_request_interval_seconds=GROQ_ASR_MIN_REQUEST_INTERVAL_SECONDS,
        )
    )


def create_cloudflare_whisper_client(
    *,
    scheduler_managed: bool = True,
) -> CloudflareWhisperClient:
    return CloudflareWhisperClient(
        CloudflareWhisperSettings(
            account_id=CLOUDFLARE_ACCOUNT_ID,
            api_token=CLOUDFLARE_API_TOKEN,
            base_url=CLOUDFLARE_ASR_BASE_URL,
            model=CLOUDFLARE_ASR_MODEL,
            timeout_seconds=CLOUDFLARE_ASR_TIMEOUT_SECONDS,
            max_retries=0 if scheduler_managed else CLOUDFLARE_ASR_MAX_RETRIES,
            fallback_wait_seconds=CLOUDFLARE_ASR_FALLBACK_WAIT_SECONDS,
            max_wait_seconds=CLOUDFLARE_ASR_MAX_WAIT_SECONDS,
        )
    )


def create_together_whisper_client() -> TogetherWhisperClient:
    return TogetherWhisperClient(
        TogetherWhisperSettings(
            api_key=TOGETHER_API_KEY,
            base_url=TOGETHER_ASR_BASE_URL,
            model=TOGETHER_ASR_MODEL,
            timeout_seconds=TOGETHER_ASR_TIMEOUT_SECONDS,
            max_retries=TOGETHER_ASR_MAX_RETRIES,
            fallback_wait_seconds=TOGETHER_ASR_FALLBACK_WAIT_SECONDS,
            max_wait_seconds=TOGETHER_ASR_MAX_WAIT_SECONDS,
        )
    )


def create_elevenlabs_whisper_client() -> ElevenLabsWhisperClient:
    return ElevenLabsWhisperClient(
        ElevenLabsWhisperSettings(
            api_key=ELEVENLABS_API_KEY,
            base_url=ELEVENLABS_ASR_BASE_URL,
            model=ELEVENLABS_ASR_MODEL,
            timeout_seconds=ELEVENLABS_ASR_TIMEOUT_SECONDS,
            max_retries=ELEVENLABS_ASR_MAX_RETRIES,
            fallback_wait_seconds=ELEVENLABS_ASR_FALLBACK_WAIT_SECONDS,
            max_wait_seconds=ELEVENLABS_ASR_MAX_WAIT_SECONDS,
        )
    )


def create_remote_whisper_client(provider: str) -> WhisperAsrClient:
    if provider == "cloudflare":
        return create_cloudflare_whisper_client()
    if provider == "groq":
        return create_groq_whisper_client()
    if provider == "together":
        return create_together_whisper_client()
    if provider == "elevenlabs":
        return create_elevenlabs_whisper_client()
    raise ValueError(f"不支援的遠端 ASR provider: {provider}")


def _cancelable_sleep(
    seconds: float,
    cancel_check: Callable[[], bool] | None,
) -> None:
    deadline = time.perf_counter() + max(0.0, seconds)
    while True:
        if cancel_check is not None and cancel_check():
            raise TranscriptionCancelled("轉譯已取消")
        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            return
        time.sleep(min(0.25, remaining))


def transcribe_remote_audio(
    asr_provider: str,
    asr_client: WhisperAsrClient,
    audio_samples: np.ndarray,
    *,
    language: str | None,
    beam_size: int,
    vad_filter: bool,
    condition_on_previous_text: bool,
    hallucination_silence_threshold: float | None,
    job_id: str | None,
    cancel_check: Callable[[], bool] | None,
    elevenlabs_keyterms: list[str] | None = None,
) -> tuple[list[Any], Any, float]:
    options = {
        "language": language,
        "beam_size": beam_size,
        "vad_filter": vad_filter,
        "condition_on_previous_text": condition_on_previous_text,
        "hallucination_silence_threshold": hallucination_silence_threshold,
    }
    if asr_provider == "elevenlabs":
        options["keyterms"] = elevenlabs_keyterms or []
    if asr_provider != "cloudflare" or not job_id:
        segments, info = asr_client.transcribe(audio_samples, **options)
        return segments, info, 0.0

    scheduler_wait_ms = 0.0
    for attempt in range(CLOUDFLARE_ASR_MAX_RETRIES + 1):
        try:
            scheduled = cloudflare_asr_scheduler.execute(
                str(job_id),
                lambda: asr_client.transcribe(audio_samples, **options),
                cancel_check,
            )
        except CloudflareAsrRequestCancelled as exc:
            raise TranscriptionCancelled("轉譯已取消") from exc
        except requests.HTTPError as exc:
            response = exc.response
            if (
                response is None
                or response.status_code != 429
                or attempt >= CLOUDFLARE_ASR_MAX_RETRIES
            ):
                raise
            wait_seconds = retry_after_seconds(response)
            if wait_seconds is None:
                wait_seconds = CLOUDFLARE_ASR_FALLBACK_WAIT_SECONDS
            if wait_seconds > CLOUDFLARE_ASR_MAX_WAIT_SECONDS:
                raise
            _cancelable_sleep(wait_seconds, cancel_check)
            continue
        scheduler_wait_ms += scheduled.queue_wait_ms
        segments, info = scheduled.value
        return segments, info, scheduler_wait_ms
    raise AssertionError("Cloudflare ASR retry loop did not return")


TOGETHER_LANGUAGE_TAG_PATTERN = re.compile(
    r"<(?P<language>[a-z]{2,3})(?:-[A-Za-z]{2,4})?>",
    re.IGNORECASE,
)


def normalize_together_multilingual_text(
    transcript: str,
) -> tuple[str, str | None]:
    value = str(transcript or "").strip()
    tagged_languages = [
        match.group("language").lower()
        for match in TOGETHER_LANGUAGE_TAG_PATTERN.finditer(value)
    ]
    value = TOGETHER_LANGUAGE_TAG_PATTERN.sub("", value).strip()
    tagged_languages = list(dict.fromkeys(tagged_languages))
    if len(tagged_languages) == 1:
        return value, tagged_languages[0]
    if len(tagged_languages) > 1:
        return value, "auto"

    script_languages: list[str] = []
    if re.search(r"[\u3040-\u30ff]", value):
        script_languages.append("ja")
    if re.search(r"[\uac00-\ud7af]", value):
        script_languages.append("ko")
    if re.search(r"[\u4e00-\u9fff]", value) and "ja" not in script_languages:
        script_languages.append("zh")
    script_languages = list(dict.fromkeys(script_languages))
    if len(script_languages) == 1:
        return value, script_languages[0]
    if len(script_languages) > 1:
        return value, "auto"
    if re.search(r"[A-Za-z]", value):
        return value, "en"
    return value, None


def float_audio_to_pcm16(audio_samples: np.ndarray) -> bytes:
    clipped = np.clip(audio_samples, -1.0, 32767.0 / 32768.0)
    return (clipped * 32768.0).astype("<i2").tobytes()


def transcribe_multilingual_diarized_audio(
    together_client: WhisperAsrClient,
    audio_samples: np.ndarray,
    *,
    beam_size: int,
    job_id: str | None,
    cancel_check: Callable[[], bool] | None,
) -> tuple[list[Any], Any, float]:
    del beam_size, job_id
    transcribe_diarized = getattr(together_client, "transcribe_diarized", None)
    if transcribe_diarized is None:
        raise RuntimeError("Together Batch ASR client 尚未建立")
    speaker_segments, _ = transcribe_diarized(
        audio_samples,
        language=None,
        min_speakers=TOGETHER_DIARIZATION_MIN_SPEAKERS,
        max_speakers=TOGETHER_DIARIZATION_MAX_SPEAKERS,
    )
    source_segments: list[Any] = []
    detected_languages: list[str] = []
    source_transcriber = TogetherRealtimeTranscriber(
        TogetherRealtimeSettings(
            api_key=TOGETHER_API_KEY,
            websocket_url=TOGETHER_REALTIME_URL,
            model=TOGETHER_REALTIME_MODEL,
            timeout_seconds=TOGETHER_ASR_TIMEOUT_SECONDS,
            max_retries=TOGETHER_REALTIME_MAX_RETRIES,
            frame_bytes=TOGETHER_REALTIME_FRAME_BYTES,
        ),
        language="auto",
    )
    max_source_samples = max(
        16000,
        int(TOGETHER_MULTILINGUAL_SOURCE_CHUNK_SECONDS * 16000),
    )
    try:
        for speaker_segment in sorted(
            speaker_segments,
            key=lambda item: float(getattr(item, "start", 0.0)),
        ):
            speaker_id = str(
                getattr(speaker_segment, "speaker_id", "") or ""
            ).strip() or None
            start_sample = max(
                0,
                int(float(getattr(speaker_segment, "start", 0.0)) * 16000),
            )
            end_sample = min(
                len(audio_samples),
                max(
                    start_sample,
                    int(float(getattr(speaker_segment, "end", 0.0)) * 16000),
                ),
            )
            for part_start in range(start_sample, end_sample, max_source_samples):
                if cancel_check is not None and cancel_check():
                    raise TranscriptionCancelled("轉譯已取消")
                part_end = min(end_sample, part_start + max_source_samples)
                if part_end <= part_start:
                    continue
                try:
                    source_result = source_transcriber.transcribe(
                        float_audio_to_pcm16(audio_samples[part_start:part_end]),
                        cancel_check=cancel_check,
                    )
                except TogetherRealtimeCancelled as exc:
                    raise TranscriptionCancelled("轉譯已取消") from exc
                source_text, source_language = normalize_together_multilingual_text(
                    source_result.text
                )
                if not source_text:
                    continue
                if source_language and source_language != "auto":
                    detected_languages.append(source_language)
                source_segments.append(
                    SimpleNamespace(
                        text=source_text,
                        start=part_start / 16000,
                        end=part_end / 16000,
                        words=[],
                        speaker_id=speaker_id,
                        language=source_language,
                        avg_logprob=0.0,
                        no_speech_prob=0.0,
                    )
                )
    finally:
        source_transcriber.close()
    dominant_language = (
        max(dict.fromkeys(detected_languages), key=detected_languages.count)
        if detected_languages
        else None
    )
    return source_segments, SimpleNamespace(
        language=dominant_language,
        language_probability=None,
    ), 0.0


def asr_source_name(provider: str) -> str:
    if provider == "cloudflare":
        return "cloudflare_whisper"
    if provider == "groq":
        return "groq_whisper"
    if provider == "together":
        return "together_whisper_http"
    if provider == "elevenlabs":
        return "elevenlabs_scribe_v2"
    return "whisper"


def normalize_elevenlabs_transcription_mode(value: str | None) -> str:
    mode = str(value or "chunks").strip().lower()
    if mode not in {"chunks", "full"}:
        raise ValueError("ElevenLabs 模式必須是 chunks 或 full")
    return mode


def normalize_elevenlabs_keyterms(value: Any) -> list[str]:
    if value is None:
        return []
    values = value.splitlines() if isinstance(value, str) else list(value)
    normalized: list[str] = []
    seen: set[str] = set()
    for item in values:
        keyterm = str(item or "").strip()
        if not keyterm:
            continue
        if len(keyterm) > ELEVENLABS_MAX_KEYTERM_CHARACTERS:
            raise ValueError(
                "ElevenLabs 每個關鍵詞最多 "
                f"{ELEVENLABS_MAX_KEYTERM_CHARACTERS} 個字元"
            )
        if keyterm in seen:
            continue
        seen.add(keyterm)
        normalized.append(keyterm)
    if len(normalized) > ELEVENLABS_MAX_KEYTERMS:
        raise ValueError(
            f"ElevenLabs 完整檔案模式最多 {ELEVENLABS_MAX_KEYTERMS} 個關鍵詞"
        )
    return normalized


def video_transcribe_task(
    job_id: str,
    processing_profile: str | None,
    asr_provider: str | None = None,
) -> dict[str, str]:
    profile = normalize_processing_profile(processing_profile)
    provider = asr_provider_for_profile(profile, asr_provider)
    return {
        "kind": "youtube_live",
        "id": job_id,
        "worker_group": (
            "remote_asr" if provider in REMOTE_ASR_PROVIDERS else "default"
        ),
        "asr_provider": provider,
    }


def remote_vad_speech_intervals(audio_samples: np.ndarray) -> list[tuple[float, float]]:
    if not YOUTUBE_WHISPER_VAD_FILTER:
        duration = len(audio_samples) / 16000
        return [(0.0, duration)] if duration > 0 else []
    from faster_whisper.vad import VadOptions, get_speech_timestamps

    timestamps = get_speech_timestamps(
        audio_samples,
        VadOptions(**YOUTUBE_WHISPER_VAD_PARAMETERS),
        sampling_rate=16000,
    )
    return [
        (float(item["start"]) / 16000, float(item["end"]) / 16000)
        for item in timestamps
        if item.get("end", 0) > item.get("start", 0)
    ]


groq_vad_speech_intervals = remote_vad_speech_intervals


def overlaps_speech(
    start: float,
    end: float,
    speech_intervals: list[tuple[float, float]],
) -> bool:
    return any(
        start < speech_end and end > speech_start
        for speech_start, speech_end in speech_intervals
    )


def transcribe_audio_stream(
    auto2lrc,
    audio_path: MediaSourceInput,
    language_hint: Optional[str],
    loop: asyncio.AbstractEventLoop,
    queue: asyncio.Queue[dict[str, Any]],
    job_id: Optional[str] = None,
    cancel_check: Optional[Callable[[], bool]] = None,
    include_word_timestamps: bool = False,
    audio_duration_hint: Optional[float] = None,
    transcription_mode: str = "accurate",
    asr_provider: str = "local",
    asr_client: WhisperAsrClient | None = None,
    telemetry_callback: Callable[[dict[str, Any]], None] | None = None,
    detected_language_fallback: str | None = None,
    speaker_diarization: bool = False,
    elevenlabs_keyterms: list[str] | None = None,
) -> dict[str, Any]:
    started_at = time.perf_counter()
    mode = (
        transcription_mode
        if transcription_mode in TRANSCRIPTION_MODE_BEAM_SIZES
        else "accurate"
    )
    beam_size = TRANSCRIPTION_MODE_BEAM_SIZES[mode]
    remote_asr = asr_provider in REMOTE_ASR_PROVIDERS
    if speaker_diarization:
        raise RuntimeError("講者辨識暫時停用；目前先測試 Together 逐字時間")
    active_asr_client = (
        asr_client or create_remote_whisper_client(asr_provider)
        if remote_asr
        else None
    )
    asr_model = (
        str(active_asr_client.model_name)
        if active_asr_client is not None
        else str(getattr(auto2lrc, "model_name", "faster-whisper"))
    )
    audio_duration = max(
        0.0,
        float(audio_duration_hint or probe_media_duration(audio_path) or 0.0),
    )
    detected_language = language_hint or detected_language_fallback
    remote_auto_language = (
        asr_provider in {"cloudflare", "elevenlabs", "together"}
        and not language_hint
    )
    stream_chunk_seconds = YOUTUBE_WHISPER_STREAM_CHUNK_SECONDS
    stream_initial_chunk_seconds = YOUTUBE_WHISPER_INITIAL_CHUNK_SECONDS
    stream_overlap_seconds = YOUTUBE_WHISPER_STREAM_OVERLAP_SECONDS
    language_probability = None
    segment_count = 0
    decoded_segment_count = 0
    chunk_count = 0
    processed_seconds = 0.0
    last_emitted_start = 0.0
    content_parts: list[str] = []
    fair_scheduler_enabled = asr_provider == "local" and bool(job_id)
    logger.info(
        "Whisper transcription started: job_id=%s asr_provider=%s "
        "asr_model=%s transcription_mode=%s "
        "beam_size=%d language_hint=%s "
        "temperature=0.0 condition_on_previous_text=false vad_filter=%s "
        "word_timestamps=true hallucination_silence_threshold=%.2f "
        "initial_chunk_seconds=%.1f chunk_seconds=%.1f "
        "overlap_seconds=%.1f pcm_queue_size=%d prefetch_chunks=%d "
        "speaker_diarization=%s",
        job_id or "unknown",
        asr_provider,
        asr_model,
        mode,
        beam_size,
        language_hint or "auto",
        YOUTUBE_WHISPER_VAD_FILTER,
        YOUTUBE_WHISPER_HALLUCINATION_SILENCE_SECONDS,
        stream_initial_chunk_seconds,
        stream_chunk_seconds,
        stream_overlap_seconds,
        YOUTUBE_WHISPER_STREAM_QUEUE_SIZE,
        YOUTUBE_WHISPER_STREAM_PREFETCH_CHUNKS,
        speaker_diarization,
    )

    try:
        if cancel_check is not None and cancel_check():
            raise TranscriptionCancelled("轉譯已取消")
        with FFmpegPcmChunkStream(
            audio_path,
            chunk_seconds=stream_chunk_seconds,
            initial_chunk_seconds=stream_initial_chunk_seconds,
            overlap_seconds=stream_overlap_seconds,
            queue_size=YOUTUBE_WHISPER_STREAM_QUEUE_SIZE,
            prefetch_chunks=YOUTUBE_WHISPER_STREAM_PREFETCH_CHUNKS,
            cancel_check=cancel_check,
        ) as audio_chunks:
            audio_chunk_iterator = iter(audio_chunks)
            while True:
                input_wait_started_at = time.perf_counter()
                try:
                    chunk = next(audio_chunk_iterator)
                except StopIteration:
                    break
                input_wait_ms = round(
                    (time.perf_counter() - input_wait_started_at) * 1000,
                    3,
                )
                if cancel_check is not None and cancel_check():
                    raise TranscriptionCancelled("轉譯已取消")
                scheduler_wait_ms = 0.0
                scheduler_turn_acquired = False
                if fair_scheduler_enabled:
                    scheduler_wait_ms = round(
                        fair_transcription_scheduler.acquire(
                            str(job_id),
                            cancel_check,
                        ),
                        3,
                    )
                    scheduler_turn_acquired = True
                chunk_started_at = time.perf_counter()
                inference_elapsed_seconds = 0.0
                event_emit_elapsed_seconds = 0.0
                decoded_before_chunk = decoded_segment_count
                emitted_before_chunk = segment_count
                chunk_count += 1
                audio_samples = np.frombuffer(
                    chunk.data,
                    dtype="<i2",
                ).astype(np.float32)
                audio_samples /= 32768.0
                logger.info(
                    "Whisper chunk started: job_id=%s chunk=%d "
                    "range=%.3f-%.3fs duration=%.3fs provider=%s "
                    "language=%s final=%s input_wait=%.3fms "
                    "scheduler_wait=%.3fms",
                    job_id or "unknown",
                    chunk_count,
                    chunk.offset_seconds,
                    chunk.offset_seconds + chunk.duration_seconds,
                    chunk.duration_seconds,
                    asr_provider,
                    detected_language or "auto",
                    chunk.is_final,
                    input_wait_ms,
                    scheduler_wait_ms,
                )
                speech_intervals: list[tuple[float, float]] = []
                inference_started_at = time.perf_counter()
                if remote_asr:
                    speech_intervals = remote_vad_speech_intervals(audio_samples)
                    if not speech_intervals:
                        segments = []
                        info = None
                    else:
                        if active_asr_client is None:  # pragma: no cover
                            raise RuntimeError("遠端 ASR client 尚未建立")
                        request_language = (
                            None if remote_auto_language else detected_language
                        )
                        segments, info, remote_scheduler_wait_ms = (
                            transcribe_remote_audio(
                                asr_provider,
                                active_asr_client,
                                audio_samples,
                                language=request_language,
                                beam_size=beam_size,
                                vad_filter=YOUTUBE_WHISPER_VAD_FILTER,
                                condition_on_previous_text=False,
                                hallucination_silence_threshold=(
                                    YOUTUBE_WHISPER_HALLUCINATION_SILENCE_SECONDS
                                ),
                                job_id=job_id,
                                cancel_check=cancel_check,
                                elevenlabs_keyterms=elevenlabs_keyterms,
                            )
                        )
                        scheduler_wait_ms += remote_scheduler_wait_ms
                else:
                    segments, info = auto2lrc.transcribe(
                        audio_samples,
                        beam_size=beam_size,
                        language=detected_language,
                        vad_filter=YOUTUBE_WHISPER_VAD_FILTER,
                        vad_parameters=(
                            YOUTUBE_WHISPER_VAD_PARAMETERS
                            if YOUTUBE_WHISPER_VAD_FILTER
                            else None
                        ),
                        word_timestamps=True,
                        hallucination_silence_threshold=(
                            YOUTUBE_WHISPER_HALLUCINATION_SILENCE_SECONDS
                        ),
                    )
                inference_elapsed_seconds += (
                    time.perf_counter() - inference_started_at
                )
                reported_chunk_language = getattr(info, "language", None)
                if remote_auto_language:
                    chunk_language = reported_chunk_language or detected_language
                    if detected_language is None and chunk_language:
                        detected_language = chunk_language
                elif asr_provider in {"cloudflare", "elevenlabs", "together"} and language_hint:
                    chunk_language = language_hint
                    detected_language = language_hint
                else:
                    chunk_language = reported_chunk_language or detected_language
                    if chunk_language:
                        detected_language = chunk_language
                if language_probability is None:
                    language_probability = getattr(info, "language_probability", None)

                ownership_start = (
                    stream_overlap_seconds / 2
                    if chunk.offset_seconds > 0
                    else 0.0
                )
                ownership_end = chunk.duration_seconds
                if not chunk.is_final:
                    ownership_end = max(
                        0.0,
                        chunk.duration_seconds
                        - stream_overlap_seconds / 2,
                    )
                segment_iterator = iter(segments)
                while True:
                    inference_started_at = time.perf_counter()
                    try:
                        segment = next(segment_iterator)
                    except StopIteration:
                        inference_elapsed_seconds += (
                            time.perf_counter() - inference_started_at
                        )
                        break
                    inference_elapsed_seconds += (
                        time.perf_counter() - inference_started_at
                    )
                    decoded_segment_count += 1
                    if cancel_check is not None and cancel_check():
                        raise TranscriptionCancelled("轉譯已取消")
                    if remote_asr and not overlaps_speech(
                        float(getattr(segment, "start", 0.0)),
                        float(getattr(segment, "end", 0.0)),
                        speech_intervals,
                    ):
                        continue
                    owned_segment = owned_whisper_segment(
                        segment,
                        ownership_start,
                        ownership_end,
                        chunk.is_final,
                    )
                    if owned_segment is None:
                        continue
                    raw_text, local_start, local_end, owned_words = owned_segment
                    segment_language = (
                        str(getattr(segment, "language", "") or "").strip()
                        or chunk_language
                    )
                    text = to_traditional_chinese(
                        raw_text,
                        segment_language,
                    )
                    if not text:
                        continue
                    global_start = chunk.offset_seconds + max(0.0, local_start)
                    global_end = chunk.offset_seconds + max(local_start, local_end)
                    timeline_adjusted = global_start + 0.001 < last_emitted_start
                    if timeline_adjusted:
                        logger.warning(
                            "Whisper segment timeline moved backwards; clamping start and "
                            "omitting word timeline: job_id=%s previous_start=%.3f "
                            "current_start=%.3f current_end=%.3f",
                            job_id or "unknown",
                            last_emitted_start,
                            global_start,
                            global_end,
                        )
                        global_start = last_emitted_start
                        global_end = max(global_end, global_start + 0.001)
                    segment_count += 1
                    content_parts.append(text)
                    payload = segment_payload(
                        segment_count,
                        global_start,
                        global_end,
                        text,
                        segment_language,
                        whisper_low_confidence_spans(
                            segment,
                            text,
                            segment_language,
                            owned_words,
                        ),
                        (
                            (
                                []
                                if timeline_adjusted
                                else whisper_word_payloads(
                                    segment,
                                    segment_language,
                                    chunk.offset_seconds,
                                    owned_words,
                                )
                            )
                            if include_word_timestamps
                            else None
                        ),
                        str(getattr(segment, "speaker_id", "") or "") or None,
                    )
                    last_emitted_start = global_start
                    payload.update(
                        transcription_progress_payload(
                            audio_duration,
                            global_end,
                            time.perf_counter() - started_at,
                        )
                    )
                    event_emit_started_at = time.perf_counter()
                    put_thread_event(
                        loop,
                        queue,
                        {
                            "event": "segment",
                            "data": payload,
                        },
                    )
                    event_emit_elapsed_seconds += (
                        time.perf_counter() - event_emit_started_at
                    )

                processed_seconds = chunk.offset_seconds + ownership_end
                if chunk.is_final:
                    audio_duration = max(audio_duration, processed_seconds)
                event_emit_started_at = time.perf_counter()
                put_thread_event(
                    loop,
                    queue,
                    {
                        "event": "progress",
                        "data": transcription_progress_payload(
                            audio_duration,
                            processed_seconds,
                            time.perf_counter() - started_at,
                        ),
                    },
                )
                event_emit_elapsed_seconds += (
                    time.perf_counter() - event_emit_started_at
                )

                chunk_elapsed_ms = round(
                    (time.perf_counter() - chunk_started_at) * 1000,
                    3,
                )
                total_elapsed = max(0.001, time.perf_counter() - started_at)
                progress = transcription_progress_payload(
                    audio_duration,
                    processed_seconds,
                    total_elapsed,
                )
                chunk_metrics = {
                    "chunk_index": chunk_count,
                    "offset_seconds": round(chunk.offset_seconds, 3),
                    "duration_seconds": round(chunk.duration_seconds, 3),
                    "processed_seconds": round(processed_seconds, 3),
                    "progress_percent": progress["progress_percent"],
                    "chunk_elapsed_ms": chunk_elapsed_ms,
                    "input_wait_ms": input_wait_ms,
                    "scheduler_wait_ms": scheduler_wait_ms,
                    "inference_ms": round(inference_elapsed_seconds * 1000, 3),
                    "event_emit_ms": round(event_emit_elapsed_seconds * 1000, 3),
                    "decoded_segments": decoded_segment_count - decoded_before_chunk,
                    "emitted_in_chunk": segment_count - emitted_before_chunk,
                    "segments_emitted": segment_count,
                    "processing_speed_x": round(processed_seconds / total_elapsed, 3),
                    "estimated_remaining_seconds": progress[
                        "estimated_remaining_seconds"
                    ],
                    "estimated_completion_at": progress[
                        "estimated_completion_at"
                    ],
                    "speech_intervals": len(speech_intervals),
                    "is_final": chunk.is_final,
                }
                if telemetry_callback is not None:
                    telemetry_callback(chunk_metrics)
                logger.info(
                    "Whisper chunk completed: job_id=%s chunk=%d "
                    "progress=%.2f%% processed=%.3f/%.3fs "
                    "chunk_elapsed=%.3fms inference=%.3fms "
                    "scheduler_wait=%.3fms input_wait=%.3fms "
                    "segments=%d/%d speech_intervals=%d speed=%.3fx "
                    "remaining=%s completion_at=%s language=%s",
                    job_id or "unknown",
                    chunk_count,
                    chunk_metrics["progress_percent"],
                    processed_seconds,
                    audio_duration,
                    chunk_elapsed_ms,
                    chunk_metrics["inference_ms"],
                    scheduler_wait_ms,
                    input_wait_ms,
                    chunk_metrics["emitted_in_chunk"],
                    chunk_metrics["decoded_segments"],
                    chunk_metrics["speech_intervals"],
                    chunk_metrics["processing_speed_x"],
                    (
                        f'{chunk_metrics["estimated_remaining_seconds"]:.1f}s'
                        if chunk_metrics["estimated_remaining_seconds"] is not None
                        else "unknown"
                    ),
                    chunk_metrics["estimated_completion_at"] or "unknown",
                    chunk_language or detected_language or "unknown",
                )
                log_structured_event(
                    "video_transcription_chunk_completed",
                    job_id=job_id or "unknown",
                    asr_provider=asr_provider,
                    **chunk_metrics,
                )
                if scheduler_turn_acquired:
                    fair_transcription_scheduler.release(
                        str(job_id),
                        completed=chunk.is_final,
                    )

        if chunk_count == 0:
            raise RuntimeError("媒體檔案沒有可轉譯的音軌")

        elapsed_seconds = time.perf_counter() - started_at
        real_time_factor = (
            elapsed_seconds / audio_duration
            if audio_duration > 0
            else None
        )
        speed_ratio = (
            audio_duration / elapsed_seconds
            if elapsed_seconds > 0 and audio_duration > 0
            else None
        )
        logger.info(
            "Whisper transcription completed: job_id=%s asr_provider=%s "
            "asr_model=%s "
            "transcription_mode=%s beam_size=%d "
            "audio_duration=%.3fs elapsed=%.3fs real_time_factor=%s "
            "speed=%s decoded_segments=%d emitted_segments=%d chunks=%d language=%s",
            job_id or "unknown",
            asr_provider,
            asr_model,
            mode,
            beam_size,
            audio_duration,
            elapsed_seconds,
            f"{real_time_factor:.4f}" if real_time_factor is not None else "unknown",
            f"{speed_ratio:.2f}x" if speed_ratio is not None else "unknown",
            decoded_segment_count,
            segment_count,
            chunk_count,
            detected_language or "unknown",
        )
        return {
            "language": detected_language,
            "language_probability": language_probability,
            "segments_count": segment_count,
            "content": "\n".join(content_parts),
            "audio_duration_seconds": round(audio_duration, 3),
            "transcription_elapsed_seconds": round(elapsed_seconds, 3),
            "processing_speed_x": round(speed_ratio, 3) if speed_ratio is not None else None,
            "transcription_mode": mode,
            "beam_size": beam_size,
            "asr_provider": asr_provider,
            "asr_model": asr_model,
            "source_transcription_provider": asr_provider,
            "source_transcription_model": asr_model,
            "timing_precision": (
                "word"
                if asr_provider in {"together", "elevenlabs"} and include_word_timestamps
                else "segment"
            ),
            "speaker_diarization": speaker_diarization,
            "transcription_delivery": (
                "http"
                if asr_provider in {"together", "elevenlabs"}
                else "realtime"
            ),
        }
    except Exception:
        logger.exception(
            "Whisper transcription failed: job_id=%s elapsed=%.3fs segments=%d",
            job_id or "unknown",
            time.perf_counter() - started_at,
            segment_count,
        )
        raise
    finally:
        if fair_scheduler_enabled:
            fair_transcription_scheduler.abort(str(job_id))
        if asr_provider == "local":
            auto2lrc.clear_model_cache()


def transcribe_elevenlabs_full_media(
    asr_client: ElevenLabsWhisperClient,
    *,
    source_url: str | None,
    media_path: Path | None,
    language_hint: str | None,
    loop: asyncio.AbstractEventLoop,
    queue: asyncio.Queue[dict[str, Any]],
    job_id: str,
    cancel_check: Callable[[], bool],
    include_word_timestamps: bool,
    keyterms: list[str] | None = None,
    audio_duration_hint: float | None = None,
) -> dict[str, Any]:
    """Run one complete ElevenLabs request and emit normalized subtitle events."""
    started_at = time.perf_counter()
    if cancel_check():
        raise TranscriptionCancelled("轉譯已取消")
    source_kind = "source_url" if source_url else "uploaded_file"
    logger.info(
        "ElevenLabs full transcription started: job_id=%s model=%s "
        "source=%s language=%s keyterms=%d duration_hint=%s",
        job_id,
        asr_client.model_name,
        source_kind,
        language_hint or "auto",
        len(keyterms or []),
        (
            f"{float(audio_duration_hint):.3f}s"
            if audio_duration_hint is not None
            else "unknown"
        ),
    )
    if source_url:
        segments, info = asr_client.transcribe_source_url(
            source_url,
            language=language_hint,
            keyterms=keyterms,
        )
        source = "elevenlabs_source_url"
    elif media_path is not None:
        segments, info = asr_client.transcribe_file(
            media_path,
            language=language_hint,
            keyterms=keyterms,
        )
        source = "elevenlabs_full_file"
    else:
        raise RuntimeError("ElevenLabs 整段模式缺少媒體來源")

    if cancel_check():
        raise TranscriptionCancelled("轉譯已取消")
    if not segments:
        raise RuntimeError("ElevenLabs 沒有回傳可用的轉譯內容")

    response_elapsed_seconds = time.perf_counter() - started_at
    logger.info(
        "ElevenLabs full transcription response received: job_id=%s "
        "model=%s source=%s elapsed=%.3fs returned_segments=%d "
        "language=%s",
        job_id,
        asr_client.model_name,
        source_kind,
        response_elapsed_seconds,
        len(segments),
        getattr(info, "language", None) or language_hint or "unknown",
    )

    detected_language = getattr(info, "language", None) or language_hint
    language_probability = getattr(info, "language_probability", None)
    duration = max(
        float(audio_duration_hint or 0.0),
        float(getattr(info, "duration", 0.0) or 0.0),
        max(float(getattr(segment, "end", 0.0) or 0.0) for segment in segments),
    )
    content_parts: list[str] = []
    last_start = 0.0
    emitted = 0
    for segment in segments:
        if cancel_check():
            raise TranscriptionCancelled("轉譯已取消")
        segment_language = (
            str(getattr(segment, "language", "") or "").strip()
            or detected_language
            or language_hint
        )
        text = to_traditional_chinese(
            str(getattr(segment, "text", "") or "").strip(),
            segment_language,
        )
        if not text:
            continue
        start = max(last_start, float(getattr(segment, "start", 0.0) or 0.0))
        end = max(start + 0.001, float(getattr(segment, "end", start) or start))
        words = (
            whisper_word_payloads(segment, segment_language)
            if include_word_timestamps
            else None
        )
        emitted += 1
        content_parts.append(text)
        payload = segment_payload(
            emitted,
            start,
            end,
            text,
            segment_language,
            whisper_low_confidence_spans(segment, text, segment_language),
            words,
            str(getattr(segment, "speaker_id", "") or "") or None,
        )
        payload.update(
            transcription_progress_payload(
                duration,
                end,
                time.perf_counter() - started_at,
            )
        )
        put_thread_event(loop, queue, {"event": "segment", "data": payload})
        last_start = start

    if emitted == 0:
        raise RuntimeError("ElevenLabs 沒有回傳可用的轉譯文字")
    elapsed_seconds = max(0.001, time.perf_counter() - started_at)
    speed_ratio = duration / elapsed_seconds if duration > 0 else None
    logger.info(
        "ElevenLabs full transcription completed: job_id=%s model=%s "
        "source=%s duration=%.3fs elapsed=%.3fs speed=%s "
        "returned_segments=%d emitted_segments=%d language=%s",
        job_id,
        asr_client.model_name,
        source_kind,
        duration,
        elapsed_seconds,
        f"{speed_ratio:.3f}x" if speed_ratio is not None else "unknown",
        len(segments),
        emitted,
        detected_language or "unknown",
    )
    put_thread_event(
        loop,
        queue,
        {
            "event": "progress",
            "data": transcription_progress_payload(
                duration,
                duration,
                elapsed_seconds,
            ),
        },
    )
    return {
        "source": source,
        "language": detected_language,
        "language_probability": language_probability,
        "segments_count": emitted,
        "content": "\n".join(content_parts),
        "audio_duration_seconds": round(duration, 3),
        "transcription_elapsed_seconds": round(elapsed_seconds, 3),
        "processing_speed_x": round(speed_ratio, 3) if speed_ratio is not None else None,
        "transcription_mode": "full",
        "beam_size": None,
        "asr_provider": "elevenlabs",
        "asr_model": asr_client.model_name,
        "source_transcription_provider": "elevenlabs",
        "source_transcription_model": asr_client.model_name,
        "timing_precision": "word" if include_word_timestamps else "segment",
        "speaker_diarization": False,
        "transcription_delivery": "full_http",
    }


def detect_audio_language(
    auto2lrc,
    audio_path: MediaSourceInput,
    job_id: Optional[str] = None,
    cancel_check: Optional[Callable[[], bool]] = None,
    audio_duration_hint: Optional[float] = None,
    asr_provider: str = "local",
    asr_client: WhisperAsrClient | None = None,
) -> dict[str, Any]:
    detection_provider = "cloudflare" if asr_provider == "together" else asr_provider
    detection_client = asr_client
    if detection_provider == "cloudflare" and asr_provider == "together":
        # Together's auto mode currently resolves multilingual audio to English.
        # Use Cloudflare only for the short language-detection pass; Together
        # remains responsible for the actual transcription after confirmation.
        detection_client = create_cloudflare_whisper_client(scheduler_managed=False)
    logger.info(
        "Whisper language detection started: job_id=%s provider=%s",
        job_id or "unknown",
        detection_provider,
    )
    started_at = time.perf_counter()
    try:
        if cancel_check is not None and cancel_check():
            raise TranscriptionCancelled("轉譯已取消")
        audio_duration = max(
            0.0,
            float(audio_duration_hint or probe_media_duration(audio_path) or 0.0),
        )
        try:
            prefix_pcm = decode_media_prefix(
                audio_path,
                YOUTUBE_WHISPER_LANGUAGE_DETECT_SECONDS,
                cancel_check,
            )
        except InterruptedError as exc:
            raise TranscriptionCancelled("轉譯已取消") from exc
        audio_samples = np.frombuffer(
            prefix_pcm,
            dtype="<i2",
        ).astype(np.float32)
        audio_samples /= 32768.0
        if detection_provider in REMOTE_ASR_PROVIDERS:
            active_asr_client = detection_client or create_remote_whisper_client(
                detection_provider
            )
            segments, info, _ = transcribe_remote_audio(
                detection_provider,
                active_asr_client,
                audio_samples,
                language=None,
                beam_size=5,
                vad_filter=YOUTUBE_WHISPER_VAD_FILTER,
                condition_on_previous_text=False,
                hallucination_silence_threshold=(
                    YOUTUBE_WHISPER_HALLUCINATION_SILENCE_SECONDS
                ),
                job_id=job_id,
                cancel_check=cancel_check,
            )
        else:
            scheduler_turn_acquired = False
            try:
                if job_id:
                    fair_transcription_scheduler.acquire(str(job_id), cancel_check)
                    scheduler_turn_acquired = True
                segments, info = auto2lrc.transcribe(
                    audio_samples,
                    beam_size=5,
                    language=None,
                )
            finally:
                if scheduler_turn_acquired:
                    fair_transcription_scheduler.release(
                        str(job_id),
                        completed=True,
                    )
                elif job_id:
                    fair_transcription_scheduler.abort(str(job_id))
        del segments
        if cancel_check is not None and cancel_check():
            raise TranscriptionCancelled("轉譯已取消")
        language = getattr(info, "language", None)
        probability = getattr(info, "language_probability", None)
        logger.info(
            "Whisper language detection completed: job_id=%s language=%s "
            "probability=%s elapsed=%.3fs",
            job_id or "unknown",
            language or "unknown",
            f"{probability:.4f}" if probability is not None else "unknown",
            time.perf_counter() - started_at,
        )
        return {
            "language": language,
            "language_probability": probability,
            "source": (
                "cloudflare_whisper"
                if detection_provider == "cloudflare"
                else asr_source_name(detection_provider)
            ),
            "duration": audio_duration,
            "detection_elapsed_seconds": round(time.perf_counter() - started_at, 3),
        }
    finally:
        if asr_provider == "local":
            logger.info(
                "Whisper language detection retained the warm local model: job_id=%s",
                job_id or "unknown",
            )


def create_youtube_live_router(auto2lrc, project_root: Path, verify_captcha_token=None) -> APIRouter:
    router = APIRouter()
    page_path = project_root / "pages" / "youtube_live.html"
    translate_page_path = project_root / "pages" / "youtube_live_translate.html"
    translate_script_path = project_root / "pages" / "youtube_live_translate.js"
    dashboard_page_path = project_root / "pages" / "dashboard.html"
    dashboard_script_path = project_root / "pages" / "dashboard.js"
    translation_usage_script_path = project_root / "pages" / "translation_usage.js"
    subtitle_display_cues_script_path = (
        project_root / "pages" / "subtitle_display_cues.js"
    )
    cookies_file = project_root / "cookies.txt"
    jobs: dict[str, dict[str, Any]] = {}
    upload_sessions: dict[str, dict[str, Any]] = {}
    telemetry_lock = threading.Lock()

    def cleanup_video_upload_sessions() -> None:
        now = time.time()
        expired_ids = [
            upload_id
            for upload_id, session in upload_sessions.items()
            if float(session.get("expires_at") or 0) <= now
        ]
        for upload_id in expired_ids:
            session = upload_sessions.pop(upload_id, None)
            if session:
                work_dir = str(session.get("work_dir") or "")
                if work_dir:
                    shutil.rmtree(work_dir, ignore_errors=True)

    def require_video_upload_session(
        upload_id: str,
        upload_token: str,
    ) -> dict[str, Any]:
        cleanup_video_upload_sessions()
        session = upload_sessions.get(upload_id)
        if session is None:
            raise HTTPException(status_code=404, detail="找不到上傳工作，請重新開始")
        if not upload_token or not secrets.compare_digest(
            str(session["upload_token"]),
            upload_token,
        ):
            raise HTTPException(status_code=403, detail="上傳憑證無效")
        session["expires_at"] = time.time() + VIDEO_UPLOAD_SESSION_TTL_SECONDS
        return session

    def video_upload_session_payload(session: dict[str, Any]) -> dict[str, Any]:
        completed_chunks, uploaded_bytes = completed_video_upload_chunks(
            Path(session["chunks_dir"]),
            int(session["size_bytes"]),
            int(session["chunk_bytes"]),
        )
        return {
            "upload_id": session["upload_id"],
            "upload_token": session["upload_token"],
            "filename": session["original_filename"],
            "size_bytes": session["size_bytes"],
            "chunk_bytes": session["chunk_bytes"],
            "chunk_count": session["chunk_count"],
            "completed_chunks": completed_chunks,
            "uploaded_bytes": uploaded_bytes,
            "expires_at": datetime.fromtimestamp(
                float(session["expires_at"]),
                timezone.utc,
            ).isoformat(),
        }

    def touch_job(job: dict[str, Any], *, phase: str | None = None) -> None:
        with telemetry_lock:
            telemetry = job.setdefault("telemetry", new_job_telemetry())
            telemetry["last_activity_at"] = time.time()
            if phase is not None:
                telemetry["phase"] = phase

    def record_transcription_chunk(job_id: str, metrics: dict[str, Any]) -> None:
        job = jobs.get(job_id)
        if job is None:
            return
        with telemetry_lock:
            telemetry = job.setdefault("telemetry", new_job_telemetry())
            transcription = telemetry["transcription"]
            chunk_count = int(metrics.get("chunk_index") or 0)
            previous_count = max(0, chunk_count - 1)
            previous_average = float(transcription.get("average_chunk_ms") or 0.0)
            chunk_ms = float(metrics.get("chunk_elapsed_ms") or 0.0)
            input_wait_ms = float(metrics.get("input_wait_ms") or 0.0)
            scheduler_wait_ms = float(metrics.get("scheduler_wait_ms") or 0.0)
            inference_ms = float(metrics.get("inference_ms") or 0.0)
            event_emit_ms = float(metrics.get("event_emit_ms") or 0.0)

            def rolling_average(field: str, current_value: float) -> float:
                previous_value = float(transcription.get(field) or 0.0)
                return round(
                    ((previous_value * previous_count) + current_value)
                    / max(1, chunk_count),
                    3,
                )

            transcription.update(
                {
                    "chunk_count": chunk_count,
                    "processed_seconds": round(float(metrics.get("processed_seconds") or 0.0), 3),
                    "progress_percent": round(float(metrics.get("progress_percent") or 0.0), 3),
                    "segments_emitted": int(metrics.get("segments_emitted") or 0),
                    "last_chunk_ms": round(chunk_ms, 3),
                    "average_chunk_ms": round(
                        ((previous_average * previous_count) + chunk_ms) / max(1, chunk_count),
                        3,
                    ),
                    "max_chunk_ms": round(
                        max(float(transcription.get("max_chunk_ms") or 0.0), chunk_ms),
                        3,
                    ),
                    "last_input_wait_ms": round(input_wait_ms, 3),
                    "average_input_wait_ms": rolling_average(
                        "average_input_wait_ms",
                        input_wait_ms,
                    ),
                    "max_input_wait_ms": round(
                        max(
                            float(transcription.get("max_input_wait_ms") or 0.0),
                            input_wait_ms,
                        ),
                        3,
                    ),
                    "last_scheduler_wait_ms": round(scheduler_wait_ms, 3),
                    "average_scheduler_wait_ms": rolling_average(
                        "average_scheduler_wait_ms",
                        scheduler_wait_ms,
                    ),
                    "max_scheduler_wait_ms": round(
                        max(
                            float(transcription.get("max_scheduler_wait_ms") or 0.0),
                            scheduler_wait_ms,
                        ),
                        3,
                    ),
                    "last_inference_ms": round(inference_ms, 3),
                    "average_inference_ms": rolling_average(
                        "average_inference_ms",
                        inference_ms,
                    ),
                    "max_inference_ms": round(
                        max(
                            float(transcription.get("max_inference_ms") or 0.0),
                            inference_ms,
                        ),
                        3,
                    ),
                    "last_event_emit_ms": round(event_emit_ms, 3),
                    "average_event_emit_ms": rolling_average(
                        "average_event_emit_ms",
                        event_emit_ms,
                    ),
                    "max_event_emit_ms": round(
                        max(
                            float(transcription.get("max_event_emit_ms") or 0.0),
                            event_emit_ms,
                        ),
                        3,
                    ),
                    "processing_speed_x": metrics.get("processing_speed_x"),
                    "last_chunk_at": time.time(),
                }
            )
            telemetry["phase"] = "transcription"
            telemetry["last_activity_at"] = time.time()

    def translation_request_started(
        job: dict[str, Any],
        operation: str,
        source_ids: set[int | str],
    ) -> None:
        with telemetry_lock:
            telemetry = job.setdefault("telemetry", new_job_telemetry())
            translation = telemetry["translation"]
            translation["active_requests"] += 1
            translation["requests_total"] += 1
            if operation == "group":
                translation["grouping_requests"] += 1
            elif operation == "translate-groups":
                translation["translation_requests"] += 1
            translation["source_ids_seen"].update(source_ids)
            translation["last_request_at"] = time.time()
            telemetry["last_activity_at"] = time.time()

    def translation_request_finished(
        job: dict[str, Any],
        *,
        status_code: int,
        latency_ms: float,
        source_ids: set[int | str],
        provider: str | None = None,
        usage: dict[str, Any] | None = None,
    ) -> None:
        with telemetry_lock:
            telemetry = job.setdefault("telemetry", new_job_telemetry())
            translation = telemetry["translation"]
            translation["active_requests"] = max(0, translation["active_requests"] - 1)
            if 200 <= status_code < 300:
                translation["successful_requests"] += 1
                translation["source_ids_succeeded"].update(source_ids)
            else:
                translation["failed_requests"] += 1
            latencies = translation["latencies_ms"]
            latencies.append(round(latency_ms, 3))
            del latencies[:-100]
            translation["last_latency_ms"] = round(latency_ms, 3)
            translation["last_status_code"] = status_code
            translation["last_provider"] = provider or translation.get("last_provider")
            translation["last_request_at"] = time.time()
            if usage:
                for key in ("prompt_tokens", "output_tokens"):
                    try:
                        translation[key] += int(usage.get(key) or 0)
                    except (TypeError, ValueError):
                        pass
                for key in ("estimated_cost_usd", "estimated_cost_twd"):
                    usage_key = key.replace("estimated_cost", "estimated_total_cost")
                    try:
                        translation[key] += float(usage.get(usage_key) or 0.0)
                    except (TypeError, ValueError):
                        pass
            telemetry["last_activity_at"] = time.time()

    def dashboard_job_payload(job_id: str, job: dict[str, Any]) -> dict[str, Any]:
        now = time.time()
        with telemetry_lock:
            telemetry = job.setdefault("telemetry", new_job_telemetry())
            transcription = dict(telemetry["transcription"])
            translation = dict(telemetry["translation"])
            latencies = list(translation.pop("latencies_ms", []))
            translation["source_ids_seen"] = len(translation["source_ids_seen"])
            translation["source_ids_succeeded"] = len(
                translation["source_ids_succeeded"]
            )
            translation["average_latency_ms"] = (
                round(sum(latencies) / len(latencies), 3) if latencies else None
            )
            translation["p95_latency_ms"] = dashboard_percentile(latencies, 0.95)
            last_activity_at = float(telemetry.get("last_activity_at") or job["created_at"])
            phase = str(telemetry.get("phase") or job.get("status") or "unknown")
        last_activity_age = max(0.0, now - last_activity_at)
        stalled_threshold = max(
            90.0,
            3.0 * float(transcription.get("last_chunk_ms") or 0.0) / 1000.0,
        )
        return {
            "job_id": job_id,
            "status": job.get("status"),
            "phase": phase,
            "source_kind": job.get("source_kind", "youtube"),
            "processing_profile": normalize_processing_profile(
                job.get("processing_profile")
            ),
            "asr_provider": asr_provider_for_profile(
                normalize_processing_profile(job.get("processing_profile")),
                job.get("asr_provider"),
            ),
            "source_language": job.get("language_hint") or job.get("detected_language") or "auto",
            "created_at": datetime.fromtimestamp(
                float(job["created_at"]), timezone.utc
            ).isoformat(),
            "age_seconds": round(max(0.0, now - float(job["created_at"])), 3),
            "last_activity_age_seconds": round(last_activity_age, 3),
            "stalled": job.get("status") == "running" and last_activity_age > stalled_threshold,
            "video_duration_seconds": job.get("video_duration_seconds"),
            "transcription": transcription,
            "translation": translation,
        }

    def job_is_cancelled(job: dict[str, Any]) -> bool:
        cancel_event = job.get("cancel_event")
        return job.get("status") == "cancelled" or (
            isinstance(cancel_event, threading.Event) and cancel_event.is_set()
        )

    def ensure_job_not_cancelled(job: dict[str, Any]) -> None:
        if job_is_cancelled(job):
            raise TranscriptionCancelled("轉譯已取消")

    async def close_event_stream(job: dict[str, Any]) -> None:
        if job.get("stream_closed"):
            return
        job["stream_closed"] = True
        await job["queue"].put({"event": "close", "data": {}})

    async def notify_job_cancelled(job: dict[str, Any]) -> None:
        if not job.get("cancel_notified"):
            job["cancel_notified"] = True
            await push_event(job, "cancelled", {"message": "轉譯已取消"})
        await close_event_stream(job)

    async def push_event(job: dict[str, Any], event: str, data: dict[str, Any]) -> None:
        await job["queue"].put({"event": event, "data": data})

    async def emit_subtitle_segments(job: dict[str, Any]) -> dict[str, Any]:
        segments = job.get("subtitle_segments") or []
        source_language = job.get("language_hint") or job.get("detected_language")
        content_parts: list[str] = []
        emitted_count = 0
        for segment in segments:
            ensure_job_not_cancelled(job)
            text = to_traditional_chinese(
                segment["text"].strip(),
                job.get("detected_language"),
            )
            if not text:
                continue
            emitted_count += 1
            content_parts.append(text)
            await push_event(
                job,
                "segment",
                segment_payload(
                    emitted_count,
                    segment["start"],
                    segment["end"],
                    text,
                    source_language,
                ),
            )
            await asyncio.sleep(0)
        return {
            "source": job.get("subtitle_source", "subtitle"),
            "language": source_language,
            "language_probability": None,
            "segments_count": emitted_count,
            "content": "\n".join(content_parts),
        }

    async def resolve_youtube_audio_source(
        job: dict[str, Any],
    ) -> tuple[MediaSourceInput, dict[str, Any]]:
        """Resolve a YouTube source, falling back to a local yt-dlp download.

        Some YouTube stream URLs work through yt-dlp but reject FFmpeg's separate
        HTTP request with a 403, even when a GVS PO token is present.  A local
        download uses yt-dlp's own downloader and is kept for the rest of this
        job so language confirmation does not trigger the same failing request.
        """
        cached_audio_path = Path(str(job.get("audio_path") or ""))
        if cached_audio_path.is_file():
            return cached_audio_path, dict(job.get("youtube_audio_info") or {})

        # Try the live path first.  With the PO provider enabled,
        # youtube_player_client_attempts() puts tv_simply first, so yt-dlp can
        # generate the matching GVS token before FFmpeg begins decoding.
        try:
            return await asyncio.to_thread(
                get_youtube_audio_stream_source,
                job["url"],
                cookies_file,
            )
        except HTTPException as stream_error:
            # Keep a local yt-dlp download only as a fallback for the small
            # number of videos where YouTube rejects FFmpeg's stream request.
            work_dir_value = job.get("work_dir")
            work_dir = (
                Path(str(work_dir_value))
                if work_dir_value
                else Path(tempfile.mkdtemp(prefix="youtube_audio_"))
            )
            work_dir.mkdir(parents=True, exist_ok=True)
            job["work_dir"] = str(work_dir)
            await push_event(
                job,
                "status",
                {
                    "message": "正在透過 YouTube 音訊下載器準備暫存音訊",
                },
            )
            logger.warning(
                "YouTube direct stream failed; falling back to yt-dlp download: %s",
                readable_exception_message(stream_error),
            )
            audio_path, video_info = await asyncio.to_thread(
                download_youtube_audio,
                job["url"],
                str(work_dir),
                cookies_file,
            )
            job["audio_path"] = str(audio_path)
            job["youtube_audio_info"] = video_info
            job["audio_transport"] = "download"
            return audio_path, video_info

    async def run_job(job_id: str) -> None:
        job = jobs.get(job_id)
        if not job:
            return
        if job_is_cancelled(job):
            cleanup_youtube_live_job_artifacts(job)
            return

        queue: asyncio.Queue[dict[str, Any]] = job["queue"]
        processing_profile = normalize_processing_profile(
            job.get("processing_profile")
        )
        asr_provider = asr_provider_for_profile(
            processing_profile,
            job.get("asr_provider"),
        )
        asr_client = (
            create_remote_whisper_client(asr_provider)
            if asr_provider in REMOTE_ASR_PROVIDERS
            else None
        )
        asr_source = asr_source_name(asr_provider)
        terminal = False
        phase = "queued"
        run_started = time.perf_counter()
        queue_wait_ms = round(
            max(0.0, run_started - float(job.get("queued_monotonic", run_started)))
            * 1000
        )
        log_structured_event(
            "video_job_started",
            job_id=job_id,
            job_status=job.get("status"),
            phase=(
                "transcription"
                if job.get("status") == "queued_for_transcription"
                else "preparation"
            ),
            queue_wait_ms=queue_wait_ms,
            source_kind=job.get("source_kind", "youtube"),
            source_language=job.get("language_hint") or "auto",
            ignore_subtitles=job.get("ignore_subtitles"),
            include_word_timestamps=job.get("include_word_timestamps"),
            processing_profile=processing_profile,
            asr_provider=asr_provider,
        )
        touch_job(
            job,
            phase=(
                "transcription"
                if job.get("status") == "queued_for_transcription"
                else "preparation"
            ),
        )
        try:
            ensure_job_not_cancelled(job)
            if job.get("source_kind") == "upload" and not job.get("metadata_emitted"):
                upload_path = Path(str(job.get("audio_path") or ""))
                if not upload_path.is_file():
                    raise RuntimeError("暫存影片已失效，請重新上傳")
                if not job.get("video_duration_seconds"):
                    job["video_duration_seconds"] = await asyncio.to_thread(
                        probe_media_duration,
                        upload_path,
                    )
                job["metadata_emitted"] = True
                await push_event(
                    job,
                    "metadata",
                    {
                        "title": job.get("filename") or "Uploaded video",
                        "duration": job.get("video_duration_seconds") or None,
                        "webpage_url": None,
                        "chapters": [],
                    },
                )
            if (
                asr_provider == "elevenlabs"
                and job.get("elevenlabs_mode") == "full"
            ):
                if not isinstance(asr_client, ElevenLabsWhisperClient):
                    raise RuntimeError("ElevenLabs 整段模式 client 尚未建立")
                phase = "full_transcription"
                job["status"] = "running"
                source_url: str | None = None
                media_path: Path | None = None
                if job.get("source_kind", "youtube") == "youtube":
                    source_url = str(job.get("url") or "").strip()
                    await push_event(
                        job,
                        "status",
                        {"message": "正在將 YouTube 網址交給 ElevenLabs 整段轉譯"},
                    )
                    try:
                        video_info = await asyncio.to_thread(
                            get_youtube_video_info,
                            source_url,
                            cookies_file,
                        )
                        job["video_duration_seconds"] = float(
                            video_info.get("duration") or 0.0
                        )
                        await push_event(
                            job,
                            "metadata",
                            {
                                "title": video_info.get("title"),
                                "duration": video_info.get("duration"),
                                "webpage_url": video_info.get("webpage_url"),
                                "chapters": chapter_payloads(video_info),
                            },
                        )
                    except Exception as exc:
                        logger.warning(
                            "Could not load metadata before ElevenLabs source_url request: %s",
                            readable_exception_message(exc),
                        )
                else:
                    media_path = Path(str(job.get("audio_path") or ""))
                    if not media_path.is_file():
                        raise RuntimeError("暫存影片已失效，請重新建立轉譯任務")
                    await push_event(
                        job,
                        "status",
                        {"message": "正在將完整影片送至 ElevenLabs 轉譯"},
                    )
                info = await asyncio.to_thread(
                    transcribe_elevenlabs_full_media,
                    asr_client,
                    source_url=source_url,
                    media_path=media_path,
                    language_hint=job.get("language_hint"),
                    loop=asyncio.get_running_loop(),
                    queue=queue,
                    job_id=job_id,
                    cancel_check=lambda: job_is_cancelled(job),
                    include_word_timestamps=bool(
                        job.get("include_word_timestamps")
                    ),
                    keyterms=job.get("elevenlabs_keyterms") or [],
                    audio_duration_hint=job.get("video_duration_seconds"),
                )
                ensure_job_not_cancelled(job)
                job["detected_language"] = info.get("language")
                job["language_probability"] = info.get("language_probability")
                job["video_duration_seconds"] = info.get(
                    "audio_duration_seconds"
                )
                job["status"] = "done"
                terminal = True
                await push_event(job, "done", info)
                log_structured_event(
                    "video_job_completed",
                    job_id=job_id,
                    job_status="done",
                    source_kind=job.get("source_kind", "youtube"),
                    subtitle_source=info.get("source"),
                    source_language=info.get("language"),
                    language_probability=info.get("language_probability"),
                    segments=info.get("segments_count"),
                    audio_duration_seconds=info.get(
                        "audio_duration_seconds"
                    ),
                    processing_speed_x=info.get("processing_speed_x"),
                    run_elapsed_ms=round(
                        (time.perf_counter() - run_started) * 1000
                    ),
                    total_elapsed_ms=round(
                        (
                            time.perf_counter()
                            - float(job.get("created_monotonic", run_started))
                        )
                        * 1000
                    ),
                )
                return
            if job.get("status") == "queued_for_transcription":
                phase = "transcription"
                job["status"] = "running"
                await push_event(job, "status", {"message": "語言已確認，開始逐段轉譯"})
                if job.get("prepared_source") == "subtitle":
                    info = await emit_subtitle_segments(job)
                else:
                    if job.get("source_kind", "youtube") == "youtube":
                        await push_event(
                            job,
                            "status",
                            {"message": "語言已確認，正在建立音訊串流"},
                        )
                        audio_source, _ = await resolve_youtube_audio_source(job)
                    else:
                        audio_source = Path(str(job.get("audio_path") or ""))
                        if not audio_source.is_file():
                            raise RuntimeError("暫存音訊已失效，請重新建立轉譯任務")
                    info = await asyncio.to_thread(
                        transcribe_audio_stream,
                        auto2lrc,
                        audio_source,
                        job.get("language_hint"),
                        asyncio.get_running_loop(),
                        queue,
                        job_id,
                        lambda: job_is_cancelled(job),
                        bool(job.get("include_word_timestamps")),
                        job.get("video_duration_seconds"),
                        job.get("transcription_mode", "accurate"),
                        asr_provider,
                        asr_client,
                        lambda metrics: record_transcription_chunk(job_id, metrics),
                        job.get("detected_language"),
                        bool(job.get("speaker_diarization")),
                        job.get("elevenlabs_keyterms") or [],
                    )
                ensure_job_not_cancelled(job)
                job["status"] = "done"
                terminal = True
                await push_event(job, "done", info)
                log_structured_event(
                    "video_job_completed",
                    job_id=job_id,
                    job_status="done",
                    source_kind=job.get("source_kind", "youtube"),
                    subtitle_source=info.get("source") or job.get("prepared_source"),
                    source_language=info.get("language") or job.get("language_hint"),
                    segments=info.get("segments_count"),
                    audio_duration_seconds=info.get("audio_duration_seconds")
                    or job.get("video_duration_seconds"),
                    processing_speed_x=info.get("processing_speed_x"),
                    run_elapsed_ms=round((time.perf_counter() - run_started) * 1000),
                    total_elapsed_ms=round(
                        (
                            time.perf_counter()
                            - float(job.get("created_monotonic", run_started))
                        )
                        * 1000
                    ),
                )
                return

            if job.get("status") != "queued":
                return

            job["status"] = "running"
            ensure_job_not_cancelled(job)
            if job.get("source_kind") == "upload":
                phase = "language_detection"
                audio_path = Path(str(job.get("audio_path") or ""))
                if not audio_path.is_file():
                    raise RuntimeError("暫存影片已失效，請重新上傳")
                await push_event(
                    job,
                    "status",
                    {"message": "影片上傳完成，正在偵測原文語言"},
                )
                detection = await asyncio.to_thread(
                    detect_audio_language,
                    auto2lrc,
                    audio_path,
                    job_id,
                    lambda: job_is_cancelled(job),
                    job.get("video_duration_seconds"),
                    asr_provider,
                    asr_client,
                )
                ensure_job_not_cancelled(job)
                job["detected_language"] = detection["language"]
                job["detection_source"] = detection.get("source") or asr_source
                job["language_probability"] = detection["language_probability"]
                job["video_duration_seconds"] = detection["duration"]
                job["status"] = "awaiting_language_confirmation"
                job["language_detected_monotonic"] = time.perf_counter()
                job["expires_at"] = time.time() + YOUTUBE_LIVE_JOB_TTL_SECONDS
                await push_event(
                    job,
                    "language_detected",
                    {
                        "language": detection["language"],
                        "language_probability": detection["language_probability"],
                        "source": detection.get("source") or asr_source,
                    },
                )
                log_structured_event(
                    "video_job_awaiting_language",
                    job_id=job_id,
                    job_status="awaiting_language_confirmation",
                    source_kind="upload",
                    detected_language=detection["language"],
                    language_probability=detection["language_probability"],
                    subtitle_source=detection.get("source") or asr_source,
                    detection_elapsed_ms=round(
                        detection["detection_elapsed_seconds"] * 1000
                    ),
                )
                return

            if job["ignore_subtitles"]:
                await push_event(job, "status", {"message": "已略過內建字幕，準備下載音訊"})
            else:
                await push_event(job, "status", {"message": "檢查 YouTube 字幕中"})

            phase = "metadata"
            metadata_started = time.perf_counter()
            video_info = await asyncio.to_thread(
                get_youtube_video_info,
                job["url"],
                cookies_file,
            )
            ensure_job_not_cancelled(job)
            chapters = chapter_payloads(video_info)
            job["video_duration_seconds"] = float(video_info.get("duration") or 0.0)
            log_structured_event(
                "video_metadata_loaded",
                job_id=job_id,
                duration_seconds=job["video_duration_seconds"],
                chapters=len(chapters),
                metadata_elapsed_ms=round(
                    (time.perf_counter() - metadata_started) * 1000
                ),
            )
            await push_event(
                job,
                "metadata",
                {
                    "title": video_info.get("title"),
                    "duration": video_info.get("duration"),
                    "webpage_url": video_info.get("webpage_url"),
                    "chapters": chapters,
                },
            )

            subtitle_track = None if job["ignore_subtitles"] else choose_subtitle_track(video_info)
            if subtitle_track is not None:
                phase = "subtitle_download"
                await push_event(
                    job,
                    "status",
                    {
                        "message": "找到內建字幕，正在解析",
                        "source": subtitle_track["source"],
                        "language": subtitle_track["language"],
                    },
                )
                subtitle_started = time.perf_counter()
                subtitle_content = await asyncio.to_thread(download_subtitle_content, subtitle_track["url"])
                ensure_job_not_cancelled(job)
                segments = parse_subtitle_content(subtitle_content, subtitle_track["extension"])
                job.update(
                    {
                        "prepared_source": "subtitle",
                        "subtitle_source": subtitle_track["source"],
                        "subtitle_segments": segments,
                        "detected_language": subtitle_track["language"],
                    }
                )
                log_structured_event(
                    "video_subtitles_prepared",
                    job_id=job_id,
                    subtitle_source=subtitle_track["source"],
                    detected_language=subtitle_track["language"],
                    segments=len(segments),
                    subtitle_elapsed_ms=round(
                        (time.perf_counter() - subtitle_started) * 1000
                    ),
                )
                if not job.get("language_hint"):
                    phase = "language_confirmation"
                    job["status"] = "awaiting_language_confirmation"
                    job["detection_source"] = "youtube_subtitles"
                    job["language_probability"] = None
                    job["language_detected_monotonic"] = time.perf_counter()
                    job["expires_at"] = time.time() + YOUTUBE_LIVE_JOB_TTL_SECONDS
                    await push_event(
                        job,
                        "language_detected",
                        {
                            "language": subtitle_track["language"],
                            "language_probability": None,
                            "source": "youtube_subtitles",
                        },
                    )

                    log_structured_event(
                        "video_job_awaiting_language",
                        job_id=job_id,
                        job_status="awaiting_language_confirmation",
                        detected_language=subtitle_track["language"],
                        subtitle_source=subtitle_track["source"],
                        segments=len(segments),
                    )
                    return

                info = await emit_subtitle_segments(job)
                job["status"] = "done"
                terminal = True
                await push_event(job, "done", info)
                log_structured_event(
                    "video_job_completed",
                    job_id=job_id,
                    job_status="done",
                    subtitle_source=info.get("source"),
                    source_language=info.get("language"),
                    segments=info.get("segments_count"),
                    audio_duration_seconds=job.get("video_duration_seconds"),
                    run_elapsed_ms=round((time.perf_counter() - run_started) * 1000),
                )
                return

            status_message = (
                "正在建立音訊串流"
                if job["ignore_subtitles"]
                else "沒有可用字幕，正在建立音訊串流"
            )
            await push_event(job, "status", {"message": status_message})
            phase = "audio_stream"
            audio_stream_started = time.perf_counter()
            audio_source, video_info = await resolve_youtube_audio_source(job)
            ensure_job_not_cancelled(job)
            job["prepared_source"] = asr_source
            log_structured_event(
                "video_audio_stream_resolved",
                job_id=job_id,
                stream_protocol=video_info.get("protocol"),
                audio_format=video_info.get("format_id"),
                audio_bitrate_kbps=video_info.get("abr"),
                audio_codec=video_info.get("acodec"),
                audio_filesize_bytes=(
                    video_info.get("filesize")
                    or video_info.get("filesize_approx")
                ),
                audio_stream_resolve_elapsed_ms=round(
                    (time.perf_counter() - audio_stream_started) * 1000
                ),
            )
            if not job.get("language_hint"):
                phase = "language_detection"
                await push_event(
                    job,
                    "status",
                    {"message": "音訊串流已建立，正在偵測原文語言"},
                )
                detection = await asyncio.to_thread(
                    detect_audio_language,
                    auto2lrc,
                    audio_source,
                    job_id,
                    lambda: job_is_cancelled(job),
                    job.get("video_duration_seconds"),
                    asr_provider,
                    asr_client,
                )
                ensure_job_not_cancelled(job)
                job["detected_language"] = detection["language"]
                job["detection_source"] = detection.get("source") or asr_source
                job["language_probability"] = detection["language_probability"]
                job["status"] = "awaiting_language_confirmation"
                job["language_detected_monotonic"] = time.perf_counter()
                job["expires_at"] = time.time() + YOUTUBE_LIVE_JOB_TTL_SECONDS
                await push_event(
                    job,
                    "language_detected",
                    {
                        "language": detection["language"],
                        "language_probability": detection["language_probability"],
                        "source": detection.get("source") or asr_source,
                    },
                )
                log_structured_event(
                    "video_job_awaiting_language",
                    job_id=job_id,
                    job_status="awaiting_language_confirmation",
                    detected_language=detection["language"],
                    language_probability=detection["language_probability"],
                    subtitle_source=detection.get("source") or asr_source,
                    detection_elapsed_ms=round(
                        detection["detection_elapsed_seconds"] * 1000
                    ),
                )
                return

            phase = "transcription"
            await push_event(job, "status", {"message": "音訊串流已建立，開始逐段轉譯"})
            info = await asyncio.to_thread(
                transcribe_audio_stream,
                auto2lrc,
                audio_source,
                job["language_hint"],
                asyncio.get_running_loop(),
                queue,
                job_id,
                lambda: job_is_cancelled(job),
                bool(job.get("include_word_timestamps")),
                job.get("video_duration_seconds"),
                job.get("transcription_mode", "accurate"),
                asr_provider,
                asr_client,
                lambda metrics: record_transcription_chunk(job_id, metrics),
                job.get("detected_language"),
                bool(job.get("speaker_diarization")),
                job.get("elevenlabs_keyterms") or [],
            )
            ensure_job_not_cancelled(job)

            job["status"] = "done"
            terminal = True
            await push_event(job, "done", {"source": asr_source, **info})
            log_structured_event(
                "video_job_completed",
                job_id=job_id,
                job_status="done",
                subtitle_source=asr_source,
                source_language=info.get("language"),
                language_probability=info.get("language_probability"),
                segments=info.get("segments_count"),
                audio_duration_seconds=info.get("audio_duration_seconds"),
                processing_speed_x=info.get("processing_speed_x"),
                run_elapsed_ms=round((time.perf_counter() - run_started) * 1000),
            )
        except TranscriptionCancelled:
            job["status"] = "cancelled"
            terminal = True
            log_structured_event(
                "video_job_cancelled",
                job_id=job_id,
                job_status="cancelled",
                source_kind=job.get("source_kind", "youtube"),
                phase=phase,
                run_elapsed_ms=round((time.perf_counter() - run_started) * 1000),
            )
            if not job.get("cancel_notified"):
                job["cancel_notified"] = True
                await push_event(job, "cancelled", {"message": "轉譯已取消"})
        except Exception as exc:
            job["status"] = "failed"
            terminal = True
            log_structured_event(
                "video_job_failed",
                job_id=job_id,
                job_status="failed",
                source_kind=job.get("source_kind", "youtube"),
                phase=phase,
                exception_type=type(exc).__name__,
                run_elapsed_ms=round((time.perf_counter() - run_started) * 1000),
            )
            logger.exception(
                "Video transcription job failed: job_id=%s phase=%s",
                job_id,
                phase,
            )
            await push_event(job, "failed", {"message": readable_exception_message(exc)})
        finally:
            job["expires_at"] = time.time() + YOUTUBE_LIVE_JOB_TTL_SECONDS
            touch_job(job, phase=str(job.get("status") or phase))
            if terminal:
                cleanup_youtube_live_job_artifacts(job)
                await close_event_stream(job)

    async def handle_youtube_live_task(task: dict[str, Any]) -> None:
        await run_job(str(task.get("id", "")))

    register_transcribe_handler("youtube_live", handle_youtube_live_task)
    register_transcribe_cleanup(lambda: cleanup_youtube_live_jobs(jobs))
    register_transcribe_cleanup(cleanup_video_upload_sessions)

    @router.get("/youtube-live", include_in_schema=False)
    def youtube_live_page():
        return HTMLResponse(page_path.read_text(encoding="utf-8"))

    @router.get("/youtube-live-translate", include_in_schema=False)
    def youtube_live_translate_page():
        return HTMLResponse(
            translate_page_path.read_text(encoding="utf-8"),
            headers={"Cache-Control": "no-store, max-age=0"},
        )

    @router.get("/video-upload-translate", include_in_schema=False)
    def video_upload_translate_page():
        return HTMLResponse(
            translate_page_path.read_text(encoding="utf-8"),
            headers={"Cache-Control": "no-store, max-age=0"},
        )

    @router.get("/youtube-live-translate.js", include_in_schema=False)
    def youtube_live_translate_script():
        return Response(
            translate_script_path.read_text(encoding="utf-8"),
            media_type="application/javascript",
            headers={"Cache-Control": "no-store, max-age=0"},
        )

    @router.get("/subtitle-display-cues.js", include_in_schema=False)
    def subtitle_display_cues_script():
        return Response(
            subtitle_display_cues_script_path.read_text(encoding="utf-8"),
            media_type="application/javascript",
            headers={"Cache-Control": "no-store, max-age=0"},
        )

    @router.get("/translation-usage.js", include_in_schema=False)
    def translation_usage_script():
        return Response(
            translation_usage_script_path.read_text(encoding="utf-8"),
            media_type="application/javascript",
            headers={"Cache-Control": "no-store, max-age=0"},
        )

    @router.get("/dashboard", include_in_schema=False)
    def dashboard_page():
        return HTMLResponse(
            dashboard_page_path.read_text(encoding="utf-8"),
            headers={"Cache-Control": "no-store, max-age=0"},
        )

    @router.get("/dashboard.js", include_in_schema=False)
    def dashboard_script():
        return Response(
            dashboard_script_path.read_text(encoding="utf-8"),
            media_type="application/javascript",
            headers={"Cache-Control": "no-store, max-age=0"},
        )

    def require_dashboard_access(request: Request) -> None:
        authorization = request.headers.get("authorization", "")
        client_host = request.client.host if request.client else ""
        if dashboard_request_authorized(
            AUDIOIO_DASHBOARD_TOKEN,
            authorization,
            client_host,
        ):
            return
        if AUDIOIO_DASHBOARD_TOKEN:
            raise HTTPException(status_code=401, detail="Dashboard Token 無效")
        raise HTTPException(
            status_code=503,
            detail="雲端 Dashboard 尚未設定 AUDIOIO_DASHBOARD_TOKEN",
        )

    @router.get(
        "/api/dashboard/status",
        tags=["System"],
        summary="取得即時 GPU、轉錄及翻譯工作遙測",
    )
    def dashboard_status(request: Request):
        require_dashboard_access(request)
        cleanup_youtube_live_jobs(jobs)
        queue_counts = get_transcribe_queue_counts()
        sorted_jobs = sorted(
            jobs.items(),
            key=dashboard_job_sort_key,
            reverse=True,
        )[:AUDIOIO_DASHBOARD_JOB_LIMIT]
        job_payloads = [
            dashboard_job_payload(job_id, job) for job_id, job in sorted_jobs
        ]
        try:
            gpu_payload = get_pynvml_gpu_info()
            gpus = [
                {
                    "index": index,
                    "name": gpu.get("name"),
                    "utilization_gpu": gpu.get(
                        "scheduling_utilization",
                        gpu.get("utilization_gpu"),
                    ),
                    "temperature_gpu": gpu.get("temperature_gpu"),
                    "memory_used": gpu.get("memory_used"),
                    "memory_total": gpu.get("memory_total"),
                    "memory_free": gpu.get("memory_free"),
                }
                for index, gpu in enumerate(gpu_payload.get("gpus", []))
            ]
        except Exception as exc:
            logger.warning("Dashboard could not read NVIDIA metrics: %s", exc)
            gpus = []
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "queue": queue_counts,
            "summary": {
                "jobs": len(job_payloads),
                "active_jobs": sum(
                    job.get("status") in {"queued", "queued_for_transcription", "running"}
                    for job in job_payloads
                ),
                "stalled_jobs": sum(bool(job.get("stalled")) for job in job_payloads),
                "translation_errors": sum(
                    int(job["translation"].get("failed_requests") or 0)
                    for job in job_payloads
                ),
            },
            "gpus": gpus,
            "jobs": job_payloads,
        }

    @router.get(
        "/api/youtube-live/playlists/preview",
        tags=["YouTube Live"],
        summary="讀取 YouTube 播放清單預覽",
    )
    async def youtube_playlist_preview(url: str):
        try:
            playlist = await asyncio.to_thread(
                get_youtube_playlist_preview,
                url,
                cookies_file,
            )
        except HTTPException:
            raise
        except Exception as exc:
            logger.warning("Could not load YouTube playlist preview: %s", exc)
            if is_youtube_rate_limit_error(exc):
                raise HTTPException(status_code=429, detail=YOUTUBE_RATE_LIMIT_MESSAGE) from exc
            raise HTTPException(status_code=502, detail="無法讀取 YouTube 播放清單") from exc
        return JSONResponse(
            playlist,
            headers={"Cache-Control": "private, max-age=300"},
        )

    @router.post("/api/youtube-live/jobs", tags=["YouTube Live"])
    async def create_youtube_live_job(
        payload: YoutubeLiveRequest,
        http_request: Request,
    ):
        cleanup_youtube_live_jobs(jobs)
        url = payload.url.strip()
        if not url:
            raise HTTPException(status_code=400, detail="請輸入 YouTube 網址")
        if verify_captcha_token is not None:
            verify_captcha_token(payload.captcha_token)

        job_id = secrets.token_urlsafe(18)
        created_monotonic = time.perf_counter()
        transcription_mode = normalize_transcription_mode(payload.transcription_mode)
        try:
            processing_profile = normalize_processing_profile(payload.processing_profile)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        try:
            asr_provider = asr_provider_for_profile(
                processing_profile,
                payload.asr_provider,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        try:
            elevenlabs_mode = normalize_elevenlabs_transcription_mode(
                payload.elevenlabs_mode
            )
            elevenlabs_keyterms = normalize_elevenlabs_keyterms(
                payload.elevenlabs_keyterms
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        if asr_provider != "elevenlabs":
            elevenlabs_mode = "chunks"
            elevenlabs_keyterms = []
        validate_speaker_diarization(
            processing_profile,
            payload.speaker_diarization,
        )
        jobs[job_id] = {
            "url": url,
            "language_hint": payload.language.strip() or None,
            "language_mode": "forced" if payload.language.strip() else "auto",
            "ignore_subtitles": (
                payload.ignore_subtitles
                or payload.speaker_diarization
                or elevenlabs_mode == "full"
            ),
            "include_word_timestamps": payload.include_word_timestamps,
            "speaker_diarization": payload.speaker_diarization,
            "transcription_mode": transcription_mode,
            "processing_profile": processing_profile,
            "asr_provider": asr_provider,
            "elevenlabs_mode": elevenlabs_mode,
            "elevenlabs_keyterms": elevenlabs_keyterms,
            "translation_token": secrets.token_urlsafe(32),
            "translation_tasks": set(),
            "cancel_token": secrets.token_urlsafe(32),
            "cancel_event": threading.Event(),
            "status": "queued",
            "queue": asyncio.Queue(),
            "created_at": time.time(),
            "created_monotonic": created_monotonic,
            "queued_monotonic": created_monotonic,
            "expires_at": time.time() + YOUTUBE_LIVE_JOB_TTL_SECONDS,
            "telemetry": new_job_telemetry(),
        }
        set_request_log_metadata(
            http_request,
            request_id=job_id,
            job_id=job_id,
            operation="video_job_create",
            job_status="queued",
            source_language=jobs[job_id]["language_hint"] or "auto",
            ignore_subtitles=payload.ignore_subtitles,
            include_word_timestamps=payload.include_word_timestamps,
            speaker_diarization=payload.speaker_diarization,
            transcription_mode=transcription_mode,
            processing_profile=processing_profile,
            asr_provider=asr_provider,
            elevenlabs_mode=elevenlabs_mode,
            elevenlabs_keyterms_count=len(elevenlabs_keyterms),
        )
        try:
            enqueue_transcribe_task(
                video_transcribe_task(job_id, processing_profile, asr_provider)
            )
        except asyncio.QueueFull:
            jobs[job_id]["status"] = "failed"
            jobs[job_id]["expires_at"] = time.time() + YOUTUBE_LIVE_JOB_TTL_SECONDS
            log_structured_event(
                "video_job_rejected",
                job_id=job_id,
                reason="queue_full",
            )
            raise HTTPException(status_code=503, detail="轉譯佇列已滿，請稍後再試")
        except RuntimeError as exc:
            jobs[job_id]["status"] = "failed"
            jobs[job_id]["expires_at"] = time.time() + YOUTUBE_LIVE_JOB_TTL_SECONDS
            log_structured_event(
                "video_job_rejected",
                job_id=job_id,
                reason="queue_not_started",
            )
            raise HTTPException(status_code=503, detail="轉譯佇列尚未啟動，請稍後再試") from exc

        queue_counts = get_transcribe_queue_counts()
        log_structured_event(
            "video_job_created",
            job_id=job_id,
            job_status="queued",
            source_language=jobs[job_id]["language_hint"] or "auto",
            ignore_subtitles=payload.ignore_subtitles,
            include_word_timestamps=payload.include_word_timestamps,
            speaker_diarization=payload.speaker_diarization,
            transcription_mode=transcription_mode,
            processing_profile=processing_profile,
            asr_provider=asr_provider,
            elevenlabs_mode=elevenlabs_mode,
            elevenlabs_keyterms_count=len(elevenlabs_keyterms),
            waiting_count=queue_counts["waiting_count"],
            transcribing_count=queue_counts["transcribing_count"],
        )

        return {
            "job_id": job_id,
            "events_url": f"/api/youtube-live/jobs/{job_id}/events",
            "translation_token": jobs[job_id]["translation_token"],
            "cancel_url": f"/api/youtube-live/jobs/{job_id}/cancel",
            "cancel_token": jobs[job_id]["cancel_token"],
            "status": "queued",
            "processing_profile": processing_profile,
            "asr_provider": asr_provider,
            "elevenlabs_mode": elevenlabs_mode,
            "elevenlabs_keyterms_count": len(elevenlabs_keyterms),
        }

    def validated_video_upload_metadata(
        filename: str,
        content_type: str,
        size_bytes: int | None = None,
    ) -> tuple[str, str]:
        original_filename = Path(filename or "video").name
        suffix = Path(original_filename).suffix.lower()
        normalized_content_type = str(content_type or "").lower()
        if (
            not normalized_content_type.startswith("video/")
            and suffix not in VIDEO_UPLOAD_EXTENSIONS
        ):
            raise HTTPException(status_code=400, detail="請選擇支援的影片檔案")
        if size_bytes is not None:
            if size_bytes <= 0:
                raise HTTPException(status_code=400, detail="上傳影片不可為空")
            if size_bytes > VIDEO_UPLOAD_MAX_BYTES:
                raise HTTPException(
                    status_code=413,
                    detail="上傳影片超過伺服器允許的大小",
                )
        return original_filename, suffix

    async def register_video_upload_job(
        http_request: Request,
        *,
        upload_path: Path,
        work_dir: Path,
        original_filename: str,
        upload_bytes: int,
        language: str,
        include_word_timestamps: bool,
        speaker_diarization: bool,
        transcription_mode: str,
        processing_profile: str,
        asr_provider: str,
        elevenlabs_mode: str,
        elevenlabs_keyterms: Any = None,
    ) -> dict[str, Any]:
        transcription_mode = normalize_transcription_mode(transcription_mode)
        try:
            normalized_profile = normalize_processing_profile(processing_profile)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        try:
            normalized_asr_provider = asr_provider_for_profile(
                normalized_profile,
                asr_provider,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        try:
            normalized_elevenlabs_mode = normalize_elevenlabs_transcription_mode(
                elevenlabs_mode
            )
            normalized_elevenlabs_keyterms = normalize_elevenlabs_keyterms(
                elevenlabs_keyterms
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        if normalized_asr_provider != "elevenlabs":
            normalized_elevenlabs_mode = "chunks"
            normalized_elevenlabs_keyterms = []
        validate_speaker_diarization(normalized_profile, speaker_diarization)
        job_id = secrets.token_urlsafe(18)
        language_hint = language.strip() or None
        created_monotonic = time.perf_counter()
        jobs[job_id] = {
            "source_kind": "upload",
            "filename": original_filename,
            "language_hint": language_hint,
            "language_mode": "forced" if language_hint else "auto",
            "ignore_subtitles": True,
            "include_word_timestamps": include_word_timestamps,
            "speaker_diarization": speaker_diarization,
            "transcription_mode": transcription_mode,
            "processing_profile": normalized_profile,
            "asr_provider": normalized_asr_provider,
            "elevenlabs_mode": normalized_elevenlabs_mode,
            "elevenlabs_keyterms": normalized_elevenlabs_keyterms,
            "prepared_source": asr_source_name(
                normalized_asr_provider
            ),
            "audio_path": str(upload_path),
            "work_dir": str(work_dir),
            "translation_token": secrets.token_urlsafe(32),
            "translation_tasks": set(),
            "cancel_token": secrets.token_urlsafe(32),
            "cancel_event": threading.Event(),
            "status": "queued_for_transcription" if language_hint else "queued",
            "queue": asyncio.Queue(),
            "created_at": time.time(),
            "created_monotonic": created_monotonic,
            "queued_monotonic": created_monotonic,
            "expires_at": time.time() + YOUTUBE_LIVE_JOB_TTL_SECONDS,
            "telemetry": new_job_telemetry(),
        }
        set_request_log_metadata(
            http_request,
            request_id=job_id,
            job_id=job_id,
            operation="video_upload_job_create",
            job_status=jobs[job_id]["status"],
            source_kind="upload",
            source_language=language_hint or "auto",
            include_word_timestamps=include_word_timestamps,
            speaker_diarization=speaker_diarization,
            transcription_mode=transcription_mode,
            processing_profile=normalized_profile,
            asr_provider=normalized_asr_provider,
            elevenlabs_mode=normalized_elevenlabs_mode,
            elevenlabs_keyterms_count=len(normalized_elevenlabs_keyterms),
        )
        try:
            enqueue_transcribe_task(
                video_transcribe_task(
                    job_id,
                    normalized_profile,
                    normalized_asr_provider,
                )
            )
        except asyncio.QueueFull:
            cleanup_youtube_live_job_artifacts(jobs.pop(job_id))
            raise HTTPException(status_code=503, detail="轉譯佇列已滿，請稍後再試")
        except RuntimeError as exc:
            cleanup_youtube_live_job_artifacts(jobs.pop(job_id))
            raise HTTPException(status_code=503, detail="轉譯佇列尚未啟動，請稍後再試") from exc

        queue_counts = get_transcribe_queue_counts()
        log_structured_event(
            "video_job_created",
            job_id=job_id,
            job_status=jobs[job_id]["status"],
            source_kind="upload",
            source_language=language_hint or "auto",
            include_word_timestamps=include_word_timestamps,
            speaker_diarization=speaker_diarization,
            transcription_mode=transcription_mode,
            processing_profile=normalized_profile,
            asr_provider=normalized_asr_provider,
            elevenlabs_mode=normalized_elevenlabs_mode,
            elevenlabs_keyterms_count=len(normalized_elevenlabs_keyterms),
            input_bytes=upload_bytes,
            waiting_count=queue_counts["waiting_count"],
            transcribing_count=queue_counts["transcribing_count"],
        )
        return {
            "job_id": job_id,
            "filename": original_filename,
            "events_url": f"/api/youtube-live/jobs/{job_id}/events",
            "translation_token": jobs[job_id]["translation_token"],
            "cancel_url": f"/api/youtube-live/jobs/{job_id}/cancel",
            "cancel_token": jobs[job_id]["cancel_token"],
            "status": jobs[job_id]["status"],
            "processing_profile": normalized_profile,
            "asr_provider": normalized_asr_provider,
            "elevenlabs_mode": normalized_elevenlabs_mode,
        }

    async def create_video_upload_job_entry(
        http_request: Request,
        file: UploadFile,
        language: str,
        include_word_timestamps: bool,
        speaker_diarization: bool,
        transcription_mode: str,
        processing_profile: str,
        asr_provider: str,
        elevenlabs_mode: str,
        elevenlabs_keyterms: Any = None,
    ) -> dict[str, Any]:
        original_filename, suffix = validated_video_upload_metadata(
            file.filename or "video",
            str(file.content_type or ""),
        )
        work_dir = Path(tempfile.mkdtemp(prefix="video_upload_"))
        upload_path = work_dir / f"input{suffix or '.video'}"
        upload_bytes = 0
        try:
            with upload_path.open("wb") as output:
                while chunk := await file.read(1024 * 1024):
                    upload_bytes += len(chunk)
                    if upload_bytes > VIDEO_UPLOAD_MAX_BYTES:
                        raise HTTPException(
                            status_code=413,
                            detail="上傳影片超過伺服器允許的大小",
                        )
                    output.write(chunk)
        except asyncio.CancelledError:
            shutil.rmtree(work_dir, ignore_errors=True)
            raise
        except HTTPException:
            shutil.rmtree(work_dir, ignore_errors=True)
            raise
        except Exception as exc:
            shutil.rmtree(work_dir, ignore_errors=True)
            raise HTTPException(
                status_code=500,
                detail=f"儲存上傳影片失敗: {exc}",
            ) from exc
        finally:
            await file.close()

        try:
            validated_video_upload_metadata(
                original_filename,
                str(file.content_type or ""),
                upload_bytes,
            )
            return await register_video_upload_job(
                http_request,
                upload_path=upload_path,
                work_dir=work_dir,
                original_filename=original_filename,
                upload_bytes=upload_bytes,
                language=language,
                include_word_timestamps=include_word_timestamps,
                speaker_diarization=speaker_diarization,
                transcription_mode=transcription_mode,
                processing_profile=processing_profile,
                asr_provider=asr_provider,
                elevenlabs_mode=elevenlabs_mode,
                elevenlabs_keyterms=elevenlabs_keyterms,
            )
        except Exception:
            if upload_path.exists() and not any(
                str(job.get("audio_path") or "") == str(upload_path)
                for job in jobs.values()
            ):
                shutil.rmtree(work_dir, ignore_errors=True)
            raise

    @router.post(
        "/api/video-upload/sessions/batch",
        tags=["Video Upload"],
        summary="建立可續傳的分塊影片上傳工作",
    )
    async def create_video_upload_sessions(
        http_request: Request,
        payload: VideoUploadBatchSessionRequest,
    ):
        cleanup_youtube_live_jobs(jobs)
        cleanup_video_upload_sessions()
        if not payload.files:
            raise HTTPException(status_code=400, detail="請至少選擇一個影片檔案")
        if len(payload.files) > VIDEO_UPLOAD_BATCH_MAX_FILES:
            raise HTTPException(
                status_code=400,
                detail=f"一次最多上傳 {VIDEO_UPLOAD_BATCH_MAX_FILES} 個影片",
            )
        validated_files: list[tuple[VideoUploadFileSessionRequest, str, str]] = []
        for file_request in payload.files:
            original_filename, suffix = validated_video_upload_metadata(
                file_request.filename,
                file_request.content_type,
                int(file_request.size_bytes),
            )
            validated_files.append((file_request, original_filename, suffix))
        if verify_captcha_token is not None:
            verify_captcha_token(payload.captcha_token)

        created_upload_ids: list[str] = []
        sessions: list[dict[str, Any]] = []
        try:
            for batch_index, (file_request, original_filename, suffix) in enumerate(
                validated_files
            ):
                upload_id = secrets.token_urlsafe(18)
                work_dir = Path(tempfile.mkdtemp(prefix="video_upload_session_"))
                chunks_dir = work_dir / "chunks"
                chunks_dir.mkdir()
                session = {
                    "upload_id": upload_id,
                    "upload_token": secrets.token_urlsafe(32),
                    "original_filename": original_filename,
                    "suffix": suffix or ".video",
                    "content_type": file_request.content_type,
                    "size_bytes": int(file_request.size_bytes),
                    "last_modified": file_request.last_modified,
                    "chunk_bytes": VIDEO_UPLOAD_CHUNK_BYTES,
                    "chunk_count": video_upload_chunk_count(
                        int(file_request.size_bytes),
                        VIDEO_UPLOAD_CHUNK_BYTES,
                    ),
                    "work_dir": str(work_dir),
                    "chunks_dir": str(chunks_dir),
                    "lock": asyncio.Lock(),
                    "created_at": time.time(),
                    "expires_at": time.time() + VIDEO_UPLOAD_SESSION_TTL_SECONDS,
                }
                upload_sessions[upload_id] = session
                created_upload_ids.append(upload_id)
                session_payload = video_upload_session_payload(session)
                session_payload["batch_index"] = batch_index
                sessions.append(session_payload)
        except Exception:
            for upload_id in created_upload_ids:
                session = upload_sessions.pop(upload_id, None)
                if session:
                    shutil.rmtree(session["work_dir"], ignore_errors=True)
            raise

        set_request_log_metadata(
            http_request,
            operation="video_upload_sessions_create",
            source_kind="upload",
            upload_session_count=len(sessions),
            input_bytes=sum(int(item.size_bytes) for item in payload.files),
        )
        return {
            "chunk_bytes": VIDEO_UPLOAD_CHUNK_BYTES,
            "sessions": sessions,
        }

    @router.get(
        "/api/video-upload/sessions/{upload_id}",
        tags=["Video Upload"],
        summary="取得分塊影片上傳進度",
    )
    async def get_video_upload_session(
        upload_id: str,
        upload_token: str = Header("", alias="X-Upload-Token"),
    ):
        session = require_video_upload_session(upload_id, upload_token)
        return video_upload_session_payload(session)

    @router.put(
        "/api/video-upload/sessions/{upload_id}/chunks/{chunk_index}",
        tags=["Video Upload"],
        summary="上傳一個影片分塊",
    )
    async def upload_video_chunk(
        http_request: Request,
        upload_id: str,
        chunk_index: int,
        upload_token: str = Header("", alias="X-Upload-Token"),
    ):
        session = require_video_upload_session(upload_id, upload_token)
        try:
            expected_bytes = expected_video_upload_chunk_bytes(
                int(session["size_bytes"]),
                int(session["chunk_bytes"]),
                chunk_index,
            )
        except ValueError as exc:
            raise HTTPException(status_code=404, detail="上傳分塊不存在") from exc
        content_length = http_request.headers.get("content-length")
        if content_length:
            try:
                if int(content_length) != expected_bytes:
                    raise HTTPException(status_code=400, detail="上傳分塊大小不正確")
            except ValueError as exc:
                raise HTTPException(status_code=400, detail="Content-Length 無效") from exc

        chunks_dir = Path(session["chunks_dir"])
        part_path = chunks_dir / f"{chunk_index:08d}.part"
        async with session["lock"]:
            if part_path.is_file() and part_path.stat().st_size == expected_bytes:
                return video_upload_session_payload(session)
            temporary_path = chunks_dir / (
                f".{chunk_index:08d}.{secrets.token_hex(6)}.uploading"
            )
            written_bytes = 0
            try:
                with temporary_path.open("wb") as output:
                    async for block in http_request.stream():
                        written_bytes += len(block)
                        if written_bytes > expected_bytes:
                            raise HTTPException(
                                status_code=413,
                                detail="上傳分塊超過允許大小",
                            )
                        output.write(block)
                if written_bytes != expected_bytes:
                    raise HTTPException(status_code=400, detail="上傳分塊不完整")
                temporary_path.replace(part_path)
            finally:
                temporary_path.unlink(missing_ok=True)

        session["expires_at"] = time.time() + VIDEO_UPLOAD_SESSION_TTL_SECONDS
        set_request_log_metadata(
            http_request,
            operation="video_upload_chunk",
            source_kind="upload",
            upload_id=upload_id,
            chunk_index=chunk_index,
            chunk_bytes=written_bytes,
        )
        return video_upload_session_payload(session)

    @router.post(
        "/api/video-upload/sessions/{upload_id}/complete",
        tags=["Video Upload"],
        summary="合併影片分塊並建立轉譯工作",
    )
    async def complete_video_upload_session(
        http_request: Request,
        upload_id: str,
        payload: VideoUploadCompleteRequest,
        upload_token: str = Header("", alias="X-Upload-Token"),
    ):
        session = require_video_upload_session(upload_id, upload_token)
        async with session["lock"]:
            completed_chunks, uploaded_bytes = completed_video_upload_chunks(
                Path(session["chunks_dir"]),
                int(session["size_bytes"]),
                int(session["chunk_bytes"]),
            )
            if len(completed_chunks) != int(session["chunk_count"]):
                raise HTTPException(
                    status_code=409,
                    detail={
                        "message": "影片尚未上傳完成",
                        "completed_chunks": completed_chunks,
                        "uploaded_bytes": uploaded_bytes,
                    },
                )
            work_dir = Path(session["work_dir"])
            upload_path = work_dir / f"input{session['suffix']}"
            try:
                await asyncio.to_thread(
                    assemble_video_upload_chunks,
                    Path(session["chunks_dir"]),
                    upload_path,
                    int(session["size_bytes"]),
                    int(session["chunk_bytes"]),
                )
                job = await register_video_upload_job(
                    http_request,
                    upload_path=upload_path,
                    work_dir=work_dir,
                    original_filename=str(session["original_filename"]),
                    upload_bytes=int(session["size_bytes"]),
                    language=payload.language,
                    include_word_timestamps=payload.include_word_timestamps,
                    speaker_diarization=payload.speaker_diarization,
                    transcription_mode=payload.transcription_mode,
                    processing_profile=payload.processing_profile,
                    asr_provider=payload.asr_provider,
                    elevenlabs_mode=payload.elevenlabs_mode,
                    elevenlabs_keyterms=payload.elevenlabs_keyterms,
                )
            except Exception:
                upload_sessions.pop(upload_id, None)
                shutil.rmtree(work_dir, ignore_errors=True)
                raise

            upload_sessions.pop(upload_id, None)
            shutil.rmtree(Path(session["chunks_dir"]), ignore_errors=True)
            job["upload_id"] = upload_id
            return job

    @router.delete(
        "/api/video-upload/sessions/{upload_id}",
        tags=["Video Upload"],
        summary="取消尚未完成的影片上傳",
    )
    async def cancel_video_upload_session(
        upload_id: str,
        upload_token: str = Header("", alias="X-Upload-Token"),
    ):
        session = require_video_upload_session(upload_id, upload_token)
        upload_sessions.pop(upload_id, None)
        shutil.rmtree(session["work_dir"], ignore_errors=True)
        return {"status": "cancelled", "upload_id": upload_id}

    @router.post(
        "/api/video-upload/jobs",
        tags=["Video Upload"],
        summary="上傳影片並建立即時轉譯工作",
    )
    async def create_video_upload_job(
        http_request: Request,
        file: UploadFile = File(...),
        language: str = Form(""),
        captcha_token: str = Form(""),
        include_word_timestamps: bool = Form(True),
        speaker_diarization: bool = Form(False),
        transcription_mode: str = Form("accurate"),
        processing_profile: str = Form("standard"),
        asr_provider: str = Form(""),
        elevenlabs_mode: str = Form("chunks"),
        elevenlabs_keyterms: str = Form(""),
    ):
        cleanup_youtube_live_jobs(jobs)
        if verify_captcha_token is not None:
            verify_captcha_token(captcha_token)
        return await create_video_upload_job_entry(
            http_request,
            file,
            language,
            include_word_timestamps,
            speaker_diarization,
            transcription_mode,
            processing_profile,
            asr_provider,
            elevenlabs_mode,
            elevenlabs_keyterms,
        )

    @router.post(
        "/api/video-upload/jobs/batch",
        tags=["Video Upload"],
        summary="上傳多個影片並建立轉譯工作",
    )
    async def create_video_upload_batch(
        http_request: Request,
        files: list[UploadFile] = File(...),
        language: str = Form(""),
        captcha_token: str = Form(""),
        include_word_timestamps: bool = Form(True),
        speaker_diarization: bool = Form(False),
        transcription_mode: str = Form("accurate"),
        processing_profile: str = Form("standard"),
        asr_provider: str = Form(""),
        elevenlabs_mode: str = Form("chunks"),
        elevenlabs_keyterms: str = Form(""),
    ):
        cleanup_youtube_live_jobs(jobs)
        if not files:
            raise HTTPException(status_code=400, detail="請至少選擇一個影片檔案")
        if len(files) > VIDEO_UPLOAD_BATCH_MAX_FILES:
            raise HTTPException(
                status_code=400,
                detail=f"一次最多上傳 {VIDEO_UPLOAD_BATCH_MAX_FILES} 個影片",
            )
        if verify_captcha_token is not None:
            verify_captcha_token(captcha_token)

        created_jobs: list[dict[str, Any]] = []
        errors: list[dict[str, Any]] = []
        for batch_index, file in enumerate(files):
            filename = Path(file.filename or f"video-{batch_index + 1}").name
            try:
                job = await create_video_upload_job_entry(
                    http_request,
                    file,
                    language,
                    include_word_timestamps,
                    speaker_diarization,
                    transcription_mode,
                    processing_profile,
                    asr_provider,
                    elevenlabs_mode,
                    elevenlabs_keyterms,
                )
                job["batch_index"] = batch_index
                created_jobs.append(job)
            except HTTPException as exc:
                errors.append(
                    {
                        "batch_index": batch_index,
                        "filename": filename,
                        "error": str(exc.detail),
                    }
                )

        return {
            "total_files": len(files),
            "jobs": created_jobs,
            "errors": errors,
        }

    @router.post(
        "/api/youtube-live/jobs/{job_id}/cancel",
        tags=["YouTube Live"],
        summary="取消 Video 轉譯工作",
    )
    async def cancel_youtube_live_job(
        job_id: str,
        payload: YoutubeCancelRequest,
        http_request: Request,
    ):
        cleanup_youtube_live_jobs(jobs)
        job = jobs.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="找不到這個影片轉譯請求")
        expected_token = str(job.get("cancel_token") or "")
        if not payload.cancel_token or not secrets.compare_digest(
            payload.cancel_token,
            expected_token,
        ):
            raise HTTPException(status_code=403, detail="取消權杖無效或已過期")

        previous_status = str(job.get("status") or "")
        translation_tasks = {
            task
            for task in job.get("translation_tasks", set())
            if isinstance(task, asyncio.Task) and not task.done()
        }
        cancelled_translation_tasks = len(translation_tasks)
        if previous_status not in {"done", "failed", "cancelled"}:
            cancel_event = job.get("cancel_event")
            if isinstance(cancel_event, threading.Event):
                cancel_event.set()
            removed_from_queue = cancel_queued_transcribe_task(
                "youtube_live",
                job_id,
            )
            cancelled_asr_requests = cloudflare_asr_scheduler.cancel_job(job_id)
            job["status"] = "cancelled"
            job["cancelled_at"] = time.time()
            job["expires_at"] = time.time() + YOUTUBE_LIVE_JOB_TTL_SECONDS
            job["translation_token"] = ""
            if removed_from_queue or previous_status == "awaiting_language_confirmation":
                cleanup_youtube_live_job_artifacts(job)
            await notify_job_cancelled(job)
            log_structured_event(
                "video_job_cancel_requested",
                job_id=job_id,
                previous_status=previous_status,
                removed_from_queue=removed_from_queue,
                cancelled_asr_requests=cancelled_asr_requests,
                cancelled_translation_tasks=cancelled_translation_tasks,
            )
        if translation_tasks:
            job["translation_token"] = ""
            for translation_task in translation_tasks:
                translation_task.cancel()

        set_request_log_metadata(
            http_request,
            request_id=job_id,
            job_id=job_id,
            operation="video_job_cancel",
            job_status=job.get("status"),
        )
        return {"job_id": job_id, "status": job["status"]}

    @router.post(
        "/api/youtube-live/jobs/{job_id}/language",
        tags=["YouTube Live"],
        summary="確認自動偵測到的原文語言",
    )
    async def confirm_youtube_live_language(
        job_id: str,
        selection: YoutubeLanguageSelection,
        http_request: Request,
        x_translation_token: str = Header(default="", alias="X-Translation-Token"),
    ):
        cleanup_youtube_live_jobs(jobs)
        job = jobs.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="找不到這個影片轉譯請求")
        if not x_translation_token or not secrets.compare_digest(
            x_translation_token,
            str(job.get("translation_token") or ""),
        ):
            raise HTTPException(status_code=401, detail="語言確認權杖無效或已過期")
        if job.get("status") != "awaiting_language_confirmation":
            raise HTTPException(status_code=409, detail="這個轉譯請求目前不需要確認語言")

        requested_language = selection.language.strip().lower()
        auto_language = requested_language in {"", "auto"}
        normalized_language = None
        if not auto_language:
            normalized_language = (
                "zh" if requested_language in {"zh", "zh-tw"} else requested_language
            )
        if normalized_language is not None and normalized_language not in {
            "en",
            "ja",
            "ko",
            "th",
            "zh",
        }:
            raise HTTPException(status_code=400, detail="目前不支援選擇的原文語言")

        confirmation_wait_ms = round(
            max(
                0.0,
                time.perf_counter()
                - float(job.get("language_detected_monotonic", time.perf_counter())),
            )
            * 1000
        )
        job["language_hint"] = normalized_language
        job["language_mode"] = "auto" if auto_language else "forced"
        job["status"] = "queued_for_transcription"
        job["queued_monotonic"] = time.perf_counter()
        job["expires_at"] = time.time() + YOUTUBE_LIVE_JOB_TTL_SECONDS
        set_request_log_metadata(
            http_request,
            request_id=job_id,
            job_id=job_id,
            operation="video_language_confirm",
            job_status="queued_for_transcription",
            detected_language=job.get("detected_language"),
            language_probability=job.get("language_probability"),
            source_language=normalized_language or "auto",
            language_mode=job["language_mode"],
            subtitle_source=job.get("detection_source"),
        )
        try:
            enqueue_transcribe_task(
                video_transcribe_task(
                    job_id,
                    job.get("processing_profile"),
                    job.get("asr_provider"),
                )
            )
        except asyncio.QueueFull:
            job["status"] = "awaiting_language_confirmation"
            set_request_log_metadata(
                http_request,
                job_status="awaiting_language_confirmation",
            )
            raise HTTPException(status_code=503, detail="轉譯佇列已滿，請稍後再試")
        except RuntimeError as exc:
            job["status"] = "awaiting_language_confirmation"
            set_request_log_metadata(
                http_request,
                job_status="awaiting_language_confirmation",
            )
            raise HTTPException(status_code=503, detail="轉譯佇列尚未啟動，請稍後再試") from exc

        queue_counts = get_transcribe_queue_counts()
        log_structured_event(
            "video_language_confirmed",
            job_id=job_id,
            job_status="queued_for_transcription",
            detected_language=job.get("detected_language"),
            language_probability=job.get("language_probability"),
            source_language=normalized_language or "auto",
            language_mode=job["language_mode"],
            subtitle_source=job.get("detection_source"),
            confirmation_wait_ms=confirmation_wait_ms,
            waiting_count=queue_counts["waiting_count"],
            transcribing_count=queue_counts["transcribing_count"],
        )

        return {
            "job_id": job_id,
            "language": normalized_language,
            "language_mode": job["language_mode"],
            "status": "queued",
        }

    @router.post(
        "/api/youtube-live/translate-batch",
        tags=["YouTube Live Translation"],
        summary="代理字幕批次翻譯請求",
        description=(
            "供即時翻譯驗證頁面使用。需要建立影片 job 時取得的短效 "
            "X-Translation-Token，JSON 內容會原樣轉送至翻譯服務。"
        ),
        responses={
            400: {"description": "JSON 或 Content-Length 無效"},
            401: {"description": "翻譯權杖無效或過期"},
            403: {"description": "request_id 與影片 job 不相符"},
            413: {"description": "請求超過 128 KiB"},
            415: {"description": "Content-Type 不是 application/json"},
            502: {"description": "無法連線至翻譯服務"},
            504: {"description": "翻譯服務逾時"},
        },
        openapi_extra={
            "requestBody": {
                "required": True,
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": [
                                "request_id",
                                "source_language",
                                "target_language",
                                "segments",
                            ],
                            "properties": {
                                "request_id": {"type": "string"},
                                "translation_type": {
                                    "type": "string",
                                    "enum": [
                                        "standard",
                                        "premium",
                                        "private",
                                        "std",
                                        "pro",
                                    ],
                                    "default": "standard",
                                },
                                "source_language": {
                                    "type": "string",
                                    "enum": ["en", "ja", "ko", "th", "zh-TW"],
                                },
                                "target_language": {
                                    "type": "string",
                                    "enum": ["en", "ja", "zh-TW"],
                                },
                                "prompt_version": {
                                    "type": "string",
                                    "default": "subtitle-v1",
                                },
                                "context_segments": {
                                    "type": "array",
                                    "maxItems": 5,
                                    "items": {
                                        "type": "object",
                                        "required": [
                                            "id",
                                            "source_text",
                                            "translated_text",
                                        ],
                                    },
                                },
                                "preceding_context_segments": {
                                    "type": "array",
                                    "maxItems": 5,
                                    "items": {
                                        "type": "object",
                                        "required": ["id", "text"],
                                    },
                                },
                                "following_context_segments": {
                                    "type": "array",
                                    "maxItems": 5,
                                    "items": {
                                        "type": "object",
                                        "required": ["id", "text"],
                                    },
                                },
                                "on_screen_terms": {
                                    "type": "array",
                                    "maxItems": 20,
                                    "items": {"type": "string"},
                                },
                                "segments": {
                                    "type": "array",
                                    "minItems": 1,
                                    "maxItems": 40,
                                    "items": {
                                        "type": "object",
                                        "required": ["id", "text"],
                                        "properties": {
                                            "id": {"type": "integer"},
                                            "text": {"type": "string"},
                                            "low_confidence_spans": {
                                                "type": "array",
                                                "maxItems": 20,
                                                "items": {"type": "string"},
                                            },
                                        },
                                    },
                                },
                            },
                        }
                    }
                },
            }
        },
    )
    async def translate_batch_proxy(
        request: Request,
        translation_token: str = Header(
            ...,
            alias="X-Translation-Token",
            description="建立影片 job 時取得的短效翻譯權杖",
        ),
    ):
        cleanup_youtube_live_jobs(jobs)
        content_type = request.headers.get("content-type", "")
        if content_type.split(";", 1)[0].strip().lower() != "application/json":
            raise HTTPException(status_code=415, detail="只接受 application/json")

        content_length = request.headers.get("content-length")
        if content_length:
            try:
                if int(content_length) > TRANSLATE_PROXY_MAX_BODY_BYTES:
                    raise HTTPException(status_code=413, detail="翻譯請求不可超過 128 KiB")
            except ValueError:
                raise HTTPException(status_code=400, detail="Content-Length 無效")

        body = await request.body()
        if len(body) > TRANSLATE_PROXY_MAX_BODY_BYTES:
            raise HTTPException(status_code=413, detail="翻譯請求不可超過 128 KiB")
        try:
            payload = json.loads(body)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise HTTPException(status_code=400, detail="JSON 格式無效") from exc
        if not isinstance(payload, dict):
            raise HTTPException(status_code=400, detail="JSON 根節點必須是物件")

        authorized_job_id = None
        for candidate_id, job in jobs.items():
            expected_token = str(job.get("translation_token") or "")
            if expected_token and secrets.compare_digest(translation_token, expected_token):
                authorized_job_id = candidate_id
                break
        if authorized_job_id is None:
            raise HTTPException(status_code=401, detail="翻譯權杖無效或已過期")

        authorized_job = jobs[authorized_job_id]
        processing_profile = normalize_processing_profile(
            authorized_job.get("processing_profile")
        )
        payload["translation_type"] = translation_type_for_profile(processing_profile)
        body = json.dumps(
            payload, ensure_ascii=False, separators=(",", ":")
        ).encode("utf-8")
        if len(body) > TRANSLATE_PROXY_MAX_BODY_BYTES:
            raise HTTPException(status_code=413, detail="翻譯請求不可超過 128 KiB")

        request_id = str(payload.get("request_id") or "")
        if not request_id.startswith(f"youtube-{authorized_job_id}-"):
            raise HTTPException(status_code=403, detail="request_id 與影片 job 不相符")

        segments = payload.get("segments")
        context_segments = payload.get("context_segments")
        preceding_context_segments = payload.get("preceding_context_segments")
        following_context_segments = payload.get("following_context_segments")
        on_screen_terms = payload.get("on_screen_terms")
        segments = segments if isinstance(segments, list) else []
        context_segments = (
            context_segments if isinstance(context_segments, list) else []
        )
        preceding_context_segments = (
            preceding_context_segments
            if isinstance(preceding_context_segments, list)
            else []
        )
        following_context_segments = (
            following_context_segments
            if isinstance(following_context_segments, list)
            else []
        )
        on_screen_terms = on_screen_terms if isinstance(on_screen_terms, list) else []
        characters = sum(
            len(str(item.get("text") or ""))
            for item in segments
            if isinstance(item, dict)
        )
        low_confidence_span_count = sum(
            len(spans)
            for item in segments + following_context_segments
            if isinstance(item, dict)
            for spans in [item.get("low_confidence_spans")]
            if isinstance(spans, list)
        )
        characters += sum(
            len(str(span or ""))
            for item in segments + following_context_segments
            if isinstance(item, dict)
            for spans in [item.get("low_confidence_spans")]
            if isinstance(spans, list)
            for span in spans
        )
        characters += sum(
            len(str(item.get(key) or ""))
            for item in context_segments
            if isinstance(item, dict)
            for key in ("source_text", "translated_text")
        )
        characters += sum(
            len(str(item.get("text") or ""))
            for item in preceding_context_segments
            if isinstance(item, dict)
        )
        characters += sum(
            len(str(item.get("text") or ""))
            for item in following_context_segments
            if isinstance(item, dict)
        )
        characters += sum(len(str(term or "")) for term in on_screen_terms)
        set_request_log_metadata(
            request,
            request_id=request_id,
            job_id=authorized_job_id,
            operation="video_translation_proxy",
            source_language=payload.get("source_language"),
            target_language=payload.get("target_language"),
            segments=len(segments),
            context_segments=len(context_segments),
            preceding_context_segments=len(preceding_context_segments),
            following_context_segments=len(following_context_segments),
            on_screen_terms=len(on_screen_terms),
            low_confidence_spans=low_confidence_span_count,
            characters=characters,
            processing_profile=processing_profile,
            translation_type=payload.get("translation_type"),
        )

        upstream_url = f"{TRANSLATE_API_BASE}/api/v1/subtitles/translate"
        timeout = aiohttp.ClientTimeout(total=TRANSLATE_API_TIMEOUT_SECONDS)
        started_at = time.perf_counter()
        translation_task = asyncio.current_task()
        if translation_task is not None:
            authorized_job.setdefault("translation_tasks", set()).add(translation_task)
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(
                    upstream_url,
                    data=body,
                    headers={
                        "Accept": "application/json",
                        "Content-Type": "application/json",
                    },
                ) as upstream_response:
                    response_body = await upstream_response.read()
                    response_content_type = upstream_response.headers.get(
                        "Content-Type",
                        "application/json",
                    )
                    logger.info(
                        "Translation proxy completed: job_id=%s status=%d "
                        "request_id=%s source=%s target=%s segments=%d context=%d "
                        "preceding_context=%d following_context=%d on_screen_terms=%d "
                        "low_confidence_spans=%d "
                        "characters=%d request_bytes=%d response_bytes=%d elapsed=%.3fs",
                        authorized_job_id,
                        upstream_response.status,
                        request_id,
                        payload.get("source_language"),
                        payload.get("target_language"),
                        len(segments),
                        len(context_segments),
                        len(preceding_context_segments),
                        len(following_context_segments),
                        len(on_screen_terms),
                        low_confidence_span_count,
                        characters,
                        len(body),
                        len(response_body),
                        time.perf_counter() - started_at,
                    )
                    return Response(
                        content=response_body,
                        status_code=upstream_response.status,
                        headers={"Content-Type": response_content_type},
                    )
        except asyncio.CancelledError:
            logger.info(
                "Translation proxy cancelled: job_id=%s request_id=%s elapsed=%.3fs",
                authorized_job_id,
                request_id,
                time.perf_counter() - started_at,
            )
            return JSONResponse(
                status_code=499,
                content={
                    "detail": "翻譯已取消",
                    "retryable": False,
                },
            )
        except asyncio.TimeoutError:
            return JSONResponse(
                status_code=504,
                content={
                    "detail": "翻譯服務逾時",
                    "retryable": True,
                },
            )
        except aiohttp.ClientError:
            logger.exception(
                "Translation proxy failed: job_id=%s elapsed=%.3fs",
                authorized_job_id,
                time.perf_counter() - started_at,
            )
            return JSONResponse(
                status_code=502,
                content={
                    "detail": "無法連線至翻譯服務",
                    "retryable": True,
                },
            )
        finally:
            if translation_task is not None:
                authorized_job.get("translation_tasks", set()).discard(
                    translation_task
                )

    @router.post(
        "/api/youtube-live/translation-workflow/{operation}",
        tags=["YouTube Live Translation"],
        summary="代理來源語分組或群組翻譯請求",
        responses={
            400: {"description": "操作、JSON 或 Content-Length 無效"},
            401: {"description": "翻譯權杖無效或過期"},
            403: {"description": "request_id 與影片 job 不相符"},
            413: {"description": "請求超過 128 KiB"},
            415: {"description": "Content-Type 不是 application/json"},
            502: {"description": "無法連線至翻譯服務"},
            504: {"description": "翻譯服務逾時"},
        },
    )
    async def translation_workflow_proxy(
        operation: str,
        request: Request,
        translation_token: str = Header(
            ...,
            alias="X-Translation-Token",
            description="建立影片 job 時取得的短效翻譯權杖",
        ),
    ):
        upstream_paths = {
            "group": "/api/v1/subtitles/group",
            "translate-groups": "/api/v1/subtitles/translate-groups",
        }
        upstream_path = upstream_paths.get(operation)
        if upstream_path is None:
            raise HTTPException(status_code=400, detail="不支援的翻譯工作流程操作")

        cleanup_youtube_live_jobs(jobs)
        content_type = request.headers.get("content-type", "")
        if content_type.split(";", 1)[0].strip().lower() != "application/json":
            raise HTTPException(status_code=415, detail="只接受 application/json")

        content_length = request.headers.get("content-length")
        if content_length:
            try:
                if int(content_length) > TRANSLATE_PROXY_MAX_BODY_BYTES:
                    raise HTTPException(status_code=413, detail="翻譯請求不可超過 128 KiB")
            except ValueError:
                raise HTTPException(status_code=400, detail="Content-Length 無效")

        body = await request.body()
        if len(body) > TRANSLATE_PROXY_MAX_BODY_BYTES:
            raise HTTPException(status_code=413, detail="翻譯請求不可超過 128 KiB")
        try:
            payload = json.loads(body)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise HTTPException(status_code=400, detail="JSON 格式無效") from exc
        if not isinstance(payload, dict):
            raise HTTPException(status_code=400, detail="JSON 根節點必須是物件")

        authorized_job_id = None
        for candidate_id, job in jobs.items():
            expected_token = str(job.get("translation_token") or "")
            if expected_token and secrets.compare_digest(translation_token, expected_token):
                authorized_job_id = candidate_id
                break
        if authorized_job_id is None:
            raise HTTPException(status_code=401, detail="翻譯權杖無效或已過期")

        authorized_job = jobs[authorized_job_id]
        processing_profile = normalize_processing_profile(
            authorized_job.get("processing_profile")
        )
        payload = route_translation_workflow_payload(
            payload,
            operation,
            processing_profile,
        )
        body = json.dumps(
            payload, ensure_ascii=False, separators=(",", ":")
        ).encode("utf-8")
        if len(body) > TRANSLATE_PROXY_MAX_BODY_BYTES:
            raise HTTPException(status_code=413, detail="翻譯請求不可超過 128 KiB")

        request_id = str(payload.get("request_id") or "")
        if not request_id.startswith(f"youtube-{authorized_job_id}-"):
            raise HTTPException(status_code=403, detail="request_id 與影片 job 不相符")

        segments = payload.get("segments")
        groups = payload.get("groups")
        segments = segments if isinstance(segments, list) else []
        groups = groups if isinstance(groups, list) else []
        preceding_source_context = payload.get("preceding_source_context")
        preceding_source_context = (
            preceding_source_context
            if isinstance(preceding_source_context, list)
            else []
        )
        source_id_count = sum(
            len(source_ids)
            for item in groups
            if isinstance(item, dict)
            for source_ids in [item.get("source_ids")]
            if isinstance(source_ids, list)
        )
        translation_source_ids: set[int | str] = {
            source_id
            for item in groups
            if isinstance(item, dict)
            for source_ids in [item.get("source_ids")]
            if isinstance(source_ids, list)
            for source_id in source_ids
            if isinstance(source_id, (int, str))
        }
        characters = sum(
            len(str(item.get("text") or ""))
            for item in segments
            if isinstance(item, dict)
        ) + sum(
            len(str(item.get("source_text") or ""))
            for item in groups
            if isinstance(item, dict)
        )
        characters += sum(
            len(str(item.get("source_text") or ""))
            for item in preceding_source_context
            if isinstance(item, dict)
        )
        set_request_log_metadata(
            request,
            request_id=request_id,
            job_id=authorized_job_id,
            operation=f"video_translation_{operation}",
            source_language=payload.get("source_language"),
            target_language=payload.get("target_language"),
            processing_profile=processing_profile,
            translation_type=payload.get("translation_type"),
            segments=len(segments),
            groups=len(groups),
            preceding_source_groups=len(preceding_source_context),
            source_ids=source_id_count,
            characters=characters,
            final=payload.get("final"),
        )

        upstream_url = f"{TRANSLATE_API_BASE}{upstream_path}"
        timeout = aiohttp.ClientTimeout(total=TRANSLATE_API_TIMEOUT_SECONDS)
        started_at = time.perf_counter()
        workflow_task = asyncio.current_task()
        if workflow_task is not None:
            authorized_job.setdefault("translation_tasks", set()).add(workflow_task)
        translation_request_started(
            authorized_job,
            operation,
            translation_source_ids,
        )
        telemetry_status_code = 500
        telemetry_provider: str | None = None
        telemetry_usage: dict[str, Any] = {}
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(
                    upstream_url,
                    data=body,
                    headers={
                        "Accept": "application/json",
                        "Content-Type": "application/json",
                    },
                ) as upstream_response:
                    response_body = await upstream_response.read()
                    response_content_type = upstream_response.headers.get(
                        "Content-Type",
                        "application/json",
                    )
                    response_payload = None
                    try:
                        response_payload = json.loads(response_body)
                    except (UnicodeDecodeError, json.JSONDecodeError):
                        pass
                    usage = (
                        response_payload.get("usage")
                        if isinstance(response_payload, dict)
                        else None
                    )
                    usage = usage if isinstance(usage, dict) else {}
                    telemetry_status_code = upstream_response.status
                    telemetry_provider = (
                        response_payload.get("provider")
                        if isinstance(response_payload, dict)
                        else None
                    )
                    telemetry_usage = usage
                    set_request_log_metadata(
                        request,
                        translation_provider=(
                            response_payload.get("provider")
                            if isinstance(response_payload, dict)
                            else None
                        ),
                        counted_input_tokens=usage.get("counted_input_tokens"),
                        prompt_tokens=usage.get("prompt_tokens"),
                        output_tokens=usage.get("output_tokens"),
                        estimated_cost_usd=usage.get("estimated_total_cost_usd"),
                        estimated_cost_twd=usage.get("estimated_total_cost_twd"),
                    )
                    logger.info(
                        "Translation workflow proxy completed: operation=%s "
                        "job_id=%s status=%d request_id=%s segments=%d groups=%d "
                        "preceding_source_groups=%d "
                        "source_ids=%d characters=%d request_bytes=%d "
                        "response_bytes=%d translation_type=%s provider=%s "
                        "counted_input_tokens=%s prompt_tokens=%s output_tokens=%s "
                        "estimated_cost_usd=%s estimated_cost_twd=%s elapsed=%.3fs",
                        operation,
                        authorized_job_id,
                        upstream_response.status,
                        request_id,
                        len(segments),
                        len(groups),
                        len(preceding_source_context),
                        source_id_count,
                        characters,
                        len(body),
                        len(response_body),
                        payload.get("translation_type", "standard"),
                        (
                            response_payload.get("provider")
                            if isinstance(response_payload, dict)
                            else None
                        ),
                        usage.get("counted_input_tokens"),
                        usage.get("prompt_tokens"),
                        usage.get("output_tokens"),
                        usage.get("estimated_total_cost_usd"),
                        usage.get("estimated_total_cost_twd"),
                        time.perf_counter() - started_at,
                    )
                    return Response(
                        content=response_body,
                        status_code=upstream_response.status,
                        headers={"Content-Type": response_content_type},
                    )
        except asyncio.CancelledError:
            telemetry_status_code = 499
            return JSONResponse(
                status_code=499,
                content={"detail": "翻譯已取消", "retryable": False},
            )
        except asyncio.TimeoutError:
            telemetry_status_code = 504
            return JSONResponse(
                status_code=504,
                content={"detail": "翻譯服務逾時", "retryable": True},
            )
        except aiohttp.ClientError:
            telemetry_status_code = 502
            logger.exception(
                "Translation workflow proxy failed: operation=%s job_id=%s "
                "request_id=%s elapsed=%.3fs",
                operation,
                authorized_job_id,
                request_id,
                time.perf_counter() - started_at,
            )
            return JSONResponse(
                status_code=502,
                content={"detail": "無法連線至翻譯服務", "retryable": True},
            )
        finally:
            translation_request_finished(
                authorized_job,
                status_code=telemetry_status_code,
                latency_ms=(time.perf_counter() - started_at) * 1000,
                source_ids=translation_source_ids,
                provider=telemetry_provider,
                usage=telemetry_usage,
            )
            if workflow_task is not None:
                authorized_job.get("translation_tasks", set()).discard(workflow_task)

    @router.get("/api/youtube-live/jobs/{job_id}/events", tags=["YouTube Live"])
    async def youtube_live_events(job_id: str, request: Request):
        cleanup_youtube_live_jobs(jobs)
        job = jobs.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="找不到這個影片轉譯請求")
        set_request_log_metadata(
            request,
            request_id=job_id,
            job_id=job_id,
            operation="video_job_events",
            job_status=job.get("status"),
        )

        async def event_stream():
            yield sse_message("status", {"message": "已連線，等待處理結果"})
            if job.get("status") == "awaiting_language_confirmation":
                yield sse_message(
                    "language_detected",
                    {
                        "language": job.get("detected_language"),
                        "language_probability": job.get("language_probability"),
                        "source": job.get("detection_source"),
                    },
                )
            while True:
                try:
                    item = await asyncio.wait_for(
                        job["queue"].get(),
                        timeout=YOUTUBE_LIVE_EVENT_TIMEOUT_SECONDS,
                    )
                except asyncio.TimeoutError:
                    yield ": keep-alive\n\n"
                    continue

                if item["event"] == "close":
                    break
                yield sse_message(item["event"], item["data"])

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Content-Encoding": "identity",
                "X-Accel-Buffering": "no",
            },
        )

    return router
