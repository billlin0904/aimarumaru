import asyncio
import json
import logging
import math
import os
import secrets
import shutil
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from collections.abc import Callable
from typing import Any, Optional

import aiohttp
import numpy as np
from fastapi import APIRouter, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse
from pydantic import BaseModel

from app_logging import log_structured_event, set_request_log_metadata
from media_audio_stream import (
    FFmpegPcmChunkStream,
    MediaSourceInput,
    decode_media_prefix,
    probe_media_duration,
)
from text_converter import to_traditional_chinese
from transcribe_queue import (
    TranscriptionCancelled,
    cancel_queued_transcribe_task,
    enqueue_transcribe_task,
    get_transcribe_queue_counts,
    register_transcribe_cleanup,
    register_transcribe_handler,
)
from youtube_srt import (
    choose_subtitle_track,
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
TRANSLATE_API_BASE = os.getenv(
    "TRANSLATE_API_BASE",
    "https://translate.audio-io.com",
).rstrip("/")
TRANSLATE_API_TIMEOUT_SECONDS = float(
    os.getenv("TRANSLATE_API_TIMEOUT_SECONDS", "150")
)
TRANSLATE_PROXY_MAX_BODY_BYTES = 128 * 1024
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
YOUTUBE_WHISPER_LANGUAGE_DETECT_SECONDS = max(
    10.0,
    float(os.getenv("YOUTUBE_WHISPER_LANGUAGE_DETECT_SECONDS", "30")),
)
VIDEO_UPLOAD_MAX_BYTES = int(
    os.getenv("VIDEO_UPLOAD_MAX_BYTES", str(2 * 1024 * 1024 * 1024))
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


def normalize_transcription_mode(value: Optional[str]) -> str:
    mode = str(value or "accurate").strip().lower()
    if mode not in TRANSCRIPTION_MODE_BEAM_SIZES:
        raise HTTPException(status_code=422, detail="不支援的轉譯模式")
    return mode


class YoutubeLiveRequest(BaseModel):
    url: str
    language: str = ""
    captcha_token: str = ""
    ignore_subtitles: bool = False
    include_word_timestamps: bool = False
    transcription_mode: str = "accurate"


class YoutubeLanguageSelection(BaseModel):
    language: str


class YoutubeCancelRequest(BaseModel):
    cancel_token: str


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
) -> dict[str, Any]:
    started_at = time.perf_counter()
    mode = (
        transcription_mode
        if transcription_mode in TRANSCRIPTION_MODE_BEAM_SIZES
        else "accurate"
    )
    beam_size = TRANSCRIPTION_MODE_BEAM_SIZES[mode]
    audio_duration = max(
        0.0,
        float(audio_duration_hint or probe_media_duration(audio_path) or 0.0),
    )
    detected_language = language_hint
    language_probability = None
    segment_count = 0
    decoded_segment_count = 0
    chunk_count = 0
    processed_seconds = 0.0
    last_emitted_start = 0.0
    content_parts: list[str] = []
    logger.info(
        "Whisper transcription started: job_id=%s transcription_mode=%s "
        "beam_size=%d language_hint=%s "
        "temperature=0.0 condition_on_previous_text=false vad_filter=%s "
        "word_timestamps=true hallucination_silence_threshold=%.2f "
        "initial_chunk_seconds=%.1f chunk_seconds=%.1f "
        "overlap_seconds=%.1f pcm_queue_size=%d",
        job_id or "unknown",
        mode,
        beam_size,
        language_hint or "auto",
        YOUTUBE_WHISPER_VAD_FILTER,
        YOUTUBE_WHISPER_HALLUCINATION_SILENCE_SECONDS,
        YOUTUBE_WHISPER_INITIAL_CHUNK_SECONDS,
        YOUTUBE_WHISPER_STREAM_CHUNK_SECONDS,
        YOUTUBE_WHISPER_STREAM_OVERLAP_SECONDS,
        YOUTUBE_WHISPER_STREAM_QUEUE_SIZE,
    )

    try:
        if cancel_check is not None and cancel_check():
            raise TranscriptionCancelled("轉譯已取消")
        with FFmpegPcmChunkStream(
            audio_path,
            chunk_seconds=YOUTUBE_WHISPER_STREAM_CHUNK_SECONDS,
            initial_chunk_seconds=YOUTUBE_WHISPER_INITIAL_CHUNK_SECONDS,
            overlap_seconds=YOUTUBE_WHISPER_STREAM_OVERLAP_SECONDS,
            queue_size=YOUTUBE_WHISPER_STREAM_QUEUE_SIZE,
            cancel_check=cancel_check,
        ) as audio_chunks:
            for chunk in audio_chunks:
                if cancel_check is not None and cancel_check():
                    raise TranscriptionCancelled("轉譯已取消")
                chunk_count += 1
                audio_samples = np.frombuffer(
                    chunk.data,
                    dtype="<i2",
                ).astype(np.float32)
                audio_samples /= 32768.0
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
                chunk_language = getattr(info, "language", detected_language)
                if chunk_language:
                    detected_language = chunk_language
                if language_probability is None:
                    language_probability = getattr(info, "language_probability", None)

                ownership_start = (
                    YOUTUBE_WHISPER_STREAM_OVERLAP_SECONDS / 2
                    if chunk.offset_seconds > 0
                    else 0.0
                )
                ownership_end = chunk.duration_seconds
                if not chunk.is_final:
                    ownership_end = max(
                        0.0,
                        chunk.duration_seconds
                        - YOUTUBE_WHISPER_STREAM_OVERLAP_SECONDS / 2,
                    )
                for segment in segments:
                    decoded_segment_count += 1
                    if cancel_check is not None and cancel_check():
                        raise TranscriptionCancelled("轉譯已取消")
                    owned_segment = owned_whisper_segment(
                        segment,
                        ownership_start,
                        ownership_end,
                        chunk.is_final,
                    )
                    if owned_segment is None:
                        continue
                    raw_text, local_start, local_end, owned_words = owned_segment
                    text = to_traditional_chinese(
                        raw_text,
                        detected_language,
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
                        detected_language,
                        whisper_low_confidence_spans(
                            segment,
                            text,
                            detected_language,
                            owned_words,
                        ),
                        (
                            (
                                []
                                if timeline_adjusted
                                else whisper_word_payloads(
                                    segment,
                                    detected_language,
                                    chunk.offset_seconds,
                                    owned_words,
                                )
                            )
                            if include_word_timestamps
                            else None
                        ),
                    )
                    last_emitted_start = global_start
                    payload.update(
                        transcription_progress_payload(
                            audio_duration,
                            global_end,
                            time.perf_counter() - started_at,
                        )
                    )
                    put_thread_event(
                        loop,
                        queue,
                        {
                            "event": "segment",
                            "data": payload,
                        },
                    )

                processed_seconds = chunk.offset_seconds + ownership_end
                if chunk.is_final:
                    audio_duration = max(audio_duration, processed_seconds)
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
            "Whisper transcription completed: job_id=%s "
            "transcription_mode=%s beam_size=%d "
            "audio_duration=%.3fs elapsed=%.3fs real_time_factor=%s "
            "speed=%s decoded_segments=%d emitted_segments=%d chunks=%d language=%s",
            job_id or "unknown",
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
        auto2lrc.clear_model_cache()


def detect_audio_language(
    auto2lrc,
    audio_path: MediaSourceInput,
    job_id: Optional[str] = None,
    cancel_check: Optional[Callable[[], bool]] = None,
    audio_duration_hint: Optional[float] = None,
) -> dict[str, Any]:
    logger.info(
        "Whisper language detection started: job_id=%s",
        job_id or "unknown",
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
        segments, info = auto2lrc.transcribe(
            audio_samples,
            beam_size=5,
            language=None,
        )
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
            "duration": audio_duration,
            "detection_elapsed_seconds": round(time.perf_counter() - started_at, 3),
        }
    finally:
        auto2lrc.clear_model_cache()


def create_youtube_live_router(auto2lrc, project_root: Path, verify_captcha_token=None) -> APIRouter:
    router = APIRouter()
    page_path = project_root / "pages" / "youtube_live.html"
    translate_page_path = project_root / "pages" / "youtube_live_translate.html"
    translate_script_path = project_root / "pages" / "youtube_live_translate.js"
    subtitle_display_cues_script_path = (
        project_root / "pages" / "subtitle_display_cues.js"
    )
    cookies_file = project_root / "cookies.txt"
    jobs: dict[str, dict[str, Any]] = {}

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

    async def run_job(job_id: str) -> None:
        job = jobs.get(job_id)
        if not job:
            return
        if job_is_cancelled(job):
            cleanup_youtube_live_job_artifacts(job)
            return

        queue: asyncio.Queue[dict[str, Any]] = job["queue"]
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
                        audio_source, _ = await asyncio.to_thread(
                            get_youtube_audio_stream_source,
                            job["url"],
                            cookies_file,
                        )
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
                )
                ensure_job_not_cancelled(job)
                job["detected_language"] = detection["language"]
                job["detection_source"] = "whisper"
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
                        "source": "whisper",
                    },
                )
                log_structured_event(
                    "video_job_awaiting_language",
                    job_id=job_id,
                    job_status="awaiting_language_confirmation",
                    source_kind="upload",
                    detected_language=detection["language"],
                    language_probability=detection["language_probability"],
                    subtitle_source="whisper",
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
            audio_source, video_info = await asyncio.to_thread(
                get_youtube_audio_stream_source,
                job["url"],
                cookies_file,
            )
            ensure_job_not_cancelled(job)
            job["prepared_source"] = "whisper"
            log_structured_event(
                "video_audio_stream_resolved",
                job_id=job_id,
                stream_protocol=video_info.get("protocol"),
                audio_format=video_info.get("format_id"),
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
                )
                ensure_job_not_cancelled(job)
                job["detected_language"] = detection["language"]
                job["detection_source"] = "whisper"
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
                        "source": "whisper",
                    },
                )
                log_structured_event(
                    "video_job_awaiting_language",
                    job_id=job_id,
                    job_status="awaiting_language_confirmation",
                    detected_language=detection["language"],
                    language_probability=detection["language_probability"],
                    subtitle_source="whisper",
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
            )
            ensure_job_not_cancelled(job)

            job["status"] = "done"
            terminal = True
            await push_event(job, "done", {"source": "whisper", **info})
            log_structured_event(
                "video_job_completed",
                job_id=job_id,
                job_status="done",
                subtitle_source="whisper",
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
            if terminal:
                cleanup_youtube_live_job_artifacts(job)
                await close_event_stream(job)

    async def handle_youtube_live_task(task: dict[str, Any]) -> None:
        await run_job(str(task.get("id", "")))

    register_transcribe_handler("youtube_live", handle_youtube_live_task)
    register_transcribe_cleanup(lambda: cleanup_youtube_live_jobs(jobs))

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
        jobs[job_id] = {
            "url": url,
            "language_hint": payload.language.strip() or None,
            "ignore_subtitles": payload.ignore_subtitles,
            "include_word_timestamps": payload.include_word_timestamps,
            "transcription_mode": transcription_mode,
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
            transcription_mode=transcription_mode,
        )
        try:
            enqueue_transcribe_task({"kind": "youtube_live", "id": job_id})
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
            transcription_mode=transcription_mode,
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
        }

    async def create_video_upload_job_entry(
        http_request: Request,
        file: UploadFile,
        language: str,
        include_word_timestamps: bool,
        transcription_mode: str,
    ) -> dict[str, Any]:
        transcription_mode = normalize_transcription_mode(transcription_mode)
        original_filename = Path(file.filename or "video").name
        suffix = Path(original_filename).suffix.lower()
        content_type = str(file.content_type or "").lower()
        if not content_type.startswith("video/") and suffix not in VIDEO_UPLOAD_EXTENSIONS:
            raise HTTPException(status_code=400, detail="請選擇支援的影片檔案")

        job_id = secrets.token_urlsafe(18)
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
            raise HTTPException(status_code=500, detail=f"儲存上傳影片失敗: {exc}") from exc
        finally:
            await file.close()

        if upload_bytes == 0:
            shutil.rmtree(work_dir, ignore_errors=True)
            raise HTTPException(status_code=400, detail="上傳影片不可為空")

        language_hint = language.strip() or None
        created_monotonic = time.perf_counter()
        jobs[job_id] = {
            "source_kind": "upload",
            "filename": original_filename,
            "language_hint": language_hint,
            "ignore_subtitles": True,
            "include_word_timestamps": include_word_timestamps,
            "transcription_mode": transcription_mode,
            "prepared_source": "whisper",
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
            transcription_mode=transcription_mode,
        )
        try:
            enqueue_transcribe_task({"kind": "youtube_live", "id": job_id})
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
            transcription_mode=transcription_mode,
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
        }

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
        transcription_mode: str = Form("accurate"),
    ):
        cleanup_youtube_live_jobs(jobs)
        if verify_captcha_token is not None:
            verify_captcha_token(captcha_token)
        return await create_video_upload_job_entry(
            http_request,
            file,
            language,
            include_word_timestamps,
            transcription_mode,
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
        transcription_mode: str = Form("accurate"),
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
                    transcription_mode,
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

        requested_language = selection.language.strip()
        normalized_language = (
            "zh"
            if requested_language.lower() in {"zh", "zh-tw"}
            else requested_language.lower()
        )
        if normalized_language not in {"en", "ja", "ko", "th", "zh"}:
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
            source_language=normalized_language,
            subtitle_source=job.get("detection_source"),
        )
        try:
            enqueue_transcribe_task({"kind": "youtube_live", "id": job_id})
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
            source_language=normalized_language,
            subtitle_source=job.get("detection_source"),
            confirmation_wait_ms=confirmation_wait_ms,
            waiting_count=queue_counts["waiting_count"],
            transcribing_count=queue_counts["transcribing_count"],
        )

        return {"job_id": job_id, "language": normalized_language, "status": "queued"}

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
        )

        upstream_url = f"{TRANSLATE_API_BASE}/api/v1/subtitles/translate"
        timeout = aiohttp.ClientTimeout(total=TRANSLATE_API_TIMEOUT_SECONDS)
        started_at = time.perf_counter()
        authorized_job = jobs[authorized_job_id]
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

        request_id = str(payload.get("request_id") or "")
        if not request_id.startswith(f"youtube-{authorized_job_id}-"):
            raise HTTPException(status_code=403, detail="request_id 與影片 job 不相符")

        segments = payload.get("segments")
        groups = payload.get("groups")
        segments = segments if isinstance(segments, list) else []
        groups = groups if isinstance(groups, list) else []
        source_id_count = sum(
            len(source_ids)
            for item in groups
            if isinstance(item, dict)
            for source_ids in [item.get("source_ids")]
            if isinstance(source_ids, list)
        )
        characters = sum(
            len(str(item.get("text") or ""))
            for item in segments
            if isinstance(item, dict)
        ) + sum(
            len(str(item.get("source_text") or ""))
            for item in groups
            if isinstance(item, dict)
        )
        set_request_log_metadata(
            request,
            request_id=request_id,
            job_id=authorized_job_id,
            operation=f"video_translation_{operation}",
            source_language=payload.get("source_language"),
            target_language=payload.get("target_language"),
            segments=len(segments),
            groups=len(groups),
            source_ids=source_id_count,
            characters=characters,
            final=payload.get("final"),
        )

        upstream_url = f"{TRANSLATE_API_BASE}{upstream_path}"
        timeout = aiohttp.ClientTimeout(total=TRANSLATE_API_TIMEOUT_SECONDS)
        started_at = time.perf_counter()
        authorized_job = jobs[authorized_job_id]
        workflow_task = asyncio.current_task()
        if workflow_task is not None:
            authorized_job.setdefault("translation_tasks", set()).add(workflow_task)
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
                        "Translation workflow proxy completed: operation=%s "
                        "job_id=%s status=%d request_id=%s segments=%d groups=%d "
                        "source_ids=%d characters=%d request_bytes=%d "
                        "response_bytes=%d elapsed=%.3fs",
                        operation,
                        authorized_job_id,
                        upstream_response.status,
                        request_id,
                        len(segments),
                        len(groups),
                        source_id_count,
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
            return JSONResponse(
                status_code=499,
                content={"detail": "翻譯已取消", "retryable": False},
            )
        except asyncio.TimeoutError:
            return JSONResponse(
                status_code=504,
                content={"detail": "翻譯服務逾時", "retryable": True},
            )
        except aiohttp.ClientError:
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
