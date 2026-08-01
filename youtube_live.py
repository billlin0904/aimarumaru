import asyncio
import json
import logging
import os
import secrets
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import aiohttp
from fastapi import APIRouter, Header, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse
from pydantic import BaseModel

from text_converter import to_traditional_chinese
from transcribe_queue import enqueue_transcribe_task, register_transcribe_cleanup, register_transcribe_handler
from youtube_srt import (
    choose_subtitle_track,
    download_subtitle_content,
    download_youtube_audio,
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


class YoutubeLiveRequest(BaseModel):
    url: str
    language: str = ""
    captcha_token: str = ""
    ignore_subtitles: bool = False


def sse_message(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def segment_payload(
    index: int,
    start: Optional[float],
    end: Optional[float],
    text: str,
    language: Optional[str] = None,
) -> dict[str, Any]:
    return {
        "index": index,
        "start": start,
        "end": end,
        "text": text.strip(),
        "language": language,
    }


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
        if job.get("expires_at", 0) <= now and job.get("status") not in {"running", "queued"}
    ]
    for job_id in expired_ids:
        jobs.pop(job_id, None)


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
    audio_path: Path,
    language_hint: Optional[str],
    loop: asyncio.AbstractEventLoop,
    queue: asyncio.Queue[dict[str, Any]],
) -> dict[str, Any]:
    started_at = time.perf_counter()
    audio_duration = 0.0
    detected_language = language_hint
    segment_count = 0
    content_parts: list[str] = []
    logger.info(
        "Whisper transcription started: beam_size=5 language_hint=%s "
        "temperature=0.0 condition_on_previous_text=false vad_filter=false",
        language_hint or "auto",
    )

    try:
        segments, info = auto2lrc.transcribe(
            str(audio_path),
            beam_size=5,
            language=language_hint,
        )
        detected_language = getattr(info, "language", language_hint)
        audio_duration = float(getattr(info, "duration", 0.0) or 0.0)
        for segment_count, segment in enumerate(segments, start=1):
            text = to_traditional_chinese(
                segment.text.strip(),
                detected_language,
            )
            if not text:
                continue
            content_parts.append(text)
            payload = segment_payload(
                segment_count,
                segment.start,
                segment.end,
                text,
                detected_language,
            )
            payload.update(
                transcription_progress_payload(
                    audio_duration,
                    float(segment.end or 0.0),
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
            "Whisper transcription completed: "
            "audio_duration=%.3fs elapsed=%.3fs real_time_factor=%s "
            "speed=%s segments=%d emitted_segments=%d language=%s",
            audio_duration,
            elapsed_seconds,
            f"{real_time_factor:.4f}" if real_time_factor is not None else "unknown",
            f"{speed_ratio:.2f}x" if speed_ratio is not None else "unknown",
            segment_count,
            len(content_parts),
            detected_language or "unknown",
        )
        return {
            "language": detected_language,
            "language_probability": getattr(info, "language_probability", None),
            "segments_count": segment_count,
            "content": "\n".join(content_parts),
        }
    except Exception:
        logger.exception(
            "Whisper transcription failed: elapsed=%.3fs segments=%d",
            time.perf_counter() - started_at,
            segment_count,
        )
        raise
    finally:
        auto2lrc.clear_model_cache()


def create_youtube_live_router(auto2lrc, project_root: Path, verify_captcha_token=None) -> APIRouter:
    router = APIRouter()
    page_path = project_root / "pages" / "youtube_live.html"
    translate_page_path = project_root / "pages" / "youtube_live_translate.html"
    translate_script_path = project_root / "pages" / "youtube_live_translate.js"
    cookies_file = project_root / "cookies.txt"
    jobs: dict[str, dict[str, Any]] = {}

    async def push_event(job: dict[str, Any], event: str, data: dict[str, Any]) -> None:
        await job["queue"].put({"event": event, "data": data})

    async def run_job(job_id: str) -> None:
        job = jobs.get(job_id)
        if not job:
            return

        queue: asyncio.Queue[dict[str, Any]] = job["queue"]
        try:
            job["status"] = "running"
            if job["ignore_subtitles"]:
                await push_event(job, "status", {"message": "已略過內建字幕，準備下載音訊"})
            else:
                await push_event(job, "status", {"message": "檢查 YouTube 字幕中"})

            video_info = await asyncio.to_thread(get_youtube_video_info, job["url"], cookies_file)
            chapters = chapter_payloads(video_info)
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
                await push_event(
                    job,
                    "status",
                    {
                        "message": "找到內建字幕，正在解析",
                        "source": subtitle_track["source"],
                        "language": subtitle_track["language"],
                    },
                )
                subtitle_content = await asyncio.to_thread(download_subtitle_content, subtitle_track["url"])
                segments = parse_subtitle_content(subtitle_content, subtitle_track["extension"])
                content_parts: list[str] = []
                for index, segment in enumerate(segments, start=1):
                    text = to_traditional_chinese(
                        segment["text"].strip(),
                        subtitle_track["language"],
                    )
                    if not text:
                        continue
                    content_parts.append(text)
                    await push_event(
                        job,
                        "segment",
                        segment_payload(
                            index,
                            segment["start"],
                            segment["end"],
                            text,
                            subtitle_track["language"],
                        ),
                    )
                    await asyncio.sleep(0)

                job["status"] = "done"
                await push_event(
                    job,
                    "done",
                    {
                        "source": subtitle_track["source"],
                        "language": subtitle_track["language"],
                        "segments_count": len(segments),
                        "content": "\n".join(content_parts),
                    },
                )
                return

            status_message = "正在下載音訊" if job["ignore_subtitles"] else "沒有可用字幕，正在下載音訊"
            await push_event(job, "status", {"message": status_message})
            with tempfile.TemporaryDirectory(prefix="yt_live_") as temp_dir:
                audio_path, video_info = await asyncio.to_thread(download_youtube_audio, job["url"], temp_dir, cookies_file)
                await push_event(job, "status", {"message": "音訊下載完成，開始逐段轉譯"})
                info = await asyncio.to_thread(
                    transcribe_audio_stream,
                    auto2lrc,
                    audio_path,
                    job["language_hint"],
                    asyncio.get_running_loop(),
                    queue,
                )

            job["status"] = "done"
            await push_event(job, "done", {"source": "whisper", **info})
        except Exception as exc:
            job["status"] = "failed"
            await push_event(job, "failed", {"message": readable_exception_message(exc)})
        finally:
            job["expires_at"] = time.time() + YOUTUBE_LIVE_JOB_TTL_SECONDS
            await queue.put({"event": "close", "data": {}})

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

    @router.get("/youtube-live-translate.js", include_in_schema=False)
    def youtube_live_translate_script():
        return Response(
            translate_script_path.read_text(encoding="utf-8"),
            media_type="application/javascript",
            headers={"Cache-Control": "no-store, max-age=0"},
        )

    @router.post("/api/youtube-live/jobs", tags=["YouTube Live"])
    async def create_youtube_live_job(request: YoutubeLiveRequest):
        cleanup_youtube_live_jobs(jobs)
        url = request.url.strip()
        if not url:
            raise HTTPException(status_code=400, detail="請輸入 YouTube 網址")
        if verify_captcha_token is not None:
            verify_captcha_token(request.captcha_token)

        job_id = secrets.token_urlsafe(18)
        jobs[job_id] = {
            "url": url,
            "language_hint": request.language.strip() or None,
            "ignore_subtitles": request.ignore_subtitles,
            "translation_token": secrets.token_urlsafe(32),
            "status": "queued",
            "queue": asyncio.Queue(),
            "created_at": time.time(),
            "expires_at": time.time() + YOUTUBE_LIVE_JOB_TTL_SECONDS,
        }
        try:
            enqueue_transcribe_task({"kind": "youtube_live", "id": job_id})
        except asyncio.QueueFull:
            jobs[job_id]["status"] = "failed"
            jobs[job_id]["expires_at"] = time.time() + YOUTUBE_LIVE_JOB_TTL_SECONDS
            raise HTTPException(status_code=503, detail="轉譯佇列已滿，請稍後再試")
        except RuntimeError as exc:
            jobs[job_id]["status"] = "failed"
            jobs[job_id]["expires_at"] = time.time() + YOUTUBE_LIVE_JOB_TTL_SECONDS
            raise HTTPException(status_code=503, detail="轉譯佇列尚未啟動，請稍後再試") from exc

        return {
            "job_id": job_id,
            "events_url": f"/api/youtube-live/jobs/{job_id}/events",
            "translation_token": jobs[job_id]["translation_token"],
            "status": "queued",
        }

    @router.post(
        "/api/youtube-live/translate-batch",
        tags=["YouTube Live Translation"],
        summary="代理字幕批次翻譯請求",
        description=(
            "供即時翻譯驗證頁面使用。需要建立 YouTube job 時取得的短效 "
            "X-Translation-Token，JSON 內容會原樣轉送至翻譯服務。"
        ),
        responses={
            400: {"description": "JSON 或 Content-Length 無效"},
            401: {"description": "翻譯權杖無效或過期"},
            403: {"description": "request_id 與 YouTube job 不相符"},
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
                                    "enum": ["en", "ja", "ko", "zh-TW"],
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
                                "segments": {
                                    "type": "array",
                                    "minItems": 1,
                                    "maxItems": 40,
                                    "items": {
                                        "type": "object",
                                        "required": ["id", "text"],
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
            description="建立 YouTube job 時取得的短效翻譯權杖",
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
            raise HTTPException(status_code=403, detail="request_id 與 YouTube job 不相符")

        upstream_url = f"{TRANSLATE_API_BASE}/api/v1/subtitles/translate"
        timeout = aiohttp.ClientTimeout(total=TRANSLATE_API_TIMEOUT_SECONDS)
        started_at = time.perf_counter()
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
                        "request_bytes=%d response_bytes=%d elapsed=%.3fs",
                        authorized_job_id,
                        upstream_response.status,
                        len(body),
                        len(response_body),
                        time.perf_counter() - started_at,
                    )
                    return Response(
                        content=response_body,
                        status_code=upstream_response.status,
                        headers={"Content-Type": response_content_type},
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

    @router.get("/api/youtube-live/jobs/{job_id}/events", tags=["YouTube Live"])
    async def youtube_live_events(job_id: str):
        cleanup_youtube_live_jobs(jobs)
        job = jobs.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="找不到這個 YouTube 轉譯請求")

        async def event_stream():
            yield sse_message("status", {"message": "已連線，等待處理結果"})
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
