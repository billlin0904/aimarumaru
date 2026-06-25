import asyncio
import json
import secrets
import tempfile
import time
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

from youtube_srt import (
    choose_subtitle_track,
    download_youtube_audio,
    get_youtube_video_info,
    parse_subtitle_content,
)


YOUTUBE_LIVE_JOB_TTL_SECONDS = 3600
YOUTUBE_LIVE_EVENT_TIMEOUT_SECONDS = 30


class YoutubeLiveRequest(BaseModel):
    url: str
    language: str = ""


def sse_message(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def segment_payload(index: int, start: Optional[float], end: Optional[float], text: str) -> dict[str, Any]:
    return {
        "index": index,
        "start": start,
        "end": end,
        "text": text.strip(),
    }


def cleanup_youtube_live_jobs(jobs: dict[str, dict[str, Any]]) -> None:
    now = time.time()
    expired_ids = [
        job_id
        for job_id, job in jobs.items()
        if job.get("expires_at", 0) <= now and job.get("status") not in {"running", "queued"}
    ]
    for job_id in expired_ids:
        jobs.pop(job_id, None)


def put_thread_event(loop: asyncio.AbstractEventLoop, queue: asyncio.Queue[dict[str, Any]], event: dict[str, Any]) -> None:
    asyncio.run_coroutine_threadsafe(queue.put(event), loop).result()


def transcribe_audio_stream(
    auto2lrc,
    audio_path: Path,
    language_hint: Optional[str],
    loop: asyncio.AbstractEventLoop,
    queue: asyncio.Queue[dict[str, Any]],
) -> dict[str, Any]:
    segments, info = auto2lrc.get_model().transcribe(
        str(audio_path),
        beam_size=5,
        language=language_hint,
    )
    segment_count = 0
    try:
        for segment_count, segment in enumerate(segments, start=1):
            text = segment.text.strip()
            if not text:
                continue
            put_thread_event(
                loop,
                queue,
                {
                    "event": "segment",
                    "data": segment_payload(segment_count, segment.start, segment.end, text),
                },
            )
        return {
            "language": getattr(info, "language", None),
            "language_probability": getattr(info, "language_probability", None),
            "segments_count": segment_count,
        }
    finally:
        auto2lrc.clear_model_cache()


def create_youtube_live_router(auto2lrc, project_root: Path) -> APIRouter:
    router = APIRouter()
    page_path = project_root / "pages" / "youtube_live.html"
    cookies_file = project_root / "cookies.txt"
    jobs: dict[str, dict[str, Any]] = {}
    transcribe_lock = asyncio.Lock()

    async def push_event(job: dict[str, Any], event: str, data: dict[str, Any]) -> None:
        await job["queue"].put({"event": event, "data": data})

    async def run_job(job_id: str) -> None:
        job = jobs.get(job_id)
        if not job:
            return

        queue: asyncio.Queue[dict[str, Any]] = job["queue"]
        try:
            job["status"] = "running"
            await push_event(job, "status", {"message": "檢查 YouTube 字幕中"})

            video_info = await asyncio.to_thread(get_youtube_video_info, job["url"], cookies_file)
            await push_event(
                job,
                "metadata",
                {
                    "title": video_info.get("title"),
                    "duration": video_info.get("duration"),
                    "webpage_url": video_info.get("webpage_url"),
                },
            )

            subtitle_track = choose_subtitle_track(video_info)
            if subtitle_track is not None:
                import requests

                await push_event(
                    job,
                    "status",
                    {
                        "message": "找到內建字幕，正在解析",
                        "source": subtitle_track["source"],
                        "language": subtitle_track["language"],
                    },
                )
                response = await asyncio.to_thread(requests.get, subtitle_track["url"], timeout=30)
                response.raise_for_status()
                response.encoding = response.encoding or "utf-8"
                segments = parse_subtitle_content(response.text, subtitle_track["extension"])
                for index, segment in enumerate(segments, start=1):
                    text = segment["text"].strip()
                    if not text:
                        continue
                    await push_event(
                        job,
                        "segment",
                        segment_payload(index, segment["start"], segment["end"], text),
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
                    },
                )
                return

            await push_event(job, "status", {"message": "沒有可用字幕，正在下載音訊"})
            with tempfile.TemporaryDirectory(prefix="yt_live_") as temp_dir:
                audio_path, video_info = await asyncio.to_thread(download_youtube_audio, job["url"], temp_dir, cookies_file)
                await push_event(job, "status", {"message": "音訊下載完成，開始逐段轉譯"})
                async with transcribe_lock:
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
            await push_event(job, "failed", {"message": str(exc)})
        finally:
            job["expires_at"] = time.time() + YOUTUBE_LIVE_JOB_TTL_SECONDS
            await queue.put({"event": "close", "data": {}})

    @router.get("/youtube-live", include_in_schema=False)
    def youtube_live_page():
        return HTMLResponse(page_path.read_text(encoding="utf-8"))

    @router.post("/api/youtube-live/jobs", tags=["YouTube Live"])
    async def create_youtube_live_job(request: YoutubeLiveRequest):
        cleanup_youtube_live_jobs(jobs)
        url = request.url.strip()
        if not url:
            raise HTTPException(status_code=400, detail="請輸入 YouTube 網址")

        job_id = secrets.token_urlsafe(18)
        jobs[job_id] = {
            "url": url,
            "language_hint": request.language.strip() or None,
            "status": "queued",
            "queue": asyncio.Queue(),
            "created_at": time.time(),
            "expires_at": time.time() + YOUTUBE_LIVE_JOB_TTL_SECONDS,
        }
        asyncio.create_task(run_job(job_id))
        return {
            "job_id": job_id,
            "events_url": f"/api/youtube-live/jobs/{job_id}/events",
            "status": "queued",
        }

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
