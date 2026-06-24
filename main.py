import multiprocessing

from fastapi import FastAPI, HTTPException, File, Form, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse, Response
import asyncio
from datetime import datetime, timezone
import html
import json
import os
import re
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Any, Optional
from urllib.parse import parse_qs, urlparse

from pydantic import BaseModel
from starlette.middleware.gzip import GZipMiddleware
import uvicorn

from auto2lrc import Auto2Lrc

app = FastAPI(
    title="Transcribe API",
    description="Upload audio, transcribe audio, and generate TXT, SRT, LRC, or JSON output.",
    version="0.2.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)
app.add_middleware(GZipMiddleware, minimum_size=1024, compresslevel=6)

ffmpeg_dir = r'C:\ProgramData\chocolatey\bin'
cuda_bin_dirs = [
    ffmpeg_dir,
    r"C:\ProgramData\Anaconda3\envs\aimarumaru\Lib\site-packages\nvidia\cudnn\bin",
    r"C:\ProgramData\Anaconda3\envs\aimarumaru\Lib\site-packages\nvidia\cublas\bin",
]
for bin_dir in cuda_bin_dirs:
    if os.path.isdir(bin_dir) and bin_dir not in os.environ["PATH"]:
        os.environ["PATH"] += os.pathsep + bin_dir

auto2lrc = Auto2Lrc()
PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_COOKIES_FILE = PROJECT_ROOT / "cookies.txt"
TRANSCRIBE_JOB_DIR = PROJECT_ROOT / "transcribe_jobs"
TRANSCRIBE_QUEUE_MAX_SIZE = 100
TRANSCRIBE_OUTPUT_FORMATS = {"txt", "srt", "lrc", "json"}
TRANSCRIBE_MEDIA_TYPES = {
    "txt": "text/plain; charset=utf-8",
    "srt": "application/x-subrip; charset=utf-8",
    "lrc": "text/plain; charset=utf-8",
    "json": "application/json; charset=utf-8",
}
transcribe_queue: Optional[asyncio.Queue[str]] = None
transcribe_jobs: dict[str, dict[str, Any]] = {}
transcribe_worker_task: Optional[asyncio.Task[Any]] = None
YOUTUBE_PLAYER_CLIENT_ATTEMPTS = [
    None,
    ["default", "-tv_simply"],
    ["default", "ios"],
    ["android"],
    ["web"],
    ["mweb"],
    ["tv_simply", "default"],
]
YOUTUBE_SUBTITLE_LANGUAGE_PRIORITY = [
    "zh-TW",
    "zh-Hant",
    "zh-Hans",
    "zh-CN",
    "zh",
    "en",
    "ja",
    "ko",
]
YOUTUBE_SUBTITLE_EXT_PRIORITY = ["vtt", "srt", "json3", "srv3", "ttml"]


class YoutubeSrtRequest(BaseModel):
    url: str
    lrc_format: bool = False


@app.on_event("startup")
async def start_transcribe_queue():
    global transcribe_queue, transcribe_worker_task
    TRANSCRIBE_JOB_DIR.mkdir(parents=True, exist_ok=True)
    transcribe_queue = asyncio.Queue(maxsize=TRANSCRIBE_QUEUE_MAX_SIZE)
    transcribe_worker_task = asyncio.create_task(transcribe_queue_worker())


@app.on_event("shutdown")
async def stop_transcribe_queue():
    if transcribe_worker_task:
        transcribe_worker_task.cancel()
        try:
            await transcribe_worker_task
        except asyncio.CancelledError:
            pass


@app.get("/", include_in_schema=False)
def home_page():
    index_path = PROJECT_ROOT / "pages" / "index.html"
    return HTMLResponse(index_path.read_text(encoding="utf-8"))


@app.get("/swagger", include_in_schema=False)
def swagger_ui():
    return RedirectResponse(url="/docs")


def extract_youtube_video_id(url: str) -> Optional[str]:
    parsed = urlparse(url)
    host = parsed.netloc.lower()
    if host.endswith("youtu.be"):
        return parsed.path.strip("/") or None
    if "youtube.com" in host:
        query_id = parse_qs(parsed.query).get("v", [None])[0]
        if query_id:
            return query_id
        if parsed.path.startswith("/shorts/"):
            return parsed.path.split("/", 2)[2].split("/")[0]
        if parsed.path.startswith("/embed/"):
            return parsed.path.split("/", 2)[2].split("/")[0]
    return None


def youtube_url_attempts(url: str, try_music_url: bool) -> list[str]:
    urls = [url]
    video_id = extract_youtube_video_id(url)
    if try_music_url and video_id:
        music_url = f"https://music.youtube.com/watch?v={video_id}"
        if music_url not in urls:
            urls.append(music_url)
    return urls


def parse_subtitle_time(time_text: str) -> float:
    time_text = time_text.replace(",", ".")
    parts = time_text.split(":")
    try:
        if len(parts) == 3:
            hours = int(parts[0])
            minutes = int(parts[1])
            seconds = float(parts[2])
            return hours * 3600 + minutes * 60 + seconds
        if len(parts) == 2:
            minutes = int(parts[0])
            seconds = float(parts[1])
            return minutes * 60 + seconds
    except ValueError:
        return 0.0
    return 0.0


def format_srt_time(seconds: float) -> str:
    millis = int(round(seconds * 1000))
    hours, remainder = divmod(millis, 3600000)
    minutes, remainder = divmod(remainder, 60000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"


def strip_subtitle_markup(text: str) -> str:
    text = re.sub(r"<\d{1,2}:\d{2}:\d{2}[.,]\d{3}>", "", text)
    text = re.sub(r"<[^>]+>", "", text)
    text = html.unescape(text)
    return re.sub(r"\s+", " ", text).strip()


def parse_vtt_subtitles(content: str) -> list[dict[str, Any]]:
    segments = []
    blocks = re.split(r"\n\s*\n", content.replace("\r\n", "\n"))
    for block in blocks:
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if not lines or lines[0].startswith(("WEBVTT", "Kind:", "Language:", "NOTE", "STYLE")):
            continue

        time_line_index = next((i for i, line in enumerate(lines) if "-->" in line), None)
        if time_line_index is None:
            continue

        time_line = lines[time_line_index]
        start_text, end_part = time_line.split("-->", 1)
        end_text = end_part.strip().split()[0]
        payload = " ".join(lines[time_line_index + 1 :])
        text = strip_subtitle_markup(payload)
        if not text:
            continue

        segments.append(
            {
                "start": parse_subtitle_time(start_text.strip()),
                "end": parse_subtitle_time(end_text.strip()),
                "text": text,
            }
        )
    return segments


def parse_srt_subtitles(content: str) -> list[dict[str, Any]]:
    segments = []
    blocks = re.split(r"\n\s*\n", content.replace("\r\n", "\n"))
    for block in blocks:
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if not lines:
            continue

        time_line_index = next((i for i, line in enumerate(lines) if "-->" in line), None)
        if time_line_index is None:
            continue

        start_text, end_part = lines[time_line_index].split("-->", 1)
        end_text = end_part.strip().split()[0]
        text = strip_subtitle_markup(" ".join(lines[time_line_index + 1 :]))
        if not text:
            continue

        segments.append(
            {
                "start": parse_subtitle_time(start_text.strip()),
                "end": parse_subtitle_time(end_text.strip()),
                "text": text,
            }
        )
    return segments


def parse_json3_subtitles(content: str) -> list[dict[str, Any]]:
    data = json.loads(content)
    segments = []
    for event in data.get("events", []):
        segs = event.get("segs")
        if not segs:
            continue

        text = strip_subtitle_markup("".join(seg.get("utf8", "") for seg in segs))
        if not text:
            continue

        start = (event.get("tStartMs") or 0) / 1000
        duration = (event.get("dDurationMs") or 0) / 1000
        segments.append({"start": start, "end": start + duration, "text": text})
    return segments


def subtitle_segments_to_text(segments: list[dict[str, Any]]) -> str:
    lines = []
    last_text = ""
    for segment in segments:
        text = segment["text"].strip()
        if text and text != last_text:
            lines.append(text)
            last_text = text
    return "\n".join(lines)


def subtitle_segments_to_srt(segments: list[dict[str, Any]]) -> str:
    blocks = []
    for index, segment in enumerate(segments, start=1):
        start = format_srt_time(float(segment["start"]))
        end = format_srt_time(float(segment["end"]))
        text = segment["text"].strip()
        if text:
            blocks.append(f"{index}\n{start} --> {end}\n{text}")
    return "\n\n".join(blocks) + ("\n" if blocks else "")


def parse_subtitle_content(content: str, extension: str) -> list[dict[str, Any]]:
    if extension == "json3":
        return parse_json3_subtitles(content)
    if extension == "srt":
        return parse_srt_subtitles(content)
    return parse_vtt_subtitles(content)


def choose_subtitle_track(video_info: dict[str, Any]) -> Optional[dict[str, Any]]:
    track_groups = [
        ("youtube subtitles", video_info.get("subtitles") or {}),
        ("youtube automatic captions", video_info.get("automatic_captions") or {}),
    ]

    for source, tracks in track_groups:
        languages = [lang for lang in YOUTUBE_SUBTITLE_LANGUAGE_PRIORITY if lang in tracks]
        languages.extend(lang for lang in tracks if lang not in languages and lang != "live_chat")

        for language in languages:
            candidates = tracks.get(language) or []
            for extension in YOUTUBE_SUBTITLE_EXT_PRIORITY:
                for candidate in candidates:
                    if candidate.get("ext") == extension and candidate.get("url"):
                        return {
                            "source": source,
                            "language": language,
                            "extension": extension,
                            "url": candidate["url"],
                        }
    return None


def get_youtube_video_info(url: str) -> dict[str, Any]:
    try:
        import yt_dlp
    except ImportError as exc:
        raise HTTPException(status_code=500, detail="缺少 yt-dlp，請先安裝 requirements.txt") from exc

    options = {
        "skip_download": True,
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "js_runtimes": {"node": {}},
    }
    if DEFAULT_COOKIES_FILE.exists():
        options["cookiefile"] = str(DEFAULT_COOKIES_FILE)

    with yt_dlp.YoutubeDL(options) as ydl:
        return ydl.extract_info(url, download=False)


def try_get_youtube_subtitle_content(url: str, lrc_format: bool) -> Optional[dict[str, Any]]:
    import requests

    try:
        video_info = get_youtube_video_info(url)
        subtitle_track = choose_subtitle_track(video_info)
        if subtitle_track is None:
            return None

        response = requests.get(subtitle_track["url"], timeout=30)
        response.raise_for_status()
        response.encoding = response.encoding or "utf-8"
        segments = parse_subtitle_content(response.text, subtitle_track["extension"])
        if not segments:
            return None

        content = subtitle_segments_to_srt(segments) if lrc_format else subtitle_segments_to_text(segments)
        return {
            "content": content,
            "video_info": video_info,
            "language": subtitle_track["language"],
            "source": subtitle_track["source"],
            "segments_count": len(segments),
        }
    except Exception as exc:
        print(f"YouTube subtitle fast path failed, falling back to Whisper: {exc}")
        return None


def download_youtube_audio(url: str, output_dir: str):
    try:
        import yt_dlp
    except ImportError as exc:
        raise HTTPException(status_code=500, detail="缺少 yt-dlp，請先安裝 requirements.txt") from exc

    def build_options(player_clients: Optional[list[str]], attempt_dir: str):
        output_path = str(Path(attempt_dir) / "%(id)s.%(ext)s")
        options = {
            "format": "bestaudio[ext=m4a]/bestaudio/best",
            "outtmpl": output_path,
            "noplaylist": True,
            "quiet": False,
            "no_warnings": False,
            "retries": 3,
            "fragment_retries": 3,
            "concurrent_fragment_downloads": 5,
            "js_runtimes": {"node": {}},
            "postprocessors": [
                {
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": "wav",
                }
            ],
        }
        if player_clients:
            options["extractor_args"] = {"youtube": {"player_client": player_clients}}
        if DEFAULT_COOKIES_FILE.exists():
            options["cookiefile"] = str(DEFAULT_COOKIES_FILE)
        options["verbose"] = True
        return options

    errors = []
    urls = youtube_url_attempts(url, True)
    attempt_index = 0
    for attempt_url in urls:
        for player_clients in YOUTUBE_PLAYER_CLIENT_ATTEMPTS:
            attempt_dir = Path(output_dir) / f"attempt_{attempt_index}"
            attempt_index += 1
            attempt_dir.mkdir(parents=True, exist_ok=True)
            options = build_options(player_clients, str(attempt_dir))
            try:
                with yt_dlp.YoutubeDL(options) as ydl:
                    info = ydl.extract_info(attempt_url, download=True)
                audio_files = list(attempt_dir.glob("*.wav"))
                if not audio_files:
                    raise RuntimeError("音軌轉換完成後找不到 WAV 檔案")
                return audio_files[0], info
            except Exception as exc:
                errors.append(f"{attempt_url} / {player_clients or 'default'}: {str(exc)}")

    last_error = errors[-1] if errors else "未知錯誤"
    if errors:
        print("yt-dlp download attempts failed:\n" + "\n".join(errors))

    raise HTTPException(
        status_code=400,
        detail=(
            "下載或抽取 YouTube 音軌失敗。yt-dlp 已嘗試多組 YouTube player client。"
            "若仍是 403，請確認 cookies.txt 是否有效，或該影片在同一台機器的瀏覽器可播放。"
            f"最後錯誤: {last_error}"
        ),
    )

@app.get("/transcribe/", tags=["Audio"])
def transcribe_audio_to_lrc(file_path: str):
    """
    接受音頻文件的路徑並返回轉錄的 LRC 文件內容字串
    """
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="音頻文件不存在")

    # 設定 LRC 文件的輸出路徑
    output_lrc_file = "output.lrc"

    try:
        # 進行轉錄並生成 LRC 文件
        auto2lrc.get_lrc(file_path, output_lrc_file)

        # 讀取 LRC 文件的內容
        with open(output_lrc_file, "r", encoding="utf-8") as f:
            lrc_content = f.read()

        # 立即刪除 LRC 文件
        os.remove(output_lrc_file)

        # 返回 LRC 文件的內容字串
        return {"lrc_content": lrc_content}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"處理音頻文件時出錯: {str(e)}")


@app.post("/youtube/srt/", tags=["YouTube"])
def transcribe_youtube_to_srt(request: YoutubeSrtRequest):
    """
    接受 YouTube URL，優先使用 YouTube 字幕，沒有字幕時下載音軌並轉錄。
    """
    subtitle_result = try_get_youtube_subtitle_content(request.url, request.lrc_format)
    if subtitle_result is not None:
        video_info = subtitle_result["video_info"]
        return {
            "content": subtitle_result["content"],
            "lrc_format": request.lrc_format,
            "title": video_info.get("title"),
            "duration": video_info.get("duration"),
            "language": subtitle_result["language"],
            "language_probability": None,
            "source": subtitle_result["source"],
            "segments_count": subtitle_result["segments_count"],
        }

    with tempfile.TemporaryDirectory(prefix="yt_srt_") as temp_dir:
        audio_path, video_info = download_youtube_audio(request.url, temp_dir)
        output_file = Path(temp_dir) / ("output.srt" if request.lrc_format else "output.txt")

        try:
            if request.lrc_format:
                transcript_info = auto2lrc.get_srt(
                    str(audio_path),
                    str(output_file),
                    language=None,
                    beam_size=5,
                )
            else:
                transcript_info = auto2lrc.get_text(
                    str(audio_path),
                    str(output_file),
                    language=None,
                    beam_size=5,
                )

            with open(output_file, "r", encoding="utf-8") as f:
                content = f.read()

            return {
                "content": content,
                "lrc_format": request.lrc_format,
                "title": video_info.get("title"),
                "duration": video_info.get("duration"),
                "language": getattr(transcript_info, "language", None),
                "language_probability": getattr(transcript_info, "language_probability", None),
                "source": "whisper",
                "segments_count": None,
            }

        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"生成 SRT 時出錯: {str(e)}")


@app.post("/transcribe_file/", tags=["Audio"])
async def transcribe_uploaded_audio(file: UploadFile = File(...)):
    """
    接受上傳的音頻文件，進行轉錄並返回 LRC 文件內容字串
    """
    try:
        # 保存上傳的文件到本地
        file_location = f"temp_{file.filename}"
        with open(file_location, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # 設定 LRC 文件的輸出路徑
        output_lrc_file = f"output_{os.path.splitext(file.filename)[0]}.lrc"

        # 進行轉錄並生成 LRC 文件
        auto2lrc.get_lrc(file_location, output_lrc_file)

        # 刪除暫存的音頻文件
        os.remove(file_location)

        # 讀取 LRC 文件的內容
        with open(output_lrc_file, "r", encoding="utf-8") as f:
            lrc_content = f.read()

        # 立即刪除 LRC 文件
        os.remove(output_lrc_file)

        # 返回 LRC 文件的內容字串
        return {"lrc_content": lrc_content}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"處理音頻文件時出錯: {str(e)}")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_transcript_stem(filename: Optional[str]) -> str:
    stem = Path(filename or "transcript").stem or "transcript"
    return re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._") or "transcript"


def write_transcription_output(
    input_path: Path,
    output_path: Path,
    output_format: str,
    source_filename: Optional[str],
    language_hint: Optional[str],
) -> None:
    if output_format == "lrc":
        auto2lrc.get_lrc(str(input_path), str(output_path))
    elif output_format == "srt":
        auto2lrc.get_srt(str(input_path), str(output_path), language=language_hint, beam_size=5)
    elif output_format == "txt":
        auto2lrc.get_text(str(input_path), str(output_path), language=language_hint, beam_size=5)
    else:
        segments, info = auto2lrc.get_model().transcribe(
            str(input_path),
            beam_size=5,
            language=language_hint,
        )
        payload = {
            "filename": source_filename,
            "language": getattr(info, "language", None),
            "language_probability": getattr(info, "language_probability", None),
            "segments": [
                {
                    "start": segment.start,
                    "end": segment.end,
                    "text": segment.text.strip(),
                }
                for segment in segments
            ],
        }
        output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        auto2lrc.clear_model_cache()


def run_transcribe_request(req_id: str) -> None:
    job = transcribe_jobs[req_id]
    write_transcription_output(
        input_path=job["input_path"],
        output_path=job["output_path"],
        output_format=job["output_format"],
        source_filename=job["filename"],
        language_hint=job["language_hint"],
    )


async def transcribe_queue_worker():
    while True:
        assert transcribe_queue is not None
        req_id = await transcribe_queue.get()
        job = transcribe_jobs.get(req_id)
        if not job:
            transcribe_queue.task_done()
            continue

        job["status"] = "running"
        job["started_at"] = utc_now_iso()
        job["updated_at"] = job["started_at"]
        try:
            await asyncio.to_thread(run_transcribe_request, req_id)
            job["status"] = "done"
            job["completed_at"] = utc_now_iso()
            job["updated_at"] = job["completed_at"]
        except Exception as exc:
            job["status"] = "failed"
            job["error"] = str(exc)
            job["completed_at"] = utc_now_iso()
            job["updated_at"] = job["completed_at"]
        finally:
            transcribe_queue.task_done()


def get_public_transcribe_request(req_id: str) -> dict[str, Any]:
    job = transcribe_jobs.get(req_id)
    if not job:
        raise HTTPException(status_code=404, detail="找不到這個轉譯請求")

    payload = {
        "req_id": req_id,
        "status": job["status"],
        "filename": job["filename"],
        "output_format": job["output_format"],
        "created_at": job["created_at"],
        "updated_at": job["updated_at"],
        "status_url": f"/api/transcribe-audio/{req_id}",
        "download_url": f"/api/transcribe-audio/{req_id}/download",
    }
    if job.get("started_at"):
        payload["started_at"] = job["started_at"]
    if job.get("completed_at"):
        payload["completed_at"] = job["completed_at"]
    if job.get("error"):
        payload["error"] = job["error"]
    if job["status"] == "queued" and transcribe_queue is not None:
        payload["queue_size"] = transcribe_queue.qsize()
    return payload


@app.post("/api/transcribe-audio/", tags=["Audio"])
async def transcribe_audio_download(
    file: UploadFile = File(...),
    output_format: str = Form("txt"),
    language: str = Form(""),
):
    """
    接受上傳音訊，建立轉譯請求並回傳 req_id。
    """
    output_format = output_format.lower().strip()
    if output_format not in TRANSCRIBE_OUTPUT_FORMATS:
        raise HTTPException(status_code=400, detail="output_format 只支援 txt、srt、lrc、json")
    if transcribe_queue is None:
        raise HTTPException(status_code=503, detail="轉譯佇列尚未啟動，請稍後再試")

    suffix = Path(file.filename or "audio").suffix or ".audio"
    safe_stem = safe_transcript_stem(file.filename)
    language_hint = language.strip() or None
    req_id = uuid.uuid4().hex
    req_dir = TRANSCRIBE_JOB_DIR / req_id
    req_dir.mkdir(parents=True, exist_ok=True)
    input_path = req_dir / f"input{suffix}"
    output_path = req_dir / f"{safe_stem}.{output_format}"
    download_name = f"{safe_stem}.{output_format}"

    try:
        with open(input_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as exc:
        shutil.rmtree(req_dir, ignore_errors=True)
        raise HTTPException(status_code=500, detail=f"儲存上傳檔案失敗: {exc}") from exc

    created_at = utc_now_iso()
    transcribe_jobs[req_id] = {
        "status": "queued",
        "filename": file.filename or "audio",
        "output_format": output_format,
        "language_hint": language_hint,
        "input_path": input_path,
        "output_path": output_path,
        "download_name": download_name,
        "media_type": TRANSCRIBE_MEDIA_TYPES[output_format],
        "created_at": created_at,
        "updated_at": created_at,
        "started_at": None,
        "completed_at": None,
        "error": None,
    }

    try:
        transcribe_queue.put_nowait(req_id)
    except asyncio.QueueFull:
        transcribe_jobs[req_id]["status"] = "failed"
        transcribe_jobs[req_id]["error"] = "轉譯佇列已滿，請稍後再試"
        shutil.rmtree(req_dir, ignore_errors=True)
        raise HTTPException(status_code=503, detail="轉譯佇列已滿，請稍後再試")

    return get_public_transcribe_request(req_id)


@app.get("/api/transcribe-audio/{req_id}", tags=["Audio"])
def get_transcribe_audio_request(req_id: str):
    """
    依 req_id 查詢轉譯狀態。
    """
    return get_public_transcribe_request(req_id)


@app.get("/api/transcribe-audio/{req_id}/download", tags=["Audio"])
def download_transcribe_audio_request(req_id: str):
    """
    下載已完成的轉譯結果。
    """
    job = transcribe_jobs.get(req_id)
    if not job:
        raise HTTPException(status_code=404, detail="找不到這個轉譯請求")
    if job["status"] != "done":
        raise HTTPException(status_code=409, detail="轉譯尚未完成")

    output_path = job["output_path"]
    if not output_path.exists():
        raise HTTPException(status_code=404, detail="找不到轉譯結果檔案")

    return Response(
        content=output_path.read_bytes(),
        media_type=job["media_type"],
        headers={"Content-Disposition": f'attachment; filename="{job["download_name"]}"'},
    )

# if __name__ == '__main__':
#     multiprocessing.freeze_support()  # For Windows support
#     uvicorn.run(app, host="127.0.0.1", port=8090, reload=False, workers=8)
