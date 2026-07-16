import multiprocessing

from fastapi import FastAPI, HTTPException, File, Form, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse, Response
import asyncio
import base64
from datetime import datetime, timezone
from io import BytesIO
import json
import os
import re
import secrets
import shutil
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel
from starlette.middleware.gzip import GZipMiddleware
import uvicorn

from auto2lrc import Auto2Lrc
from transcribe_queue import (
    enqueue_transcribe_task,
    get_transcribe_queue_size,
    is_transcribe_queue_started,
    register_transcribe_cleanup,
    register_transcribe_handler,
    start_transcribe_queue,
    stop_transcribe_queue,
)
from youtube_live import create_youtube_live_router
from youtube_srt import create_youtube_router

app = FastAPI(
    title="Transcribe API",
    description="Upload audio or video and generate TXT, SRT, LRC, or JSON transcription output.",
    version="0.4.0",
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
TRANSCRIBE_JOB_DIR = PROJECT_ROOT / "transcribe_jobs"
TRANSCRIBE_JOB_TTL_SECONDS = 3600
CAPTCHA_TTL_SECONDS = 300
CAPTCHA_ENABLED = False # os.getenv("AUDIOIO_CAPTCHA_ENABLED", "true").strip().lower() not in {"0", "false", "no", "off"}
TRANSCRIBE_OUTPUT_FORMATS = {"txt", "srt", "lrc", "json"}
TRANSCRIBE_VIDEO_EXTENSIONS = {
    ".avi",
    ".flv",
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
TRANSCRIBE_MEDIA_TYPES = {
    "txt": "text/plain; charset=utf-8",
    "srt": "application/x-subrip; charset=utf-8",
    "lrc": "text/plain; charset=utf-8",
    "json": "application/json; charset=utf-8",
}
transcribe_jobs: dict[str, dict[str, Any]] = {}
captcha_challenges: dict[str, dict[str, Any]] = {}
captcha_verifications: dict[str, dict[str, Any]] = {}
app.include_router(create_youtube_router(auto2lrc, PROJECT_ROOT))
app.include_router(create_youtube_live_router(auto2lrc, PROJECT_ROOT, lambda token: verify_captcha_token_or_raise(token)))


class CaptchaVerifyRequest(BaseModel):
    captcha_id: str
    captcha_answer: str


@app.on_event("startup")
async def start_background_transcribe_queue():
    TRANSCRIBE_JOB_DIR.mkdir(parents=True, exist_ok=True)
    cleanup_stale_transcribe_job_dirs()
    await start_transcribe_queue()


@app.on_event("shutdown")
async def stop_background_transcribe_queue():
    await stop_transcribe_queue()


@app.get("/", include_in_schema=False)
def home_page():
    index_path = PROJECT_ROOT / "pages" / "index.html"
    return HTMLResponse(index_path.read_text(encoding="utf-8"))


def serve_page(filename: str) -> HTMLResponse:
    page_path = PROJECT_ROOT / "pages" / filename
    if not page_path.exists():
        raise HTTPException(status_code=404, detail="找不到頁面")
    return HTMLResponse(page_path.read_text(encoding="utf-8"))


@app.get("/about", include_in_schema=False)
def about_page():
    return serve_page("about.html")


@app.get("/privacy", include_in_schema=False)
def privacy_page():
    return serve_page("privacy.html")


@app.get("/terms", include_in_schema=False)
def terms_page():
    return serve_page("terms.html")


@app.get("/contact", include_in_schema=False)
def contact_page():
    return serve_page("contact.html")


@app.get("/favicon.svg", include_in_schema=False)
def favicon():
    icon_path = PROJECT_ROOT / "pages" / "favicon.svg"
    return Response(content=icon_path.read_text(encoding="utf-8"), media_type="image/svg+xml")


@app.get("/legal_i18n.js", include_in_schema=False)
def legal_i18n_script():
    script_path = PROJECT_ROOT / "pages" / "legal_i18n.js"
    return Response(content=script_path.read_text(encoding="utf-8"), media_type="application/javascript")


@app.get("/swagger", include_in_schema=False)
def swagger_ui():
    return RedirectResponse(url="/docs")


@app.get("/api/public-config", tags=["Config"])
def get_public_config():
    return {
        "captcha_enabled": CAPTCHA_ENABLED,
    }


def cleanup_expired_captchas() -> None:
    now = time.time()
    expired_challenge_ids = [
        captcha_id
        for captcha_id, challenge in captcha_challenges.items()
        if challenge["expires_at"] <= now
    ]
    expired_verification_ids = [
        token
        for token, verification in captcha_verifications.items()
        if verification["expires_at"] <= now
    ]
    for captcha_id in expired_challenge_ids:
        captcha_challenges.pop(captcha_id, None)
    for token in expired_verification_ids:
        captcha_verifications.pop(token, None)


def issue_captcha_token(captcha_id: str, captcha_answer: str) -> str:
    if not CAPTCHA_ENABLED:
        return "captcha-disabled"
    cleanup_expired_captchas()
    captcha_id = captcha_id.strip()
    captcha_answer = captcha_answer.strip().lower()
    if not captcha_id or not captcha_answer:
        raise HTTPException(status_code=400, detail="請先完成驗證碼")

    challenge = captcha_challenges.pop(captcha_id, None)
    if not challenge:
        raise HTTPException(status_code=400, detail="驗證碼已過期，請重新取得")
    if captcha_answer != challenge["answer"]:
        raise HTTPException(status_code=400, detail="驗證碼錯誤，請重新輸入")

    token = uuid.uuid4().hex
    captcha_verifications[token] = {
        "expires_at": time.time() + CAPTCHA_TTL_SECONDS,
    }
    return token


def verify_captcha_token_or_raise(captcha_token: str) -> None:
    if not CAPTCHA_ENABLED:
        return
    cleanup_expired_captchas()
    captcha_token = captcha_token.strip()
    if not captcha_token:
        raise HTTPException(status_code=400, detail="請先完成驗證碼")
    if captcha_verifications.pop(captcha_token, None) is None:
        raise HTTPException(status_code=400, detail="驗證已過期，請重新驗證")


@app.get("/api/captcha", tags=["Captcha"])
def create_captcha():
    """
    建立一次性圖片驗證碼。
    """
    try:
        from multicolorcaptcha import CaptchaGenerator
    except ImportError as exc:
        raise HTTPException(status_code=500, detail="缺少 multicolorcaptcha，請先安裝 requirements.txt") from exc

    cleanup_expired_captchas()
    generator = CaptchaGenerator(2)
    captcha = generator.gen_captcha_image(difficult_level=2)
    buffer = BytesIO()
    captcha.image.save(buffer, "PNG")

    captcha_id = uuid.uuid4().hex
    captcha_challenges[captcha_id] = {
        "answer": captcha.characters.lower(),
        "expires_at": time.time() + CAPTCHA_TTL_SECONDS,
    }
    image_base64 = base64.b64encode(buffer.getvalue()).decode("ascii")
    return {
        "captcha_id": captcha_id,
        "image": f"data:image/png;base64,{image_base64}",
        "expires_in": CAPTCHA_TTL_SECONDS,
    }


@app.post("/api/captcha/verify", tags=["Captcha"])
def verify_captcha(request: CaptchaVerifyRequest):
    """
    驗證圖片驗證碼，成功後回傳一次性 token。
    """
    token = issue_captcha_token(request.captcha_id, request.captcha_answer)
    return {
        "verified": True,
        "captcha_token": token,
        "expires_in": CAPTCHA_TTL_SECONDS,
    }


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


def build_transcribe_download_url(req_id: str, token: str) -> str:
    return f"/api/transcribe-audio/{req_id}/download?token={token}"


def remove_transcribe_job(req_id: str, job: Optional[dict[str, Any]] = None) -> None:
    job = job or transcribe_jobs.pop(req_id, None)
    if not job:
        return
    transcribe_jobs.pop(req_id, None)
    req_dir = job.get("req_dir")
    if isinstance(req_dir, Path):
        shutil.rmtree(req_dir, ignore_errors=True)


def cleanup_stale_transcribe_job_dirs() -> None:
    if not TRANSCRIBE_JOB_DIR.exists():
        return

    active_dirs = {
        job["req_dir"].resolve()
        for job in transcribe_jobs.values()
        if isinstance(job.get("req_dir"), Path)
    }
    now = time.time()
    for child in TRANSCRIBE_JOB_DIR.iterdir():
        if not child.is_dir():
            continue
        try:
            if child.resolve() in active_dirs:
                continue
            if now - child.stat().st_mtime >= TRANSCRIBE_JOB_TTL_SECONDS:
                shutil.rmtree(child, ignore_errors=True)
        except OSError:
            continue


def cleanup_expired_transcribe_jobs() -> None:
    now = time.time()
    expired_job_ids = [
        req_id
        for req_id, job in transcribe_jobs.items()
        if job["status"] != "running" and job["expires_at"] <= now
    ]
    for req_id in expired_job_ids:
        remove_transcribe_job(req_id)
    cleanup_stale_transcribe_job_dirs()


def transcribe_job_is_expired(req_id: str, job: dict[str, Any]) -> bool:
    if job["status"] == "running" or job["expires_at"] > time.time():
        return False
    remove_transcribe_job(req_id, job)
    return True


def write_transcription_output(
    input_path: Path,
    output_path: Path,
    output_format: str,
    source_filename: Optional[str],
    language_hint: Optional[str],
    use_vocal_separation: bool,
) -> None:
    if output_format == "lrc":
        auto2lrc.get_lrc(str(input_path), str(output_path), use_vocal_separation=use_vocal_separation)
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


def is_video_upload(filename: Optional[str], content_type: Optional[str]) -> bool:
    normalized_content_type = (content_type or "").lower().strip()
    suffix = Path(filename or "").suffix.lower()
    return normalized_content_type.startswith("video/") or suffix in TRANSCRIBE_VIDEO_EXTENSIONS


def extract_video_audio(video_path: Path, audio_path: Path) -> Path:
    ffmpeg_path = shutil.which("ffmpeg")
    if not ffmpeg_path:
        raise RuntimeError("伺服器未安裝 ffmpeg，無法從影片抽取音軌")

    process = subprocess.run(
        [
            ffmpeg_path,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(video_path),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            str(audio_path),
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if process.returncode != 0 or not audio_path.exists() or audio_path.stat().st_size == 0:
        detail = process.stderr.strip() or "影片沒有可用的音軌"
        raise RuntimeError(f"影片音軌抽取失敗: {detail}")
    return audio_path


def prepare_transcription_input(job: dict[str, Any]) -> Path:
    input_path = job["input_path"]
    if job.get("media_kind") != "video":
        return input_path

    extracted_audio_path = job["req_dir"] / "extracted_audio.wav"
    job["prepared_input_path"] = extract_video_audio(input_path, extracted_audio_path)
    return job["prepared_input_path"]


def run_transcribe_request(req_id: str) -> None:
    job = transcribe_jobs[req_id]
    transcription_input = prepare_transcription_input(job)
    write_transcription_output(
        input_path=transcription_input,
        output_path=job["output_path"],
        output_format=job["output_format"],
        source_filename=job["filename"],
        language_hint=job["language_hint"],
        use_vocal_separation=job.get("use_vocal_separation", False),
    )


async def handle_audio_transcribe_task(task: dict[str, Any]) -> None:
    req_id = str(task.get("id", ""))
    job = transcribe_jobs.get(req_id)
    if not job:
        return

    job["status"] = "running"
    job["started_at"] = utc_now_iso()
    job["updated_at"] = job["started_at"]
    try:
        await asyncio.to_thread(run_transcribe_request, req_id)
        job["status"] = "done"
        job["completed_at"] = utc_now_iso()
        job["updated_at"] = job["completed_at"]
        job["expires_at"] = time.time() + TRANSCRIBE_JOB_TTL_SECONDS
    except Exception as exc:
        job["status"] = "failed"
        job["error"] = str(exc)
        job["completed_at"] = utc_now_iso()
        job["updated_at"] = job["completed_at"]
        job["expires_at"] = time.time() + TRANSCRIBE_JOB_TTL_SECONDS


register_transcribe_handler("audio", handle_audio_transcribe_task)
register_transcribe_cleanup(cleanup_expired_transcribe_jobs)


def get_public_transcribe_request(req_id: str) -> dict[str, Any]:
    job = transcribe_jobs.get(req_id)
    if not job or transcribe_job_is_expired(req_id, job):
        raise HTTPException(status_code=404, detail="找不到這個轉譯請求")

    payload = {
        "req_id": req_id,
        "status": job["status"],
        "filename": job["filename"],
        "output_format": job["output_format"],
        "media_kind": job.get("media_kind", "audio"),
        "created_at": job["created_at"],
        "updated_at": job["updated_at"],
        "expires_at": datetime.fromtimestamp(job["expires_at"], timezone.utc).isoformat(),
        "status_url": f"/api/transcribe-audio/{req_id}",
    }
    if job["status"] == "done":
        payload["download_url"] = build_transcribe_download_url(req_id, job["download_token"])
    if job.get("started_at"):
        payload["started_at"] = job["started_at"]
    if job.get("completed_at"):
        payload["completed_at"] = job["completed_at"]
    if job.get("error"):
        payload["error"] = job["error"]
    if job["status"] == "queued" and is_transcribe_queue_started():
        payload["queue_size"] = get_transcribe_queue_size()
    return payload


@app.post("/api/transcribe-audio/", tags=["Audio"])
async def transcribe_audio_download(
    file: UploadFile = File(...),
    output_format: str = Form("txt"),
    language: str = Form(""),
    vocal_separation: bool = Form(False),
    captcha_token: str = Form(""),
):
    """
    接受上傳音訊或影片，建立轉譯請求並回傳 req_id。

    影片會先在背景佇列中抽取音軌，再使用與音訊相同的轉譯流程。
    """
    output_format = output_format.lower().strip()
    if output_format not in TRANSCRIBE_OUTPUT_FORMATS:
        raise HTTPException(status_code=400, detail="output_format 只支援 txt、srt、lrc、json")
    if not is_transcribe_queue_started():
        raise HTTPException(status_code=503, detail="轉譯佇列尚未啟動，請稍後再試")
    verify_captcha_token_or_raise(captcha_token)

    suffix = Path(file.filename or "audio").suffix or ".audio"
    media_kind = "video" if is_video_upload(file.filename, file.content_type) else "audio"
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
    download_token = secrets.token_urlsafe(32)
    transcribe_jobs[req_id] = {
        "status": "queued",
        "filename": file.filename or "audio",
        "content_type": file.content_type,
        "media_kind": media_kind,
        "output_format": output_format,
        "language_hint": language_hint,
        "use_vocal_separation": vocal_separation,
        "req_dir": req_dir,
        "input_path": input_path,
        "output_path": output_path,
        "download_name": download_name,
        "download_token": download_token,
        "media_type": TRANSCRIBE_MEDIA_TYPES[output_format],
        "created_at": created_at,
        "updated_at": created_at,
        "started_at": None,
        "completed_at": None,
        "expires_at": time.time() + TRANSCRIBE_JOB_TTL_SECONDS,
        "error": None,
    }

    try:
        enqueue_transcribe_task({"kind": "audio", "id": req_id})
    except asyncio.QueueFull:
        transcribe_jobs[req_id]["status"] = "failed"
        transcribe_jobs[req_id]["error"] = "轉譯佇列已滿，請稍後再試"
        remove_transcribe_job(req_id)
        raise HTTPException(status_code=503, detail="轉譯佇列已滿，請稍後再試")
    except RuntimeError as exc:
        transcribe_jobs[req_id]["status"] = "failed"
        transcribe_jobs[req_id]["error"] = "轉譯佇列尚未啟動，請稍後再試"
        remove_transcribe_job(req_id)
        raise HTTPException(status_code=503, detail="轉譯佇列尚未啟動，請稍後再試") from exc

    return get_public_transcribe_request(req_id)


@app.get("/api/transcribe-audio/{req_id}", tags=["Audio"])
def get_transcribe_audio_request(req_id: str):
    """
    依 req_id 查詢轉譯狀態。
    """
    return get_public_transcribe_request(req_id)


@app.get("/api/transcribe-audio/{req_id}/download", tags=["Audio"])
def download_transcribe_audio_request(req_id: str, token: str = ""):
    """
    下載已完成的轉譯結果。
    """
    job = transcribe_jobs.get(req_id)
    if not job or transcribe_job_is_expired(req_id, job):
        raise HTTPException(status_code=404, detail="找不到這個轉譯請求")
    if not token or not secrets.compare_digest(token, job["download_token"]):
        raise HTTPException(status_code=403, detail="下載連結無效或已過期")
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
