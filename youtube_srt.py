import html
import json
import logging
import math
import os
import re
import tempfile
import time
from pathlib import Path
from typing import Any, Optional
from urllib.parse import parse_qs, urlparse

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from app_logging import log_structured_event, set_request_log_metadata
from media_audio_stream import MediaAudioSource
from text_converter import to_traditional_chinese


logger = logging.getLogger(__name__)


class YtDlpLogger:
    def debug(self, message: str) -> None:
        if message.startswith("[debug] "):
            logger.debug(message)
        else:
            logger.info(message)

    def info(self, message: str) -> None:
        logger.info(message)

    def warning(self, message: str) -> None:
        logger.warning(message)

    def error(self, message: str) -> None:
        logger.error(message)


yt_dlp_logger = YtDlpLogger()


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
YOUTUBE_TRADITIONAL_SUBTITLE_LANGUAGE_PRIORITY = [
    "zh-TW",
    "zh-Hant",
    "zh-HK",
    "zh-CN",
    "zh-Hans",
    "zh",
]
YOUTUBE_SUBTITLE_EXT_PRIORITY = ["vtt", "srt", "json3", "srv3", "ttml"]
YOUTUBE_RATE_LIMIT_MESSAGE = "429 Client Error: Too Many Requests"
YOUTUBE_AUDIO_FORMAT = os.getenv(
    "YOUTUBE_AUDIO_FORMAT",
    "bestaudio[abr<=96]/bestaudio[abr<=128]/bestaudio",
).strip() or "bestaudio[abr<=96]/bestaudio[abr<=128]/bestaudio"


class YoutubeSrtRequest(BaseModel):
    url: str
    lrc_format: bool = False


class YoutubeSrtResponse(BaseModel):
    content: str
    lrc_format: bool
    title: Optional[str]
    duration: Optional[float]
    language: Optional[str]
    language_probability: Optional[float]
    source: str
    segments_count: Optional[int]
    total_elapsed_seconds: float = Field(
        description="從收到請求到產生結果的端到端總耗時（秒）。"
    )
    processing_speed_x: Optional[float] = Field(
        description="每秒可處理的影片秒數；例如 2.5 表示 2.5 倍即時速度。"
    )


def build_processing_metrics(
    duration: Any,
    started_at: float,
) -> dict[str, Optional[float]]:
    elapsed_seconds = max(0.0, time.perf_counter() - started_at)
    try:
        duration_seconds = float(duration)
    except (TypeError, ValueError):
        duration_seconds = 0.0

    has_duration = math.isfinite(duration_seconds) and duration_seconds > 0
    has_elapsed = math.isfinite(elapsed_seconds) and elapsed_seconds > 0
    return {
        "total_elapsed_seconds": round(elapsed_seconds, 3),
        "processing_speed_x": (
            round(duration_seconds / elapsed_seconds, 3)
            if has_duration and has_elapsed
            else None
        ),
    }


def log_processing_metrics(
    source: str,
    metrics: dict[str, Optional[float]],
) -> None:
    speed = metrics["processing_speed_x"]
    logger.info(
        "YouTube SRT request completed: source=%s total_elapsed=%.3fs "
        "processing_speed=%s",
        source,
        metrics["total_elapsed_seconds"] or 0.0,
        f"{speed:.3f}x" if speed is not None else "unknown",
    )


def is_youtube_rate_limit_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return "429" in text or "too many requests" in text


def raise_youtube_rate_limit_error(exc: Exception) -> None:
    raise HTTPException(status_code=429, detail=f"{YOUTUBE_RATE_LIMIT_MESSAGE} 原始錯誤: {exc}") from exc


def download_subtitle_content(url: str, retries: int = 2, timeout: int = 30) -> str:
    import requests

    last_error: Optional[Exception] = None
    for attempt in range(retries + 1):
        try:
            response = requests.get(url, timeout=timeout)
            response.raise_for_status()
            response.encoding = response.encoding or "utf-8"
            return response.text
        except requests.HTTPError as exc:
            last_error = exc
            status_code = exc.response.status_code if exc.response is not None else None
            if status_code == 429 and attempt < retries:
                time.sleep(2 * (attempt + 1))
                continue
            if status_code == 429:
                raise_youtube_rate_limit_error(exc)
            raise
        except Exception as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(1 * (attempt + 1))
                continue
            raise
    assert last_error is not None
    raise last_error


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


def normalize_language_code(language: Any) -> str:
    return str(language or "").strip().lower().replace("_", "-")


def matching_video_languages(
    video_info: dict[str, Any],
    tracks: dict[str, Any],
) -> list[str]:
    available = [language for language in tracks if language != "live_chat"]
    matches: list[str] = []
    preferred = [
        video_info.get("original_language"),
        video_info.get("language"),
    ]
    for preferred_language in preferred:
        normalized_preferred = normalize_language_code(preferred_language)
        if not normalized_preferred:
            continue
        preferred_base = normalized_preferred.split("-", 1)[0]
        for language in available:
            normalized_language = normalize_language_code(language)
            if (
                normalized_language == normalized_preferred
                or normalized_language.split("-", 1)[0] == preferred_base
            ) and language not in matches:
                matches.append(language)
    return matches


def ordered_subtitle_languages(
    video_info: dict[str, Any],
    tracks: dict[str, Any],
) -> list[str]:
    available = [language for language in tracks if language != "live_chat"]
    ordered = matching_video_languages(video_info, tracks)

    for language in YOUTUBE_SUBTITLE_LANGUAGE_PRIORITY:
        if language in tracks and language not in ordered:
            ordered.append(language)
    ordered.extend(language for language in available if language not in ordered)
    return ordered


def subtitle_candidate_is_translated(candidate: dict[str, Any]) -> bool:
    url = str(candidate.get("url") or "")
    return bool(parse_qs(urlparse(url).query).get("tlang"))


def select_subtitle_candidate(
    source: str,
    tracks: dict[str, Any],
    languages: list[str],
    untranslated_only: bool = False,
) -> Optional[dict[str, Any]]:
    for language in languages:
        candidates = tracks.get(language) or []
        for extension in YOUTUBE_SUBTITLE_EXT_PRIORITY:
            for candidate in candidates:
                if candidate.get("ext") != extension or not candidate.get("url"):
                    continue
                if untranslated_only and subtitle_candidate_is_translated(candidate):
                    continue
                return {
                    "source": source,
                    "language": language,
                    "extension": extension,
                    "url": candidate["url"],
                }
    return None


def choose_subtitle_track(video_info: dict[str, Any]) -> Optional[dict[str, Any]]:
    subtitles = video_info.get("subtitles") or {}
    preferred_subtitle_languages = matching_video_languages(video_info, subtitles)
    if preferred_subtitle_languages:
        preferred_subtitle = select_subtitle_candidate(
            "youtube subtitles",
            subtitles,
            preferred_subtitle_languages,
        )
        if preferred_subtitle is not None:
            return preferred_subtitle

    automatic_captions = video_info.get("automatic_captions") or {}
    automatic_languages = ordered_subtitle_languages(video_info, automatic_captions)
    original_automatic_caption = select_subtitle_candidate(
        "youtube automatic captions",
        automatic_captions,
        automatic_languages,
        untranslated_only=True,
    )
    if original_automatic_caption is not None:
        return original_automatic_caption

    subtitle = select_subtitle_candidate(
        "youtube subtitles",
        subtitles,
        ordered_subtitle_languages(video_info, subtitles),
    )
    if subtitle is not None:
        return subtitle

    translated_automatic_caption = select_subtitle_candidate(
        "youtube automatic captions",
        automatic_captions,
        automatic_languages,
    )
    if translated_automatic_caption is not None:
        return translated_automatic_caption
    return None


def choose_traditional_subtitle_track(
    video_info: dict[str, Any],
) -> Optional[dict[str, Any]]:
    for source, tracks in [
        ("youtube subtitles", video_info.get("subtitles") or {}),
        ("youtube automatic captions", video_info.get("automatic_captions") or {}),
    ]:
        languages = [
            language
            for language in YOUTUBE_TRADITIONAL_SUBTITLE_LANGUAGE_PRIORITY
            if language in tracks
        ]
        subtitle = select_subtitle_candidate(source, tracks, languages)
        if subtitle is not None:
            return subtitle
    return None


def subtitle_track_to_content(
    subtitle_track: dict[str, Any],
    lrc_format: bool,
) -> tuple[str, int]:
    subtitle_content = download_subtitle_content(subtitle_track["url"])
    segments = parse_subtitle_content(
        subtitle_content,
        subtitle_track["extension"],
    )
    if not segments:
        return "", 0
    content = (
        subtitle_segments_to_srt(segments)
        if lrc_format
        else subtitle_segments_to_text(segments)
    )
    return content, len(segments)


def get_youtube_video_info(url: str, cookies_file: Path) -> dict[str, Any]:
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
    if cookies_file.exists():
        options["cookiefile"] = str(cookies_file)

    with yt_dlp.YoutubeDL(options) as ydl:
        return ydl.extract_info(url, download=False)


def get_youtube_playlist_preview(
    url: str,
    cookies_file: Path,
    max_items: int = 50,
) -> dict[str, Any]:
    try:
        import yt_dlp
    except ImportError as exc:
        raise HTTPException(status_code=500, detail="缺少 yt-dlp，請先安裝 requirements.txt") from exc

    parsed = urlparse(url.strip())
    host = parsed.hostname.lower().removeprefix("www.") if parsed.hostname else ""
    if host != "youtube.com" and not host.endswith(".youtube.com"):
        raise HTTPException(status_code=400, detail="請輸入有效的 YouTube 播放清單網址")

    playlist_id = (parse_qs(parsed.query).get("list") or [""])[0].strip()
    if not re.fullmatch(r"[A-Za-z0-9_-]{2,100}", playlist_id):
        raise HTTPException(status_code=400, detail="網址中沒有有效的 YouTube 播放清單")

    item_limit = max(1, min(int(max_items), 100))
    playlist_url = f"https://www.youtube.com/playlist?list={playlist_id}"
    options = {
        "skip_download": True,
        "extract_flat": "in_playlist",
        "playlistend": item_limit,
        "lazy_playlist": False,
        "quiet": True,
        "no_warnings": True,
        "js_runtimes": {"node": {}},
    }
    if cookies_file.exists():
        options["cookiefile"] = str(cookies_file)

    with yt_dlp.YoutubeDL(options) as ydl:
        info = ydl.extract_info(playlist_url, download=False)

    items: list[dict[str, Any]] = []
    for entry in info.get("entries") or []:
        video_id = str(entry.get("id") or "").strip()
        if not re.fullmatch(r"[A-Za-z0-9_-]{11}", video_id):
            continue
        duration = entry.get("duration")
        try:
            duration = round(float(duration), 3) if duration is not None else None
        except (TypeError, ValueError):
            duration = None
        items.append(
            {
                "id": video_id,
                "title": str(entry.get("title") or video_id),
                "duration": duration,
                "thumbnail": f"https://i.ytimg.com/vi/{video_id}/mqdefault.jpg",
            }
        )

    total_items = info.get("playlist_count")
    try:
        total_items = int(total_items)
    except (TypeError, ValueError):
        total_items = len(items)
    total_items = max(total_items, len(items))
    return {
        "id": playlist_id,
        "title": str(info.get("title") or "YouTube Playlist"),
        "items": items,
        "total_items": total_items,
        "is_truncated": total_items > len(items),
    }


def try_get_youtube_subtitle_content(
    url: str,
    lrc_format: bool,
    cookies_file: Path,
) -> Optional[dict[str, Any]]:
    try:
        video_info = get_youtube_video_info(url, cookies_file)
        subtitle_track = choose_subtitle_track(video_info)
        if subtitle_track is None:
            return None

        source_content, segments_count = subtitle_track_to_content(
            subtitle_track,
            lrc_format,
        )
        if not source_content:
            return None

        content = ""
        if normalize_language_code(subtitle_track["language"]).startswith("zh"):
            content = to_traditional_chinese(
                source_content,
                subtitle_track["language"],
            )
        else:
            traditional_track = choose_traditional_subtitle_track(video_info)
            if traditional_track is not None:
                translated_content, _ = subtitle_track_to_content(
                    traditional_track,
                    lrc_format,
                )
                content = to_traditional_chinese(
                    translated_content,
                    traditional_track["language"],
                )

        if not content:
            content = source_content
        return {
            "content": content,
            "video_info": video_info,
            "language": subtitle_track["language"],
            "source": subtitle_track["source"],
            "segments_count": segments_count,
        }
    except HTTPException as exc:
        if exc.status_code == 429:
            raise
        logger.warning(
            "YouTube subtitle fast path failed, falling back to Whisper: %s",
            exc,
        )
        return None
    except Exception as exc:
        if is_youtube_rate_limit_error(exc):
            raise_youtube_rate_limit_error(exc)
        logger.warning(
            "YouTube subtitle fast path failed, falling back to Whisper: %s",
            exc,
        )
        return None


def download_youtube_audio(url: str, output_dir: str, cookies_file: Path):
    try:
        import yt_dlp
    except ImportError as exc:
        raise HTTPException(status_code=500, detail="缺少 yt-dlp，請先安裝 requirements.txt") from exc

    def build_options(player_clients: Optional[list[str]], attempt_dir: str):
        output_path = str(Path(attempt_dir) / "%(id)s.%(ext)s")
        options = {
            "format": YOUTUBE_AUDIO_FORMAT,
            "outtmpl": output_path,
            "noplaylist": True,
            "quiet": False,
            "no_warnings": False,
            "retries": 3,
            "fragment_retries": 3,
            "concurrent_fragment_downloads": 5,
            "js_runtimes": {"node": {}},
            "logger": yt_dlp_logger,
        }
        if player_clients:
            options["extractor_args"] = {"youtube": {"player_client": player_clients}}
        if cookies_file.exists():
            options["cookiefile"] = str(cookies_file)
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
                    candidate_paths: list[Path] = []
                    for download in info.get("requested_downloads") or []:
                        filepath = download.get("filepath")
                        if filepath:
                            candidate_paths.append(Path(filepath))
                    prepared_filename = ydl.prepare_filename(info)
                    if prepared_filename:
                        candidate_paths.append(Path(prepared_filename))

                ignored_suffixes = {
                    ".description",
                    ".json",
                    ".jpg",
                    ".jpeg",
                    ".part",
                    ".png",
                    ".srt",
                    ".vtt",
                    ".webp",
                    ".ytdl",
                }
                candidate_paths.extend(
                    path
                    for path in attempt_dir.iterdir()
                    if path.is_file() and path.suffix.lower() not in ignored_suffixes
                )
                audio_files = {
                    path.resolve()
                    for path in candidate_paths
                    if path.is_file() and path.stat().st_size > 0
                }
                if not audio_files:
                    raise RuntimeError("音軌下載完成後找不到媒體檔案")
                audio_path = max(audio_files, key=lambda path: path.stat().st_size)
                return audio_path, info
            except Exception as exc:
                errors.append(f"{attempt_url} / {player_clients or 'default'}: {str(exc)}")

    last_error = errors[-1] if errors else "未知錯誤"
    if errors:
        logger.error("yt-dlp download attempts failed")
        for error in errors:
            logger.error("yt-dlp attempt failed: %s", error)
    if any("429" in error or "Too Many Requests" in error for error in errors):
        raise HTTPException(status_code=429, detail=f"{YOUTUBE_RATE_LIMIT_MESSAGE} 最後錯誤: {last_error}")

    raise HTTPException(
        status_code=400,
        detail=(
            "下載或抽取 YouTube 音軌失敗。yt-dlp 已嘗試多組 YouTube player client。"
            "若仍是 403，請確認 cookies.txt 是否有效，或該影片在同一台機器的瀏覽器可播放。"
            f"最後錯誤: {last_error}"
        ),
    )


def get_youtube_audio_stream_source(
    url: str,
    cookies_file: Path,
) -> tuple[MediaAudioSource, dict[str, Any]]:
    try:
        import yt_dlp
    except ImportError as exc:
        raise HTTPException(status_code=500, detail="缺少 yt-dlp，請先安裝 requirements.txt") from exc

    errors: list[str] = []
    for attempt_url in youtube_url_attempts(url, True):
        for player_clients in YOUTUBE_PLAYER_CLIENT_ATTEMPTS:
            options: dict[str, Any] = {
                "format": YOUTUBE_AUDIO_FORMAT,
                "noplaylist": True,
                "quiet": True,
                "no_warnings": True,
                "retries": 3,
                "fragment_retries": 3,
                "js_runtimes": {"node": {}},
                "logger": yt_dlp_logger,
            }
            if player_clients:
                options["extractor_args"] = {
                    "youtube": {"player_client": player_clients}
                }
            if cookies_file.exists():
                options["cookiefile"] = str(cookies_file)

            try:
                with yt_dlp.YoutubeDL(options) as ydl:
                    info = ydl.extract_info(attempt_url, download=False)
                stream_url = str(info.get("url") or "").strip()
                if not stream_url:
                    for selected_format in info.get("requested_formats") or []:
                        if selected_format.get("vcodec") == "none" and selected_format.get("url"):
                            stream_url = str(selected_format["url"]).strip()
                            break
                if not stream_url:
                    raise RuntimeError("yt-dlp 沒有回傳可串流的音軌網址")

                headers = {
                    str(name): str(value)
                    for name, value in (info.get("http_headers") or {}).items()
                    if value is not None
                }
                source = MediaAudioSource(
                    location=stream_url,
                    headers=headers,
                    label=f"youtube-{info.get('id') or 'audio'}",
                )
                return source, info
            except Exception as exc:
                errors.append(
                    f"{attempt_url} / {player_clients or 'default'}: {str(exc)}"
                )

    last_error = errors[-1] if errors else "未知錯誤"
    if any("429" in error or "Too Many Requests" in error for error in errors):
        raise HTTPException(
            status_code=429,
            detail=f"{YOUTUBE_RATE_LIMIT_MESSAGE} 最後錯誤: {last_error}",
        )
    raise HTTPException(
        status_code=400,
        detail=f"無法建立 YouTube 音軌串流。最後錯誤: {last_error}",
    )


def create_youtube_router(auto2lrc, project_root: Path) -> APIRouter:
    router = APIRouter(tags=["YouTube"])
    cookies_file = project_root / "cookies.txt"

    @router.post("/youtube/srt/", response_model=YoutubeSrtResponse)
    def transcribe_youtube_to_srt(
        request: YoutubeSrtRequest,
        http_request: Request,
    ):
        """
        接受 YouTube URL，優先使用 YouTube 字幕，沒有字幕時下載音軌並轉錄。
        """
        started_at = time.perf_counter()
        set_request_log_metadata(
            http_request,
            operation="youtube_srt",
            job_status="running",
            output_format="srt" if request.lrc_format else "txt",
            source_language="auto",
        )
        subtitle_result = try_get_youtube_subtitle_content(
            request.url,
            request.lrc_format,
            cookies_file,
        )
        if subtitle_result is not None:
            video_info = subtitle_result["video_info"]
            metrics = build_processing_metrics(video_info.get("duration"), started_at)
            log_processing_metrics(subtitle_result["source"], metrics)
            set_request_log_metadata(
                http_request,
                job_status="done",
                source_language=subtitle_result["language"],
                subtitle_source=subtitle_result["source"],
                segments=subtitle_result["segments_count"],
            )
            log_structured_event(
                "youtube_srt_completed",
                job_status="done",
                output_format="srt" if request.lrc_format else "txt",
                source_language=subtitle_result["language"],
                subtitle_source=subtitle_result["source"],
                segments=subtitle_result["segments_count"],
                audio_duration_seconds=video_info.get("duration"),
                total_elapsed_seconds=metrics["total_elapsed_seconds"],
                processing_speed_x=metrics["processing_speed_x"],
            )
            return {
                "content": subtitle_result["content"],
                "lrc_format": request.lrc_format,
                "title": video_info.get("title"),
                "duration": video_info.get("duration"),
                "language": subtitle_result["language"],
                "language_probability": None,
                "source": subtitle_result["source"],
                "segments_count": subtitle_result["segments_count"],
                **metrics,
            }

        with tempfile.TemporaryDirectory(prefix="yt_srt_") as temp_dir:
            audio_path, video_info = download_youtube_audio(request.url, temp_dir, cookies_file)
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

                metrics = build_processing_metrics(video_info.get("duration"), started_at)
                log_processing_metrics("whisper", metrics)
                detected_language = getattr(transcript_info, "language", None)
                language_probability = getattr(
                    transcript_info,
                    "language_probability",
                    None,
                )
                set_request_log_metadata(
                    http_request,
                    job_status="done",
                    detected_language=detected_language,
                    language_probability=language_probability,
                    subtitle_source="whisper",
                )
                log_structured_event(
                    "youtube_srt_completed",
                    job_status="done",
                    output_format="srt" if request.lrc_format else "txt",
                    detected_language=detected_language,
                    language_probability=language_probability,
                    subtitle_source="whisper",
                    audio_duration_seconds=video_info.get("duration"),
                    total_elapsed_seconds=metrics["total_elapsed_seconds"],
                    processing_speed_x=metrics["processing_speed_x"],
                )
                return {
                    "content": content,
                    "lrc_format": request.lrc_format,
                    "title": video_info.get("title"),
                    "duration": video_info.get("duration"),
                    "language": detected_language,
                    "language_probability": language_probability,
                    "source": "whisper",
                    "segments_count": None,
                    **metrics,
                }

            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"生成 SRT 時出錯: {str(e)}") from e

    return router
