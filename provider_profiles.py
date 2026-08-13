from __future__ import annotations

import base64
import io
import os
import re
import time
import wave
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Literal, Optional, Protocol, cast

import numpy as np
import requests


ProcessingProfile = Literal["standard", "premium", "private"]
AsrProviderName = Literal["cloudflare", "groq", "local"]
TranslationType = Literal["standard", "premium", "private"]
REMOTE_ASR_PROVIDERS = frozenset({"cloudflare", "groq"})

PROFILE_ALIASES = {
    "standard": "standard",
    "std": "standard",
    "premium": "premium",
    "pro": "premium",
    "private": "private",
}


def normalize_processing_profile(value: Optional[str]) -> ProcessingProfile:
    profile = PROFILE_ALIASES.get(str(value or "standard").strip().lower())
    if profile is None:
        raise ValueError("不支援的處理方案")
    return cast(ProcessingProfile, profile)


def asr_provider_for_profile(profile: ProcessingProfile) -> AsrProviderName:
    del profile
    provider = os.getenv("AUDIOIO_ASR_PROVIDER", "local").strip().lower()
    if provider not in {"cloudflare", "groq", "local"}:
        raise ValueError(
            "AUDIOIO_ASR_PROVIDER 必須是 local、cloudflare 或 groq"
        )
    return cast(AsrProviderName, provider)


def translation_type_for_profile(profile: ProcessingProfile) -> TranslationType:
    return profile


def route_translation_workflow_payload(
    payload: dict[str, Any],
    operation: str,
    profile: ProcessingProfile,
) -> dict[str, Any]:
    """Apply provider routing only to operations that actually call a model."""
    routed = dict(payload)
    if operation == "translate-groups":
        routed["translation_type"] = translation_type_for_profile(profile)
    else:
        routed.pop("translation_type", None)
    return routed


def pcm_float32_to_wav(audio_samples: np.ndarray, sample_rate: int = 16000) -> bytes:
    samples = np.asarray(audio_samples, dtype=np.float32)
    pcm = np.clip(samples, -1.0, 1.0)
    pcm = (pcm * 32767.0).astype("<i2", copy=False)
    output = io.BytesIO()
    with wave.open(output, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm.tobytes())
    return output.getvalue()


def retry_after_seconds(response: requests.Response) -> float | None:
    value = response.headers.get("retry-after")
    if value:
        try:
            parsed = float(value)
        except ValueError:
            parsed = -1
        if parsed >= 0:
            return parsed
    try:
        body = response.json()
    except ValueError:
        return None
    error = body.get("error") if isinstance(body, dict) else None
    message = error.get("message") if isinstance(error, dict) else None
    if not isinstance(message, str) and isinstance(body, dict):
        errors = body.get("errors")
        if isinstance(errors, list):
            message = " ".join(
                str(item.get("message") or "")
                for item in errors
                if isinstance(item, dict)
            )
    if not isinstance(message, str):
        return None
    match = re.search(
        r"try again in\s+([0-9]+(?:\.[0-9]+)?)\s*(ms|s)\b",
        message,
        flags=re.IGNORECASE,
    )
    if match is None:
        return None
    seconds = float(match.group(1))
    return seconds / 1000 if match.group(2).lower() == "ms" else seconds


groq_retry_after_seconds = retry_after_seconds


class WhisperAsrClient(Protocol):
    provider_name: AsrProviderName
    model_name: str

    def transcribe(
        self,
        audio_samples: np.ndarray,
        *,
        language: str | None = None,
        beam_size: int = 5,
        vad_filter: bool = False,
        condition_on_previous_text: bool = False,
        hallucination_silence_threshold: float | None = None,
    ) -> tuple[list[SimpleNamespace], SimpleNamespace]: ...


@dataclass(frozen=True)
class GroqWhisperSettings:
    api_key: str
    base_url: str = "https://api.groq.com/openai/v1"
    model: str = "whisper-large-v3"
    timeout_seconds: float = 120.0
    max_retries: int = 2
    fallback_wait_seconds: float = 10.0
    max_wait_seconds: float = 30.0
    min_request_interval_seconds: float = 3.1


class GroqWhisperClient:
    """OpenAI-compatible ASR adapter with a faster-whisper-shaped result."""

    provider_name: AsrProviderName = "groq"

    def __init__(
        self,
        settings: GroqWhisperSettings,
        session: requests.Session | None = None,
    ) -> None:
        self.settings = settings
        self.session = session or requests.Session()
        self._last_request_started = 0.0

    @property
    def model_name(self) -> str:
        return self.settings.model

    def _wait_for_rate_slot(self) -> None:
        remaining = (
            self.settings.min_request_interval_seconds
            - (time.monotonic() - self._last_request_started)
        )
        if remaining > 0:
            time.sleep(remaining)
        self._last_request_started = time.monotonic()

    def transcribe(
        self,
        audio_samples: np.ndarray,
        *,
        language: str | None = None,
        beam_size: int = 5,
        vad_filter: bool = False,
        condition_on_previous_text: bool = False,
        hallucination_silence_threshold: float | None = None,
    ) -> tuple[list[SimpleNamespace], SimpleNamespace]:
        del beam_size, vad_filter, condition_on_previous_text
        del hallucination_silence_threshold
        if not self.settings.api_key:
            raise RuntimeError("Standard/Premium 尚未設定 GROQ_API_KEY")
        payload: list[tuple[str, str]] = [
            ("model", self.settings.model),
            ("response_format", "verbose_json"),
            ("temperature", "0"),
            ("timestamp_granularities[]", "segment"),
            ("timestamp_granularities[]", "word"),
        ]
        normalized_language = normalize_asr_language(language)
        if normalized_language:
            payload.append(("language", normalized_language))
        wav_bytes = pcm_float32_to_wav(audio_samples)
        response: requests.Response | None = None
        for attempt in range(self.settings.max_retries + 1):
            self._wait_for_rate_slot()
            response = self.session.post(
                f"{self.settings.base_url.rstrip('/')}/audio/transcriptions",
                headers={"Authorization": f"Bearer {self.settings.api_key}"},
                data=payload,
                files={"file": ("audio.wav", wav_bytes, "audio/wav")},
                timeout=self.settings.timeout_seconds,
            )
            if response.status_code != 429:
                response.raise_for_status()
                break
            if attempt >= self.settings.max_retries:
                response.raise_for_status()
            wait_seconds = retry_after_seconds(response)
            if wait_seconds is None:
                wait_seconds = self.settings.fallback_wait_seconds
            if wait_seconds > self.settings.max_wait_seconds:
                response.raise_for_status()
            time.sleep(wait_seconds)
        if response is None:  # pragma: no cover
            raise AssertionError("Groq ASR retry loop did not issue a request")
        data = response.json()
        if not isinstance(data, dict):
            raise RuntimeError("Groq Whisper 回傳格式不正確")
        return groq_response_to_segments(data, normalized_language)


@dataclass(frozen=True)
class CloudflareWhisperSettings:
    account_id: str
    api_token: str
    base_url: str = "https://api.cloudflare.com/client/v4"
    model: str = "@cf/openai/whisper-large-v3-turbo"
    timeout_seconds: float = 120.0
    max_retries: int = 2
    fallback_wait_seconds: float = 5.0
    max_wait_seconds: float = 30.0


class CloudflareWhisperClient:
    """Cloudflare Workers AI adapter with a faster-whisper-shaped result."""

    provider_name: AsrProviderName = "cloudflare"

    def __init__(
        self,
        settings: CloudflareWhisperSettings,
        session: requests.Session | None = None,
    ) -> None:
        self.settings = settings
        self.session = session or requests.Session()

    @property
    def model_name(self) -> str:
        return self.settings.model

    def transcribe(
        self,
        audio_samples: np.ndarray,
        *,
        language: str | None = None,
        beam_size: int = 5,
        vad_filter: bool = False,
        condition_on_previous_text: bool = False,
        hallucination_silence_threshold: float | None = None,
    ) -> tuple[list[SimpleNamespace], SimpleNamespace]:
        if not self.settings.account_id or not self.settings.api_token:
            raise RuntimeError(
                "Cloudflare ASR 尚未設定 CLOUDFLARE_ACCOUNT_ID 或 "
                "CLOUDFLARE_API_TOKEN"
            )
        payload: dict[str, Any] = {
            "audio": base64.b64encode(pcm_float32_to_wav(audio_samples)).decode(
                "ascii"
            ),
            "task": "transcribe",
            "beam_size": max(1, int(beam_size)),
            "vad_filter": bool(vad_filter),
            "condition_on_previous_text": bool(condition_on_previous_text),
        }
        normalized_language = normalize_asr_language(language)
        if normalized_language:
            payload["language"] = normalized_language
        if hallucination_silence_threshold is not None:
            payload["hallucination_silence_threshold"] = float(
                hallucination_silence_threshold
            )

        endpoint = (
            f"{self.settings.base_url.rstrip('/')}/accounts/"
            f"{self.settings.account_id}/ai/run/{self.settings.model}"
        )
        response: requests.Response | None = None
        for attempt in range(self.settings.max_retries + 1):
            response = self.session.post(
                endpoint,
                headers={"Authorization": f"Bearer {self.settings.api_token}"},
                json=payload,
                timeout=self.settings.timeout_seconds,
            )
            if response.status_code != 429:
                response.raise_for_status()
                break
            if attempt >= self.settings.max_retries:
                response.raise_for_status()
            wait_seconds = retry_after_seconds(response)
            if wait_seconds is None:
                wait_seconds = self.settings.fallback_wait_seconds
            if wait_seconds > self.settings.max_wait_seconds:
                response.raise_for_status()
            time.sleep(wait_seconds)

        if response is None:  # pragma: no cover
            raise AssertionError("Cloudflare ASR retry loop did not issue a request")
        envelope = response.json()
        if not isinstance(envelope, dict):
            raise RuntimeError("Cloudflare Workers AI 回傳格式不正確")
        if envelope.get("success") is False:
            errors = envelope.get("errors")
            raise RuntimeError(
                f"Cloudflare Workers AI 請求失敗: {errors or 'unknown error'}"
            )
        result = envelope.get("result")
        if not isinstance(result, dict):
            raise RuntimeError("Cloudflare Workers AI 缺少 result")
        return cloudflare_response_to_segments(result, normalized_language)


def normalize_asr_language(language: str | None) -> str | None:
    value = str(language or "").strip().lower()
    if not value or value == "auto":
        return None
    if value.startswith("zh"):
        return "zh"
    return value.split("-", 1)[0]


def normalize_detected_language(language: object, fallback: str | None) -> str | None:
    value = str(language or fallback or "").strip().lower()
    names = {
        "english": "en",
        "japanese": "ja",
        "korean": "ko",
        "thai": "th",
        "chinese": "zh",
    }
    return names.get(value, value or None)


def detokenize_word_texts(raw_words: list[dict[str, Any]]) -> list[str]:
    """Create joinable word pieces when Groq text alignment is unavailable."""
    pieces: list[str] = []
    closing_punctuation = set(".,!?;:%)]}〉》」』】、。，！？：；%")
    opening_punctuation = set("([{〈《「『【")

    def is_cjk(character: str) -> bool:
        codepoint = ord(character)
        return (
            0x3040 <= codepoint <= 0x30FF
            or 0x3400 <= codepoint <= 0x9FFF
            or 0xAC00 <= codepoint <= 0xD7AF
        )

    for raw_word in raw_words:
        token = str(raw_word.get("word") or "").strip()
        if not token:
            continue
        if not pieces:
            pieces.append(token)
            continue
        previous = pieces[-1]
        needs_space = (
            token[0] not in closing_punctuation
            and previous[-1] not in opening_punctuation
            and not token.startswith(("'", "’"))
            and not (is_cjk(previous[-1]) and is_cjk(token[0]))
        )
        pieces.append((" " if needs_space else "") + token)
    return pieces


def align_word_texts(segment_text: str, raw_words: list[dict[str, Any]]) -> list[str]:
    """Preserve the whitespace and punctuation omitted by Groq word tokens."""
    cursor = 0
    aligned: list[str] = []
    lowered_text = segment_text.lower()
    for raw_word in raw_words:
        token = str(raw_word.get("word") or "").strip()
        if not token:
            return detokenize_word_texts(raw_words)
        match_start = segment_text.find(token, cursor)
        if match_start < 0:
            match_start = lowered_text.find(token.lower(), cursor)
        if match_start < 0:
            return detokenize_word_texts(raw_words)
        match_end = match_start + len(token)
        aligned.append(segment_text[cursor:match_end])
        cursor = match_end
    if aligned and cursor < len(segment_text):
        aligned[-1] += segment_text[cursor:]
    return aligned


def whisper_response_to_segments(
    data: dict[str, Any],
    language_hint: str | None,
) -> tuple[list[SimpleNamespace], SimpleNamespace]:
    raw_segments = data.get("segments")
    if not isinstance(raw_segments, list):
        text = data.get("text")
        duration = data.get("duration")
        if isinstance(text, str) and text.strip():
            raw_segments = [
                {
                    "start": 0.0,
                    "end": float(duration or 0.0),
                    "text": text,
                }
            ]
        else:
            raw_segments = []
    top_words = data.get("words")
    top_words = top_words if isinstance(top_words, list) else []
    segments: list[SimpleNamespace] = []
    for raw in raw_segments:
        if not isinstance(raw, dict):
            continue
        text = str(raw.get("text") or "").strip()
        if not text:
            continue
        start = max(0.0, float(raw.get("start") or 0.0))
        end = max(start, float(raw.get("end") or start))
        raw_words = raw.get("words")
        if not isinstance(raw_words, list):
            raw_words = [
                word
                for word in top_words
                if isinstance(word, dict)
                and float(word.get("start") or 0.0) < end + 0.001
                and float(word.get("end") or 0.0) > start - 0.001
            ]
        usable_words = [
            word
            for word in raw_words
            if isinstance(word, dict) and str(word.get("word") or "").strip()
        ]
        aligned_word_texts = align_word_texts(text, usable_words)
        words = [
            SimpleNamespace(
                word=aligned_word_texts[word_index],
                start=float(word.get("start") or start),
                end=float(word.get("end") or end),
                probability=(
                    float(word["probability"])
                    if isinstance(word.get("probability"), (int, float))
                    else None
                ),
            )
            for word_index, word in enumerate(usable_words)
        ]
        segments.append(
            SimpleNamespace(
                text=text,
                start=start,
                end=end,
                words=words,
                avg_logprob=float(raw.get("avg_logprob") or 0.0),
                no_speech_prob=float(raw.get("no_speech_prob") or 0.0),
            )
        )
    detected_language = normalize_detected_language(
        data.get("language"), language_hint
    )
    probability = data.get("language_probability")
    probability = float(probability) if isinstance(probability, (int, float)) else None
    return segments, SimpleNamespace(
        language=detected_language,
        language_probability=probability,
    )


def groq_response_to_segments(
    data: dict[str, Any],
    language_hint: str | None,
) -> tuple[list[SimpleNamespace], SimpleNamespace]:
    return whisper_response_to_segments(data, language_hint)


def cloudflare_response_to_segments(
    data: dict[str, Any],
    language_hint: str | None,
) -> tuple[list[SimpleNamespace], SimpleNamespace]:
    normalized = dict(data)
    transcription_info = data.get("transcription_info")
    if isinstance(transcription_info, dict):
        for key in ("duration", "language", "language_probability"):
            if normalized.get(key) is None and transcription_info.get(key) is not None:
                normalized[key] = transcription_info[key]
    return whisper_response_to_segments(normalized, language_hint)
