import io
import unittest
import wave

import numpy as np

from provider_profiles import (
    GroqWhisperClient,
    GroqWhisperSettings,
    asr_provider_for_profile,
    detokenize_word_texts,
    normalize_processing_profile,
    pcm_float32_to_wav,
    route_translation_workflow_payload,
    translation_type_for_profile,
)


class FakeResponse:
    def __init__(self, status_code, payload, headers=None):
        self.status_code = status_code
        self._payload = payload
        self.headers = headers or {}

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class FakeSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def post(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return self.responses.pop(0)


class ProviderProfileTests(unittest.TestCase):
    def test_groq_word_fallback_adds_english_spaces(self):
        pieces = detokenize_word_texts(
            [
                {"word": "Watch."},
                {"word": "Each"},
                {"word": "step"},
                {"word": "works!"},
            ]
        )

        self.assertEqual("".join(pieces), "Watch. Each step works!")

    def test_groq_word_fallback_does_not_space_cjk_characters(self):
        pieces = detokenize_word_texts(
            [{"word": "今天"}, {"word": "天氣"}, {"word": "很好"}, {"word": "。"}]
        )

        self.assertEqual("".join(pieces), "今天天氣很好。")

    def test_profiles_swap_only_asr_and_translation_provider(self):
        expected = {
            "standard": ("local", "standard"),
            "std": ("local", "standard"),
            "premium": ("local", "premium"),
            "pro": ("local", "premium"),
            "private": ("local", "private"),
        }
        for value, providers in expected.items():
            profile = normalize_processing_profile(value)
            self.assertEqual(
                (asr_provider_for_profile(profile), translation_type_for_profile(profile)),
                providers,
            )

    def test_workflow_profile_is_only_added_to_translation_requests(self):
        grouping = route_translation_workflow_payload(
            {"request_id": "group-1", "translation_type": "premium"},
            "group",
            "standard",
        )
        translation = route_translation_workflow_payload(
            {"request_id": "translate-1", "translation_type": "private"},
            "translate-groups",
            "premium",
        )

        self.assertNotIn("translation_type", grouping)
        self.assertEqual(translation["translation_type"], "premium")

    def test_pcm_encoder_writes_mono_16khz_wav(self):
        payload = pcm_float32_to_wav(np.zeros(16000, dtype=np.float32))
        with wave.open(io.BytesIO(payload), "rb") as wav:
            self.assertEqual(wav.getnchannels(), 1)
            self.assertEqual(wav.getframerate(), 16000)
            self.assertEqual(wav.getsampwidth(), 2)
            self.assertEqual(wav.getnframes(), 16000)

    def test_groq_adapter_uses_openai_audio_endpoint_and_normalizes_result(self):
        session = FakeSession(
            [
                FakeResponse(
                    200,
                    {
                        "language": "english",
                        "duration": 1.0,
                        "text": "hello world",
                        "segments": [
                            {"start": 0.0, "end": 1.0, "text": "hello world"}
                        ],
                        "words": [
                            {"start": 0.0, "end": 0.4, "word": "hello"},
                            {"start": 0.4, "end": 1.0, "word": "world"},
                        ],
                    },
                )
            ]
        )
        client = GroqWhisperClient(
            GroqWhisperSettings(api_key="test-key"),
            session=session,
        )

        segments, info = client.transcribe(
            np.zeros(16000, dtype=np.float32), language="en"
        )

        self.assertEqual(info.language, "en")
        self.assertEqual(segments[0].text, "hello world")
        self.assertEqual(
            [word.word for word in segments[0].words],
            ["hello", " world"],
        )
        self.assertEqual(
            "".join(word.word for word in segments[0].words),
            segments[0].text,
        )
        url, request = session.calls[0]
        self.assertTrue(url.endswith("/audio/transcriptions"))
        self.assertEqual(request["headers"]["Authorization"], "Bearer test-key")
        self.assertIn(("model", "whisper-large-v3"), request["data"])
        self.assertEqual(request["files"]["file"][2], "audio/wav")


if __name__ == "__main__":
    unittest.main()
