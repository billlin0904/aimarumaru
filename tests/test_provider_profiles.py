import io
import unittest
import wave
from unittest import mock

import numpy as np

from provider_profiles import (
    CloudflareWhisperClient,
    CloudflareWhisperSettings,
    GroqWhisperClient,
    GroqWhisperSettings,
    TogetherWhisperClient,
    TogetherWhisperSettings,
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
        with mock.patch.dict("os.environ", {"AUDIOIO_ASR_PROVIDER": "local"}):
            for value, providers in expected.items():
                profile = normalize_processing_profile(value)
                self.assertEqual(
                    (
                        asr_provider_for_profile(profile),
                        translation_type_for_profile(profile),
                    ),
                    providers,
                )

    def test_asr_provider_can_be_switched_to_cloudflare(self):
        with mock.patch.dict(
            "os.environ",
            {"AUDIOIO_ASR_PROVIDER": "cloudflare"},
        ):
            self.assertEqual(
                asr_provider_for_profile("standard"),
                "cloudflare",
            )

    def test_asr_provider_can_be_switched_to_together(self):
        with mock.patch.dict(
            "os.environ",
            {"AUDIOIO_ASR_PROVIDER": "together"},
        ):
            self.assertEqual(
                asr_provider_for_profile("standard"),
                "together",
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

    def test_cloudflare_adapter_normalizes_workers_ai_result(self):
        session = FakeSession(
            [
                FakeResponse(
                    200,
                    {
                        "success": True,
                        "errors": [],
                        "result": {
                            "transcription_info": {
                                "language": "en",
                                "language_probability": 0.99,
                                "duration": 1.0,
                            },
                            "text": "hello world",
                            "segments": [
                                {
                                    "start": 0.0,
                                    "end": 1.0,
                                    "text": "hello world",
                                    "avg_logprob": -0.1,
                                    "no_speech_prob": 0.01,
                                    "words": [
                                        {"start": 0.0, "end": 0.4, "word": "hello"},
                                        {"start": 0.4, "end": 1.0, "word": "world"},
                                    ],
                                }
                            ],
                            "vtt": "WEBVTT",
                        },
                    },
                )
            ]
        )
        client = CloudflareWhisperClient(
            CloudflareWhisperSettings(
                account_id="test-account",
                api_token="test-token",
            ),
            session=session,
        )

        segments, info = client.transcribe(
            np.zeros(16000, dtype=np.float32),
            language=None,
            beam_size=1,
            vad_filter=True,
            condition_on_previous_text=False,
            hallucination_silence_threshold=1.0,
        )

        self.assertEqual(info.language, "en")
        self.assertEqual(info.language_probability, 0.99)
        self.assertEqual(segments[0].text, "hello world")
        self.assertEqual(
            "".join(word.word for word in segments[0].words),
            segments[0].text,
        )
        self.assertIsNone(segments[0].words[0].probability)
        url, request = session.calls[0]
        self.assertTrue(
            url.endswith(
                "/accounts/test-account/ai/run/"
                "@cf/openai/whisper-large-v3-turbo"
            )
        )
        self.assertEqual(
            request["headers"]["Authorization"],
            "Bearer test-token",
        )
        self.assertEqual(request["json"]["task"], "transcribe")
        self.assertEqual(request["json"]["beam_size"], 1)
        self.assertTrue(request["json"]["vad_filter"])
        self.assertNotIn("language", request["json"])

    def test_cloudflare_adapter_sends_manual_language(self):
        session = FakeSession(
            [
                FakeResponse(
                    200,
                    {
                        "success": True,
                        "errors": [],
                        "result": {
                            "text": "こんにちは",
                            "language": "ja",
                            "segments": [
                                {"start": 0.0, "end": 1.0, "text": "こんにちは"}
                            ],
                        },
                    },
                )
            ]
        )
        client = CloudflareWhisperClient(
            CloudflareWhisperSettings(
                account_id="test-account",
                api_token="test-token",
            ),
            session=session,
        )

        _, info = client.transcribe(
            np.zeros(16000, dtype=np.float32),
            language="ja",
        )

        self.assertEqual(info.language, "ja")
        self.assertEqual(session.calls[0][1]["json"]["language"], "ja")

    def test_together_adapter_uses_auto_language_for_detection(self):
        session = FakeSession(
            [
                FakeResponse(
                    200,
                    {
                        "language": "korean",
                        "language_probability": 0.97,
                        "duration": 1.0,
                        "text": "안녕하세요",
                        "words": [
                            {"start": 0.0, "end": 1.0, "word": "안녕하세요"}
                        ],
                    },
                )
            ]
        )
        client = TogetherWhisperClient(
            TogetherWhisperSettings(api_key="test-key"),
            session=session,
        )

        segments, info = client.transcribe(
            np.zeros(16000, dtype=np.float32),
            language=None,
        )

        self.assertEqual(segments[0].text, "안녕하세요")
        self.assertEqual(segments[0].words[0].word, "안녕하세요")
        self.assertEqual(segments[0].words[0].start, 0.0)
        self.assertEqual(segments[0].words[0].end, 1.0)
        self.assertEqual(info.language, "ko")
        self.assertEqual(info.language_probability, 0.97)
        url, request = session.calls[0]
        self.assertTrue(url.endswith("/audio/transcriptions"))
        self.assertEqual(request["headers"]["Authorization"], "Bearer test-key")
        self.assertIn(("model", "openai/whisper-large-v3"), request["data"])
        self.assertIn(("language", "auto"), request["data"])
        self.assertIn(("response_format", "verbose_json"), request["data"])
        self.assertIn(("timestamp_granularities[]", "word"), request["data"])
        self.assertIn(("timestamp_granularities[]", "segment"), request["data"])
        self.assertNotIn(("diarize", "true"), request["data"])

    def test_together_adapter_sends_manual_language(self):
        session = FakeSession(
            [
                FakeResponse(
                    200,
                    {
                        "language": "japanese",
                        "text": "こんにちは",
                        "segments": [
                            {"start": 0.0, "end": 1.0, "text": "こんにちは"}
                        ],
                    },
                )
            ]
        )
        client = TogetherWhisperClient(
            TogetherWhisperSettings(api_key="test-key"),
            session=session,
        )

        _, info = client.transcribe(
            np.zeros(16000, dtype=np.float32),
            language="ja-JP",
        )

        self.assertEqual(info.language, "ja")
        self.assertIn(("language", "ja"), session.calls[0][1]["data"])

    def test_together_adapter_splits_provider_block_into_timed_sentences(self):
        session = FakeSession(
            [
                FakeResponse(
                    200,
                    {
                        "language": "english",
                        "duration": 6.0,
                        "text": "- Hello everyone. Concerto No. 1. We are ready.",
                        "segments": [
                            {
                                "start": 0.0,
                                "end": 6.0,
                                "text": "- Hello everyone. Concerto No. 1. We are ready.",
                            }
                        ],
                        "words": [
                            {"word": "-", "start": 0.0, "end": 0.0},
                            {"word": "Hello", "start": 0.1, "end": 0.5},
                            {"word": "everyone.", "start": 0.6, "end": 1.2},
                            {"word": "Concerto", "start": 1.5, "end": 1.8},
                            {"word": "No.", "start": 1.9, "end": 2.1},
                            {"word": "1.", "start": 2.2, "end": 2.7},
                            {"word": "We", "start": 3.0, "end": 3.2},
                            {"word": "are", "start": 3.3, "end": 3.5},
                            {"word": "ready.", "start": 3.6, "end": 4.1},
                        ],
                    },
                )
            ]
        )
        client = TogetherWhisperClient(
            TogetherWhisperSettings(api_key="test-key"),
            session=session,
        )

        segments, _ = client.transcribe(np.zeros(16000 * 6, dtype=np.float32))

        self.assertEqual(
            [segment.text for segment in segments],
            ["- Hello everyone.", "Concerto No. 1.", "We are ready."],
        )
        self.assertEqual(
            [(segment.start, segment.end) for segment in segments],
            [(0.1, 1.2), (1.5, 2.7), (3.0, 4.1)],
        )
        self.assertEqual([len(segment.words) for segment in segments], [3, 3, 3])
        self.assertEqual(segments[0].words[0].start, 0.1)

    def test_together_diarization_uses_speaker_segments(self):
        session = FakeSession(
            [
                FakeResponse(
                    200,
                    {
                        "language": "english",
                        "text": "Hello. Welcome.",
                        "words": [
                            {
                                "start": 0.0,
                                "end": 0.5,
                                "word": "Hello.",
                                "speaker_id": "SPEAKER_00",
                            },
                            {
                                "start": 0.6,
                                "end": 1.2,
                                "word": "Welcome.",
                                "speaker_id": "SPEAKER_01",
                            },
                        ],
                        "speaker_segments": [
                            {
                                "start": 0.0,
                                "end": 0.5,
                                "text": "Hello.",
                                "speaker_id": "SPEAKER_00",
                            },
                            {
                                "start": 0.6,
                                "end": 1.2,
                                "text": "Welcome.",
                                "speaker_id": "SPEAKER_01",
                            },
                        ],
                    },
                )
            ]
        )
        client = TogetherWhisperClient(
            TogetherWhisperSettings(api_key="test-key"),
            session=session,
        )

        segments, info = client.transcribe_diarized(
            np.zeros(16000 * 2, dtype=np.float32),
            language=None,
            min_speakers=1,
            max_speakers=4,
        )

        self.assertEqual(info.language, "en")
        self.assertEqual(
            [segment.speaker_id for segment in segments],
            ["SPEAKER_00", "SPEAKER_01"],
        )
        self.assertEqual(segments[1].words[0].speaker_id, "SPEAKER_01")
        request_data = session.calls[0][1]["data"]
        self.assertIn(("diarize", "true"), request_data)
        self.assertIn(("min_speakers", "1"), request_data)
        self.assertIn(("max_speakers", "4"), request_data)


if __name__ == "__main__":
    unittest.main()
