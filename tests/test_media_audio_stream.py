import asyncio
import shutil
import subprocess
import tempfile
import unittest
import wave
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from media_audio_stream import (
    FFmpegPcmChunkStream,
    MediaAudioSource,
    PcmAudioChunk,
    _STREAM_END,
    decode_media_prefix,
    probe_media_duration,
)
from youtube_live import (
    detect_audio_language,
    owned_whisper_segment,
    transcribe_audio_stream,
    whisper_word_payloads,
)


class FakeAuto2Lrc:
    def __init__(self) -> None:
        self.sample_counts: list[int] = []
        self.options: list[dict] = []
        self.clear_count = 0

    def transcribe(self, samples, **options):
        call_index = len(self.sample_counts)
        self.sample_counts.append(len(samples))
        self.options.append(options)
        duration = len(samples) / 16000
        word_start = 0.2 if call_index == 0 or duration < 2.0 else 1.2
        words = [
            SimpleNamespace(
                word=" first",
                start=word_start,
                end=min(word_start + 0.6, duration),
                probability=0.99,
            )
        ]
        segments = [
            SimpleNamespace(start=0.0, end=min(3.0, duration), text="first", words=words),
            SimpleNamespace(
                start=max(0.0, duration - 1.0),
                end=duration,
                text="tail",
                words=[],
            ),
        ]
        info = SimpleNamespace(language="en", language_probability=0.98)
        return iter(segments), info

    def clear_model_cache(self) -> None:
        self.clear_count += 1


class BoundaryOverlapAuto2Lrc(FakeAuto2Lrc):
    def transcribe(self, samples, **options):
        call_index = len(self.sample_counts)
        self.sample_counts.append(len(samples))
        duration = len(samples) / 16000
        if call_index == 0:
            segments = [
                SimpleNamespace(start=0.0, end=8.0, text="before boundary", words=[]),
                SimpleNamespace(
                    start=8.3,
                    end=9.6,
                    text="which I'll label WQ",
                    words=[
                        SimpleNamespace(word=" which", start=8.3, end=8.7, probability=0.99),
                        SimpleNamespace(word=" I'll", start=8.7, end=9.0, probability=0.99),
                        SimpleNamespace(word=" label", start=9.0, end=9.3, probability=0.99),
                        SimpleNamespace(word=" WQ", start=9.3, end=9.6, probability=0.99),
                    ],
                ),
            ]
        elif call_index == 1:
            segments = [
                SimpleNamespace(
                    start=0.0,
                    end=4.0,
                    text="matrix which I'll label WQ and multiplying",
                    words=[
                        SimpleNamespace(word=" matrix", start=0.0, end=0.4, probability=0.99),
                        SimpleNamespace(word=" which", start=0.4, end=0.8, probability=0.99),
                        SimpleNamespace(word=" I'll", start=0.8, end=1.0, probability=0.99),
                        SimpleNamespace(word=" label", start=1.0, end=1.3, probability=0.99),
                        SimpleNamespace(word=" WQ", start=1.3, end=1.6, probability=0.99),
                        SimpleNamespace(word=" and", start=1.6, end=2.0, probability=0.99),
                        SimpleNamespace(word=" multiplying", start=2.0, end=4.0, probability=0.99),
                    ],
                )
            ]
        else:
            segments = [
                SimpleNamespace(start=1.0, end=min(3.0, duration), text="after", words=[])
            ]
        return iter(segments), SimpleNamespace(language="en", language_probability=0.98)


class FakeRemoteWhisperClient:
    def __init__(
        self,
        model_name="@cf/openai/whisper-large-v3-turbo",
        reported_languages=None,
        include_words=False,
    ) -> None:
        self.calls = 0
        self.model_name = model_name
        self.options = []
        self.diarized_options = []
        self.reported_languages = list(reported_languages or [])
        self.include_words = include_words

    def transcribe(self, samples, *, language=None, **options):
        self.calls += 1
        self.options.append({"language": language, **options})
        duration = len(samples) / 16000
        reported_language = (
            self.reported_languages[self.calls - 1]
            if self.calls <= len(self.reported_languages)
            else (language or "en")
        )
        return [
            SimpleNamespace(
                start=0.1,
                end=min(1.0, duration),
                text="cloud transcript",
                words=(
                    [
                        SimpleNamespace(
                            word="cloud transcript",
                            start=0.1,
                            end=min(1.0, duration),
                            probability=0.99,
                        )
                    ]
                    if self.include_words
                    else []
                ),
            )
        ], SimpleNamespace(language=reported_language, language_probability=0.97)

    def transcribe_diarized(
        self,
        samples,
        *,
        language=None,
        min_speakers=1,
        max_speakers=5,
    ):
        self.diarized_options.append(
            {
                "language": language,
                "min_speakers": min_speakers,
                "max_speakers": max_speakers,
            }
        )
        duration = len(samples) / 16000
        speaker_id = f"SPEAKER_{len(self.diarized_options) % 2:02d}"
        return [
            SimpleNamespace(
                start=0.1,
                end=min(1.0, duration),
                text="speaker transcript",
                speaker_id=speaker_id,
                words=[
                    SimpleNamespace(
                        word=" speaker",
                        start=0.1,
                        end=min(0.8, duration),
                        probability=0.99,
                        speaker_id=speaker_id,
                    )
                ],
            )
        ], SimpleNamespace(language=language or "en", language_probability=0.97)


class FakeTogetherRealtimeTranscriber:
    instances = []
    result_texts = []
    total_calls = 0

    def __init__(self, settings, *, language=None) -> None:
        self.settings = settings
        self.language = language
        self.calls = []
        self.closed = False
        self.__class__.instances.append(self)

    def transcribe(self, pcm_audio, *, on_delta=None, cancel_check=None):
        del cancel_check
        call_number = len(self.calls) + 1
        self.calls.append(pcm_audio)
        if on_delta is not None:
            on_delta(f"partial {call_number}")
        result_index = self.__class__.total_calls
        self.__class__.total_calls += 1
        text = (
            self.__class__.result_texts[result_index]
            if result_index < len(self.__class__.result_texts)
            else f"final {call_number}"
        )
        return SimpleNamespace(text=text, delta_count=1)

    def close(self):
        self.closed = True


@unittest.skipUnless(shutil.which("ffmpeg"), "FFmpeg is required")
class MediaAudioStreamTests(unittest.TestCase):
    def test_local_language_detection_keeps_whisper_model_warm(self) -> None:
        local = FakeAuto2Lrc()
        prefix_pcm = b"\0\0" * (5 * 16000)

        with mock.patch("youtube_live.decode_media_prefix", return_value=prefix_pcm):
            result = detect_audio_language(
                local,
                self.audio_path,
                job_id="warm-model-job",
                audio_duration_hint=25.0,
                asr_provider="local",
            )

        self.assertEqual(result["language"], "en")
        self.assertEqual(local.clear_count, 0)
        self.assertEqual(local.sample_counts, [5 * 16000])

    def test_together_language_detection_uses_http_transcription(self) -> None:
        prefix_pcm = b"\0\0" * (5 * 16000)
        FakeTogetherRealtimeTranscriber.instances.clear()
        remote = FakeRemoteWhisperClient(
            model_name="openai/whisper-large-v3",
            reported_languages=["ja"],
        )

        with mock.patch("youtube_live.decode_media_prefix", return_value=prefix_pcm):
            result = detect_audio_language(
                FakeAuto2Lrc(),
                self.audio_path,
                job_id="together-language-job",
                audio_duration_hint=25.0,
                asr_provider="together",
                asr_client=remote,
            )

        self.assertEqual(result["language"], "ja")
        self.assertEqual(result["language_probability"], 0.97)
        self.assertEqual(remote.calls, 1)
        self.assertIsNone(remote.options[0]["language"])
        self.assertEqual(FakeTogetherRealtimeTranscriber.instances, [])

    def test_full_groq_segment_uses_authoritative_segment_text(self) -> None:
        segment = SimpleNamespace(
            text="Watch. Each step adds a few more numbers.",
            start=0.0,
            end=2.0,
            words=[
                SimpleNamespace(word="Watch.", start=0.0, end=0.4),
                SimpleNamespace(word="Each", start=0.4, end=0.8),
                SimpleNamespace(word="step", start=0.8, end=1.1),
                SimpleNamespace(word="adds", start=1.1, end=1.4),
                SimpleNamespace(word="a", start=1.4, end=1.5),
                SimpleNamespace(word="few", start=1.5, end=1.7),
                SimpleNamespace(word="more", start=1.7, end=1.8),
                SimpleNamespace(word="numbers.", start=1.8, end=2.0),
            ],
        )

        owned = owned_whisper_segment(segment, 0.0, 3.0, True)

        self.assertIsNotNone(owned)
        self.assertEqual(owned[0], segment.text)

    def test_word_payloads_clamp_locally_reversed_timestamps(self) -> None:
        words = [
            SimpleNamespace(word=" so", start=18.6, end=18.84, probability=0.99),
            SimpleNamespace(word=" far?", start=18.2, end=19.32, probability=0.99),
        ]

        payloads = whisper_word_payloads(
            SimpleNamespace(words=words),
            "en",
        )

        self.assertEqual([item["start"] for item in payloads], [18.6, 18.6])
        self.assertGreater(payloads[1]["end"], payloads[1]["start"])

    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.audio_path = Path(self.temporary_directory.name) / "audio.wav"
        with wave.open(str(self.audio_path), "wb") as output:
            output.setnchannels(1)
            output.setsampwidth(2)
            output.setframerate(16000)
            output.writeframes(b"\0\0" * (25 * 16000))

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_ffmpeg_stream_uses_bounded_overlapping_chunks(self) -> None:
        with FFmpegPcmChunkStream(
            MediaAudioSource(
                location=str(self.audio_path),
                label="local-test-audio",
            ),
            chunk_seconds=10,
            initial_chunk_seconds=6,
            overlap_seconds=2,
            queue_size=1,
        ) as stream:
            chunks = list(stream)

        self.assertEqual(
            [chunk.offset_seconds for chunk in chunks],
            [0.0, 4.0, 12.0, 20.0],
        )
        self.assertEqual(
            [round(chunk.duration_seconds) for chunk in chunks],
            [6, 10, 10, 5],
        )
        self.assertEqual(
            [chunk.is_final for chunk in chunks],
            [False, False, False, True],
        )
        self.assertLessEqual(max(len(chunk.data) for chunk in chunks), 10 * 16000 * 2)

    def test_stream_prefetches_configured_chunks_before_first_yield(self) -> None:
        stream = FFmpegPcmChunkStream(
            self.audio_path,
            queue_size=3,
            prefetch_chunks=2,
        )
        chunks = [
            PcmAudioChunk(b"\0\0", 0.0, 1.0, False),
            PcmAudioChunk(b"\0\0", 1.0, 1.0, True),
        ]
        for item in (*chunks, _STREAM_END):
            stream._queue.put_nowait(item)

        with mock.patch.object(stream, "start"):
            iterator = iter(stream)
            self.assertEqual(next(iterator), chunks[0])
            self.assertEqual(stream._queue.qsize(), 1)
            self.assertEqual(list(iterator), [chunks[1]])

    def test_language_prefix_is_bounded(self) -> None:
        prefix = decode_media_prefix(self.audio_path, 3)
        self.assertEqual(len(prefix), 3 * 16000 * 2)
        self.assertAlmostEqual(probe_media_duration(self.audio_path), 25.0, places=2)

    def test_stream_rejects_media_without_an_audio_track(self) -> None:
        video_path = Path(self.temporary_directory.name) / "silent-video.mp4"
        subprocess.run(
            [
                shutil.which("ffmpeg"),
                "-nostdin",
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "lavfi",
                "-i",
                "color=size=32x32:rate=1:duration=1",
                "-an",
                "-y",
                str(video_path),
            ],
            check=True,
        )
        with self.assertRaisesRegex(RuntimeError, "音軌|FFmpeg"):
            with FFmpegPcmChunkStream(video_path, chunk_seconds=10) as stream:
                list(stream)

    def test_transcription_offsets_segments_and_emits_chunk_progress(self) -> None:
        async def run_test():
            event_queue: asyncio.Queue = asyncio.Queue()
            fake = FakeAuto2Lrc()
            telemetry = []
            with (
                mock.patch("youtube_live.YOUTUBE_WHISPER_STREAM_CHUNK_SECONDS", 10.0),
                mock.patch("youtube_live.YOUTUBE_WHISPER_STREAM_OVERLAP_SECONDS", 2.0),
                mock.patch("youtube_live.YOUTUBE_WHISPER_STREAM_QUEUE_SIZE", 1),
            ):
                result = await asyncio.to_thread(
                    transcribe_audio_stream,
                    fake,
                    self.audio_path,
                    "en",
                    asyncio.get_running_loop(),
                    event_queue,
                    "test-job",
                    None,
                    True,
                    25.0,
                    "accurate",
                    "local",
                    None,
                    telemetry.append,
                )
            events = []
            while not event_queue.empty():
                events.append(event_queue.get_nowait())
            return fake, result, events, telemetry

        fake, result, events, telemetry = asyncio.run(run_test())
        segment_events = [event["data"] for event in events if event["event"] == "segment"]
        progress_events = [event["data"] for event in events if event["event"] == "progress"]

        self.assertEqual(result["segments_count"], 4)
        self.assertEqual(
            [event["start"] for event in segment_events],
            [0.2, 9.2, 17.2, 24.0],
        )
        self.assertEqual(segment_events[1]["words"][0]["start"], 9.2)
        self.assertEqual(len(progress_events), 3)
        self.assertEqual(round(progress_events[-1]["progress_percent"]), 100)
        self.assertLessEqual(max(fake.sample_counts), 10 * 16000)
        self.assertTrue(all(options["beam_size"] == 5 for options in fake.options))
        self.assertEqual(fake.clear_count, 1)
        self.assertEqual(len(telemetry), 3)
        self.assertEqual(telemetry[-1]["chunk_index"], 3)
        self.assertEqual(round(telemetry[-1]["progress_percent"]), 100)
        self.assertEqual(telemetry[-1]["segments_emitted"], 4)
        self.assertGreaterEqual(telemetry[-1]["input_wait_ms"], 0)
        self.assertGreaterEqual(telemetry[-1]["inference_ms"], 0)
        self.assertGreaterEqual(telemetry[-1]["event_emit_ms"], 0)

    def test_fast_transcription_mode_uses_greedy_search(self) -> None:
        async def run_test():
            event_queue: asyncio.Queue = asyncio.Queue()
            fake = FakeAuto2Lrc()
            with (
                mock.patch("youtube_live.YOUTUBE_WHISPER_STREAM_CHUNK_SECONDS", 10.0),
                mock.patch("youtube_live.YOUTUBE_WHISPER_STREAM_OVERLAP_SECONDS", 2.0),
                mock.patch("youtube_live.YOUTUBE_WHISPER_STREAM_QUEUE_SIZE", 1),
            ):
                result = await asyncio.to_thread(
                    transcribe_audio_stream,
                    fake,
                    self.audio_path,
                    "en",
                    asyncio.get_running_loop(),
                    event_queue,
                    "fast-job",
                    None,
                    False,
                    25.0,
                    "fast",
                )
            return fake, result

        fake, result = asyncio.run(run_test())

        self.assertTrue(all(options["beam_size"] == 1 for options in fake.options))
        self.assertEqual(result["transcription_mode"], "fast")
        self.assertEqual(result["beam_size"], 1)

    def test_remote_asr_uses_common_vad_and_does_not_load_local_model(self) -> None:
        async def run_test():
            event_queue: asyncio.Queue = asyncio.Queue()
            local = FakeAuto2Lrc()
            remote = FakeRemoteWhisperClient()
            with (
                mock.patch("youtube_live.YOUTUBE_WHISPER_STREAM_CHUNK_SECONDS", 30.0),
                mock.patch("youtube_live.YOUTUBE_WHISPER_INITIAL_CHUNK_SECONDS", 30.0),
                mock.patch("youtube_live.YOUTUBE_WHISPER_STREAM_OVERLAP_SECONDS", 0.0),
                mock.patch(
                    "youtube_live.remote_vad_speech_intervals",
                    return_value=[(0.0, 25.0)],
                ),
            ):
                result = await asyncio.to_thread(
                    transcribe_audio_stream,
                    local,
                    self.audio_path,
                    "en",
                    asyncio.get_running_loop(),
                    event_queue,
                    "cloudflare-job",
                    None,
                    False,
                    25.0,
                    "accurate",
                    "cloudflare",
                    remote,
                )
            return local, remote, result

        local, remote, result = asyncio.run(run_test())
        self.assertEqual(remote.calls, 1)
        self.assertEqual(remote.options[0]["beam_size"], 5)
        self.assertTrue(remote.options[0]["vad_filter"])
        self.assertEqual(local.sample_counts, [])
        self.assertEqual(local.clear_count, 0)
        self.assertEqual(result["asr_provider"], "cloudflare")
        self.assertEqual(
            result["asr_model"],
            "@cf/openai/whisper-large-v3-turbo",
        )

    def test_cloudflare_auto_omits_language_for_every_chunk(self) -> None:
        async def run_test():
            event_queue: asyncio.Queue = asyncio.Queue()
            remote = FakeRemoteWhisperClient(
                reported_languages=["ja", "ko", "en"],
            )
            with (
                mock.patch("youtube_live.YOUTUBE_WHISPER_STREAM_CHUNK_SECONDS", 10.0),
                mock.patch("youtube_live.YOUTUBE_WHISPER_INITIAL_CHUNK_SECONDS", 10.0),
                mock.patch("youtube_live.YOUTUBE_WHISPER_STREAM_OVERLAP_SECONDS", 0.0),
                mock.patch(
                    "youtube_live.remote_vad_speech_intervals",
                    side_effect=lambda samples: [(0.0, len(samples) / 16000)],
                ),
            ):
                await asyncio.to_thread(
                    transcribe_audio_stream,
                    FakeAuto2Lrc(),
                    self.audio_path,
                    None,
                    asyncio.get_running_loop(),
                    event_queue,
                    "cloudflare-auto-job",
                    None,
                    False,
                    25.0,
                    "accurate",
                    "cloudflare",
                    remote,
                )
            events = []
            while not event_queue.empty():
                events.append(event_queue.get_nowait())
            return remote, events

        remote, events = asyncio.run(run_test())
        segment_languages = [
            event["data"]["language"]
            for event in events
            if event["event"] == "segment"
        ]

        self.assertEqual(remote.calls, 3)
        self.assertEqual([item["language"] for item in remote.options], [None] * 3)
        self.assertEqual(segment_languages, ["ja", "ko", "en"])

    def test_cloudflare_manual_language_is_sent_for_every_chunk(self) -> None:
        async def run_test():
            event_queue: asyncio.Queue = asyncio.Queue()
            remote = FakeRemoteWhisperClient(
                reported_languages=["ko", "en", "zh"],
            )
            with (
                mock.patch("youtube_live.YOUTUBE_WHISPER_STREAM_CHUNK_SECONDS", 10.0),
                mock.patch("youtube_live.YOUTUBE_WHISPER_INITIAL_CHUNK_SECONDS", 10.0),
                mock.patch("youtube_live.YOUTUBE_WHISPER_STREAM_OVERLAP_SECONDS", 0.0),
                mock.patch(
                    "youtube_live.remote_vad_speech_intervals",
                    side_effect=lambda samples: [(0.0, len(samples) / 16000)],
                ),
            ):
                await asyncio.to_thread(
                    transcribe_audio_stream,
                    FakeAuto2Lrc(),
                    self.audio_path,
                    "ja",
                    asyncio.get_running_loop(),
                    event_queue,
                    "cloudflare-manual-job",
                    None,
                    False,
                    25.0,
                    "accurate",
                    "cloudflare",
                    remote,
                )
            events = []
            while not event_queue.empty():
                events.append(event_queue.get_nowait())
            return remote, events

        remote, events = asyncio.run(run_test())
        segment_languages = [
            event["data"]["language"]
            for event in events
            if event["event"] == "segment"
        ]

        self.assertEqual(remote.calls, 3)
        self.assertEqual([item["language"] for item in remote.options], ["ja"] * 3)
        self.assertEqual(segment_languages, ["ja", "ja", "ja"])

    def test_together_http_emits_word_timestamps_without_websocket(self) -> None:
        async def run_test():
            event_queue: asyncio.Queue = asyncio.Queue()
            remote = FakeRemoteWhisperClient(
                model_name="openai/whisper-large-v3",
                include_words=True,
            )
            FakeTogetherRealtimeTranscriber.instances.clear()
            with (
                mock.patch("youtube_live.YOUTUBE_WHISPER_STREAM_CHUNK_SECONDS", 10.0),
                mock.patch("youtube_live.YOUTUBE_WHISPER_INITIAL_CHUNK_SECONDS", 10.0),
                mock.patch("youtube_live.YOUTUBE_WHISPER_STREAM_OVERLAP_SECONDS", 0.0),
                mock.patch(
                    "youtube_live.remote_vad_speech_intervals",
                    side_effect=lambda samples: [(0.0, len(samples) / 16000)],
                ),
            ):
                result = await asyncio.to_thread(
                    transcribe_audio_stream,
                    FakeAuto2Lrc(),
                    self.audio_path,
                    None,
                    asyncio.get_running_loop(),
                    event_queue,
                    "together-job",
                    None,
                    True,
                    25.0,
                    "accurate",
                    "together",
                    remote,
                    None,
                    "ko",
                )
            events = []
            while not event_queue.empty():
                events.append(event_queue.get_nowait())
            return remote, result, events

        remote, result, events = asyncio.run(run_test())
        segment_events = [
            event["data"] for event in events if event["event"] == "segment"
        ]

        self.assertEqual(remote.calls, 3)
        self.assertEqual(FakeTogetherRealtimeTranscriber.instances, [])
        self.assertEqual([item["language"] for item in segment_events], ["en"] * 3)
        self.assertTrue(all(len(item["words"]) == 1 for item in segment_events))
        self.assertEqual(
            [item["words"][0]["start"] for item in segment_events],
            [0.1, 10.1, 20.1],
        )
        self.assertEqual(result["timing_precision"], "word")
        self.assertEqual(result["transcription_delivery"], "http")

    def test_together_speaker_diarization_is_temporarily_disabled(self) -> None:
        async def run_test():
            event_queue: asyncio.Queue = asyncio.Queue()
            remote = FakeRemoteWhisperClient(model_name="openai/whisper-large-v3")
            with self.assertRaisesRegex(RuntimeError, "講者辨識暫時停用"):
                await asyncio.to_thread(
                    transcribe_audio_stream,
                    FakeAuto2Lrc(),
                    self.audio_path,
                    None,
                    asyncio.get_running_loop(),
                    event_queue,
                    "together-batch-job",
                    None,
                    True,
                    25.0,
                    "accurate",
                    "together",
                    remote,
                    None,
                    "en",
                    True,
                )

        asyncio.run(run_test())

    def test_overlap_boundary_splits_by_word_without_losing_speech(self) -> None:
        async def run_test():
            event_queue: asyncio.Queue = asyncio.Queue()
            fake = BoundaryOverlapAuto2Lrc()
            with (
                mock.patch("youtube_live.YOUTUBE_WHISPER_STREAM_CHUNK_SECONDS", 10.0),
                mock.patch("youtube_live.YOUTUBE_WHISPER_INITIAL_CHUNK_SECONDS", 10.0),
                mock.patch("youtube_live.YOUTUBE_WHISPER_STREAM_OVERLAP_SECONDS", 2.0),
                mock.patch("youtube_live.YOUTUBE_WHISPER_STREAM_QUEUE_SIZE", 1),
            ):
                result = await asyncio.to_thread(
                    transcribe_audio_stream,
                    fake,
                    self.audio_path,
                    "en",
                    asyncio.get_running_loop(),
                    event_queue,
                    "boundary-job",
                    None,
                    True,
                    25.0,
                )
            events = []
            while not event_queue.empty():
                events.append(event_queue.get_nowait())
            return result, events

        result, events = asyncio.run(run_test())
        segments = [event["data"] for event in events if event["event"] == "segment"]
        texts = [segment["text"] for segment in segments]
        starts = [segment["start"] for segment in segments]

        self.assertIn("which I'll", texts)
        self.assertIn("label WQ and multiplying", texts)
        boundary_text = " ".join(texts)
        for word in ("which", "I'll", "label", "WQ", "multiplying"):
            self.assertEqual(boundary_text.split().count(word), 1)
        self.assertEqual(starts, sorted(starts))
        self.assertEqual(result["segments_count"], len(segments))


if __name__ == "__main__":
    unittest.main()
