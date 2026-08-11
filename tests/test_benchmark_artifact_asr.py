import tempfile
import unittest
from pathlib import Path

from benchmark_artifact_asr import (
    BenchmarkResult,
    comparison_rows,
    format_srt_timestamp,
    transcript_similarity,
)


class ArtifactAsrBenchmarkTests(unittest.TestCase):
    def test_formats_srt_timestamp(self) -> None:
        self.assertEqual(format_srt_timestamp(3661.234), "01:01:01,234")

    def test_similarity_ignores_case_and_whitespace(self) -> None:
        self.assertEqual(transcript_similarity("Hello World", "hello\nworld"), 1.0)

    def test_comparison_calculates_speed_and_output_delta(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            baseline_srt = root / "baseline.srt"
            candidate_srt = root / "candidate.srt"
            baseline_srt.write_text("Hello world", encoding="utf-8")
            candidate_srt.write_text("Hello world!", encoding="utf-8")
            baseline_json = root / "baseline.json"
            candidate_json = root / "candidate.json"
            baseline_json.write_text(
                '{"segments": [{"text": "Hello world"}]}', encoding="utf-8"
            )
            candidate_json.write_text(
                '{"segments": [{"text": "Hello world!"}]}', encoding="utf-8"
            )
            common = {
                "case": "en",
                "label": "English",
                "language": "en",
                "source": "source.mp4",
                "audio_seconds": 60.0,
                "speed_x": 0.0,
                "realtime_factor": 0.0,
                "characters": 11,
                "word_timestamps": 2,
                "low_confidence_spans": 0,
            }
            baseline = BenchmarkResult(
                model="large-v3",
                elapsed_seconds=12.0,
                segments=3,
                srt_file=str(baseline_srt),
                json_file=str(baseline_json),
                **common,
            )
            candidate = BenchmarkResult(
                model="turbo",
                elapsed_seconds=4.0,
                segments=4,
                srt_file=str(candidate_srt),
                json_file=str(candidate_json),
                **common,
            )

            rows = comparison_rows([baseline, candidate], "large-v3", "turbo")

            self.assertEqual(rows[0]["candidate_speedup"], 3.0)
            self.assertEqual(rows[0]["segment_delta"], 1)


if __name__ == "__main__":
    unittest.main()
