import argparse
import json
import shutil
import subprocess
import tempfile
import time
from collections import Counter
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from faster_whisper import WhisperModel


TEMPERATURES = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
CASES = {
    "A_current": {
        "temperature": TEMPERATURES,
        "condition_on_previous_text": True,
        "vad_filter": False,
    },
    "B_fixed_temperature": {
        "temperature": 0.0,
        "condition_on_previous_text": True,
        "vad_filter": False,
    },
    "C_no_previous_text": {
        "temperature": 0.0,
        "condition_on_previous_text": False,
        "vad_filter": False,
    },
    "D_vad": {
        "temperature": 0.0,
        "condition_on_previous_text": True,
        "vad_filter": True,
    },
}


@dataclass
class CaseResult:
    name: str
    elapsed_seconds: float
    audio_seconds: float
    audio_seconds_after_vad: float
    realtime_factor: float
    speed_x: float
    segment_count: int
    observed_window_count: int
    observed_fallback_window_count: int
    temperature_distribution: dict[str, int]
    repeated_segment_ratio: float
    average_log_probability: float | None
    maximum_compression_ratio: float | None
    character_count: int
    text_file: str
    options: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare faster-whisper fallback, previous-text, and VAD settings."
    )
    parser.add_argument("input", type=Path, help="Audio or video file to benchmark.")
    parser.add_argument("--output-dir", type=Path, default=Path("benchmark_results"))
    parser.add_argument("--model", default="large-v3")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--compute-type", default="float16")
    parser.add_argument("--language", default="", help="Empty means automatic detection.")
    parser.add_argument("--beam-size", type=int, default=5)
    parser.add_argument("--start", type=float, default=0, help="Clip start time in seconds.")
    parser.add_argument(
        "--duration",
        type=float,
        default=3600,
        help="Benchmark clip duration in seconds. Use 0 for the full file.",
    )
    parser.add_argument(
        "--cases",
        nargs="+",
        choices=CASES,
        default=list(CASES),
        help="Cases to run.",
    )
    return parser.parse_args()


def run_ffmpeg(source: Path, destination: Path, start: float, duration: float) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg is required when --start or --duration is used.")

    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        str(max(0, start)),
        "-i",
        str(source),
    ]
    if duration > 0:
        command.extend(["-t", str(duration)])
    command.extend(["-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(destination)])
    subprocess.run(command, check=True)


def normalize_segment(text: str) -> str:
    return "".join(text.lower().split())


def repeated_segment_ratio(texts: list[str]) -> float:
    normalized = [normalize_segment(text) for text in texts]
    normalized = [text for text in normalized if text]
    if not normalized:
        return 0.0
    counts = Counter(normalized)
    repeats = sum(count - 1 for count in counts.values() if count > 1)
    return repeats / len(normalized)


def run_case(
    model: WhisperModel,
    case_name: str,
    audio_path: Path,
    output_dir: Path,
    language: str,
    beam_size: int,
) -> tuple[CaseResult, str]:
    options = dict(CASES[case_name])
    started_at = time.perf_counter()
    segments, info = model.transcribe(
        str(audio_path),
        language=language or None,
        beam_size=beam_size,
        **options,
    )

    segment_rows: list[dict[str, Any]] = []
    window_temperatures: dict[int, float] = {}
    for segment in segments:
        text = segment.text.strip()
        segment_rows.append(
            {
                "seek": segment.seek,
                "start": segment.start,
                "end": segment.end,
                "text": text,
                "temperature": segment.temperature,
                "avg_logprob": segment.avg_logprob,
                "compression_ratio": segment.compression_ratio,
            }
        )
        temperature = float(segment.temperature or 0)
        window_temperatures[segment.seek] = max(
            temperature,
            window_temperatures.get(segment.seek, 0),
        )

    elapsed = time.perf_counter() - started_at
    texts = [row["text"] for row in segment_rows if row["text"]]
    full_text = "\n".join(texts)
    text_path = output_dir / f"{case_name}.txt"
    segments_path = output_dir / f"{case_name}_segments.json"
    text_path.write_text(full_text, encoding="utf-8")
    segments_path.write_text(
        json.dumps(segment_rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    audio_seconds = float(info.duration)
    duration_after_vad = float(info.duration_after_vad)
    temperatures = Counter(f"{value:.1f}" for value in window_temperatures.values())
    log_probabilities = [row["avg_logprob"] for row in segment_rows]
    compression_ratios = [row["compression_ratio"] for row in segment_rows]
    result = CaseResult(
        name=case_name,
        elapsed_seconds=round(elapsed, 3),
        audio_seconds=round(audio_seconds, 3),
        audio_seconds_after_vad=round(duration_after_vad, 3),
        realtime_factor=round(elapsed / audio_seconds, 4) if audio_seconds else 0,
        speed_x=round(audio_seconds / elapsed, 3) if elapsed else 0,
        segment_count=len(segment_rows),
        observed_window_count=len(window_temperatures),
        observed_fallback_window_count=sum(
            temperature > 0 for temperature in window_temperatures.values()
        ),
        temperature_distribution=dict(sorted(temperatures.items())),
        repeated_segment_ratio=round(repeated_segment_ratio(texts), 4),
        average_log_probability=(
            round(sum(log_probabilities) / len(log_probabilities), 4)
            if log_probabilities
            else None
        ),
        maximum_compression_ratio=(
            round(max(compression_ratios), 4) if compression_ratios else None
        ),
        character_count=len(full_text),
        text_file=str(text_path),
        options=options,
    )
    return result, full_text


def warm_up(model: WhisperModel, audio_path: Path, language: str) -> None:
    segments, _ = model.transcribe(
        str(audio_path),
        language=language or None,
        beam_size=1,
        temperature=0.0,
        condition_on_previous_text=False,
        clip_timestamps="0,30",
    )
    list(segments)


def write_markdown(
    output_path: Path,
    results: list[CaseResult],
    texts: dict[str, str],
) -> None:
    lines = [
        "# faster-whisper benchmark",
        "",
        "| Case | Time (s) | Speed | RTF | VAD audio (s) | Fallback windows | Repeated segments | Characters |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for result in results:
        lines.append(
            f"| {result.name} | {result.elapsed_seconds:.3f} | "
            f"{result.speed_x:.3f}x | {result.realtime_factor:.4f} | "
            f"{result.audio_seconds_after_vad:.3f} | "
            f"{result.observed_fallback_window_count}/{result.observed_window_count} | "
            f"{result.repeated_segment_ratio:.2%} | {result.character_count} |"
        )

    lines.extend(["", "## Text similarity", ""])
    baseline = texts.get("A_current")
    if baseline is None and texts:
        baseline = next(iter(texts.values()))
    for name, text in texts.items():
        ratio = SequenceMatcher(None, baseline or "", text).ratio()
        lines.append(f"- `{name}`: {ratio:.2%} compared with the first baseline")

    lines.extend(
        [
            "",
            "> Fallback counts only include windows that emitted at least one segment.",
            "",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    source = args.input.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="whisper_benchmark_") as temp_dir:
        if args.start > 0 or args.duration > 0:
            audio_path = Path(temp_dir) / "benchmark.wav"
            run_ffmpeg(source, audio_path, args.start, args.duration)
        else:
            audio_path = source

        print(f"Loading {args.model} on {args.device} ({args.compute_type})")
        model = WhisperModel(
            args.model,
            device=args.device,
            compute_type=args.compute_type,
        )
        print("Warming up model")
        warm_up(model, audio_path, args.language)

        results: list[CaseResult] = []
        texts: dict[str, str] = {}
        for case_name in args.cases:
            print(f"Running {case_name}")
            result, text = run_case(
                model,
                case_name,
                audio_path,
                output_dir,
                args.language,
                args.beam_size,
            )
            results.append(result)
            texts[case_name] = text
            print(
                f"  {result.elapsed_seconds:.1f}s, {result.speed_x:.2f}x, "
                f"fallback {result.observed_fallback_window_count}/"
                f"{result.observed_window_count}"
            )

    payload = {
        "input": str(source),
        "model": args.model,
        "device": args.device,
        "compute_type": args.compute_type,
        "language": args.language or None,
        "beam_size": args.beam_size,
        "clip_start_seconds": args.start,
        "clip_duration_seconds": args.duration,
        "results": [asdict(result) for result in results],
    }
    (output_dir / "benchmark.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_markdown(output_dir / "benchmark.md", results, texts)
    print(f"Results written to {output_dir}")


if __name__ == "__main__":
    main()
