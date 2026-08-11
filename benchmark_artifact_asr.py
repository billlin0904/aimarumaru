"""A/B benchmark the production local-ASR pipeline on English/Japanese/Korean artifacts."""

from __future__ import annotations

import argparse
import asyncio
import gc
import json
import shutil
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).with_name(".env"))
except ImportError:
    pass

from auto2lrc import Auto2Lrc
from youtube_live import transcribe_audio_stream


@dataclass(frozen=True)
class ArtifactCase:
    language: str
    filename: str
    label: str


ARTIFACT_CASES = {
    "en": ArtifactCase(
        "en",
        "Anthropic, OpenAI Should Not Be Allowed to IPO, Says Ed Zitron_1080p.mp4",
        "English interview",
    ),
    "ja": ArtifactCase("ja", "日文 Netflix 訪談.mp4", "Japanese Netflix interview"),
    "ko": ArtifactCase("ko", "韓文頒獎.mp4", "Korean awards"),
}


@dataclass
class BenchmarkResult:
    case: str
    label: str
    language: str
    model: str
    source: str
    audio_seconds: float
    elapsed_seconds: float
    speed_x: float
    realtime_factor: float
    segments: int
    characters: int
    word_timestamps: int
    low_confidence_spans: int
    srt_file: str
    json_file: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare faster-whisper models through Aimarumaru's production "
            "30-second streaming/VAD/timestamp pipeline."
        )
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "kotobamaru" / "artifacts",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmark_results") / "artifact_asr_ab",
    )
    parser.add_argument("--models", nargs="+", default=["large-v3", "turbo"])
    parser.add_argument("--cases", nargs="+", choices=ARTIFACT_CASES, default=list(ARTIFACT_CASES))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--compute-type", default="float16")
    parser.add_argument(
        "--duration",
        type=float,
        default=60,
        help="Seconds from each video; 0 runs the complete videos.",
    )
    parser.add_argument(
        "--transcription-mode",
        choices=["accurate", "fast"],
        default="accurate",
    )
    return parser.parse_args()


def format_srt_timestamp(seconds: float) -> str:
    milliseconds = max(0, int(round(float(seconds) * 1000)))
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, milliseconds = divmod(remainder, 1000)
    return f"{hours:02}:{minutes:02}:{secs:02},{milliseconds:03}"


def write_srt(path: Path, segments: list[dict[str, Any]]) -> None:
    blocks = []
    for index, segment in enumerate(segments, start=1):
        blocks.append(
            "\n".join(
                [
                    str(index),
                    f"{format_srt_timestamp(segment['start'])} --> "
                    f"{format_srt_timestamp(segment['end'])}",
                    str(segment["text"]).strip(),
                ]
            )
        )
    path.write_text("\n\n".join(blocks) + ("\n" if blocks else ""), encoding="utf-8")


def normalized_transcript(value: str) -> str:
    return "".join(str(value).casefold().split())


def transcript_similarity(first: str, second: str) -> float:
    return SequenceMatcher(
        None,
        normalized_transcript(first),
        normalized_transcript(second),
        autojunk=False,
    ).ratio()


def transcript_from_result(result: BenchmarkResult) -> str:
    payload = json.loads(Path(result.json_file).read_text(encoding="utf-8"))
    return "\n".join(
        str(segment.get("text", "")) for segment in payload.get("segments", [])
    )


def create_clip(source: Path, output: Path, duration: float) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg is required for --duration clips")
    subprocess.run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(source),
            "-t",
            str(duration),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            str(output),
        ],
        check=True,
    )


def warm_up_model(auto2lrc: Auto2Lrc, source: Path, language: str) -> None:
    model = auto2lrc.get_model()
    segments, _ = model.transcribe(
        str(source),
        language=language,
        beam_size=1,
        temperature=0.0,
        condition_on_previous_text=False,
        vad_filter=False,
        clip_timestamps="0,10",
    )
    list(segments)


async def run_pipeline_case(
    auto2lrc: Auto2Lrc,
    case_key: str,
    case: ArtifactCase,
    source: Path,
    output_dir: Path,
    transcription_mode: str,
) -> BenchmarkResult:
    event_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
    chunk_metrics: list[dict[str, Any]] = []
    loop = asyncio.get_running_loop()
    result = await asyncio.to_thread(
        transcribe_audio_stream,
        auto2lrc,
        source,
        case.language,
        loop,
        event_queue,
        f"benchmark-{case_key}-{auto2lrc.model_name}",
        None,
        True,
        None,
        transcription_mode,
        "local",
        None,
        chunk_metrics.append,
    )

    segments: list[dict[str, Any]] = []
    while not event_queue.empty():
        event = event_queue.get_nowait()
        if event.get("event") == "segment":
            segments.append(event["data"])

    safe_model = auto2lrc.model_name.replace("/", "_").replace(":", "_")
    stem = f"{case_key}-{safe_model}"
    srt_path = output_dir / f"{stem}.srt"
    json_path = output_dir / f"{stem}.json"
    write_srt(srt_path, segments)
    json_path.write_text(
        json.dumps(
            {"summary": result, "segments": segments, "chunks": chunk_metrics},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    elapsed = float(result["transcription_elapsed_seconds"])
    audio_seconds = float(result["audio_duration_seconds"])
    return BenchmarkResult(
        case=case_key,
        label=case.label,
        language=case.language,
        model=auto2lrc.model_name,
        source=str(source),
        audio_seconds=audio_seconds,
        elapsed_seconds=elapsed,
        speed_x=float(result.get("processing_speed_x") or 0),
        realtime_factor=elapsed / audio_seconds if audio_seconds else 0,
        segments=len(segments),
        characters=sum(len(str(segment.get("text", ""))) for segment in segments),
        word_timestamps=sum(len(segment.get("words", [])) for segment in segments),
        low_confidence_spans=sum(
            len(segment.get("low_confidence_spans", [])) for segment in segments
        ),
        srt_file=str(srt_path),
        json_file=str(json_path),
    )


def comparison_rows(
    results: list[BenchmarkResult], baseline_model: str, candidate_model: str
) -> list[dict[str, Any]]:
    indexed = {(row.case, row.model): row for row in results}
    rows = []
    for case_key in ARTIFACT_CASES:
        baseline = indexed.get((case_key, baseline_model))
        candidate = indexed.get((case_key, candidate_model))
        if not baseline or not candidate:
            continue
        baseline_text = transcript_from_result(baseline)
        candidate_text = transcript_from_result(candidate)
        rows.append(
            {
                "case": case_key,
                "baseline_model": baseline_model,
                "candidate_model": candidate_model,
                "candidate_speedup": round(
                    baseline.elapsed_seconds / candidate.elapsed_seconds, 3
                ) if candidate.elapsed_seconds else None,
                "segment_delta": candidate.segments - baseline.segments,
                "character_delta": candidate.characters - baseline.characters,
                "transcript_similarity": round(
                    transcript_similarity(baseline_text, candidate_text), 4
                ),
            }
        )
    return rows


def write_report(
    output_dir: Path,
    args: argparse.Namespace,
    results: list[BenchmarkResult],
) -> None:
    comparisons = (
        comparison_rows(results, args.models[0], args.models[1])
        if len(args.models) >= 2
        else []
    )
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "artifacts_dir": str(args.artifacts_dir.resolve()),
        "duration_seconds": args.duration,
        "transcription_mode": args.transcription_mode,
        "results": [asdict(row) for row in results],
        "comparisons": comparisons,
        "note": "Similarity is model-to-model agreement, not accuracy against human ground truth.",
    }
    (output_dir / "benchmark.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    lines = [
        "# English/Japanese/Korean ASR benchmark",
        "",
        f"- Clip duration: {'full video' if args.duration <= 0 else f'{args.duration:g} seconds'}",
        f"- Pipeline mode: `{args.transcription_mode}`",
        "- Model loading and a 10-second warm-up are excluded from timed results.",
        "",
        "| Language | Model | Audio | Elapsed | Speed | Segments | Characters | Words |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in results:
        lines.append(
            f"| {row.case} | `{row.model}` | {row.audio_seconds:.1f}s | "
            f"{row.elapsed_seconds:.2f}s | {row.speed_x:.2f}x | {row.segments} | "
            f"{row.characters} | {row.word_timestamps} |"
        )
    if comparisons:
        lines.extend(
            [
                "",
                f"## `{args.models[1]}` compared with `{args.models[0]}`",
                "",
                "| Language | Speed-up | Segment delta | Character delta | Transcript agreement |",
                "| --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in comparisons:
            lines.append(
                f"| {row['case']} | {row['candidate_speedup']:.2f}x | "
                f"{row['segment_delta']:+d} | {row['character_delta']:+d} | "
                f"{row['transcript_similarity']:.1%} |"
            )
        lines.extend(
            [
                "",
                "> Transcript agreement only measures how similar the two model outputs are. "
                "Use the generated SRT files for human accuracy review.",
            ]
        )
    (output_dir / "benchmark.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


async def async_main(args: argparse.Namespace) -> None:
    artifacts_dir = args.artifacts_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cases = [(key, ARTIFACT_CASES[key]) for key in args.cases]
    for _, case in cases:
        source = artifacts_dir / case.filename
        if not source.is_file():
            raise FileNotFoundError(source)

    results: list[BenchmarkResult] = []
    with tempfile.TemporaryDirectory(prefix="aimarumaru-asr-benchmark-") as temp_dir:
        prepared: dict[str, Path] = {}
        for case_key, case in cases:
            source = artifacts_dir / case.filename
            if args.duration > 0:
                clip = Path(temp_dir) / f"{case_key}.wav"
                create_clip(source, clip, args.duration)
                prepared[case_key] = clip
            else:
                prepared[case_key] = source

        for model_name in args.models:
            auto2lrc = Auto2Lrc(
                model_name=model_name,
                device=args.device,
                compute_type=args.compute_type,
            )
            first_key, first_case = cases[0]
            print(f"[benchmark] loading and warming up {model_name}", flush=True)
            await asyncio.to_thread(
                warm_up_model, auto2lrc, prepared[first_key], first_case.language
            )
            for case_key, case in cases:
                print(f"[benchmark] {model_name}: {case.label}", flush=True)
                result = await run_pipeline_case(
                    auto2lrc,
                    case_key,
                    case,
                    prepared[case_key],
                    output_dir,
                    args.transcription_mode,
                )
                results.append(result)
                print(
                    f"  {result.elapsed_seconds:.2f}s, {result.speed_x:.2f}x, "
                    f"{result.segments} segments",
                    flush=True,
                )
            auto2lrc.model = None
            gc.collect()
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass

    write_report(output_dir, args, results)
    print(f"[benchmark] report: {output_dir / 'benchmark.md'}", flush=True)


def main() -> None:
    asyncio.run(async_main(parse_args()))


if __name__ == "__main__":
    main()
