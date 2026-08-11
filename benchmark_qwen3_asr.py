"""Benchmark Qwen3-ASR + ForcedAligner on the shared artifact videos.

Run this with the isolated qwen3-asr environment.  The output JSON/SRT can be
compared with benchmark_results/artifact_asr_ab from benchmark_artifact_asr.py.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from qwen_asr import Qwen3ASRModel


CASES = {
    "en": ("Anthropic, OpenAI Should Not Be Allowed to IPO, Says Ed Zitron_1080p.mp4", "English interview"),
    "ja": ("日文 Netflix 訪談.mp4", "Japanese Netflix interview"),
    "ko": ("韓文頒獎.mp4", "Korean awards"),
}
LANGUAGE_NAMES = {"en": "English", "ja": "Japanese", "ko": "Korean"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts-dir", type=Path, default=Path(__file__).resolve().parent.parent / "kotobamaru" / "artifacts")
    parser.add_argument("--output-dir", type=Path, default=Path("benchmark_results") / "artifact_asr_qwen3")
    parser.add_argument("--duration", type=float, default=60)
    parser.add_argument("--model", default="Qwen/Qwen3-ASR-1.7B")
    parser.add_argument("--aligner", default="Qwen/Qwen3-ForcedAligner-0.6B")
    parser.add_argument("--cases", nargs="+", choices=CASES, default=list(CASES))
    return parser.parse_args()


def format_timestamp(seconds: float) -> str:
    ms = max(0, int(round(float(seconds) * 1000)))
    hours, rem = divmod(ms, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    secs, ms = divmod(rem, 1000)
    return f"{hours:02}:{minutes:02}:{secs:02},{ms:03}"


def make_clip(source: Path, target: Path, duration: float) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg is required")
    command = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-i", str(source)]
    if duration > 0:
        command += ["-t", str(duration)]
    command += ["-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(target)]
    subprocess.run(command, check=True)


def timestamp_fields(item: object) -> tuple[float, float, str]:
    text = str(getattr(item, "text", "")).strip()
    start = float(getattr(item, "start_time", getattr(item, "start", 0.0)))
    end = float(getattr(item, "end_time", getattr(item, "end", start)))
    return start, end, text


def result_segments(result: object) -> list[dict[str, object]]:
    stamps = getattr(result, "time_stamps", None) or []
    rows = []
    for item in stamps:
        start, end, text = timestamp_fields(item)
        if text:
            rows.append({"start": start, "end": end, "text": text})
    if not rows:
        text = str(getattr(result, "text", "")).strip()
        if text:
            rows.append({"start": 0.0, "end": 0.0, "text": text})
    return rows


def write_srt(path: Path, rows: list[dict[str, object]]) -> None:
    blocks = []
    for index, row in enumerate(rows, start=1):
        blocks.append(
            f"{index}\n{format_timestamp(float(row['start']))} --> "
            f"{format_timestamp(float(row['end']))}\n{row['text']}"
        )
    path.write_text("\n\n".join(blocks) + ("\n" if blocks else ""), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for key in args.cases:
        source = args.artifacts_dir / CASES[key][0]
        if not source.is_file():
            raise FileNotFoundError(source)

    print(f"[qwen3] loading {args.model} + {args.aligner}", flush=True)
    model = Qwen3ASRModel.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        device_map="cuda:0",
        max_inference_batch_size=1,
        max_new_tokens=512,
        forced_aligner=args.aligner,
        forced_aligner_kwargs={"dtype": torch.bfloat16, "device_map": "cuda:0"},
    )
    results = []
    with tempfile.TemporaryDirectory(prefix="qwen3-asr-benchmark-") as temp_dir:
        warmup_source = args.artifacts_dir / CASES[args.cases[0]][0]
        warmup_path = Path(temp_dir) / "warmup.wav"
        make_clip(warmup_source, warmup_path, 10)
        print("[qwen3] warming up (excluded from timings)", flush=True)
        model.transcribe(
            audio=str(warmup_path),
            language=LANGUAGE_NAMES[args.cases[0]],
            return_time_stamps=False,
        )
        for key in args.cases:
            source = args.artifacts_dir / CASES[key][0]
            input_path = source
            if args.duration > 0:
                input_path = Path(temp_dir) / f"{key}.wav"
                make_clip(source, input_path, args.duration)
            print(f"[qwen3] {key}: {CASES[key][1]}", flush=True)
            started = time.perf_counter()
            output = model.transcribe(
                audio=str(input_path),
                language=LANGUAGE_NAMES[key],
                return_time_stamps=True,
            )[0]
            elapsed = time.perf_counter() - started
            rows = result_segments(output)
            stem = f"{key}-qwen3-asr-1.7b"
            srt_path = args.output_dir / f"{stem}.srt"
            json_path = args.output_dir / f"{stem}.json"
            write_srt(srt_path, rows)
            payload = {
                "case": key,
                "label": CASES[key][1],
                "model": args.model,
                "aligner": args.aligner,
                "language": getattr(output, "language", LANGUAGE_NAMES[key]),
                "text": getattr(output, "text", ""),
                "segments": rows,
                "audio_seconds": args.duration,
                "elapsed_seconds": round(elapsed, 3),
                "speed_x": round(args.duration / elapsed, 3) if args.duration > 0 else None,
                "srt_file": str(srt_path),
            }
            json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            results.append(payload)
            print(f"  {elapsed:.2f}s, {len(rows)} timestamp rows", flush=True)

    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "duration_seconds": args.duration,
        "model": args.model,
        "aligner": args.aligner,
        "results": results,
    }
    (args.output_dir / "benchmark.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# Qwen3-ASR artifact benchmark", "", "- Model loading and a 10-second warm-up are excluded from timings.", "", "| Language | Elapsed | Speed | Timestamp rows |", "| --- | ---: | ---: | ---: |"]
    for row in results:
        lines.append(f"| {row['case']} | {row['elapsed_seconds']:.3f}s | {row['speed_x']:.2f}x | {len(row['segments'])} |")
    (args.output_dir / "benchmark.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[qwen3] report: {args.output_dir / 'benchmark.md'}", flush=True)


if __name__ == "__main__":
    main()
