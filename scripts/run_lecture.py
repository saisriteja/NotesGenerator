#!/usr/bin/env python3
"""One-shot lecture pipeline with per-step timing analysis.

Example:
  python3 scripts/run_lecture.py \\
    --video "/devwork/teja/note_generator/lecture/Stanford CS329A....mp4" \\
    --output lecture_1
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent


@dataclass
class StepResult:
    name: str
    script: str
    started_at: str
    elapsed_sec: float
    status: str = "ok"
    error: str = ""


@dataclass
class PipelineRun:
    video: str
    output: str
    run_dir: str
    started_at: str
    finished_at: str = ""
    total_elapsed_sec: float = 0.0
    steps: list[StepResult] = field(default_factory=list)


def fmt_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, secs = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes}m {secs}s"
    return f"{minutes}m {secs}s"


def run_step(name: str, script: str, args: list[str], env: dict) -> StepResult:
    started = datetime.now(timezone.utc).isoformat()
    t0 = time.perf_counter()
    cmd = [sys.executable, str(SCRIPT_DIR / script), *args]
    print(f"\n{'=' * 60}")
    print(f"▶ {name}")
    print(f"  {' '.join(cmd)}")
    print(f"{'=' * 60}")
    try:
        subprocess.run(cmd, check=True, env=env, cwd=str(ROOT))
        elapsed = time.perf_counter() - t0
        print(f"✓ {name} — {fmt_duration(elapsed)}")
        return StepResult(name, script, started, elapsed, "ok")
    except subprocess.CalledProcessError as exc:
        elapsed = time.perf_counter() - t0
        print(f"✗ {name} failed (exit {exc.returncode}) — {fmt_duration(elapsed)}")
        return StepResult(
            name, script, started, elapsed, "failed", f"exit code {exc.returncode}"
        )


def print_timing_table(run: PipelineRun) -> None:
    print(f"\n{'=' * 60}")
    print("PIPELINE TIME ANALYSIS")
    print(f"{'=' * 60}")
    print(f"Output:  {run.run_dir}")
    print(f"Total:   {fmt_duration(run.total_elapsed_sec)}")
    print(f"{'-' * 60}")
    print(f"{'Step':<28} {'Time':>10}  {'%':>6}")
    print(f"{'-' * 60}")
    total = run.total_elapsed_sec or 1.0
    for step in run.steps:
        pct = 100.0 * step.elapsed_sec / total
        mark = "✓" if step.status == "ok" else "✗"
        print(f"{mark} {step.name:<26} {fmt_duration(step.elapsed_sec):>10}  {pct:5.1f}%")
    print(f"{'-' * 60}")
    print(f"{'TOTAL':<28} {fmt_duration(run.total_elapsed_sec):>10}  100.0%")
    print(f"{'=' * 60}")
    print(f"Report:  {Path(run.run_dir) / 'report' / 'report.md'}")
    print(f"Timing:  {Path(run.run_dir) / 'pipeline_timing.json'}")


def build_env() -> dict:
    env = os.environ.copy()
    env.setdefault("HF_HOME", "/workspace/teja/models/hub")
    env.setdefault("HUGGINGFACE_HUB_CACHE", "/workspace/teja/models/hub")
    env.setdefault("TORCH_HOME", "/workspace/torch-cache")
    env.setdefault("QWEN_MODEL", "Qwen/Qwen3-VL-4B-Instruct")
    env.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    return env


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full lecture pipeline (one shot)")
    parser.add_argument(
        "--video",
        type=Path,
        required=True,
        help="Path to input video (.mp4/.webm). Split yt-dlp streams auto-detected.",
    )
    parser.add_argument(
        "--audio",
        type=Path,
        default=None,
        help="Separate audio file (.webm). Auto-detected when video has no audio track.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output run name (e.g. lecture_1 → runs/lecture_1/)",
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        default=ROOT / "runs",
        help="Parent folder for output runs",
    )
    parser.add_argument("--whisper-model", default="medium")
    parser.add_argument("--scene-threshold", type=float, default=35.0)
    parser.add_argument("--min-scene-len", type=float, default=4.0)
    parser.add_argument("--vlm-batch-size", type=int, default=8)
    parser.add_argument("--window-minutes", type=int, default=8)
    parser.add_argument("--refine-passes", type=int, default=2)
    parser.add_argument("--skip-prepare", action="store_true")
    parser.add_argument("--force-all", action="store_true", help="Force re-run all overwrite steps")
    args = parser.parse_args()

    video = args.video.resolve()
    if not video.exists():
        print(f"Error: video not found: {video}", file=sys.stderr)
        sys.exit(1)

    audio = args.audio.resolve() if args.audio else None
    if audio and not audio.exists():
        print(f"Error: audio not found: {audio}", file=sys.stderr)
        sys.exit(1)

    run_dir = (args.run_root / args.output).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    env = build_env()
    pipeline_started = datetime.now(timezone.utc).isoformat()
    t_pipeline = time.perf_counter()

    run = PipelineRun(
        video=str(video),
        output=args.output,
        run_dir=str(run_dir),
        started_at=pipeline_started,
    )

    print(f"\n🎬 Lecture pipeline")
    print(f"   Input:  {video}")
    if audio:
        print(f"   Audio:  {audio}")
    print(f"   Output: {run_dir}")

    rd = str(run_dir)
    force_flag = ["--force"] if args.force_all else []

    steps_spec = []

    if not args.skip_prepare:
        prepare_args = [str(video), "--run-dir", rd, *force_flag]
        if audio:
            prepare_args.extend(["--audio", str(audio)])
        steps_spec.append(
            (
                "00 Prepare video+audio",
                "00_prepare_local.py",
                prepare_args,
            )
        )

    steps_spec.extend(
        [
            (
                "02 Transcribe (Whisper)",
                "02_transcribe.py",
                ["--run-dir", rd, "--model", args.whisper_model],
            ),
            (
                "03 Detect scenes",
                "03_detect_scenes.py",
                [
                    "--run-dir", rd,
                    "--threshold", str(args.scene_threshold),
                    "--min-scene-len", str(args.min_scene_len),
                    *force_flag,
                ],
            ),
            (
                "04 Extract keyframes",
                "04_extract_keyframes.py",
                ["--run-dir", rd],
            ),
            (
                "05 VLM caption slides",
                "05_caption_slides.py",
                [
                    "--run-dir", rd,
                    "--batch-size", str(args.vlm_batch_size),
                    *(["--force"] if args.force_all else []),
                ],
            ),
            (
                "06 Align transcript",
                "06_align.py",
                ["--run-dir", rd],
            ),
            (
                "07 Merge checkpoint",
                "07_merge.py",
                ["--run-dir", rd],
            ),
            (
                "09 Generate report",
                "09_generate_report.py",
                [
                    "--run-dir", rd,
                    "--force-outline",
                    "--max-model-len", "16384",
                    "--window-minutes", str(args.window_minutes),
                    "--section-tokens", "6000",
                    "--refine-passes", str(args.refine_passes),
                    "--refine-tokens", "4096",
                    "--gpu-mem", "0.82",
                ],
            ),
        ]
    )

    for name, script, step_args in steps_spec:
        result = run_step(name, script, step_args, env)
        run.steps.append(result)
        if result.status != "ok":
            run.total_elapsed_sec = time.perf_counter() - t_pipeline
            run.finished_at = datetime.now(timezone.utc).isoformat()
            timing_path = run_dir / "pipeline_timing.json"
            timing_path.write_text(
                json.dumps(
                    {
                        **asdict(run),
                        "steps": [asdict(s) for s in run.steps],
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            print_timing_table(run)
            sys.exit(1)

    run.total_elapsed_sec = time.perf_counter() - t_pipeline
    run.finished_at = datetime.now(timezone.utc).isoformat()

    timing_path = run_dir / "pipeline_timing.json"
    timing_path.write_text(
        json.dumps(
            {**asdict(run), "steps": [asdict(s) for s in run.steps]},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print_timing_table(run)


if __name__ == "__main__":
    main()
