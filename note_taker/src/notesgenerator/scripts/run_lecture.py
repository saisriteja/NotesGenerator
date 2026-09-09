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
ROOT = Path(os.environ.get("NOTE_TAKER_ROOT", Path.cwd()))

try:
    from notesgenerator.gpu_defaults import report_vllm_defaults
except ImportError:
    _pkg_root = SCRIPT_DIR.parent
    if str(_pkg_root) not in sys.path:
        sys.path.insert(0, str(_pkg_root))
    from gpu_defaults import report_vllm_defaults  # type: ignore[import-untyped,no-redef]


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
    cmd = [sys.executable, "-u", str(SCRIPT_DIR / script), *args]
    print(f"\n{'=' * 60}", flush=True)
    print(f"▶ {name}", flush=True)
    print(f"  {' '.join(cmd)}", flush=True)
    print(f"{'=' * 60}", flush=True)
    try:
        subprocess.run(cmd, check=True, env=env, cwd=str(ROOT))
        elapsed = time.perf_counter() - t0
        print(f"✓ {name} — {fmt_duration(elapsed)}", flush=True)
        return StepResult(name, script, started, elapsed, "ok")
    except subprocess.CalledProcessError as exc:
        elapsed = time.perf_counter() - t0
        print(f"✗ {name} failed (exit {exc.returncode}) — {fmt_duration(elapsed)}", flush=True)
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
    models = ROOT / "models"
    env.setdefault("NOTE_TAKER_ROOT", str(ROOT))
    env.setdefault("HF_HOME", str(models / "hub"))
    env.setdefault("HUGGINGFACE_HUB_CACHE", str(models / "hub"))
    env.setdefault("TORCH_HOME", str(models / "torch"))
    env.setdefault("QWEN_MODEL", "Qwen/Qwen3-VL-4B-Instruct")
    env.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    env["PYTHONUNBUFFERED"] = "1"
    scripts = str(SCRIPT_DIR)
    env["PYTHONPATH"] = (
        scripts if "PYTHONPATH" not in env else f"{scripts}{os.pathsep}{env['PYTHONPATH']}"
    )
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
    parser.add_argument("--verify-passes", type=int, default=2)
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=None,
        help="vLLM context for report step (default: auto from GPU — 8192 on T4, 16384 on A100+)",
    )
    parser.add_argument(
        "--gpu-mem",
        type=float,
        default=None,
        help="vLLM gpu_memory_utilization for report step (default: auto — 0.90 on T4, 0.82 on larger GPUs)",
    )
    parser.add_argument("--skip-prepare", action="store_true")
    parser.add_argument("--force-all", action="store_true", help="Force re-run all overwrite steps")
    parser.add_argument(
        "--no-compress",
        action="store_true",
        help="Skip ImageMagick compression of keyframes and report images",
    )
    parser.add_argument(
        "--compress-strategy",
        choices=["optimized-png", "palette", "jpeg", "webp"],
        default="palette",
        help="ImageMagick compression strategy (default: palette — good space savings)",
    )
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

    auto_max_len, auto_gpu_mem = report_vllm_defaults()
    report_max_model_len = args.max_model_len if args.max_model_len is not None else auto_max_len
    report_gpu_mem = args.gpu_mem if args.gpu_mem is not None else auto_gpu_mem

    env = build_env()
    pipeline_started = datetime.now(timezone.utc).isoformat()
    t_pipeline = time.perf_counter()

    run = PipelineRun(
        video=str(video),
        output=args.output,
        run_dir=str(run_dir),
        started_at=pipeline_started,
    )

    print(f"\n🎬 Lecture pipeline", flush=True)
    print(f"   Input:  {video}", flush=True)
    if audio:
        print(f"   Audio:  {audio}", flush=True)
    print(f"   Output: {run_dir}", flush=True)
    if not args.no_compress:
        print(f"   Compress: {args.compress_strategy} at end (step 10)", flush=True)
    print(
        f"   Report vLLM: max_model_len={report_max_model_len}, gpu_mem={report_gpu_mem}",
        flush=True,
    )

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
                    "--max-model-len", str(report_max_model_len),
                    "--window-minutes", str(args.window_minutes),
                    "--section-tokens", "6000",
                    "--refine-passes", str(args.refine_passes),
                    "--refine-tokens", "4096",
                    "--verify-passes", str(args.verify_passes),
                    "--verify-tokens", "4096",
                    "--gpu-mem", str(report_gpu_mem),
                ],
            ),
            *(
                [
                    (
                        "10 Compress all images (ImageMagick)",
                        "compress_keyframes.py",
                        [
                            "--run-dir", rd,
                            "--include-report",
                            "--attachments",
                            "--strategy", args.compress_strategy,
                            "--in-place",
                        ],
                    ),
                ]
                if not args.no_compress
                else []
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
