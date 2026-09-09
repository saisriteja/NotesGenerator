#!/usr/bin/env python3
"""Entry point for the lecture report pipeline."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from notesgenerator._paths import data_root, runs_dir, scripts_dir
from notesgenerator.colab_env import configure_colab_env


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run NotesGenerator pipeline from a local video or YouTube URL"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--video", type=Path, help="Path to local lecture video")
    group.add_argument("--url", help="YouTube URL or video id")
    parser.add_argument("--audio", type=Path, default=None, help="Separate audio file")
    parser.add_argument(
        "--output",
        required=True,
        help="Run name under runs/ (e.g. my_lecture)",
    )
    parser.add_argument("--force-all", action="store_true", help="Re-run all steps")
    parser.add_argument("--whisper-model", default="medium")
    parser.add_argument("--vlm-batch-size", type=int, default=8)
    args = parser.parse_args()

    configure_colab_env(
        offline=os.environ.get("HF_HUB_OFFLINE") == "1",
        model=os.environ.get("QWEN_MODEL", "Qwen/Qwen3-VL-4B-Instruct"),
    )

    run_dir = runs_dir() / args.output
    scripts = scripts_dir()

    if args.url:
        cmd = [
            sys.executable,
            str(scripts / "01_download.py"),
            args.url,
            "--run-dir",
            str(run_dir),
        ]
        subprocess.run(cmd, check=True, cwd=str(data_root()))
        video = run_dir / "raw" / "video.mp4"
        skip_prepare = True
    else:
        video = args.video.resolve()
        if not video.exists():
            raise SystemExit(f"Video not found: {video}")
        skip_prepare = False

    cmd = [
        sys.executable,
        str(scripts / "run_lecture.py"),
        "--video",
        str(video),
        "--output",
        args.output,
        "--run-root",
        str(runs_dir()),
        "--whisper-model",
        args.whisper_model,
        "--vlm-batch-size",
        str(args.vlm_batch_size),
    ]
    if args.audio:
        cmd.extend(["--audio", str(args.audio.resolve())])
    if skip_prepare:
        cmd.append("--skip-prepare")
    if args.force_all:
        cmd.append("--force-all")

    subprocess.run(cmd, check=True, cwd=str(data_root()))
    report = run_dir / "report" / "report.md"
    print(f"\nDone. Report: {report}")


if __name__ == "__main__":
    main()
