#!/usr/bin/env python3
"""Entry point for the lecture report pipeline."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

from notesgenerator._paths import data_root, runs_dir, scripts_dir
from notesgenerator.colab_env import configure_colab_env


def _log(msg: str) -> None:
    print(msg, flush=True)


def _check_imagemagick() -> None:
    if shutil.which("convert") is None:
        _log(
            "WARNING: ImageMagick not found (needed for image compression).\n"
            "  Install with: apt-get install -y imagemagick"
        )
    else:
        _log("ImageMagick: OK (convert found)")


def _zip_report(run_dir: Path, output_name: str) -> Path:
    report_dir = run_dir / "report"
    if not report_dir.is_dir():
        _log(f"WARNING: no report/ to zip at {report_dir}")
        return run_dir

    zip_path = data_root() / f"{output_name}_report"
    archive = shutil.make_archive(str(zip_path), "zip", report_dir)
    _log(f"Zipped report → {archive}")
    return Path(archive)


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
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Root for runs/ and models/ (default: /content on Colab, else cwd)",
    )
    parser.add_argument("--force-all", action="store_true", help="Re-run all steps")
    parser.add_argument("--whisper-model", default="medium")
    parser.add_argument("--vlm-batch-size", type=int, default=8)
    parser.add_argument(
        "--no-zip",
        action="store_true",
        help="Skip creating {output}_report.zip at the end",
    )
    parser.add_argument(
        "--no-compress",
        action="store_true",
        help="Skip ImageMagick compression steps",
    )
    args = parser.parse_args()

    configure_colab_env(
        offline=os.environ.get("HF_HUB_OFFLINE") == "1",
        model=os.environ.get("QWEN_MODEL", "Qwen/Qwen3-VL-4B-Instruct"),
        data_path=args.data_dir,
    )

    run_dir = runs_dir() / args.output
    scripts = scripts_dir()
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    _log("=" * 60)
    _log("NotesGenerator pipeline")
    _log("=" * 60)
    if args.video:
        _log(f"Video:  {args.video.resolve()}")
    else:
        _log(f"URL:    {args.url}")
    _log(f"Output: {run_dir}")
    _log(f"Data:   {data_root()}")
    _check_imagemagick()
    _log("=" * 60)

    if args.url:
        cmd = [
            sys.executable,
            "-u",
            str(scripts / "01_download.py"),
            args.url,
            "--run-dir",
            str(run_dir),
        ]
        _log("\n▶ Downloading video ...")
        subprocess.run(cmd, check=True, cwd=str(data_root()), env=env)
        video = run_dir / "raw" / "video.mp4"
        skip_prepare = True
    else:
        video = args.video.resolve()
        if not video.exists():
            raise SystemExit(f"Video not found: {video}")
        skip_prepare = False

    cmd = [
        sys.executable,
        "-u",
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
    if args.no_compress:
        cmd.append("--no-compress")

    subprocess.run(cmd, check=True, cwd=str(data_root()), env=env)

    report = run_dir / "report" / "report.md"
    _log("\n" + "=" * 60)
    _log("PIPELINE COMPLETE")
    _log("=" * 60)
    _log(f"Report:  {report}")
    _log(f"Run dir: {run_dir}")

    if not args.no_zip:
        _zip_report(run_dir, args.output)

    _log("=" * 60)


if __name__ == "__main__":
    main()
