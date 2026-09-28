#!/usr/bin/env python3
"""Run NotesGenerator on lectures that do not yet have output under notes/."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from notesgenerator._paths import data_root
from notesgenerator.colab_env import configure_colab_env
from notesgenerator.notes_layout import (
    build_video_map,
    has_notes_output,
    lectures_dir,
    notes_dir,
    organize_output,
    output_name_for,
)


def _log(msg: str) -> None:
    print(msg, flush=True)


def _find_notesgenerator() -> list[str]:
    repo_root = data_root()
    venv_bin = repo_root / ".venv" / "bin" / "notesgenerator"
    if venv_bin.is_file() and os.access(venv_bin, os.X_OK):
        return [str(venv_bin)]
    return ["notesgenerator"]


def run_one(video: Path, extra_args: list[str], env: dict[str, str]) -> bool:
    output = output_name_for(video)
    cmd = [
        *_find_notesgenerator(),
        "--video",
        str(video.resolve()),
        "--output",
        output,
        "--data-dir",
        str(data_root()),
        *extra_args,
    ]

    vdir = video.parent
    papers = sorted(vdir.glob("*.pdf"))
    if papers:
        cmd.extend(["--papers", *[str(p.resolve()) for p in papers]])

    _log("=" * 60)
    _log(f"RUN: {output}")
    _log(f"  Video: {video}")
    _log(f"  Papers: {len(papers)}")
    _log("=" * 60)

    try:
        subprocess.run(cmd, check=True, cwd=str(data_root()), env=env)
    except subprocess.CalledProcessError:
        _log(f"FAILED: {output} ({video})")
        return False

    dest = organize_output(output)
    if dest:
        _log(f"Organized → {dest.relative_to(data_root())}")
    else:
        _log(f"WARNING: no report zip found for {output}")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run NotesGenerator on lectures missing from notes/"
    )
    parser.add_argument(
        "--course",
        action="append",
        default=None,
        help="Only process lectures under this top-level course folder (repeatable)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List pending lectures without running the pipeline",
    )
    parser.add_argument(
        "--no-zip",
        action="store_true",
        help="Forward --no-zip to notesgenerator",
    )
    parser.add_argument(
        "--no-compress",
        action="store_true",
        help="Forward --no-compress to notesgenerator",
    )
    args = parser.parse_args()

    configure_colab_env(data_path=data_root())
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    lectures = lectures_dir()
    if not lectures.is_dir():
        raise SystemExit(f"Lectures directory not found: {lectures}")

    log_dir = data_root() / "runs" / "_batch_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    batch_log = log_dir / f"missing_lectures_{datetime.now():%Y%m%d_%H%M%S}.log"

    video_map = build_video_map()
    courses = set(args.course or [])
    videos = sorted(lectures.rglob("*.mp4"))

    pending: list[Path] = []
    skipped = 0
    for video in videos:
        rel = video.relative_to(lectures)
        if courses and rel.parts[0] not in courses:
            continue
        if has_notes_output(video, video_map):
            skipped += 1
            continue
        pending.append(video)

    _log(f"Batch log: {batch_log}")
    _log(f"CUDA_VISIBLE_DEVICES={env.get('CUDA_VISIBLE_DEVICES', '(unset)')}")
    _log(f"Repo: {data_root()}")
    _log(f"Notes: {notes_dir()}")
    _log(f"Total videos: {len(videos)}  skipped (in notes): {skipped}  pending: {len(pending)}")

    if args.dry_run:
        for video in pending:
            _log(f"  pending: {output_name_for(video)} <- {video.relative_to(data_root())}")
        return

    extra_args: list[str] = []
    if args.no_zip:
        extra_args.append("--no-zip")
    if args.no_compress:
        extra_args.append("--no-compress")

    ran = 0
    failed = 0
    with batch_log.open("a", encoding="utf-8") as log_file:
        for video in pending:
            log_file.write(
                f"Processing {output_name_for(video)} ({video.relative_to(data_root())})\n"
            )
            log_file.flush()
            if run_one(video, extra_args, env):
                ran += 1
            else:
                failed += 1
        log_file.write(
            f"Done. skipped={skipped} ran={ran} failed={failed} pending_was={len(pending)}\n"
        )

    _log(f"Done. skipped={skipped} ran={ran} failed={failed} pending_was={len(pending)}")
    _log(f"Log: {batch_log}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
