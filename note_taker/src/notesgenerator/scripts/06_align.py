#!/usr/bin/env python3
"""Module 6 — Align Whisper transcript segments to scene windows."""

from __future__ import annotations

import argparse
from pathlib import Path

from common import add_run_dir_arg, load_json, save_json


def align_segments(scenes: list[dict], segments: list[dict]) -> list[dict]:
    aligned = []
    for scene in scenes:
        start, end = scene["start"], scene["end"]
        texts = [
            seg["text"]
            for seg in segments
            if start <= seg["start"] < end and seg["text"]
        ]
        aligned.append(
            {
                "scene_id": scene["scene_id"],
                "start": scene["start"],
                "end": scene["end"],
                "transcript": " ".join(texts).strip(),
            }
        )
    return aligned


def main() -> None:
    parser = argparse.ArgumentParser(description="Align transcript to scenes")
    add_run_dir_arg(parser)
    args = parser.parse_args()

    run_dir = args.run_dir
    scenes = load_json(run_dir / "scenes" / "scene_list.json")
    segments = load_json(run_dir / "transcript" / "whisper_segments.json")

    aligned = align_segments(scenes, segments)
    out_path = run_dir / "merged" / "aligned_scenes.json"
    save_json(out_path, aligned)
    print(f"Aligned {len(aligned)} scenes -> {out_path}")


if __name__ == "__main__":
    main()
