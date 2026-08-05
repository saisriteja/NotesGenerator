#!/usr/bin/env python3
"""Module 7 — Merge scenes, captions, and aligned transcript (checkpoint)."""

from __future__ import annotations

import argparse
from pathlib import Path

from common import add_run_dir_arg, load_json, save_json


def merge_run(run_dir: Path, output_name: str = "scenes_merged.json") -> Path:
    scenes = load_json(run_dir / "scenes" / "scene_list.json")
    captions = {
        row["scene_id"]: row["vlm_description"]
        for row in load_json(run_dir / "captions" / "vlm_captions.json")
    }

    aligned_path = run_dir / "merged" / "aligned_scenes.json"
    if aligned_path.exists():
        aligned = {row["scene_id"]: row["transcript"] for row in load_json(aligned_path)}
    else:
        segments = load_json(run_dir / "transcript" / "whisper_segments.json")
        aligned = {}
        for scene in scenes:
            start, end = scene["start"], scene["end"]
            texts = [
                seg["text"]
                for seg in segments
                if start <= seg["start"] < end and seg["text"]
            ]
            aligned[scene["scene_id"]] = " ".join(texts).strip()

    merged = []
    for scene in scenes:
        sid = scene["scene_id"]
        merged.append(
            {
                "scene_id": sid,
                "start": scene["start"],
                "end": scene["end"],
                "image_path": f"scenes/keyframes/scene_{sid:03d}.png",
                "vlm_description": captions.get(sid, ""),
                "transcript": aligned.get(sid, ""),
            }
        )

    out_path = run_dir / "merged" / output_name
    save_json(out_path, merged)
    print(f"Merged {len(merged)} scenes -> {out_path}")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge pipeline outputs")
    add_run_dir_arg(parser)
    args = parser.parse_args()
    merge_run(args.run_dir)


if __name__ == "__main__":
    main()
