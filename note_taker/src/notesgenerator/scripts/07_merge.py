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
    aligned_rows: dict[int, dict] = {}
    if aligned_path.exists():
        for row in load_json(aligned_path):
            aligned_rows[row["scene_id"]] = row
    else:
        segments = load_json(run_dir / "transcript" / "whisper_segments.json")
        for scene in scenes:
            start, end = scene["start"], scene["end"]
            matched = [
                {
                    "start": seg["start"],
                    "end": seg["end"],
                    "text": seg["text"].strip(),
                }
                for seg in segments
                if seg.get("text")
                and seg["start"] < end
                and seg["end"] > start
            ]
            matched.sort(key=lambda row: row["start"])
            aligned_rows[scene["scene_id"]] = {
                "transcript": " ".join(row["text"] for row in matched).strip(),
                "segments": matched,
            }

    merged = []
    for scene in scenes:
        sid = scene["scene_id"]
        row = aligned_rows.get(sid, {})
        merged.append(
            {
                "scene_id": sid,
                "start": scene["start"],
                "end": scene["end"],
                "image_path": f"scenes/keyframes/scene_{sid:03d}.png",
                "vlm_description": captions.get(sid, ""),
                "transcript": row.get("transcript", ""),
                "segments": row.get("segments", []),
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
