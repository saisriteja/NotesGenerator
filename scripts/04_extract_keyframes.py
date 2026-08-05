#!/usr/bin/env python3
"""Module 4 — Extract midpoint keyframe PNG for each scene."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from common import add_run_dir_arg, die, load_json


def extract_keyframes(
    video_path: Path,
    scene_list_path: Path,
    keyframes_dir: Path,
) -> None:
    scenes = load_json(scene_list_path)
    if not scenes:
        die(f"No scenes in {scene_list_path}")

    keyframes_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        die(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    saved = 0

    for scene in scenes:
        scene_id = scene["scene_id"]
        out_path = keyframes_dir / f"scene_{scene_id:03d}.png"
        if out_path.exists():
            saved += 1
            continue

        midpoint = (scene["start"] + scene["end"]) / 2.0
        frame_idx = int(midpoint * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok:
            print(f"Warning: failed to read frame for scene {scene_id} at {midpoint:.2f}s")
            continue

        cv2.imwrite(str(out_path), frame)
        saved += 1

    cap.release()
    print(f"Keyframes ready: {saved}/{len(scenes)} in {keyframes_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract scene keyframes")
    add_run_dir_arg(parser)
    args = parser.parse_args()

    run_dir = args.run_dir
    extract_keyframes(
        run_dir / "raw" / "video.mp4",
        run_dir / "scenes" / "scene_list.json",
        run_dir / "scenes" / "keyframes",
    )


if __name__ == "__main__":
    main()
