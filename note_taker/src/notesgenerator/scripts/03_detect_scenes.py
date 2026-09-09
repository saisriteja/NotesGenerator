#!/usr/bin/env python3
"""Module 3 — Scene detection with PySceneDetect ContentDetector."""

from __future__ import annotations

import argparse
from pathlib import Path

from common import add_run_dir_arg, die, save_json


def detect_scenes(
    video_path: Path,
    output_path: Path,
    threshold: float,
    min_scene_len_sec: float,
    force: bool,
) -> None:
    if output_path.exists() and not force:
        print(f"Scene list already exists: {output_path}")
        return

    from scenedetect import SceneManager, open_video
    from scenedetect.detectors import ContentDetector

    print(f"Detecting scenes in {video_path} (threshold={threshold})...")
    video = open_video(str(video_path))
    manager = SceneManager()
    manager.add_detector(
        ContentDetector(
            threshold=threshold,
            min_scene_len=int(min_scene_len_sec * video.frame_rate),
        )
    )
    manager.detect_scenes(video)
    scene_list = manager.get_scene_list()

    records = []
    for idx, (start, end) in enumerate(scene_list, start=1):
        records.append(
            {
                "scene_id": idx,
                "start": round(start.get_seconds(), 3),
                "end": round(end.get_seconds(), 3),
            }
        )

    if not records:
        die(
            "Scene detection found 0 scenes. "
            "The video may use an unsupported codec (e.g. AV1) — "
            "re-run step 00 with --force to transcode to h264."
        )

    save_json(output_path, records)
    print(f"Wrote {len(records)} scenes -> {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect slide/scene changes")
    add_run_dir_arg(parser)
    parser.add_argument(
        "--threshold",
        type=float,
        default=35.0,
        help="ContentDetector threshold — higher = fewer, more meaningful slide changes",
    )
    parser.add_argument(
        "--min-scene-len",
        type=float,
        default=4.0,
        help="Minimum scene length in seconds (filters animation micro-cuts)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing scene_list.json",
    )
    args = parser.parse_args()

    video_path = args.run_dir / "raw" / "video.mp4"
    output_path = args.run_dir / "scenes" / "scene_list.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    detect_scenes(
        video_path, output_path, args.threshold, args.min_scene_len, args.force
    )


if __name__ == "__main__":
    main()
