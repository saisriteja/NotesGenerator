#!/usr/bin/env python3
"""Prepare run folder from a local video file (copy mp4 + extract 16 kHz audio).

Supports single muxed files and yt-dlp split streams (video-only .mp4/.webm +
audio-only .webm in the same folder, e.g. gpumode downloads).
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from common import add_run_dir_arg, die, ensure_run_layout, resolve_media_inputs, stream_codecs


def video_codec(path: Path) -> str:
    return stream_codecs(path)["video"] or ""


def transcode_to_h264(src: Path, dst: Path) -> None:
    streams = stream_codecs(src)
    cmd = [
        "ffmpeg", "-y", "-i", str(src),
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
    ]
    if streams["audio"]:
        cmd.extend(["-c:a", "copy"])
    else:
        cmd.append("-an")
    cmd.append(str(dst))
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True, capture_output=True)


def extract_audio(src: Path, dst: Path) -> None:
    cmd = [
        "ffmpeg", "-y", "-i", str(src),
        "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
        str(dst),
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True, capture_output=True)


def prepare(
    video_path: Path,
    run_dir: Path,
    force: bool,
    audio_path: Path | None = None,
) -> None:
    video_src, audio_src = resolve_media_inputs(video_path, audio_path)

    folders = ensure_run_layout(run_dir)
    raw = folders["raw"]
    video_out = raw / "video.mp4"
    audio_out = raw / "audio.wav"

    if video_out.exists() and audio_out.exists() and not force:
        codec = video_codec(video_out)
        if codec == "h264":
            print(f"Already prepared: {video_out}, {audio_out}")
            return
        print(f"Existing video is {codec}; re-transcoding to h264 (--force not required)")

    codec = video_codec(video_src)
    if codec == "h264":
        print(f"Copying {video_src} -> {video_out}")
        shutil.copy2(video_src, video_out)
    else:
        print(f"Transcoding {codec} -> h264: {video_src} -> {video_out}")
        transcode_to_h264(video_src, video_out)

    audio_input = audio_src or video_out
    if audio_src:
        print(f"Extracting audio from separate source: {audio_src}")
    extract_audio(audio_input, audio_out)

    print(f"Ready: {video_out} ({video_out.stat().st_size // 1024 // 1024} MB)")
    print(f"Ready: {audio_out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare local lecture video")
    parser.add_argument(
        "video",
        type=Path,
        help="Path to input video (.mp4/.webm) or audio when paired with video",
    )
    parser.add_argument(
        "--audio",
        type=Path,
        default=None,
        help="Separate audio file (.webm/.mp4). Auto-detected for yt-dlp split streams.",
    )
    add_run_dir_arg(parser)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    prepare(args.video.resolve(), args.run_dir, args.force, args.audio)


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        die(f"ffmpeg failed with exit code {exc.returncode}", exc.returncode)
