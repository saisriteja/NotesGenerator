#!/usr/bin/env python3
"""Module 1 — Download YouTube video and extract 16 kHz mono audio."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from common import default_run_dir, die, ensure_run_layout, extract_video_id


def download(
    url: str,
    run_dir: Path,
    cookies_from_browser: str | None,
    cookies_file: Path | None,
) -> None:
    folders = ensure_run_layout(run_dir)
    raw = folders["raw"]
    video_path = raw / "video.mp4"
    audio_path = raw / "audio.wav"

    if video_path.exists() and audio_path.exists():
        print(f"Already downloaded: {video_path}, {audio_path}")
        return

    cmd = [
        "yt-dlp",
        "--no-playlist",
        "--extractor-args",
        "youtube:player_client=android,web",
        "-f",
        "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "--merge-output-format",
        "mp4",
        "-o",
        str(video_path),
        url,
    ]
    if cookies_from_browser:
        cmd[1:1] = ["--cookies-from-browser", cookies_from_browser]
    elif cookies_file:
        cmd[1:1] = ["--cookies", str(cookies_file)]

    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)

    if not video_path.exists():
        # yt-dlp may append format suffix
        candidates = sorted(raw.glob("video*.mp4"))
        if not candidates:
            die(f"Download finished but no mp4 found in {raw}")
        candidates[0].rename(video_path)

    ffmpeg_cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        "16000",
        "-ac",
        "1",
        str(audio_path),
    ]
    print("Running:", " ".join(ffmpeg_cmd))
    subprocess.run(ffmpeg_cmd, check=True, capture_output=True)
    print(f"Saved {video_path} and {audio_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download lecture video + audio")
    parser.add_argument("url", help="YouTube URL or video id")
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Output run directory (default: runs/run_<video_id>)",
    )
    parser.add_argument(
        "--cookies-from-browser",
        default=None,
        help="Browser for YouTube cookies, e.g. chrome or firefox",
    )
    parser.add_argument(
        "--cookies",
        type=Path,
        default=None,
        help="Path to Netscape cookies.txt exported from browser",
    )
    args = parser.parse_args()

    video_id = extract_video_id(args.url)
    run_dir = args.run_dir or default_run_dir(video_id)
    run_dir.mkdir(parents=True, exist_ok=True)

    url = args.url if args.url.startswith("http") else f"https://www.youtube.com/watch?v={video_id}"
    download(url, run_dir, args.cookies_from_browser, args.cookies)


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        die(f"Command failed with exit code {exc.returncode}", exc.returncode)
