#!/usr/bin/env python3
"""Prepare run folder from a local video file (copy mp4 + extract 16 kHz audio).

Supports single muxed files and yt-dlp split streams (video-only .mp4/.webm +
audio-only .webm in the same folder, e.g. gpumode downloads).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from common import (
    add_run_dir_arg,
    die,
    ensure_run_layout,
    h264_encoder_args,
    pipeline_video_filters,
    resolve_media_inputs,
    run_ffmpeg,
    stream_codecs,
    transcode_wait_hint,
)


def remux_video_copy(src: Path, dst: Path) -> None:
    """Fast remux — copy bitstreams without re-encoding."""
    cmd = ["ffmpeg", "-y", "-i", str(src), "-c", "copy", str(dst)]
    run_ffmpeg(
        cmd,
        hint=f"Remuxing {src.name} with stream copy (no re-encode; seconds, not minutes).",
    )


def transcode_to_h264(src: Path, dst: Path, *, fast: bool = True) -> None:
    streams = stream_codecs(src)
    codec = streams["video"] or "unknown"
    cmd = [
        "ffmpeg", "-y", "-threads", "0", "-i", str(src),
        *pipeline_video_filters(fast=fast),
        *h264_encoder_args(fast=fast),
    ]
    if streams["audio"]:
        cmd.extend(["-c:a", "copy"])
    else:
        cmd.append("-an")
    cmd.append(str(dst))
    run_ffmpeg(cmd, hint=transcode_wait_hint(src, codec, fast=fast))


def extract_audio(src: Path, dst: Path) -> None:
    cmd = [
        "ffmpeg", "-y", "-i", str(src),
        "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
        str(dst),
    ]
    run_ffmpeg(cmd)


def prepare(
    video_path: Path,
    run_dir: Path,
    force: bool,
    audio_path: Path | None = None,
    *,
    transcode_h264: bool = False,
    fast: bool = True,
) -> None:
    video_src, audio_src = resolve_media_inputs(video_path, audio_path)

    folders = ensure_run_layout(run_dir)
    raw = folders["raw"]
    video_out = raw / "video.mp4"
    audio_out = raw / "audio.wav"

    if video_out.exists() and audio_out.exists() and not force:
        print(f"Already prepared: {video_out}, {audio_out}")
        return

    need_audio = not audio_out.exists() or force
    src_streams = stream_codecs(video_src)
    src_codec = src_streams["video"] or "unknown"

    def write_video() -> None:
        if transcode_h264:
            print(f"Transcoding {src_codec} -> h264: {video_src} -> {video_out}")
            transcode_to_h264(video_src, video_out, fast=fast)
        else:
            print(f"Remuxing {src_codec} -> {video_out} (stream copy)")
            remux_video_copy(video_src, video_out)

    can_parallel_audio = need_audio and not audio_src and src_streams["audio"]
    if can_parallel_audio:
        print("Extracting audio in parallel with video remux ...", flush=True)
        with ThreadPoolExecutor(max_workers=2) as pool:
            video_future = pool.submit(write_video)
            audio_future = pool.submit(extract_audio, video_src, audio_out)
            video_future.result()
            audio_future.result()
        need_audio = False
    else:
        write_video()

    if need_audio:
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
    parser.add_argument(
        "--transcode-h264",
        action="store_true",
        help="Re-encode to h264 (slow). Default is ffmpeg -c copy remux.",
    )
    parser.add_argument(
        "--full-quality",
        action="store_true",
        help="With --transcode-h264: keep source fps/resolution (default is 15fps/1280px)",
    )
    args = parser.parse_args()
    prepare(
        args.video.resolve(),
        args.run_dir,
        args.force,
        args.audio,
        transcode_h264=args.transcode_h264,
        fast=not args.full_quality,
    )


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        die(f"ffmpeg failed with exit code {exc.returncode}", exc.returncode)
