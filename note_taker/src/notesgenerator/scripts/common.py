"""Shared helpers for the lecture-notes pipeline."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

# yt-dlp split-stream names, e.g. Lecture_48.f137.mp4 + Lecture_48.f251.webm
YT_DLP_STREAM_SUFFIX = re.compile(r"\.f\d+\.(mp4|webm|webp|mkv)$", re.I)


def extract_video_id(url_or_id: str) -> str:
    """Return a YouTube video id from a URL or bare id."""
    url_or_id = url_or_id.strip()
    match = re.search(
        r"(?:v=|youtu\.be/|embed/|shorts/)([A-Za-z0-9_-]{11})",
        url_or_id,
    )
    if match:
        return match.group(1)
    if re.fullmatch(r"[A-Za-z0-9_-]{11}", url_or_id):
        return url_or_id
    raise ValueError(f"Could not parse YouTube video id from: {url_or_id!r}")


def default_run_dir(video_id: str, base: Path | None = None) -> Path:
    if base is not None:
        root = base
    else:
        data = Path(os.environ.get("NOTE_TAKER_ROOT", Path.cwd()))
        root = data / "runs"
    return root / f"run_{video_id}"


def ensure_run_layout(run_dir: Path) -> dict[str, Path]:
    """Create standard pipeline folders; return name -> path map."""
    folders = {
        "raw": run_dir / "raw",
        "transcript": run_dir / "transcript",
        "scenes": run_dir / "scenes",
        "keyframes": run_dir / "scenes" / "keyframes",
        "captions": run_dir / "captions",
        "merged": run_dir / "merged",
        "notes": run_dir / "notes",
        "attachments": run_dir / "attachments",
    }
    for path in folders.values():
        path.mkdir(parents=True, exist_ok=True)
    return folders


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")


def add_run_dir_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Pipeline run directory (e.g. runs/run_<video_id>)",
    )


def die(message: str, code: int = 1) -> None:
    print(message, file=sys.stderr)
    sys.exit(code)


_NVENC_AVAILABLE: bool | None = None


def nvenc_available() -> bool:
    """True when ffmpeg can encode H.264 with NVENC (Colab T4, etc.)."""
    global _NVENC_AVAILABLE
    if _NVENC_AVAILABLE is None:
        if os.environ.get("NOTE_TAKER_DISABLE_NVENC") == "1":
            _NVENC_AVAILABLE = False
        else:
            probe = subprocess.run(
                [
                    "ffmpeg",
                    "-hide_banner",
                    "-f",
                    "lavfi",
                    "-i",
                    "nullsrc=s=256x256:d=0.1",
                    "-c:v",
                    "h264_nvenc",
                    "-f",
                    "null",
                    "-",
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            _NVENC_AVAILABLE = probe.returncode == 0
    return _NVENC_AVAILABLE


def media_duration_sec(path: Path) -> float | None:
    """Return container duration in seconds, or None if ffprobe cannot read it."""
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    try:
        return float(result.stdout.strip())
    except ValueError:
        return None


def h264_encoder_args(*, fast: bool = True) -> list[str]:
    """Encoder flags for pipeline-compatible H.264 output."""
    if nvenc_available():
        if fast:
            return [
                "-c:v",
                "h264_nvenc",
                "-preset",
                "p1",
                "-tune",
                "ll",
                "-rc",
                "vbr",
                "-cq",
                "28",
                "-pix_fmt",
                "yuv420p",
            ]
        return [
            "-c:v",
            "h264_nvenc",
            "-preset",
            "p4",
            "-rc",
            "vbr",
            "-cq",
            "23",
            "-pix_fmt",
            "yuv420p",
        ]
    if fast:
        return [
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-tune",
            "fastdecode",
            "-crf",
            "28",
            "-threads",
            "0",
            "-pix_fmt",
            "yuv420p",
        ]
    return [
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "23",
        "-threads",
        "0",
        "-pix_fmt",
        "yuv420p",
    ]


def pipeline_video_filters(*, fast: bool = True) -> list[str]:
    """Video filters tuned for slide/scene detection (not final viewing quality)."""
    if not fast:
        return []
    # 15 fps + 720p-class width: much less work for AV1 decode/encode on long lectures.
    return [
        "-vf",
        "fps=15,scale='min(1280,iw)':-2:flags=fast_bilinear",
    ]


def transcode_wait_hint(src: Path, src_codec: str, *, fast: bool = True) -> str:
    """Human-readable note before a long ffmpeg transcode."""
    duration = media_duration_sec(src)
    duration_txt = f" (~{int(duration // 60)} min)" if duration else ""
    encoder = "GPU h264_nvenc (p1)" if nvenc_available() else "CPU libx264 (ultrafast)"
    speed_note = (
        "Fast prepare: 15 fps, max width 1280px."
        if fast
        else "Full-quality prepare (slower)."
    )
    return (
        f"Transcoding {src_codec} -> h264{duration_txt} via {encoder}. "
        f"{speed_note} Progress lines appear below every ~5s."
    )


def run_ffmpeg(cmd: list[str], *, hint: str = "") -> None:
    """Run ffmpeg with live stderr progress (notebook/Colab friendly)."""
    if hint:
        print(hint, flush=True)

    full = list(cmd)
    extras = ["-hide_banner", "-stats_period", "5"]
    if full and full[0] == "ffmpeg":
        insert_at = 2 if len(full) > 1 and full[1] == "-y" else 1
        for offset, flag in enumerate(extras):
            if flag not in full:
                full.insert(insert_at + offset, flag)

    print("Running:", " ".join(full), flush=True)
    subprocess.run(full, check=True)


def media_stem(path: Path) -> str:
    """Strip yt-dlp format suffix: ``Lecture_48.f137.mp4`` → ``Lecture_48``."""
    name = path.name
    match = YT_DLP_STREAM_SUFFIX.search(name)
    if match:
        return name[: match.start()]
    return path.stem


def stream_codecs(path: Path) -> dict[str, str | None]:
    """Return ``{'video': codec_name|None, 'audio': codec_name|None}`` via ffprobe."""
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "stream=codec_type,codec_name",
        "-of",
        "csv=p=0",
        str(path),
    ]
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    video: str | None = None
    audio: str | None = None
    for line in result.stdout.splitlines():
        parts = [part.strip().lower() for part in line.split(",") if part.strip()]
        if len(parts) < 2:
            continue
        if parts[1] == "video" and video is None:
            video = parts[0] or None
        elif parts[1] == "audio" and audio is None:
            audio = parts[0] or None
    return {"video": video, "audio": audio}


def _same_lecture_candidates(path: Path) -> list[Path]:
    """Sibling files that share the same yt-dlp lecture stem."""
    parent = path.parent
    stem = media_stem(path)
    return sorted(
        p
        for p in parent.iterdir()
        if p.is_file() and media_stem(p) == stem and p != path
    )


def find_audio_companion(video_path: Path) -> Path | None:
    """Find a separate audio-only file for a video-only source (e.g. .f251.webm)."""
    audio_only: list[Path] = []
    muxed: list[Path] = []
    for candidate in _same_lecture_candidates(video_path):
        if candidate.suffix.lower() not in {".webm", ".webp", ".mp4", ".m4a", ".mkv"}:
            continue
        streams = stream_codecs(candidate)
        if streams["audio"] and not streams["video"]:
            audio_only.append(candidate)
        elif streams["audio"] and streams["video"]:
            muxed.append(candidate)

    if audio_only:
        # Prefer yt-dlp opus audio (f251) when present, else largest file.
        audio_only.sort(
            key=lambda p: (".f251." not in p.name.lower(), -p.stat().st_size),
        )
        return audio_only[0]
    return muxed[0] if muxed else None


def find_video_companion(media_path: Path) -> Path | None:
    """Find a separate video-only file for an audio-only source."""
    video_only: list[Path] = []
    muxed: list[Path] = []
    for candidate in _same_lecture_candidates(media_path):
        if candidate.suffix.lower() not in {".webm", ".webp", ".mp4", ".mkv"}:
            continue
        streams = stream_codecs(candidate)
        if streams["video"] and not streams["audio"]:
            video_only.append(candidate)
        elif streams["video"] and streams["audio"]:
            muxed.append(candidate)

    if video_only:
        # Prefer mp4 over webm, then yt-dlp f137/f399, then largest file.
        def rank(p: Path) -> tuple:
            suffix = p.suffix.lower()
            name = p.name.lower()
            return (
                suffix != ".mp4",
                ".f137." not in name and ".f399." not in name,
                -p.stat().st_size,
            )

        video_only.sort(key=rank)
        return video_only[0]
    return muxed[0] if muxed else None


def resolve_media_inputs(
    video: Path,
    audio: Path | None = None,
) -> tuple[Path, Path | None]:
    """Resolve video/audio sources, including yt-dlp split-stream pairs."""
    video = video.resolve()
    if not video.exists():
        die(f"Video not found: {video}")

    if audio is not None:
        audio = audio.resolve()
        if not audio.exists():
            die(f"Audio not found: {audio}")
        return video, audio

    streams = stream_codecs(video)
    if streams["video"] and streams["audio"]:
        return video, None

    if streams["video"] and not streams["audio"]:
        companion = find_audio_companion(video)
        if companion:
            print(f"Auto-detected audio: {companion.name}")
            return video, companion
        return video, None

    if streams["audio"] and not streams["video"]:
        companion = find_video_companion(video)
        if companion:
            print(f"Auto-detected video: {companion.name}")
            return companion, video
        die(f"No video stream in {video} and no video companion found")

    die(f"No usable video or audio streams in: {video}")
