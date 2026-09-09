#!/usr/bin/env python3
"""Module 2 — Transcribe audio with faster-whisper (GPU)."""

from __future__ import annotations

import argparse
from pathlib import Path

from common import add_run_dir_arg, die, save_json


def transcribe(
    audio_path: Path,
    output_path: Path,
    model_size: str,
    device: str,
    compute_type: str,
) -> None:
    if output_path.exists():
        print(f"Transcript already exists: {output_path}")
        return

    if not audio_path.exists():
        die(f"Missing audio: {audio_path}")

    from faster_whisper import WhisperModel

    print(f"Loading Whisper model '{model_size}' on {device} ({compute_type})...")
    model = WhisperModel(model_size, device=device, compute_type=compute_type)

    print(f"Transcribing {audio_path} ...")
    segments_iter, info = model.transcribe(str(audio_path), beam_size=5, vad_filter=True)
    print(f"Detected language: {info.language} (prob={info.language_probability:.2f})")

    segments = []
    for seg in segments_iter:
        segments.append(
            {
                "start": round(seg.start, 3),
                "end": round(seg.end, 3),
                "text": seg.text.strip(),
            }
        )
        if len(segments) % 50 == 0:
            print(f"  ... {len(segments)} segments")

    save_json(output_path, segments)
    print(f"Wrote {len(segments)} segments -> {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Transcribe lecture audio")
    add_run_dir_arg(parser)
    parser.add_argument("--model", default="medium", help="Whisper model size")
    parser.add_argument("--device", default="cuda", help="cuda or cpu")
    parser.add_argument(
        "--compute-type",
        default="float16",
        help="float16, int8_float16, etc.",
    )
    args = parser.parse_args()

    audio_path = args.run_dir / "raw" / "audio.wav"
    output_path = args.run_dir / "transcript" / "whisper_segments.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    transcribe(audio_path, output_path, args.model, args.device, args.compute_type)


if __name__ == "__main__":
    main()
