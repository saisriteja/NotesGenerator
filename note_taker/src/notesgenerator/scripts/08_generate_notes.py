#!/usr/bin/env python3
"""Module 8 — Generate Obsidian lecture notes (Qwen3-VL text mode + vLLM)."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from common import add_run_dir_arg, load_json
from qwen_vllm import DEFAULT_MODEL, VllmConfig, qwen_vllm_session, text_messages

NOTE_SYSTEM = (
    "You write clear Obsidian lecture notes for an AI/math course. "
    "Use the slide description and spoken transcript. Keep equations in LaTeX."
)


def fmt_time(seconds: float) -> str:
    minutes, secs = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def scene_prompt(scene: dict) -> str:
    sid = scene["scene_id"]
    return f"""{NOTE_SYSTEM}

Write Obsidian markdown for this lecture scene.

Scene {sid} ({fmt_time(scene['start'])}–{fmt_time(scene['end'])})

Slide description:
{scene.get('vlm_description', '')}

Spoken transcript:
{scene.get('transcript', '')}

Use exactly this template:

## Scene {sid} — [short topic] ({fmt_time(scene['start'])}–{fmt_time(scene['end'])})
![[scene_{sid:03d}.png]]

**Slide content:** ...
**Explanation (from audio):** ...
**Key equations:** ...
"""


def polish_scene_block(block: str, scene_id: int) -> str:
    if f"![[scene_{scene_id:03d}.png]]" not in block:
        header = f"## Scene {scene_id}"
        if header in block:
            block = block.replace(header, f"{header}\n![[scene_{scene_id:03d}.png]]", 1)
    return block.strip()


def generate_notes(
    run_dir: Path,
    model: str,
    max_new_tokens: int,
    batch_size: int,
) -> Path:
    merged_path = run_dir / "merged" / "scenes_merged.json"
    scenes = load_json(merged_path)

    notes_dir = run_dir / "notes"
    attachments_dir = run_dir / "attachments"
    notes_dir.mkdir(parents=True, exist_ok=True)
    attachments_dir.mkdir(parents=True, exist_ok=True)

    keyframes_dir = run_dir / "scenes" / "keyframes"
    for scene in scenes:
        sid = scene["scene_id"]
        src = keyframes_dir / f"scene_{sid:03d}.png"
        dst = attachments_dir / f"scene_{sid:03d}.png"
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)

    config = VllmConfig(model=model, max_new_tokens=max_new_tokens)
    blocks: list[str] = ["# Lecture Notes\n"]

    with qwen_vllm_session(config) as engine:
        for start in range(0, len(scenes), batch_size):
            batch = scenes[start : start + batch_size]
            end = min(start + batch_size, len(scenes))
            print(f"Generating notes for scenes {start + 1}–{end} / {len(scenes)} ...")

            messages_list = [text_messages(scene_prompt(s)) for s in batch]
            texts = engine.generate_batch(messages_list)

            for scene, text in zip(batch, texts):
                blocks.append(polish_scene_block(text, scene["scene_id"]))
                blocks.append("")

    out_path = notes_dir / "lecture_notes.md"
    out_path.write_text("\n".join(blocks).strip() + "\n", encoding="utf-8")
    print(f"Wrote {out_path} ({len(scenes)} scenes)")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Obsidian notes with Qwen3-VL (vLLM offline)"
    )
    add_run_dir_arg(parser)
    parser.add_argument(
        "--model",
        default=None,
        help="Qwen3-VL model id (default: $QWEN_MODEL or Qwen/Qwen3-VL-4B-Instruct)",
    )
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Scenes per vLLM batch",
    )
    args = parser.parse_args()

    generate_notes(
        args.run_dir,
        args.model or DEFAULT_MODEL,
        args.max_new_tokens,
        args.batch_size,
    )


if __name__ == "__main__":
    main()
