#!/usr/bin/env python3
"""Module 5 — VLM captioning for slide keyframes (Qwen3-VL + vLLM, batched)."""

from __future__ import annotations

import argparse
from pathlib import Path

from common import add_run_dir_arg, load_json, save_json
from qwen_vllm import VllmConfig, image_messages, qwen_vllm_session

CAPTION_PROMPT = (
    "This is a slide from an AI/math lecture. Transcribe any equations (LaTeX), "
    "list key terms/diagrams, and summarize what the slide shows in 2–3 sentences."
)


def caption_slides(
    scene_list_path: Path,
    keyframes_dir: Path,
    output_path: Path,
    model: str,
    max_new_tokens: int,
    batch_size: int,
    force: bool,
) -> None:
    if output_path.exists() and not force:
        print(f"Captions already exist: {output_path}")
        return

    scenes = load_json(scene_list_path)
    work: list[tuple[int, Path]] = []
    for scene in scenes:
        scene_id = scene["scene_id"]
        image_path = keyframes_dir / f"scene_{scene_id:03d}.png"
        if image_path.exists():
            work.append((scene_id, image_path))
        else:
            print(f"Skipping scene {scene_id}: missing {image_path}")

    config = VllmConfig(model=model, max_new_tokens=max_new_tokens)
    captions: list[dict] = []

    with qwen_vllm_session(config) as engine:
        for start in range(0, len(work), batch_size):
            batch = work[start : start + batch_size]
            end = min(start + batch_size, len(work))
            print(f"Captioning scenes {start + 1}–{end} / {len(work)} ...")

            messages_list = [
                image_messages(f"file://{path.resolve()}", CAPTION_PROMPT)
                for _, path in batch
            ]
            texts = engine.generate_batch(messages_list)

            for (scene_id, _), text in zip(batch, texts):
                captions.append({"scene_id": scene_id, "vlm_description": text})

            save_json(output_path, captions)

    save_json(output_path, captions)
    print(f"Wrote {len(captions)} captions -> {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Caption slide keyframes with Qwen3-VL (vLLM offline)"
    )
    add_run_dir_arg(parser)
    parser.add_argument(
        "--model",
        default=None,
        help="Qwen3-VL model id (default: $QWEN_MODEL or Qwen/Qwen3-VL-4B-Instruct)",
    )
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Images per vLLM batch (raise on A100, lower on T4)",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite existing captions")
    args = parser.parse_args()

    from qwen_vllm import DEFAULT_MODEL

    run_dir = args.run_dir
    caption_slides(
        run_dir / "scenes" / "scene_list.json",
        run_dir / "scenes" / "keyframes",
        run_dir / "captions" / "vlm_captions.json",
        args.model or DEFAULT_MODEL,
        args.max_new_tokens,
        args.batch_size,
        args.force,
    )


if __name__ == "__main__":
    main()
