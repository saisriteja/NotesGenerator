#!/usr/bin/env python3
"""Combined VLM step — caption slides + generate report in one model session."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from common import add_run_dir_arg, die

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import importlib.util


def _load_script(name: str):
    path = SCRIPT_DIR / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load script: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


caption_mod = _load_script("05_caption_slides")
merge_mod = _load_script("07_merge")
report_mod = _load_script("09_generate_report")

caption_slides = caption_mod.caption_slides
merge_run = merge_mod.merge_run
generate_report = report_mod.generate_report

from qwen_vllm import DEFAULT_MODEL, VllmConfig, qwen_vllm_session

try:
    from notesgenerator.gpu_defaults import clamp_gpu_mem_for_free_vram, report_vllm_defaults
except ImportError:
    _pkg_root = SCRIPT_DIR.parent
    if str(_pkg_root) not in sys.path:
        sys.path.insert(0, str(_pkg_root))
    from gpu_defaults import (  # type: ignore[import-untyped]
        clamp_gpu_mem_for_free_vram,
        report_vllm_defaults,
    )


def _require_run_artifacts(run_dir: Path, force_caption: bool) -> None:
    """Fail fast before loading vLLM if upstream pipeline outputs are missing."""
    if not run_dir.is_dir():
        die(
            f"Run directory not found: {run_dir}\n"
            "Run the full pipeline first (steps 00–06), e.g.:\n"
            '  notesgenerator --video "lectures/lecture 2.mp4" --output lecture_2_test --no-zip'
        )

    scene_list = run_dir / "scenes" / "scene_list.json"
    if not scene_list.is_file():
        die(
            f"Missing {scene_list} — scene detection (step 03) has not been run.\n"
            "Re-run the full pipeline or at least steps 03–06 before this script."
        )

    captions = run_dir / "captions" / "vlm_captions.json"
    if force_caption or not captions.is_file():
        keyframes = run_dir / "scenes" / "keyframes"
        if not keyframes.is_dir() or not any(keyframes.glob("scene_*.png")):
            die(
                f"No keyframes in {keyframes} — run step 04 (extract keyframes) first."
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Caption slides and generate report with a single VLM load"
    )
    add_run_dir_arg(parser)
    parser.add_argument("--model", default=None)
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--gpu-mem", type=float, default=None)
    parser.add_argument(
        "--vlm-batch-size",
        type=int,
        default=8,
        help="Slide captions per vLLM batch",
    )
    parser.add_argument(
        "--outline-batch-size",
        type=int,
        default=4,
        help="Outline windows per vLLM batch",
    )
    parser.add_argument(
        "--topic-batch-size",
        type=int,
        default=2,
        help="Topic write/refine requests per vLLM batch",
    )
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--max-main-scenes", type=int, default=25)
    parser.add_argument("--window-minutes", type=int, default=8)
    parser.add_argument("--section-tokens", type=int, default=6000)
    parser.add_argument("--refine-tokens", type=int, default=4096)
    parser.add_argument("--refine-passes", type=int, default=2)
    parser.add_argument("--verify-passes", type=int, default=2)
    parser.add_argument("--verify-tokens", type=int, default=4096)
    parser.add_argument("--force-caption", action="store_true")
    parser.add_argument("--force-outline", action="store_true")
    parser.add_argument("--force-topics", action="store_true")
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    _require_run_artifacts(run_dir, args.force_caption)
    model = args.model or DEFAULT_MODEL
    auto_max_len, auto_gpu_mem = report_vllm_defaults()
    max_model_len = args.max_model_len if args.max_model_len is not None else auto_max_len
    gpu_mem = args.gpu_mem if args.gpu_mem is not None else auto_gpu_mem
    gpu_mem = clamp_gpu_mem_for_free_vram(gpu_mem)
    config = VllmConfig(
        model=model,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_mem,
        max_new_tokens=args.max_new_tokens,
        tensor_parallel_size=1,
    )

    print("Loading VLM once for caption + report ...")
    with qwen_vllm_session(config) as engine:
        print("▶ VLM caption slides")
        caption_slides(
            run_dir / "scenes" / "scene_list.json",
            run_dir / "scenes" / "keyframes",
            run_dir / "captions" / "vlm_captions.json",
            model,
            args.max_new_tokens,
            args.vlm_batch_size,
            args.force_caption,
            engine=engine,
        )

        print("▶ Merge captions + transcript")
        merge_run(run_dir)

        print("▶ VLM generate report")
        generate_report(
            run_dir,
            model,
            max_model_len,
            args.max_main_scenes,
            args.window_minutes,
            args.section_tokens,
            args.refine_tokens,
            args.refine_passes,
            args.verify_passes,
            args.verify_tokens,
            args.force_outline,
            args.force_topics,
            gpu_mem,
            engine=engine,
            outline_batch_size=args.outline_batch_size,
            topic_batch_size=args.topic_batch_size,
        )

    print("Combined VLM step complete.")


if __name__ == "__main__":
    main()
