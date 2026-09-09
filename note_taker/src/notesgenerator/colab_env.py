"""Colab-friendly environment setup for NotesGenerator."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from notesgenerator._paths import default_data_root, is_colab, models_dir, runs_dir, scripts_dir
from notesgenerator.colab_torch import ensure_torch_stack


def configure_colab_env(
    *,
    offline: bool = False,
    model: str = "Qwen/Qwen3-VL-4B-Instruct",
    hf_token: str | None = None,
    data_path: Path | str | None = None,
) -> Path:
    """Set env vars and ``sys.path`` for Colab / notebook use."""
    root = Path(data_path) if data_path else default_data_root()
    root.mkdir(parents=True, exist_ok=True)
    os.environ["NOTE_TAKER_ROOT"] = str(root)
    runs_dir().mkdir(parents=True, exist_ok=True)
    (models_dir() / "hub").mkdir(parents=True, exist_ok=True)
    os.environ["PYTHONUNBUFFERED"] = "1"
    os.environ["HF_HOME"] = str(models_dir() / "hub")
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(models_dir() / "hub")
    os.environ["TORCH_HOME"] = str(models_dir() / "torch")
    os.environ["QWEN_MODEL"] = model
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

    if offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    else:
        os.environ.pop("HF_HUB_OFFLINE", None)
        os.environ.pop("TRANSFORMERS_OFFLINE", None)

    if hf_token:
        os.environ["HF_TOKEN"] = hf_token
        os.environ["HUGGINGFACE_HUB_TOKEN"] = hf_token

    scripts = str(scripts_dir())
    if scripts not in sys.path:
        sys.path.insert(0, scripts)

    print(f"NotesGenerator data root: {root}")
    print(f"  runs/   → {runs_dir()}")
    print(f"  models/ → {models_dir()}")
    if is_colab():
        print("  (Colab detected — outputs go under /content/)")
        ensure_torch_stack()
    sys.stdout.flush()
    return root
