"""Package and runtime data directory paths."""

from __future__ import annotations

import os
from pathlib import Path

_PKG_ROOT = Path(__file__).resolve().parent


def package_root() -> Path:
    """Installed package directory (contains scripts/, colab/, etc.)."""
    return _PKG_ROOT


def is_colab() -> bool:
    return bool(os.environ.get("COLAB_RELEASE_TAG")) or Path("/content").is_dir()


def default_data_root() -> Path:
    if override := os.environ.get("NOTE_TAKER_ROOT"):
        return Path(override)
    if is_colab():
        return Path("/content")
    return Path.cwd()


def data_root() -> Path:
    return default_data_root()


def scripts_dir() -> Path:
    return package_root() / "scripts"


def runs_dir() -> Path:
    return data_root() / "runs"


def models_dir() -> Path:
    return data_root() / "models"
