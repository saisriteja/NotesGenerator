"""Package and runtime data directory paths."""

from __future__ import annotations

import os
from pathlib import Path

_PKG_ROOT = Path(__file__).resolve().parent


def package_root() -> Path:
    """Installed package directory (contains scripts/, colab/, etc.)."""
    return _PKG_ROOT


def data_root() -> Path:
    """Writable root for runs/, models/, and pipeline outputs."""
    override = os.environ.get("NOTE_TAKER_ROOT")
    if override:
        return Path(override)
    return Path.cwd()


def scripts_dir() -> Path:
    return package_root() / "scripts"


def runs_dir() -> Path:
    return data_root() / "runs"


def models_dir() -> Path:
    return data_root() / "models"
