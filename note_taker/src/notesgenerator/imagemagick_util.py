"""ImageMagick detection and command helpers."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

from notesgenerator._paths import is_colab


def available() -> bool:
    return bool(
        shutil.which("convert")
        or shutil.which("identify")
        or shutil.which("magick")
    )


def im_cmd(subcommand: str, *args: str) -> list[str]:
    """Build an ImageMagick command (supports IM6 binaries and IM7 ``magick``)."""
    if shutil.which(subcommand):
        return [subcommand, *args]
    if shutil.which("magick"):
        return ["magick", subcommand, *args]
    raise FileNotFoundError(
        f"ImageMagick '{subcommand}' not found. Install: apt-get install -y imagemagick"
    )


def ensure_installed(*, auto_install_colab: bool = True) -> bool:
    """Return True if ImageMagick is usable; optionally apt-install on Colab."""
    if available():
        return True

    if auto_install_colab and is_colab():
        print("ImageMagick not found — installing on Colab ...", flush=True)
        subprocess.run(["apt-get", "update", "-qq"], check=False)
        subprocess.run(
            ["apt-get", "install", "-y", "-qq", "imagemagick"],
            check=False,
        )
        if available():
            print("ImageMagick installed OK.", flush=True)
            return True

    print(
        "WARNING: ImageMagick not available — image compression will be skipped.\n"
        "  Install manually: apt-get install -y imagemagick",
        flush=True,
    )
    return False


def identify_size(path: Path) -> tuple[int, int]:
    result = subprocess.run(
        im_cmd("identify", "-format", "%w %h", str(path)),
        check=True,
        capture_output=True,
        text=True,
    )
    w, h = result.stdout.strip().split()
    return int(w), int(h)


def convert(src: Path, dst: Path, extra_args: list[str]) -> None:
    subprocess.run(
        im_cmd("convert", str(src), *extra_args, str(dst)),
        check=True,
        capture_output=True,
    )
