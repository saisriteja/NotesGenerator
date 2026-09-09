#!/usr/bin/env python3
"""One-shot Colab setup: system deps + Python packages."""

from __future__ import annotations

import subprocess
import sys


def run(cmd: list[str], *, check: bool = True) -> None:
    print("$", " ".join(cmd))
    subprocess.run(cmd, check=check)


def install_system_deps() -> None:
    run(["apt-get", "update", "-qq"])
    run(
        [
            "apt-get",
            "install",
            "-y",
            "-qq",
            "ffmpeg",
            "libsm6",
            "libxext6",
            "libgl1",
        ]
    )


def install_python_deps() -> None:
    run([sys.executable, "-m", "pip", "install", "-q", "-U", "pip"])
    run([sys.executable, "-m", "pip", "install", "-q", "NotesGenerator[gpu]"])


def main() -> None:
    install_system_deps()
    install_python_deps()
    print("\nSetup complete.")
    print("Next: from notesgenerator.colab_env import configure_colab_env")


if __name__ == "__main__":
    main()
