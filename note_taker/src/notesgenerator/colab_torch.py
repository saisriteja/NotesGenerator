"""Fix Colab PyTorch / TorchAudio CUDA mismatches after pip installs."""

from __future__ import annotations

import importlib
import subprocess
import sys

from notesgenerator._paths import is_colab


def _torch_cuda_tag() -> str | None:
    import torch

    cuda = getattr(torch.version, "cuda", None)
    if not cuda:
        return None
    parts = cuda.split(".")
    major = parts[0]
    minor = parts[1] if len(parts) > 1 else "0"
    return f"cu{major}{minor}"


def _torchaudio_ok() -> bool:
    try:
        import torchaudio  # noqa: F401

        return True
    except RuntimeError as exc:
        return "CUDA versions" not in str(exc)
    except ImportError:
        return True


def ensure_torch_stack(*, quiet: bool = False) -> None:
    """Reconcile torchaudio with Colab's PyTorch if pip left them mismatched."""
    if not is_colab():
        return
    if _torchaudio_ok():
        return

    import torch

    torch_ver = torch.__version__.split("+")[0]
    tag = _torch_cuda_tag()
    if not quiet:
        print(
            f"Fixing torchaudio for PyTorch {torch.__version__} (CUDA {torch.version.cuda}) ...",
            flush=True,
        )

    if tag:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "-q",
                "-U",
                f"torchaudio=={torch_ver}",
                "--index-url",
                f"https://download.pytorch.org/whl/{tag}",
            ],
            check=False,
        )

    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "-U", "torchaudio"],
        check=False,
    )

    if _torchaudio_ok():
        if not quiet:
            print("torchaudio OK.", flush=True)
        return

    if not quiet:
        print("Removing broken torchaudio (not needed for slide captioning) ...", flush=True)
    subprocess.run(
        [sys.executable, "-m", "pip", "uninstall", "-y", "torchaudio"],
        check=False,
    )
    sys.modules.pop("torchaudio", None)
    importlib.invalidate_caches()
