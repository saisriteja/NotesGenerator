"""GPU-aware defaults for vLLM (Qwen3-VL) memory settings."""

from __future__ import annotations


def report_vllm_defaults() -> tuple[int, float]:
    """Return ``(max_model_len, gpu_memory_utilization)`` for step 09.

    Qwen3-VL-4B on a 16 GB T4 cannot allocate enough KV cache for a 16k
    context window. Use 8192 / 0.90 on T4-class GPUs; keep 16384 / 0.82
    on larger GPUs (A100, L4, etc.).
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return 8192, 0.90

        props = torch.cuda.get_device_properties(0)
        compute_cap = props.major + props.minor / 10.0
        vram_gib = props.total_memory / (1024**3)
        name = props.name

        # FlashAttention2 needs CC >= 8; 16 GB VRAM cannot hold 16k ctx + weights.
        if compute_cap < 8.0 or vram_gib <= 16.5:
            print(
                f"GPU {name!r} ({vram_gib:.1f} GiB, CC {compute_cap}) "
                "→ report step uses max_model_len=8192, gpu_mem=0.90"
            )
            return 8192, 0.90

        print(
            f"GPU {name!r} ({vram_gib:.1f} GiB, CC {compute_cap}) "
            "→ report step uses max_model_len=16384, gpu_mem=0.82"
        )
        return 16384, 0.82
    except Exception:
        return 8192, 0.90


# Rough floor for Qwen3-VL-4B weights + vLLM overhead (bf16, single GPU).
_MIN_FREE_GIB_FOR_VLLM = 12.0


def clamp_gpu_mem_for_free_vram(requested_util: float, margin: float = 0.95) -> float:
    """Lower gpu_memory_utilization when other processes already occupy VRAM.

    Colab/T4 defaults (0.82 / 0.90) stay unchanged on an empty GPU.
    vLLM requires ``gpu_memory_utilization * total_vram <= free_vram`` at startup.
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return requested_util

        free_bytes, total_bytes = torch.cuda.mem_get_info()
        if total_bytes <= 0:
            return requested_util

        free_gib = free_bytes / (1024**3)
        total_gib = total_bytes / (1024**3)
        needed_gib = requested_util * total_gib
        max_util = (free_bytes / total_bytes) * margin

        if free_gib < _MIN_FREE_GIB_FOR_VLLM:
            raise SystemExit(
                f"Not enough free VRAM on the visible GPU "
                f"({free_gib:.1f}/{total_gib:.1f} GiB free; need ~{_MIN_FREE_GIB_FOR_VLLM:.0f}+ GiB).\n"
                "Other processes are using this GPU. Options:\n"
                "  • export CUDA_VISIBLE_DEVICES=1   (use a freer GPU)\n"
                "  • stop competing GPU jobs, then retry\n"
                "  • pass --gpu-mem 0.72 if only slightly over budget"
            )

        if needed_gib <= free_gib * margin:
            return requested_util

        adjusted = min(requested_util, max_util)
        print(
            f"GPU memory shared: free {free_gib:.1f}/{total_gib:.1f} GiB — "
            f"lowering gpu_mem {requested_util:.2f} → {adjusted:.2f} "
            f"(needed ~{needed_gib:.1f} GiB at requested util)"
        )
        return round(adjusted, 3)
    except SystemExit:
        raise
    except Exception:
        return requested_util
