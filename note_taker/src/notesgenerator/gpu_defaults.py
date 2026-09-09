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
