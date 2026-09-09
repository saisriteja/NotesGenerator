"""Qwen3-VL offline inference via vLLM.

Loads the model when a script starts and tears it down on exit — no separate
``vllm serve`` process required (Colab-friendly).
"""

from __future__ import annotations

import gc
import os
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

# Colab / notebook safe multiprocessing start method
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

DEFAULT_MODEL = os.environ.get("QWEN_MODEL", "Qwen/Qwen3-VL-4B-Instruct")


def image_messages(image_uri: str, prompt: str) -> list[dict]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_uri},
                {"type": "text", "text": prompt},
            ],
        }
    ]


def text_messages(prompt: str) -> list[dict]:
    return [{"role": "user", "content": [{"type": "text", "text": prompt}]}]


def prepare_vllm_input(messages: list[dict], processor) -> dict[str, Any]:
    from qwen_vl_utils import process_vision_info

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs, video_kwargs = process_vision_info(
        messages,
        image_patch_size=processor.image_processor.patch_size,
        return_video_kwargs=True,
        return_video_metadata=True,
    )

    mm_data: dict[str, Any] = {}
    if image_inputs is not None:
        mm_data["image"] = image_inputs
    if video_inputs is not None:
        mm_data["video"] = video_inputs

    payload: dict[str, Any] = {"prompt": text}
    if mm_data:
        payload["multi_modal_data"] = mm_data
    if video_kwargs:
        payload["mm_processor_kwargs"] = video_kwargs
    return payload


@dataclass
class VllmConfig:
    model: str = DEFAULT_MODEL
    max_model_len: int = 8192
    gpu_memory_utilization: float = 0.90
    tensor_parallel_size: int = 1
    max_new_tokens: int = 512
    temperature: float = 0.0


class QwenVllmEngine:
    """One-shot vLLM engine: load → generate → close."""

    def __init__(self, config: VllmConfig):
        self.config = config
        self._llm = None
        self._processor = None

    def load(self) -> None:
        try:
            from notesgenerator.colab_torch import ensure_torch_stack
        except ImportError:
            import sys
            from pathlib import Path

            root = Path(__file__).resolve().parents[1]
            if str(root) not in sys.path:
                sys.path.insert(0, str(root))
            from colab_torch import ensure_torch_stack

        ensure_torch_stack()

        import torch
        from transformers import AutoProcessor
        from vllm import LLM

        print(f"Loading Qwen3-VL via vLLM: {self.config.model}")
        self._processor = AutoProcessor.from_pretrained(self.config.model)

        tp = self.config.tensor_parallel_size
        if tp <= 0:
            tp = max(1, torch.cuda.device_count())

        self._llm = LLM(
            model=self.config.model,
            tensor_parallel_size=tp,
            gpu_memory_utilization=self.config.gpu_memory_utilization,
            max_model_len=self.config.max_model_len,
            trust_remote_code=True,
        )
        print("vLLM engine ready.")

    def _sampling_params(self):
        from vllm import SamplingParams

        return SamplingParams(
            temperature=self.config.temperature,
            max_tokens=self.config.max_new_tokens,
            top_k=-1,
        )

    def generate_batch(self, messages_list: list[list[dict]]) -> list[str]:
        if self._llm is None or self._processor is None:
            raise RuntimeError("Engine not loaded")

        inputs = [
            prepare_vllm_input(msgs, self._processor) for msgs in messages_list
        ]
        outputs = self._llm.generate(inputs, sampling_params=self._sampling_params())
        return [out.outputs[0].text.strip() for out in outputs]

    def generate_one(self, messages: list[dict]) -> str:
        return self.generate_batch([messages])[0]

    def close(self) -> None:
        import torch

        if self._llm is not None:
            del self._llm
            self._llm = None
        if self._processor is not None:
            del self._processor
            self._processor = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("vLLM engine shut down.")


@contextmanager
def qwen_vllm_session(config: VllmConfig):
    engine = QwenVllmEngine(config)
    try:
        engine.load()
        yield engine
    finally:
        engine.close()
