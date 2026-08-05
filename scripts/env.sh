#!/usr/bin/env bash
# Source before running any pipeline script:
#   source scripts/env.sh

export HF_HOME="${HF_HOME:-/workspace/teja/models/hub}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-/workspace/teja/models/hub}"
export TORCH_HOME="${TORCH_HOME:-/workspace/torch-cache}"

# Use cached models only — avoids HuggingFace 403 via IDE proxy
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY all_proxy
unset SOCKS_PROXY SOCKS5_PROXY socks_proxy socks5_proxy
unset GIT_HTTP_PROXY GIT_HTTPS_PROXY
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/workspace/users/${USER}/.pip-cache}"

# Qwen3-VL via vLLM (offline — no separate serve process)
export QWEN_MODEL="${QWEN_MODEL:-Qwen/Qwen3-VL-4B-Instruct}"
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"

source /devwork/MiniConda/miniconda3/etc/profile.d/conda.sh
conda activate gsplat_env

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export NOTE_GENERATOR_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
