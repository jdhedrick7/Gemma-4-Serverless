#!/usr/bin/env bash
#
# GPU quant bootstrap for a CLEAN base image:
#   nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04   (no torch/torchvision preinstalled)
#
# Why a clean CUDA base: RunPod's runpod/pytorch images bundle torchvision pinned to
# torch, so any torch change orphans it (the nms ABI break). A clean base has nothing
# to orphan, and cu128 matches our pinned torch 2.11.0+cu128 (Blackwell-ready).
#
# Prereqs on the pod:
#   * 1x Blackwell GPU (RTX PRO 6000 or B200), EUR-IS-1 (to attach the staged volume)
#   * network volume ayzcrd0zx1 mounted at /workspace, PRE-STAGED with:
#       - HF cache (bf16 model + eagle head + dataset) at /workspace/hf-cache
#       - warm pip wheel cache at /workspace/pip-cache  (makes this install fast)
#   * env: HF_TOKEN (read private source + push quantized repo)
#
# Usage on the pod:
#   export HF_TOKEN=hf_...
#   export DST_MODEL=jdfelo/gemma-4-31B-v2-NVFP4     # optional push target
#   bash run_pod.sh
#
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

VOLUME="${VOLUME:-/workspace}"
export HF_HOME="${HF_HOME:-${VOLUME}/hf-cache}"              # pre-staged model+dataset (no download)
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${VOLUME}/pip-cache}" # warm wheel cache (fast install)
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
export PIP_BREAK_SYSTEM_PACKAGES=1

SRC_MODEL="${SRC_MODEL:-jdfelo/gemma-4-31B-v2}"
DST_DIR="${DST_DIR:-${VOLUME}/gemma-4-31B-v2-NVFP4}"
DST_MODEL="${DST_MODEL:-}"                                    # empty = don't push
TORCH_PIN="${TORCH_PIN:-torch==2.11.0+cu128}"
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu128}"

: "${HF_TOKEN:?set HF_TOKEN (read private source + push)}"
export HF_TOKEN

echo "[gpu] HF_HOME=${HF_HOME}  PIP_CACHE_DIR=${PIP_CACHE_DIR}"

# ---- 1. System Python (clean CUDA base ships none) --------------------------
if ! command -v python3.12 >/dev/null 2>&1; then
    echo "[gpu] installing python3.12 + pip + git"
    apt-get update -q
    apt-get install -y -q python3.12 python3.12-venv python3-pip git >/dev/null
fi
PY=python3.12
$PY -m pip install -q -U pip >/dev/null 2>&1 || true

# ---- 2. Install toolchain to LOCAL disk, wheels from the warm cache ---------
# torch FIRST from the cu128 index (pinned build) so pip never re-resolves it.
echo "[gpu] installing ${TORCH_PIN} (cache-warm -> fast)"
$PY -m pip install -q "${TORCH_PIN}" --index-url "${TORCH_INDEX}"
echo "[gpu] installing pinned llmcompressor stack (must NOT bump torch)"
$PY -m pip install -q -r "${HERE}/requirements-quant.txt"

# ---- 3. Verify the pins held (the two things that broke before) -------------
$PY - <<'PY'
import torch, transformers, llmcompressor
assert torch.__version__.startswith("2.11.0"), f"torch bumped: {torch.__version__}"
print(f"[gpu] torch {torch.__version__} | transformers {transformers.__version__} | llmcompressor {llmcompressor.__version__}")
try:
    import torchvision
    print(f"[gpu] WARNING torchvision present ({torchvision.__version__}) — should be absent")
except Exception:
    print("[gpu] torchvision absent (correct)")
from transformers import Gemma4ForConditionalGeneration  # must import without torchvision
print("[gpu] Gemma4ForConditionalGeneration import OK")
PY

# ---- 4. Quantize (reads staged bf16 from HF_HOME; writes NVFP4 to DST_DIR) --
PUSH=(); [ -n "${DST_MODEL}" ] && PUSH=(--push-repo "${DST_MODEL}")
echo "[gpu] launching NVFP4 quantization"
$PY "${HERE}/quantize_nvfp4.py" --src "${SRC_MODEL}" --dst-dir "${DST_DIR}" "${PUSH[@]}"

echo "[gpu] DONE. NVFP4 checkpoint at: ${DST_DIR}"
du -sh "${DST_DIR}" 2>/dev/null || true
