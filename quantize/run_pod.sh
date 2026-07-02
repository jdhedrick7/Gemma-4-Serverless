#!/usr/bin/env bash
#
# GPU quant bootstrap. Recommended base image (has sshd + keepalive so RunPod
# won't crash-loop it, unlike a bare nvidia/cuda image):
#
#   runpod/pytorch:1.0.7-cu1281-torch291-ubuntu2404
#   (CUDA 12.8.1 = cu128 = Blackwell-ready; Python 3.12; ships torch 2.9.1)
#
# This script makes ANY such base churn-free:
#   1. drop the bundled torchvision/torchaudio (the ABI-break source),
#   2. install the PINNED torch 2.11.0+cu128 (llmcompressor 0.12.0's range),
#   3. install the pinned llmcompressor stack (must NOT bump torch),
#   4. verify the pins held, then NVFP4-quantize from the pre-staged cache.
#
# Prereqs on the pod:
#   * 1x Blackwell GPU (RTX PRO 6000 or B200), EUR-IS-1 (to attach the volume)
#   * network volume ayzcrd0zx1 at /workspace, PRE-STAGED:
#       - HF cache (bf16 model + eagle + dataset) at /workspace/hf-cache
#       - warm pip wheel cache at /workspace/pip-cache
#   * env: HF_TOKEN (read private source + push)
#
# Usage on the pod:
#   export HF_TOKEN=hf_... ; export DST_MODEL=jdfelo/gemma-4-31B-v2-NVFP4
#   bash /workspace/quantize/run_pod.sh
#
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

VOLUME="${VOLUME:-/workspace}"
export HF_HOME="${HF_HOME:-${VOLUME}/hf-cache}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${VOLUME}/pip-cache}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
export PIP_BREAK_SYSTEM_PACKAGES=1
export DEBIAN_FRONTEND=noninteractive

SRC_MODEL="${SRC_MODEL:-jdfelo/gemma-4-31B-v2}"
DST_DIR="${DST_DIR:-${VOLUME}/gemma-4-31B-v2-NVFP4}"
DST_MODEL="${DST_MODEL:-}"
TORCH_PIN="${TORCH_PIN:-torch==2.11.0+cu128}"
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu128}"
PY="${PYTHON:-python3}"

: "${HF_TOKEN:?set HF_TOKEN (read private source + push)}"
export HF_TOKEN

echo "[gpu] $(date -u)  HF_HOME=${HF_HOME}  PIP_CACHE_DIR=${PIP_CACHE_DIR}"

# ---- 0. Python guard (no-op on runpod/pytorch; installs on a bare base) -----
if ! command -v "$PY" >/dev/null 2>&1; then
    echo "[gpu] installing python3.12 + pip"
    apt-get update -q && apt-get install -y -q python3.12 python3.12-venv python3-pip git >/dev/null
    PY=python3.12
fi
$PY -m pip install -q -U pip >/dev/null 2>&1 || true

# ---- 1. Remove bundled torchvision/torchaudio (the churn/ABI-break source) --
echo "[gpu] removing bundled torchvision/torchaudio (unused; text-only quant)"
$PY -m pip uninstall -y torchvision torchaudio 2>/dev/null || true

# ---- 2. Pin torch FIRST from the cu128 index (upgrades 2.9.1 -> 2.11.0) -----
echo "[gpu] installing ${TORCH_PIN}"
$PY -m pip install -q "${TORCH_PIN}" --index-url "${TORCH_INDEX}"

# ---- 3. Pinned llmcompressor stack (must NOT bump torch) --------------------
echo "[gpu] installing pinned llmcompressor stack"
$PY -m pip install -q -r "${HERE}/requirements-quant.txt"

# ---- 4. Verify the pins held (the two things that broke before) -------------
$PY - <<'PY' || { echo "[gpu] PIN VERIFY FAILED — aborting before GPU time"; exit 1; }
import torch, transformers, llmcompressor
assert torch.__version__.startswith("2.11.0"), f"torch bumped: {torch.__version__}"
print(f"[gpu] torch {torch.__version__} | transformers {transformers.__version__} | llmcompressor {llmcompressor.__version__}")
try:
    import torchvision
    raise SystemExit(f"[gpu] ABORT: torchvision still present ({torchvision.__version__})")
except ImportError:
    print("[gpu] torchvision absent (correct)")
from transformers import Gemma4ForConditionalGeneration  # must import without torchvision
print("[gpu] Gemma4ForConditionalGeneration import OK")
PY

# ---- 5. Quantize (reads staged bf16 from HF_HOME; writes NVFP4 to DST_DIR) --
PUSH=(); [ -n "${DST_MODEL}" ] && PUSH=(--push-repo "${DST_MODEL}")
echo "[gpu] launching NVFP4 quantization"
$PY "${HERE}/quantize_nvfp4.py" --src "${SRC_MODEL}" --dst-dir "${DST_DIR}" "${PUSH[@]}"

echo "[gpu] $(date -u)  DONE. NVFP4 checkpoint at: ${DST_DIR}"
du -sh "${DST_DIR}" 2>/dev/null || true
