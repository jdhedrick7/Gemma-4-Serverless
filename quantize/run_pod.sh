#!/usr/bin/env bash
#
# One-shot bootstrap for the B200 quantization pod.
#
# Prereqs on the pod:
#   * 1x B200 (180 GB), attached network volume mounted at /workspace
#   * base image vllm/vllm-openai:gemma4 (torch + CUDA 12.9 + transformers 5.5.3)
#   * env: HF_TOKEN (required, to read the private source + push the quant)
#
# What it does:
#   1. Pins HF cache onto the network volume (persists across pod restarts).
#   2. Installs the quant toolchain (transformers 5.8.1 + llm-compressor git main).
#   3. Runs NVFP4 quantization -> /workspace/gemma-4-31B-v2-NVFP4.
#   4. Optionally pushes to a private HF repo (DST_MODEL) for clean serving.
#
# Usage on the pod:
#   export HF_TOKEN=hf_...              # required
#   export DST_MODEL=jdfelo/gemma-4-31B-v2-NVFP4   # optional: push target
#   bash run_pod.sh
#
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

VOLUME="${VOLUME:-/workspace}"
SRC_MODEL="${SRC_MODEL:-jdfelo/gemma-4-31B-v2}"
DST_DIR="${DST_DIR:-${VOLUME}/gemma-4-31B-v2-NVFP4}"
DST_MODEL="${DST_MODEL:-}"                 # empty = don't push
NUM_CALIBRATION_SAMPLES="${NUM_CALIBRATION_SAMPLES:-512}"
MAX_SEQUENCE_LENGTH="${MAX_SEQUENCE_LENGTH:-2048}"

: "${HF_TOKEN:?set HF_TOKEN (needs read on the private source repo + write to push)}"

# ---- HF cache on the volume (persist downloads, count against the 200 GB) ----
export HF_HOME="${HF_HOME:-${VOLUME}/hf-cache}"
export HF_HUB_ENABLE_HF_TRANSFER=1
mkdir -p "${HF_HOME}"
echo "[run_pod] HF_HOME=${HF_HOME}"
echo "[run_pod] source=${SRC_MODEL}  dst_dir=${DST_DIR}  push=${DST_MODEL:-<none>}"

# ---- free space guard: need ~90 GB (src 62.5 + nvfp4 ~18 + cache) ------------
avail_gb="$(df -BG --output=avail "${VOLUME}" 2>/dev/null | tail -1 | tr -dc '0-9' || echo 0)"
echo "[run_pod] volume free: ${avail_gb} GB"
if [ "${avail_gb:-0}" -lt 90 ]; then
    echo "[run_pod] WARNING: <90 GB free on ${VOLUME}; quant may fail to stage weights." >&2
fi

# ---- toolchain --------------------------------------------------------------
echo "[run_pod] installing quant toolchain…"
pip install --no-cache-dir -r "${HERE}/requirements-quant.txt"

# ---- run --------------------------------------------------------------------
echo "[run_pod] starting NVFP4 quantization…"
PUSH_ARGS=()
[ -n "${DST_MODEL}" ] && PUSH_ARGS+=(--push-repo "${DST_MODEL}")

python3 "${HERE}/quantize_nvfp4.py" \
    --src "${SRC_MODEL}" \
    --dst-dir "${DST_DIR}" \
    --calib-dataset "${CALIB_DATASET:-HuggingFaceH4/ultrachat_200k}" \
    --num-samples "${NUM_CALIBRATION_SAMPLES}" \
    --max-seq-len "${MAX_SEQUENCE_LENGTH}" \
    "${PUSH_ARGS[@]}"

echo "[run_pod] DONE. NVFP4 checkpoint at: ${DST_DIR}"
du -sh "${DST_DIR}" 2>/dev/null || true
echo "[run_pod] next: serve it (see ../serve) — set MODEL to ${DST_MODEL:-${DST_DIR}}"
