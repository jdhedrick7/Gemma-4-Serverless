#!/usr/bin/env bash
# Launch Gemma-4-31B-v2 NVFP4 on vLLM (OpenAI server) + RunPod /ping sidecar,
# tuned for MAX SINGLE-STREAM DECODE tok/s on 1x B200 (Blackwell, sm_100).
#
# Throughput stack (all defaults below):
#   * NVFP4 W4A4 weights        -> native B200 FP4 tensor cores (~23 GB, TP1)
#   * EAGLE3 speculative decode  -> RedHatAI draft head, the #1 single-stream lever
#   * FP8 KV cache               -> halves KV bandwidth on Blackwell
#   * FlashInfer attention       -> vLLM default on Blackwell (forced explicit)
#   * CUDA graphs (batch=1)      -> max-num-seqs 1 captures the tight decode path
#   * text-only (image=0,audio=0)-> skips vision profiling; vision tower is bf16
#
# LB routes external traffic to $PORT (vLLM OpenAI API) and probes /ping on
# $PORT_HEALTH (our sidecar), which returns 204 while weights load, then 200.
set -euo pipefail

# ---- Model ---------------------------------------------------------------
# Default to the public NVFP4 repo; override to a local /runpod-volume path
# or the bf16 source for A/B. SERVED_MODEL_NAME is the id clients call.
MODEL_NAME="${MODEL_NAME:-jdfelo/gemma-4-31B-v2-NVFP4}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-${MODEL_NAME}}"

# ---- Ports (RunPod injects; keep in sync with Dockerfile) ----------------
PORT="${PORT:-8000}"
PORT_HEALTH="${PORT_HEALTH:-8080}"

# ---- Single-GPU: no tensor parallel (23 GB NVFP4 fits 180 GB B200) --------
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"

# ---- PROFILE: one switch picks the whole tuning set ----------------------
# latency    = single-stream decode max (default): EAGLE3 spec, fp8 KV, seqs=1.
# throughput = concurrent serving: no spec, bigger batches, more seqs.
PROFILE="${PROFILE:-latency}"
_EAGLE3='{"model": "RedHatAI/gemma-4-31B-it-speculator.eagle3", "num_speculative_tokens": 3, "method": "eagle3"}'
case "${PROFILE,,}" in
  latency)
    _GPU_MEM="0.90"; _MAX_MODEL_LEN="32768"; _MAX_SEQS="1"; _MAX_BATCHED="8192"; _SPEC="${_EAGLE3}" ;;
  throughput)
    _GPU_MEM="0.92"; _MAX_MODEL_LEN="16384"; _MAX_SEQS="256"; _MAX_BATCHED="16384"; _SPEC="" ;;
  *)
    echo "[start] unknown PROFILE='${PROFILE}' (use latency|throughput)" >&2; exit 2 ;;
esac

# Per-knob env overrides win over the profile default.
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-$_GPU_MEM}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-$_MAX_MODEL_LEN}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-$_MAX_SEQS}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-$_MAX_BATCHED}"
SPECULATIVE_CONFIG="${SPECULATIVE_CONFIG:-$_SPEC}"

# KV cache dtype. fp8 = B200 native, halves KV bandwidth. Set '' for auto(bf16).
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8}"
# Text-only by default: skip the vision/audio encoders' profiling + KV budget.
# Set MULTIMODAL=true to serve images (vision tower is bf16 in the checkpoint).
MULTIMODAL="${MULTIMODAL:-false}"
# Attention backend. FlashInfer is vLLM's Blackwell default; force it explicit.
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASHINFER}"

# ---- HF cache (cached-models / network volume) ---------------------------
export HF_HOME="${HF_HOME:-/runpod-volume/huggingface-cache}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"

# ---- Start health sidecar (backgrounded) ---------------------------------
python3 /health_server.py &
HEALTH_PID=$!
term() { kill -TERM "$HEALTH_PID" "${VLLM_PID:-}" 2>/dev/null || true; }
trap term TERM INT

# ---- Assemble vLLM args --------------------------------------------------
ARGS=(
  --model "$MODEL_NAME"
  --served-model-name "$SERVED_MODEL_NAME"
  --host 0.0.0.0
  --port "$PORT"
  --tensor-parallel-size "$TENSOR_PARALLEL_SIZE"
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
  --max-model-len "$MAX_MODEL_LEN"
  --max-num-seqs "$MAX_NUM_SEQS"
  --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"
  --async-scheduling
)

# Text-only: skip vision+audio encoder allocation entirely (latency + VRAM win).
if [ "${MULTIMODAL,,}" != "true" ]; then
  ARGS+=( --limit-mm-per-prompt '{"image":0,"audio":0}' )
fi

# KV cache dtype (fp8 on B200).
if [ -n "$KV_CACHE_DTYPE" ]; then
  ARGS+=( --kv-cache-dtype "$KV_CACHE_DTYPE" )
fi

# Speculative decoding: pass the JSON as ONE arg (never word-split it).
if [ -n "$SPECULATIVE_CONFIG" ]; then
  ARGS+=( --speculative-config "$SPECULATIVE_CONFIG" )
fi

# Escape hatch: append any additional vLLM engine args verbatim.
if [ -n "${EXTRA_VLLM_ARGS:-}" ]; then
  # shellcheck disable=SC2206
  ARGS+=( ${EXTRA_VLLM_ARGS} )
fi

echo "[start] Launching vLLM: profile=$PROFILE model=$MODEL_NAME TP=$TENSOR_PARALLEL_SIZE ctx=$MAX_MODEL_LEN seqs=$MAX_NUM_SEQS mm=$MULTIMODAL spec=${SPECULATIVE_CONFIG:+eagle3} kv=${KV_CACHE_DTYPE:-auto} attn=$VLLM_ATTENTION_BACKEND port=$PORT health=$PORT_HEALTH"

python3 -m vllm.entrypoints.openai.api_server "${ARGS[@]}" &
VLLM_PID=$!
wait "$VLLM_PID"
