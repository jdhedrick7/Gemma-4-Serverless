#!/usr/bin/env bash
# TRT-LLM fallback: serve gemma-4 + our DFlash head with in-engine spec decode.
# Run on the B200 ONLY after vLLM finetune measured < 1000 (kill vLLM first — one GPU).
#
# Assumes the TRT-LLM release container is the runtime (it ships tensorrt_llm +
# trtllm-serve). If running on the bare pod instead, set TRTLLM_PIP=1 to pip-install.
#
# Usage:
#   DRAFT=/workspace/dflash_ft_vllm bash train/trtllm_serve.sh        # finetuned head
#   DRAFT=RedHatAI/gemma-4-31B-it-speculator.dflash bash train/trtllm_serve.sh  # stock
set -uo pipefail

# --- MPI env (VALIDATED live on rc20): SSH sessions bypass nvidia_entrypoint.sh,
# so OpenMPI can't find its runtime files -> `import tensorrt_llm` aborts at
# opal_init. OPAL_PREFIX relocates it; run-as-root vars needed (we're root).
export OPAL_PREFIX="${OPAL_PREFIX:-/opt/hpcx/ompi}"
export OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1

# TARGET: NVFP4 loader bug (#12764) is FIXED on rc20 (built from recent main), so
# the NVFP4 text-only extraction is the throughput target (~18GB, ~3.4x less HBM
# read than bf16 -> higher single-stream floor). Both bf16 and NVFP4 targets must
# be TEXT-ONLY (Gemma4ForCausalLM); the HF NVFP4 repo is multimodal, so extract.
TARGET="${TARGET:-/workspace/gemma4_v2_nvfp4_text}"    # text-only NVFP4 (see extract_text_nvfp4.py)
DRAFT="${DRAFT:-/workspace/dflash_ft_vllm}"
KDRAFT="${KDRAFT:-8}"                                   # max_draft_len (block_size)
PORT="${PORT:-8000}"
CFG=/workspace/trtllm_dflash.yaml

command -v trtllm-serve >/dev/null 2>&1 || { echo "[x] trtllm-serve not on PATH — run inside nvcr.io/nvidia/tensorrt-llm/release:<latest>"; exit 1; }
echo "trtllm: $(python3 -c 'import tensorrt_llm; print(tensorrt_llm.__version__)' 2>/dev/null || echo '?')"

# 1) wire Gemma4 aux-capture (idempotent; no-op if already patched or upstream added it)
python3 "$(dirname "$0")/patch_trtllm_gemma4.py" || { echo "[x] patch failed"; exit 1; }

# 2) DFlash spec config. VALIDATED live: DFLASH sends K+1 tokens/step but
# FlashInfer's decode path expects 1 token/seq, and Gemma4's head_dim=256
# auto-selects FlashInfer -> must force attn_backend: TRTLLM. mask_token_id
# from head config (4); target_layer_ids = head's aux_hidden_state_layer_ids.
cat > "$CFG" <<YAML
attn_backend: TRTLLM
speculative_config:
  decoding_type: DFlash
  max_draft_len: ${KDRAFT}
  speculative_model: ${DRAFT}
  target_layer_ids: [1, 17, 29, 47, 58]
  mask_token_id: 4
YAML
echo "[cfg] $CFG:"; cat "$CFG"

# 3) free the GPU (kill any vLLM) and wait for VRAM
pkill -9 -f "vllm serve" 2>/dev/null; pkill -9 -f "VLLM::EngineCore" 2>/dev/null
for i in $(seq 1 45); do
  u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  [ "${u:-99999}" -lt 5000 ] && break; sleep 2
done

# 4) serve (single-stream: batch 1). setsid so an SSH drop can't kill it.
echo "[serve] target=$TARGET draft=$DRAFT k=$KDRAFT port=$PORT"
setsid trtllm-serve serve "$TARGET" \
  --backend pytorch --host 0.0.0.0 --port "$PORT" \
  --max_batch_size 1 --max_seq_len 4096 --trust_remote_code \
  --config "$CFG" > /workspace/trtllm_serve.log 2>&1 &
echo "[serve] pid=$! log=/workspace/trtllm_serve.log"
echo "[next] wait for 'Application startup complete' in the log, then:"
echo "       python3 benchmark.py --base-url http://localhost:$PORT --model $TARGET --trials 10 --max-tokens 512 --warmup 3"
