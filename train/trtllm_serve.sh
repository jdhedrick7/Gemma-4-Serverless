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

TARGET="${TARGET:-/workspace/gemma4_v2_text}"          # bf16 target (sidesteps NVFP4 loader #12764)
DRAFT="${DRAFT:-/workspace/dflash_ft_vllm}"
KDRAFT="${KDRAFT:-8}"                                   # max_draft_len (block_size)
PORT="${PORT:-8000}"
CFG=/workspace/trtllm_dflash.yaml

command -v trtllm-serve >/dev/null 2>&1 || { echo "[x] trtllm-serve not on PATH — run inside nvcr.io/nvidia/tensorrt-llm/release:<latest>"; exit 1; }
echo "trtllm: $(python3 -c 'import tensorrt_llm; print(tensorrt_llm.__version__)' 2>/dev/null || echo '?')"

# 1) wire Gemma4 aux-capture (idempotent; no-op if already patched or upstream added it)
python3 "$(dirname "$0")/patch_trtllm_gemma4.py" || { echo "[x] patch failed"; exit 1; }

# 2) DFlash spec config — target_layer_ids from the head's aux_hidden_state_layer_ids
cat > "$CFG" <<YAML
speculative_config:
  decoding_type: DFlash
  max_draft_len: ${KDRAFT}
  speculative_model: ${DRAFT}
  target_layer_ids: [1, 17, 29, 47, 58]
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
  --max_batch_size 1 --max_seq_len 4096 \
  --config "$CFG" > /workspace/trtllm_serve.log 2>&1 &
echo "[serve] pid=$! log=/workspace/trtllm_serve.log"
echo "[next] wait for 'Application startup complete' in the log, then:"
echo "       python3 benchmark.py --base-url http://localhost:$PORT --model $TARGET --trials 10 --max-tokens 512 --warmup 3"
