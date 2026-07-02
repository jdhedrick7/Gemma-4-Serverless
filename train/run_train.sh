#!/usr/bin/env bash
# Finetune the DFlash head on v2's own greedy outputs (warm start from the
# RedHat head). Waits for gen_data.py to finish (DONE marker) so it can be
# queued while generation still owns the GPU.
#
# Usage (pod): bash train/run_train.sh
set -uo pipefail

DATA=/workspace/train_data/v2_greedy.jsonl
GENLOG=/workspace/gen_data.log
TARGET=/workspace/gemma4_v2_text
INIT=/workspace/dflash_init
OUT=/workspace/dflash_ft
SF=/workspace/SpecForge

echo "[wait] for data-gen DONE marker in $GENLOG"
for ((i=0; i<720; i++)); do   # up to 6 h
  grep -q "^DONE" "$GENLOG" 2>/dev/null && break
  if ! pgrep -f "train/gen_data.py" >/dev/null 2>&1; then
    grep -q "^DONE" "$GENLOG" 2>/dev/null && break
    echo "[!] gen_data.py not running and no DONE marker — tail:"; tail -5 "$GENLOG"
    # proceed anyway if we have a usable dataset (>30k rows)
    n=$(wc -l < "$DATA" 2>/dev/null || echo 0)
    [ "$n" -ge 30000 ] && { echo "[i] proceeding with $n rows"; break; } || exit 1
  fi
  sleep 30
done
echo "[data] rows: $(wc -l < "$DATA")"

# wait for VRAM to free after gen engine exits
for ((i=0; i<60; i++)); do
  u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  [ "${u:-99999}" -lt 5000 ] && break
  sleep 5
done

. /workspace/sfvenv/bin/activate
cd "$SF"
export HF_HOME=/workspace/.huggingface
# cross_entropy over full 262144 vocab OOM'd at bs8; expandable_segments cuts
# fragmentation, bs2+accum4 keeps effective batch 8 within 178 GB.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# venv python explicitly: `torchrun` resolves to system Python where SpecForge
# (installed --no-deps into the venv) is invisible -> ModuleNotFoundError.
/workspace/sfvenv/bin/python -m torch.distributed.run --nproc_per_node=1 --master_port 29571 scripts/train_dflash.py \
  --target-model-path "$TARGET" \
  --target-model-backend hf \
  --draft-config-path "$INIT" \
  --draft-init-path "$INIT" \
  --train-data-path "$DATA" \
  --chat-template gemma4 \
  --attention-backend sdpa \
  --block-size 8 \
  --batch-size 2 \
  --accumulation-steps 4 \
  --num-epochs 1 \
  --learning-rate 2e-4 \
  --warmup-ratio 0.02 \
  --max-length 2048 \
  --log-interval 20 \
  --save-interval 2000 \
  --report-to none \
  --output-dir "$OUT" \
  --cache-dir /workspace/.cache/specforge
