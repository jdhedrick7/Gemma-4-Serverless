#!/usr/bin/env bash
# One-shot, self-detaching restart of data-gen (new fp8 config) + train waiter.
# Runs under its own setsid so it survives the launching SSH session dropping
# (pod SSH has been unstable under load). Group-kills the old gen (parent +
# vLLM EngineCore child) properly, waits for VRAM, relaunches both.
set -uo pipefail
LOG=/workspace/restart_gen.log
exec >"$LOG" 2>&1
echo "[restart] $(date)"

# kill any gen / train waiters by PROCESS GROUP (negative pid), plus EngineCore
for p in $(pgrep -f "gen_data.py|run_train.sh"); do
  pgid=$(ps -o pgid= -p "$p" 2>/dev/null | tr -d ' ')
  [ -n "$pgid" ] && kill -9 -- "-$pgid" 2>/dev/null || true
done
pkill -9 -f "multiprocessing.spawn" 2>/dev/null || true
pkill -9 -f "VLLM::EngineCore" 2>/dev/null || true   # offline LLM() engine proctitle (no "vllm"/"spawn" substring)
pkill -9 -f "gen_data.py" 2>/dev/null || true

# wait for VRAM to free (up to ~2 min)
for i in $(seq 1 60); do
  u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  echo "[restart] vram=$u"
  [ "${u:-99999}" -lt 5000 ] && break
  sleep 2
done

cd /workspace/Gemma-4-Serverless
export HF_HOME=/workspace/.huggingface   # public model+datasets: no token needed
echo "[restart] rows before: $(wc -l < /workspace/train_data/v2_greedy.jsonl)"

setsid bash -c 'HF_HOME='"$HF_HOME"' python3 train/gen_data.py > /workspace/gen_data.log 2>&1' &
echo "[restart] gen relaunched pid=$!"
setsid bash -c 'bash train/run_train.sh > /workspace/train.log 2>&1' &
echo "[restart] train waiter relaunched pid=$!"
echo "[restart] done $(date)"
