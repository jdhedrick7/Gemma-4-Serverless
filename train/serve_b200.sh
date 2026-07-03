#!/usr/bin/env bash
# Restart trtllm-serve on B200: gemma-4-31B-v2 NVFP4 text target + RedHat DFlash
# base drafter, in-engine spec decode. Detached; own argv never matches the
# pkill pattern (bracket trick). Env sourced for OPAL_PREFIX + tensorrt libs.
set -u
source /etc/profile.d/zz_ngc_env.sh 2>/dev/null || true
export OPAL_PREFIX="${OPAL_PREFIX:-/opt/hpcx/ompi}"
export OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1

TARGET="${TARGET:-/workspace/gemma4_v2_nvfp4_text}"
CFG="${CFG:-/workspace/setup/gemma4_dflash_b200.yaml}"
PORT="${PORT:-8000}"

pkill -9 -f "[t]rtllm-serve" 2>/dev/null
sleep 3
cd /workspace/setup
nohup trtllm-serve "$TARGET" \
  --host 0.0.0.0 --port "$PORT" --backend pytorch \
  --trust_remote_code \
  --config "$CFG" </dev/null > /workspace/serve.log 2>&1 &
disown
sleep 3
echo "serve pid(s): $(pgrep -f '[t]rtllm-serve' | tr '\n' ' ')"
echo "log: /workspace/serve.log"
