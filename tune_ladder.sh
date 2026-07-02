#!/usr/bin/env bash
#
# Single-stream tok/s tuning ladder for gemma-4-31B-v2-NVFP4 on vLLM (Blackwell).
#
# For each config: launch `vllm serve`, wait for /health, run benchmark.py
# (decode tok/s @ concurrency 1), tear down, move on. Prints a summary table.
# One bad rung (unknown flag) is isolated — it logs and the sweep continues.
#
# Usage (on the pod, model pulls from HF on first launch):
#   # Phase 1 — STANDARD: stable vllm/vllm-openai:v0.24.0 image
#   MODEL=jdfelo/gemma-4-31B-v2-NVFP4 bash tune_ladder.sh
#   # Phase 2 — REACH: DFlash (~1.5x over EAGLE3) on a vLLM NIGHTLY image only
#   RUN_DFLASH=1 bash tune_ladder.sh
#
# Pre-flight (do ONCE before running, to catch version-renamed flags):
#   vllm serve --help | grep -E "kv-cache-dtype|async-scheduling|speculative-config|limit-mm-per-prompt|max-model-len"
#
set -uo pipefail

MODEL="${MODEL:-jdfelo/gemma-4-31B-v2-NVFP4}"
EAGLE="${EAGLE:-RedHatAI/gemma-4-31B-it-speculator.eagle3}"
PORT="${PORT:-8000}"
BASE="http://localhost:${PORT}"
TRIALS="${TRIALS:-10}"
MAXTOK="${MAXTOK:-512}"
STARTUP_TRIES="${STARTUP_TRIES:-240}"   # x3s = 12 min max cold start (first pull can be slow)
RESULTS="${RESULTS:-/workspace/tune_results}"
BENCH="${BENCH:-$(cd "$(dirname "$0")" && pwd)/benchmark.py}"
mkdir -p "$RESULTS"

# Rock-solid flags only — shared by EVERY rung (a bad one here kills all rungs).
COMMON_ARGS=(--max-model-len 32768 --gpu-memory-utilization 0.90 --port "$PORT")

# EAGLE3 speculative-decode JSON (no spaces; json.loads on the vLLM side).
SPEC()  { echo "{\"model\":\"$EAGLE\",\"num_speculative_tokens\":$1,\"method\":\"eagle3\"}"; }
# DFlash (block-diffusion) speculative-decode JSON. block_size=8 in the head,
# so num_speculative_tokens=8 is the designed config. NVIDIA: ~1.5x over EAGLE3.
DFLASH="${DFLASH:-RedHatAI/gemma-4-31B-it-speculator.dflash}"
SPECDF(){ echo "{\"model\":\"$DFLASH\",\"num_speculative_tokens\":$1,\"method\":\"dflash\"}"; }
NOIMG='{"image":0}'

# benchmark client needs httpx
python3 -c "import httpx" 2>/dev/null || pip install -q httpx || true

run_cfg () {
  local label="$1"; shift
  local -a extra=("$@")
  local log="$RESULTS/serve_${label}.log"
  local res="$RESULTS/bench_${label}.txt"
  echo
  echo "==================== ${label} ===================="
  echo "flags: ${COMMON_ARGS[*]} ${extra[*]}"

  vllm serve "$MODEL" "${COMMON_ARGS[@]}" "${extra[@]}" >"$log" 2>&1 &
  local pid=$!

  local ok=0 i
  for ((i=0; i<STARTUP_TRIES; i++)); do
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "[!] vllm exited during startup — tail $log:"; tail -15 "$log"; break
    fi
    if curl -sf "$BASE/health" >/dev/null 2>&1; then ok=1; break; fi
    sleep 3
  done

  if [ "$ok" = 1 ]; then
    echo "[ready] benchmarking $label…"
    # confirm what ACTUALLY engaged (catch silent fallbacks that fake a number)
    echo "--- engaged (from $log) ---"
    grep -iE "Using .*(FlashInfer|Flash Attention|backend)" "$log" | tail -1
    grep -iE "async scheduling (enabled|disabled)|Disabling async" "$log" | tail -1
    grep -iE "Speculative|eagle|num_speculative|draft" "$log" | grep -ivE "warning" | tail -2
    grep -iE "Capturing|CUDA graph|cudagraph|enforce_eager" "$log" | tail -1
    grep -iE "kv.?cache.*(fp8|dtype)|GPU KV cache size|Maximum concurrency" "$log" | tail -2
    echo "----------------------------"
    python3 "$BENCH" --base-url "$BASE" --api-key "${VLLM_API_KEY:-EMPTY}" --trials "$TRIALS" --max-tokens "$MAXTOK" --warmup 3 2>&1 | tee "$res"
  else
    echo "[!] $label never became ready — skipped (see $log)" | tee "$res"
  fi

  # teardown: TERM, then KILL, then wait for port to free
  kill "$pid" 2>/dev/null
  for ((i=0; i<40; i++)); do kill -0 "$pid" 2>/dev/null || break; sleep 2; done
  kill -9 "$pid" 2>/dev/null
  pkill -9 -f "vllm serve" 2>/dev/null
  for ((i=0; i<30; i++)); do curl -sf "$BASE/health" >/dev/null 2>&1 || break; sleep 2; done
  sleep 5
}

# ---- the ladder (each rung adds one lever) ----------------------------------
run_cfg baseline
run_cfg fp8kv      --kv-cache-dtype fp8
run_cfg async      --kv-cache-dtype fp8 --async-scheduling
run_cfg eagle3_k3  --kv-cache-dtype fp8 --async-scheduling --limit-mm-per-prompt "$NOIMG" --speculative-config "$(SPEC 3)"
run_cfg eagle3_k2  --kv-cache-dtype fp8 --async-scheduling --limit-mm-per-prompt "$NOIMG" --speculative-config "$(SPEC 2)"
run_cfg eagle3_k5  --kv-cache-dtype fp8 --async-scheduling --limit-mm-per-prompt "$NOIMG" --speculative-config "$(SPEC 5)"
# DFlash needs vLLM NIGHTLY (not the stable v0.24.0 image) — RedHat card says so,
# and v0.24.0 registers the drafter only as qwen3_dflash. Off by default; run the
# stable ladder first, then: RUN_DFLASH=1 bash tune_ladder.sh  (on a nightly image).
if [ "${RUN_DFLASH:-0}" = 1 ]; then
  run_cfg dflash_k8 --kv-cache-dtype fp8 --async-scheduling --limit-mm-per-prompt "$NOIMG" --speculative-config "$(SPECDF 8)"
fi

# ---- summary ----------------------------------------------------------------
echo
echo "==================== SUMMARY (decode tok/s @ concurrency 1) ===================="
printf "%-12s %-40s %s\n" "config" "decode tok/s" "spec accept"
for f in "$RESULTS"/bench_*.txt; do
  [ -e "$f" ] || continue
  lbl=$(basename "$f" .txt | sed 's/^bench_//')
  line=$(grep "decode tok/s :" "$f" 2>/dev/null | head -1 | sed 's/decode tok\/s ://')
  acc=$(grep "EAGLE3 accept:" "$f" 2>/dev/null | head -1 | sed 's/EAGLE3 accept://')
  printf "%-12s %-40s %s\n" "$lbl" "${line:-no result}" "${acc:-}"
done
