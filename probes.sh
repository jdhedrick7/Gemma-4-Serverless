#!/usr/bin/env bash
#
# Post-ladder probes toward max single-stream tok/s (goal: 1000 on B200).
# Ladder findings this builds on (2026-07-02, vLLM v0.24.0, 1x B200):
#   baseline 140.5 | fp8kv 126.3 (-10%!) | async 134.2 | e3_k3 239 | e3_k5 251.6
# Probes:
#   e5_nofp8   winner minus the fp8-KV regression (KV auto)
#   e7_nofp8   deeper speculation (acceptance was monotone k2<k3<k5)
#   e5_fi      + VLLM_ATTENTION_BACKEND=FLASHINFER (TRITON_ATTN was a fallback;
#              Blackwell trtllm-gen decode kernels support head_dim 256)
#   e7_fi      both levers
#   peak_<w>   winner re-run WITH FlashInfer autotune = the ceiling number
#
# Usage: MODEL=jdfelo/gemma-4-31B-v2-NVFP4 bash probes.sh
set -uo pipefail

MODEL="${MODEL:-jdfelo/gemma-4-31B-v2-NVFP4}"
EAGLE="${EAGLE:-RedHatAI/gemma-4-31B-it-speculator.eagle3}"
PORT="${PORT:-8000}"
BASE="http://localhost:${PORT}"
TRIALS="${TRIALS:-10}"
MAXTOK="${MAXTOK:-512}"
STARTUP_TRIES="${STARTUP_TRIES:-200}"
PEAK_STARTUP_TRIES="${PEAK_STARTUP_TRIES:-600}"
RESULTS="${RESULTS:-/workspace/probe_results}"
BENCH="${BENCH:-$(cd "$(dirname "$0")" && pwd)/benchmark.py}"
PEAK_RERUN="${PEAK_RERUN:-1}"
mkdir -p "$RESULTS"
if ls "$RESULTS"/bench_*.txt >/dev/null 2>&1; then
  old="$RESULTS/prev_$(date +%s)"; mkdir -p "$old"
  mv "$RESULTS"/bench_*.txt "$RESULTS"/serve_*.log "$old"/ 2>/dev/null
  echo "[i] archived previous results to $old"
fi

export HF_HOME="${HF_HOME:-/workspace/.huggingface}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-/workspace/.cache/vllm}"
mkdir -p "$HF_HOME" "$VLLM_CACHE_ROOT"

COMMON_ARGS=(--max-model-len 32768 --gpu-memory-utilization 0.90 --port "$PORT")
SPEC()  { echo "{\"model\":\"$EAGLE\",\"num_speculative_tokens\":$1,\"method\":\"eagle3\"}"; }
NOIMG='{"image":0}'

python3 -c "import httpx" 2>/dev/null || pip install -q httpx || true

port_busy () { curl -s -o /dev/null --max-time 2 "$BASE/health" 2>/dev/null; [ $? -ne 7 ]; }
purge_stale () {
  pkill -9 -f "vllm serve" 2>/dev/null
  pkill -9 -f "from multiprocessing.spawn import spawn_main" 2>/dev/null
  sleep 2
}
if port_busy; then
  echo "[!] :$PORT busy — purging stale vllm/EngineCore"; purge_stale
  port_busy && { echo "[x] :$PORT still occupied — aborting"; exit 1; }
fi
CURRENT_PGID=""
cleanup () { [ -n "$CURRENT_PGID" ] && { kill -9 -- "-$CURRENT_PGID" 2>/dev/null; CURRENT_PGID=""; }; }
trap cleanup EXIT
trap 'echo "[!] interrupted — tearing down"; cleanup; exit 130' INT TERM HUP

# run_cfg <autotune 0|1> <attn AUTO|FLASHINFER> <label> [flags...]
run_cfg () {
  local autotune="$1" attn="$2" label="$3"; shift 3
  local -a extra=("$@")
  local tries="$STARTUP_TRIES"
  if [ "$autotune" = 0 ]; then extra+=(--no-enable-flashinfer-autotune); else tries="$PEAK_STARTUP_TRIES"; fi
  local -a envp=()
  [ "$attn" = "FLASHINFER" ] && envp=(env VLLM_ATTENTION_BACKEND=FLASHINFER)
  local log="$RESULTS/serve_${label}.log"
  local res="$RESULTS/bench_${label}.txt"
  echo
  echo "==================== ${label} ===================="
  echo "attn=$attn autotune=$autotune flags: ${COMMON_ARGS[*]} ${extra[*]}"

  setsid ${envp[@]+"${envp[@]}"} vllm serve "$MODEL" "${COMMON_ARGS[@]}" "${extra[@]}" >"$log" 2>&1 &
  local pid=$!
  CURRENT_PGID="$pid"

  local ok=0 i
  for ((i=0; i<tries; i++)); do
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "[!] vllm exited during startup — tail $log:"; tail -15 "$log"; break
    fi
    if curl -sf "$BASE/health" >/dev/null 2>&1; then ok=1; break; fi
    sleep 3
  done

  if [ "$ok" = 1 ]; then
    echo "[ready] benchmarking ${label}..."
    echo "--- engaged (from $log) ---"
    grep -iE "Using .* attention backend|FLASHINFER|TRITON_ATTN" "$log" | tail -2
    grep -iE "Skipping FlashInfer autotune|Autotuning process starts" "$log" | tail -1
    grep -iE "Speculative|num_speculative" "$log" | grep -ivE "warning" | tail -1
    echo "----------------------------"
    python3 "$BENCH" --base-url "$BASE" --model "$MODEL" --api-key "${VLLM_API_KEY:-EMPTY}" \
      --trials "$TRIALS" --max-tokens "$MAXTOK" --warmup 3 2>&1 | tee "$res"
  else
    echo "[!] $label never became ready — skipped (see $log)" | tee "$res"
  fi

  kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null
  for ((i=0; i<20; i++)); do kill -0 "$pid" 2>/dev/null || break; sleep 2; done
  kill -9 -- "-$pid" 2>/dev/null
  wait "$pid" 2>/dev/null
  CURRENT_PGID=""
  nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null \
    | while read -r gp; do [ -n "$gp" ] && kill -9 "$gp" 2>/dev/null; done
  local u=0
  for ((i=0; i<45; i++)); do
    u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
    [ "${u:-99999}" -lt 5000 ] 2>/dev/null && break
    sleep 2
  done
  if [ "${u:-0}" -ge 5000 ] 2>/dev/null; then
    echo "[!] WARNING: ${u} MiB VRAM still held after teardown — next rung may OOM"
  fi
  sleep 3
}

RUNG_LABELS=(); RUNG_ATTN=(); RUNG_FLAGS=()
rung () {
  local attn="$1" label="$2"; shift 2
  RUNG_LABELS+=("$label"); RUNG_ATTN+=("$attn"); RUNG_FLAGS+=("$*")
  run_cfg 0 "$attn" "$label" "$@"
}

# ---- probes -------------------------------------------------------------------
rung AUTO       e5_nofp8 --async-scheduling --limit-mm-per-prompt "$NOIMG" --speculative-config "$(SPEC 5)"
rung AUTO       e7_nofp8 --async-scheduling --limit-mm-per-prompt "$NOIMG" --speculative-config "$(SPEC 7)"
rung FLASHINFER e5_fi    --async-scheduling --limit-mm-per-prompt "$NOIMG" --speculative-config "$(SPEC 5)"
rung FLASHINFER e7_fi    --async-scheduling --limit-mm-per-prompt "$NOIMG" --speculative-config "$(SPEC 7)"

# ---- winner + autotuned peak ----------------------------------------------------
mean_of () { awk '/decode tok\/s :/ {for(i=1;i<=NF;i++) if($i=="mean"){print $(i+1); exit}}' "$1" 2>/dev/null; }
best_label="" best_mean=0 best_i=-1
for i in "${!RUNG_LABELS[@]}"; do
  m=$(mean_of "$RESULTS/bench_${RUNG_LABELS[$i]}.txt"); [ -n "$m" ] || continue
  if awk -v a="$m" -v b="$best_mean" 'BEGIN{exit !(a>b)}'; then
    best_mean="$m"; best_label="${RUNG_LABELS[$i]}"; best_i="$i"
  fi
done
if [ -n "$best_label" ] && [ "$PEAK_RERUN" = 1 ]; then
  echo
  echo ">>> probe winner: $best_label (${best_mean} tok/s) — autotuned peak rerun"
  # shellcheck disable=SC2086
  run_cfg 1 "${RUNG_ATTN[$best_i]}" "peak_${best_label}" ${RUNG_FLAGS[$best_i]}
fi

echo
echo "==================== PROBE SUMMARY (decode tok/s @ concurrency 1) ===================="
printf "%-16s %s\n" "config" "decode tok/s"
for f in "$RESULTS"/bench_*.txt; do
  [ -e "$f" ] || continue
  lbl=$(basename "$f" .txt | sed 's/^bench_//')
  line=$(grep "decode tok/s :" "$f" 2>/dev/null | head -1 | sed 's/decode tok\/s ://')
  printf "%-16s %s\n" "$lbl" "${line:-no result}"
done
[ -n "$best_label" ] && echo "probe winner: $best_label"
