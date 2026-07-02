#!/usr/bin/env bash
#
# Single-stream tok/s tuning ladder for gemma-4-31B-v2-NVFP4 on vLLM (Blackwell).
#
# Two phases in one run:
#   1) LADDER — each config launches `vllm serve` with FlashInfer autotune
#      DISABLED (--no-enable-flashinfer-autotune). v0.24.0 hardcodes the
#      autotune persistent cache OFF (_FLASHINFER_USE_PERSISTENT_CACHE=False,
#      vllm/model_executor/warmup/kernel_warmup.py), so with autotune on every
#      rung would re-pay a ~13 min fp4_gemm tune. Heuristic kernel tactics are
#      identical across rungs -> fair A/B, ~4-6 min startup per rung.
#   2) PEAK — the winning rung is re-run once WITH autotune (~+13 min startup)
#      for the true peak decode tok/s. Autotune cost paid exactly once.
#
# Teardown kills the whole process GROUP (setsid + kill -- -PGID): the
# EngineCore child is a `python -c multiprocessing.spawn` process whose cmdline
# does NOT contain "vllm", so pkill -f vllm misses it and it strands ~160 GB
# of VRAM (observed). Group-kill reaps it; a VRAM-free wait guards the next rung.
#
# Usage (on the pod, model pulls from HF on first launch):
#   MODEL=jdfelo/gemma-4-31B-v2-NVFP4 bash tune_ladder.sh
#   PEAK_RERUN=0 bash tune_ladder.sh          # ladder only, skip autotuned peak
#   RUN_DFLASH=1 bash tune_ladder.sh          # extra rung, vLLM NIGHTLY image only
#
# Pre-flight (do ONCE before running, to catch version-renamed flags):
#   vllm serve --help | grep -E "kv-cache-dtype|async-scheduling|speculative-config|limit-mm-per-prompt|max-model-len|enable-flashinfer-autotune"
#
set -uo pipefail

MODEL="${MODEL:-jdfelo/gemma-4-31B-v2-NVFP4}"
EAGLE="${EAGLE:-RedHatAI/gemma-4-31B-it-speculator.eagle3}"
PORT="${PORT:-8000}"
BASE="http://localhost:${PORT}"
TRIALS="${TRIALS:-10}"
MAXTOK="${MAXTOK:-512}"
# Separate startup budgets: ladder rungs skip autotune (~4-6 min incl. compile);
# the peak rerun adds the ~13 min FlashInfer autotune.
STARTUP_TRIES="${STARTUP_TRIES:-200}"            # x3s = 10 min per ladder rung
PEAK_STARTUP_TRIES="${PEAK_STARTUP_TRIES:-600}"  # x3s = 30 min for the autotuned peak
RESULTS="${RESULTS:-/workspace/tune_results}"
BENCH="${BENCH:-$(cd "$(dirname "$0")" && pwd)/benchmark.py}"
PEAK_RERUN="${PEAK_RERUN:-1}"
mkdir -p "$RESULTS"
# stale bench files from a previous crashed run would pollute winner selection
# and the summary — archive them out of the way.
if ls "$RESULTS"/bench_*.txt >/dev/null 2>&1; then
  old="$RESULTS/prev_$(date +%s)"; mkdir -p "$old"
  mv "$RESULTS"/bench_*.txt "$RESULTS"/serve_*.log "$old"/ 2>/dev/null
  echo "[i] archived previous results to $old"
fi

# Persist HF weights + vLLM compile/JIT caches on the network volume so pod
# stop/start (or a fresh pod on the same volume) skips re-download/re-compile.
export HF_HOME="${HF_HOME:-/workspace/.huggingface}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-/workspace/.cache/vllm}"
mkdir -p "$HF_HOME" "$VLLM_CACHE_ROOT"

# Rock-solid flags only — shared by EVERY rung (a bad one here kills all rungs).
COMMON_ARGS=(--max-model-len 32768 --gpu-memory-utilization 0.90 --port "$PORT")

# EAGLE3 speculative-decode JSON. IMPORTANT: all rung flags must stay
# SPACE-FREE tokens — the winner's flags are stored word-split for the peak
# rerun (see RUNG_FLAGS below).
SPEC()  { echo "{\"model\":\"$EAGLE\",\"num_speculative_tokens\":$1,\"method\":\"eagle3\"}"; }
# DFlash (block-diffusion) speculative-decode JSON. block_size=8 in the head,
# so num_speculative_tokens=8 is the designed config. NVIDIA: ~1.5x over EAGLE3.
DFLASH="${DFLASH:-RedHatAI/gemma-4-31B-it-speculator.dflash}"
SPECDF(){ echo "{\"model\":\"$DFLASH\",\"num_speculative_tokens\":$1,\"method\":\"dflash\"}"; }
NOIMG='{"image":0}'

# benchmark client needs httpx
python3 -c "import httpx" 2>/dev/null || pip install -q httpx || true

# ---- stale-server guard + crash traps ----------------------------------------
# If a previous crashed/interrupted run left a listener on $PORT, the next
# rung's /health probe would hit the STALE server and benchmark the wrong
# config. Probe the port; purge anything vllm-ish; abort if still occupied.
port_busy () { curl -s -o /dev/null --max-time 2 "$BASE/health" 2>/dev/null; [ $? -ne 7 ]; }  # rc 7 = conn refused = free
purge_stale () {
  pkill -9 -f "vllm serve" 2>/dev/null
  pkill -9 -f "from multiprocessing.spawn import spawn_main" 2>/dev/null  # EngineCore workers
  sleep 2
}
if port_busy; then
  echo "[!] something is already listening on :$PORT — purging stale vllm/EngineCore"
  purge_stale
  if port_busy; then
    echo "[x] :$PORT still occupied after purge — refusing to benchmark a stale server. Investigate: ss -ltnp | grep :$PORT"
    exit 1
  fi
fi
# On Ctrl-C / TERM / normal exit: reap the current rung's process group so a
# setsid'd vllm (immune to terminal SIGINT) can't outlive the ladder.
CURRENT_PGID=""
cleanup () {
  [ -n "$CURRENT_PGID" ] && { kill -9 -- "-$CURRENT_PGID" 2>/dev/null; CURRENT_PGID=""; }
}
trap cleanup EXIT
trap 'echo "[!] interrupted — tearing down"; cleanup; exit 130' INT TERM HUP

# AUTOTUNE=0 -> append --no-enable-flashinfer-autotune (ladder rungs)
# AUTOTUNE=1 -> autotune on (peak rerun)
run_cfg () {
  local autotune="$1" label="$2"; shift 2
  local -a extra=("$@")
  local tries="$STARTUP_TRIES"
  if [ "$autotune" = 0 ]; then
    extra+=(--no-enable-flashinfer-autotune)
  else
    tries="$PEAK_STARTUP_TRIES"
  fi
  local log="$RESULTS/serve_${label}.log"
  local res="$RESULTS/bench_${label}.txt"
  echo
  echo "==================== ${label} ===================="
  echo "flags: ${COMMON_ARGS[*]} ${extra[*]}"

  # setsid: new session + process group (pgid = pid). Script runs
  # non-interactively (no job control), so the fork-less setsid path applies
  # and $! is the vllm pid itself.
  setsid vllm serve "$MODEL" "${COMMON_ARGS[@]}" "${extra[@]}" >"$log" 2>&1 &
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
    # confirm what ACTUALLY engaged (catch silent fallbacks that fake a number)
    echo "--- engaged (from $log) ---"
    grep -iE "Using .*(FlashInfer|Flash Attention|backend)" "$log" | tail -1
    grep -iE "Skipping FlashInfer autotune|Autotuning process starts" "$log" | tail -1
    grep -iE "async scheduling (enabled|disabled)|Disabling async" "$log" | tail -1
    grep -iE "Speculative|eagle|num_speculative|draft" "$log" | grep -ivE "warning" | tail -2
    grep -iE "Capturing|CUDA graph|cudagraph|enforce_eager" "$log" | tail -1
    grep -iE "kv.?cache.*(fp8|dtype)|GPU KV cache size|Maximum concurrency" "$log" | tail -2
    echo "----------------------------"
    python3 "$BENCH" --base-url "$BASE" --model "$MODEL" --api-key "${VLLM_API_KEY:-EMPTY}" \
      --trials "$TRIALS" --max-tokens "$MAXTOK" --warmup 3 2>&1 | tee "$res"
  else
    echo "[!] $label never became ready — skipped (see $log)" | tee "$res"
  fi

  # ---- teardown: TERM the whole group, escalate, verify VRAM freed ----------
  kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null
  for ((i=0; i<20; i++)); do kill -0 "$pid" 2>/dev/null || break; sleep 2; done
  kill -9 -- "-$pid" 2>/dev/null
  wait "$pid" 2>/dev/null            # reap; silences bash "Terminated" job noise
  CURRENT_PGID=""
  # best-effort: reap anything still holding the GPU (container PID caveat)
  nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null \
    | while read -r gp; do [ -n "$gp" ] && kill -9 "$gp" 2>/dev/null; done
  # wait for VRAM to actually free (prevents next-rung OOM), up to ~90s
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

# rung: record label+flags for winner selection, run with autotune OFF.
# Parallel indexed arrays (portable to bash 3.x, unlike declare -A).
RUNG_LABELS=()
RUNG_FLAGS=()
rung () {
  local label="$1"; shift
  RUNG_LABELS+=("$label")
  RUNG_FLAGS+=("$*")     # safe: every rung flag is a space-free token
  run_cfg 0 "$label" "$@"
}
flags_of () {  # flags_of <label> -> echoes recorded flags
  local i
  for i in "${!RUNG_LABELS[@]}"; do
    [ "${RUNG_LABELS[$i]}" = "$1" ] && { echo "${RUNG_FLAGS[$i]}"; return; }
  done
}

# ---- phase 1: the ladder (each rung adds one lever) --------------------------
rung baseline
rung fp8kv      --kv-cache-dtype fp8
rung async      --kv-cache-dtype fp8 --async-scheduling
rung eagle3_k3  --kv-cache-dtype fp8 --async-scheduling --limit-mm-per-prompt "$NOIMG" --speculative-config "$(SPEC 3)"
rung eagle3_k2  --kv-cache-dtype fp8 --async-scheduling --limit-mm-per-prompt "$NOIMG" --speculative-config "$(SPEC 2)"
rung eagle3_k5  --kv-cache-dtype fp8 --async-scheduling --limit-mm-per-prompt "$NOIMG" --speculative-config "$(SPEC 5)"
# DFlash needs vLLM NIGHTLY (not the stable v0.24.0 image) — RedHat card says so,
# and v0.24.0 registers the drafter only as qwen3_dflash. Off by default; run the
# stable ladder first, then: RUN_DFLASH=1 bash tune_ladder.sh  (on a nightly image).
if [ "${RUN_DFLASH:-0}" = 1 ]; then
  rung dflash_k8 --kv-cache-dtype fp8 --async-scheduling --limit-mm-per-prompt "$NOIMG" --speculative-config "$(SPECDF 8)"
fi

# ---- pick the winner (highest mean decode tok/s across ladder rungs) ---------
mean_of () {  # extract "decode tok/s : mean N" from a bench file
  awk '/decode tok\/s :/ {for(i=1;i<=NF;i++) if($i=="mean"){print $(i+1); exit}}' "$1" 2>/dev/null
}
best_label="" best_mean=0
for lbl in "${RUNG_LABELS[@]}"; do
  m=$(mean_of "$RESULTS/bench_${lbl}.txt")
  [ -n "$m" ] || continue
  if awk -v a="$m" -v b="$best_mean" 'BEGIN{exit !(a>b)}'; then
    best_mean="$m"; best_label="$lbl"
  fi
done

# ---- phase 2: peak rerun of the winner WITH FlashInfer autotune --------------
if [ -n "$best_label" ] && [ "$PEAK_RERUN" = 1 ]; then
  echo
  echo ">>> winner: $best_label (${best_mean} tok/s heuristic) — re-running WITH autotune (~+13 min startup)"
  # shellcheck disable=SC2086  # intentional word-split of space-free tokens
  run_cfg 1 "peak_${best_label}" $(flags_of "$best_label")
elif [ -z "$best_label" ]; then
  echo "[!] no rung produced a decode tok/s number — skipping peak rerun"
fi

# ---- summary ------------------------------------------------------------------
echo
echo "==================== SUMMARY (decode tok/s @ concurrency 1) ===================="
printf "%-16s %-40s %s\n" "config" "decode tok/s" "spec accept"
for f in "$RESULTS"/bench_*.txt; do
  [ -e "$f" ] || continue
  lbl=$(basename "$f" .txt | sed 's/^bench_//')
  line=$(grep "decode tok/s :" "$f" 2>/dev/null | head -1 | sed 's/decode tok\/s ://')
  acc=$(grep "EAGLE3 accept:" "$f" 2>/dev/null | head -1 | sed 's/EAGLE3 accept://')
  printf "%-16s %-40s %s\n" "$lbl" "${line:-no result}" "${acc:-}"
done
[ -n "$best_label" ] && echo "ladder winner: $best_label — peak run: peak_${best_label}"
