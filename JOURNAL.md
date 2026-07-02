# Tuning Journal — gemma-4-31B-v2-NVFP4 → max single-stream tok/s on 1× B200

Goal: **1000 tok/s** single-stream decode (concurrency 1, greedy, 512-tok
completions). Hardware: RunPod 1× B200 (183 GB), image `vllm/vllm-openai:v0.24.0`.
Model: `jdfelo/gemma-4-31B-v2-NVFP4` (our NVFP4 W4A16-of-A4 quant of the
abliterated v2; 23.3 GB). Bench: `benchmark.py` — decode tok/s excludes TTFT.

## Scoreboard (chronological)

| config | decode tok/s mean (median / max) | delta | notes |
|---|---|---|---|
| baseline | 140.5 (140.5 / 140.6) | 1.00× | TRITON_ATTN, no autotune |
| fp8kv | 126.3 | 0.90× | **regression** — see L2 |
| async (fp8kv+async) | 134.2 | 0.96× | async recovers some of fp8kv loss |
| eagle3_k2 (fp8kv+async) | 227.2 | 1.62× | |
| eagle3_k3 (fp8kv+async) | 239.1 | 1.70× | |
| eagle3_k5 (fp8kv+async) | 251.6 | 1.79× | ladder winner |
| **peak_eagle3_k5** (autotuned) | 268.2 (249.8 / 364.8) | 1.91× | +6.6% from autotune |
| e5_nofp8 (async, auto KV) | 258.2 | 1.84× | dropping fp8kv: +2.6% over e5 |
| e7_nofp8 | 268.0 (243.0 / 377.9) | 1.91× | k7>k5; ties autotuned k5 untuned |
| e5_fi / e7_fi | 258.3 / 256.3 | — | FLASHINFER env **ignored** (L4) |
| peak_e7_nofp8 (autotuned) | 255.7 | 1.82× | **autotune hurt e7** (L6) |
| dflash d8 (async, auto KV) | 309.6 (268.1 / 489.7) | 2.20× | stock RedHat head |
| **peak_d8 (autotuned)** | **336.5 (290.1 / 525.2)** | **2.40×** | best so far |
| d4 | 279.0 | 1.99× | shallower block loses |
| d16 (max-num-seqs 32) | 320.6 (281.1 / 501.0) | 2.28× | un-autotuned; ~ties d8 |

## Learnings

**L1 — vLLM v0.24.0 re-runs FlashInfer fp4_gemm autotune (~13 min) on EVERY
launch.** `VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR` is dead code:
`_FLASHINFER_USE_PERSISTENT_CACHE = False` hardcoded in
`vllm/model_executor/warmup/kernel_warmup.py` (upstream cache-collision TODO).
Fix: `--no-enable-flashinfer-autotune` for A/B rungs (identical heuristics =
fair), pay autotune once on the winner. Ladder went ~16 min/rung → ~5 min/rung.

**L2 — fp8 KV cache is a ~10% REGRESSION at batch=1.** Weight reads dominate
single-stream decode; KV reads are noise at 512-tok contexts. fp8 KV just adds
dequant work in the attention kernel. (It pays at high concurrency / long ctx,
not here.) All later configs use auto KV.

**L3 — EAGLE3 acceptance is monotone k2<k3<k5<k7 on this model**, so deeper
speculation keeps winning until drafter latency eats the gain (k7 ≈ k5-autotuned).

**L4 — FlashInfer ATTENTION is impossible on v0.24.0 for this model:**
head_size=256 → backend selector offers only `['TRITON_ATTN', 'FLEX_ATTENTION']`;
`VLLM_ATTENTION_BACKEND=FLASHINFER` silently ignored. Also FA4 rejects
head_size=256/512 on Blackwell (TMEM limits) → FA2. Attention-backend lever: dead.
(FlashInfer *GEMM* for NVFP4 linears is active and is what autotune tunes.)

**L5 — DFlash loads fine on STABLE v0.24.0** despite the RedHat card saying
"nightly": registry maps `DFlashDraftModel → qwen3_dflash` and the drafter is
target-agnostic (KV injection). No nightly needed. (Nightly wheel index is
`0.23.1rc1.dev730` — *lower* version than stable; pip prefers stable even
with `--pre`. Would need exact `==` pin to install. Untested, deferred.)

**L6 — Autotune is NOT uniformly positive:** +6.6% on eagle3_k5, **−4.6% on
eagle3_k7** (255.7 vs 268.0). Tuner optimizes GEMM tactics for
max_num_batched_tokens shapes, not the small verify-batch shapes spec-decode
actually runs. Always re-measure after autotune; never assume.

**L7 — d16 needs `--max-num-seqs 32`:** default max_num_seqs(1024)×(k+1)
overflows the 8192-token scheduling budget →
`max_num_scheduled_tokens = -7168` ValidationError at startup.

**L8 — Stock DFlash head accepts only 2.11/8 drafted tokens on v2**
(per-position: p0=0.74 p1=0.51 p2=0.34 p3=0.23 on d4). Root cause: heads were
trained on pristine `gemma-4-31B-it`; our target is the **abliterated v2** —
distribution shift. NVIDIA's pristine-model numbers imply ~4.5-6/8. This is
the single biggest remaining lever → head finetune (in progress, see Plan).

**L9 — Prometheus parsing traps:** vLLM v0.24.0 metric names are
`vllm:spec_decode_num_{drafts,draft_tokens,accepted_tokens}_total` +
`..._accepted_tokens_per_pos_total{position=N}`. Substring matching collides
(per_pos contains accepted_tokens) → bogus 1.000; `_created` samples are unix
timestamps, must be skipped. benchmark.py now exact-matches and reports
accept-rate + accepted/draft + per-position.

## Infra learnings

**I1 — RunPod + vllm-openai image:** the image has ENTRYPOINT `["vllm","serve"]`;
only the JSON form of "Container Start Command"
(`{"entrypoint":["bash","-c"],"cmd":[...]}`) overrides it. Plain string = args
appended to `vllm serve` = crash loop.

**I2 — Orphaned EngineCore = stranded 160 GB VRAM.** vLLM v1's engine child is
`python -c from multiprocessing.spawn import spawn_main...` — cmdline does NOT
contain "vllm", so `pkill -f vllm` misses it. Fix: launch under `setsid`, kill
the whole process group (`kill -- -PGID`), then wait for VRAM < 5 GB before the
next rung. Also purge pattern `multiprocessing.spawn import spawn_main`.

**I3 — 12-min health budget was the first-run killer:** engine init with
autotune took 951 s (load 117 s + compile 82 s + autotune ~13 min + graphs);
STARTUP_TRIES=240×3s timed out, killed the rung, orphaned the engine, and the
next rung OOM-blocked on the orphan. Cascade cost a full failed session.

**I4 — Everything is dry-run-tested locally against shims** (fake `vllm` SSE
server + fake `nvidia-smi` + fake EngineCore child): full ladder, crash path,
interrupt path, orphan-reaping all proven before burning GPU time. macOS
gotchas: no `setsid` binary, bash 3.2 chokes on `$label…` (UTF-8 glued to
varname), `(cmd &)` in the harness gets HUP'd — use its async job facility.

**I5 — RunPod network volume** (`/workspace`, NFS4) persists across pods in the
same DC: HF cache + torch.compile cache (`VLLM_CACHE_ROOT=/workspace/.cache/vllm`)
+ all results live there. Weight load from warm volume: 117 s for 23 GB.

## Current best

```
vllm serve jdfelo/gemma-4-31B-v2-NVFP4 \
  --max-model-len 32768 --gpu-memory-utilization 0.90 \
  --async-scheduling --limit-mm-per-prompt '{"image":0}' \
  --speculative-config '{"model":"RedHatAI/gemma-4-31B-it-speculator.dflash","num_speculative_tokens":8,"method":"dflash"}'
# + FlashInfer autotune ON (default) => 336.5 tok/s mean, 525 max
```

## Plan to 1000 (active — head finetune pipeline built + running)

**L10 — SpecForge DFlash draft maps 1:1 onto the RedHat head.** Converter kept
58 tensors (layers.*, fc, hidden_norm, norm = 2.54 B), dropped 4
(embed_tokens/lm_head/d2t/t2d — SpecForge borrows target embed+head at runtime,
full 262144 vocab, no compression). Warm-start smoke: loaded 58, missing 0,
unexpected 0. So finetune = true warm start from RedHat, not from scratch.

**I6 — SpecForge installs alongside vLLM without conflict** via
`venv --system-site-packages` + `pip install --no-deps` (reuses pod
torch 2.11+cu128). Its hard dep sglang==0.5.14 is unwanted; two module-level
sglang imports guarded/stubbed (annotations eval at class-def → stub the NAMES,
not just the import). `yunchang` is a real dep (seq-parallel), installed.
`--draft-init-path` patched into train_dflash.py for warm start.

### Pipeline (all scripts in `train/`, dry-run/smoke-validated)
1. `gen_data.py` — v2-NVFP4 batch-regenerates 119,642 ultrachat+magpie prompts,
   greedy (matches serving). RUNNING on pod → `/workspace/train_data/v2_greedy.jsonl`.
2. `extract_text_model.py` — DONE. bf16 flat-config Gemma4ForCausalLM (5376
   hidden, 60 layers) at `/workspace/gemma4_v2_text` (SpecForge HF target needs
   flat config; AutoModelForCausalLM kept it nested → explicit transplant).
3. `convert_head.py` — DONE. RedHat head → `/workspace/dflash_init` (SpecForge fmt).
4. `setup_specforge.sh` — DONE. venv + patches + warm-start smoke pass.
5. `run_train.sh` — QUEUED (waits for gen DONE). warm-start finetune, block_size 8,
   bs4×accum2, 2 epochs, lr 2e-4, sdpa attn → `/workspace/dflash_ft/`.
6. `export_head.py` — merge finetuned 58 tensors back onto RedHat template
   (keep embed/lm_head/d2t/t2d + speculators config) → vLLM serves unchanged.
7. re-benchmark d8/d16 with finetuned head; autotuned peak.

Expected: accepted/draft 2.11 → ~4-5 ⇒ ~1.8× ⇒ **~600+ tok/s**; then d16 +
autotune. If short of 1000: Domino (DFlash+GRU, SpecForge), deeper drafter,
TRT-LLM compare.

## Scoreboard peak so far: **peak_d8 = 336.5 tok/s mean (525 max), 2.40× baseline.**
