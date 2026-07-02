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

## More learnings (data-gen phase)

**I7 — the offline `LLM()` engine proctitle is `VLLM::EngineCore`** — no "vllm"
or "multiprocessing.spawn" substring, so `pkill -f vllm` AND the serve-era
`pkill -f "multiprocessing.spawn"` BOTH miss it. It stranded 170 GB across a
gen restart and OOM'd the relaunch. Reap with `pkill -9 -f "VLLM::EngineCore"`
(now in restart_gen.sh). Group-kill by PGID also works if you catch the parent.

**L11 — fp8 KV lifted data-gen concurrency 40x→82x but tok/s only ~4.5k→~4.9k.**
Batch offline generation is NOT the clean decode-bound case: it's dominated by
prefill of the prompt queue + per-`generate()`-chunk tail drain (batch shrinks
137→1 as seqs finish). The concurrency ceiling wasn't the binding constraint,
so doubling it barely helped. Lesson: fp8 KV is a *serving/long-context* win,
not a batch-gen win. (Serving single-stream: L2 says it's a regression there
too — fp8 KV only pays at high concurrency + long context.)

**Decision — cap training data at 50K (not 120K).** Warm-start alignment
finetune from the RedHat head doesn't need 120K; 50K starts training ~2.5 h
sooner. `cap_gen.sh` watches row count, group-kills gen at 50K, writes the
DONE marker the train waiter blocks on.

**Re-benchmark is pre-wired:** `dflash.sh` takes `DFLASH=<path>`, so post-export
it's `DFLASH=/workspace/dflash_ft_vllm SKIP_CONTROL=1 bash dflash.sh`.

## Ceiling math (why 1000 is plausible)
Decode reads ~19 GB active weights ⇒ ~2.4 ms/pass @ 8 TB/s ⇒ ~420 tok/s
unspeculated. DFlash: 1 draft pass (~0.6 ms, whole 8-block) + 1 verify
(~2.4 ms) = ~3 ms/cycle producing (accepted+1) tokens. At today's 3.11
acc/pass ⇒ ~336 measured. Finetuned head at ~5-6 acc ⇒ ~1.6-2× ⇒ ~550-670;
with k16 + autotune + Domino stacking, ~1000 is reachable but hinges on
accepted/draft climbing from 2.11/8 toward ~5-6/8.

## L12 — Gemma-4 changed its chat format; SpecForge's `gemma` template is WRONG for it

**Highest-value catch of the session.** SpecForge ships a `gemma` chat template
using `<start_of_turn>…<end_of_turn>` (Gemma-2/3). **Gemma-4 changed to**
`<bos><|turn>user\n…<turn|>\n<|turn>model\n…<turn|>\n` (verified live against the
v2 tokenizer). SpecForge's DFlash loss mask locates assistant spans by string:
`end_of_turn_token + assistant_header`. With the wrong template that regex
matches **nothing** → every training sample has an all-zero loss mask → either
filtered by `min_loss_tokens` or trained on noise. Would have burned the whole
~5 h finetune producing a useless head, silently (no crash).
Fix: `train/register_gemma4.py` registers a `gemma4` template
(`assistant_header="<|turn>model\n"`, `user_header="<|turn>user\n"`,
`end_of_turn_token="<turn|>\n"`, `system_prompt=""`); `run_train.sh` uses
`--chat-template gemma4`. (Inline heredoc registration mangled the `\n` escapes —
use a real .py file, never a shell heredoc, for code with escapes.)

**Meta-lesson:** dry-run/smoke tests validated *construction + warm-start* but
NOT the *data path*. The template mismatch lived in the one stage the smoke test
didn't cover. De-risking the data path before the long run (rendering one sample
through the real tokenizer) is what caught it.

## L13 — Domino is NOT servable on our stack (fallback crossed off)

vLLM v0.24.0 `SpeculativeMethod` = ngram/medusa/mlp_speculator/draft_model/
suffix/custom_class + EagleModelTypes + NgramGPUTypes. **No "domino"** anywhere
(0 mentions in speculative.py, registry only maps `DFlashDraftModel→qwen3_dflash`).
Nightly is `0.23.1rc1.dev730` — a *downgrade*. So even though SpecForge can
*train* Domino (DFlash+GRU logit correction, higher accept), vLLM can't *serve*
it here. Fallback ladder is now: finetuned DFlash → k-sweep+autotune →
(if still short) TRT-LLM, NOT Domino.

## Sharper ceiling analysis (measured overhead is the real wall)

Naive math said DFlash cycle ≈ 3 ms → but peak_d8 = 336 tok/s at 2.11 acc/pass
means **real cycle ≈ (2.11+1)/336 = 9.2 ms**, ~3× the 3 ms ideal. That gap is
per-step overhead: drafter fwd + sampling + detokenize + scheduler + CUDA-graph
seams, NOT weight bandwidth. Implication: **acceptance alone caps out ~600-760**
(7 acc/pass ÷ 9.2 ms ≈ 760), so 1000 needs BOTH higher acceptance AND lower
per-cycle overhead. peak_d8's spread (mean 336 / median 290 / **max 525**) shows
high-acceptance runs already near ~525 — the finetune's job is to make that the
*floor*, not the outlier. Honest projection: finetune ⇒ ~500-700. See L14 —
the lower-overhead engine (TRT-LLM) that could break past that is ruled out too.

## L14 — TRT-LLM ruled out for gemma-4 + NVFP4 + DFlash (both off-vLLM levers dead)

Checked GPU-free before burning hours. Three independent blockers:
1. **Open bug NVIDIA/TensorRT-LLM#12764 (Apr 2026):** "Gemma4 HF NVFP4 export
   cannot be loaded by TRT-LLM builtin runtime (tokenizer/runtime skew)" — our
   exact checkpoint type.
2. **NVFP4 GEMM is Llama/Mixtral-only** in TRT-LLM release notes; Gemma not on
   the NVFP4-GEMM list.
3. **DFlash is not a TRT-LLM spec method** (EAGLE-1/2 in-engine, EAGLE3 via a
   2-model PyTorch path only). And Gemma-4 itself is buggy on current TRT-LLM
   (#14942: FlashInfer arch, kv_layout, AutoDeploy use_cache).

**Consequence:** with Domino (L13) and TRT-LLM both out, **vLLM + finetuned
DFlash head + k-sweep + autotune is the entire remaining game.** The realistic
ceiling for THIS model+stack is ~500-700 tok/s (3.6-5× baseline). 1000 tok/s
single-stream may be physically out of reach for a 31B dense model on one B200
via vLLM's spec-decode loop — I'll prove the finetuned number, report it
honestly, and not pretend a blocked path exists. If a TRT-LLM EAGLE3 route on
the *bf16* model (sidestepping the NVFP4 loader bug) is worth trying, that's a
separate experiment with its own engine build — flagged, not assumed.

## L15 — CORRECTION to L14: TRT-LLM IS viable on B200 (deep-read of the threads + docs)

L14 was too pessimistic — it read search snippets, not the source. Re-reading the
actual issue threads + main-branch docs (user directive: fix TRT-LLM myself if
vLLM < 1000):

1. **#12764 (gemma4 NVFP4 load) is CLOSED** — "Gemma 4 is now supported on the
   latest main" (2026-05-09). Failures were **DGX Spark/GB10 (aarch64)** +
   tokenizer `extra_special_tokens`-as-list + transformers skew — NOT B200.
   Official `nvidia/Gemma-4-31B-IT-NVFP4` exists.
2. **#14942 is an L4 (SM89)** — "FlashInfer unsupported architecture" = trtllm-gen
   FMHA has no SM89 cubins. **B200 = SM100, the exact trtllm-gen target.** Not our bug.
3. **DFlash is a FIRST-CLASS TRT-LLM spec method** (`DFlashDecodingConfig`,
   `decoding_type: DFlash`, params `speculative_model` + `target_layer_ids` =
   our head's `aux_hidden_state_layer_ids [1,17,29,47,58]`). Our finetuned DFlash
   head can drop straight in. EAGLE3 also available (also target-agnostic drafter).

**The only real gap:** support matrix shows `Gemma4 EAGLE-3/DFlash: No` = the
**aux-hidden-state capture isn't wired into the `Gemma4ForConditionalGeneration`
class** in TRT-LLM (Llama4/Qwen3/GptOss have it). That's the ownable fix the user
authorized — mirror those classes' `aux_hidden_states` exposure for Gemma4.
TRT-LLM's in-engine spec loop is exactly what beats vLLM's 9.2 ms Python-side wall.

**Endgame:** (1) measure finetuned DFlash on vLLM. (2) if < 1000, stand up
TRT-LLM latest-main on the B200, serve gemma-4 + our DFlash head via
`DFlashDecodingConfig`; if Gemma4 aux-capture is missing, patch the model class.
That path is now the active fallback, not "flagged/not assumed".

## L16 — vLLM CUDA-graph lever is already maxed (overhead is inherent, proven)

Checked whether spec-decode was silently falling back to PIECEWISE cudagraphs
(a fixable per-step overhead source). It is NOT: `AttentionCGSupport` enum is
`ALWAYS=3 > UNIFORM_BATCH=2`; the FULL-decode-under-spec-decode gate only trips
when support `< UNIFORM_BATCH(2)`. Our backend TRITON_ATTN declares
`_cudagraph_support = ALWAYS(3)`, so peak_d8's spec-decode path **already ran
under FULL cudagraphs**. No `--compilation-config`/cudagraph knob left to pull.

**Consequence:** the measured 9.2 ms DFlash cycle (vs ~3 ms ideal) is inherent
to vLLM's Python-orchestrated draft→verify→accept loop, not a misconfig. Every
vLLM-internal overhead lever is now exhausted with proof. => the finetuned head
lifts *acceptance* (→ ~500-700), but breaking past that to 1000 REQUIRES the
lower-overhead in-engine loop = TRT-LLM (L15, staged in TRT_LLM_PLAN.md +
patch_trtllm_gemma4.py + trtllm_serve.sh). TRT-LLM is now necessary, not optional.

## L17 — DFlash head finetune is LIVE (3 live-run bugs fixed, none caught by smoke tests)

Training running on the B200: loss 4.3→1.8, draft acc 0.32→0.58 in the first
~180 steps, GPU 99%, VRAM 141/178 GB. gemma4 target-forward + gemma4 loss-mask
+ warm-started RedHat head all validated by actually running.

Three bugs, each only surfaced live (smoke tests covered construction/warm-start,
not the runtime data path):
1. **`torchrun` runs system Python** — SpecForge installed `--no-deps` into the
   venv is invisible → `ModuleNotFoundError`. Fix: `/workspace/sfvenv/bin/python
   -m torch.distributed.run` (run_train.sh).
2. **Data schema**: SpecForge reads a `conversations` column; gen_data.py wrote
   `messages`. Fix: rename key (gen_data.py + one-shot sed on existing 37K rows).
3. **cross_entropy OOM over full 262144 vocab at bs8** (tried +16 GB, 15.5 free).
   Fix: bs2 × accum4 (effective batch 8 preserved) + PYTORCH_CUDA_ALLOC_CONF=
   expandable_segments:True. Steady 141 GB.

Config: 36,864 rows (capped chunk-1; enough for warm-start), block_size 8, 1 epoch,
lr 2e-4, maxlen 2048. ~4.3 h/epoch; save-interval 2000 optimizer-steps →
first exportable ckpt ~2.6 h, final at ~4604 steps. Then export_head.py →
DFLASH=/workspace/dflash_ft_vllm bash dflash.sh (d8/d12/d16 + autotuned peak).

Also: pod git pulls abort when an on-pod `sed` leaves run_train.sh modified —
always `git checkout -- <file>` before pull. And `pkill -f <pat>` self-matches
the SSH command's own cmdline (→ 255); kill by explicit PID instead.

## L18 — export pipeline validated on a REAL checkpoint (not just unit test)

At training step 2000, CPU-exported `epoch_0_step_2000` while training continued
on GPU (export is CPU-only, zero contention). Result: `export_head.py` overwrote
58 trained tensors, kept 4 frozen (embed/lm_head/d2t/t2d), missing 0, extra 0 →
**62-tensor vLLM head**, config.json in original RedHat speculators format
(vLLM loads unchanged). Confirms: SpecForge saves with RedHat-matching keys (no
`draft_model.` prefix, single unsharded file, no domino modules). The whole
downstream chain (export → dflash.sh DFLASH=path) is now proven on real
artifacts. Final checkpoint export will be a known-good one-liner.

## L19 - parallel TRT-LLM validation pod (user-authorized 2nd B200)

Pod `j5orkuylapq83x` (gemma4-trtllm-validate), NVIDIA B200, US-CA-2, $5.89/hr,
attached to volume `5vd1uvstkm` (same as training pod, read-only use).
Image `nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc20` (commit c25c23f):
transformers==5.5.4 (gemma4 OK), CUDA_ARCH_LIST has 10.0 (sm_100/B200), ships
trtllm-serve, anonymously pullable. Purpose: validate the untested TRT-LLM
modeling_gemma4.py patch + serve base RedHat EAGLE3 head IN PARALLEL with
training. MUST TERMINATE when done (cost). Training pod: vkkadsdhy2w7mg.
