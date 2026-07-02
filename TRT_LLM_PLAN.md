# TRT-LLM fallback plan (execute only if finetuned vLLM DFlash < 1000 tok/s)

Authorized by user: "fix the bugs in TRT-LLM yourself and get it running." Fully
scoped GPU-free (see JOURNAL L15). The B200-blocking bugs were myths (aarch64/L4
specific or fixed on main). One real gap: Gemma4 aux-hidden-state capture.

## Why TRT-LLM can beat vLLM here
vLLM's measured spec cycle is **9.2 ms** (peak_d8, 2.11 acc/pass) vs ~3 ms ideal —
per-step Python/scheduler overhead is the wall. TRT-LLM runs draft+verify+accept
**in-engine (C++/CUDA)**, removing that overhead. NVIDIA's DFlash 15× numbers are
TRT-LLM. DFlash is a first-class TRT-LLM spec method (`DFlashDecodingConfig`).

## Bug status (from deep-read of the actual threads)
- **#12764** (gemma4 NVFP4 won't load): CLOSED, "supported on latest main"
  2026-05-09. Failures were DGX Spark/GB10 (aarch64) + tokenizer
  `extra_special_tokens`-as-list + transformers skew. NOT B200.
  Mitigation if it recurs: normalize `extra_special_tokens` list→dict in
  tokenizer_config.json; use official `nvidia/Gemma-4-31B-IT-NVFP4` or our
  bf16 `/workspace/gemma4_v2_text`.
- **#14942** (FlashInfer "unsupported architecture"): L4/SM89 only — trtllm-gen
  FMHA has no SM89 cubins. B200=SM100 is the native trtllm-gen target. N/A.

## The one code gap: Gemma4 aux-hidden-state capture
Support matrix shows `Gemma4 EAGLE-3/DFlash: No` because
`tensorrt_llm/_torch/models/modeling_gemma4.py` never calls
`spec_metadata.maybe_capture_hidden_states(...)` (Llama/Qwen3/GptOss do).

### Patch (mirror modeling_llama.py lines ~572-592 / 863-885)
File: `tensorrt_llm/_torch/models/modeling_gemma4.py`

1. `Gemma4DecoderLayer.forward` (currently line ~641): add explicit param
   `spec_metadata: Optional[SpecMetadata] = None` to the signature. Because the
   model-forward passes `**kwargs` to the layer, Python binds spec_metadata here
   automatically AND it stops leaking into `self.self_attn(**kwargs)` (which would
   otherwise raise on the unexpected kwarg — this is likely why it isn't wired yet).

2. Just before `hidden_states = hidden_states * self.layer_scalar` (line ~714),
   insert (guard for older spec_metadata without the method):
   ```python
   if spec_metadata is not None and getattr(spec_metadata, "is_layer_capture", None) \
           and spec_metadata.is_layer_capture(self.layer_idx):
       spec_metadata.maybe_capture_hidden_states(self.layer_idx, hidden_states, residual)
   ```
   (`residual` at that point = pre-`layer_scalar` accumulated stream; validate
   against the DFlash drafter's `fc` expectation — RedHat head concatenates the
   5 aux layers' hidden states, dim 5*5376=26880 = fc.weight in-features. Confirmed.)

3. Import `SpecMetadata` if not already imported (check top of file).

4. If executor gates capture on a model attribute (some paths check the ForCausalLM
   exposes `spec` support), add whatever Llama4's `Gemma4ForCausalLM` is missing —
   verify against `DecoderModelForCausalLM` base + how Llama advertises. (TBD live.)

### Serve command (trtllm-serve, config.yaml)
```yaml
# /workspace/trtllm_dflash.yaml
speculative_config:
  decoding_type: DFlash
  max_draft_len: 8
  speculative_model: /workspace/dflash_ft_vllm   # our finetuned head (or RedHat)
  target_layer_ids: [1, 17, 29, 47, 58]
```
```bash
trtllm-serve serve /workspace/gemma4_v2_text \   # bf16 target (sidesteps NVFP4 loader)
  --backend pytorch --host 0.0.0.0 --port 8000 \
  --max_batch_size 1 --max_seq_len 4096 --config /workspace/trtllm_dflash.yaml
```
Then benchmark.py against :8000 (same harness).

## Container
`nvcr.io/nvidia/tensorrt-llm/release:<latest>` (or build main). Editable install
of the patched `modeling_gemma4.py` (site-packages or `-e` checkout).

## Order of operations
1. Measure finetuned DFlash on vLLM first (running now).
2. If < 1000: spin TRT-LLM on the SAME B200 (kill vLLM first — one GPU), apply
   the modeling_gemma4.py patch, serve + bench. NVFP4 target if loader is clean
   on main; else bf16 target (still fast — decode is spec-bound, not weight-bound).
3. Compare; keep the winner. Record in JOURNAL + MODEL_CARD.
