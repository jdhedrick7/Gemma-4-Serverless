# Gemma-4-31B-v2 → NVFP4 → max single-stream tok/s on 1× B200

Serve [`jdfelo/gemma-4-31B-v2`](https://huggingface.co/jdfelo/gemma-4-31B-v2)
(abliterated Gemma 4 31B, multimodal, bf16) at **maximum single-stream decode
tok/s** on a single **NVIDIA B200** via RunPod Serverless.

The model ships only in bf16 (62.5 GB), so there are **two phases**:

1. **Quantize** — one-time job (~1 GPU-hour) on **any recent NVIDIA GPU**
   (H100/H200 cheapest; Blackwell not required to *pack* FP4): bf16 → **NVFP4**
   (W4A4, group 16, ~18 GB), pushed to a private HF repo. (`quantize/`)
2. **Serve** — RunPod Serverless load-balancer worker running vLLM, tuned for
   single-stream decode. (`serve/`)

## Why this is fast (the tok/s stack)

| Lever | Choice | Why it wins on B200 |
|---|---|---|
| Weights | **NVFP4** W4A4, group 16 | Native Blackwell FP4 tensor cores; 62.5 → ~18 GB. Weight bandwidth dominates single-stream decode → ~3–4× vs bf16. |
| Spec decode | **EAGLE3** (`RedHatAI/gemma-4-31B-it-speculator.eagle3`) | Purpose-built draft for the exact base model; verifies ~3 tokens/step. The #1 single-stream multiplier (2–3×). |
| KV cache | **FP8** | Halves KV read bandwidth per decode step. |
| Attention | **FlashInfer** | vLLM's Blackwell default; best decode kernels. |
| Parallelism | **TP1 (single GPU)** | 18 GB fits 180 GB with room to spare; TP would only add cross-GPU comm latency. |
| Modality | **text-only** (`image=0,audio=0`) | Skips vision/audio profiling + KV budget. Vision tower stays bf16 (unquantized) in the checkpoint for optional multimodal serving. |
| Batching | **max-num-seqs 1** + CUDA graphs | Captures the tight batch-1 decode path; async scheduling overlaps CPU scheduling with GPU decode. |

Everything is a known-good recipe: the NVFP4 config mirrors
`RedHatAI/gemma-4-31B-it-NVFP4` and the EAGLE3 draft is Red Hat's official
speculator for this base — we only re-quantize because v2 is an *abliterated
finetune* of that base.

## Layout

```
quantize/
  quantize_nvfp4.py        NVFP4 oneshot (llm-compressor), saves + optional HF push
  requirements-quant.txt   transformers 5.8.1 + llm-compressor git main
  run_pod.sh               one-command bootstrap for the B200 quant pod
serve/
  Dockerfile               FROM vllm/vllm-openai:gemma4
  start.sh                 vLLM launcher (PROFILE=latency|throughput) + sidecar
  health_server.py         RunPod /ping LB sidecar (204 loading → 200 ready)
  .dockerignore
benchmark.py               single-stream decode tok/s + EAGLE3 accept rate
smoke_test.py              health → models → chat → streaming
DEPLOY.md                  screen-by-screen RunPod runbook
.env                       secrets (gitignored)
```

## Quick start

**Phase 1 — quantize** (1× B200, network volume at `/workspace`, base image
`vllm/vllm-openai:gemma4`):
```bash
export HF_TOKEN=hf_...                          # read private source + push
export DST_MODEL=jdfelo/gemma-4-31B-v2-NVFP4    # push target (optional)
bash quantize/run_pod.sh
```

**Phase 2 — serve** (RunPod Serverless): import this repo, Dockerfile
`serve/Dockerfile`, GPU **B200 ×1**, env `PORT=8000 PORT_HEALTH=8080
RUNPOD_INIT_TIMEOUT=800`. Full walkthrough in [DEPLOY.md](DEPLOY.md).

**Verify:**
```bash
python smoke_test.py --base-url https://<ENDPOINT_ID>.api.runpod.ai --api-key $RUNPOD_API_KEY
python benchmark.py  --base-url https://<ENDPOINT_ID>.api.runpod.ai --api-key $RUNPOD_API_KEY --trials 10 --max-tokens 512
```

## Tuning knobs (env; override the profile default)

`PROFILE` (`latency`\|`throughput`), `MODEL_NAME`, `MAX_MODEL_LEN`,
`MAX_NUM_SEQS`, `GPU_MEMORY_UTILIZATION`, `KV_CACHE_DTYPE` (`fp8`\|empty),
`SPECULATIVE_CONFIG` (JSON; empty disables EAGLE3), `MULTIMODAL` (`true` serves
images), `VLLM_ATTENTION_BACKEND`, `EXTRA_VLLM_ARGS`.

If decode tok/s underwhelms, check the EAGLE3 accept rate in `benchmark.py`
output (or vLLM `/metrics`). The draft was trained on the *base* model; on the
abliterated finetune accept may be lower — if so, try
`num_speculative_tokens: 2`, or set `SPECULATIVE_CONFIG=""` to isolate.
