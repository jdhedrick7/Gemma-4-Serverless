# Deploy — Gemma-4-31B-v2 NVFP4 on RunPod (1× B200)

Two phases: **(A) quantize once** on a B200 pod, **(B) serve** on RunPod
Serverless. The serverless image bakes no weights — the host supplies the NVFP4
checkpoint via the HF cache / network volume.

---

## 0. Prerequisites
- RunPod account + billing; GitHub connected to RunPod.
- This repo pushed to `main`.
- `.env` filled (already gitignored): `HF_TOKEN`, RunPod S3 keys.
- Source model `jdfelo/gemma-4-31B-v2` is **private** → `HF_TOKEN` needs read.

---

## Phase A — Quantize (one-time, ~1 GPU-hour)

> **You do NOT need a Blackwell GPU to quantize.** Producing the NVFP4
> checkpoint (calibration + packing) runs on any recent NVIDIA GPU; FP4 tensor
> cores are only needed to *serve* it fast (Phase B). Quantizing on H100/H200 is
> cheaper. Quantize on B200 only if you want to validate FP4 on the target HW in
> one pod.

> ✅ **Pre-staged (already done):** the EUR-IS-1 volume `ayzcrd0zx1` holds the
> bf16 source model, the EAGLE3 head, and the calibration dataset in an HF cache
> at `/workspace/hf-cache` (67 GB, integrity-verified). A quant pod that
> **attaches this volume** downloads **zero weights** — `run_pod.sh` already
> points `HF_HOME` there. So "attach the volume in EUR-IS-1" is the recommended
> path below.

### A1. Pod
Pods → **Deploy** →
| Field | Value | Why |
|---|---|---|
| GPU | **RTX PRO 6000 Blackwell (96 GB)** ×1 — recommended | Holds the 62.5 GB bf16 model outright (no offload), and being Blackwell it runs the sanity-gen on the **native FP4 path** (H100/H200 can't). Cheaper than H200. H200 141 GB / H100 80 GB / B200 also work; H100 needs sequential offload. |
| CPU RAM | **≥ 120 GB** | bf16 model (62.5 GB) stays in host RAM while layers stream to the GPU. |
| Region | **EUR-IS-1** 🚨 | The pre-staged volume is region-locked here — the quant pod must be in EUR-IS-1 to attach it and skip all downloads. |
| Network volume | attach **`ayzcrd0zx1`** (200 GB) → `/workspace` | Pre-staged: bf16 model + EAGLE3 head + dataset already cached at `/workspace/hf-cache`. |
| Template | `vllm/vllm-openai:gemma4` | torch + CUDA 12.9 + transformers 5.5.3. Any recent CUDA image works. |
| Container disk | 30 GB | Scratch (pip, compile); weights are on the volume. |

> Disk math: 67 GB already staged (model 62.5 + eagle 4.5) + NVFP4 output ~18 GB ≈ 85 GB of 200 GB. ✓
> On a **non-Blackwell** quant pod (H100/H200), the script's end-of-run sanity
> generation uses vLLM's Marlin FP4 fallback — confirms the checkpoint is
> coherent but its speed is meaningless. On a **Blackwell** quant pod (PRO 6000 /
> B200) sanity-gen runs native FP4 (validates the kernels), but single-stream
> tok/s still only reflects the serving GPU — PRO 6000 is ~1.6 TB/s vs B200 ~8
> TB/s, so real decode speed only shows on the B200 in Phase B.

### A2. Run
SSH / web terminal into the pod, then:
```bash
git clone https://github.com/jdhedrick7/Gemma-4-Serverless.git
cd Gemma-4-Serverless
export HF_TOKEN=hf_...                          # read private source + push
export DST_MODEL=jdfelo/gemma-4-31B-v2-NVFP4    # push target (omit to skip push)
bash quantize/run_pod.sh
```
`run_pod.sh` pins `HF_HOME=/workspace/hf-cache`, installs transformers 5.8.1 +
llm-compressor git main, runs NVFP4 oneshot → `/workspace/gemma-4-31B-v2-NVFP4`,
then pushes to `DST_MODEL` (private).

Because the volume is pre-staged, the model + dataset resolve from
`/workspace/hf-cache` with **no download** — only the ~18 GB NVFP4 output is
written. (Optionally `export HF_HUB_OFFLINE=1` to force fully offline resolution.)

Watch for: `oneshot done`, a coherent `SANITY GENERATION`, `push complete`.

### A3. Teardown
Once pushed to HF (or left on the volume), **terminate the pod**. Phase B never
needs it again.

---

## Phase B — Serve (RunPod Serverless, load balancer)

### B1. Get weights on the host (pick one)
| Option | How | Notes |
|---|---|---|
| **Cached models** ✅ | Endpoint **Model** field = `jdfelo/gemma-4-31B-v2-NVFP4` | RunPod prefetches to `/runpod-volume/huggingface-cache`, $0 download billing. Needs `HF_TOKEN` (private repo). |
| Network volume | Attach `ayzcrd0zx1`; set `MODEL_NAME=/runpod-volume/gemma-4-31B-v2-NVFP4` | Serve straight off the volume you quantized onto (no HF round-trip). Pins to EUR-IS-1. |

### B2. New Endpoint → Import Git Repository
Serverless → **New Endpoint** → **Import Git Repository** →
`jdhedrick7/Gemma-4-Serverless`.
| Field | Value |
|---|---|
| Branch | `main` |
| Dockerfile path | `serve/Dockerfile` |
| Build context | `serve` |

### B3. Configure endpoint
| Field | Value | Why |
|---|---|---|
| Endpoint type | **Load balancer** | Direct HTTP to vLLM, native streaming, max tok/s. |
| GPU | **B200 (180 GB)** | Native FP4 tensor cores — required for NVFP4 speed. |
| **GPUs / worker** | **1** | 18 GB NVFP4 + KV fits one B200; TP adds only latency. |
| Region | **EUR-IS-1** if serving off the network volume | Cached-models path is region-flexible. |
| Container disk | 20 GB | Scratch (CUDA-graph cache); weights live on `/runpod-volume`. |

### B4. Model, ports, env
| Field | Value |
|---|---|
| **Model** | `jdfelo/gemma-4-31B-v2-NVFP4` (triggers cached-models prefetch) |
| **Expose HTTP ports** | `8000, 8080` (API + health — both required) |
| Container start command | leave blank (image ENTRYPOINT runs vLLM + sidecar) |

**Environment variables**

🚨 Always required (LB reads these from *endpoint env*, not the Dockerfile —
miss them and it probes port 80, finds nothing, 502s after ~8 min):
```
PORT               = 8000
PORT_HEALTH        = 8080
HF_TOKEN           = hf_...          # private NVFP4 repo (cached-models path)
RUNPOD_INIT_TIMEOUT= 800             # NVFP4 load + EAGLE3 draft fetch < cutoff
```

Regime (on top of the above):
- **Single-stream / latency (default):** nothing else — image defaults to
  `PROFILE=latency` → EAGLE3 spec, fp8 KV, 32K ctx, seqs 1, text-only.
- **Concurrent / throughput:** add `PROFILE = throughput`.

Serving off the volume instead of HF: add
`MODEL_NAME = /runpod-volume/gemma-4-31B-v2-NVFP4` (drop `HF_TOKEN`).

### B5. Scaling
| Setting | Single-user (scale-to-zero) | Production |
|---|---|---|
| Active workers | **0** (pay only while generating) | ≥ 1 (no cold starts) |
| Max workers | 1–2 | ~20% over peak concurrency |
| `RUNPOD_INIT_TIMEOUT` | `800` | `800` |
| Idle timeout | 60–300 s (keeps worker warm between turns) | 60–300 s |
| CUDA versions | floor **12.9** (image is cu129; use `:gemma4-cu130` + 13.0 to bump) | 12.9 + 13.0 |

> ⚠️ **LB ~5.5-min per-request cap:** a single streamed request is bounded by
> ~330 s × tok/s. At ~200 tok/s that's ~66k tokens — fine for chat, but cap
> `max_tokens` for very long generations, or use a queue endpoint for unbounded.

---

## Verify
```bash
export EP=https://<ENDPOINT_ID>.api.runpod.ai
export RUNPOD_API_KEY=<key>

python smoke_test.py --base-url $EP --api-key $RUNPOD_API_KEY
python benchmark.py  --base-url $EP --api-key $RUNPOD_API_KEY --trials 10 --max-tokens 512
```
Client (OpenAI SDK) — base URL is `/v1`:
```python
from openai import OpenAI
client = OpenAI(api_key="RUNPOD_API_KEY", base_url="https://<ENDPOINT_ID>.api.runpod.ai/v1")
r = client.chat.completions.create(
    model="jdfelo/gemma-4-31B-v2-NVFP4",
    messages=[{"role": "user", "content": "..."}],
    max_tokens=512, temperature=0.7)
print(r.choices[0].message.content)
```

---

## Troubleshooting
| Symptom | Cause / fix |
|---|---|
| OOM on load | Wrong GPU. NVFP4 needs **B200** (FP4 cores); confirm GPUs/worker = 1 on 180 GB. |
| Worker cycles, `502` after ~8 min | Ports. Set env `PORT=8000` + `PORT_HEALTH=8080` **and** expose both. |
| `401` fetching model | Private repo. Set `HF_TOKEN` in endpoint env, or serve off the volume. |
| Marked unhealthy during load | Raise `RUNPOD_INIT_TIMEOUT=800`; sidecar holds `204` while loading. |
| EAGLE3 draft fetch fails | Rare (HF cache writable). Fallback: `SPECULATIVE_CONFIG=""` to serve without spec. |
| Low decode tok/s | Check EAGLE3 accept in `benchmark.py` / `/metrics`. If low (abliterated drift), try `num_speculative_tokens: 2`. |
| `no workers available` | Cold start in flight; retry with backoff (3×, 5–10 s). |
| Startup arg error on `--speculative-config` | Pass as one JSON arg (start.sh already does); check `vllm/vllm-openai:gemma4` is current. |

## Cost
- B200 ≈ **$0.00240/s/GPU** → ~**$8.6/hr** while a worker runs.
- Active workers = 0 → $0 idle; you pay only during generation + idle tail.
