---
license: mit
base_model: jdfelo/gemma-4-31B-v2
base_model_relation: quantized
pipeline_tag: text-generation
library_name: transformers
tags:
- nvfp4
- fp4
- vllm
- llm-compressor
- compressed-tensors
- gemma4
- abliterated
- blackwell
---

# gemma-4-31B-v2-NVFP4

**NVFP4 (4-bit) quantization of [`jdfelo/gemma-4-31B-v2`](https://huggingface.co/jdfelo/gemma-4-31B-v2)** — an *abliterated* Gemma 4 31B — built for **maximum single-stream decode throughput on NVIDIA Blackwell** (B200 / RTX PRO 6000), served with vLLM.

## What this is

- Weights quantized to **NVFP4** (FP4, group size 16) with [LLM Compressor](https://github.com/vllm-project/llm-compressor).
- **~62.5 GB (bf16) → ~23 GB.** Fits a single Blackwell GPU with plenty of headroom for a long-context KV cache. (The NVFP4 language model is ~16 GB; the vision tower + tied embeddings stay bf16, adding ~7 GB.)
- **Vision tower, embeddings, and `lm_head` kept at bf16** — only the language-model `Linear` layers are quantized, so multimodal understanding is untouched.
- **Abliterated:** the base finetune has standard safety refusals removed — see *Responsible use*.
- **Parameter count: 32.7 B** (same as the base model). Hugging Face's widget shows *~20B* because it counts stored tensor elements and cannot see that NVFP4 packs **two 4-bit weights per byte** — the 14.6 B packed bytes are really 29.3 B params, plus 3.4 B bf16 params. This undercount is normal for every NVFP4/packed-4-bit checkpoint; no weights are missing (all 60 layers + vision tower + embeddings are present).

## Quantization details

| | |
|---|---|
| Scheme | **NVFP4**, `targets="Linear"`, group size 16 |
| Ignored (kept bf16) | `re:.*vision.*`, `re:.*audio.*`, `lm_head`, `re:.*embed.*` |
| Tooling | llm-compressor 0.12.0 · compressed-tensors 0.17.1 · transformers 5.10.1 · torch 2.11.0+cu128 |
| Calibration | 512 samples @ 2048 tokens, `HuggingFaceH4/ultrachat_200k` (text-only) |

The ignore set and scheme mirror the reference [`RedHatAI/gemma-4-31B-it-NVFP4`](https://huggingface.co/RedHatAI/gemma-4-31B-it-NVFP4) recipe for the base architecture.

## Serving with vLLM (Blackwell)

```bash
vllm serve jdfelo/gemma-4-31B-v2-NVFP4 \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.90 \
  --kv-cache-dtype fp8 \
  --limit-mm-per-prompt '{"image":0,"audio":0}' \
  --async-scheduling
```

**For maximum single-stream tok/s**, add EAGLE3 speculative decoding:

```bash
  --speculative-config '{"model":"RedHatAI/gemma-4-31B-it-speculator.eagle3","num_speculative_tokens":3,"method":"eagle3"}'
```

> The EAGLE3 draft is trained on the base `google/gemma-4-31B-it`; acceptance rate on this abliterated finetune may differ — smoke-test and drop to `num_speculative_tokens: 2` if it's low.

### Requirements

- A **Blackwell** GPU with FP4 tensor cores: 1× **B200** (180 GB) or 1× **RTX PRO 6000** (96 GB). Weights are ~23 GB; the rest is KV cache.
- On pre-Blackwell GPUs, NVFP4 loads via a Marlin fallback (smaller footprint, **not** faster than FP8).

### Client (OpenAI-compatible)

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")
r = client.chat.completions.create(
    model="jdfelo/gemma-4-31B-v2-NVFP4",
    messages=[{"role": "user", "content": "Explain speculative decoding in one paragraph."}],
    max_tokens=512, temperature=0.7,
)
print(r.choices[0].message.content)
```

## Responsible use

This is an **abliterated / uncensored** model: standard safety refusals have been removed from the base finetune. You are solely responsible for how you use it and for compliance with applicable law and the terms of the upstream models.

## Lineage & license

Quantized from `jdfelo/gemma-4-31B-v2`, a finetune/abliteration of `google/gemma-4-31B-it`. Released under the **MIT license** by the repository owner. Note that upstream Gemma components remain subject to Google's [Gemma Terms of Use](https://ai.google.dev/gemma/terms); downstream users should review that lineage.
