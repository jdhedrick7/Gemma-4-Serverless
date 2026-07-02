#!/usr/bin/env python3
"""Generate DFlash training data: v2 model regenerates responses to real prompts.

Speculative-decoding drafters must model the TARGET's distribution. Our target
is the abliterated v2, so training text = v2's own greedy outputs (greedy
matches the serving/benchmark setting, temperature=0).

Prompt sources (same families the stock heads used, so gains are attributable
to alignment, not data):
  - HuggingFaceH4/ultrachat_200k (train_sft, first user turn)
  - Magpie-Align/Magpie-Llama-3.1-Pro-300K-Filtered (instruction field)

Output: JSONL {"messages":[{role:user,...},{role:assistant,...}]} — the format
both SpecForge and ModelOpt ingest.

Run (pod): python3 train/gen_data.py --model jdfelo/gemma-4-31B-v2-NVFP4 \
             --out /workspace/train_data/v2_greedy.jsonl --n 120000
"""
import argparse
import json
import os
import random
from pathlib import Path


def load_prompts(n_total: int, seed: int = 0) -> list[str]:
    from datasets import load_dataset

    half = n_total // 2
    prompts: list[str] = []

    uc = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft", streaming=True)
    for row in uc:
        msgs = row.get("messages") or []
        if msgs and msgs[0].get("role") == "user":
            p = msgs[0]["content"].strip()
            if 8 <= len(p) <= 6000:
                prompts.append(p)
        if len(prompts) >= half:
            break

    mg = load_dataset(
        "Magpie-Align/Magpie-Llama-3.1-Pro-300K-Filtered", split="train", streaming=True
    )
    for row in mg:
        p = (row.get("instruction") or "").strip()
        if 8 <= len(p) <= 6000:
            prompts.append(p)
        if len(prompts) >= n_total:
            break

    random.Random(seed).shuffle(prompts)
    # dedupe, keep order
    seen, out = set(), []
    for p in prompts:
        k = p[:256]
        if k in seen:
            continue
        seen.add(k)
        out.append(p)
    return out[:n_total]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="jdfelo/gemma-4-31B-v2-NVFP4")
    ap.add_argument("--out", default="/workspace/train_data/v2_greedy.jsonl")
    ap.add_argument("--n", type=int, default=120000)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--batch-log", type=int, default=2000)
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    done = 0
    if out.exists():  # resume: skip prompts already generated
        with out.open() as f:
            done = sum(1 for _ in f)
        print(f"[resume] {done} rows already present", flush=True)

    prompts = load_prompts(args.n)
    todo = prompts[done:]
    print(f"prompts total={len(prompts)} todo={len(todo)}", flush=True)
    if not todo:
        print("nothing to do")
        return

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tok = AutoTokenizer.from_pretrained(args.model)
    llm = LLM(
        model=args.model,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=0.95,
        kv_cache_dtype="fp8",          # 2x KV pool -> ~2x concurrency (greedy: precision irrelevant)
        max_num_seqs=512,              # allow the bigger pool to actually fill
        max_num_batched_tokens=32768,  # more prefill+decode scheduling headroom
        enforce_eager=False,
        limit_mm_per_prompt={"image": 0},
    )
    sp = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)

    CHUNK = 24576   # ~5 chunks over 120k -> few drain/render gaps vs 30
    with out.open("a") as f:
        for i in range(0, len(todo), CHUNK):
            batch = todo[i : i + CHUNK]
            texts = [
                tok.apply_chat_template(
                    [{"role": "user", "content": p}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for p in batch
            ]
            outs = llm.generate(texts, sp)
            for p, o in zip(batch, outs):
                resp = o.outputs[0].text.strip()
                if not resp:
                    continue
                f.write(
                    json.dumps(
                        {
                            "conversations": [
                                {"role": "user", "content": p},
                                {"role": "assistant", "content": resp},
                            ]
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            f.flush()
            print(f"[gen] {min(i + CHUNK, len(todo))}/{len(todo)} written", flush=True)

    print("DONE", flush=True)


if __name__ == "__main__":
    main()
