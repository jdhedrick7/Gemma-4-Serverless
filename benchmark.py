#!/usr/bin/env python3
"""
Single-stream decode benchmark for Gemma-4-31B-v2 NVFP4 on vLLM.

Goal metric: DECODE tok/s at concurrency 1 (the number this build maximizes).
Streams completions from the OpenAI-compatible endpoint, measuring:
  * TTFT           — time to first token (s)
  * decode tok/s   — output_tokens / (wall - ttft)   [the headline number]
  * end-to-end tok/s

Also queries vLLM /metrics for EAGLE3 acceptance (draft_acceptance_rate /
num_accepted_tokens): decode tok/s scales ~linearly with accepted tokens, so a
low accept rate is the first thing to check if tok/s underwhelms.

Usage:
  # Local pod (vLLM on :8000):
  python benchmark.py --base-url http://localhost:8000 --trials 10 --max-tokens 512

  # RunPod LB endpoint:
  python benchmark.py --base-url https://<ENDPOINT_ID>.api.runpod.ai \
      --api-key $RUNPOD_API_KEY --trials 10 --max-tokens 512
"""
import argparse
import asyncio
import os
import statistics
import time

import httpx


def load_env(path: str = ".env") -> None:
    """Minimal .env loader (no python-dotenv dependency)."""
    if not os.path.exists(path):
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


PROMPTS = [
    "Write a detailed technical explanation of how speculative decoding accelerates LLM inference.",
    "Explain the tradeoffs between NVFP4, FP8, and BF16 for large language model serving.",
    "Describe step by step how a transformer decodes one token during autoregressive generation.",
    "Write a Python function that implements binary search, then explain its complexity.",
]


async def one_request(client, base_url, headers, model, prompt, max_tokens):
    """Stream one chat completion; return timing dict."""
    url = f"{base_url}/v1/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    t0 = time.perf_counter()
    ttft = None
    n_chunks = 0
    usage = {}
    async with client.stream("POST", url, json=payload, headers=headers) as r:
        r.raise_for_status()
        async for line in r.aiter_lines():
            if not line or not line.startswith("data: "):
                continue
            data = line[6:]
            if data == "[DONE]":
                break
            import json as _json
            obj = _json.loads(data)
            if obj.get("usage"):
                usage = obj["usage"]
            choices = obj.get("choices") or []
            if choices and choices[0].get("delta", {}).get("content"):
                if ttft is None:
                    ttft = time.perf_counter() - t0
                n_chunks += 1
    wall = time.perf_counter() - t0
    out_toks = usage.get("completion_tokens") or n_chunks
    ttft = ttft or wall
    decode_s = max(wall - ttft, 1e-6)
    return {
        "ttft": ttft,
        "wall": wall,
        "out_tokens": out_toks,
        "decode_tok_s": out_toks / decode_s,
        "e2e_tok_s": out_toks / wall,
    }


async def fetch_accept_rate(client, base_url, headers):
    """Scrape EAGLE3 acceptance from vLLM /metrics.

    Returns (mean_accepted_per_draft, per_pos list) or None.
    v0.24.0 names (exact-match; substring matching would collide with
    vllm:spec_decode_num_accepted_tokens_per_pos):
      vllm:spec_decode_num_drafts_total
      vllm:spec_decode_num_draft_tokens_total
      vllm:spec_decode_num_accepted_tokens_total
      vllm:spec_decode_num_accepted_tokens_per_pos_total{position=...}
    """
    try:
        r = await client.get(f"{base_url}/metrics", headers=headers, timeout=10)
        if r.status_code != 200:
            return None
        drafts = draft_toks = acc_toks = None
        per_pos = {}
        for line in r.text.splitlines():
            if line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            name = parts[0].split("{")[0]
            val = float(parts[-1])
            if name in ("vllm:spec_decode_num_drafts_total", "vllm:spec_decode_num_drafts"):
                drafts = (drafts or 0.0) + val
            elif name in ("vllm:spec_decode_num_draft_tokens_total", "vllm:spec_decode_num_draft_tokens"):
                draft_toks = (draft_toks or 0.0) + val
            elif name in ("vllm:spec_decode_num_accepted_tokens_total", "vllm:spec_decode_num_accepted_tokens"):
                acc_toks = (acc_toks or 0.0) + val
            elif name.startswith("vllm:spec_decode_num_accepted_tokens_per_pos"):
                pos = parts[0].split('position="')[-1].split('"')[0] if 'position="' in parts[0] else str(len(per_pos))
                per_pos[pos] = per_pos.get(pos, 0.0) + val
        if acc_toks is None or not draft_toks:
            return None
        rate = acc_toks / draft_toks
        pos_rates = None
        if drafts and per_pos:
            pos_rates = [per_pos[k] / drafts for k in sorted(per_pos, key=lambda x: int(x) if x.isdigit() else 0)]
        return rate, (acc_toks / drafts if drafts else None), pos_rates
    except Exception:
        return None
    return None


async def wait_ready(client, base_url, headers, timeout_s):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        for probe in ("/ping", "/health"):
            try:
                r = await client.get(f"{base_url}{probe}", headers=headers, timeout=10)
                if r.status_code == 200:
                    return True
            except Exception:
                pass
        await asyncio.sleep(3)
    return False


async def main():
    load_env()
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default=os.getenv("BASE_URL", "http://localhost:8000"))
    ap.add_argument("--api-key", default=os.getenv("RUNPOD_API_KEY", "EMPTY"))
    ap.add_argument("--model", default=os.getenv("SERVED_MODEL_NAME", "jdfelo/gemma-4-31B-v2-NVFP4"))
    ap.add_argument("--trials", type=int, default=10)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--warmup", type=int, default=2, help="Warmup runs (CUDA graph capture) excluded from stats.")
    ap.add_argument("--wait", type=int, default=0, help="Seconds to wait for readiness before starting.")
    args = ap.parse_args()

    base = args.base_url.rstrip("/")
    headers = {"Authorization": f"Bearer {args.api_key}"} if args.api_key and args.api_key != "EMPTY" else {}

    async with httpx.AsyncClient(timeout=httpx.Timeout(600.0)) as client:
        if args.wait:
            print(f"waiting up to {args.wait}s for readiness…")
            if not await wait_ready(client, base, headers, args.wait):
                print("worker not ready; proceeding anyway")

        print(f"model={args.model}  base={base}  trials={args.trials}  max_tokens={args.max_tokens}")
        print("warming up (CUDA graph capture + EAGLE3 draft load)…")
        for i in range(args.warmup):
            await one_request(client, base, headers, args.model, PROMPTS[i % len(PROMPTS)], min(args.max_tokens, 128))

        results = []
        for i in range(args.trials):
            res = await one_request(client, base, headers, args.model, PROMPTS[i % len(PROMPTS)], args.max_tokens)
            results.append(res)
            print(f"  trial {i+1:2d}: decode {res['decode_tok_s']:6.1f} tok/s  "
                  f"ttft {res['ttft']*1000:6.0f}ms  {res['out_tokens']} tok in {res['wall']:.1f}s")

        dec = [r["decode_tok_s"] for r in results]
        ttfts = [r["ttft"] for r in results]
        e2e = [r["e2e_tok_s"] for r in results]
        print("\n===== SINGLE-STREAM DECODE (concurrency 1) =====")
        print(f"decode tok/s : mean {statistics.mean(dec):6.1f}  median {statistics.median(dec):6.1f}  "
              f"min {min(dec):6.1f}  max {max(dec):6.1f}")
        print(f"e2e tok/s    : mean {statistics.mean(e2e):6.1f}")
        print(f"ttft         : mean {statistics.mean(ttfts)*1000:6.0f}ms  median {statistics.median(ttfts)*1000:6.0f}ms")

        acc = await fetch_accept_rate(client, base, headers)
        if acc is not None:
            rate, per_draft, pos_rates = acc
            extra = f"  accepted/draft {per_draft:.2f}" if per_draft is not None else ""
            print(f"EAGLE3 accept: {rate:.3f} of draft tokens{extra}")
            if pos_rates:
                print("  accept by pos: " + " ".join(f"p{i}={r:.2f}" for i, r in enumerate(pos_rates)))
        else:
            print("EAGLE3 accept: n/a (metrics not exposed on this endpoint)")


if __name__ == "__main__":
    asyncio.run(main())
