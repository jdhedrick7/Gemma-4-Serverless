#!/usr/bin/env python3
"""
Smoke test for the Gemma-4-31B-v2 NVFP4 vLLM worker.

Checks, in order:
  1. /ping (health sidecar) or /health (vLLM) returns ready.
  2. /v1/models lists the served model.
  3. A non-streaming chat completion returns coherent text.
  4. A streaming completion yields tokens (confirms the streaming path).

Usage:
  python smoke_test.py --base-url http://localhost:8000
  python smoke_test.py --base-url https://<ENDPOINT_ID>.api.runpod.ai --api-key $RUNPOD_API_KEY
"""
import argparse
import os
import sys
import time

import httpx


def load_env(path: str = ".env") -> None:
    if not os.path.exists(path):
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


def main() -> int:
    load_env()
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default=os.getenv("BASE_URL", "http://localhost:8000"))
    ap.add_argument("--api-key", default=os.getenv("RUNPOD_API_KEY", "EMPTY"))
    ap.add_argument("--model", default=os.getenv("SERVED_MODEL_NAME", "jdfelo/gemma-4-31B-v2-NVFP4"))
    ap.add_argument("--wait", type=int, default=900, help="Max seconds to wait for readiness (cold start).")
    args = ap.parse_args()

    base = args.base_url.rstrip("/")
    headers = {"Authorization": f"Bearer {args.api_key}"} if args.api_key and args.api_key != "EMPTY" else {}
    client = httpx.Client(timeout=httpx.Timeout(120.0), headers=headers)

    # 1. readiness
    print(f"[1/4] waiting for readiness (up to {args.wait}s)…")
    deadline = time.time() + args.wait
    ready = False
    while time.time() < deadline:
        for probe in ("/ping", "/health"):
            try:
                r = client.get(f"{base}{probe}")
                if r.status_code == 200:
                    ready = True
                    print(f"      ready via {probe}")
                    break
            except Exception:
                pass
        if ready:
            break
        time.sleep(5)
    if not ready:
        print("      FAIL: never became ready")
        return 1

    # 2. models
    print("[2/4] GET /v1/models")
    r = client.get(f"{base}/v1/models")
    r.raise_for_status()
    ids = [m["id"] for m in r.json().get("data", [])]
    print(f"      models: {ids}")
    if not ids:
        print("      FAIL: no models listed")
        return 1

    model = args.model if args.model in ids else ids[0]

    # 3. non-streaming completion
    print(f"[3/4] chat completion (model={model})")
    r = client.post(
        f"{base}/v1/chat/completions",
        json={
            "model": model,
            "messages": [{"role": "user", "content": "In one sentence, what is speculative decoding?"}],
            "max_tokens": 128,
            "temperature": 0.0,
        },
    )
    r.raise_for_status()
    text = r.json()["choices"][0]["message"]["content"]
    print(f"      -> {text.strip()[:300]}")
    if not text.strip():
        print("      FAIL: empty completion")
        return 1

    # 4. streaming completion
    print("[4/4] streaming completion")
    got = 0
    with client.stream(
        "POST",
        f"{base}/v1/chat/completions",
        json={
            "model": model,
            "messages": [{"role": "user", "content": "Count from 1 to 5."}],
            "max_tokens": 64,
            "temperature": 0.0,
            "stream": True,
        },
    ) as s:
        s.raise_for_status()
        for line in s.iter_lines():
            if line and line.startswith("data: ") and line[6:] != "[DONE]":
                got += 1
    print(f"      streamed {got} chunks")
    if got == 0:
        print("      FAIL: no streamed chunks")
        return 1

    print("\nSMOKE TEST PASSED ✓")
    return 0


if __name__ == "__main__":
    sys.exit(main())
