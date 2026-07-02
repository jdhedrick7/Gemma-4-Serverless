#!/usr/bin/env python3
"""RunPod load-balancer health sidecar (FastAPI) for the vLLM Gemma-4 worker.

Exposes ``GET /ping`` on ``PORT_HEALTH`` implementing RunPod's LB contract:
    200 -> healthy      (worker added to the routing pool)
    204 -> initializing (cold start in progress; LB keeps waiting)
    503 -> unhealthy    (evicted from the pool)

Why a sidecar: vLLM's own ``/health`` only answers 200 after weights load and
refuses connections before that. A raw connection refusal reads as "unhealthy"
to the LB and can kill the worker mid-load. This sidecar polls vLLM in the
background and maps its state to the contract: 204 until vLLM first reports
ready, 200 after, and 503 if it later stops answering (crashed engine gets
evicted).

FastAPI + uvicorn ship in the ``vllm/vllm-openai`` image. The explicit
``@app.get("/ping")`` route is also what RunPod's pre-deploy analyzer scans for.
"""
import os
import threading
import time
import urllib.request

from fastapi import FastAPI, Response

VLLM_PORT = int(os.environ.get("PORT", "8000"))
HEALTH_PORT = int(os.environ.get("PORT_HEALTH", "8080"))
POLL_INTERVAL_S = float(os.environ.get("HEALTH_POLL_INTERVAL", "3"))

# _ready: set once vLLM first reports 200 (cold start done).
# _last_ok: tracks current health after ready, so a later failure -> 503.
_ready = threading.Event()
_last_ok = threading.Event()

app = FastAPI()


def _probe_vllm() -> bool:
    try:
        with urllib.request.urlopen(
            f"http://127.0.0.1:{VLLM_PORT}/health", timeout=5
        ) as resp:
            return resp.status == 200
    except Exception:
        return False


def _poll_loop() -> None:
    while True:
        if _probe_vllm():
            _ready.set()
            _last_ok.set()
        else:
            _last_ok.clear()
        time.sleep(POLL_INTERVAL_S)


@app.get("/ping")
def ping() -> Response:
    if not _ready.is_set():
        return Response(status_code=204)   # cold start in progress
    if _last_ok.is_set():
        return Response(status_code=200)   # ready + engine currently healthy
    return Response(status_code=503)        # was ready, engine now failing


def _start_poller() -> None:
    threading.Thread(target=_poll_loop, daemon=True).start()
    print(
        f"[health] /ping sidecar (FastAPI) on :{HEALTH_PORT} -> vLLM /health :{VLLM_PORT}",
        flush=True,
    )


if __name__ == "__main__":
    import uvicorn

    _start_poller()
    uvicorn.run(app, host="0.0.0.0", port=HEALTH_PORT, log_level="warning")
