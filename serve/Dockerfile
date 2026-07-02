# Gemma-4-31B-v2 NVFP4 — RunPod Serverless (Load Balancing) worker.
# Target: 1x NVIDIA B200 (Blackwell, sm_100). Max single-stream decode tok/s.
#
# Base: pinned vLLM v0.24.0 (latest stable, CUDA 12.8 / torch 2.11.0 /
# transformers>=5.5.3). Verified to ship: gemma4_mm (Gemma4ForConditionalGeneration),
# NVFP4 compressed-tensors, EAGLE3 "speculators" spec-decode, --async-scheduling
# (with the Eagle-method exemption), and FlashInfer Blackwell attention.
FROM vllm/vllm-openai:v0.24.0

# hf_transfer gives multi-GB/s pulls for the ~23 GB NVFP4 checkpoint + 4.5 GB eagle head.
RUN pip install --no-cache-dir "hf_transfer>=0.1.8"

# HF_HOME points at the RunPod mount used by both the "cached models" feature
# and attached network volumes. Weights resolved there load far faster than a
# cold HF download and are billed as $0 download time on RunPod.
ENV HF_HOME=/runpod-volume/huggingface-cache \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    PORT=8000 \
    PORT_HEALTH=8080 \
    PYTHONUNBUFFERED=1

COPY health_server.py /health_server.py
COPY start.sh /start.sh
RUN chmod +x /start.sh

# PORT = OpenAI API (routed to by the LB); PORT_HEALTH = /ping sidecar.
EXPOSE 8000 8080

# Override the base image's vLLM entrypoint with our launcher (vLLM + sidecar).
ENTRYPOINT ["/start.sh"]
