# Gemma-4-31B-v2 NVFP4 — RunPod Serverless (Load Balancing) worker.
# Target: 1x NVIDIA B200 (Blackwell, sm_100). Max single-stream decode tok/s.
#
# Base: vLLM's official Gemma 4 image (CUDA 12.9, transformers 5.5.3, native
# Gemma4ForConditionalGeneration + NVFP4 compressed-tensors + EAGLE3 spec-decode
# + FlashInfer Blackwell attention). Use :gemma4-cu130 for CUDA 13.0 hosts.
FROM vllm/vllm-openai:gemma4

# hf_transfer gives multi-GB/s pulls for the ~18 GB NVFP4 checkpoint + eagle head.
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
