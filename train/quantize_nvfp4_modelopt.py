#!/usr/bin/env python3
"""Quantize the text-only bf16 gemma-4-31B-v2 to NVFP4 in MODELOPT HF format
(the format TRT-LLM's pytorch backend loads natively — unlike llm-compressor's
compressed-tensors, which TRT-LLM can't load: keys weight_packed vs weight).

NVFP4_DEFAULT_CFG uses dynamic input quant, so calibration is light (weight amax
is data-free); a few forward passes satisfy the API. lm_head + embed kept BF16
(quality; matches official nvidia/Gemma-4-31B-IT-NVFP4 structure).

GPU (~58GB load + activations, fits B200). Run with the serve stopped:
    python3 train/quantize_nvfp4_modelopt.py \
        --src /workspace/gemma4_v2_text --dst /workspace/gemma4_v2_nvfp4_mo
"""
import argparse
import torch


CALIB = [
    "The capital of Australia is",
    "def fibonacci(n):",
    "Explain quantum entanglement in simple terms.",
    "Write a haiku about the ocean.",
    "What is 17 times 23? Show your reasoning.",
    "The three laws of thermodynamics are",
    "Translate 'good morning' into French, Spanish, and German.",
    "A recipe for chocolate chip cookies starts with",
    "The key differences between TCP and UDP are",
    "In machine learning, gradient descent works by",
    "Summarize the plot of Romeo and Juliet.",
    "The most important events of the 20th century include",
    "How does photosynthesis convert sunlight into energy?",
    "class BinarySearchTree:",
    "The Pythagorean theorem states that",
    "Describe the water cycle step by step.",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="/workspace/gemma4_v2_text")
    ap.add_argument("--dst", default="/workspace/gemma4_v2_nvfp4_mo")
    ap.add_argument("--calib-seq", type=int, default=256)
    args = ap.parse_args()

    import modelopt.torch.quantization as mtq
    from modelopt.torch.export import export_hf_checkpoint
    from transformers import AutoTokenizer
    from transformers.models.gemma4 import Gemma4ForCausalLM

    tok = AutoTokenizer.from_pretrained(args.src)
    print("loading bf16 target (~58GB) to GPU...")
    model = Gemma4ForCausalLM.from_pretrained(
        args.src, dtype=torch.bfloat16, device_map="cuda"
    ).eval()

    cfg = mtq.NVFP4_DEFAULT_CFG
    # keep lm_head full precision (quality; softcap after it)
    # MLP-only NVFP4 (matches nvidia/Gemma-4-31B-IT-NVFP4): ALL self_attn stays
    # bf16. The 10 global layers use unified-KV (no v_proj); the fused NVFP4
    # QKV loader requires q/k/v -> KeyError 'v' if attn is quantized. MLP is
    # ~4x the attention weight, so this keeps most of the bandwidth win AND
    # avoids the attention-quant acceptance regression the NextStep journal hit.
    cfg = {**cfg, "quant_cfg": {**cfg["quant_cfg"],
                                "*lm_head*": {"enable": False},
                                "*embed_tokens*": {"enable": False},
                                "*self_attn*": {"enable": False}}}

    def forward_loop(m):
        with torch.no_grad():
            for i, p in enumerate(CALIB):
                ids = tok(p, return_tensors="pt", truncation=True,
                          max_length=args.calib_seq).input_ids.to("cuda")
                m(ids)
                if (i + 1) % 4 == 0:
                    print(f"  calib {i+1}/{len(CALIB)}")

    print("quantizing NVFP4 (dynamic act, block16)...")
    model = mtq.quantize(model, cfg, forward_loop)
    mtq.print_quant_summary(model)

    print(f"exporting modelopt HF checkpoint -> {args.dst}")
    export_hf_checkpoint(model, export_dir=args.dst)
    tok.save_pretrained(args.dst)
    print("DONE:", args.dst)


if __name__ == "__main__":
    main()
