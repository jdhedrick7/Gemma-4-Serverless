#!/usr/bin/env python3
"""Extract the text-only decoder from jdfelo/gemma-4-31B-v2 (multimodal) as a
standalone Gemma4ForCausalLM checkpoint in bf16.

Why: SpecForge's DFlash HF target backend loads via AutoModelForCausalLM and
reads flat config fields (hidden_size, num_hidden_layers, ...). The v2 repo is
Gemma4ForConditionalGeneration with a nested text_config + vision tower; a
clean text-only export sidesteps both problems and drops ~1.5 GB of vision
weights the trainer never uses.

CPU-only (device_map on CPU, ~65 GB RAM; pod has 1.5 TB). Run while the GPU
does something useful.

Run (pod): python3 train/extract_text_model.py \
             --src jdfelo/gemma-4-31B-v2 --dst /workspace/gemma4_v2_text
"""
import argparse

import torch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="jdfelo/gemma-4-31B-v2")
    ap.add_argument("--dst", default="/workspace/gemma4_v2_text")
    args = ap.parse_args()

    from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor, AutoTokenizer

    cfg = AutoConfig.from_pretrained(args.src)
    text_cfg = getattr(cfg, "text_config", cfg)
    print("text_config:", text_cfg.__class__.__name__)

    # Load the CAUSAL-LM view of the checkpoint: transformers >=5.5 maps
    # Gemma4ForConditionalGeneration checkpoints into Gemma4ForCausalLM when
    # asked via AutoModelForCausalLM (vision weights are skipped). If this
    # ever regresses, fall back to loading the full model and pulling
    # .language_model manually.
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.src, torch_dtype=torch.bfloat16, device_map="cpu"
        )
    except Exception as e:  # fallback: full multimodal load -> extract
        print(f"[fallback] AutoModelForCausalLM failed ({type(e).__name__}: {e})")
        from transformers import AutoModelForImageTextToText

        full = AutoModelForImageTextToText.from_pretrained(
            args.src, torch_dtype=torch.bfloat16, device_map="cpu"
        )
        model = full.language_model if hasattr(full, "language_model") else full.model.language_model

    model = model.eval()
    n = sum(p.numel() for p in model.parameters())
    print(f"params: {n/1e9:.2f} B  class: {model.__class__.__name__}")
    assert n > 25e9, "text decoder should be ~31B params — wrong submodule?"

    model.save_pretrained(args.dst, safe_serialization=True, max_shard_size="10GB")
    AutoTokenizer.from_pretrained(args.src).save_pretrained(args.dst)
    try:
        AutoProcessor.from_pretrained(args.src).save_pretrained(args.dst)
    except Exception:
        pass
    print("saved ->", args.dst)


if __name__ == "__main__":
    main()
