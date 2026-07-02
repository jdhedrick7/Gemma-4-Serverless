#!/usr/bin/env python3
"""Extract the text decoder from jdfelo/gemma-4-31B-v2 (multimodal) as a clean,
FLAT-config Gemma4ForCausalLM checkpoint in bf16.

Why explicit construction (not AutoModelForCausalLM): on transformers 5.x the
auto-mapping hands back Gemma4ForConditionalGeneration with the nested
text_config intact — SpecForge's HF target backend reads flat fields
(hidden_size, num_hidden_layers, ...) and gets None. So: build
Gemma4ForCausalLM(text_config) and copy weights by stripped prefix.

CPU-only (~65 GB RAM). Run: python3 train/extract_text_model.py \
    --src jdfelo/gemma-4-31B-v2 --dst /workspace/gemma4_v2_text
"""
import argparse
import shutil
from pathlib import Path

import torch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="jdfelo/gemma-4-31B-v2")
    ap.add_argument("--dst", default="/workspace/gemma4_v2_text")
    args = ap.parse_args()

    from transformers import AutoConfig, AutoTokenizer
    from transformers.models.gemma4 import Gemma4ForCausalLM

    cfg = AutoConfig.from_pretrained(args.src)
    text_cfg = getattr(cfg, "text_config", cfg)
    print("text_config class:", text_cfg.__class__.__name__)
    print("hidden_size:", text_cfg.hidden_size, "layers:", text_cfg.num_hidden_layers)

    # Load the full multimodal wrapper on CPU, then transplant.
    from transformers import AutoModelForImageTextToText

    full = AutoModelForImageTextToText.from_pretrained(
        args.src, dtype=torch.bfloat16, device_map="cpu"
    ).eval()
    full_sd = full.state_dict()

    text = Gemma4ForCausalLM(text_cfg)
    text = text.to(torch.bfloat16).eval()
    want = text.state_dict()

    # Map wrapper keys -> text-model keys by suffix stripping. Wrapper layouts
    # seen in the wild: "model.language_model.<X>" or "language_model.model.<X>";
    # lm_head at "lm_head.*" (often tied to embeddings).
    prefixes = ("model.language_model.", "language_model.model.", "language_model.")
    new_sd = {}
    for k, v in full_sd.items():
        for p in prefixes:
            if k.startswith(p):
                cand = "model." + k[len(p):]
                if cand in want:
                    new_sd[cand] = v
                break
        else:
            if k.startswith("lm_head.") and k in want:
                new_sd[k] = v

    missing = [k for k in want if k not in new_sd]
    # tied lm_head: fill from embeddings if absent
    if "lm_head.weight" in missing and "model.embed_tokens.weight" in new_sd:
        new_sd["lm_head.weight"] = new_sd["model.embed_tokens.weight"]
        missing.remove("lm_head.weight")
    print(f"mapped {len(new_sd)}/{len(want)} tensors; missing={len(missing)}")
    if missing[:5]:
        print("  first missing:", missing[:5])
    assert not missing, "unmapped tensors — wrapper layout changed, inspect keys"

    text.load_state_dict(new_sd, strict=True)
    n = sum(p.numel() for p in text.parameters())
    print(f"text params: {n/1e9:.2f} B  class: {text.__class__.__name__}")

    dst = Path(args.dst)
    if dst.exists():
        shutil.rmtree(dst)  # avoid 2x 60 GB on a tight volume
    text.save_pretrained(dst, safe_serialization=True, max_shard_size="10GB")
    tok = AutoTokenizer.from_pretrained(args.src)
    tok.save_pretrained(dst)

    import json
    saved = json.load(open(dst / "config.json"))
    print("saved architectures:", saved.get("architectures"))
    print("saved hidden_size:", saved.get("hidden_size"))
    assert saved.get("hidden_size") == text_cfg.hidden_size, "config not flat!"
    print("saved ->", dst)


if __name__ == "__main__":
    main()
