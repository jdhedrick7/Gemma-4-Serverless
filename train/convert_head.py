#!/usr/bin/env python3
"""Convert RedHatAI/gemma-4-31B-it-speculator.dflash (vLLM speculators format)
into a SpecForge DFlashDraftModel init directory.

Mapping (verified against both sources):
  - keep: layers.*  fc.weight  hidden_norm.weight  norm.weight   (name-identical)
  - drop: embed_tokens / lm_head / d2t / t2d — SpecForge's draft has no such
    modules; it borrows the TARGET model's embeddings + lm_head at runtime
    (TargetEmbeddingsAndHead), i.e. full 262144 vocab, no compression.
  - config: flatten transformer_layer_config -> top level, model_type qwen3
    (DFlashDraftModel.config_class = Qwen3Config), plus block_size and
    dflash_config{mask_token_id, target_layer_ids} from the head config.

Run (pod): python3 train/convert_head.py --dst /workspace/dflash_init
"""
import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="RedHatAI/gemma-4-31B-it-speculator.dflash")
    ap.add_argument("--dst", default="/workspace/dflash_init")
    args = ap.parse_args()

    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file, save_file

    src = Path(snapshot_download(args.src))
    cfg = json.loads((src / "config.json").read_text())
    tl = cfg["transformer_layer_config"]

    out_cfg = {
        "architectures": ["DFlashDraftModel"],
        "model_type": "qwen3",
        "attention_bias": tl.get("attention_bias", False),
        "attention_dropout": tl.get("attention_dropout", 0.0),
        "head_dim": tl["head_dim"],
        "hidden_act": tl.get("hidden_act", "silu"),
        "hidden_size": tl["hidden_size"],
        "initializer_range": tl.get("initializer_range", 0.02),
        "intermediate_size": tl["intermediate_size"],
        "max_position_embeddings": tl.get("max_position_embeddings", 262144),
        "num_attention_heads": tl["num_attention_heads"],
        "num_hidden_layers": tl["num_hidden_layers"],
        "num_key_value_heads": tl["num_key_value_heads"],
        "rms_norm_eps": tl.get("rms_norm_eps", 1e-6),
        "rope_theta": (tl.get("rope_parameters") or {}).get("rope_theta", 10000.0),
        "rope_scaling": None,
        "sliding_window": None,
        "use_sliding_window": False,
        "layer_types": ["full_attention"] * tl["num_hidden_layers"],
        "vocab_size": tl["vocab_size"],
        "tie_word_embeddings": False,
        "use_cache": True,
        "dtype": "bfloat16",
        # DFlash-specific
        "block_size": cfg["block_size"],
        "num_target_layers": 60,  # gemma-4-31B text decoder depth
        "dflash_config": {
            "mask_token_id": cfg["mask_token_id"],
            "target_layer_ids": cfg["aux_hidden_state_layer_ids"],
        },
    }

    sd = load_file(src / "model.safetensors")
    keep, drop = {}, []
    for k, v in sd.items():
        if k.startswith("layers.") or k in ("fc.weight", "hidden_norm.weight", "norm.weight"):
            keep[k] = v
        else:
            drop.append(k)
    n = sum(v.numel() for v in keep.values())
    print(f"kept {len(keep)} tensors ({n/1e9:.2f} B params); dropped: {drop}")

    dst = Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)
    (dst / "config.json").write_text(json.dumps(out_cfg, indent=1))
    save_file(keep, dst / "model.safetensors")
    print("wrote ->", dst)


if __name__ == "__main__":
    main()
