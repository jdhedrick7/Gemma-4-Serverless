#!/usr/bin/env python3
"""Extract the text decoder from an NVFP4 (compressed-tensors) multimodal
gemma-4 checkpoint as a FLAT Gemma4ForCausalLM checkpoint, PRESERVING the
NVFP4 quantization.

Unlike train/extract_text_model.py (which rebuilds an nn.Module and thus would
DEQUANTIZE), this is a pure safetensors key-filter: the NVFP4 weights live as
`weight_packed` + `weight_scale` + `weight_global_scale` + `input_global_scale`
aux tensors under each Linear, so copying tensors by prefix keeps them bit-exact.

Verified key layout of jdfelo/gemma-4-31B-v2-NVFP4 (single 18 GB file):
  model.language_model.*  (2062)  -> KEEP, remap to model.*
  lm_head.weight          (1)     -> KEEP as-is
  model.vision_tower.*    (355)   -> DROP
  model.embed_vision      (1)     -> DROP

Why text-only: TRT-LLM's DFlash path serves Gemma4ForCausalLM (flat config);
the HF repo is Gemma4ForConditionalGeneration (nested text_config + vision).
This matches what gemma4_v2_text did for bf16, but keeps NVFP4 (~18 GB, ~3.4x
less HBM read than bf16 -> higher single-stream decode floor).

CPU/disk only (~20 GB RAM). Run on the pod:
    python3 train/extract_text_nvfp4.py \
        --src jdfelo/gemma-4-31B-v2-NVFP4 --dst /workspace/gemma4_v2_nvfp4_text
"""
import argparse
import json
import shutil
from pathlib import Path


LM_PREFIX = "model.language_model."
KEEP_ASIS = {"lm_head.weight"}
DROP_PREFIXES = ("model.vision_tower.", "model.embed_vision")
# text-only aux files to carry over if present
AUX_FILES = (
    "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
    "tokenizer.model", "generation_config.json", "chat_template.jinja",
)


def build_flat_config(src_dir: Path) -> dict:
    """Promote text_config to top level, keep quantization_config, set arch."""
    parent = json.loads((src_dir / "config.json").read_text())
    tc = dict(parent.get("text_config") or parent)
    tc["architectures"] = ["Gemma4ForCausalLM"]
    tc.setdefault("model_type", parent.get("model_type", "gemma4"))
    # carry the compressed-tensors quant config verbatim (its ignore patterns
    # for vision simply match nothing after extraction; lm_head ignore stays)
    if "quantization_config" in parent:
        tc["quantization_config"] = parent["quantization_config"]
    for k in ("dtype", "torch_dtype", "bos_token_id", "eos_token_id", "pad_token_id"):
        if k in parent and k not in tc:
            tc[k] = parent[k]
    return tc


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="jdfelo/gemma-4-31B-v2-NVFP4",
                    help="HF repo id or local dir of the NVFP4 multimodal ckpt")
    ap.add_argument("--dst", default="/workspace/gemma4_v2_nvfp4_text")
    args = ap.parse_args()

    from huggingface_hub import snapshot_download
    from safetensors import safe_open
    from safetensors.torch import save_file

    src = Path(args.src if Path(args.src).is_dir() else snapshot_download(args.src))
    dst = Path(args.dst)
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)

    # locate weight shards (single file or sharded)
    shards = sorted(src.glob("model*.safetensors"))
    assert shards, f"no safetensors in {src}"

    new_sd = {}
    kept = dropped = 0
    for shard in shards:
        with safe_open(shard, framework="pt") as f:
            for k in f.keys():
                if any(k.startswith(p) for p in DROP_PREFIXES):
                    dropped += 1
                    continue
                if k in KEEP_ASIS:
                    new_sd[k] = f.get_tensor(k); kept += 1
                elif k.startswith(LM_PREFIX):
                    new_sd["model." + k[len(LM_PREFIX):]] = f.get_tensor(k); kept += 1
                else:
                    # unknown top-level (e.g. stray) — drop but report
                    dropped += 1
    print(f"kept {kept} tensors, dropped {dropped}")
    assert "model.embed_tokens.weight" in new_sd, "embed_tokens missing after remap"
    assert "lm_head.weight" in new_sd, "lm_head missing"

    # save (single file; NVFP4 text-only is ~15 GB, under safetensors limits but
    # shard at 10 GB for safety/parity with bf16 extraction)
    save_file(new_sd, dst / "model.safetensors", metadata={"format": "pt"})

    cfg = build_flat_config(src)
    (dst / "config.json").write_text(json.dumps(cfg, indent=2))
    assert cfg.get("hidden_size"), "flat config missing hidden_size"
    assert "quantization_config" in cfg, "quant config not preserved"

    for fn in AUX_FILES:
        p = src / fn
        if p.exists():
            shutil.copy2(p, dst / fn)

    print("saved ->", dst)
    print("  architectures:", cfg.get("architectures"))
    print("  hidden_size:", cfg.get("hidden_size"), "layers:", cfg.get("num_hidden_layers"))
    print("  quant format:", (cfg.get("quantization_config") or {}).get("format"))


if __name__ == "__main__":
    main()
