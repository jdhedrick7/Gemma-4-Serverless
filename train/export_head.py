#!/usr/bin/env python3
"""Merge a finetuned SpecForge DFlash draft back into the RedHat speculators
checkpoint so vLLM serves it unchanged.

Inverse of convert_head.py:
  - trained SpecForge ckpt has 58 tensors: layers.*, fc.weight,
    hidden_norm.weight, norm.weight  (the only trainable params).
  - RedHat head has those 58 + embed_tokens, lm_head, d2t, t2d (frozen,
    borrowed-from-target / vocab-compression buffers).
  => take RedHat as template, overwrite the 58, keep the 4 extras and the
     ORIGINAL speculators config.json + aux files verbatim.

Run (pod): python3 train/export_head.py \
    --ft /workspace/dflash_ft/epoch_1 --dst /workspace/dflash_ft_vllm
"""
import argparse
import shutil
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--redhat", default="RedHatAI/gemma-4-31B-it-speculator.dflash")
    ap.add_argument("--ft", default="/workspace/dflash_ft",
                    help="SpecForge ckpt dir (model.safetensors), OR a parent dir "
                         "of epoch_*_step_* — latest is auto-picked")
    ap.add_argument("--dst", default="/workspace/dflash_ft_vllm")
    args = ap.parse_args()

    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file, save_file

    rh = Path(snapshot_download(args.redhat))
    ft = Path(args.ft)
    if not (ft / "model.safetensors").exists():
        cands = sorted(
            ft.glob("epoch_*_step_*"),
            key=lambda d: int(str(d).rsplit("_", 1)[-1]) if str(d).rsplit("_", 1)[-1].isdigit() else -1,
        )
        cands = [d for d in cands if (d / "model.safetensors").exists()]
        if not cands:
            raise SystemExit(f"no model.safetensors under {ft} or its epoch_*_step_* subdirs")
        ft = cands[-1]
        print(f"[auto] using latest checkpoint: {ft}")
    dst = Path(args.dst)
    if dst.exists():
        shutil.rmtree(dst)
    # copy the entire RedHat repo (config.json speculators-format, tokenizer, etc.)
    shutil.copytree(rh, dst)

    base = load_file(rh / "model.safetensors")
    trained = load_file(ft / "model.safetensors")

    TRAINABLE = lambda k: (
        k.startswith("layers.")
        or k in ("fc.weight", "hidden_norm.weight", "norm.weight")
    )
    merged = dict(base)
    n_over, n_skip, missing = 0, 0, []
    for k in base:
        if TRAINABLE(k):
            if k in trained:
                assert trained[k].shape == base[k].shape, f"shape mismatch {k}: {trained[k].shape} vs {base[k].shape}"
                merged[k] = trained[k]
                n_over += 1
            else:
                missing.append(k)
        else:
            n_skip += 1
    extra = [k for k in trained if not (k in base)]
    print(f"overwrote {n_over} trained tensors; kept {n_skip} frozen/buffer tensors")
    print(f"missing-from-ft (kept base): {len(missing)}  extra-in-ft (ignored): {extra[:5]}")
    assert n_over >= 55, "too few trained tensors merged — wrong --ft dir?"

    save_file(merged, dst / "model.safetensors", metadata={"format": "pt"})
    print("wrote ->", dst)


if __name__ == "__main__":
    main()
