#!/usr/bin/env python3
"""Fix upstream TRT-LLM NVFP4 bug: dynamic-activation checkpoints (no static
`input_scale`, e.g. compressed-tensors `"dynamic":"local"`) get gibberish output.

Root cause (verified live on rc20, modules/linear.py NVFP4LinearMethod):
`create_weights` inits `module.input_scale = Parameter(torch.empty([1]))` (garbage).
The 3 `process_weights_after_loading_*` finalizers only `copy_weight` when the
checkpoint HAS a static input_scale:
    if input_scale is not None:
        copy_weight(module.input_scale, input_scale)
...with NO `else`. For dynamic checkpoints `_finalize_nvfp4_scales` returns
input_scale=None, so the garbage empty Parameter survives. Then `_input_prepare`
(line ~1366) tests `if module.input_scale is None or force_dynamic_quantization`
-> the garbage Parameter is NOT None and force_dynamic defaults False -> the
STATIC quant path runs with a garbage scale -> gibberish.

Fix: add `else: module.input_scale = None; module.inv_input_scale = None` to each
of the 3 finalizers, so the forward correctly takes the dynamic path.

(The FP8 methods already self-heal via has_static_input_scale; only NVFP4 is broken.)

Idempotent, anchored, fails loudly. Run: python3 patch_linear_nvfp4_inputscale.py [--check]
"""
import argparse
import sys
from pathlib import Path

ELSE_BLOCK_MARK = "# === NVFP4-DYNAMIC-INPUTSCALE FIX ==="

# fused qkv + fused gate_up share this exact 3-line shape (2 occurrences):
FUSED_OLD = (
    "        if input_scale is not None:\n"
    "            copy_weight(module.input_scale, input_scale)\n"
    "        if alpha is not None:\n"
)
FUSED_NEW = (
    "        if input_scale is not None:\n"
    "            copy_weight(module.input_scale, input_scale)\n"
    f"        else:  {ELSE_BLOCK_MARK}\n"
    "            module.input_scale = None\n"
    "            module.inv_input_scale = None\n"
    "        if alpha is not None:\n"
)

# vanilla has the extra E2M1_MAX / inv_input_scale lines (1 occurrence):
VANILLA_OLD = (
    "            module.inv_input_scale.data = module.input_scale / E2M1_MAX\n"
    "        if alpha is not None:\n"
)
VANILLA_NEW = (
    "            module.inv_input_scale.data = module.input_scale / E2M1_MAX\n"
    f"        else:  {ELSE_BLOCK_MARK}\n"
    "            module.input_scale = None\n"
    "            module.inv_input_scale = None\n"
    "        if alpha is not None:\n"
)


def find_target() -> Path:
    import tensorrt_llm
    p = Path(tensorrt_llm.__file__).parent / "_torch" / "modules" / "linear.py"
    if not p.exists():
        sys.exit(f"FATAL: {p} not found")
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    p = find_target()
    src = p.read_text()

    if ELSE_BLOCK_MARK in src:
        print("ALREADY PATCHED:", p)
        return

    if args.check:
        print("NOT patched:", p)
        return

    # vanilla (1 occurrence)
    n_van = src.count(VANILLA_OLD)
    if n_van != 1:
        sys.exit(f"FATAL: vanilla anchor found {n_van}x (expected 1) — drift, inspect linear.py")
    src = src.replace(VANILLA_OLD, VANILLA_NEW, 1)

    # fused qkv + gate_up (2 occurrences)
    n_fused = src.count(FUSED_OLD)
    if n_fused != 2:
        sys.exit(f"FATAL: fused anchor found {n_fused}x (expected 2) — drift, inspect linear.py")
    src = src.replace(FUSED_OLD, FUSED_NEW, 2)

    p.with_suffix(".py.nvfp4bak").write_text(p.read_text())
    p.write_text(src)
    added = src.count(ELSE_BLOCK_MARK)
    print(f"PATCHED: {p}  ({added} finalizers fixed: 1 vanilla + 2 fused)")
    print(f"backup:  {p}.nvfp4bak")


if __name__ == "__main__":
    main()
