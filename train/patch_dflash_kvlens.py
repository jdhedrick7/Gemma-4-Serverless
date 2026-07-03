#!/usr/bin/env python3
"""Fix TRT-LLM DFlash worker for the FlashInfer backend: guard the kv_lens_cuda
prepare_for_spec_dec call (journal Patch #5).

dflash.py:258 unconditionally does:
    attn_metadata.prepare_for_spec_dec("_seq_lens", "_seq_lens_cuda", "kv_lens_cuda")
in the cuda-graph branch. But FlashInferAttentionMetadata has no `kv_lens_cuda`
(it computes kv lens in plan(); only the TRTLLM backend metadata carries it) ->
AttributeError at serve init. Lines 270/286 already `hasattr`-guard it; only 258
does not.

Fix: add `and hasattr(attn_metadata, "kv_lens_cuda")` to the branch condition, so
FlashInfer falls through to the 2-arg else path.

Idempotent, anchored, fails loudly. Run: python3 patch_dflash_kvlens.py [--check]
"""
import argparse
import sys
from pathlib import Path

OLD = (
    "        if spec_metadata.is_cuda_graph and not is_capturing:\n"
    '            attn_metadata.prepare_for_spec_dec("_seq_lens", "_seq_lens_cuda", "kv_lens_cuda")\n'
)
NEW = (
    '        if spec_metadata.is_cuda_graph and not is_capturing and hasattr(attn_metadata, "kv_lens_cuda"):  # === DFLASH-KVLENS-FLASHINFER FIX ===\n'
    '            attn_metadata.prepare_for_spec_dec("_seq_lens", "_seq_lens_cuda", "kv_lens_cuda")\n'
)
MARK = "# === DFLASH-KVLENS-FLASHINFER FIX ==="


def find_target() -> Path:
    import tensorrt_llm
    p = Path(tensorrt_llm.__file__).parent / "_torch" / "speculative" / "dflash.py"
    if not p.exists():
        sys.exit(f"FATAL: {p} not found")
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    p = find_target()
    src = p.read_text()
    if MARK in src:
        print("ALREADY PATCHED:", p)
        return
    if args.check:
        print("NOT patched:", p)
        return
    n = src.count(OLD)
    if n != 1:
        sys.exit(f"FATAL: anchor found {n}x (expected 1) — drift, inspect dflash.py:258")
    src = src.replace(OLD, NEW, 1)
    p.with_suffix(".py.kvlensbak").write_text(p.read_text())
    p.write_text(src)
    print(f"PATCHED: {p}")


if __name__ == "__main__":
    main()
