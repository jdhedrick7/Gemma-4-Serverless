#!/usr/bin/env python3
"""Wire aux-hidden-state capture into TRT-LLM's Gemma4 so EAGLE3/DFlash spec
decode works (support matrix shows Gemma4 EAGLE-3/DFlash = No because this hook
is absent; Llama4/Qwen3/GptOss have it).

Idempotent, anchor-based, verifies with py_compile. Run inside the TRT-LLM
container: python3 patch_trtllm_gemma4.py

Mechanism (verified against main @ 2026-07):
- interface.py: `maybe_capture_hidden_states(layer_id, hidden_states, residual)`
  + `is_layer_capture(layer_id)`.
- dflash.py: `to_save = hidden_states + residual if residual is not None else
  hidden_states`, written to a per-layer buffer indexed by target_layer_ids.
- Gemma4DecoderLayer already does `hidden_states = residual + hidden_states`
  inline, so at `return` hidden_states == HF output_hidden_states[i+1] (what the
  RedHat/finetuned DFlash head was trained on). => pass (hidden_states, None);
  do NOT pass a separate residual (would double-count).
- spec_metadata reaches the layer via **kwargs but leaks into self_attn(**kwargs)
  and errors -> add an explicit `spec_metadata=None` param to intercept it.
"""
import importlib.util
import py_compile
import re
from pathlib import Path

MARK = "# [dflash-patch]"


def find_modeling_gemma4() -> Path:
    spec = importlib.util.find_spec("tensorrt_llm")
    if spec is None or not spec.submodule_search_locations:
        raise SystemExit("tensorrt_llm not importable — run inside the TRT-LLM container")
    root = Path(spec.submodule_search_locations[0])
    p = root / "_torch" / "models" / "modeling_gemma4.py"
    if not p.exists():
        raise SystemExit(f"not found: {p}")
    return p


def main() -> None:
    p = find_modeling_gemma4()
    s = p.read_text()
    if MARK in s:
        print("already patched")
        return
    orig = s

    # 1) add explicit spec_metadata param to Gemma4DecoderLayer.forward.
    #    Anchor on the unique per_layer_input param immediately preceding **kwargs.
    sig_anchor = "        per_layer_input: Optional[torch.Tensor] = None,\n        **kwargs,"
    if sig_anchor not in s:
        raise SystemExit("signature anchor not found — inspect modeling_gemma4.py forward()")
    s = s.replace(
        sig_anchor,
        "        per_layer_input: Optional[torch.Tensor] = None,\n"
        f"        spec_metadata=None,  {MARK} intercept from kwargs so it doesn't reach self_attn\n"
        "        **kwargs,",
        1,
    )

    # 2) capture the layer's final residual-stream value right before return.
    ret_anchor = "        # Layer scalar\n        hidden_states = hidden_states * self.layer_scalar\n\n        return hidden_states"
    if ret_anchor not in s:
        raise SystemExit("return anchor not found — inspect end of Gemma4DecoderLayer.forward")
    capture = (
        "        # Layer scalar\n"
        "        hidden_states = hidden_states * self.layer_scalar\n\n"
        f"        {MARK} EAGLE3/DFlash aux capture: post-residual value == HF output_hidden_states[i+1]\n"
        "        if spec_metadata is not None and getattr(spec_metadata, \"is_layer_capture\", None) \\\n"
        "                and spec_metadata.is_layer_capture(self.layer_idx):\n"
        "            spec_metadata.maybe_capture_hidden_states(self.layer_idx, hidden_states, None)\n\n"
        "        return hidden_states"
    )
    s = s.replace(ret_anchor, capture, 1)

    if s == orig:
        raise SystemExit("no changes applied")
    bak = p.with_suffix(".py.bak")
    if not bak.exists():
        bak.write_text(orig)
    p.write_text(s)
    py_compile.compile(str(p), doraise=True)
    print(f"patched {p}")
    print(f"backup  {bak}")
    print("verify: grep -n '\\[dflash-patch\\]'", p)


if __name__ == "__main__":
    main()
