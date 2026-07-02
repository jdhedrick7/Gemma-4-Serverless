#!/usr/bin/env bash
# SpecForge (DFlash trainer) in an ISOLATED venv on the pod.
#
# Why this shape:
#  - venv --system-site-packages: reuse pod's torch 2.11+cu128 / transformers /
#    datasets (SpecForge pins torch==2.11.0 = same). vllm env stays untouched.
#  - install --no-deps: SpecForge hard-deps sglang==0.5.14, which we do NOT
#    want (multi-GB, conflicts with vllm). The HF target backend never touches
#    it; two module-level imports get try/except guards below.
#  - patch --draft-init-path into train_dflash.py: upstream always random-inits
#    DFlashDraftModel(config); we warm-start from the converted RedHat head.
set -euo pipefail

VENV=/workspace/sfvenv
SF=/workspace/SpecForge

[ -d "$VENV" ] || python3 -m venv --system-site-packages "$VENV"
. "$VENV/bin/activate"

[ -d "$SF" ] || git clone -q https://github.com/sgl-project/SpecForge "$SF"
cd "$SF"

pip install -q --no-deps -e .
pip install -q tensorboard wandb psutil pydantic ninja packaging 2>&1 | tail -1 || true

# ---- guard sglang imports (HF backend never uses them) -----------------------
python3 - <<'PY'
from pathlib import Path

# args.py: ATTENTION_BACKEND_CHOICES constant
p = Path("specforge/args.py")
s = p.read_text()
old = "from sglang.srt.server_args import ATTENTION_BACKEND_CHOICES"
if old in s and "except ImportError" not in s:
    s = s.replace(old, (
        "try:\n"
        "    from sglang.srt.server_args import ATTENTION_BACKEND_CHOICES\n"
        "except ImportError:  # hf backend: sglang not installed\n"
        "    ATTENTION_BACKEND_CHOICES = ['flashinfer', 'triton', 'torch_native', 'fa3']\n"
    ))
    p.write_text(s)
    print("patched args.py")

# dflash_target_model.py: whole sglang import block + SGLang class
p = Path("specforge/modeling/target/dflash_target_model.py")
s = p.read_text()
if "_SGLANG_AVAILABLE" not in s:
    lines = s.splitlines(keepends=True)
    out = []
    for l in lines:
        if l.startswith("from sglang") or l.startswith("import sglang") or l.strip() == "from .sglang_backend import SGLangRunner":
            out.append("# [patched-out sglang import] " + l)
        else:
            out.append(l)
    s = "".join(out)
    stub = (
        "\n# [patch] sglang unavailable: stub names referenced in annotations/bodies\n"
        "_SGLANG_AVAILABLE = False\n"
        "ModelConfig = Req = ScheduleBatch = Scheduler = CacheInitParams = None\n"
        "RadixCache = CaptureHiddenMode = ForwardBatch = SamplingParams = None\n"
        "ServerArgs = SpeculativeAlgorithm = SGLangRunner = None\n"
        "def require_mlp_sync(*a, **k): raise RuntimeError('sglang not installed')\n"
        "def require_mlp_tp_gather(*a, **k): raise RuntimeError('sglang not installed')\n\n"
    )
    anchor = "from specforge.distributed import get_tp_group"
    assert anchor in s, "stub anchor moved"
    s = s.replace(anchor, anchor + "\n" + stub, 1)
    p.write_text(s)
    print("patched dflash_target_model.py")
PY

# ---- add --draft-init-path warm start ----------------------------------------
python3 - <<'PY'
from pathlib import Path
p = Path("scripts/train_dflash.py")
s = p.read_text()
if "--draft-init-path" not in s:
    s = s.replace(
        'model_group.add_argument("--draft-config-path", type=str, default=None)',
        'model_group.add_argument("--draft-config-path", type=str, default=None)\n'
        '    model_group.add_argument("--draft-init-path", type=str, default=None,\n'
        '                             help="dir with model.safetensors to warm-start the draft")',
    )
    anchor = "draft_model = DFlashDraftModel(draft_config).to(device=device, dtype=torch.bfloat16)"
    inject = anchor + """
    if getattr(args, "draft_init_path", None):
        from safetensors.torch import load_file as _lf
        _sd = _lf(os.path.join(args.draft_init_path, "model.safetensors"))
        _missing, _unexpected = draft_model.load_state_dict(_sd, strict=False)
        print_on_rank0(f"[warm-start] loaded {len(_sd)} tensors from {args.draft_init_path}; "
                       f"missing={len(_missing)} unexpected={len(_unexpected)}")
        assert not _unexpected, f"unexpected tensors: {_unexpected[:5]}"
"""
    assert anchor in s, "anchor moved — inspect train_dflash.py"
    s = s.replace(anchor, inject)
    p.write_text(s)
    print("patched train_dflash.py (--draft-init-path)")
PY

# ---- smoke: imports + draft init + warm start --------------------------------
python3 - <<'PY'
import torch, os
from transformers import AutoConfig
from specforge.modeling.draft.dflash import DFlashDraftModel
init_dir = "/workspace/dflash_init"
cfg = AutoConfig.from_pretrained(init_dir)
m = DFlashDraftModel(cfg)
from safetensors.torch import load_file
sd = load_file(os.path.join(init_dir, "model.safetensors"))
missing, unexpected = m.load_state_dict(sd, strict=False)
print("draft params:", sum(p.numel() for p in m.parameters())/1e9, "B")
print("warm-start: loaded", len(sd), "missing", len(missing), "unexpected", len(unexpected))
assert not unexpected, unexpected[:5]
# anything missing must be non-RedHat extras (rotary buffers etc.), print them
print("missing keys:", missing[:10])
PY

echo "SPECFORGE SETUP OK"
