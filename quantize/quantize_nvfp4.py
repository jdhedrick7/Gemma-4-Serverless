#!/usr/bin/env python3
"""
Quantize jdfelo/gemma-4-31B-v2 (bf16, multimodal Gemma4ForConditionalGeneration)
to NVFP4 (W4A4, group 16) with llm-compressor, for max single-stream decode
tok/s on a B200 (native FP4 tensor cores) served by vLLM.

Recipe mirrors the known-good RedHatAI/gemma-4-31B-it-NVFP4 checkpoint (same base
model this v2 finetune derives from):
  * scheme = NVFP4  (4-bit float weights + 4-bit float input activations, group 16)
  * targets = Linear
  * ignore  = lm_head + the entire vision stack (kept at bf16 so multimodal
              understanding is untouched; only the language_model linears shrink)

Only the language-model Linears are quantized, so TEXT calibration is correct and
sufficient (identical approach to the official NVFP4 llama3 example). 512 samples
@ 2048 tokens run through THIS model's own chat template.

Requires (installed by run_pod.sh):
  transformers==5.8.1   (Gemma4 needs transformers>=5.5; 5.8.1 is the llm-compressor pin)
  llm-compressor @ git main   (PyPI 0.10.0.1 can't load Gemma4)
  compressed-tensors, datasets, accelerate, huggingface_hub, hf_transfer

Usage:
  python quantize_nvfp4.py \
      --src jdfelo/gemma-4-31B-v2 \
      --dst-dir /workspace/gemma-4-31B-v2-NVFP4 \
      --push-repo jdfelo/gemma-4-31B-v2-NVFP4     # optional HF push (private)
"""
import argparse
import os
import sys
import time

import torch

# --- model class: prefer the explicit Gemma4 multimodal class ----------------
try:
    from transformers import Gemma4ForConditionalGeneration as ModelCls
    _MODEL_CLS_NAME = "Gemma4ForConditionalGeneration"
except Exception:  # pragma: no cover - older transformers
    from transformers import AutoModelForCausalLM as ModelCls
    _MODEL_CLS_NAME = "AutoModelForCausalLM"

from transformers import AutoProcessor, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier


# Language model stays; everything multimodal is preserved at bf16.
# Mirrors RedHatAI/gemma-4-31B-it-NVFP4's effective ignore set.
IGNORE = [
    "lm_head",
    "re:.*vision_tower.*",
    "re:.*multi_modal_projector.*",
    "re:.*embed_vision.*",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="NVFP4 quantize Gemma-4-31B v2")
    p.add_argument("--src", default=os.getenv("SRC_MODEL", "jdfelo/gemma-4-31B-v2"))
    p.add_argument("--dst-dir", default=os.getenv("DST_DIR", "/workspace/gemma-4-31B-v2-NVFP4"))
    p.add_argument("--push-repo", default=os.getenv("DST_MODEL", ""),
                   help="HF repo id to push to (empty = don't push).")
    p.add_argument("--private", action="store_true", default=True,
                   help="Create the HF push repo as private.")
    p.add_argument("--calib-dataset", default=os.getenv("CALIB_DATASET", "HuggingFaceH4/ultrachat_200k"))
    p.add_argument("--calib-split", default=os.getenv("CALIB_SPLIT", "train_sft"))
    p.add_argument("--num-samples", type=int, default=int(os.getenv("NUM_CALIBRATION_SAMPLES", "512")))
    p.add_argument("--max-seq-len", type=int, default=int(os.getenv("MAX_SEQUENCE_LENGTH", "2048")))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--skip-sanity", action="store_true", help="Skip the post-quant sample generation.")
    return p.parse_args()


def build_calibration(tokenizer, dataset_id: str, split: str, n: int, max_len: int, seed: int):
    """Text calibration set: chat-templated, tokenized, ready for oneshot."""
    from datasets import load_dataset

    ds = load_dataset(dataset_id, split=f"{split}[:{n}]")
    ds = ds.shuffle(seed=seed)

    def _to_text(example):
        # ultrachat-style {"messages":[...]}; fall back to a raw "text" field.
        msgs = example.get("messages")
        if msgs:
            return {"text": tokenizer.apply_chat_template(msgs, tokenize=False)}
        return {"text": example.get("text", "")}

    ds = ds.map(_to_text)

    def _tok(sample):
        return tokenizer(
            sample["text"],
            padding=False,
            max_length=max_len,
            truncation=True,
            add_special_tokens=False,
        )

    return ds.map(_tok, remove_columns=ds.column_names)


def main() -> int:
    args = parse_args()
    t0 = time.time()

    print(f"[quant] model class : {_MODEL_CLS_NAME}", flush=True)
    print(f"[quant] source      : {args.src}", flush=True)
    print(f"[quant] dst dir     : {args.dst_dir}", flush=True)
    print(f"[quant] scheme      : NVFP4 (W4A4, group 16)  ignore={IGNORE}", flush=True)
    print(f"[quant] calibration : {args.calib_dataset}[{args.calib_split}] "
          f"n={args.num_samples} seq={args.max_seq_len}", flush=True)

    # bf16 load on CPU; llm-compressor onloads each layer to GPU sequentially.
    print("[quant] loading model (bf16, CPU)…", flush=True)
    model = ModelCls.from_pretrained(args.src, torch_dtype="auto", low_cpu_mem_usage=True)
    tokenizer = AutoTokenizer.from_pretrained(args.src)

    print("[quant] building calibration set…", flush=True)
    ds = build_calibration(
        tokenizer, args.calib_dataset, args.calib_split,
        args.num_samples, args.max_seq_len, args.seed,
    )

    recipe = QuantizationModifier(targets="Linear", scheme="NVFP4", ignore=IGNORE)

    print("[quant] running oneshot NVFP4…", flush=True)
    oneshot(
        model=model,
        dataset=ds,
        recipe=recipe,
        max_seq_length=args.max_seq_len,
        num_calibration_samples=args.num_samples,
    )
    print(f"[quant] oneshot done in {time.time() - t0:.0f}s", flush=True)

    if not args.skip_sanity:
        try:
            from compressed_tensors.offload import dispatch_model
            print("\n[quant] ===== SANITY GENERATION =====", flush=True)
            dispatch_model(model)
            ids = tokenizer("The capital of France is", return_tensors="pt").input_ids.to(model.device)
            out = model.generate(ids, max_new_tokens=40, disable_compile=True)
            print(tokenizer.decode(out[0], skip_special_tokens=True), flush=True)
            print("[quant] ================================\n", flush=True)
        except Exception as e:  # sanity gen is best-effort; never fail the run on it
            print(f"[quant] sanity gen skipped: {e!r}", flush=True)

    # Save compressed. Also save the PROCESSOR so multimodal + chat template survive.
    os.makedirs(args.dst_dir, exist_ok=True)
    print(f"[quant] saving compressed checkpoint -> {args.dst_dir}", flush=True)
    model.save_pretrained(args.dst_dir, save_compressed=True)
    tokenizer.save_pretrained(args.dst_dir)
    try:
        AutoProcessor.from_pretrained(args.src).save_pretrained(args.dst_dir)
        print("[quant] saved processor (vision preprocessing + chat template)", flush=True)
    except Exception as e:
        print(f"[quant] processor save skipped: {e!r}", flush=True)

    if args.push_repo:
        from huggingface_hub import HfApi
        print(f"[quant] pushing to HF repo (private={args.private}): {args.push_repo}", flush=True)
        api = HfApi()
        api.create_repo(args.push_repo, private=args.private, exist_ok=True)
        api.upload_folder(folder_path=args.dst_dir, repo_id=args.push_repo, repo_type="model")
        print("[quant] push complete", flush=True)

    print(f"[quant] ALL DONE in {time.time() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
