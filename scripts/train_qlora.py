"""QLoRA fine-tune with Unsloth, driven by a YAML config from experiments/.

TEMPLATE. Written before the hardware arrived, against the Unsloth/TRL APIs as of mid-2026.
Updated 2026-10-17 for transformers 5.x / TRL 1.x: SFTConfig uses max_length and warmup_steps (no
warmup_ratio), and SFTTrainer takes processing_class= (not tokenizer=).

Usage:
  python scripts/train_qlora.py --config experiments/2026-XX-XX_name.yaml --smoke-test   # first, always
  python scripts/train_qlora.py --config experiments/2026-XX-XX_name.yaml

Outputs under models/<run_name>/: adapter/ (the LoRA), checkpoints/, train_summary.json,
and merged-bf16/ if merge_on_finish is true.

Config keys that pick the training mode:
  loader: FastLanguageModel (dense models, the default) | FastModel (MoE / multimodal variants; unused here)
  load_in_4bit: true = QLoRA;  load_in_8bit: true = 8-bit LoRA;  both false = bf16 LoRA
  max_load_gib: abort if the loaded model already uses more than this (catches the MoE case below)
Two things to know (September 2026):
  - Unsloth advises against 4-bit on Qwen3.5-family models. For a dense 20B-40B model on this box,
    8-bit LoRA is the default and 4-bit the fallback.
  - bitsandbytes does not quantise the fused expert tensors of MoE models, which is why this project
    uses a dense model. The max_load_gib guard is there so that any load that comes in far above the
    expected size fails in ten seconds with a clear message instead of an OOM twenty minutes in.
See docs/Model-Candidates.md.
"""
import argparse
import json
import os
import time

import yaml


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--smoke-test", action="store_true", help="200 samples, 30 steps, short sequences")
    args = p.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if args.smoke_test:
        cfg.update(max_steps=30, max_samples=200, seq_len=min(int(cfg.get("seq_len", 1024)), 512))
        cfg["run_name"] = cfg["run_name"] + "-smoke"
    out = os.path.join("models", cfg["run_name"])
    os.makedirs(out, exist_ok=True)
    print("Config:", json.dumps(cfg, indent=2))

    # Unsloth must be imported before transformers/trl so its patches apply.
    from unsloth import FastLanguageModel, FastModel  # noqa: E402
    import torch  # noqa: E402
    from datasets import load_dataset  # noqa: E402
    from trl import SFTConfig, SFTTrainer  # noqa: E402

    # Dense models use FastLanguageModel (the default). FastModel is Unsloth's loader for MoE and
    # multimodal variants; kept selectable for completeness, not used by this project's config.
    Loader = FastModel if cfg.get("loader", "FastLanguageModel") == "FastModel" else FastLanguageModel
    load_in_4bit = bool(cfg.get("load_in_4bit", True))
    load_in_8bit = bool(cfg.get("load_in_8bit", False))
    if load_in_4bit and load_in_8bit:
        raise SystemExit("Set only one of load_in_4bit / load_in_8bit.")
    quantised = load_in_4bit or load_in_8bit
    model, tok = Loader.from_pretrained(
        model_name=cfg["base_model"],
        max_seq_length=int(cfg["seq_len"]),
        load_in_4bit=load_in_4bit,
        load_in_8bit=load_in_8bit,
        dtype=None if quantised else torch.bfloat16,     # bf16 LoRA when nothing is quantised
    )
    torch.cuda.synchronize()
    loaded_gib = torch.cuda.memory_allocated() / 2**30
    print(f"Model loaded: {loaded_gib:.1f} GiB allocated on the GPU "
          f"({'4-bit' if load_in_4bit else '8-bit' if load_in_8bit else 'bf16'})")
    limit = cfg.get("max_load_gib")
    if limit and loaded_gib > float(limit):
        raise SystemExit(
            f"Loaded model uses {loaded_gib:.1f} GiB, above max_load_gib={limit}. "
            "For a MoE model this means bitsandbytes left the fused experts in bf16 (they were not "
            "quantised), so training will not fit. Stop here and see docs/Model-Candidates.md."
        )
    # FastModel may hand back a multimodal processor; the text tokenizer then lives on .tokenizer.
    text_tok = getattr(tok, "tokenizer", tok)
    model = Loader.get_peft_model(
        model,
        r=int(cfg["lora_r"]),
        lora_alpha=int(cfg["lora_alpha"]),
        lora_dropout=float(cfg.get("lora_dropout", 0.0)),
        target_modules=cfg["target_modules"],
        bias="none",
        use_gradient_checkpointing="unsloth",   # trades compute for memory; required at this scale
        random_state=int(cfg.get("seed", 42)),
    )
    model.print_trainable_parameters()

    ds = load_dataset("json", data_files={"train": cfg["train_file"], "val": cfg["val_file"]})
    if cfg.get("max_samples"):
        n = min(int(cfg["max_samples"]), len(ds["train"]))
        ds["train"] = ds["train"].select(range(n))
    template_kwargs = cfg.get("chat_template_kwargs") or {}

    def to_text(ex):
        return {"text": text_tok.apply_chat_template(ex["messages"], tokenize=False,
                                                     add_generation_prompt=False, **template_kwargs)}

    train = ds["train"].map(to_text, remove_columns=ds["train"].column_names)
    val = ds["val"].map(to_text, remove_columns=ds["val"].column_names)
    print("Example formatted row:\n", train[0]["text"][:600])

    # Current SFTConfig takes warmup_steps (int), not warmup_ratio: 3% of the planned optimiser steps.
    bs, accum = int(cfg.get("batch_size", 1)), int(cfg.get("grad_accum", 8))
    max_steps = int(cfg.get("max_steps", -1))
    planned = max_steps if max_steps > 0 else int(len(train) * float(cfg.get("epochs", 1)) / (bs * accum))
    warmup_steps = int(cfg.get("warmup_steps", max(1, round(0.03 * planned))))

    targs = SFTConfig(
        output_dir=os.path.join(out, "checkpoints"),
        per_device_train_batch_size=bs,
        gradient_accumulation_steps=accum,
        learning_rate=float(cfg.get("lr", 2e-4)),
        num_train_epochs=float(cfg.get("epochs", 1)),
        max_steps=max_steps,
        warmup_steps=warmup_steps,
        lr_scheduler_type="cosine",
        logging_steps=5,
        eval_strategy="steps",
        eval_steps=int(cfg.get("eval_steps", 100)),
        save_steps=int(cfg.get("save_steps", 200)),
        save_total_limit=2,
        bf16=True,
        optim=cfg.get("optim", "adamw_8bit"),   # needs bitsandbytes; fall back to "adamw_torch"
        max_length=int(cfg["seq_len"]),          # older TRL: max_seq_length
        dataset_text_field="text",
        seed=int(cfg.get("seed", 42)),
        report_to="none",
    )
    trainer = SFTTrainer(model=model, processing_class=text_tok, train_dataset=train, eval_dataset=val, args=targs)

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    result = trainer.train()
    seconds = time.time() - t0
    peak_gib = torch.cuda.max_memory_reserved() / 2**30

    adapter_dir = os.path.join(out, "adapter")
    model.save_pretrained(adapter_dir)
    text_tok.save_pretrained(adapter_dir)

    summary = {
        "config": cfg,
        "train_seconds": round(seconds),
        "peak_reserved_gib": round(peak_gib, 1),
        "metrics": result.metrics,
        "final_eval": trainer.evaluate(),
    }
    with open(os.path.join(out, "train_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nDone in {seconds / 3600:.2f} h, peak reserved {peak_gib:.1f} GiB. Adapter: {adapter_dir}")
    print("Compare peak memory to scripts/memory_estimate.py and adjust its constants.")

    if cfg.get("merge_on_finish"):
        merged = os.path.join(out, "merged-bf16")
        model.save_pretrained_merged(merged, tok, save_method="merged_16bit")
        print(f"Merged bf16 model: {merged}  -> now run scripts/export_gguf.py, then delete this folder.")


if __name__ == "__main__":
    main()
