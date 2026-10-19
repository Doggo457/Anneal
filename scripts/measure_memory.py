"""Measure, don't estimate: load a model at a chosen precision, attach LoRA, run one training step,
and print peak GPU memory. Run this on the EVO-X2 for each precision before trusting any table.

  python scripts/measure_memory.py --model Qwen/Qwen3.8-27B --precision 8bit --seq-len 1024
  python scripts/measure_memory.py --model Qwen/Qwen3.8-27B --precision 4bit --seq-len 2048
  python scripts/measure_memory.py --model Qwen/Qwen3.8-27B --precision bf16            # expected to OOM on 64GB
  python scripts/measure_memory.py --model Qwen/Qwen3.8-27B --precision 8bit --load-only   # just the weights

It prints three numbers: memory after loading the weights, after adding the LoRA adapter, and the
peak during one forward+backward+optimizer step at the given sequence length. Copy them into
docs/Hardware-Notes.md next to the published figures. If it dies with an out-of-memory error, that
is also a measurement: write down the precision and sequence length that did not fit.

Needs the training stack installed (setup/02). Uses Unsloth if present, plain transformers+PEFT otherwise.
"""
import argparse
import json
import os
import time


def gib(n):
    return n / 2**30


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--precision", choices=["bf16", "8bit", "4bit"], default="8bit")
    p.add_argument("--seq-len", type=int, default=1024)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--load-only", action="store_true", help="stop after loading the weights")
    p.add_argument("--no-unsloth", action="store_true", help="use plain transformers + PEFT even if Unsloth is installed")
    p.add_argument("--out", default="results/memory_measurements.jsonl")
    args = p.parse_args()

    import torch

    if not torch.cuda.is_available():
        raise SystemExit("No GPU visible to torch. Run scripts/check_env.py first.")
    free, total = torch.cuda.mem_get_info()
    print(f"GPU: {torch.cuda.get_device_name(0)} | total {gib(total):.1f} GiB, free {gib(free):.1f} GiB")
    torch.cuda.reset_peak_memory_stats()
    record = {"model": args.model, "precision": args.precision, "seq_len": args.seq_len, "batch": args.batch,
              "lora_r": args.lora_r, "gpu_total_gib": round(gib(total), 1)}

    use_unsloth = not args.no_unsloth
    if use_unsloth:
        try:
            from unsloth import FastLanguageModel
        except Exception as e:  # noqa: BLE001
            print(f"Unsloth unavailable ({type(e).__name__}); falling back to transformers + PEFT")
            use_unsloth = False

    t0 = time.time()
    if use_unsloth:
        model, tok = FastLanguageModel.from_pretrained(
            model_name=args.model,
            max_seq_length=args.seq_len,
            load_in_4bit=args.precision == "4bit",
            load_in_8bit=args.precision == "8bit",
            dtype=None if args.precision != "bf16" else torch.bfloat16,
        )
        tok = getattr(tok, "tokenizer", tok)
    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        qcfg = None
        if args.precision == "4bit":
            qcfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                      bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
        elif args.precision == "8bit":
            qcfg = BitsAndBytesConfig(load_in_8bit=True)
        tok = AutoTokenizer.from_pretrained(args.model)
        model = AutoModelForCausalLM.from_pretrained(
            args.model, quantization_config=qcfg, dtype=torch.bfloat16, device_map={"": 0},
        )
    torch.cuda.synchronize()
    record["load_seconds"] = round(time.time() - t0)
    record["after_load_gib"] = round(gib(torch.cuda.memory_allocated()), 2)
    print(f"[1] weights loaded ({args.precision}): {record['after_load_gib']:.1f} GiB allocated, "
          f"{record['load_seconds']}s")
    if args.load_only:
        finish(record, args)
        return

    targets = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    if use_unsloth:
        model = FastLanguageModel.get_peft_model(model, r=args.lora_r, lora_alpha=2 * args.lora_r, lora_dropout=0,
                                                 target_modules=targets, use_gradient_checkpointing="unsloth")
    else:
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

        if args.precision != "bf16":
            model = prepare_model_for_kbit_training(model)
        model.gradient_checkpointing_enable()
        model = get_peft_model(model, LoraConfig(r=args.lora_r, lora_alpha=2 * args.lora_r, lora_dropout=0,
                                                 target_modules=targets, task_type="CAUSAL_LM"))
    torch.cuda.synchronize()
    record["after_lora_gib"] = round(gib(torch.cuda.memory_allocated()), 2)
    print(f"[2] LoRA r={args.lora_r} attached: {record['after_lora_gib']:.1f} GiB allocated")

    # One real training step at the requested shape.
    model.train()
    trainable = [p_ for p_ in model.parameters() if p_.requires_grad]
    try:
        import bitsandbytes as bnb

        opt = bnb.optim.AdamW8bit(trainable, lr=1e-4)
        record["optimizer"] = "adamw_8bit"
    except Exception:  # noqa: BLE001
        opt = torch.optim.AdamW(trainable, lr=1e-4)
        record["optimizer"] = "adamw_torch"
    ids = torch.randint(100, tok.vocab_size - 1, (args.batch, args.seq_len), device="cuda")
    t0 = time.time()
    out = model(input_ids=ids, labels=ids)
    out.loss.backward()
    opt.step()
    opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    record["step_seconds"] = round(time.time() - t0, 1)
    record["peak_step_gib"] = round(gib(torch.cuda.max_memory_allocated()), 2)
    record["peak_reserved_gib"] = round(gib(torch.cuda.max_memory_reserved()), 2)
    print(f"[3] one train step @ seq {args.seq_len}, batch {args.batch}: peak {record['peak_step_gib']:.1f} GiB "
          f"allocated, {record['peak_reserved_gib']:.1f} GiB reserved, {record['step_seconds']}s "
          f"(loss {out.loss.item():.3f})")
    print(f"    headroom vs this GPU: {gib(total) - record['peak_reserved_gib']:.1f} GiB")
    finish(record, args)


def finish(record, args):
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
    print(f"appended to {args.out}")


if __name__ == "__main__":
    main()
