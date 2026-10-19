"""Measure memory and speed of 4-bit QLoRA training on fixed-length rows (default 8,192 tokens).

Loads the model in 4-bit with Unsloth, adds LoRA (r=16 on q/k/v/o/gate/up/down, Unsloth gradient
checkpointing), and trains a few steps through the TRL trainer (the real training path). Rows: CodeAlpaca
pairs packed into multi-turn chats of at most --max-len tokens, with loss on the final assistant turn only
(anneal/data.py); --loss full instead cuts the chat token stream into rows of exactly --max-len tokens and
scores every token, as the runs before 19 Oct 2026 did. Records GPU memory after load, after LoRA and at every step,
plus what Windows sees (the portal's per-process GPU dedicated + shared counters) and system RAM, which
matter on this unified-memory machine: the GPU's "shared" memory and Unsloth's offloaded activations both
come out of the same RAM.

Usage (the Training page adds --run-dir):
  measure_memory.py --model unsloth/Qwen3.8-27B-unsloth-bnb-4bit --max-len 8192 --steps 5
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # C:\AI\training
from anneal.runtime import RESULTS_DIR, CountingCollator, Run, gib, trainer_callback  # noqa: E402
from anneal.runtime import WindowsMemorySampler as Sampler  # noqa: E402

os.environ.setdefault("UNSLOTH_ENABLE_LOGGING", "1")


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", default="unsloth/Qwen3.8-27B-unsloth-bnb-4bit")
    ap.add_argument("--dataset", default="sahil2801/CodeAlpaca-20k")
    ap.add_argument("--max-len", type=int, default=8192)
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--gdn-heads", choices=["stock", "auto", "interleave", "tile"], default="stock",
                    help="GatedDeltaNet value-head pairing (anneal/gdn_heads.py): stock (default; also auto, "
                         "interleave) is transformers' own, value head j -> key head j // 3, and matches llama.cpp; "
                         "tile (j -> j %% 16) is the old, wrong patch, only to reproduce runs made before 18 Oct 2026")
    ap.add_argument("--short-conv", choices=["torch", "shift"], default="shift",
                    help="GatedDeltaNet short convolution: shift (anneal/short_conv.py; 84.5 vs 134.6 s per 27B step "
                         "at 8K, results/mem-27b-8k-shiftconv) or torch (transformers' F.conv1d fallback)")
    ap.add_argument("--loss", choices=["assistant", "full"], default="assistant",
                    help="assistant: packed chats of at most --max-len tokens, labels -100 before the final assistant "
                         "turn (anneal/data.py); full: rows of exactly --max-len tokens, every token scored")
    ap.add_argument("--tag", default="measure_memory")
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--resume", action="store_true", help="accepted for the Training page; a measurement starts over")
    return ap.parse_args()


def build_packed_rows(tokenizer, name, max_len, n_rows):
    """CodeAlpaca pairs packed into multi-turn chats of at most max_len tokens, loss on the final assistant turn."""
    from datasets import Dataset, load_dataset

    from anneal.data import MaskStats, codealpaca_messages, encode_chat, pack_pairs

    def pairs():
        for chunk in range(0, 20000, 500):
            for r in load_dataset(name, split=f"train[{chunk}:{chunk + 500}]"):
                yield codealpaca_messages(r)

    stats = MaskStats()
    encoded = [e for e in (encode_chat(tokenizer, m, max_len=max_len, stats=stats)
                           for m in pack_pairs(tokenizer, pairs(), max_len, n_rows)) if e]
    lens = [len(e["input_ids"]) for e in encoded]
    print(f"rows: {len(encoded)} packed chats of {lens} tokens (limit {max_len}) from {name}; trained tokens "
          f"{[e['n_trained'] for e in encoded]}; mask stats {stats.as_dict()}")
    ds = Dataset.from_dict({"input_ids": [e["input_ids"] for e in encoded], "labels": [e["labels"] for e in encoded]})
    return ds, encoded, stats


def build_rows(tokenizer, name, max_len, n_rows):
    """Concatenate CodeAlpaca chats and cut the token stream into rows of exactly max_len tokens (--loss full)."""
    from datasets import Dataset, load_dataset

    need = max_len * n_rows
    ids, used = [], 0
    for chunk in range(0, 20000, 500):
        data = load_dataset(name, split=f"train[{chunk}:{chunk + 500}]")
        for r in data:
            prompt = r["instruction"] + (("\n\n" + r["input"]) if r.get("input") else "")
            text = tokenizer.apply_chat_template([{"role": "user", "content": prompt},
                                                  {"role": "assistant", "content": r["output"]}], tokenize=False)
            ids += tokenizer(text, add_special_tokens=False)["input_ids"]
            used += 1
        if len(ids) >= need:
            break
    rows = [ids[i * max_len:(i + 1) * max_len] for i in range(min(n_rows, len(ids) // max_len))]
    print(f"rows: {len(rows)} x {max_len} tokens from {used} rows of {name}")
    return Dataset.from_dict({"input_ids": rows, "attention_mask": [[1] * max_len for _ in rows]})


def main():
    args = parse_args()
    run = Run(args.run_dir or os.path.join(RESULTS_DIR, args.tag))
    run.capture_output()
    run.write_status(state="starting", model=args.model, script=os.path.abspath(__file__))
    print(f"== measure_memory.py {time.strftime('%Y-%m-%d %H:%M:%S')}  args: {vars(args)}")
    if args.resume:
        print("(--resume: a measurement has no checkpoint to continue from, so it starts over)")
    try:
        return measure(args, run)
    except Exception as exc:
        import traceback
        traceback.print_exc()
        run.write_status(state="failed", error=f"{type(exc).__name__}: {exc}")
        run.metric(event="failed", error=f"{type(exc).__name__}: {exc}")
        return 1


def measure(args, run):
    import unsloth  # noqa: F401
    from unsloth import FastLanguageModel

    import torch
    from trl import SFTConfig, SFTTrainer

    from anneal import gdn_heads, short_conv
    print(gdn_heads.apply(args.model, args.gdn_heads))
    print(short_conv.apply(args.short_conv))

    sampler = Sampler()
    sampler.start()
    out = {"args": vars(args), "torch": torch.__version__, "hip": torch.version.hip,
           "device": torch.cuda.get_device_name(0), "phases": {}}

    def phase(name):
        torch.cuda.synchronize()
        free, total = torch.cuda.mem_get_info()
        out["phases"][name] = {"allocated_gib": gib(torch.cuda.memory_allocated()),
                               "reserved_gib": gib(torch.cuda.memory_reserved()),
                               "peak_reserved_gib": gib(torch.cuda.max_memory_reserved()),
                               "mem_get_info_free_gib": gib(free), "mem_get_info_total_gib": gib(total),
                               "windows_view": dict(sampler.peak)}
        print(f"[{name}] {json.dumps(out['phases'][name])}")
        run.metric(event="phase", phase=name, **{k: v for k, v in out["phases"][name].items() if k != "windows_view"})

    phase("start")
    run.write_status(state="loading")
    t0 = time.time()
    model, tokenizer = FastLanguageModel.from_pretrained(model_name=args.model, max_seq_length=args.max_len,
                                                         load_in_4bit=True, dtype=torch.bfloat16)
    tok = getattr(tokenizer, "tokenizer", tokenizer)
    out["load_seconds"] = round(time.time() - t0, 1)
    print(f"loaded {args.model} in {out['load_seconds']} s ({type(model).__name__})")
    if args.short_conv != "torch":
        print(f"short conv patched in: {short_conv.patch_loaded() or short_conv.report()['modules']}")
    phase("after_load")

    model = FastLanguageModel.get_peft_model(
        model, r=args.rank, lora_alpha=args.rank, lora_dropout=0, bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing="unsloth", random_state=3407)
    out["lora_trainable_params"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"LoRA trainable parameters: {out['lora_trainable_params']:,}")
    phase("after_lora")

    counter = {}
    if args.loss == "full":
        dataset = build_rows(tok, args.dataset, args.max_len, args.steps)
        out["loss_mask"] = {"loss": "full"}
        tokens_per_step = args.max_len
    else:
        from anneal.data import print_sample
        dataset, encoded, stats = build_packed_rows(tok, args.dataset, args.max_len, args.steps)
        out["loss_mask"] = {"loss": "assistant", "stats": stats.as_dict(),
                            "row_tokens": [len(e["input_ids"]) for e in encoded],
                            "row_trained_tokens": [e["n_trained"] for e in encoded],
                            "sample_row": print_sample(tok, encoded[0], "sample row 0 (loss mask)")}
        tokens_per_step = sum(len(e["input_ids"]) for e in encoded) / len(encoded)
    cb = trainer_callback(run, tokens_per_step=tokens_per_step,
                          step_tokens=lambda step: counter["c"].step_tokens(step) if "c" in counter else None)
    trainer = SFTTrainer(
        model=model, processing_class=tok, train_dataset=dataset, callbacks=[cb],
        args=SFTConfig(output_dir=run.dir, max_length=args.max_len, packing=False, per_device_train_batch_size=1,
                       gradient_accumulation_steps=1, max_steps=args.steps, learning_rate=1e-4, warmup_steps=0,
                       optim="adamw_8bit", bf16=True, fp16=False, logging_steps=1, save_strategy="no",
                       report_to="none", seed=3407, dataset_num_proc=1))
    if args.loss != "full":
        row = trainer.train_dataset[0]
        batch = trainer.data_collator([row])
        ids, labels = batch["input_ids"][0].tolist(), batch["labels"][0].tolist()
        want = next((e for e in encoded if e["input_ids"] == ids), None)
        out["loss_mask"]["collated_check"] = {"collator": type(trainer.data_collator).__name__,
                                              "labels_match_helper": bool(want and labels == want["labels"])}
        print(f"collated batch check: {out['loss_mask']['collated_check']}")
        if not out["loss_mask"]["collated_check"]["labels_match_helper"]:
            raise RuntimeError("the trainer's labels are not the helper's loss mask")
    counter["c"] = trainer.data_collator = CountingCollator(trainer.data_collator)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    trainer.train()
    out["train_wall_seconds"] = round(time.time() - t0, 1)
    phase("after_training")
    sampler.stop.set()

    losses = [m["loss"] for m in trainer.state.log_history if "loss" in m]
    steady = cb.step_times[2:] or cb.step_times[1:] or cb.step_times
    sps = sum(steady) / len(steady) if steady else None
    batches = counter["c"].batches[:len(cb.step_times)]
    steady_b = batches[len(cb.step_times) - len(steady):]
    out["train"] = {
        "steps": len(cb.step_times), "step_seconds": [round(t, 2) for t in cb.step_times],
        "step_tokens": [b[0] for b in batches], "step_trained_tokens": [b[1] for b in batches],
        "losses": losses, "any_nan_loss": any(x != x for x in losses),
        "seconds_per_step_steady": round(sps, 2) if sps else None,
        "tokens_per_second_steady": round(tokens_per_step / sps, 1) if sps else None,
        "exact_tokens_per_second_steady": round(sum(b[0] for b in steady_b) / sum(steady), 1)
        if steady and len(steady_b) == len(steady) else None,
        "exact_trained_tokens_per_second_steady": round(sum(b[1] for b in steady_b) / sum(steady), 1)
        if steady and len(steady_b) == len(steady) else None,
        "peak_max_memory_reserved_gib": gib(torch.cuda.max_memory_reserved()),
        "peak_max_memory_allocated_gib": gib(torch.cuda.max_memory_allocated()),
    }
    out["windows_peak"] = dict(sampler.peak, portal_metrics=bool(sampler.portal_ok))
    out["gdn_heads"] = gdn_heads.report()
    out["short_conv"] = short_conv.report()
    print("== summary")
    print(json.dumps(out, indent=2, default=str))
    with open(os.path.join(run.dir, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, default=str)
    ok = bool(losses) and not out["train"]["any_nan_loss"]
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
