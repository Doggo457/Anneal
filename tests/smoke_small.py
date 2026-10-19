"""Smoke test for the training stack on native Windows + ROCm (Radeon 8060S, gfx1151).

QLoRA (4-bit NF4) with Unsloth on a qwen3_5-architecture model (Gated DeltaNet + attention, like
Qwen3.8-27B): LoRA r=16 on q/k/v/o/gate/up/down, Unsloth gradient checkpointing, bf16, adamw_8bit,
sequences of up to --max-len tokens built by packing short CodeAlpaca rows into multi-turn chats.
Loss on the final assistant turn only (anneal/data.py; --loss full scores every token, as the runs before
19 Oct 2026 did). --save-adapter writes the LoRA adapter to <run folder>\\adapter.

Defaults are the original small test (Qwen/Qwen3.5-0.8B, 30 steps at 4096). The 27B smoke test is
  smoke_small.py --model unsloth/Qwen3.8-27B-unsloth-bnb-4bit --tag smoke_27b --steps 10

Everything goes to the run folder (C:\\AI\\training\\results\\<tag>, or --run-dir): run.log, metrics.jsonl,
status.json, summary.json. Creating a file named STOP there pauses the run: it saves a checkpoint and exits
cleanly; --resume continues from the newest checkpoint.
"""

import argparse
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # C:\AI\training
from anneal.runtime import (RESULTS_DIR, CountingCollator, Run, WindowsMemorySampler, gib,  # noqa: E402
                            last_checkpoint, trainer_callback)

os.environ.setdefault("UNSLOTH_ENABLE_LOGGING", "1")


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    ap.add_argument("--dataset", default="sahil2801/CodeAlpaca-20k")
    ap.add_argument("--rows", type=int, default=200)
    ap.add_argument("--max-len", type=int, default=4096)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--gdn-heads", choices=["stock", "auto", "interleave", "tile"], default="stock",
                    help="GatedDeltaNet value-head pairing (anneal/gdn_heads.py): stock (default; also auto, "
                         "interleave) is transformers' own, value head j -> key head j // 3, and matches llama.cpp; "
                         "tile (j -> j %% 16) is the old, wrong patch, only to reproduce runs made before 18 Oct 2026")
    ap.add_argument("--short-conv", choices=["torch", "shift"], default="shift",
                    help="GatedDeltaNet short convolution: shift (anneal/short_conv.py, 1.6x faster 27B steps at 8K) "
                         "or torch (transformers' F.conv1d fallback)")
    ap.add_argument("--loss", choices=["assistant", "full"], default="assistant",
                    help="assistant: labels -100 before the final assistant turn (anneal/data.py); full: every token")
    ap.add_argument("--save-adapter", action="store_true", help="save the LoRA adapter to <run folder>\\adapter")
    ap.add_argument("--tag", default="smoke_small")
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--resume", action="store_true", help="continue from the newest checkpoint in the run folder")
    return ap.parse_args()


def versions():
    from importlib.metadata import PackageNotFoundError, version

    import torch
    out = {"python": sys.version.split()[0], "torch": torch.__version__, "torch.version.hip": torch.version.hip}
    for pkg in ("triton-windows", "unsloth", "unsloth_zoo", "transformers", "trl", "peft", "bitsandbytes",
                "accelerate", "datasets"):
        try:
            out[pkg] = version(pkg)
        except PackageNotFoundError:
            out[pkg] = None
    try:
        from unsloth_zoo._vendored import fla
        out["fla (vendored in unsloth_zoo)"] = getattr(fla, "__version__", "present")
    except Exception as exc:
        out["fla (vendored in unsloth_zoo)"] = f"not importable: {exc}"
    return out


def fla_status():
    """Which gated-delta / short-conv implementation the qwen3_5 layers will actually run."""
    info = {}
    try:
        from transformers.utils.import_utils import is_flash_linear_attention_available
        info["is_flash_linear_attention_available"] = bool(is_flash_linear_attention_available())
    except Exception as exc:
        info["is_flash_linear_attention_available"] = f"error: {exc}"
    try:
        from unsloth_zoo.temporary_patches.fla_vendor import fla_unavailable_reason
        info["fla_unavailable_reason"] = fla_unavailable_reason()
    except Exception as exc:
        info["fla_unavailable_reason"] = f"error: {exc}"
    # transformers' qwen3_5 modeling calls these module-level functions; a kernel-hub wrapper keeps the
    # implementation it dispatches to in its closure (fla Triton kernel, or the pure PyTorch fallback).
    impls = {}
    try:
        from transformers.models.qwen3_5 import modeling_qwen3_5 as mq
        for name in ("torch_chunk_gated_delta_rule", "torch_recurrent_gated_delta_rule", "causal_conv1d_fn",
                     "causal_conv1d_update"):
            fn = getattr(mq, name, None)
            target = fn
            for cell in getattr(fn, "__closure__", None) or ():
                try:
                    if callable(cell.cell_contents):
                        target = cell.cell_contents
                        break
                except ValueError:
                    pass
            impls[name] = None if fn is None else \
                f"{getattr(target, '__module__', '?')}.{getattr(target, '__name__', type(target).__name__)}"
    except Exception as exc:
        impls["error"] = str(exc)
    info["qwen3_5 kernels resolve to"] = impls
    return info


def build_dataset(tokenizer, name, rows, max_len, loss="assistant"):
    """Pack short instruction/answer pairs into multi-turn chats of close to max_len tokens, then encode each with
    anneal/data.py (labels -100 before the final assistant turn, or every token with loss="full")."""
    from datasets import Dataset, load_dataset

    from anneal.data import MaskStats, codealpaca_messages, encode_chat, template_ids

    data = load_dataset(name, split=f"train[:{rows}]")
    chats, messages, length = [], [], 0
    for r in data:
        turn = codealpaca_messages(r)
        n = len(template_ids(tokenizer, turn))
        if messages and length + n > max_len - 64:
            chats.append(messages)
            messages, length = [], 0
        messages += turn
        length += n
    if messages:
        chats.append(messages)
    stats = MaskStats()
    encoded = [e for e in (encode_chat(tokenizer, m, max_len=max_len, loss=loss, stats=stats) for m in chats) if e]
    # Drop a short leftover chat so every step trains on a near-max_len sequence (steady timing).
    short = [e for e in encoded if len(e["input_ids"]) < max_len // 2]
    if short and len(short) < len(encoded):
        encoded = [e for e in encoded if len(e["input_ids"]) >= max_len // 2]
        stats.kept -= len(short)
        stats.tokens_total -= sum(len(e["input_ids"]) for e in short)
        stats.tokens_trained -= sum(e["n_trained"] for e in short)
        stats.dropped_short_leftover = len(short)
    lens = [len(e["input_ids"]) for e in encoded]
    trained = [e["n_trained"] for e in encoded]
    print(f"dataset: {rows} rows of {name} -> {len(encoded)} chats, tokens min/mean/max "
          f"{min(lens)}/{sum(lens) // len(lens)}/{max(lens)}; loss on {loss} tokens: trained min/mean/max "
          f"{min(trained)}/{sum(trained) // len(trained)}/{max(trained)}; mask stats {stats.as_dict()}")
    ds = Dataset.from_dict({"input_ids": [e["input_ids"] for e in encoded], "labels": [e["labels"] for e in encoded]})
    return ds, encoded, stats


def check_collated_mask(trainer, encoded):
    """The labels the trainer will actually see for one row (after TRL/Unsloth dataset prep and the collator)
    must be the helper's labels."""
    row = trainer.train_dataset[0]
    batch = trainer.data_collator([row])
    ids = batch["input_ids"][0].tolist()
    labels = batch["labels"][0].tolist()
    want = next((e for e in encoded if e["input_ids"] == ids), None)
    ok = want is not None and labels == want["labels"]
    out = {"collator": type(trainer.data_collator).__name__, "row_found": want is not None,
           "labels_match_helper": ok, "batch_tokens": len(ids),
           "batch_trained": sum(1 for x in labels[1:] if x != -100)}
    print(f"collated batch check: {out}")
    return out


def main():
    args = parse_args()
    run = Run(args.run_dir or os.path.join(RESULTS_DIR, args.tag))
    run.capture_output()
    run.write_status(state="starting", tag=args.tag, model=args.model, script=os.path.abspath(__file__))
    print(f"== smoke_small.py {time.strftime('%Y-%m-%d %H:%M:%S')}  args: {vars(args)}")
    print("env: HF_HOME=%s HF_HUB_DISABLE_XET=%s CC=%s BNB_ROCM_VERSION=%s" % (
        os.environ.get("HF_HOME"), os.environ.get("HF_HUB_DISABLE_XET"), os.environ.get("CC"),
        os.environ.get("BNB_ROCM_VERSION")))
    try:
        return train(args, run)
    except Exception as exc:
        import traceback
        traceback.print_exc()
        run.write_status(state="failed", error=f"{type(exc).__name__}: {exc}")
        run.metric(event="failed", error=f"{type(exc).__name__}: {exc}")
        return 1


def train(args, run):
    import unsloth  # noqa: F401  (must be imported before transformers / trl / peft)
    from unsloth import FastLanguageModel

    from anneal import gdn_heads, short_conv
    print(gdn_heads.apply(args.model, args.gdn_heads))
    print(short_conv.apply(args.short_conv))

    import torch
    import bitsandbytes
    from bitsandbytes.cextension import lib as bnb_lib
    from trl import SFTConfig, SFTTrainer

    if os.environ.get("ANNEAL_GUARD_DEBUG") == "1":
        from anneal.runtime import install_guard_debug
        install_guard_debug()

    summary = {"args": vars(args), "versions": versions()}
    for k, v in summary["versions"].items():
        print(f"  {k}: {v}")
    summary["device"] = torch.cuda.get_device_name(0)
    summary["arch"] = getattr(torch.cuda.get_device_properties(0), "gcnArchName", "?")
    free, total = torch.cuda.mem_get_info()
    summary["mem_get_info_before_gib"] = {"free": gib(free), "total": gib(total)}
    summary["bitsandbytes_native_lib"] = str(getattr(bnb_lib, "_name", None) or bnb_lib)
    print(f"device: {summary['device']} ({summary['arch']}), mem_get_info free {gib(free)} / total {gib(total)} GiB")
    print(f"bitsandbytes {bitsandbytes.__version__} native lib: {summary['bitsandbytes_native_lib']}")

    sampler = WindowsMemorySampler()
    sampler.start()
    run.write_status(state="loading")
    t0 = time.time()
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model, max_seq_length=args.max_len, load_in_4bit=True, dtype=torch.bfloat16)
    tok = getattr(tokenizer, "tokenizer", tokenizer)  # a processor for vision-language checkpoints
    short_conv.patch_loaded()  # Unsloth's compiled copy of the model code is imported during the load
    summary["load_seconds"] = round(time.time() - t0, 1)
    summary["after_load_allocated_gib"] = gib(torch.cuda.memory_allocated())
    print(f"loaded {args.model} in 4-bit in {summary['load_seconds']} s; class {type(model).__name__}; "
          f"{summary['after_load_allocated_gib']} GiB allocated")

    model = FastLanguageModel.get_peft_model(
        model, r=args.rank, lora_alpha=args.rank, lora_dropout=0, bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing="unsloth", random_state=3407)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"LoRA trainable parameters: {trainable:,}")
    summary["lora_trainable_params"] = trainable
    summary["fla"] = fla_status()
    print("FLA / Gated DeltaNet:", json.dumps(summary["fla"], indent=2, default=str))

    from anneal.data import print_sample
    dataset, encoded, mask_stats = build_dataset(tok, args.dataset, args.rows, args.max_len, args.loss)
    lens = [len(e["input_ids"]) for e in encoded]
    trained_lens = [e["n_trained"] for e in encoded]
    mean_tokens = sum(lens) / len(lens)
    summary["dataset"] = {"name": args.dataset, "rows": args.rows, "chats": len(lens),
                          "tokens_mean": int(mean_tokens), "tokens_min": min(lens), "tokens_max": max(lens),
                          "row_tokens": lens, "row_trained_tokens": trained_lens}
    summary["loss_mask"] = {"loss": args.loss, "stats": mask_stats.as_dict(),
                            "sample_row": print_sample(tok, encoded[0], "sample row 0 (loss mask)")}

    counter = {}
    cb = trainer_callback(run, tokens_per_step=mean_tokens,
                          step_tokens=lambda step: counter["c"].step_tokens(step) if "c" in counter else None)
    trainer = SFTTrainer(
        model=model,
        processing_class=tok,
        train_dataset=dataset,
        callbacks=[cb],
        args=SFTConfig(
            output_dir=run.dir, max_length=args.max_len, packing=False,
            per_device_train_batch_size=1, gradient_accumulation_steps=1, max_steps=args.steps,
            learning_rate=args.lr, warmup_steps=min(3, args.steps // 3), lr_scheduler_type="linear",
            weight_decay=0.0, optim="adamw_8bit", bf16=True, fp16=False, logging_steps=1,
            save_strategy="no", report_to="none", seed=3407, dataset_num_proc=1),
    )
    summary["loss_mask"]["collated_check"] = check_collated_mask(trainer, encoded)
    if not summary["loss_mask"]["collated_check"]["labels_match_helper"]:
        raise RuntimeError("the trainer's labels are not the helper's loss mask (see collated batch check)")
    counter["c"] = trainer.data_collator = CountingCollator(trainer.data_collator)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    resume = last_checkpoint(run.dir) if args.resume else None
    if resume:
        print(f"resuming from {resume}")
    t0 = time.time()
    result = trainer.train(resume_from_checkpoint=resume)
    wall = time.time() - t0

    sampler.stop.set()
    losses = [m["loss"] for m in trainer.state.log_history if "loss" in m]
    steady = cb.step_times[2:] or cb.step_times
    sps = sum(steady) / len(steady) if steady else None
    batches = counter["c"].batches[:len(cb.step_times)]
    steady_b = batches[2:] if len(cb.step_times) > 2 else batches
    exact = len(steady_b) == len(steady) and steady_b
    low = min(range(len(losses)), key=lambda i: losses[i]) if losses else None
    summary["train"] = {
        "steps": len(cb.step_times),
        "first_loss": losses[0] if losses else None,
        "last_loss": losses[-1] if losses else None,
        "lowest_loss": losses[low] if losses else None,
        "lowest_loss_step": low + 1 if losses else None,
        "any_nan_loss": any(math.isnan(x) or math.isinf(x) for x in losses),
        "losses": losses,
        "step_seconds": [round(t, 2) for t in cb.step_times],
        "step_tokens": [b[0] for b in batches],
        "step_trained_tokens": [b[1] for b in batches],
        "first_step_seconds": round(cb.step_times[0], 2) if cb.step_times else None,
        "seconds_per_step_after_step_2": round(sps, 3) if sps else None,
        "tokens_per_second_after_step_2": round(mean_tokens / sps, 1) if sps else None,
        "exact_tokens_per_second_after_step_2": round(sum(b[0] for b in steady_b) / sum(steady), 1) if exact else None,
        "exact_trained_tokens_per_second_after_step_2":
            round(sum(b[1] for b in steady_b) / sum(steady), 1) if exact else None,
        "wall_seconds": round(wall, 1),
        "train_runtime_reported": result.metrics.get("train_runtime"),
        "peak_max_memory_reserved_gib": gib(torch.cuda.max_memory_reserved()),
        "peak_max_memory_allocated_gib": gib(torch.cuda.max_memory_allocated()),
        "paused": cb.paused,
    }
    summary["windows_peak"] = dict(sampler.peak, portal_metrics=bool(sampler.portal_ok))
    summary["gdn_heads"] = gdn_heads.report()
    summary["short_conv"] = short_conv.report()
    if args.save_adapter and not cb.paused:
        path = os.path.join(run.dir, "adapter")
        model.save_pretrained(path)
        tok.save_pretrained(path)
        summary["adapter"] = {"path": path, "files": {f: os.path.getsize(os.path.join(path, f))
                                                      for f in sorted(os.listdir(path))}}
        print(f"adapter saved to {path}: {summary['adapter']['files']}")
        run.metric(event="adapter_saved", path=path)
    free, total = torch.cuda.mem_get_info()
    summary["mem_get_info_after_gib"] = {"free": gib(free), "total": gib(total)}
    print("== summary")
    print(json.dumps(summary, indent=2, default=str))
    with open(os.path.join(run.dir, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=str)
    ok = bool(losses) and not summary["train"]["any_nan_loss"]
    print("RESULT:", "PAUSED" if cb.paused else ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
