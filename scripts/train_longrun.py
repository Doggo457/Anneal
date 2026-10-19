"""Multi-week QLoRA run: warmup-stable-decay schedule, crash-safe resume, decay branches, completion-only loss.

Same YAML config as train_qlora.py, plus the long-run keys below. Written for runs that last weeks on the
EVO-X2, where Windows updates, power cuts and needing the GPU back are certain to interrupt them.

Usage:
  python scripts/train_longrun.py --config experiments/<run>.yaml            # start, or resume after any stop
  python scripts/train_longrun.py --config experiments/<run>.yaml --branch-from models/<run>/milestones/checkpoint-1000
      # cool-down branch: decays the learning rate from that checkpoint to zero over branch_decay_steps,
      # saving a finished adapter at that data size without touching the main run
  To stop cleanly at the next step (saves a checkpoint first): create the file models/<run>/STOP.

Long-run config keys (defaults in brackets):
  schedule: wsd [wsd] | cosine
  warmup_steps: [20]           decay_fraction: [0.2] share of the main run spent decaying at the end
  min_lr_ratio: [0.0]          epochs: [1]
  save_steps: [20]             save_total_limit: [3] rolling checkpoints for resume
  milestone_steps: []          steps whose checkpoints are kept for good (decay branches start from them)
  branch_decay_fraction: [0.1] decay length of a branch, as a share of its starting step
  branch_decay_steps: [0]      if set, overrides branch_decay_fraction
  completion_only: [true]      loss on the assistant turn only, not on the prompt
  eval_steps: [0]              0 = no in-process eval (evaluate checkpoints out of process with llama-server)
  engine: [unsloth] | hf       hf = plain transformers + PEFT + bitsandbytes (the fallback path, and CPU tests)

Resume: re-running the same command continues from the newest checkpoint in models/<run>/checkpoints with
the optimizer, scheduler, RNG and data position restored (Hugging Face Trainer). It refuses to resume if
train_file has changed since the run started (SHA-256 in models/<run>/data.sha256), because the data
position would then point at different rows.

Decay branches: a branch resumes from a milestone checkpoint at step S with the same warmup and a stable
phase that ends at S, then decays over D steps on the rows that come next in the data order. Its
adapter lands in models/<run>/branches/<checkpoint-S>/adapter. Trainer restores the scheduler's step
count from the checkpoint but takes the decay shape from this run's settings (LambdaLR does not save the
schedule function), which is what lets one checkpoint seed a differently shaped schedule.
"""
import argparse
import hashlib
import json
import math
import os
import shutil
import time

import yaml


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def latest_checkpoint(folder):
    if not os.path.isdir(folder):
        return None
    steps = [int(d.split("-")[1]) for d in os.listdir(folder)
             if d.startswith("checkpoint-") and d.split("-")[1].isdigit()
             and os.path.exists(os.path.join(folder, d, "trainer_state.json"))]
    return os.path.join(folder, f"checkpoint-{max(steps)}") if steps else None


def tokenize_rows(rows, tok, template_kwargs, seq_len, completion_only):
    """input_ids + labels per row. Labels are -100 over the prompt when completion_only. Rows over seq_len
    are dropped (never truncated) and counted."""
    out, too_long, misaligned = [], 0, 0
    for ex in rows:
        msgs = ex["messages"]
        full = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False, **template_kwargs)
        ids = tok(full, add_special_tokens=False)["input_ids"]
        if len(ids) > seq_len:
            too_long += 1
            continue
        labels = list(ids)
        if completion_only:
            prompt = tok.apply_chat_template(msgs[:-1], tokenize=False, add_generation_prompt=True, **template_kwargs)
            p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
            n = 0
            while n < min(len(p_ids), len(ids)) and p_ids[n] == ids[n]:
                n += 1
            if n < len(p_ids):
                misaligned += 1  # template renders the prompt differently on its own; mask up to the divergence
            labels[:n] = [-100] * n
        out.append({"input_ids": ids, "labels": labels})
    return out, too_long, misaligned


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--branch-from", default=None, help="milestone checkpoint to start a decay branch from")
    p.add_argument("--smoke-test", action="store_true", help="200 rows, 30 steps, 512-token rows, own folder")
    args = p.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if args.smoke_test:
        cfg.update(max_samples=200, seq_len=min(int(cfg.get("seq_len", 1024)), 512), save_steps=10,
                   milestone_steps=[10], warmup_steps=3)
        cfg["run_name"] = cfg["run_name"] + "-smoke"
    run_dir = os.path.join(cfg.get("models_dir", "models"), cfg["run_name"])
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    stop_file = os.path.join(run_dir, "STOP")
    if os.path.exists(stop_file):
        os.remove(stop_file)  # starting again means the stop request has been handled

    # Same data or nothing: resuming against a changed file would silently train on different rows.
    digest = sha256_file(cfg["train_file"])
    digest_path = os.path.join(run_dir, "data.sha256")
    if os.path.exists(digest_path):
        with open(digest_path, encoding="utf-8") as f:
            if f.read().strip() != digest:
                raise SystemExit(f"{cfg['train_file']} has changed since this run started. Start a new run_name "
                                 "for new data, or restore the original file to resume.")
    else:
        with open(digest_path, "w", encoding="utf-8") as f:
            f.write(digest + "\n")

    engine = cfg.get("engine", "unsloth")
    if engine == "unsloth":
        from unsloth import FastLanguageModel, FastModel  # noqa: E402  (must come before transformers/trl)
    import torch  # noqa: E402
    from datasets import Dataset, load_dataset  # noqa: E402
    from transformers import TrainerCallback, set_seed  # noqa: E402
    from trl import SFTConfig, SFTTrainer  # noqa: E402

    set_seed(int(cfg.get("seed", 42)))  # before the adapter is created, so its initial weights are reproducible

    load_in_4bit = bool(cfg.get("load_in_4bit", True))
    load_in_8bit = bool(cfg.get("load_in_8bit", False))
    use_bf16 = bool(cfg.get("bf16", True))
    on_gpu = torch.cuda.is_available()
    lora = dict(r=int(cfg["lora_r"]), lora_alpha=int(cfg["lora_alpha"]),
                lora_dropout=float(cfg.get("lora_dropout", 0.0)), target_modules=cfg["target_modules"], bias="none")
    if engine == "unsloth":
        Loader = FastModel if cfg.get("loader", "FastLanguageModel") == "FastModel" else FastLanguageModel
        model, tok = Loader.from_pretrained(model_name=cfg["base_model"], max_seq_length=int(cfg["seq_len"]),
                                            load_in_4bit=load_in_4bit, load_in_8bit=load_in_8bit,
                                            dtype=None if (load_in_4bit or load_in_8bit) else torch.bfloat16)
        model = Loader.get_peft_model(model, use_gradient_checkpointing="unsloth",
                                      random_state=int(cfg.get("seed", 42)), **lora)
    else:
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        quant = None
        if load_in_4bit or load_in_8bit:
            quant = BitsAndBytesConfig(load_in_4bit=load_in_4bit, load_in_8bit=load_in_8bit,
                                       bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                                       bnb_4bit_compute_dtype=torch.bfloat16)
        model = AutoModelForCausalLM.from_pretrained(cfg["base_model"], quantization_config=quant,
                                                     dtype=torch.bfloat16 if use_bf16 else torch.float32,
                                                     device_map={"": 0} if on_gpu else None)
        tok = AutoTokenizer.from_pretrained(cfg["base_model"])
        if quant is not None:
            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
        model = get_peft_model(model, LoraConfig(task_type="CAUSAL_LM", **lora))
    model.print_trainable_parameters()
    text_tok = getattr(tok, "tokenizer", tok)  # multimodal processors carry the text tokenizer inside

    raw = load_dataset("json", data_files={"train": cfg["train_file"]})["train"]
    if cfg.get("max_samples"):
        raw = raw.select(range(min(int(cfg["max_samples"]), len(raw))))  # a prefix keeps the mixer's shares
    rows, too_long, misaligned = tokenize_rows(raw, text_tok, cfg.get("chat_template_kwargs") or {},
                                               int(cfg["seq_len"]), bool(cfg.get("completion_only", True)))
    print(f"Tokenised {len(rows)} rows; dropped {too_long} over seq_len {cfg['seq_len']}"
          + (f"; {misaligned} rows where the prompt didn't render as a prefix (masked to the divergence)"
             if misaligned else ""))
    train = Dataset.from_list(rows)
    n_tokens = sum(len(r["input_ids"]) for r in rows)
    eval_ds = None
    eval_steps = int(cfg.get("eval_steps", 0))
    if eval_steps and cfg.get("val_file"):
        val_raw = load_dataset("json", data_files={"val": cfg["val_file"]})["val"]
        eval_ds = Dataset.from_list(tokenize_rows(val_raw, text_tok, cfg.get("chat_template_kwargs") or {},
                                                  int(cfg["seq_len"]), bool(cfg.get("completion_only", True)))[0])

    bs, accum = int(cfg.get("batch_size", 1)), int(cfg.get("grad_accum", 8))
    steps_per_epoch = math.ceil(len(train) / (bs * accum))
    main_steps = int(cfg.get("max_steps") or steps_per_epoch * int(cfg.get("epochs", 1)))
    if args.smoke_test:
        main_steps = min(main_steps, 30)
    warmup = int(cfg.get("warmup_steps", 20))
    milestones = sorted(int(s) for s in cfg.get("milestone_steps") or [])

    if args.branch_from:
        with open(os.path.join(args.branch_from, "trainer_state.json"), encoding="utf-8") as f:
            start_step = int(json.load(f)["global_step"])
        decay = int(cfg.get("branch_decay_steps") or max(1, round(float(cfg.get("branch_decay_fraction", 0.1))
                                                                  * start_step)))
        total_steps = start_step + decay
        out_dir = os.path.join(run_dir, "branches", os.path.basename(os.path.normpath(args.branch_from)))
        resume = latest_checkpoint(out_dir) or args.branch_from
        milestones = []
        print(f"Decay branch from step {start_step}: {decay} decay steps, ending at step {total_steps}")
    else:
        total_steps = main_steps
        decay = max(1, round(float(cfg.get("decay_fraction", 0.2)) * total_steps))
        out_dir = ckpt_dir
        resume = latest_checkpoint(ckpt_dir)
    if total_steps > steps_per_epoch * int(cfg.get("epochs", 1)):
        print(f"Warning: {total_steps} steps is more than one pass over {len(train)} rows; rows will repeat.")

    if cfg.get("schedule", "wsd") == "wsd":
        sched = dict(lr_scheduler_type="warmup_stable_decay",
                     lr_scheduler_kwargs={"num_decay_steps": decay, "min_lr_ratio": float(cfg.get("min_lr_ratio", 0.0)),
                                          "decay_type": cfg.get("decay_type", "cosine")})
    else:
        sched = dict(lr_scheduler_type="cosine")

    run_t0 = time.time()

    class LongRun(TrainerCallback):
        """Milestone checkpoints kept for good, a STOP file for a clean pause, and a throughput log."""

        def __init__(self):
            self.last = None

        def on_step_end(self, a, state, control, **kw):
            if state.global_step in milestones:
                control.should_save = True
            if os.path.exists(stop_file):
                print(f"\nSTOP file found: saving at step {state.global_step} and exiting.")
                control.should_save = True
                control.should_training_stop = True
            return control

        def on_save(self, a, state, control, **kw):
            if state.global_step in milestones:
                src = os.path.join(a.output_dir, f"checkpoint-{state.global_step}")
                dst = os.path.join(run_dir, "milestones", f"checkpoint-{state.global_step}")
                if os.path.isdir(src) and not os.path.exists(dst):
                    shutil.copytree(src, dst)
                    print(f"  milestone kept: {dst}")
            return control

        def on_log(self, a, state, control, logs=None, **kw):
            now, seen = time.time(), state.num_input_tokens_seen
            if self.last is not None and seen > self.last[1]:
                rate = (seen - self.last[1]) / max(now - self.last[0], 1e-9)
                with open(os.path.join(run_dir, "throughput.jsonl"), "a", encoding="utf-8") as f:
                    f.write(json.dumps({"step": state.global_step, "tokens_seen": seen,
                                        "tokens_per_s": round(rate, 2), **(logs or {})}) + "\n")
            self.last = (now, seen)

    targs = SFTConfig(
        output_dir=out_dir,
        per_device_train_batch_size=bs,
        gradient_accumulation_steps=accum,
        learning_rate=float(cfg.get("lr", 1e-4)),
        max_steps=total_steps,
        warmup_steps=warmup,
        **sched,
        logging_steps=int(cfg.get("logging_steps", 1)),
        eval_strategy="steps" if eval_ds is not None else "no",
        eval_steps=eval_steps or None,
        save_strategy="steps",
        save_steps=int(cfg.get("save_steps", 20)),
        save_total_limit=int(cfg.get("save_total_limit", 3)),
        bf16=use_bf16 and on_gpu,
        optim=cfg.get("optim", "adamw_8bit" if on_gpu else "adamw_torch"),
        gradient_checkpointing=engine == "hf" and bool(cfg.get("gradient_checkpointing", True)),
        max_length=int(cfg["seq_len"]),
        include_num_input_tokens_seen="non_padding",
        seed=int(cfg.get("seed", 42)),
        report_to="none",
    )
    trainer = SFTTrainer(model=model, processing_class=text_tok, train_dataset=train, eval_dataset=eval_ds,
                         args=targs, callbacks=[LongRun()])
    if on_gpu:
        torch.cuda.reset_peak_memory_stats()
    print(f"{len(train)} rows, {n_tokens:,} tokens, {total_steps} optimizer steps "
          f"({bs * accum} rows each), warmup {warmup}, decay {decay}."
          + (f" Resuming from {resume}." if resume else " Starting fresh."))
    result = trainer.train(resume_from_checkpoint=resume)
    stopped_early = trainer.state.global_step < total_steps
    summary_dir = out_dir if args.branch_from else run_dir
    summary = {
        "config": cfg,
        "branch_from": args.branch_from,
        "global_step": trainer.state.global_step,
        "total_steps": total_steps,
        "stopped_early": stopped_early,
        "session_seconds": round(time.time() - run_t0),
        "tokens_seen": trainer.state.num_input_tokens_seen,
        "peak_reserved_gib": round(torch.cuda.max_memory_reserved() / 2**30, 1) if on_gpu else None,
        "metrics": result.metrics,
        "data_sha256": digest,
    }
    if not stopped_early:
        adapter_dir = os.path.join(summary_dir, "adapter")
        model.save_pretrained(adapter_dir)
        text_tok.save_pretrained(adapter_dir)
        summary["adapter"] = adapter_dir
    with open(os.path.join(summary_dir, "train_summary.json"), "a", encoding="utf-8") as f:
        f.write(json.dumps(summary, default=str) + "\n")
    print("Stopped early; re-run the same command to resume." if stopped_early
          else f"Done at step {trainer.state.global_step}. Adapter: {summary['adapter']}")


if __name__ == "__main__":
    main()
