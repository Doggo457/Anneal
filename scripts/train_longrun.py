"""Multi-week QLoRA run: warmup-stable-decay schedule, crash-safe resume, decay branches, completion-only loss.

Same YAML config as train_qlora.py, plus the long-run keys below. Written for runs that last weeks on the
EVO-X2, where Windows updates, power cuts and needing the GPU back are certain to interrupt them.

Usage:
  python scripts/train_longrun.py --config experiments/<run>.yaml            # start, or resume after any stop
  python scripts/train_longrun.py --config experiments/<run>.yaml --branch-from <run dir>/milestones/checkpoint-1000
      # cool-down branch: decays the learning rate from that checkpoint to zero over branch_decay_steps,
      # saving a finished adapter at that data size without touching the main run
  To stop cleanly at the next step (saves a checkpoint first): create the file <run dir>/STOP (the Training page's
  Pause button does this).

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
  gdn_heads: [stock] | tile    short_conv: [shift] | torch     (EVO-X2 additions, below)
  expect_trainable_params: []  if set, the run refuses to start with any other LoRA parameter count
  empty_cache_between_microbatches: [true]   release the CUDA/HIP allocator cache after every micro-batch
  pinned_offload: [false]      pinned host buffers for Unsloth's offloaded checkpoints (false = pageable)
  warm_up_longest_row: [true]  one forward + backward on the longest row before step 1 (gradients dropped)
  ce_target_gb: []             fixed memory budget for Unsloth's fused cross entropy (UNSLOTH_CE_LOSS_TARGET_GB)
  mem_debug: [false]           log memory just before the loss in every forward
  sdpa_probe, sdpa_contiguous, blas_backend, profile_steps, profile_warmup: speed checks, all off by default
                               (anneal/train_probe.py)
  init_adapter: []             start the LoRA from a saved adapter folder instead of fresh weights (the improvement
                               loop continues its best adapter). Same r, lora_alpha and target modules required; every
                               tensor is checked equal after loading. Ignored once a checkpoint exists (resume wins).

Resume: re-running the same command continues from the newest checkpoint in <run dir>/checkpoints with
the optimizer, scheduler, RNG and data position restored (Hugging Face Trainer). It refuses to resume if
train_file has changed since the run started (SHA-256 in <run dir>/data.sha256), because the data
position would then point at different rows.

Decay branches: a branch resumes from a milestone checkpoint at step S with the same warmup and a stable
phase that ends at S, then decays over D steps on the rows that come next in the data order. Its
adapter lands in <run dir>/branches/<checkpoint-S>/adapter. Trainer restores the scheduler's step
count from the checkpoint but takes the decay shape from this run's settings (LambdaLR does not save the
schedule function), which is what lets one checkpoint seed a differently shaped schedule.

EVO-X2 adaptation (19 Oct 2026; the cloud original is cloud_originals/train_longrun.py.cloud):
  - Portal run contract (anneal/runtime.py): the run folder is --run-dir (default results/<run_name>), with
    run.log, metrics.jsonl, status.json and summary.json; STOP there pauses with a checkpoint; --resume is accepted
    (resuming is automatic whenever a checkpoint exists).
  - Model setup of run smoke-27b-stock (tests/smoke_small.py): import unsloth first, UNSLOTH_BLOCK_COMPILE=0,
    stock GatedDeltaNet head pairing (anneal/gdn_heads.py), shift-and-add short conv (anneal/short_conv.py),
    FastLanguageModel 4-bit in bf16, Unsloth gradient checkpointing.
  - Labels come from anneal/data.py encode_chat (same prefix rule as tokenize_rows below, exact token IDs from the
    tokenizer's own template rendering), and the trainer's collated labels for a row are checked against them
    before training starts.
  - Qwen3.8's chat template renders reasoning only from message["reasoning_content"]; it never splits
    "<think>...</think>" out of the content. Rows from prepare_dataset.py / generate_traces.py keep the trace in
    the content, so it is moved to reasoning_content first (split_trace) and the row renders as the model writes
    at inference: "<think>\\nTRACE\\n</think>\\n\\nANSWER" after the generation prompt "<think>\\n". Rows without a
    trace use enable_thinking=False. Rows over seq_len are still dropped, never truncated; one row per sequence.
"""
import argparse
import hashlib
import json
import math
import os
import shutil
import sys
import time

import yaml

TRAINING_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, TRAINING_ROOT)
from anneal.runtime import RESULTS_DIR, Run, WindowsMemorySampler, gib, trainer_callback  # noqa: E402

os.environ.setdefault("UNSLOTH_BLOCK_COMPILE", "0")  # block compile fails at guard building under TRL (30+ min lost)
os.environ.setdefault("UNSLOTH_ENABLE_LOGGING", "1")


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


def split_trace(messages):
    """Assistant content "<think>\\nTRACE\\n</think>\\n\\nANSWER" -> reasoning_content TRACE, content ANSWER (stripped,
    as the template trims both). Returns (messages, has_trace)."""
    out, has_trace = [], False
    for m in messages:
        c = m.get("content")
        if m.get("role") == "assistant" and isinstance(c, str) and c.lstrip().startswith("<think>") and "</think>" in c:
            head, _, tail = c.partition("</think>")
            m = dict(m, reasoning_content=head.lstrip()[len("<think>"):].strip(), content=tail.strip())
            has_trace = has_trace or bool(m["reasoning_content"])
        out.append(m)
    return out, has_trace


def tokenize_rows(rows, tok, template_kwargs, seq_len, completion_only, stats=None):
    """input_ids + labels per row. Labels are -100 over the prompt when completion_only (anneal/data.py encode_chat:
    the longest common prefix of the full rendering and the prompt + generation prompt is masked). Rows over seq_len
    are dropped (never truncated) and counted. Rows with a trace render with template_kwargs (enable_thinking True),
    rows without one with enable_thinking=False."""
    from anneal.data import MaskStats, encode_chat

    stats = stats if stats is not None else MaskStats()
    out, too_long, misaligned = [], 0, 0
    no_trace = 0
    for ex in rows:
        msgs, has_trace = split_trace(ex["messages"])
        kw = dict(template_kwargs) if has_trace else dict(template_kwargs, enable_thinking=False)
        no_trace += not has_trace
        before = stats.prompt_not_prefix
        enc = encode_chat(tok, msgs, max_len=seq_len, loss="assistant" if completion_only else "full", stats=stats,
                          template_kwargs=kw)
        if enc is None:
            too_long += 1
            continue
        misaligned += stats.prompt_not_prefix - before
        out.append({"input_ids": enc["input_ids"], "labels": enc["labels"], "n_trained": enc["n_trained"],
                    "source": ex.get("source")})
    stats.rows_without_trace = no_trace
    return out, too_long, misaligned


def check_collated_mask(trainer, rows):
    """The labels the trainer will actually see for a row (after TRL's dataset prep and the collator) must be
    encode_chat's labels."""
    row = trainer.train_dataset[0]
    batch = trainer.data_collator([row])
    ids = batch["input_ids"][0].tolist()
    labels = batch["labels"][0].tolist()
    want = next((r for r in rows if r["input_ids"] == ids), None)
    ok = want is not None and labels == want["labels"]
    out = {"collator": type(trainer.data_collator).__name__, "row_found": want is not None,
           "labels_match_helper": ok, "batch_tokens": len(ids),
           "batch_trained": sum(1 for x in labels[1:] if x != -100)}
    print(f"collated batch check: {out}")
    return out


def install_mem_probe(run, model):
    """mem_debug: true. Log free device memory (HIP's view), torch's reserved/allocated and the sequence length right
    after the decoder's final norm (i.e. just before the loss) in every forward, and the Windows view of this
    process's GPU memory (the portal's counters)."""
    import torch

    from anneal.runtime import WindowsMemorySampler
    portal = WindowsMemorySampler.PORTAL_METRICS
    norm = None
    for name, mod in model.named_modules():
        if name.endswith("language_model.norm") or name.endswith("model.norm"):
            norm = (name, mod)
    if norm is None:
        print("mem probe: no final norm found")
        return

    def windows_view():
        try:
            import urllib.request
            with urllib.request.urlopen(portal, timeout=2) as r:
                m = json.loads(r.read()).get("latest") or {}
            other = (m.get("gpu_procs") or {}).get("other") or {}
            return {"win_dedicated_gib": round(other.get("ded", 0) / 2**30, 3),
                    "win_shared_gib": round(other.get("shr", 0) / 2**30, 3)}
        except Exception:
            return {}

    def hook(mod, inp, out):
        free, _ = torch.cuda.mem_get_info()
        run.metric(event="pre_loss_mem", tokens=int(out.shape[-2]) if hasattr(out, "shape") else None,
                   grad=torch.is_grad_enabled(), hip_free_gib=round(free / 2**30, 3),
                   torch_reserved_gib=round(torch.cuda.memory_reserved() / 2**30, 3),
                   torch_allocated_gib=round(torch.cuda.memory_allocated() / 2**30, 3), **windows_view())

    norm[1].register_forward_hook(hook)
    print(f"mem probe on {norm[0]}")


def warm_up(trainer, collator, dataset, index, run):
    """One training micro-step (Trainer.training_step: the same code path, autocast and loss as every real step) on
    the longest training row, called from on_train_begin; then the gradients are dropped (no optimizer step; dropout
    is 0, so nothing random is drawn). On this GPU part of the training process's memory outside torch's allocator
    is sized by the longest row seen so far and isn't given back when a longer row comes (dry runs 19 Oct: Windows'
    shared GPU memory 3.4 -> 4.3 -> 12.0 GiB as a 7,891-token row followed 6,019- and 2,569-token rows, but a flat
    8.3 GiB when the first row was the longest). Seeing the longest row first sizes it once."""
    import torch
    batch = collator([dataset[index]])
    n_items = int((batch["labels"][..., 1:] != -100).sum())
    model = getattr(trainer, "model_wrapped", None) or trainer.model
    t0 = time.time()
    loss = trainer.training_step(model, batch, torch.tensor(n_items, device="cuda"))
    loss = float(loss)
    trainer.model.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()
    free, _ = torch.cuda.mem_get_info()
    info = {"row": index, "tokens": int(batch["input_ids"].shape[-1]), "trained_tokens": n_items,
            "loss": round(loss, 5), "seconds": round(time.time() - t0, 1), "hip_free_gib": round(free / 2**30, 3),
            "torch_reserved_gib": round(torch.cuda.memory_reserved() / 2**30, 3)}
    print(f"warm-up on the longest row: {info}")
    run.metric(event="warm_up", **info)
    return info


class StepCounter:
    """Wraps the collator and records (tokens, trained tokens) of every micro-batch in the order they are built, so
    optimizer step N of this session covers micro-batches [(N - first) * accum, (N - first + 1) * accum)."""

    def __init__(self, collator, accum):
        self.collator, self.accum, self.batches, self.first_step = collator, accum, [], None

    def __call__(self, examples):
        out = self.collator(examples)
        labels = out["labels"]
        mask = out.get("attention_mask")
        tokens = int(mask.sum()) if mask is not None else int(labels.numel())
        self.batches.append((tokens, int((labels[..., 1:] != -100).sum())))
        return out

    def step_tokens(self, step):
        if self.first_step is None:
            return None
        i = (step - self.first_step) * self.accum
        part = self.batches[i:i + self.accum]
        if i < 0 or len(part) < self.accum:
            return None
        return sum(t for t, _ in part), sum(t for _, t in part)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--branch-from", default=None, help="milestone checkpoint to start a decay branch from")
    p.add_argument("--smoke-test", action="store_true", help="200 rows, 30 steps, 512-token rows, own folder")
    p.add_argument("--run-dir", default=None, help="run folder (the Training page sets it); default results/<run_name>")
    p.add_argument("--resume", action="store_true", help="accepted for the Training page; resuming is automatic")
    p.add_argument("--max-steps", type=int, default=None, help="override max_steps (dry runs)")
    p.add_argument("--stop-at-step", type=int, default=None, help="tests: create STOP at this step (pause + checkpoint)")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                   help="override one config key (YAML value), e.g. --set sdpa_probe=true; repeatable")
    args = p.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if args.smoke_test:
        cfg.update(max_samples=200, seq_len=min(int(cfg.get("seq_len", 1024)), 512), save_steps=10,
                   milestone_steps=[10], warmup_steps=3)
        cfg["run_name"] = cfg["run_name"] + "-smoke"
    if args.max_steps:
        cfg["max_steps"] = args.max_steps
    for item in args.set:
        key, sep, value = item.partition("=")
        if not sep or not key.strip():
            raise SystemExit(f"--set wants KEY=VALUE, got {item!r}")
        cfg[key.strip()] = yaml.safe_load(value)
    run_dir = args.run_dir or os.path.join(RESULTS_DIR, cfg["run_name"])
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    run = Run(run_dir)
    run.capture_output()
    run.write_status(state="starting", run_name=cfg["run_name"], script=os.path.abspath(__file__))
    print(f"== train_longrun.py {time.strftime('%Y-%m-%d %H:%M:%S')}  config {args.config}  args {vars(args)}")
    print(json.dumps(cfg, indent=1, default=str))
    stop_file = run.stop_path
    if os.path.exists(stop_file):
        os.remove(stop_file)  # starting again means the stop request has been handled
    try:
        return train(args, cfg, run, run_dir, ckpt_dir, stop_file)
    except BaseException as exc:
        import traceback
        traceback.print_exc()
        run.write_status(state="failed", error=f"{type(exc).__name__}: {exc}")
        run.metric(event="failed", error=f"{type(exc).__name__}: {exc}")
        return 1


def load_init_adapter(model, path, lora):
    """Copy a saved adapter's LoRA weights into the freshly created LoRA (shapes must match), then prove every tensor
    arrived. Returns a small report for summary.json."""
    import torch
    from peft import get_peft_model_state_dict, set_peft_model_state_dict
    from safetensors.torch import load_file

    with open(os.path.join(path, "adapter_config.json"), encoding="utf-8") as f:
        ac = json.load(f)
    want = {"r": lora["r"], "lora_alpha": lora["lora_alpha"], "target_modules": sorted(lora["target_modules"])}
    have = {"r": int(ac["r"]), "lora_alpha": int(ac["lora_alpha"]),
            "target_modules": sorted(ac["target_modules"]) if isinstance(ac["target_modules"], list) else ac["target_modules"]}
    if want != have:
        raise RuntimeError(f"init_adapter {path} has {have}, this run wants {want}")
    saved = load_file(os.path.join(path, "adapter_model.safetensors"))
    current = get_peft_model_state_dict(model)
    if set(saved) != set(current):
        missing, extra = sorted(set(current) - set(saved)), sorted(set(saved) - set(current))
        raise RuntimeError(f"init_adapter keys differ: {len(missing)} missing (e.g. {missing[:2]}), "
                           f"{len(extra)} unexpected (e.g. {extra[:2]})")
    res = set_peft_model_state_dict(model, saved)
    if getattr(res, "unexpected_keys", None):
        raise RuntimeError(f"init_adapter: unexpected keys {res.unexpected_keys[:3]}")
    after = get_peft_model_state_dict(model)
    bad = [k for k, v in saved.items() if not torch.equal(after[k].detach().cpu().to(v.dtype), v)]
    if bad:
        raise RuntimeError(f"init_adapter: {len(bad)} tensors differ after loading (e.g. {bad[:2]})")
    rep = {"path": path, "tensors": len(saved), "parameters": sum(v.numel() for v in saved.values())}
    print(f"init_adapter: loaded and verified {rep['tensors']} tensors ({rep['parameters']:,} parameters) from {path}")
    return rep


def train(args, cfg, run, run_dir, ckpt_dir, stop_file):
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
    from anneal import train_probe  # noqa: E402  (imports torch only when a switch below is on)
    blas_expect = train_probe.prefer_blas(cfg.get("blas_backend"))  # before anything imports torch
    # Before Unsloth and transformers, so a module that copies the function by name gets the wrapper too.
    sdpa = train_probe.install_sdpa(contiguous=bool(cfg.get("sdpa_contiguous", False)),
                                    probe=bool(cfg.get("sdpa_probe", False)))
    if cfg.get("ce_target_gb"):
        # Unsloth sizes its fused cross-entropy chunks from half the free device memory (read at import time).
        os.environ["UNSLOTH_CE_LOSS_TARGET_GB"] = str(cfg["ce_target_gb"])
    if engine == "unsloth":
        import unsloth  # noqa: F401,E402  (must come before transformers/trl/peft)
        from unsloth import FastLanguageModel, FastModel  # noqa: E402

        from anneal import gdn_heads, short_conv  # noqa: E402
        print(gdn_heads.apply(cfg["base_model"], cfg.get("gdn_heads", "stock")))
        print(short_conv.apply(cfg.get("short_conv", "shift")))
    import torch  # noqa: E402
    from datasets import Dataset, load_dataset  # noqa: E402
    from transformers import TrainerCallback, set_seed  # noqa: E402
    from trl import SFTConfig, SFTTrainer  # noqa: E402

    from anneal.data import MaskStats, print_sample  # noqa: E402

    blas = train_probe.blas_report(cfg.get("blas_backend"), blas_expect) if torch.cuda.is_available() else None
    set_seed(int(cfg.get("seed", 42)))  # before the adapter is created, so its initial weights are reproducible

    load_in_4bit = bool(cfg.get("load_in_4bit", True))
    load_in_8bit = bool(cfg.get("load_in_8bit", False))
    use_bf16 = bool(cfg.get("bf16", True))
    on_gpu = torch.cuda.is_available()
    lora = dict(r=int(cfg["lora_r"]), lora_alpha=int(cfg["lora_alpha"]),
                lora_dropout=float(cfg.get("lora_dropout", 0.0)), target_modules=cfg["target_modules"], bias="none")
    sampler = WindowsMemorySampler()
    sampler.start()
    run.write_status(state="loading")
    t_load = time.time()
    if engine == "unsloth":
        Loader = FastModel if cfg.get("loader", "FastLanguageModel") == "FastModel" else FastLanguageModel
        # As smoke-27b-stock: 4-bit NF4 weights, bf16 compute.
        model, tok = Loader.from_pretrained(model_name=cfg["base_model"], max_seq_length=int(cfg["seq_len"]),
                                            load_in_4bit=load_in_4bit, load_in_8bit=load_in_8bit,
                                            dtype=torch.bfloat16)
        short_conv.patch_loaded()  # Unsloth's compiled copy of the model code is imported during the load
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
    init_report = load_init_adapter(model, cfg["init_adapter"], lora) if cfg.get("init_adapter") else None
    if cfg.get("mem_debug"):
        install_mem_probe(run, model)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    vision = sum(p.numel() for n, p in model.named_parameters() if p.requires_grad
                 and ("visual" in n or "vision" in n))
    print(f"LoRA trainable parameters: {trainable:,} ({vision:,} in the vision tower); loaded in "
          f"{time.time() - t_load:.0f} s")
    if vision:
        raise RuntimeError(f"{vision:,} trainable parameters in the vision tower")
    if cfg.get("expect_trainable_params") and trainable != int(cfg["expect_trainable_params"]):
        raise RuntimeError(f"{trainable:,} trainable parameters, expected {int(cfg['expect_trainable_params']):,}")
    text_tok = getattr(tok, "tokenizer", tok)  # multimodal processors carry the text tokenizer inside

    raw = load_dataset("json", data_files={"train": cfg["train_file"]})["train"]
    if cfg.get("max_samples"):
        raw = raw.select(range(min(int(cfg["max_samples"]), len(raw))))  # a prefix keeps the mixer's shares
    mask_stats = MaskStats()
    rows, too_long, misaligned = tokenize_rows(raw, text_tok, cfg.get("chat_template_kwargs") or {},
                                               int(cfg["seq_len"]), bool(cfg.get("completion_only", True)), mask_stats)
    print(f"Tokenised {len(rows)} rows; dropped {too_long} over seq_len {cfg['seq_len']}"
          + (f"; {misaligned} rows where the prompt didn't render as a prefix (masked to the divergence)"
             if misaligned else "") + f"; mask stats {mask_stats.as_dict()}")
    if not rows:
        raise SystemExit("no training rows left")
    sample = print_sample(text_tok, rows[0], "row 0 (loss mask)")
    train_ds = Dataset.from_list([{"input_ids": r["input_ids"], "labels": r["labels"]} for r in rows])
    n_tokens = sum(len(r["input_ids"]) for r in rows)
    eval_ds = None
    eval_steps = int(cfg.get("eval_steps", 0))
    if eval_steps and cfg.get("val_file"):
        val_raw = load_dataset("json", data_files={"val": cfg["val_file"]})["val"]
        eval_ds = Dataset.from_list([{"input_ids": r["input_ids"], "labels": r["labels"]} for r in
                                     tokenize_rows(val_raw, text_tok, cfg.get("chat_template_kwargs") or {},
                                                   int(cfg["seq_len"]), bool(cfg.get("completion_only", True)))[0]])

    bs, accum = int(cfg.get("batch_size", 1)), int(cfg.get("grad_accum", 8))
    steps_per_epoch = math.ceil(len(train_ds) / (bs * accum))
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
        print(f"Warning: {total_steps} steps is more than one pass over {len(train_ds)} rows; rows will repeat.")

    if cfg.get("schedule", "wsd") == "wsd":
        sched = dict(lr_scheduler_type="warmup_stable_decay",
                     lr_scheduler_kwargs={"num_decay_steps": decay, "min_lr_ratio": float(cfg.get("min_lr_ratio", 0.0)),
                                          "decay_type": cfg.get("decay_type", "cosine")})
    else:
        sched = dict(lr_scheduler_type="cosine")

    run_t0 = time.time()
    # Rows vary in length (one row per sequence, up to seq_len). On this GPU (32 GiB dedicated, HIP also counts the
    # shared pool: 43.9 GiB) an 8k-token row needs ~38 GiB as Windows counts it, and blocks the caching allocator kept
    # from a shorter row before it leave no room: Unsloth's fused cross entropy then finds no free memory (dry run
    # 19 Oct: a 7,891-token row after a 6,019-token one). Releasing the cache between micro-batches gives every row
    # the clean pool the fixed-length 8k memory runs had.
    empty_cache = on_gpu and bool(cfg.get("empty_cache_between_microbatches", True))
    # The same pool also holds Unsloth's pinned host buffers for offloaded checkpoints: they grow with resize_ when a
    # longer row arrives and PyTorch's host allocator keeps the old pinned blocks, so variable-length rows pile up
    # pinned memory the GPU counts as its own. Pageable buffers (Unsloth's own switch; results unchanged, the copies
    # just stop overlapping compute) keep the offload out of it.
    if not cfg.get("pinned_offload", False):
        os.environ["UNSLOTH_DISABLE_PINNED_MEMORY"] = "1"
    print(f"allocator cache released between micro-batches: {empty_cache}; pinned offload buffers: "
          f"{os.environ.get('UNSLOTH_DISABLE_PINNED_MEMORY') != '1'}")

    cache_releases = [0]

    def release_cache():
        if empty_cache:
            torch.cuda.empty_cache()
            cache_releases[0] += 1

    class LongRun(TrainerCallback):
        """Milestone checkpoints kept for good, a STOP file for a clean pause, and a throughput log."""

        def __init__(self):
            self.last = None

        def on_train_begin(self, a, state, control, **kw):
            counter.first_step = state.global_step + 1
            if on_gpu and cfg.get("warm_up_longest_row", True) and not warm:
                longest = max(range(len(rows)), key=lambda i: len(rows[i]["input_ids"]))
                warm.update(warm_up(trainer, counter.collator, train_ds, longest, run))

        def on_step_end(self, a, state, control, **kw):
            release_cache()
            if on_gpu:
                free, total = torch.cuda.mem_get_info()
                run.metric(event="mem", step=state.global_step, hip_free_gib=gib(free), hip_total_gib=gib(total),
                           reserved_gib=gib(torch.cuda.memory_reserved()))
            if args.stop_at_step and state.global_step == args.stop_at_step and not os.path.exists(stop_file):
                with open(stop_file, "w", encoding="utf-8") as fh:
                    fh.write(f"--stop-at-step {args.stop_at_step}\n")
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

        def on_substep_end(self, a, state, control, **kw):
            release_cache()

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
        fp16=False,
        optim=cfg.get("optim", "adamw_8bit" if on_gpu else "adamw_torch"),
        weight_decay=float(cfg.get("weight_decay", 0.0)),
        gradient_checkpointing=engine == "hf" and bool(cfg.get("gradient_checkpointing", True)),
        max_length=int(cfg["seq_len"]),
        packing=False,
        dataset_num_proc=1,
        include_num_input_tokens_seen="non_padding",
        seed=int(cfg.get("seed", 42)),
        report_to="none",
    )
    counter = StepCounter(None, accum)
    profile = None
    if int(cfg.get("profile_steps", 0) or 0) > 0:
        profile = train_probe.Profile(run_dir, warmup=int(cfg.get("profile_warmup", 1)),
                                      active=int(cfg["profile_steps"]))

    class Profiling(TrainerCallback):
        """profile_steps: starts after the warm-up row (LongRun.on_train_begin runs first) and steps with the
        optimizer."""

        def on_train_begin(self, a, state, control, **kw):
            profile.start()

        def on_step_end(self, a, state, control, **kw):
            profile.step()

        def on_train_end(self, a, state, control, **kw):
            profile.stop()

    # LongRun first: with --stop-at-step it creates STOP before the run-contract callback looks for it.
    callbacks = [LongRun(), trainer_callback(run, step_tokens=counter.step_tokens)] + ([Profiling()] if profile else [])
    trainer = SFTTrainer(model=model, processing_class=text_tok, train_dataset=train_ds, eval_dataset=eval_ds,
                         args=targs, callbacks=callbacks)
    collated = check_collated_mask(trainer, rows)
    if not collated["labels_match_helper"]:
        raise RuntimeError("the trainer's labels are not encode_chat's loss mask (see collated batch check)")
    counter.collator = trainer.data_collator
    trainer.data_collator = counter
    warm = {}  # filled by LongRun.on_train_begin (warm_up)
    if on_gpu:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    print(f"{len(train_ds)} rows, {n_tokens:,} tokens, {total_steps} optimizer steps "
          f"({bs * accum} rows each), warmup {warmup}, decay {decay}."
          + (f" Resuming from {resume}." if resume else " Starting fresh."))
    run.metric(event="plan", rows=len(train_ds), tokens=n_tokens, total_steps=total_steps, warmup=warmup, decay=decay,
               resume=resume)
    result = trainer.train(resume_from_checkpoint=resume)
    sampler.stop.set()
    stopped_early = trainer.state.global_step < total_steps
    summary_dir = out_dir if args.branch_from else run_dir
    losses = [(h["step"], h["loss"]) for h in trainer.state.log_history if "loss" in h]
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
        "init_adapter": init_report,
        "blas": blas,
        "sdpa": sdpa.as_dict() if sdpa else None,
        "profile": profile.result if profile else None,
    }
    if not stopped_early:
        adapter_dir = os.path.join(summary_dir, "adapter")
        model.save_pretrained(adapter_dir)
        text_tok.save_pretrained(adapter_dir)
        summary["adapter"] = adapter_dir
        run.metric(event="adapter_saved", path=adapter_dir)
    with open(os.path.join(summary_dir, "train_summary.json"), "a", encoding="utf-8") as f:
        f.write(json.dumps(summary, default=str) + "\n")
    # The run contract's summary.json: everything above plus the data and mask checks.
    summary.update(
        trainable_params=trainable, rows=len(train_ds), rows_dropped_too_long=too_long, tokens=n_tokens,
        rows_prompt_not_prefix=misaligned, mask_stats=mask_stats.as_dict(), sample_row=sample,
        collated_check=collated, losses=losses, cache_releases_this_session=cache_releases[0], warm_up=warm,
        any_nan_loss=any(math.isnan(x) or math.isinf(x) for _, x in losses),
        windows_peak=dict(sampler.peak, portal_metrics=bool(sampler.portal_ok)),
        peak_allocated_gib=gib(torch.cuda.max_memory_allocated()) if on_gpu else None)
    if engine == "unsloth":
        summary["gdn_heads"] = gdn_heads.report()
        summary["short_conv"] = short_conv.report()
    with open(os.path.join(run_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1, default=str)
    print("Stopped early; re-run the same command to resume." if stopped_early
          else f"Done at step {trainer.state.global_step}. Adapter: {summary['adapter']}")
    if summary["any_nan_loss"]:
        raise RuntimeError("NaN/inf loss")
    return 0


if __name__ == "__main__":
    sys.exit(main())
