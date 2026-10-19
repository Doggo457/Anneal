"""Rough QLoRA training-memory estimator.

This is back-of-envelope arithmetic, not a measurement. Use it to rule models in or out before
downloading 60GB, then calibrate the constants below against the first real run
(compare its output to torch.cuda.max_memory_reserved() in models/<run>/train_summary.json).

Usage:
  python scripts/memory_estimate.py --preset qwen3-32b
  python scripts/memory_estimate.py --preset llama-70b --seq-len 1024 --lora-r 8
  python scripts/memory_estimate.py --params-b 24 --layers 40 --hidden 5120 --vocab 131072
  python scripts/memory_estimate.py --all
"""
import argparse

# Presets: total params (billions), layers, hidden size, vocab, attention heads.
# Values from each model's config.json as of mid-2026; re-check before relying on them.
#
# dense_fraction (MoE only): share of parameters OUTSIDE the expert stack (attention, embeddings,
# norms). Matters because, as of September 2026, bitsandbytes only quantises nn.Linear modules and
# skips the fused 3-D expert tensors that transformers v5 uses. So load_in_4bit / load_in_8bit on a
# MoE model leaves the experts in bf16. The report prints both scenarios for MoE presets.
PRESETS = {
    "gpt-oss-20b": dict(params_b=21, layers=24, hidden=2880, vocab=201088, heads=64, moe=True, dense_fraction=0.08),
    "qwen3-30b-a3b": dict(params_b=30.5, layers=48, hidden=2048, vocab=151936, heads=32, moe=True, dense_fraction=0.04),
    # Qwen 27B dense line (Apache-2.0): 3.5 (Feb 2026), 3.6 (Apr 2026), 3.8 (Aug 2026) share this shape.
    # THE project model is qwen3.8-27b. Other presets are here for the "why not X" arithmetic only.
    "qwen3.8-27b": dict(params_b=27.0, layers=64, hidden=5120, vocab=248320, heads=24, moe=False),
    "qwen3.6-27b": dict(params_b=27.0, layers=64, hidden=5120, vocab=248320, heads=24, moe=False),
    "qwen3-32b": dict(params_b=32.8, layers=64, hidden=5120, vocab=151936, heads=64, moe=False),
    "qwen2.5-coder-32b": dict(params_b=32.5, layers=64, hidden=5120, vocab=152064, heads=40, moe=False),
    "devstral-24b": dict(params_b=24, layers=40, hidden=5120, vocab=131072, heads=32, moe=False),
    "llama-70b": dict(params_b=70.6, layers=80, hidden=8192, vocab=128256, heads=64, moe=False),
    "qwen2.5-72b": dict(params_b=72.7, layers=80, hidden=8192, vocab=152064, heads=64, moe=False),
    "gpt-oss-120b": dict(params_b=117, layers=36, hidden=2880, vocab=201088, heads=64, moe=True, dense_fraction=0.03),
}

# Windows: the dedicated pool tops out at 48GB on this unit and it is unverified whether training can
# use the OS's "shared" memory beyond it. Linux: GTT hands the GPU 52-56GB with no penalty. Both are
# driver policy, not hardware: the 64GB is one physical pool. See docs/Hardware-Notes.md.
BUDGETS = {"Windows dedicated (48GB)": 48.0, "Linux GTT (~56GB)": 56.0}


def estimate(params_b, layers, hidden, vocab, heads, bits=4, seq_len=1024, batch=1, lora_r=16, moe=False,
             dense_fraction=1.0, experts_bf16=False):
    gib = 2**30
    # Frozen base weights. 4-bit NF4 plus quantisation constants and the parts kept in higher
    # precision (embeddings, norms) come out around 10% over the raw bit count.
    if moe and experts_bf16 and bits < 16:
        # Today's bitsandbytes: only the non-expert part gets quantised; experts stay bf16.
        expert_params = params_b * 1e9 * (1 - dense_fraction)
        dense_params = params_b * 1e9 * dense_fraction
        weights = expert_params * 2 + dense_params * bits / 8 * 1.10
    else:
        weights = params_b * 1e9 * bits / 8 * 1.10

    # LoRA adapters on all 7 linear projections. Per layer, in+out dims sum to roughly
    # 8*hidden (attention) + 3*(hidden + 3.5*hidden) (MLP) = ~21.5*hidden. MoE models put
    # adapters on the shared/attention parts only in most recipes, so scale down.
    lora_params = layers * lora_r * 21.5 * hidden * (0.4 if moe else 1.0)
    adapter = lora_params * 2          # bf16 weights
    adapter_grads = lora_params * 2
    optimizer = lora_params * 8        # AdamW fp32 m and v (adamw_8bit roughly quarters this)

    # Activations with gradient checkpointing: one saved activation per layer boundary plus one
    # layer's worth of recompute. ~4 * hidden * 2 bytes per token per layer is a workable rule of thumb.
    activations = layers * batch * seq_len * hidden * 2 * 4
    # Attention scores for the recomputed layer (fp32 softmax on seq^2).
    attention = batch * heads * seq_len * seq_len * 4
    # Output logits in fp32 for the loss.
    logits = batch * seq_len * vocab * 4
    # OS, driver, allocator fragmentation, CUDA/HIP context.
    overhead = 5 * gib

    parts = {
        "base weights": weights,
        "lora adapter + grads": adapter + adapter_grads,
        "optimizer state": optimizer,
        "activations": activations + attention,
        "logits": logits,
        "overhead": overhead,
    }
    total = sum(parts.values())
    return {k: v / gib for k, v in parts.items()}, total / gib


def verdicts(total):
    for label, budget in BUDGETS.items():
        margin = budget - total
        # Real runs tend to land 10-20% above estimates, so under ~8 GiB of margin is "tight", not "fits".
        verdict = "fits" if margin > 8 else ("tight" if margin > 0 else "does NOT fit")
        print(f"  vs {label:24s} {verdict:12s} (margin {margin:+.1f} GiB)")


def report(name, cfg, args):
    moe = cfg.get("moe", False)
    common = dict(bits=args.bits, seq_len=args.seq_len, batch=args.batch, lora_r=args.lora_r, moe=moe,
                  dense_fraction=cfg.get("dense_fraction", 1.0))
    parts, total = estimate(cfg["params_b"], cfg["layers"], cfg["hidden"], cfg["vocab"], cfg["heads"], **common)
    print(f"\n{name}: {cfg['params_b']}B params, {cfg['layers']} layers, hidden {cfg['hidden']}"
          f"{' (MoE)' if moe else ''} | {args.bits}-bit, seq {args.seq_len}, batch {args.batch}, r={args.lora_r}")
    for k, v in parts.items():
        print(f"  {k:24s} {v:6.1f} GiB")
    print(f"  {'TOTAL (estimate)':24s} {total:6.1f} GiB")
    verdicts(total)
    if moe and args.bits < 16:
        _, total_bf16 = estimate(cfg["params_b"], cfg["layers"], cfg["hidden"], cfg["vocab"], cfg["heads"],
                                 experts_bf16=True, **common)
        print(f"  -- MoE caveat: with today's bitsandbytes the fused experts are NOT quantised and stay in bf16 --")
        print(f"  {'TOTAL, experts in bf16':24s} {total_bf16:6.1f} GiB   <- what you should expect in practice")
        verdicts(total_bf16)
    # Inference-only footprint for comparison (llama.cpp GGUF does quantise experts, so this one is real).
    inf = cfg["params_b"] * 1e9 * args.bits / 8 * 1.10 / 2**30
    print(f"  inference-only weights   {inf:6.1f} GiB  (GGUF at this bit width, plus KV cache)")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--preset", choices=sorted(PRESETS))
    p.add_argument("--all", action="store_true", help="run every preset")
    p.add_argument("--params-b", type=float)
    p.add_argument("--layers", type=int)
    p.add_argument("--hidden", type=int)
    p.add_argument("--vocab", type=int, default=150000)
    p.add_argument("--heads", type=int, default=64)
    p.add_argument("--moe", action="store_true")
    p.add_argument("--bits", type=int, default=4)
    p.add_argument("--seq-len", type=int, default=1024)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--lora-r", type=int, default=16)
    args = p.parse_args()

    if args.all:
        for name, cfg in PRESETS.items():
            report(name, cfg, args)
    elif args.preset:
        report(args.preset, PRESETS[args.preset], args)
    elif args.params_b and args.layers and args.hidden:
        report("custom", dict(params_b=args.params_b, layers=args.layers, hidden=args.hidden,
                              vocab=args.vocab, heads=args.heads, moe=args.moe), args)
    else:
        p.error("give --preset, --all, or --params-b/--layers/--hidden")

    print("\nThese are estimates. Calibrate against the first real run and adjust the constants in this file.")


if __name__ == "__main__":
    main()
