"""Reproduce Unsloth's per-layer (block) compile on Qwen3.5-0.8B with tracebacks, and time a train step.

Usage: python block_compile_debug.py [--tokens 1024] [--steps 4]
"""

import argparse
import os
import time

os.environ.setdefault("UNSLOTH_BLOCK_COMPILE_DEBUG", "1")

import unsloth  # noqa: F401,E402
from unsloth import FastLanguageModel  # noqa: E402

import torch  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--mask", action="store_true", help="pass an attention_mask, as the TRL trainer does")
    args = ap.parse_args()

    # Print the guard source that fails while dynamo builds guards (the compile error hides it).
    import torch._dynamo.guards as dg
    _orig_get = dg.GuardBuilder.get

    def _get(self, guard, *a, **k):
        try:
            return _orig_get(self, guard, *a, **k)
        except Exception as exc:
            print(f"[guard-debug] failed guard source: {getattr(guard, 'name', guard)!r}: {exc}")
            raise
    dg.GuardBuilder.get = _get
    model, tok = FastLanguageModel.from_pretrained("Qwen/Qwen3.5-0.8B", max_seq_length=args.tokens,
                                                   load_in_4bit=True, dtype=torch.bfloat16)
    model = FastLanguageModel.get_peft_model(
        model, r=16, lora_alpha=16, lora_dropout=0, bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing="unsloth", random_state=3407)
    model.train()
    ids = torch.randint(1000, 50000, (1, args.tokens), device="cuda")
    for step in range(args.steps):
        torch.cuda.synchronize()
        t0 = time.time()
        extra = {"attention_mask": torch.ones_like(ids)} if args.mask else {}
        out = model(input_ids=ids, labels=ids, **extra)
        out.loss.backward()
        torch.cuda.synchronize()
        print(f"step {step}: loss {out.loss.item():.4f}  {time.time() - t0:.2f} s")
        model.zero_grad(set_to_none=True)
    try:
        from unsloth_zoo.temporary_patches.gdn_causal_conv1d import GDN_CAUSAL_CONV1D_STATS
        print("GDN_CAUSAL_CONV1D_STATS:", GDN_CAUSAL_CONV1D_STATS)
    except Exception as exc:
        print("no conv stats:", exc)
    modes = {}
    for name, m in model.named_modules():
        mode = m.__dict__.get("_unsloth_block_mode")
        if mode is not None:
            modes[str(mode)] = modes.get(str(mode), 0) + 1
    print("block modes per layer:", modes)


if __name__ == "__main__":
    main()
