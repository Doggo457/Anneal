"""Load the base model plus a saved LoRA adapter from disk and continue short coding prompts greedily, to confirm the trained model still writes sensible text (no loops).

The base is loaded in 4-bit with Unsloth exactly as for training (stock GatedDeltaNet head pairing), the adapter
with PEFT from its folder (smoke_small.py --save-adapter writes <run folder>\\adapter). Each prompt is a user turn
in the model's chat template with the generation prompt (system prompt + "<|im_start|>assistant\\n<think>\\n"; with
--no-thinking the template's non-thinking form, no system prompt and "<think>\\n\\n</think>\\n\\n"), continued
for --new-tokens tokens greedily, with the adapter and, for comparison, with it switched off. Repetition is
measured on the generated tokens: the share of distinct 4-grams and the most repeats of any 8-gram; a looping
continuation has few distinct 4-grams and an 8-gram repeated 3 or more times.

Usage (the Training page adds --run-dir):
  adapter_generate.py --adapter C:\\AI\\training\\results\\smoke-27b-stock\\adapter [--new-tokens 64]
"""

import argparse
import json
import os
import sys
import time
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # C:\AI\training
from anneal.runtime import RESULTS_DIR, Run, gib  # noqa: E402

PROMPTS = [
    ("palindrome", "Write a Python function that checks whether a string is a palindrome, ignoring case and "
                   "non-alphanumeric characters."),
    ("sql", "Write a SQL query that returns the five customers with the highest total order value from tables "
            "customers(id, name) and orders(id, customer_id, amount)."),
]


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", default="unsloth/Qwen3.8-27B-unsloth-bnb-4bit")
    ap.add_argument("--adapter", required=True, help="folder with adapter_config.json + adapter_model.safetensors")
    ap.add_argument("--new-tokens", type=int, default=64)
    ap.add_argument("--no-base", action="store_true", help="skip the comparison with the adapter switched off")
    ap.add_argument("--no-thinking", action="store_true",
                    help="render the prompts with the template's enable_thinking=False (generation prompt "
                         "'<think>\\n\\n</think>\\n\\n', no system prompt): the context the trained answers follow")
    ap.add_argument("--gdn-heads", choices=["stock", "auto", "interleave", "tile"], default="stock",
                    help="GatedDeltaNet value-head pairing (anneal/gdn_heads.py); stock is correct")
    ap.add_argument("--tag", default="adapter_generate")
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--resume", action="store_true", help="accepted for the Training page; starts over")
    return ap.parse_args()


def repetition(tokens):
    def grams(n):
        return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
    g4, g8 = grams(4), grams(8)
    top8 = Counter(g8).most_common(1)
    return {"distinct_4gram_share": round(len(set(g4)) / len(g4), 3) if g4 else None,
            "max_8gram_repeats": top8[0][1] if top8 else 0,
            "loops": bool(top8 and top8[0][1] >= 3)}


def main():
    args = parse_args()
    run = Run(args.run_dir or os.path.join(RESULTS_DIR, args.tag))
    run.capture_output()
    run.write_status(state="starting", model=args.model, script=os.path.abspath(__file__))
    print(f"== adapter_generate.py {time.strftime('%Y-%m-%d %H:%M:%S')}  args: {vars(args)}")
    try:
        return generate(args, run)
    except Exception as exc:
        import traceback
        traceback.print_exc()
        run.write_status(state="failed", error=f"{type(exc).__name__}: {exc}")
        return 1


def generate(args, run):
    import unsloth  # noqa: F401
    from unsloth import FastLanguageModel

    import torch
    from peft import PeftModel

    from anneal import gdn_heads
    print(gdn_heads.apply(args.model, args.gdn_heads))
    run.write_status(state="loading")
    t0 = time.time()
    model, tokenizer = FastLanguageModel.from_pretrained(model_name=args.model, max_seq_length=4096,
                                                         load_in_4bit=True, dtype=torch.bfloat16)
    tok = getattr(tokenizer, "tokenizer", tokenizer)
    model = PeftModel.from_pretrained(model, args.adapter)
    FastLanguageModel.for_inference(model)
    model.eval()
    with open(os.path.join(args.adapter, "adapter_config.json"), encoding="utf-8") as fh:
        acfg = json.load(fh)
    lora_params = sum(p.numel() for n, p in model.named_parameters() if "lora_" in n)
    out = {"args": vars(args), "load_seconds": round(time.time() - t0, 1),
           "allocated_gib": gib(torch.cuda.memory_allocated()), "lora_params_loaded": lora_params,
           "adapter_config": {k: acfg.get(k) for k in ("base_model_name_or_path", "r", "lora_alpha", "target_modules")},
           "prompts": []}
    print(f"loaded base + adapter in {out['load_seconds']} s, {out['allocated_gib']} GiB, "
          f"{lora_params:,} LoRA parameters; adapter config {out['adapter_config']}")
    run.write_status(state="running", step=0, max_steps=len(PROMPTS))

    def greedy(ids):
        x = torch.tensor([ids], device="cuda")
        t1 = time.time()
        with torch.no_grad():
            g = model.generate(input_ids=x, attention_mask=torch.ones_like(x), max_new_tokens=args.new_tokens,
                               do_sample=False, temperature=None, top_p=None, top_k=None)
        new = g[0, x.shape[1]:].tolist()
        dt = time.time() - t1
        return {"tokens": new, "text": tok.decode(new), "seconds": round(dt, 1),
                "tok_per_s": round(len(new) / dt, 2), "repetition": repetition(new)}

    for i, (name, text) in enumerate(PROMPTS):
        kw = {"enable_thinking": False} if args.no_thinking else {}
        ids = tok.apply_chat_template([{"role": "user", "content": text}], tokenize=True, add_generation_prompt=True,
                                      **kw)
        ids = list(ids["input_ids"] if hasattr(ids, "keys") else ids)
        row = {"name": name, "prompt": text, "prompt_tokens": len(ids), "adapter": greedy(ids)}
        print(f"\n[{name}] with adapter ({row['adapter']['seconds']} s, {row['adapter']['repetition']}):\n"
              f"{row['adapter']['text']}")
        if not args.no_base:
            with model.disable_adapter():
                row["base"] = greedy(ids)
            a, b = row["adapter"]["tokens"], row["base"]["tokens"]
            row["first_divergence_from_base"] = next((j for j, (x, y) in enumerate(zip(a, b)) if x != y),
                                                     None if a == b else min(len(a), len(b)))
            print(f"[{name}] adapter off ({row['base']['repetition']}):\n{row['base']['text']}")
            print(f"[{name}] first token where adapter and base differ: {row['first_divergence_from_base']}")
        out["prompts"].append(row)
        run.metric(step=i + 1, prompt=name, loops=row["adapter"]["repetition"]["loops"],
                   distinct_4gram_share=row["adapter"]["repetition"]["distinct_4gram_share"])
        run.write_status(state="running", step=i + 1)
    out["any_loops_with_adapter"] = any(p["adapter"]["repetition"]["loops"] for p in out["prompts"])
    out["gdn_heads"] = gdn_heads.report()
    with open(os.path.join(run.dir, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, default=str)
    run.write_status(state="finished")
    print("RESULT:", "LOOPS" if out["any_loops_with_adapter"] else "OK (no loops)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
