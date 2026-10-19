"""Answer-token loss on held-out chat rows for the base model and base + LoRA adapter, in one process.

Rows are encoded exactly as for training (scripts/train_longrun.py tokenize_rows: trace moved to reasoning_content,
anneal/data.py encode_chat labels), so "trained tokens" are the tokens training scores: the reasoning, the answer
and <|im_end|>. "answer" is the subset after </think> and before <|im_end|>. Base and adapter are scored on the same
4-bit model (smoke-27b-stock setup), the base with the adapter disabled. The LM head is applied in chunks of
positions, so a full 8,192 x 248k logit matrix is never held.

  val_loss.py --adapter <run>\\train\\adapter --val-file data\\processed\\mix_first_2k\\val.jsonl [--run-dir DIR]
"""

import argparse
import json
import math
import os
import sys
import time

TRAINING_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, TRAINING_ROOT)
sys.path.insert(0, os.path.join(TRAINING_ROOT, "scripts"))
from anneal.runtime import RESULTS_DIR, Run, gib  # noqa: E402

os.environ.setdefault("UNSLOTH_BLOCK_COMPILE", "0")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--val-file", required=True)
    ap.add_argument("--base-model", default=None, help="default: the training snapshot")
    ap.add_argument("--seq-len", type=int, default=8192)
    ap.add_argument("--max-rows", type=int, default=0)
    ap.add_argument("--chunk", type=int, default=1024, help="positions per LM-head chunk")
    ap.add_argument("--run-dir", default=os.path.join(RESULTS_DIR, "val-loss"))
    ap.add_argument("--resume", action="store_true", help="accepted for the Training page; starts over")
    args = ap.parse_args()
    run = Run(args.run_dir)
    run.capture_output()
    run.write_status(state="loading")
    try:
        return score(args, run)
    except Exception as exc:
        import traceback
        traceback.print_exc()
        run.write_status(state="failed", error=f"{type(exc).__name__}: {exc}")
        return 1


def score(args, run):
    import unsloth  # noqa: F401
    from unsloth import FastLanguageModel

    from anneal import gdn_heads, llama, short_conv
    base_model = args.base_model or llama.SNAPSHOT
    print(gdn_heads.apply(base_model, "stock"))
    print(short_conv.apply("shift"))
    import torch
    from peft import PeftModel

    from anneal.data import IGNORE_INDEX, MaskStats
    from train_longrun import tokenize_rows

    t0 = time.time()
    model, tokenizer = FastLanguageModel.from_pretrained(model_name=base_model, max_seq_length=args.seq_len,
                                                         load_in_4bit=True, dtype=torch.bfloat16)
    short_conv.patch_loaded()
    model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()
    tok = getattr(tokenizer, "tokenizer", tokenizer)
    print(f"loaded base + adapter in {time.time() - t0:.0f} s; {gib(torch.cuda.memory_allocated())} GiB")

    with open(args.val_file, encoding="utf-8") as fh:
        raw = [json.loads(line) for line in fh if line.strip()]
    if args.max_rows:
        raw = raw[:args.max_rows]
    stats = MaskStats()
    rows, too_long, _ = tokenize_rows(raw, tok, {"enable_thinking": True}, args.seq_len, True, stats)
    print(f"{len(rows)} rows ({too_long} over {args.seq_len} dropped); {stats.as_dict()}")

    # find the decoder (hidden states) and the LM head under the PEFT wrapper
    inner = model.get_base_model()
    decoder, head = inner.model, inner.lm_head
    think_end = tok.convert_tokens_to_ids("</think>")
    im_end = tok.convert_tokens_to_ids("<|im_end|>")

    def nll_of(ids):
        with torch.no_grad():
            h = decoder(input_ids=ids).last_hidden_state[0, :-1]
            out = []
            for s in range(0, h.shape[0], args.chunk):
                logits = head(h[s:s + args.chunk]).float()
                out.append(torch.nn.functional.cross_entropy(logits, ids[0, 1 + s:1 + s + logits.shape[0]],
                                                             reduction="none"))
            return torch.cat(out)

    per_row = []
    totals = {k: {"trained_nll": 0.0, "trained": 0, "answer_nll": 0.0, "answer": 0} for k in ("base", "adapter")}
    run.write_status(state="running", step=0, max_steps=len(rows))
    for k, r in enumerate(rows):
        ids = torch.tensor([r["input_ids"]], device="cuda")
        labels = torch.tensor(r["labels"][1:], device="cuda")
        mask = labels != IGNORE_INDEX
        tgt = r["input_ids"][1:]
        start = next((i for i, t in enumerate(tgt) if t == think_end and mask[i]), None)
        end = next((i for i in range(len(tgt) - 1, -1, -1) if tgt[i] == im_end), len(tgt))
        answer = torch.zeros_like(mask)
        if start is not None:
            answer[start + 1:end] = True
        else:
            answer[:end] = mask[:end]
        row = {"row": k, "source": r.get("source"), "tokens": len(r["input_ids"]), "trained": int(mask.sum()),
               "answer_tokens": int(answer.sum())}
        for name in ("base", "adapter"):
            if name == "base":
                with model.disable_adapter():
                    nll = nll_of(ids)
            else:
                nll = nll_of(ids)
            row[f"{name}_loss"] = float(nll[mask].mean())
            row[f"{name}_answer_loss"] = float(nll[answer].mean()) if answer.any() else None
            t = totals[name]
            t["trained_nll"] += float(nll[mask].sum())
            t["trained"] += int(mask.sum())
            t["answer_nll"] += float(nll[answer].sum())
            t["answer"] += int(answer.sum())
        per_row.append(row)
        print(f"row {k} ({row['source']}, {row['tokens']} tokens): base {row['base_loss']:.4f} / adapter "
              f"{row['adapter_loss']:.4f}; answer {row['base_answer_loss']} / {row['adapter_answer_loss']}")
        run.write_status(state="running", step=k + 1, max_steps=len(rows))
        run.metric(step=k + 1, **{x: row[x] for x in ("base_loss", "adapter_loss")})

    def summarise(subset):
        out = {}
        for name in ("base", "adapter"):
            tn = sum(r["trained"] for r in subset)
            an = sum(r["answer_tokens"] for r in subset)
            out[name] = {
                "trained_token_loss": sum(r[f"{name}_loss"] * r["trained"] for r in subset) / tn if tn else None,
                "answer_token_loss": sum((r[f"{name}_answer_loss"] or 0) * r["answer_tokens"] for r in subset) / an
                if an else None,
                "mean_row_loss": sum(r[f"{name}_loss"] for r in subset) / len(subset) if subset else None}
        out["rows"], out["trained_tokens"] = len(subset), sum(r["trained"] for r in subset)
        return out

    summary = {"adapter": args.adapter, "val_file": args.val_file, "rows": len(rows), "dropped_too_long": too_long,
               "all": summarise(per_row)}
    for src in sorted({r["source"] for r in per_row if r["source"]}):
        summary[f"source={src}"] = summarise([r for r in per_row if r["source"] == src])
    summary["per_row"] = per_row
    for key in [k for k in summary if k == "all" or k.startswith("source=")]:
        s = summary[key]
        print(f"{key}: trained-token loss base {s['base']['trained_token_loss']:.4f} -> adapter "
              f"{s['adapter']['trained_token_loss']:.4f}; answer-token loss {s['base']['answer_token_loss']:.4f} -> "
              f"{s['adapter']['answer_token_loss']:.4f} ({s['rows']} rows)")
    with open(os.path.join(run.dir, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=1)
    run.write_status(state="finished", step=len(rows), max_steps=len(rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
