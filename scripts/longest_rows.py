"""Pick the longest training rows of a JSONL file, measured as the trainer renders them.

For memory tests of scripts/train_longrun.py at a new row-length cap: the test should train on real rows close to the
cap, not on short ones that happen to fit. Lengths are counted as scripts/public_source.py and the trainer count them
(train_longrun.split_trace moves each reasoning trace into reasoning_content, Qwen3.8's template renders it, thinking
on exactly when the row has a trace). Rows longer than --max-tokens are skipped, never cut.

Usage:
  longest_rows.py --in data/processed/memtest/glm16k/train.jsonl --tokenizer MODEL_SNAPSHOT --max-tokens 16384
                  --min-tokens 14000 --n 12 --out data/processed/memtest/glm-longest-16k.jsonl
"""
import argparse
import heapq
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--in", dest="inp", required=True, help="JSONL rows with a messages field")
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--max-tokens", type=int, required=True)
    ap.add_argument("--min-tokens", type=int, default=0)
    ap.add_argument("--n", type=int, required=True, help="how many of the longest rows to keep")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from transformers import AutoTokenizer

    from anneal.data import template_ids
    from train_longrun import split_trace

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    tok = getattr(tok, "tokenizer", tok)
    heap, seen, over, under = [], 0, 0, 0
    with open(args.inp, encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if not line.strip():
                continue
            row = json.loads(line)
            split, thinking = split_trace(row["messages"])
            n = len(template_ids(tok, split, add_generation_prompt=False, enable_thinking=thinking))
            seen += 1
            if n > args.max_tokens:
                over += 1
                continue
            if n < args.min_tokens:
                under += 1
                continue
            item = (n, -i, line if line.endswith("\n") else line + "\n")
            if len(heap) < args.n:
                heapq.heappush(heap, item)
            else:
                heapq.heappushpop(heap, item)
    picked = sorted(heap, reverse=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.writelines(item[2] for item in picked)
    lens = [item[0] for item in picked]
    print(json.dumps({"read": seen, "over_max": over, "under_min": under,
                      "between": seen - over - under, "kept": len(picked), "kept_tokens": lens,
                      "out": args.out}))
    return 0 if picked else 1


if __name__ == "__main__":
    sys.exit(main())
