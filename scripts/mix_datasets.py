"""Combine processed sources into one training mix with fixed proportions in every prefix.

Each source is a folder written by prepare_dataset.py (train.jsonl, val.jsonl, data_card.md). Rows are
taken in each source's own shuffled order and interleaved so that ANY prefix of the output keeps the
target proportions (to within one row per source). That matters for a long run: a learning-rate decay
branch started at row 8,000 or 20,000 has then trained on the same mix as the full run, so the
checkpoints form a clean data-scaling curve.

Usage:
  python scripts/mix_datasets.py \
      --source self_oci=data/processed/self_oci:12000 \
      --source self_cc=data/processed/self_cc:6000 \
      --source ocr=data/processed/ocr:20000 \
      --source general=data/processed/general:2000 \
      --tokenizer Qwen/Qwen3.8-27B --out data/processed/mix_main
A count of "all" takes every training row of that source. If a source has fewer rows than asked, all
are used and the card says so.

Output: <out>/train.jsonl, val.jsonl, data_card.md. Every row keeps "messages" and gains "source".
Validation rows whose question also appears in train (from any source) are dropped. The card records
the SHA-256 of train.jsonl; the long-run trainer refuses to resume if the file has changed.
"""
import argparse
import collections
import hashlib
import json
import os
import random
import sys


def read_jsonl(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def question(row):
    return row["messages"][-2]["content"].strip().lower()


def interleave(pools, seed):
    """Largest-deficit interleave: after i rows, source s has taken round(i * n_s / N) rows, +-1."""
    total = sum(len(p) for p in pools.values())
    taken = {s: 0 for s in pools}
    rng = random.Random(seed)
    order = list(pools)
    rng.shuffle(order)  # tie-break order, fixed by the seed
    out = []
    for i in range(1, total + 1):
        best = max(order, key=lambda s: (i * len(pools[s]) / total - taken[s]) if taken[s] < len(pools[s]) else -1e18)
        out.append(pools[best][taken[best]])
        taken[best] += 1
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", action="append", required=True, help="name=folder:count (count may be 'all')")
    p.add_argument("--val-per-source", type=int, default=200, help="validation rows taken from each source")
    p.add_argument("--tokenizer", default=None, help="count tokens through the chat template (e.g. Qwen/Qwen3.8-27B)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    rng = random.Random(args.seed)
    pools, vals, card_rows = {}, [], []
    for spec in args.source:
        name, _, rest = spec.partition("=")
        folder, _, count = rest.rpartition(":")
        if not name or not folder or not count:
            raise SystemExit(f"--source must be name=folder:count, got {spec!r}")
        rows = read_jsonl(os.path.join(folder, "train.jsonl"))
        if not rows:
            raise SystemExit(f"{folder}/train.jsonl is missing or empty")
        want = len(rows) if count == "all" else int(count)
        use = rows[:want]  # prepare_dataset.py already shuffled with its own seed
        for r in use:
            r["source"] = name
        pools[name] = use
        v = read_jsonl(os.path.join(folder, "val.jsonl"))[: args.val_per_source]
        for r in v:
            r["source"] = name
        vals += v
        card_rows.append((name, folder, want, len(rows), len(use)))
        if len(use) < want:
            print(f"  note: {name} has {len(rows)} rows, asked for {want}; using all")

    train = interleave(pools, args.seed)
    train_q = {question(r) for r in train}
    val = [r for r in vals if question(r) not in train_q]
    dropped_val = len(vals) - len(val)
    rng.shuffle(val)

    by_source_q = collections.defaultdict(set)
    for r in train:
        by_source_q[question(r)].add(r["source"])
    shared = sum(1 for s in by_source_q.values() if len(s) > 1)

    tok_stats = {}
    if args.tokenizer:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from prepare_dataset import TokenCounter

        count_tokens = TokenCounter(args.tokenizer)
        sums = collections.Counter()
        for r in train:
            sums[r["source"]] += count_tokens(r["messages"])
        tok_stats = dict(sums)

    os.makedirs(args.out, exist_ok=True)
    digest = hashlib.sha256()
    with open(os.path.join(args.out, "train.jsonl"), "w", encoding="utf-8") as f:
        for r in train:
            line = json.dumps(r, ensure_ascii=False) + "\n"
            digest.update(line.encode("utf-8"))
            f.write(line)
    with open(os.path.join(args.out, "val.jsonl"), "w", encoding="utf-8") as f:
        for r in val:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    n = len(train)
    with open(os.path.join(args.out, "data_card.md"), "w", encoding="utf-8") as f:
        f.write("# Mixed training set\n\n")
        f.write("| Source | Folder | Asked | Available | Used | Share | Tokens | Mean tokens/row |\n|---|---|---|---|---|---|---|---|\n")
        for name, folder, want, avail, used in card_rows:
            t = tok_stats.get(name)
            f.write(f"| {name} | `{folder}` | {want} | {avail} | {used} | {100 * used / n:.1f}% | "
                    f"{t if t is not None else '-'} | {round(t / used) if t else '-'} |\n")
        f.write(f"\nTrain rows: {n}. Val rows: {len(val)} ({dropped_val} dropped because the question is in train).\n")
        f.write(f"Questions appearing in more than one source: {shared}.\n")
        f.write(f"Order: interleaved so every prefix keeps the shares above (seed {args.seed}).\n")
        f.write(f"train.jsonl SHA-256: `{digest.hexdigest()}`\n")
        if tok_stats:
            total = sum(tok_stats.values())
            f.write(f"\nTotal training tokens: {total:,}. One pass takes "
                    + ", ".join(f"{total / r / 86400:.1f} days at {r} tok/s" for r in (25, 40, 100)) + ".\n")
        f.write("\nLicences and filters: see each source folder's data_card.md.\n")
    print(f"  wrote {n} train rows, {len(val)} val rows -> {args.out} (sha256 {digest.hexdigest()[:12]})")


if __name__ == "__main__":
    main()
