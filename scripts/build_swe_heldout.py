"""Build the held-out bug-fix steps for the loop's loss rule (CPU only).

  python scripts/build_swe_heldout.py [--rows 200]

The loop's openswe_v4flash pool (experiments/loop.yaml) leaves out every repository whose name hashes to bucket 0
of 20 (scripts/public_source.py --holdout-col repo --holdout-mod 20). This script takes the other side: the same
filters as the pool (scripts/loop.py public_source_args: DeepSeek V4-Flash-0731 runs that resolved their issue,
outside Python, each step with the teacher's reasoning; at most 8,192 tokens, decontaminated against EvalPlus and the
gate), but only repositories in bucket 0, one step per run (evenly spaced over the session), then --rows of them by
a seeded draw. No round ever trains on these repositories, so a fall in loss on them is learning that carries over
to unseen code bases.

Writes data/eval/swe_heldout.jsonl and data/eval/swe_heldout_card.json. Like the gate, it must not change once
rounds are scored on it; --force only before the first training round.
"""

import argparse
import json
import os
import random
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from anneal import evalsets as E  # noqa: E402

OUT = os.path.join(E.EVAL_DIR, "swe_heldout.jsonl")
CARD = os.path.join(E.EVAL_DIR, "swe_heldout_card.json")
WORK = os.path.join(ROOT, "data", "eval", "swe_heldout_src")


def source_args(cfg):
    """The openswe_v4flash source of experiments/loop.yaml as public_source.py arguments for its held-out side: every
    filter the pool uses (scripts/loop.py public_source_args), only the repositories the pool leaves out, one step
    per run, rows up to 8,192 tokens."""
    import loop as L
    src = next(s for s in cfg["sources"] if s["name"] == "openswe_v4flash")
    if not src.get("holdout"):
        raise SystemExit("experiments/loop.yaml: openswe_v4flash has no holdout {col, mod}")
    return L.public_source_args(src, side="heldout") + ["--max-steps-per-conversation", "1", "--max-tokens", "8192"], src


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--rows", type=int, default=200)
    ap.add_argument("--config", default=os.path.join(ROOT, "experiments", "loop.yaml"))
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    if os.path.exists(OUT) and not args.force:
        raise SystemExit(f"{OUT} exists; rounds compare against it, so it must not change (--force only before the "
                         "first training round)")
    import yaml
    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    ps_args, src = source_args(cfg)
    from anneal import llama
    cmd = [sys.executable, "-u", os.path.join(ROOT, "scripts", "public_source.py")] + ps_args + [
        "--tokenizer", llama.SNAPSHOT, "--decontaminate", "--decontaminate-gate", E.DECONTAM_EXTRA, "--out", WORK]
    print(" ".join(cmd), flush=True)
    rc = subprocess.run(cmd, cwd=ROOT).returncode
    if rc != 0:
        raise SystemExit(f"public_source.py exited with {rc}")
    rows = E.read_jsonl(os.path.join(WORK, "train.jsonl")) + E.read_jsonl(os.path.join(WORK, "val.jsonl"))
    random.Random(42).shuffle(rows)
    pick = rows[:args.rows]
    E.write_jsonl(OUT, pick)
    measure = json.load(open(os.path.join(WORK, "measure.json"), encoding="utf-8"))
    card = {"built": time.strftime("%Y-%m-%d %H:%M:%S"), "source": src["name"], "dataset": src["dataset"],
            "revision": measure.get("revision"), "holdout": src["holdout"], "available_steps": len(rows),
            "rows": len(pick), "counts": measure.get("counts"), "report_fields": measure.get("report_fields"),
            "kept_tokens": measure.get("kept_tokens")}
    with open(CARD, "w", encoding="utf-8") as fh:
        json.dump(card, fh, indent=1)
    print(f"wrote {OUT}: {len(pick)} of {len(rows)} held-out steps", flush=True)


if __name__ == "__main__":
    main()
