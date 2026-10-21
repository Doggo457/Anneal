"""Build the held-out bug-fix steps (and, with --set terminal, the held-out terminal steps) for the loop's loss rule
(CPU only).

  python scripts/build_swe_heldout.py [--set swe|terminal] [--rows 200]

The loop's openswe_v4flash pool (experiments/loop.yaml) leaves out every repository whose name hashes to bucket 0
of 20 (scripts/public_source.py --holdout-col repo --holdout-mod 20). This script takes the other side: the same
filters as the pool (scripts/loop.py public_source_args: DeepSeek V4-Flash-0731 runs that resolved their issue,
outside Python, each step with the teacher's reasoning; at most 8,192 tokens, decontaminated against EvalPlus and the
gate), but only repositories in bucket 0, one step per run (evenly spaced over the session), then --rows of them by
a seeded draw. No round ever trains on these repositories, so a fall in loss on them is learning that carries over
to unseen code bases.

Writes data/eval/swe_heldout.jsonl and data/eval/swe_heldout_card.json. Like the gate, it must not change once
rounds are scored on it; --force only before the first training round.

--set terminal (21 Oct 2026; the six-language gate can't see terminal skill): the GLM-5.3 terminal runs. The
glm53_all pools leave out every task group whose id hashes to bucket 0 of their holdout; this takes that side of the
non-Python pool, with all of its filters (test-passed runs, the teacher's reasoning, parsed replies only, near-duplicate
steps dropped, decontaminated), read best-ranked run first, at most 3 runs per task (the best-ranked) so no task
dominates, one step per run (its middle step), rows up to 12,288 tokens. On 21 Oct 2026 bucket 0 of 20 held 51 task
groups (94 tasks, 300 non-Python runs). Writes data/eval/term_heldout.jsonl and term_heldout_card.json.
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

SETS = {
    # name prefix of the loop.yaml source, extra public_source.py arguments, file stem
    "swe": ("openswe_v4flash", ["--max-steps-per-conversation", "1", "--max-tokens", "8192"], "swe_heldout"),
    "terminal": ("glm53_all_nopy", ["--max-steps-per-conversation", "1", "--max-conversations-per", "task_id=3",
                                    "--max-tokens", "12288"], "term_heldout"),
}


def paths(kind):
    stem = SETS[kind][2]
    return (os.path.join(E.EVAL_DIR, f"{stem}.jsonl"), os.path.join(E.EVAL_DIR, f"{stem}_card.json"),
            os.path.join(ROOT, "data", "eval", f"{stem}_src"))


OUT, CARD, WORK = paths("swe")


def source_args(cfg, kind="swe"):
    """The loop.yaml source of this set (swe: the Open-SWE source, openswe_v4flash, renamed openswe_v4flash_12k when
    the row cap went to 12,288; terminal: the non-Python GLM-5.3 source, glm53_all_nopy_12k) as public_source.py
    arguments for its held-out side: every filter the pool uses (scripts/loop.py public_source_args), only what the
    pool leaves out, one step per run (swe rows up to 8,192 tokens: the set was built at that length and never
    changes)."""
    import loop as L
    prefix, extra, _ = SETS[kind]
    held = [s for s in cfg["sources"] if s.get("holdout") and s["name"].startswith(prefix)]
    if len(held) != 1:
        raise SystemExit(f"experiments/loop.yaml: expected one {prefix} source with a holdout, found "
                         f"{[s['name'] for s in held]}")
    src = held[0]
    return L.public_source_args(src, side="heldout") + extra, src


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--set", choices=sorted(SETS), default="swe")
    ap.add_argument("--rows", type=int, default=200)
    ap.add_argument("--config", default=os.path.join(ROOT, "experiments", "loop.yaml"))
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    out, card_path, work = paths(args.set)
    if os.path.exists(out) and not args.force:
        raise SystemExit(f"{out} exists; rounds compare against it, so it must not change (--force only before the "
                         "first training round)")
    import yaml
    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    ps_args, src = source_args(cfg, args.set)
    from anneal import llama
    cmd = [sys.executable, "-u", os.path.join(ROOT, "scripts", "public_source.py")] + ps_args + [
        "--tokenizer", llama.SNAPSHOT, "--decontaminate", "--decontaminate-gate", E.DECONTAM_EXTRA, "--out", work]
    print(" ".join(cmd), flush=True)
    rc = subprocess.run(cmd, cwd=ROOT).returncode
    if rc != 0:
        raise SystemExit(f"public_source.py exited with {rc}")
    rows = E.read_jsonl(os.path.join(work, "train.jsonl")) + E.read_jsonl(os.path.join(work, "val.jsonl"))
    random.Random(42).shuffle(rows)
    pick = rows[:args.rows]
    E.write_jsonl(out, pick)
    measure = json.load(open(os.path.join(work, "measure.json"), encoding="utf-8"))
    card = {"built": time.strftime("%Y-%m-%d %H:%M:%S"), "set": args.set, "source": src["name"], "dataset": src["dataset"],
            "revision": measure.get("revision"), "holdout": src["holdout"], "available_steps": len(rows),
            "rows": len(pick), "counts": measure.get("counts"), "report_fields": measure.get("report_fields"),
            "kept_tokens": measure.get("kept_tokens")}
    with open(card_path, "w", encoding="utf-8") as fh:
        json.dump(card, fh, indent=1)
    print(f"wrote {out}: {len(pick)} of {len(rows)} held-out steps", flush=True)


if __name__ == "__main__":
    main()
