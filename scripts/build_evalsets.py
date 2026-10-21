"""Build the improvement loop's held-out sets once (CPU only; safe to run next to a training run).

RETIRED for the gate (21 Oct 2026): the gate is now six languages, write and fix (scripts/build_gate.py), which archives
this script's Python gate in data/eval/archive/python-gate/; this script then refuses to run. The non-coding check
and the decontamination list it built stay in use.

  data/eval/gate.jsonl            CodeContests valid + test problems whose own Python 3 reference passes our sandboxed
                                  runner, plus 300 OpenCodeInstruct problems from shard train-00025 (no run reads it)
                                  whose reference passes its assert tests. Problems that overlap EvalPlus, any data
                                  trained on so far (data/processed/*/{train,val}.jsonl) or any prompt in the shards
                                  the first run generates its own rows from (train-00000 and train-00049; OpenCodeInstruct
                                  has near-duplicate prompts across shards) are left out.
  data/eval/noncoding_100.jsonl   100 MMLU-Pro test questions outside computer science (random.Random(42)).
  data/eval/decontam_extra.jsonl  every gate problem and non-coding question, for decontaminating later training data
                                  (prepare_dataset.py --decontaminate-gate, public_source.py).
  data/eval/card.md               sources with their exact revisions, counts at every filter, harness results.

The rules (anneal/evalsets.py): the gate is never trained on and never reported; adapters are kept or rejected on it,
and EvalPlus / LiveCodeBench stay untouched for the reported numbers.

  python scripts/build_evalsets.py [--oci-target 300] [--parallel 4] [--force]
"""

import argparse
import glob
import json
import os
import random
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from anneal import evalsets as E  # noqa: E402

HUB = os.path.join(ROOT, "models", "hf", "hub")
OCI_REPO, OCI_REV = "nvidia/OpenCodeInstruct", "8f3ba5bafe4d6e8db46082cf7ae6741bc370604d"  # the first run's revision
OCI_GATE_SHARD = "data/train-00025-of-00050.parquet"
# the first run generates its self rows from these shards (scripts/first_run.py OCI_FILES)
FIRST_RUN_SHARDS = ["data/train-00000-of-00050.parquet", "data/train-00049-of-00050.parquet"]
CC_REPO, MMLU_REPO = "deepmind/code_contests", "TIGER-Lab/MMLU-Pro"
CC_COLUMNS = ["name", "description", "public_tests", "private_tests", "generated_tests", "solutions", "input_file",
              "output_file", "cf_rating", "difficulty", "source"]


def fetch(repo, patterns, revision=None):
    """Download the dataset files matching any of the prefixes into models/hf/hub. Returns (paths, commit sha)."""
    from huggingface_hub import HfApi, hf_hub_download
    api = HfApi()
    sha = revision or api.dataset_info(repo).sha
    files = [f for f in api.list_repo_files(repo, repo_type="dataset", revision=sha)
             if any(f.startswith(p) for p in patterns) and f.endswith(".parquet")]
    if not files:
        raise SystemExit(f"{repo}@{sha}: no files match {patterns}")
    paths = [hf_hub_download(repo, f, repo_type="dataset", revision=sha, cache_dir=HUB) for f in sorted(files)]
    return paths, sha


def parquet_rows(paths, columns=None):
    import pyarrow.parquet as pq
    for p in paths:
        pf = pq.ParquetFile(p)
        cols = [c for c in columns if c in pf.schema_arrow.names] if columns else None
        for batch in pf.iter_batches(batch_size=256, columns=cols):
            yield from batch.to_pylist()


def trained_texts():
    """(prompt, response) of every row in data/processed/*/{train,val}.jsonl: all data trained on so far."""
    out = []
    for path in sorted(glob.glob(os.path.join(ROOT, "data", "processed", "*", "*.jsonl"))):
        if os.path.basename(path) not in ("train.jsonl", "val.jsonl"):
            continue
        for r in E.read_jsonl(path):
            m = r.get("messages") or []
            if len(m) >= 2:
                out.append((m[-2].get("content") or "", m[-1].get("content") or ""))
    return out


def evalplus_index():
    from evalplus.data import get_human_eval_plus, get_mbpp_plus
    items = [(f"evalplus/{k}", v["prompt"]) for ds in (get_human_eval_plus(), get_mbpp_plus()) for k, v in ds.items()]
    return E.OverlapIndex(items)


def select_oci(pool, target, ep, harness, pool_texts, chunk=64, margin=16, log=print):
    """OCI gate problems in pool order: not overlapping EvalPlus (ep: OverlapIndex), reference passing (harness:
    list -> kept list), and not overlapped by any prompt from the first run's generation shards (pool_texts: callable
    returning an iterable of prompts, scanned once per top-up). -> (kept, counts)."""
    kept, examined, ep_hits, pool_hits = [], 0, 0, 0
    while len(kept) < target and examined < len(pool):
        fresh = []
        while len(fresh) < target - len(kept) + margin and examined < len(pool):
            batch = []
            while len(batch) < chunk and examined < len(pool):
                c = pool[examined]
                examined += 1
                if ep.hits(c["raw_prompt"]):
                    ep_hits += 1
                    continue
                batch.append(c)
            fresh += harness(batch)
        idx = E.OverlapIndex([(c["id"], c["raw_prompt"]) for c in fresh])
        hit = set()
        for text in pool_texts():
            hit |= idx.hits(text)
        pool_hits += len(hit)
        log(f"OCI top-up: {len(fresh)} pass their reference, {len(hit)} overlap the first run's generation shards")
        kept += [c for c in fresh if c["id"] not in hit]
    return kept[:target], {"examined": examined, "evalplus_overlap": ep_hits, "first_run_pool_overlap": pool_hits}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--oci-target", type=int, default=300)
    ap.add_argument("--parallel", type=int, default=4)
    ap.add_argument("--force", action="store_true", help="rebuild even if gate.jsonl exists")
    args = ap.parse_args()
    if os.path.isdir(os.path.join(E.EVAL_DIR, "archive", "python-gate")):
        raise SystemExit("The Python gate was retired on 21 Oct 2026 (Python is the lowest priority) and is archived in "
                         "data/eval/archive/python-gate/. The gate is built by scripts/build_gate.py.")
    if os.path.exists(E.GATE_PATH) and not args.force:
        raise SystemExit(f"{E.GATE_PATH} exists. The gate must not change once rounds are scored on it; use --force "
                         "only before the loop's round 0.")
    import generate_traces as gt
    from anneal import sandbox
    sandbox.require()
    t0 = time.time()
    card = {"started": time.strftime("%Y-%m-%d %H:%M:%S")}
    log = lambda m: print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)  # noqa: E731

    # ---- overlap references: EvalPlus prompts, and everything trained on so far
    ep = evalplus_index()
    trained = trained_texts()
    log(f"overlap references: {len(ep)} EvalPlus prompts, {len(trained)} trained rows")

    # ---- CodeContests valid + test
    cc_paths, cc_sha = fetch(CC_REPO, ["data/valid-", "data/test-"])
    cands = []
    for split in ("valid", "test"):
        rows = list(parquet_rows([p for p in cc_paths if os.path.basename(p).startswith(split + "-")], CC_COLUMNS))
        c = E.cc_candidates(rows, split, gt.STDIO_SUFFIX)
        card[f"cc_{split}_rows"], card[f"cc_{split}_candidates"] = len(rows), len(c)
        cands += c
    log(f"CodeContests: {len(cands)} candidates with stdin/stdout cases and a Python 3 reference")
    cc = E.harness_check(cands, gt.run_io_tests, gt.run_assert_tests, args.parallel, log)
    card["cc_reference_passes"] = len(cc)

    # ---- OpenCodeInstruct shard 25: average_test_score 1.0, assert tests, reference passes
    oci_paths, _ = fetch(OCI_REPO, [OCI_GATE_SHARD], revision=OCI_REV)
    pool = []
    for r in parquet_rows(oci_paths, ["id", "input", "output", "unit_tests", "average_test_score", "domain"]):
        try:
            if float(r["average_test_score"]) < 1.0:
                continue
        except (TypeError, ValueError):
            continue
        tests = gt.parse_tests([r["unit_tests"]], "auto", 20)
        if tests is None or tests[0] != "asserts" or not (r.get("input") or "").strip():
            continue
        pool.append({"id": f"oci/{r['id']}", "kind": "oci", "prompt": r["input"].strip(), "raw_prompt": r["input"].strip(),
                     "tests": tests[1], "ref": gt.extract_code(r["output"] or ""), "domain": r.get("domain")})
    random.Random(42).shuffle(pool)
    card["oci_pool"] = len(pool)
    fr_paths, _ = fetch(OCI_REPO, FIRST_RUN_SHARDS, revision=OCI_REV)

    def first_run_prompts():
        return (r["input"] for r in parquet_rows(fr_paths, ["input"]) if r.get("input"))

    oci, n = select_oci(pool, args.oci_target, ep,
                        lambda batch: E.harness_check(batch, gt.run_io_tests, gt.run_assert_tests, args.parallel, log),
                        first_run_prompts, log=log)
    examined, ep_hits, fr_hits = n["examined"], n["evalplus_overlap"], n["first_run_pool_overlap"]
    card.update(oci_examined=examined, oci_evalplus_overlap=ep_hits, oci_first_run_pool_overlap=fr_hits,
                oci_kept=len(oci))
    log(f"OpenCodeInstruct: {len(oci)} kept of {examined} examined ({ep_hits} overlap EvalPlus, {fr_hits} overlap "
        f"the first run's generation shards)")

    # ---- drop gate problems that overlap EvalPlus or anything trained on so far
    gate = cc + oci
    gidx = E.OverlapIndex([(g["id"], g["raw_prompt"]) for g in gate])
    dropped = set()
    for p, resp in trained:
        dropped |= gidx.hits(p + "\n" + resp, jaccard_text=p)
    for g in gate:
        if g["kind"] == "cc" and ep.hits(g["raw_prompt"]):
            dropped.add(g["id"])
    card["gate_dropped_overlap"] = sorted(dropped)
    gate = [g for g in gate if g["id"] not in dropped]
    for g in gate:
        g.pop("raw_prompt", None)
    log(f"gate: {len(gate)} problems ({sum(g['kind'] == 'cc' for g in gate)} CodeContests, "
        f"{sum(g['kind'] == 'oci' for g in gate)} OpenCodeInstruct); {len(dropped)} dropped for overlap")

    # ---- MMLU-Pro non-coding check
    mm_paths, mm_sha = fetch(MMLU_REPO, ["data/test-"])
    mm_rows = list(parquet_rows(mm_paths, ["question_id", "question", "options", "answer", "answer_index", "category"]))
    nc = E.mmlu_pro_questions(mm_rows, n=100, seed=42)
    cats = {}
    for q in nc:
        cats[q["category"]] = cats.get(q["category"], 0) + 1

    # ---- write
    E.write_jsonl(E.GATE_PATH, gate)
    E.write_jsonl(E.NONCODING_PATH, nc)
    extra = [{"id": g["id"], "prompt": g["prompt"]} for g in gate] + [{"id": q["id"], "question": q["question"]} for q in nc]
    E.write_jsonl(E.DECONTAM_EXTRA, extra)
    card.update(cc_revision=cc_sha, oci_revision=OCI_REV, oci_shard=OCI_GATE_SHARD, mmlu_pro_revision=mm_sha,
                mmlu_pro_test_rows=len(mm_rows), noncoding_categories=cats, gate_size=len(gate),
                seconds=round(time.time() - t0))
    with open(os.path.join(E.EVAL_DIR, "card.json"), "w", encoding="utf-8") as fh:
        json.dump(card, fh, indent=1)
    lines = ["# Held-out sets for the improvement loop", "",
             f"Built {card['started']} by scripts/build_evalsets.py in {card['seconds']} s.", "",
             "| Set | Source | Kept |", "|---|---|---|",
             f"| gate: CodeContests | `{CC_REPO}` @ `{cc_sha}`, valid {card['cc_valid_rows']} + test {card['cc_test_rows']} "
             f"rows; {card['cc_valid_candidates'] + card['cc_test_candidates']} with stdin/stdout cases and a Python 3 "
             f"reference; {card['cc_reference_passes']} whose reference passes | "
             f"{sum(g['kind'] == 'cc' for g in gate)} |",
             f"| gate: OpenCodeInstruct | `{OCI_REPO}` @ `{OCI_REV}` `{OCI_GATE_SHARD}`, {card['oci_pool']} rows with score 1.0 "
             f"and assert tests; {examined} examined, {ep_hits} overlap EvalPlus, {fr_hits} overlap a prompt in the first "
             f"run's generation shards ({', '.join(FIRST_RUN_SHARDS)}) | {sum(g['kind'] == 'oci' for g in gate)} |",
             f"| non-coding | `{MMLU_REPO}` @ `{mm_sha}` test ({len(mm_rows)} rows), outside computer science, "
             f"random.Random(42) | {len(nc)} |", "",
             f"Gate problems dropped for overlap with EvalPlus or data trained on so far: {len(dropped)}.",
             f"Non-coding categories: {json.dumps(cats)}.", "",
             "Overlap rule (anneal/evalsets.py OverlapIndex): a text overlaps a problem if it holds at least 30% of the "
             "problem's 13-grams or its word set has Jaccard > 0.8 with the problem's."]
    with open(os.path.join(E.EVAL_DIR, "card.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    log(f"wrote {E.GATE_PATH}, {E.NONCODING_PATH}, {E.DECONTAM_EXTRA}, card.md")


if __name__ == "__main__":
    main()
