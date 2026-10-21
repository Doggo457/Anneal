"""Build the loop's extra no-worse checks (experiments/loop.yaml guards). CPU and network only; safe to run next to
scoring or training.

  python scripts/build_guards.py --bfcl

--bfcl  data/eval/bfcl_java_js.jsonl: BFCL V4's 100 simple_java and 50 simple_javascript items as tool-call requests
        (anneal/bfcl.py), from the pinned bfcl-eval 2026.3.23 wheel (downloaded to data/raw/bfcl and sha256-checked;
        Apache-2.0, code and data). The rows must come out byte for byte as pinned below, then BFCL's checker is
        loaded from the wheel and judges six known answers (right, wrong value, wrong type, no call, cut off, bad
        JSON) as expected. Running it again changes nothing. Last, a report (it blocks nothing): how many rows of the
        loop's prepared pools (data/processed/loop/*/train.jsonl) have a first user message that overlaps a BFCL
        question by the decontamination rule (anneal/evalsets.py OverlapIndex). The pools were built before this
        check, so BFCL is not among the sets they were decontaminated against.
"""

import argparse
import glob
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from anneal import bfcl  # noqa: E402
from anneal import evalsets as E  # noqa: E402

EVAL_DIR = os.path.join(ROOT, "data", "eval")
POOLS = os.path.join(ROOT, "data", "processed", "loop")
BFCL_OUT = os.path.join(EVAL_DIR, "bfcl_java_js.jsonl")
BFCL_ROWS_SHA256 = "f69b618c3639563895c25942b16c101ca941ee5a10e48e567dd23452f09537ea"


def write_rows(path, rows):
    tmp = path + ".tmp"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    return tmp


def bfcl_selftest(rows):
    """Six answers with known verdicts through the real checker (zip-imported from the wheel)."""
    by = {r["id"]: r for r in rows}
    j0, s0 = by["simple_java_0"], by["simple_javascript_0"]

    def call(name, args, fin="tool_calls"):
        fn = {"name": name, "arguments": json.dumps(args)}
        return {"content": "", "finish_reason": fin, "tool_calls": [{"type": "function", "id": "t", "function": fn}]}
    cases = [("right java", j0, call("GeometryPresentation_createPresentation",
                                     {"controller": "mapController", "parent": "mapArea"}), True),
             ("swapped values", j0, call("GeometryPresentation_createPresentation",
                                         {"controller": "mapArea", "parent": "mapController"}), False),
             ("right javascript", s0, call("validateUserInput", {"inputField": "userInputField", "isComplete": "true"}),
              True),
             ("JSON true, not a string", s0, call("validateUserInput", {"inputField": "userInputField",
                                                                         "isComplete": True}), False),
             ("no call", j0, {"content": "I need more information.", "finish_reason": "stop", "tool_calls": []}, False),
             ("cut off", s0, call("validateUserInput", {"inputField": "userInputField", "isComplete": "true"},
                                  "length"), False)]
    bad = []
    for name, item, resp, want in cases:
        ok, verdict = bfcl.judge(item, resp)
        print(f"  self-test {name}: {ok} ({verdict})" + ("" if ok == want else f"  WANTED {want}"))
        if ok != want:
            bad.append(name)
    if bad:
        raise SystemExit(f"BFCL checker self-test failed: {bad}")


def first_user(row):
    for m in row.get("messages") or []:
        if m.get("role") == "user":
            c = m.get("content")
            return c if isinstance(c, str) else json.dumps(c, ensure_ascii=False)
    return ""


def overlap_report(items, pools=POOLS):
    """{pool: [(row, item ids)]} for pool rows whose first user message overlaps an item's question."""
    idx = E.OverlapIndex([(it["id"], " ".join(m["content"] for m in it["messages"] if m.get("role") == "user"))
                          for it in items])
    out = {}
    for path in sorted(glob.glob(os.path.join(pools, "*", "train.jsonl"))):
        name, hits, n = os.path.basename(os.path.dirname(path)), [], 0
        with open(path, encoding="utf-8") as fh:
            for i, line in enumerate(fh):
                if not line.strip():
                    continue
                n += 1
                h = idx.hits(first_user(json.loads(line)))
                if h:
                    hits.append((i, sorted(h)))
        out[name] = {"rows": n, "overlapping": hits}
        print(f"  overlap with BFCL: {name}: {len(hits)} of {n} rows" + (f" {hits[:5]}" if hits else ""))
    return out


def build_bfcl():
    wheel = bfcl.fetch_wheel()
    print(f"wheel {wheel} sha256 {bfcl.WHEEL_SHA256}: ok")
    rows = bfcl.build_rows(wheel)
    tmp = write_rows(BFCL_OUT, rows)
    sha = bfcl.file_sha(tmp)
    if sha != BFCL_ROWS_SHA256:
        os.remove(tmp)
        raise SystemExit(f"BFCL rows sha256 {sha}, want {BFCL_ROWS_SHA256}: not written")
    os.replace(tmp, BFCL_OUT)
    langs = {}
    for r in rows:
        langs[r["language"]] = langs.get(r["language"], 0) + 1
    print(f"{BFCL_OUT}: {len(rows)} items {langs}, sha256 {sha}")
    bfcl.prepare()
    bfcl_selftest(rows)
    print("BFCL check ready")
    rep = overlap_report(rows)
    with open(os.path.join(EVAL_DIR, "bfcl_overlap.json"), "w", encoding="utf-8") as fh:
        json.dump({k: dict(v, overlapping=v["overlapping"][:200]) for k, v in rep.items()}, fh, indent=1)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--bfcl", action="store_true", help="build data/eval/bfcl_java_js.jsonl")
    args = ap.parse_args()
    if not args.bfcl:
        ap.error("name a check to build (--bfcl)")
    if args.bfcl:
        build_bfcl()
    return 0


if __name__ == "__main__":
    sys.exit(main())
