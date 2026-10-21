"""Build the loop's extra no-worse checks (experiments/loop.yaml guards). CPU and network only; safe to run next to
scoring or training.

  python scripts/build_guards.py --bfcl
  python scripts/build_guards.py --smoke    (only while the loop is paused: it needs the GPU)

--bfcl  data/eval/bfcl_java_js.jsonl: BFCL V4's 100 simple_java and 50 simple_javascript items as tool-call requests
        (anneal/bfcl.py), from the pinned bfcl-eval 2026.3.23 wheel (downloaded to data/raw/bfcl and sha256-checked;
        Apache-2.0, code and data). The rows must come out byte for byte as pinned below, then BFCL's checker is
        loaded from the wheel and judges six known answers (right, wrong value, wrong type, no call, cut off, bad
        JSON) as expected. Running it again changes nothing. Last, a report (it blocks nothing): how many rows of the
        loop's prepared pools (data/processed/loop/*/train.jsonl) have a first user message that overlaps a BFCL
        question by the decontamination rule (anneal/evalsets.py OverlapIndex). The pools were built before this
        check, so BFCL is not among the sets they were decontaminated against.
--smoke Only while the loop is paused (nothing on port 18090, the GPU free): starts the scoring server the loop uses
        (the base model, the gate's flags and loop.yaml's thinking budget), asks the first Java and the first
        JavaScript item exactly as a pass does, prints each answer's verdict, calls and tokens, and stops the server.
        Exit code 0 when both answers made a tool call (right or wrong), 1 otherwise.
"""

import argparse
import glob
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
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


def smoke_bfcl(port=18090):
    """Two real requests on the loop's scoring server settings (see --smoke). Returns the exit code."""
    import yaml
    import evalplus_win as ew
    import first_run as fr
    from anneal import llama
    try:
        llama.http(f"http://127.0.0.1:{port}/health", timeout=5)
        print(f"something answers on port {port} (the loop's scoring server?): pause the loop first")
        return 2
    except Exception:  # noqa: BLE001 - nothing there, as wanted
        pass
    cfg = yaml.safe_load(open(os.path.join(ROOT, "experiments", "loop.yaml"), encoding="utf-8"))
    g, seed = cfg["gate"], int(cfg["guards"]["bfcl"].get("seed", 1))
    items = [json.loads(line) for line in open(BFCL_OUT, encoding="utf-8") if line.strip()]
    picks = [next(i for i, it in enumerate(items) if it["language"] == lg) for lg in ("java", "javascript")]
    flags = llama.scoring_flags(extra=["--reasoning-budget", str(g["budget"])] + fr.SAMPLER_FLAGS)
    bfcl.prepare()
    called = 0
    log = os.path.join(ROOT, "logs", "bfcl_smoke_server.log")
    with llama.Server(flags, log, port=port) as srv:
        print(f"server up in {srv.load_s} s: {srv.describe()}")
        for i in picks:
            r = bfcl.request(srv.url, items[i], 100000 * seed + i, g["max_tokens"], ew.SAMPLING)
            ok, verdict = bfcl.judge(items[i], r)
            called += bool(r["tool_calls"])
            print(json.dumps({"id": items[i]["id"], "finish_reason": r["finish_reason"], "pass": ok, "verdict": verdict,
                              "tool_calls": r["tool_calls"], "completion_tokens": r["completion_tokens"],
                              "reasoning_chars": len(r["reasoning"]), "seconds": r["seconds"],
                              "content_head": r["content"][:300]}, ensure_ascii=False))
    print(f"{called} of {len(picks)} answers made a tool call")
    return 0 if called == len(picks) else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--bfcl", action="store_true", help="build data/eval/bfcl_java_js.jsonl")
    ap.add_argument("--smoke", action="store_true", help="two BFCL requests on the scoring server (loop paused)")
    args = ap.parse_args()
    if not (args.bfcl or args.smoke):
        ap.error("name a check to build (--bfcl) or --smoke")
    if args.bfcl:
        build_bfcl()
    if args.smoke:
        return smoke_bfcl()
    return 0


if __name__ == "__main__":
    sys.exit(main())
