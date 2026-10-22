"""Build the loop's extra no-worse checks (experiments/loop.yaml guards). CPU and network only; safe to run next to
scoring or training.

  python scripts/build_guards.py --bfcl
  python scripts/build_guards.py --cweval [--cweval-src DIR] [--local]
  python scripts/build_guards.py --smoke    (only while the loop is paused: it needs the GPU)

--bfcl  data/eval/bfcl_java_js.jsonl: BFCL V4's 100 simple_java and 50 simple_javascript items as tool-call requests
        (anneal/bfcl.py), from the pinned bfcl-eval 2026.3.23 wheel (downloaded to data/raw/bfcl and sha256-checked;
        Apache-2.0, code and data). The rows must come out byte for byte as pinned below, then BFCL's checker is
        loaded from the wheel and judges six known answers (right, wrong value, wrong type, no call, cut off, bad
        JSON) as expected. Running it again changes nothing. Last, a report (it blocks nothing): how many rows of the
        loop's prepared pools (data/processed/loop/*/train.jsonl) have a first user message that overlaps a BFCL
        question by the decontamination rule (anneal/evalsets.py OverlapIndex). The pools were built before this
        check, so BFCL is not among the sets they were decontaminated against.
--cweval data/eval/cweval_go_js.jsonl: the 14 Go and 18 JavaScript CWEval tasks that run here (anneal/cweval.py;
        Co1lin/CWEval at e9a2a12, Apache-2.0), from the pinned files of anneal/cweval_lock.json (downloaded to
        data/raw/cweval, each sha256-checked; --cweval-src DIR copies CWEval's own files from a clone instead, when
        their sha256 match). Unpacks them to sandbox/cweval and has the Go toolchain make the module cache and
        goimports there (build container, offline from the pinned zips). The rows must come out byte for byte as
        pinned below and the copies of CWEval's prompt and code extraction must match its source. Then the self-test,
        through the executor: every task's reference must pass, as a whole program and sent as an answer (its code in
        a fence, through goimports); every insecure solution must fail its security tests; six known answers (a Go
        answer missing its imports, an insecure one, empty, cut off, not building, throwing) judged as expected.
        Writes data/eval/cweval_selftest.json, then the same overlap report as --bfcl. --local (Linux tests only):
        no sandbox at all, the go and node on PATH.
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
from anneal import bfcl, cweval  # noqa: E402
from anneal import evalsets as E  # noqa: E402

EVAL_DIR = os.path.join(ROOT, "data", "eval")
POOLS = os.path.join(ROOT, "data", "processed", "loop")
BFCL_OUT = os.path.join(EVAL_DIR, "bfcl_java_js.jsonl")
BFCL_ROWS_SHA256 = "f69b618c3639563895c25942b16c101ca941ee5a10e48e567dd23452f09537ea"
CWEVAL_OUT = os.path.join(EVAL_DIR, "cweval_go_js.jsonl")
CWEVAL_ROWS_SHA256 = "d4f3580e86ca2c9394f9c9ff96d36acc862d0cbd7d4690baed81e44b6954d20b"


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


def overlap_report(items, pools=POOLS, label="BFCL"):
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
        print(f"  overlap with {label}: {name}: {len(hits)} of {n} rows" + (f" {hits[:5]}" if hits else ""))
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


def cweval_tools(local):
    """(executor, toolchain): the sandbox and the pinned toolchains, or (--local, Linux tests only) no isolation and
    the go and node on PATH, with a Go build cache kept between runs."""
    from anneal import polyglot as P
    if not local:
        return P.SandboxExecutor(), P.Toolchain.windows()
    if os.name == "nt":
        raise SystemExit("--local is for Linux tests only: on Windows the check runs in the sandbox")
    return P.LocalExecutor(), P.Toolchain.local(gocache=os.path.join(ROOT, "sandbox", "cweval-gocache"))


def cweval_answer(item, code, fin="stop"):
    """code before the task's entry point, fenced as a chat answer."""
    head = code.split(item["entry_line"])[0].rstrip()
    return {"content": f"Here is the implementation.\n\n```{item['lang']}\n{head}\n```\n", "finish_reason": fin}


def cweval_selftest(rows, out_path):
    """Through the executor, two at a time: each reference must pass (as a whole program and sent as an answer, which
    goes through goimports) and each insecure solution must fail its security tests; then known answers through
    judge(). A reference that only ran out of time is judged once more, and the line says so. Writes every result to
    out_path."""
    import re
    import time
    from concurrent.futures import ThreadPoolExecutor

    def timed_out(d):
        return bool(d.get("failed")) and all("Timeout >" in e for e in d["failed"].values())

    def one(item):
        t0, res = time.time(), {"id": item["id"]}
        answer = cweval.program(item, cweval_answer(item, item["reference"])["content"])
        for kind, code, fix in (("reference", item["reference"], False), ("unsafe", item["unsafe"], False),
                                ("answer", answer, True)):
            ok, verdict, d = cweval.judge_program(item, code, fix_imports=fix)
            if kind != "unsafe" and not ok and timed_out(d):
                first = d
                ok, verdict, d = cweval.judge_program(item, code, fix_imports=fix)
                d = dict(d, retried_after=first["failed"])
            res[kind] = dict(d, passed=ok, verdict=verdict)
        res["seconds"] = round(time.time() - t0, 1)
        return res
    with ThreadPoolExecutor(cweval.MAX_PARALLEL) as pool:
        results = list(pool.map(one, rows))
    bad = []
    for res in results:
        ref, uns, ans = res["reference"], res["unsafe"], res["answer"]
        why = [w for w, b in (("reference fails", not ref["passed"]), ("insecure passes security", uns.get("secure")),
                              ("reference as an answer fails", not ans["passed"])) if b]
        slow = max(c.get("slowest") or 0 for c in (ref, ans))
        retried = [k for k, c in (("reference", ref), ("as an answer", ans)) if "retried_after" in c]
        print(f"  {res['id']}: reference {ref['verdict']} {ref.get('tests')}, insecure {uns['verdict']}"
              f" (functional {uns.get('functional')}), as an answer {ans['verdict']}; slowest reference test"
              f" {slow:.2f} s, {res['seconds']} s in all" + (f"  TIMED OUT ONCE: {retried}" if retried else "")
              + (f"  BAD: {why}" if why else ""))
        if why:
            bad.append((res["id"], why))
    by = {r["id"]: r for r in rows}
    go_jwt, go0, js0 = by["cwe_347_0_go"], by["cwe_020_0_go"], by["cwe_020_0_js"]
    no_imports = re.sub(r"(?ms)^import \(\n.*?^\)\n", "", go_jwt["reference"], count=1)
    cases = [("Go answer without its imports (goimports adds them, jwt from the module cache)", go_jwt,
              cweval_answer(go_jwt, no_imports), "ok"),
             ("insecure Go answer", go0, cweval_answer(go0, go0["unsafe"]), "insecure"),
             ("empty answer", js0, {"content": "", "finish_reason": "stop"}, "empty"),
             ("cut off", js0, cweval_answer(js0, js0["reference"], "length"), "finish_length"),
             ("Go that doesn't build", go0, {"content": "```go\npackage main\n\nfunc broken( {\n```",
                                             "finish_reason": "stop"}, "compile_error"),
             ("JavaScript that throws", js0, {"content": "```js\nthrow new Error('no');\n```", "finish_reason": "stop"},
              "not_functional")]
    for name, item, resp, want in cases:
        ok, verdict = cweval.judge(item, resp)
        right = verdict == want and ok == (want == "ok")
        print(f"  self-test {name}: {ok} ({verdict})" + ("" if right else f"  WANTED {want}"))
        results.append({"id": item["id"], "case": name, "passed": ok, "verdict": verdict})
        if not right:
            bad.append((name, verdict))
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=1)
    if bad:
        raise SystemExit(f"CWEval self-test failed: {bad}")


def build_cweval(src=None, local=False):
    ex, tc = cweval_tools(local)
    cweval.install(ex, tc, src=src)
    bad = cweval.parity()
    if bad:
        raise SystemExit(f"CWEval copies differ from the pinned source: {bad}")
    print("prompt template, language lines, get_code_from, default prompt and test timeout match the pinned source")
    rows = cweval.build_rows()
    tmp = write_rows(CWEVAL_OUT, rows)
    sha = cweval.file_sha(tmp)
    if sha != CWEVAL_ROWS_SHA256:
        os.remove(tmp)
        raise SystemExit(f"CWEval rows sha256 {sha}, want {CWEVAL_ROWS_SHA256}: not written")
    os.replace(tmp, CWEVAL_OUT)
    langs = {}
    for r in rows:
        langs[r["language"]] = langs.get(r["language"], 0) + 1
    print(f"{CWEVAL_OUT}: {len(rows)} items {langs}, sha256 {sha}")
    cweval.configure(ex, tc)
    cweval.prepare()
    cweval_selftest(rows, os.path.join(EVAL_DIR, "cweval_selftest.json"))
    print("CWEval check ready")
    rep = overlap_report(rows, label="CWEval")
    with open(os.path.join(EVAL_DIR, "cweval_overlap.json"), "w", encoding="utf-8") as fh:
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
    ap.add_argument("--cweval", action="store_true", help="build data/eval/cweval_go_js.jsonl and sandbox/cweval")
    ap.add_argument("--cweval-src", help="a CWEval clone at the pinned commit to take its files from")
    ap.add_argument("--local", action="store_true", help="--cweval on Linux without the sandbox (tests only)")
    ap.add_argument("--smoke", action="store_true", help="two BFCL requests on the scoring server (loop paused)")
    args = ap.parse_args()
    if not (args.bfcl or args.cweval or args.smoke):
        ap.error("name a check to build (--bfcl, --cweval) or --smoke")
    if args.bfcl:
        build_bfcl()
    if args.cweval:
        build_cweval(args.cweval_src, args.local)
    if args.smoke:
        return smoke_bfcl()
    return 0


if __name__ == "__main__":
    sys.exit(main())
