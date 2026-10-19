"""Do OpenCodeInstruct's own reference solutions pass their unit tests in our sandboxed test runner?

The rows generate_traces.py draws (average_test_score 1.0, same shuffle seed) were judged passing by NVIDIA's
harness, so their reference "output" should pass here too. Failures point at our harness (sandbox, missing modules,
Windows differences, timeouts) rather than the model. Runs on CPU only, next to generation.

  python tests\\oci_reference_check.py [--n 200] [--run-dir DIR]
"""

import argparse
import collections
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import generate_traces as gt  # noqa: E402
from anneal import sandbox  # noqa: E402
from anneal.runtime import RESULTS_DIR, Run  # noqa: E402

OCI_FILES = [os.path.join(ROOT, "models", "hf", "hub", "datasets--nvidia--OpenCodeInstruct", "snapshots",
                          "8f3ba5bafe4d6e8db46082cf7ae6741bc370604d", "data", f"train-000{n}-of-00050.parquet")
             for n in ("00", "49")]


def classify(r):
    err = r.stderr.decode("utf-8", "replace")
    if r.timed_out:
        return "timeout", ""
    if r.returncode == 0:
        return "pass", ""
    last = [ln for ln in err.strip().splitlines() if ln.strip()][-1:] or [""]
    head = last[0].split(":")[0]
    return head or f"rc {r.returncode}", last[0][:200]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--run-dir", default=os.path.join(RESULTS_DIR, "oci-reference-check"))
    args = ap.parse_args()
    run = Run(args.run_dir)
    run.capture_output()
    from datasets import load_dataset
    ds = load_dataset("parquet", None, data_files=OCI_FILES, split="train")
    ds = ds.select_columns(["input", "output", "unit_tests", "average_test_score"])
    cases = []
    for ex in ds.shuffle(seed=42):  # generate_traces.py's order
        try:
            if float(ex["average_test_score"]) < 1.0:
                continue
        except (TypeError, ValueError):
            continue
        tests = gt.parse_tests([ex["unit_tests"]], "auto", 20)
        if tests is None or tests[0] != "asserts":
            continue
        cases.append((ex["input"], gt.extract_code(ex["output"]), tests[1]))
        if len(cases) >= args.n:
            break
    t0 = time.time()

    def one(c):
        prompt, code, tests = c
        r = sandbox.run(["candidate.py"], files={"candidate.py": code + "\n\n" + tests + "\n"}, timeout_s=20,
                        mem_bytes=gt.TEST_MEM_BYTES)
        kind, detail = classify(r)
        return {"kind": kind, "detail": detail, "prompt": prompt[:160]}

    with ThreadPoolExecutor(4) as ex:
        res = list(ex.map(one, cases))
    counts = collections.Counter(r["kind"] for r in res)
    summary = {"n": len(res), "seconds": round(time.time() - t0, 1), "counts": dict(counts),
               "pass_rate": counts["pass"] / len(res) if res else None,
               "failures": [r for r in res if r["kind"] != "pass"]}
    print(json.dumps({k: v for k, v in summary.items() if k != "failures"}, indent=1))
    for r in summary["failures"][:40]:
        print(f"  {r['kind']}: {r['detail']}  | {r['prompt'][:90]!r}")
    with open(os.path.join(run.dir, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
