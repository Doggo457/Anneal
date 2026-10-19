"""One compact status digest of the first run, the improvement loop and the watchdog, for the hourly check-ins.

  python scripts\\loop_status.py            # prints JSON (about 2-4 KB)
"""

import json
import os
import shutil
import sys
import time
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")


def read_json(path):
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def tail(path, n):
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            return [ln.rstrip() for ln in fh.readlines()[-n:]]
    except OSError:
        return []


def count_lines(path):
    try:
        with open(path, encoding="utf-8") as fh:
            return sum(1 for ln in fh if ln.strip())
    except OSError:
        return None


def main():
    out = {"now_utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
           "disk_free_gb": round(shutil.disk_usage(ROOT).free / 2 ** 30, 1)}
    try:
        with urllib.request.urlopen("http://127.0.0.1:8080/api/training", timeout=20) as r:
            t = json.loads(r.read())
        out["portal"] = {"active": t.get("active"), "hold": t.get("hold"),
                         "runs": [{k: r.get(k) for k in ("tag", "state", "step", "max_steps", "resumes", "error")}
                                  for r in (t.get("runs") or []) if r.get("tag") in ("first-run", "loop")]}
    except Exception as exc:  # noqa: BLE001
        out["portal"] = f"unreachable: {type(exc).__name__}: {exc}"
    fr = os.path.join(RESULTS, "first-run")
    out["first_run"] = {"status": read_json(os.path.join(fr, "status.json")),
                        "stages_done": sorted(f[:-10] for f in os.listdir(os.path.join(fr, "stages"))
                                              if f.endswith(".done.json")) if os.path.isdir(os.path.join(fr, "stages")) else [],
                        "self_rows_kept": count_lines(os.path.join(ROOT, "data", "raw", "self_oci_r1.jsonl")),
                        "report_exists": os.path.exists(os.path.join(fr, "report.md")),
                        "log_tail": tail(os.path.join(fr, "run.log"), 3)}
    lp = os.path.join(RESULTS, "loop")
    state = read_json(os.path.join(lp, "state.json")) or {}
    out["loop"] = {"status": read_json(os.path.join(lp, "status.json")),
                   "best": {k: (state.get("best") or {}).get(k) for k in ("round", "name", "gate_seed")},
                   "rejects_in_row": state.get("rejects_in_row"), "recipes_done": state.get("recipes_done"),
                   "offsets": state.get("offsets"),
                   "rounds": [{k: r.get(k) for k in ("round", "status", "recipe", "init", "rows", "gate", "gate_diff",
                                                      "p", "noncoding", "reasons", "failures", "last_error", "finished")}
                              for r in (state.get("rounds") or [])[-4:]],
                   "log_tail": tail(os.path.join(lp, "run.log"), 3)}
    out["eval_sets"] = read_json(os.path.join(ROOT, "data", "eval", "card.json"))
    out["eval_sets"] = {k: out["eval_sets"].get(k) for k in ("gate_size", "cc_reference_passes", "oci_kept")} \
        if out["eval_sets"] else None
    out["watchdog_tail"] = tail(os.path.join(RESULTS, "watchdog", "watchdog.log"), 3)
    json.dump(out, sys.stdout, indent=1, default=str)
    print()


if __name__ == "__main__":
    main()
