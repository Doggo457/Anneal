"""Resume test for scripts/train_longrun.py before the first real run.

tiny (experiments/resume-check-tiny.yaml, Qwen/Qwen3.5-0.8B, same qwen3_5 code as the 27B):
  A  6 optimizer steps straight
  B  --stop-at-step 3 (pause + checkpoint-3), then the same command again (resumes from checkpoint-3) to step 6
  -> steps 4-6 of B must match A: same learning rate exactly, loss within 2% (GPU kernels are not bit-exact).
27b (experiments/dryrun-27b.yaml, the first-run model setup): --stop-at-step 2, resume, finish step 3; checks the
  trainable parameter count (79,691,776), the collated-label check, finite losses, step time and peak memory.

  python tests\\resume_check.py [--run-dir DIR] [--parts tiny,27b]     (through the Training page: GPU)
"""

import argparse
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from anneal.runtime import RESULTS_DIR, Run  # noqa: E402

TRAIN = os.path.join(ROOT, "scripts", "train_longrun.py")


def train(run, config, run_dir, extra=()):
    cmd = [sys.executable, "-u", TRAIN, "--config", config, "--run-dir", run_dir] + list(extra)
    print(">>", subprocess.list2cmdline(cmd))
    t0 = time.time()
    with open(os.path.join(run.dir, "children.log"), "a", encoding="utf-8", errors="replace") as log:
        rc = subprocess.call(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                             creationflags=0x08000000)
    status = json.load(open(os.path.join(run_dir, "status.json"), encoding="utf-8"))
    print(f"   rc {rc}, state {status.get('state')}, step {status.get('step')}, {time.time() - t0:.0f} s")
    return rc, status


def steps(run_dir):
    out = {}
    with open(os.path.join(run_dir, "metrics.jsonl"), encoding="utf-8") as fh:
        for line in fh:
            m = json.loads(line)
            if "loss" in m and "step" in m and "event" not in m:
                out[m["step"]] = m
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run-dir", default=os.path.join(RESULTS_DIR, "resume-check"))
    ap.add_argument("--parts", default="tiny,27b")
    ap.add_argument("--config-27b", default="dryrun-27b.yaml", help="config in experiments/ for the 27b part")
    ap.add_argument("--resume", action="store_true", help="accepted for the Training page; starts over")
    args = ap.parse_args()
    run = Run(args.run_dir)
    run.capture_output()
    run.write_status(state="running")
    summary, ok = {}, True
    exp = os.path.join(ROOT, "experiments")
    try:
        if "tiny" in args.parts:
            cfg = os.path.join(exp, "resume-check-tiny.yaml")
            a_dir, b_dir = os.path.join(run.dir, "tiny-straight"), os.path.join(run.dir, "tiny-resumed")
            rc_a, st_a = train(run, cfg, a_dir)
            rc_b1, st_b1 = train(run, cfg, b_dir, ["--stop-at-step", "3"])
            ck = sorted(os.listdir(os.path.join(b_dir, "checkpoints")))
            rc_b2, st_b2 = train(run, cfg, b_dir, ["--resume"])
            a, b = steps(a_dir), steps(b_dir)
            rows = []
            for s in sorted(a):
                if s in b:
                    rows.append({"step": s, "loss_straight": a[s]["loss"], "loss_resumed": b[s]["loss"],
                                 "lr_straight": a[s].get("learning_rate"), "lr_resumed": b[s].get("learning_rate"),
                                 "rel_diff": abs(a[s]["loss"] - b[s]["loss"]) / max(abs(a[s]["loss"]), 1e-9)})
            after = [r for r in rows if r["step"] > 3]
            tiny_ok = (rc_a == 0 and st_a.get("state") == "finished" and st_b1.get("state") == "paused"
                       and st_b1.get("step") == 3 and "checkpoint-3" in ck and rc_b2 == 0
                       and st_b2.get("state") == "finished" and st_b2.get("step") == 6 and len(after) == 3
                       and all(r["lr_straight"] == r["lr_resumed"] and r["rel_diff"] < 0.02 for r in after))
            summary["tiny"] = {"ok": tiny_ok, "checkpoints_after_pause": ck, "steps": rows,
                               "states": [st_a.get("state"), st_b1.get("state"), st_b2.get("state")]}
            print(json.dumps(summary["tiny"], indent=1))
            ok &= tiny_ok
        for name in (args.config_27b.split(",") if "27b" in args.parts else []):
            import yaml
            cfg = os.path.join(exp, name)
            n_steps = int(yaml.safe_load(open(cfg, encoding="utf-8"))["max_steps"])
            d = os.path.join(run.dir, os.path.splitext(name)[0])
            rc1, st1 = train(run, cfg, d, ["--stop-at-step", "2"])
            rc2, st2 = train(run, cfg, d, ["--resume"])
            s = json.load(open(os.path.join(d, "summary.json"), encoding="utf-8"))
            m = steps(d)
            big_ok = (st1.get("state") == "paused" and st1.get("step") == 2 and rc2 == 0
                      and st2.get("state") == "finished" and st2.get("step") == n_steps
                      and s.get("trainable_params") == 79691776 and s["collated_check"]["labels_match_helper"]
                      and not s["any_nan_loss"] and os.path.isdir(os.path.join(d, "adapter")))
            summary[name] = {"ok": big_ok, "warm_up": s.get("warm_up"), "trainable_params": s.get("trainable_params"),
                              "collated_check": s.get("collated_check"), "rows": s.get("rows"),
                              "rows_dropped_too_long": s.get("rows_dropped_too_long"),
                              "mask_stats": s.get("mask_stats"),
                              "sample_first_trained_tokens": (s.get("sample_row") or {}).get("first_trained_tokens"),
                              "steps": {k: {x: v.get(x) for x in ("loss", "learning_rate", "seconds", "tokens",
                                                                  "trained_tokens", "tokens_per_second",
                                                                  "peak_reserved_gib")} for k, v in m.items()},
                              "peak_reserved_gib": s.get("peak_reserved_gib"), "windows_peak": s.get("windows_peak")}
            print(json.dumps(summary[name], indent=1, default=str))
            ok &= big_ok
        summary["result"] = "PASS" if ok else "FAIL"
        print("RESULT:", summary["result"])
        run.write_status(state="finished" if ok else "failed")
    except Exception as exc:
        import traceback
        traceback.print_exc()
        summary["result"] = f"ERROR {exc}"
        run.write_status(state="failed", error=str(exc))
    with open(os.path.join(run.dir, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=1, default=str)
    return 0 if summary.get("result") == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
