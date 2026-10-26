"""Compare short training runs on the same rows and seed (26 Oct 2026, speed-up items 3 and 4).

Usage:
  python scripts/compare_train_runs.py BASELINE_DIR OTHER_DIR [OTHER_DIR ...] [--skip 1] [--min-gain 0.05]
      [--loss-tol 0.01] [--max-reserved 29] [--json out.json]

Each folder is a train_longrun.py run folder (metrics.jsonl, summary.json). The first optimizer steps (--skip) are left
out of the timing; the rest give seconds per step and tokens per second. Losses are compared step by step with the
baseline's. A change passes when it is at least --min-gain faster, every step loss is within --loss-tol (relative) of
the baseline's, and its torch peak reserved memory stays at or under --max-reserved GiB. A second baseline run passed as
OTHER shows the noise floor (its gain and loss drift should be near zero).
"""
import argparse
import json
import os
import sys


def read_run(folder, skip=1):
    steps = {}
    with open(os.path.join(folder, "metrics.jsonl"), encoding="utf-8") as f:
        for line in f:
            try:
                m = json.loads(line)
            except ValueError:
                continue
            if "step" in m and "seconds" in m and "loss" in m and not m.get("event"):
                steps[int(m["step"])] = m
    summary = {}
    path = os.path.join(folder, "summary.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            summary = json.load(f)
    order = sorted(steps)
    timed = [steps[s] for s in order if s > skip]
    secs = sum(float(m["seconds"]) for m in timed)
    toks = sum(int(m.get("tokens") or 0) for m in timed)
    return {"folder": folder, "steps": order, "timed_steps": [int(m["step"]) for m in timed],
            "seconds_per_step": round(secs / len(timed), 3) if timed else None,
            "tokens_per_second": round(toks / secs, 2) if timed and secs and toks else None,
            "loss": {s: float(steps[s]["loss"]) for s in order},
            "grad_norm": {s: steps[s].get("grad_norm") for s in order},
            "peak_reserved_gib": summary.get("peak_reserved_gib",
                                             max((float(m.get("peak_reserved_gib") or 0) for m in steps.values()),
                                                 default=None)),
            "windows_peak": summary.get("windows_peak"), "blas": summary.get("blas"), "sdpa": summary.get("sdpa"),
            "profile": summary.get("profile")}


def compare(base, other, min_gain=0.05, loss_tol=0.01, max_reserved=29.0):
    common = sorted(set(base["loss"]) & set(other["loss"]))
    drift = [abs(other["loss"][s] - base["loss"][s]) / abs(base["loss"][s]) for s in common if base["loss"][s]]
    speed = (base["seconds_per_step"] / other["seconds_per_step"]
             if base["seconds_per_step"] and other["seconds_per_step"] else None)
    tps = (other["tokens_per_second"] / base["tokens_per_second"]
           if base["tokens_per_second"] and other["tokens_per_second"] else None)
    same_timed = base["timed_steps"] == other["timed_steps"]
    checks = {
        "faster": speed is not None and speed >= 1 + min_gain,
        "loss_within_tol": bool(common) and max(drift, default=0) <= loss_tol,
        "memory_ok": other["peak_reserved_gib"] is not None and float(other["peak_reserved_gib"]) <= max_reserved,
        "same_steps": same_timed and bool(common),
    }
    return {"folder": other["folder"], "speed_ratio": round(speed, 4) if speed else None,
            "tokens_per_second_ratio": round(tps, 4) if tps else None,
            "max_loss_drift": round(max(drift), 5) if drift else None,
            "mean_loss_drift": round(sum(drift) / len(drift), 5) if drift else None,
            "peak_reserved_gib": other["peak_reserved_gib"], "checks": checks, "passes": all(checks.values())}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("baseline")
    p.add_argument("others", nargs="+")
    p.add_argument("--skip", type=int, default=1, help="leading optimizer steps left out of the timing")
    p.add_argument("--min-gain", type=float, default=0.05)
    p.add_argument("--loss-tol", type=float, default=0.01)
    p.add_argument("--max-reserved", type=float, default=29.0)
    p.add_argument("--json", default=None)
    a = p.parse_args(argv)
    base = read_run(a.baseline, a.skip)
    runs = [read_run(f, a.skip) for f in a.others]
    out = {"baseline": base, "runs": runs,
           "comparisons": [compare(base, r, a.min_gain, a.loss_tol, a.max_reserved) for r in runs],
           "rule": {"skip": a.skip, "min_gain": a.min_gain, "loss_tol": a.loss_tol, "max_reserved_gib": a.max_reserved}}
    print(f"baseline {base['folder']}: {base['seconds_per_step']} s/step, {base['tokens_per_second']} tok/s, "
          f"peak reserved {base['peak_reserved_gib']} GiB, timed steps {base['timed_steps']}, blas {base['blas']}")
    for r, c in zip(runs, out["comparisons"]):
        print(f"{r['folder']}: {r['seconds_per_step']} s/step, {r['tokens_per_second']} tok/s, speed x{c['speed_ratio']}, "
              f"loss drift max {c['max_loss_drift']} mean {c['mean_loss_drift']}, peak reserved {c['peak_reserved_gib']} "
              f"GiB, checks {c['checks']} -> {'PASS' if c['passes'] else 'no'}")
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=1, default=str)
    return 0


if __name__ == "__main__":
    sys.exit(main())
