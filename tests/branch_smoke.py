"""Half-length run check for the tuning pass (30 Oct 2026), run from C:\\AI\\training while the GPU is free.

The tuning pass gets a half-length run by branching from the full run's checkpoint where a half-length run would
start its decay (train_longrun.py short_run_fractions), then decaying over the rest of its steps. This checks, on the
12 longest 12k rows (experiments\\dryrun-27b-12k.yaml, KL 1.0, the loop's BLAS library), that the branch is that
shorter run:

  A  6 steps with short_run_fractions [0.5]: must keep milestones\\checkpoint-2 (a 3-step run decays from step 2)
  C  the same settings for 3 steps from the start: the run the branch should reproduce
  B  a decay branch of 1 step from A's checkpoint-2: must end at step 3 with its adapter under branches\\checkpoint-2

Pass: losses at steps 1-2 of A and C agree, and B's step-3 loss agrees with C's, each within 1% (same weights and the
same rows, so the data position was restored); B's adapter differs from C's by at most a tenth of what step 3
changed in C (so the learning rate schedule was restored). Results in results\\research\\tuning\\branch_smoke.json.

  .venv\\Scripts\\python tests\\branch_smoke.py
Each run is skipped when its summary.json already exists, so a rerun picks up where it stopped.
"""
import json
import math
import os
import subprocess
import sys

PY = sys.executable
CFG = os.path.join("experiments", "dryrun-27b-12k.yaml")
OUT = os.path.join("results", "research", "tuning")
COMMON = ["--set", "mem_debug=false", "--set", "blas_backend=hipblaslt", "--set", "kl_think_beta=1.0"]
LOSS_TOL = 0.01
WEIGHT_TOL = 0.1


def gpu_free():
    out = subprocess.run(["tasklist", "/FI", "IMAGENAME eq llama-server.exe"], capture_output=True, text=True).stdout
    return "llama-server.exe" not in out


def run(name, extra):
    d = os.path.join(OUT, f"smoke_{name}")
    if os.path.exists(os.path.join(d, "summary.json")):
        print(f"{name}: done before, skipped", flush=True)
        return d
    if not gpu_free():
        raise SystemExit("llama-server.exe is running: the GPU is not free, nothing started")
    cmd = [PY, os.path.join("scripts", "train_longrun.py"), "--config", CFG] + COMMON + extra + ["--run-dir", d]
    print(f"{name}: {subprocess.list2cmdline(cmd)}", flush=True)
    r = subprocess.run(cmd)
    print(f"{name}: exit {r.returncode}", flush=True)
    if r.returncode != 0 or not os.path.exists(os.path.join(d, "summary.json")):
        raise SystemExit(f"{name} failed (exit {r.returncode}); later runs not started")
    return d


def summary(d):
    with open(os.path.join(d, "summary.json"), encoding="utf-8") as f:
        return json.load(f)


def losses(d):
    return {int(s): float(x) for s, x in summary(d)["losses"]}


def tensors(folder):
    from safetensors.torch import load_file
    return {k: v.float() for k, v in load_file(os.path.join(folder, "adapter_model.safetensors")).items()}


def dist(a, b):
    if set(a) != set(b):
        raise SystemExit(f"adapters hold different tensors ({len(set(a) ^ set(b))} names differ)")
    return math.sqrt(sum(float(((a[k] - b[k]) ** 2).sum()) for k in a))


def main():
    os.makedirs(OUT, exist_ok=True)
    a = run("A", ["--max-steps", "6", "--set", "short_run_fractions=[0.5]"])
    ms = os.path.join(a, "milestones", "checkpoint-2")
    if not os.path.isdir(ms):
        raise SystemExit(f"A kept no {ms}: short_run_fractions did not keep the branch point")
    c = run("C", ["--max-steps", "3"])
    b = run("B", ["--set", "branch_decay_steps=1", "--branch-from", ms])
    sb = summary(b)
    b_adapter = os.path.join(b, "branches", "checkpoint-2", "adapter")
    res = {"A_steps": summary(a)["global_step"], "C_steps": summary(c)["global_step"], "B_steps": sb["global_step"],
           "B_adapter": b_adapter, "B_adapter_exists": os.path.isdir(b_adapter)}
    la, lb, lc = losses(a), losses(b), losses(c)
    res["loss"] = {"A": la, "B": lb, "C": lc}
    rel = lambda x, y: abs(x - y) / max(abs(y), 1e-9)  # noqa: E731
    res["loss_rel_diff"] = {"A_vs_C_step1": rel(la[1], lc[1]), "A_vs_C_step2": rel(la[2], lc[2]),
                            "B_vs_C_step3": rel(lb[3], lc[3])}
    tb, tc, tm = tensors(b_adapter), tensors(os.path.join(c, "adapter")), tensors(ms)
    res["weights"] = {"B_vs_C": dist(tb, tc), "step3_change_in_C": dist(tc, tm)}
    res["weights"]["ratio"] = res["weights"]["B_vs_C"] / max(res["weights"]["step3_change_in_C"], 1e-12)
    checks = {"A ran 6 steps": res["A_steps"] == 6, "C ran 3 steps": res["C_steps"] == 3,
              "B ended at step 3": res["B_steps"] == 3, "B saved its adapter": res["B_adapter_exists"],
              "losses agree": all(v <= LOSS_TOL for v in res["loss_rel_diff"].values()),
              "B matches C": res["weights"]["step3_change_in_C"] > 0 and res["weights"]["ratio"] <= WEIGHT_TOL}
    res["checks"] = checks
    res["ok"] = all(checks.values())
    with open(os.path.join(OUT, "branch_smoke.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    print(json.dumps({k: res[k] for k in ("loss_rel_diff", "weights", "checks", "ok")}, indent=1))
    print("BRANCH SMOKE " + ("PASSED" if res["ok"] else "FAILED"))
    return 0 if res["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
