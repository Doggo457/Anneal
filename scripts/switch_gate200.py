"""Put the fixed 200 in use as the loop's gate (run once, with the loop paused; CPU only).

  python scripts/switch_gate200.py [--reuse results/gate200/base-rejudged.jsonl]

My rule (25 Oct 2026): after the base has answered all 495, the gate is a fixed 200 of them (scripts/build_gate200.py)
and the base's score is its share of the 200. Rounds can only be compared on one gate, so round 0 starts again on the
200:
  - data/eval/gate.jsonl (the 495) moves to data/eval/archive/gate495/ and data/eval/gate200/gate.jsonl takes its
    place (data/eval/card.* stay the 495's build card; the 200's card is data/eval/gate200/card.*);
  - results/loop/rounds/r000 and r001 move to results/loop/archive/gate495/, and state.json (copied first to
    state.gate495.json) forgets the gate's checksum, the base, the best and their gate server, so round 0 scores the
    base again: the gate, the non-coding check, then the base's no-worse checks. Everything else in state.json
    (source offsets, the EvalPlus base, the toolchains) stays.
--reuse: the base's re-judged answers (scripts/rejudge_gate.py) become round 0's gate rows, so the base is not asked
again. Only for a gate server that hasn't changed since they were answered: state.json then keeps its gate server
record, and the loop waits if loop.yaml asks for another. Without --reuse the base answers the 200 again (8 slots,
25 Oct 2026: answers differ with the server's settings, so every score must come from the same ones).
"""

import argparse
import json
import os
import shutil
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from anneal import evalsets as E  # noqa: E402

FORGET = ("gate_sha", "gate_server", "base", "best", "base_guards")


def switch(run_dir, gate200_dir, reuse_rows=None, seed=None, log=print):
    import loop as L
    try:
        pid = int(open(os.path.join(run_dir, "loop.pid"), encoding="utf-8").read().strip() or 0)
    except (OSError, ValueError):
        pid = 0
    if pid and L.pid_alive(pid):
        raise SystemExit(f"loop.py (pid {pid}) is running: pause it on the Training page first")
    new_gate = os.path.join(gate200_dir, "gate.jsonl")
    items = E.read_jsonl(new_gate)
    old = E.read_jsonl(E.GATE_PATH)
    if not items or any("judge" not in it for it in items):
        raise SystemExit(f"{new_gate} isn't a built fixed 200 (scripts/build_gate200.py)")
    if all("judge" in it for it in old):
        raise SystemExit(f"{E.GATE_PATH} is already the fixed 200")
    if len(old) != 495:
        raise SystemExit(f"{E.GATE_PATH} has {len(old)} items, not the 495")
    if not {it["id"] for it in items} <= {it["id"] for it in old}:
        raise SystemExit("the fixed 200 has items the 495 doesn't")
    for it in items:
        E.item_cases(it)  # every test file present and unchanged
    state_path = os.path.join(run_dir, "state.json")
    state = json.load(open(state_path, encoding="utf-8"))
    rows = []
    if reuse_rows:
        have = {r["id"]: r for r in E.read_jsonl(reuse_rows) if r["seed"] == seed}
        rows = [have[it["id"]] for it in items if it["id"] in have]
        log(f"reusing {len(rows)} re-judged base answers; {len(items) - len(rows)} items will be answered")
    arch_eval = os.path.join(E.EVAL_DIR, "archive", "gate495")
    arch_run = os.path.join(run_dir, "archive", "gate495")
    for d in (arch_eval, arch_run):
        if os.path.exists(d):
            raise SystemExit(f"{d} exists: the switch has already started; finish it by hand")
    os.makedirs(arch_eval)
    os.makedirs(os.path.join(arch_run, "rounds"))
    shutil.copy2(state_path, os.path.join(run_dir, "state.gate495.json"))
    shutil.move(E.GATE_PATH, os.path.join(arch_eval, "gate.jsonl"))
    shutil.copy2(new_gate, E.GATE_PATH)
    for n in ("r000", "r001"):
        src = os.path.join(run_dir, "rounds", n)
        if os.path.exists(src):
            shutil.move(src, os.path.join(arch_run, "rounds", n))
    forget = [k for k in FORGET if not (reuse_rows and k == "gate_server")]
    for k in forget:
        state.pop(k, None)
    state["rounds"] = [r for r in state.get("rounds") or [] if r["round"] >= 2]
    state["gate200"] = {"switched": time.strftime("%Y-%m-%d %H:%M:%S"), "items": len(items),
                        "reused_base_answers": len(rows), "archive": os.path.relpath(arch_run, run_dir)}
    if rows:
        r0 = os.path.join(run_dir, "rounds", "r000")
        E.write_jsonl(os.path.join(r0, f"gate.seed{seed}.jsonl"), rows)
    tmp = state_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(state, fh, indent=1)
    os.replace(tmp, state_path)
    log(f"the gate is now the fixed 200 ({len(items)} items); the 495 and rounds r000-r001 are archived; "
        f"state.json forgot {', '.join(forget)}; resume the loop from the Training page")
    return state


def main():
    import loop as L
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run-dir", default=os.path.join(L.RESULTS_DIR, "loop"))
    ap.add_argument("--gate200", default=os.path.join(E.EVAL_DIR, "gate200"))
    ap.add_argument("--reuse", default=None, help="re-judged base rows (scripts/rejudge_gate.py --out)")
    args = ap.parse_args()
    seed = int(L.load_config(L.CONFIG)["gate"]["seed"])
    switch(args.run_dir, args.gate200, args.reuse, seed)


if __name__ == "__main__":
    main()
