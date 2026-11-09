r"""A round's strength curve as one Training page run (9 Nov 2026, my ask: is 0.25 still round 3's best strength after
it trained on from round 2?).

It runs scripts/lora_scale_check.py --part all once per strength, in the order given, and then writes curve.json: the
gate's 60 problems (write and fix), BFCL's 150 tool calls and the held-out losses at each strength, with the base and
the round as tested. Each strength resumes from its saved answers, so a pause loses only the answers in flight.

The Training page starts it with --run-dir and --resume; its pause writes STOP in the run folder, which this passes on
to the running strength's folder (lora_scale_check stops its BFCL and gate parts cleanly), then it ends as paused.
Every 30 s it also checks memory: under 1.5 GiB free or over 80 GB committed (the job watcher's line) it stops the
same way and ends as paused. It never kills anything. It refuses to start while loop.py runs.

  python scripts\scale_curve.py --round 4 --scales 0.25,0.125,0.5 [--n 60] [--run-dir results\scale-r004]
"""

import argparse
import json
import os
import subprocess
import sys
import time
from operator import gt, lt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from anneal.runtime import RESULTS_DIR, Run  # noqa: E402

CHECK = os.path.join(ROOT, "scripts", "lora_scale_check.py")
OUT_ROOT = os.path.join(RESULTS_DIR, "research", "lora_scale")
LOOP_PID = os.path.join(RESULTS_DIR, "loop", "loop.pid")
MIN_FREE_GIB, MAX_COMMIT_GB = 1.5, 80.0
EVERY_S, MEM_EVERY_S = 5, 30


def log(msg):
    print(time.strftime("%Y-%m-%d %H:%M:%S ") + msg, flush=True)


def read_json(path):
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def write_json(path, data):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=1)
    os.replace(tmp, path)


def scale_name(scale):  # lora_scale_check's folder name
    return f"{scale:.2f}" if round(scale, 2) == scale else f"{scale:g}"


def out_dir(rnum, scale):
    return os.path.join(OUT_ROOT, f"r{rnum:03d}_x{scale_name(scale)}")


def memory():
    """(free GiB, committed GB), or None off Windows."""
    if os.name != "nt":
        return None
    import job_watcher
    free, committed, _ = job_watcher.Machine().memory()
    return free, committed


def loop_running():
    try:
        with open(LOOP_PID, encoding="utf-8") as fh:
            pid = int(fh.read().strip() or 0)
    except (OSError, ValueError):
        return 0
    return pid if pid_alive(pid) else 0


def pid_alive(pid):  # loop.py's pid_alive, without loading loop.py
    if not pid:
        return False
    if os.name == "nt":
        import ctypes
        h = ctypes.windll.kernel32.OpenProcess(0x1000, False, int(pid))  # PROCESS_QUERY_LIMITED_INFORMATION
        if not h:
            return False
        code = ctypes.c_ulong()
        ok = ctypes.windll.kernel32.GetExitCodeProcess(h, ctypes.byref(code))
        ctypes.windll.kernel32.CloseHandle(h)
        return bool(ok) and code.value == 259  # STILL_ACTIVE
    try:
        os.kill(int(pid), 0)
        return True
    except OSError:
        return False


def run_one(run, rnum, scale, n, step, total):
    """lora_scale_check at one strength: returns "done", "stopped" (a pause or the memory line) or "failed"."""
    out = out_dir(rnum, scale)
    os.makedirs(os.path.join(run.dir, "logs"), exist_ok=True)
    child_log = os.path.join(run.dir, "logs", f"x{scale_name(scale)}.log")
    cmd = [sys.executable, "-u", CHECK, "--round", str(rnum), "--scale", repr(scale), "--n", str(n), "--part", "all"]
    log(f"x{scale:g} ({step}/{total}): {subprocess.list2cmdline(cmd)} (log {child_log})")
    run.write_status(state="running", step=step - 1, max_steps=total, detail=f"strength x{scale:g} ({step} of {total})")
    stop_file, why, last_mem = os.path.join(out, "STOP"), None, 0.0
    with open(child_log, "a", encoding="utf-8", errors="replace") as fh:
        p = subprocess.Popen(cmd, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                             env=dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1"))
        while p.poll() is None:
            time.sleep(EVERY_S)
            if why is None and run.stop_requested():
                why = "the Training page asked to pause"
            if why is None and gt(time.time() - last_mem, MEM_EVERY_S):
                last_mem = time.time()
                m = memory()
                if m and (lt(m[0], MIN_FREE_GIB) or gt(m[1], MAX_COMMIT_GB)):
                    why = f"memory line: {m[0]:.2f} GiB free, {m[1]:.2f} GB committed"
            if why and not os.path.exists(stop_file):
                os.makedirs(out, exist_ok=True)
                open(stop_file, "w").close()
                log(f"x{scale:g}: {why}; STOP written to {out} (answers so far are kept)")
    rc = p.returncode
    if why:
        if os.path.exists(stop_file):
            os.remove(stop_file)  # resuming must not stop at once
        log(f"x{scale:g}: stopped ({why}); exit code {rc}")
        run.write_status(state="paused", step=step - 1, max_steps=total, detail=f"x{scale:g} stopped: {why}")
        return "stopped"
    if rc != 0:
        log(f"x{scale:g}: lora_scale_check.py exited with {rc}; see {child_log}")
        run.write_status(state="failed", step=step - 1, max_steps=total, detail=f"x{scale:g} exited with {rc}")
        return "failed"
    write_json(os.path.join(run.dir, f"done_x{scale_name(scale)}.json"), {"scale": scale, "out": out,
                                                                         "time": time.strftime("%Y-%m-%d %H:%M:%S")})
    log(f"x{scale:g}: done")
    return "done"


def pick(d, *keys):
    for k in keys:
        if not isinstance(d, dict):
            return None
        d = d.get(k)
    return d


def curve(rnum, scales):
    """Numbers only: per strength the gate (all, write, fix), BFCL (all) and held-out losses, with base and round."""
    rows, ref = [], {}
    for s in sorted(scales):
        summ = read_json(os.path.join(out_dir(rnum, s), "summary.json")) or {}
        row = {"scale": s, "round_scale": summ.get("round_scale")}
        for g in ("all", "write", "fix"):
            row[f"gate_{g}"] = pick(summ, "gate", g, "part", "pass")
            row[f"gate_{g}_n"] = pick(summ, "gate", g, "n")
            row[f"gate_{g}_vs_base"] = pick(summ, "gate", g, "part_vs_base")
            row[f"gate_{g}_vs_round"] = pick(summ, "gate", g, "part_vs_full")
            for who in ("base", "full"):
                if pick(summ, "gate", g, who, "pass") is not None:
                    ref[f"gate_{g}_{who}"] = pick(summ, "gate", g, who, "pass")
        row["reasoning_mean"] = pick(summ, "gate", "all", "part", "reasoning_mean")
        row["budget_hits"] = pick(summ, "gate", "all", "part", "budget_hits")
        row["bfcl"] = pick(summ, "bfcl", "all", "part")
        row["bfcl_vs_base"] = pick(summ, "bfcl", "all", "part_vs_base")
        if pick(summ, "bfcl", "all", "base") is not None:
            ref["bfcl_base"] = pick(summ, "bfcl", "all", "base")
        for kind, s_ in (summ.get("loss") or {}).items():
            row[f"loss_{kind}"] = s_.get("part")
            row[f"loss_{kind}_vs_base"] = s_.get("part_rel_to_base")
            ref[f"loss_{kind}_base"], ref[f"loss_{kind}_round"] = s_.get("base"), s_.get("full")
        rows.append(row)
    return {"round": rnum, "time": time.strftime("%Y-%m-%d %H:%M:%S"), "reference": ref, "strengths": rows}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--round", type=int, required=True)
    ap.add_argument("--scales", required=True, help="comma-separated strengths of the full adapter, run in this order")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--resume", action="store_true", help="accepted for the Training page; it always resumes")
    args = ap.parse_args()
    scales = [float(x) for x in args.scales.split(",") if x.strip()]
    if not scales or any(not (gt(s, 0) and lt(s, 1)) for s in scales):
        raise SystemExit("--scales must be strengths between 0 and 1")
    run = Run(args.run_dir or os.path.join(RESULTS_DIR, f"scale-r{args.round:03d}"))
    run.capture_output()
    if os.path.exists(run.stop_path):
        os.remove(run.stop_path)  # starting again means the stop request has been handled
    log(f"== scale_curve.py round {args.round} strengths {scales} n {args.n} (pid {os.getpid()})")
    pid = loop_running()
    if pid:
        log(f"not started: loop.py is running (pid {pid})")
        run.write_status(state="failed", detail=f"loop.py is running (pid {pid})")
        return 1
    total = len(scales)
    for i, s in enumerate(scales, 1):
        if os.path.exists(os.path.join(run.dir, f"done_x{scale_name(s)}.json")):
            log(f"x{s:g}: done before, skipped")
            continue
        res = run_one(run, args.round, s, args.n, i, total)
        if res != "done":
            return 0 if res == "stopped" else 1
    c = curve(args.round, scales)
    write_json(os.path.join(run.dir, "curve.json"), c)
    log("curve: " + json.dumps(c))
    run.write_status(state="finished", step=total, max_steps=total, detail=f"{total} strengths done")
    log("SCALE CURVE DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
