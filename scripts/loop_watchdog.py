"""Keeps the first run and the improvement loop going without anyone at the machine (no admin rights needed).

Run hourly by a user-level scheduled task and once at sign-in from the Startup folder (setup/loop-watchdog.md). Each run
asks the portal (http://127.0.0.1:8080/api/training) what state the runs are in and does at most one thing:
  - first-run ended by a crash or reboot ("ended" / "failed")    -> resume it through the portal (at most 3 times a day)
  - first-run finished, no "loop" run yet, six-language gate built -> start scripts/loop.py as the Training page run
                                                                   "loop" (not on the retired Python gate)
  - loop ended by a crash or reboot                             -> resume it
  - loop failed                                                 -> resume it, at most 3 times in 24 hours
  - loop waiting (no unused rows / low disk / old scripts)      -> resume it once experiments/loop.yaml has changed or
                                                                   staged scripts were installed, or 6 hours later
  - loop paused by training's memory cut-out (anneal/mem_guard): resume it, from the last checkpoint, once the trip is
                                                                   5 minutes old and Windows has 12 GiB free again; at
                                                                   most twice from the same checkpoint (a third trip
                                                                   there stays paused for a look) and 6 times a day
Anything paused or stopped by a person is left alone, and nothing happens while another run holds the GPU.
The memory cut-out resume is my call of 26 Nov 2026 ("ok so restart it when it triggered"): the cut-out ends training
before Windows runs out of memory, and training then carries on from its last checkpoint without me. Training resumed
from a checkpoint goes through the same rows in the same order, so what the model learns does not change.
Every decision is appended to results/watchdog/watchdog.log; restarts are counted in results/watchdog/state.json.

Staged scripts: the loop needs newer prepare_dataset.py and train_longrun.py than the running first run loads. They
wait in staged/ and are copied into scripts/ only after the first run has finished the last stage that runs them
(prepare_dataset.py after "generate", train_longrun.py after "train"); the replaced file is kept in staged/replaced/.

  pythonw scripts\\loop_watchdog.py [--delay SECONDS] [--dry-run] [--every SECONDS]
"""

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
import urllib.error
import urllib.request
from operator import ge, lt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PORTAL = "http://127.0.0.1:8080"
RESULTS = os.path.join(ROOT, "results")
WD_DIR = os.path.join(RESULTS, "watchdog")
LOOP_CONFIG = os.path.join(ROOT, "experiments", "loop.yaml")
GATE = os.path.join(ROOT, "data", "eval", "gate.jsonl")
MAX_RESTARTS_PER_DAY = 3
MEM_RESUME_FREE_GIB = 12.0  # Windows free memory needed before training starts again after a memory cut-out
MEM_RESUME_AFTER_S = 300  # and the cut-out at least this old
MEM_SAME_CHECKPOINT_MAX = 2  # resumes from one checkpoint; a trip after that leaves the loop paused
MEM_RESUMES_PER_DAY = 6
WAIT_RECHECK_S = 6 * 3600
BUSY = ("starting", "running", "pausing", "stopping")
# script -> the first run's last stage that runs it (scripts/first_run.py STAGES)
STAGED_AFTER = (("prepare_dataset.py", "generate"), ("train_longrun.py", "train"))


def log(msg):
    os.makedirs(WD_DIR, exist_ok=True)
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    with open(os.path.join(WD_DIR, "watchdog.log"), "a", encoding="utf-8") as fh:
        fh.write(line + "\n")
    if sys.stdout:
        try:
            print(line, flush=True)
        except (OSError, ValueError):
            pass


def api(path, body=None, method=None, timeout=60):
    data = None if body is None else json.dumps(body).encode("utf-8")
    req = urllib.request.Request(PORTAL + path, data=data, method=method or ("GET" if body is None else "POST"),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as exc:
        try:
            return json.loads(exc.read() or b"{}")
        except ValueError:
            return {"ok": False, "error": f"HTTP {exc.code}"}


def read_json(path):
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def load_state():
    return read_json(os.path.join(WD_DIR, "state.json")) or {"restarts": []}


def save_state(st):
    os.makedirs(WD_DIR, exist_ok=True)
    tmp = os.path.join(WD_DIR, "state.json.tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(st, fh, indent=1)
    os.replace(tmp, os.path.join(WD_DIR, "state.json"))


def recent_restarts(st, tag, now, window=86400):
    return [r for r in st["restarts"] if r["tag"] == tag and now - r["t"] < window]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def install_staged(root, first_dir, dry_run=False):
    """Copy each staged script into scripts/ once the first run is past the stages that run it. -> names installed."""
    staged, done = os.path.join(root, "staged"), []
    for name, stage in STAGED_AFTER:
        src, dst = os.path.join(staged, name), os.path.join(root, "scripts", name)
        if not os.path.exists(src) or not os.path.exists(os.path.join(first_dir, "stages", f"{stage}.done.json")):
            continue
        if os.path.exists(dst) and sha256(src) == sha256(dst):
            continue
        if dry_run:
            log(f"dry run: would install staged/{name} (first run's {stage} stage is done)")
            continue
        if os.path.exists(dst):
            os.makedirs(os.path.join(staged, "replaced"), exist_ok=True)
            shutil.copyfile(dst, os.path.join(staged, "replaced", f"{name}.{time.strftime('%Y%m%d-%H%M%S')}"))
        shutil.copyfile(src, dst + ".tmp")
        os.replace(dst + ".tmp", dst)
        log(f"installed staged/{name} -> scripts/{name} (sha256 {sha256(dst)[:16]}; first run's {stage} stage is done)")
        done.append(name)
    return done


def gate_ready(path=GATE):
    """The six-language gate is built (scripts/build_gate.py): its first item is a write or fix item. The retired Python
    gate (cc and oci items) does not count, so the loop never starts on it."""
    try:
        with open(path, encoding="utf-8") as fh:
            first = json.loads(fh.readline() or "{}")
    except (OSError, ValueError):
        return False
    return first.get("kind") in ("write", "fix")


def mem_trip(loop_dir=os.path.join(RESULTS, "loop")):
    """The loop's training paused by the memory cut-out: {"round", "checkpoint" (newest checkpoint step, 0 = none),
    "t" (when), "error"}, or None. Only a running round's training whose status.json says paused_by mem_guard counts."""
    state = read_json(os.path.join(loop_dir, "state.json")) or {}
    for rec in state.get("rounds") or []:
        if rec.get("status") != "running":
            continue
        tdir = os.path.join(loop_dir, "rounds", f"r{int(rec['round']):03d}", "train")
        ts = read_json(os.path.join(tdir, "status.json")) or {}
        if ts.get("state") != "paused" or ts.get("paused_by") != "mem_guard":
            continue
        ck = os.path.join(tdir, "checkpoints")
        steps = [int(d.split("-")[1]) for d in (os.listdir(ck) if os.path.isdir(ck) else [])
                 if d.startswith("checkpoint-") and d.split("-")[1].isdigit()]
        return {"round": int(rec["round"]), "checkpoint": max(steps) if steps else 0,
                "t": float(ts.get("updated") or 0), "error": str(ts.get("error") or "")[:300]}
    return None


def free_gib():
    """Windows' available physical memory in GiB (anneal/mem_guard's reading), or None if it can't be read."""
    try:
        sys.path.insert(0, ROOT)
        from anneal.mem_guard import read_system
        return float(read_system()["avail_gib"])
    except Exception:  # noqa: BLE001
        return None


def mem_resume(now, st, mem, avail_gib, loop_stopped):
    """After a memory cut-out: (action, tag, reason) as decide gives them."""
    where = f"round {mem['round']}'s training paused by the memory cut-out at checkpoint {mem['checkpoint']}"
    if loop_stopped:
        return None, "loop", where + ", and Pause pressed since: left alone"
    if lt(now - mem["t"], MEM_RESUME_AFTER_S):
        return None, "loop", where + f" {int(now - mem['t'])} s ago: waiting {MEM_RESUME_AFTER_S} s"
    if avail_gib is None or lt(avail_gib, MEM_RESUME_FREE_GIB):
        return None, "loop", where + f"; Windows free memory {avail_gib} GiB, under {MEM_RESUME_FREE_GIB}: waiting"
    mine = [r for r in st["restarts"] if r.get("tag") == "loop-mem"]
    same = [r for r in mine if (r.get("mem") or {}).get("round") == mem["round"]
            and (r.get("mem") or {}).get("checkpoint") == mem["checkpoint"]]
    if ge(len(same), MEM_SAME_CHECKPOINT_MAX):
        return None, "loop", (where + f", again after {len(same)} resumes from that checkpoint: left paused for a look "
                              "(resume it from the Training page)")
    if ge(len([r for r in mine if lt(now - r["t"], 86400)]), MEM_RESUMES_PER_DAY):
        return None, "loop", where + f", but already resumed {MEM_RESUMES_PER_DAY} times in 24 h: left paused"
    return "resume", "loop", where + f"; Windows has {avail_gib} GiB free again: resuming from the last checkpoint"


def decide(training, now, st, first_status, loop_status, gate_exists, config_mtime, scripts_installed=False,
           mem=None, avail_gib=None, loop_stopped=False):
    """-> (action, tag, reason) with action in None / "resume" / "start". Pure, so it can be tested."""
    # mem: mem_trip(); avail_gib: free_gib(); loop_stopped: the loop's STOP file exists (Pause pressed)
    if training.get("active"):
        return None, None, f"{training['active']} holds the GPU"
    runs = {r["tag"]: r for r in training.get("runs") or []}
    fr, lp = runs.get("first-run"), runs.get("loop")

    def restart(tag, reason):
        if len(recent_restarts(st, tag, now)) >= MAX_RESTARTS_PER_DAY:
            return None, tag, f"{tag}: {reason}, but already restarted {MAX_RESTARTS_PER_DAY} times in 24 h"
        return "resume", tag, reason

    if fr is not None and fr.get("state") != "finished":
        state = fr.get("state")
        if state in BUSY or state in ("paused", "stopped"):
            return None, "first-run", f"first-run is {state}"
        return restart("first-run", f"first-run {state} (crash or reboot)")
    if fr is None or (first_status or {}).get("state") != "finished":
        return None, None, "first run not finished"
    if lp is None:
        if not gate_exists:
            return None, "loop", ("first run finished but the six-language gate is not built "
                                  "(scripts/setup_toolchains.py, then scripts/build_gate.py)")
        return "start", "loop", "first run finished: starting the improvement loop"
    state = lp.get("state")
    if state == "paused" and (loop_status or {}).get("state") == "paused" and mem:
        return mem_resume(now, st, mem, avail_gib, loop_stopped)
    if state in BUSY or state in ("paused", "stopped"):
        return None, "loop", f"loop is {state}"
    said = (loop_status or {}).get("state")
    if said == "waiting":
        changed = config_mtime and config_mtime > float((loop_status or {}).get("config_mtime") or 0)
        stale = now - float((loop_status or {}).get("updated") or 0) > WAIT_RECHECK_S
        if changed or stale or scripts_installed:
            why = "loop.yaml changed" if changed else ("staged scripts were installed" if scripts_installed else
                                                      "6 hours passed")
            return restart("loop", "loop was waiting and " + why)
        return None, "loop", f"loop waiting: {(loop_status or {}).get('detail')}"
    if said == "finished" and state == "finished":
        return None, "loop", "loop finished (round limit)"
    return restart("loop", f"loop {state} (status {said})")


def check(dry_run=False):
    """One look at the runs; at most one action."""
    try:
        installed = install_staged(ROOT, os.path.join(RESULTS, "first-run"), dry_run)
    except OSError as exc:
        log(f"installing staged scripts failed ({exc}); will retry next run")
        installed = []
    try:
        training = api("/api/training")
    except Exception as exc:  # noqa: BLE001
        log(f"portal not reachable ({type(exc).__name__}: {exc}); nothing done")
        return
    now = time.time()
    st = load_state()
    mem = mem_trip()
    action, tag, reason = decide(training, now, st, read_json(os.path.join(RESULTS, "first-run", "status.json")),
                                 read_json(os.path.join(RESULTS, "loop", "status.json")), gate_ready(),
                                 os.path.getmtime(LOOP_CONFIG) if os.path.exists(LOOP_CONFIG) else None,
                                 bool(installed), mem, free_gib() if mem else None,
                                 os.path.exists(os.path.join(RESULTS, "loop", "STOP")))
    if action is None:
        log(f"no action: {reason}")
        return
    if dry_run:
        log(f"dry run: would {action} {tag}: {reason}")
        return
    if action == "start":
        res = api("/api/training/runs", {"script": "scripts/loop.py", "args": "", "tag": "loop"})
    else:
        res = api(f"/api/training/runs/{tag}/resume", {})
    rec = {"t": now, "tag": tag, "action": action, "reason": reason, "ok": bool(res.get("ok"))}
    if mem and tag == "loop" and "memory cut-out" in reason:
        rec.update(tag="loop-mem", mem={"round": mem["round"], "checkpoint": mem["checkpoint"]})
    st["restarts"].append(rec)
    st["restarts"] = st["restarts"][-200:]
    save_state(st)
    log(f"{action} {tag}: {reason} -> {'ok' if res.get('ok') else res.get('error')}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--delay", type=int, default=0, help="seconds to wait first (at sign-in the portal needs a moment)")
    ap.add_argument("--every", type=int, default=0,
                    help="keep running and check every SECONDS (only if no hourly scheduled task can be created)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    if args.delay:
        time.sleep(args.delay)
    while True:
        try:
            check(args.dry_run)
        except Exception as exc:  # noqa: BLE001  (a bad hour must not end a --every watchdog)
            log(f"check failed: {type(exc).__name__}: {exc}")
        if not args.every:
            return 0
        time.sleep(args.every)


if __name__ == "__main__":
    sys.exit(main())
