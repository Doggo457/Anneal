r"""Runs the PC's long jobs one at a time from a queue, and starts a job again from its saved work if it stops.

Jobs started from a remote session stop when that session goes idle. This watcher runs on its own, from a user-level
scheduled task (setup/job-watcher.md) that starts it every hour; Windows ignores the start while one is running.

Queue: results\jobs\queue\*.json, run in file-name order. Each file holds
  {"name": "x0.25", "args": ["scripts\\lora_scale_check.py", "--scale", "0.25"], "done": ["results\\...\\DONE"],
   "stop_dirs": ["results\\..."], "max_attempts": 4}
"args" run with the venv's python.exe in C:\AI\training. The job is finished when every "done" path exists.

Every minute:
  - while a job runs: if free RAM is under 1.5 GiB or committed memory is over 74 GiB, it writes STOP into each
    existing stop_dirs folder (the job saves and exits). It never kills anything.
  - once the job has exited: finished, so it goes to done\. A STOP file is there, so it goes to held\ and a person
    decides. It exited within 2 minutes without finishing (it refused to start), so it is tried again 10 minutes
    later, 12 times at most. Otherwise it is started again (the jobs skip saved work), up to max_attempts, and then
    goes to failed\.
  - with no job running: a queued job whose done paths all exist goes to done\. The first other job starts once the
    GPU has been free for 3 checks in a row: no llama-server.exe and no other training or scoring script running.
Files in results\jobs: state.json, status.json (rewritten every check), watcher.log, logs\NAME.log (job output).

  pythonw scripts\job_watcher.py                    run it (what the scheduled task does)
  python scripts\job_watcher.py --status            print the status and the queue
  python scripts\job_watcher.py --once --dry-run    one check that changes nothing; prints what it would do
"""

import argparse
import ctypes
import json
import os
import subprocess
import sys
import time
import traceback
from operator import ge, gt, lt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JOBS = os.path.join(ROOT, "results", "jobs")
MIN_FREE_GIB, MAX_COMMIT_GIB = 1.5, 74.0
FREE_CHECKS, EVERY_S = 3, 60
QUICK_EXIT_S, QUICK_RETRY_S, MAX_QUICK = 120, 600, 12
BUSY = ("lora_scale_check.py", "strength_sweep.py", "train_longrun.py", "train_qlora.py", "val_loss.py",
        "branch_smoke.py", "sweep_runner", "loop.py", "first_run.py", "evalplus_win.py")
FOLDERS = ("queue", "done", "held", "failed", "logs")


def jpath(*parts):
    return os.path.join(JOBS, *parts)


class Machine:
    """Processes, memory and job starts on this PC. Tests pass a fake with the same methods."""

    def __init__(self):
        self.procs = {}

    def now(self):
        return time.time()

    def processes(self):
        import psutil
        out = []
        for p in psutil.process_iter(["pid", "name", "cmdline"]):
            info = p.info
            out.append((info["pid"], (info["name"] or "").lower(), " ".join(info["cmdline"] or [])))
        return out

    def alive(self, pid, created):
        import psutil
        try:
            p = psutil.Process(pid)
            return p.is_running() and lt(abs(p.create_time() - created), 2) and p.status() != psutil.STATUS_ZOMBIE
        except psutil.Error:
            return False

    def created(self, pid):
        import psutil
        try:
            return psutil.Process(pid).create_time()
        except psutil.Error:
            return 0.0

    def memory(self):
        """Free RAM, committed memory and the commit limit, in GiB."""
        class MS(ctypes.Structure):
            _fields_ = [("n", ctypes.c_ulong), ("load", ctypes.c_ulong)] + \
                       [(k, ctypes.c_ulonglong) for k in ("tp", "ap", "tf", "af", "tv", "av", "ae")]
        st = MS()
        st.n = ctypes.sizeof(MS)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st))
        return st.ap / 2 ** 30, (st.tf - st.af) / 2 ** 30, st.tf / 2 ** 30

    def start(self, args, log_path):
        exe = os.path.join(ROOT, ".venv", "Scripts", "python.exe")
        if not os.path.exists(exe):
            exe = sys.executable
        with open(log_path, "ab") as fh:
            p = subprocess.Popen([exe] + list(args), cwd=ROOT, stdin=subprocess.DEVNULL, stdout=fh,
                                 stderr=subprocess.STDOUT, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        self.procs[p.pid] = p
        return p.pid, self.created(p.pid)


def read_json(p):
    try:
        with open(p, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def write_json(p, obj):
    tmp = p + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=1)
    os.replace(tmp, p)


def utc(t):
    return time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime(t))


def queue():
    """(file name, job or None if the file is not a valid job), in file-name order."""
    out = []
    for f in sorted(os.listdir(jpath("queue"))):
        if f.endswith(".json"):
            spec = read_json(jpath("queue", f))
            ok = (isinstance(spec, dict) and isinstance(spec.get("name"), str) and isinstance(spec.get("args"), list)
                  and isinstance(spec.get("done"), list) and spec["done"])
            out.append((f, spec if ok else None))
    return out


def finished(spec):
    return all(os.path.exists(os.path.join(ROOT, d)) for d in spec["done"])


def stop_files(spec):
    return [os.path.join(ROOT, d, "STOP") for d in spec.get("stop_dirs", [])]


def busy(m):
    me = os.getpid()
    for pid, name, cmd in m.processes():
        if pid == me:
            continue
        if name == "llama-server.exe":
            return f"llama-server.exe {pid}"
        if name.startswith("python"):
            for b in BUSY:
                if b in cmd:
                    return f"{name} {pid} ({b})"
    return None


def move(f, folder, dry, counts=None):
    if not dry:
        os.replace(jpath("queue", f), jpath(folder, f))
        if counts is not None:
            counts.pop(f, None)


def step(m, st, dry=False):
    """One check. Updates the state dict st and returns the log lines."""
    out = []
    free, commit, limit = m.memory()
    st["memory"] = {"free_gib": round(free, 2), "committed_gib": round(commit, 2), "limit_gib": round(limit, 2)}
    counts = st.setdefault("counts", {})
    job = st.get("job")
    if job and m.alive(job["pid"], job["created"]):
        job["low_free_gib"] = round(min(job.get("low_free_gib", free), free), 2)
        job["high_commit_gib"] = round(max(job.get("high_commit_gib", commit), commit), 2)
        if lt(free, MIN_FREE_GIB) or gt(commit, MAX_COMMIT_GIB):
            spec = read_json(jpath("queue", job["file"])) or {}
            for f in stop_files(spec):
                if os.path.isdir(os.path.dirname(f)) and not os.path.exists(f):
                    out.append(f"{job['name']}: RAM hard line (free {free:.2f} GiB, committed {commit:.1f} GiB), "
                               f"STOP written in {os.path.dirname(f)}")
                    if not dry:
                        open(f, "w").close()
        return out
    if job:
        st["job"] = None
        f, name = job["file"], job["name"]
        spec = read_json(jpath("queue", f))
        ran = m.now() - job["started"]
        c = counts.setdefault(f, {"attempts": 0, "quick": 0})
        lows = f"low free {job.get('low_free_gib')} GiB, high committed {job.get('high_commit_gib')} GiB"
        if spec is None or not isinstance(spec.get("done"), list):
            out.append(f"{name}: exited and its job file is gone or broken; left as it is")
        elif finished(spec):
            out.append(f"{name}: finished after {ran / 3600:.2f} h ({lows})")
            move(f, "done", dry, counts)
        elif any(os.path.exists(s) for s in stop_files(spec)):
            out.append(f"{name}: stopped by a STOP file after {ran / 3600:.2f} h; moved to held for a person")
            move(f, "held", dry, counts)
        elif lt(ran, QUICK_EXIT_S):
            c["quick"] += 1
            if ge(c["quick"], MAX_QUICK):
                out.append(f"{name}: exited within {QUICK_EXIT_S} s without finishing {c['quick']} times; failed")
                move(f, "failed", dry, counts)
            else:
                st["retry_after"] = m.now() + QUICK_RETRY_S
                out.append(f"{name}: exited after {ran:.0f} s without finishing; trying again in "
                           f"{QUICK_RETRY_S // 60} minutes")
        else:
            c["attempts"] += 1
            if ge(c["attempts"], int(spec.get("max_attempts", 4))):
                out.append(f"{name}: stopped without finishing {c['attempts']} times; failed")
                move(f, "failed", dry, counts)
            else:
                out.append(f"{name}: stopped after {ran / 3600:.2f} h without finishing ({lows}); "
                           f"it starts again from its saved work")
    if lt(m.now(), st.get("retry_after", 0)):
        return out
    q = queue()
    while q and (q[0][1] is None or finished(q[0][1])):
        f, spec = q.pop(0)
        if spec is None:
            out.append(f"{f}: not a valid job file; moved to failed")
            move(f, "failed", dry)
        else:
            out.append(f"{spec['name']}: already finished; moved to done")
            move(f, "done", dry)
    if not q:
        st["free_checks"], st["busy"] = 0, None
        return out
    f, spec = q[0]
    if any(os.path.exists(s) for s in stop_files(spec)):
        out.append(f"{spec['name']}: a STOP file is in its folder; moved to held for a person, not started")
        move(f, "held", dry)
        return out
    why = busy(m)
    if why:
        if why != st.get("busy"):
            out.append(f"waiting: the GPU is busy ({why})")
        st["free_checks"], st["busy"] = 0, why
        return out
    st["busy"] = None
    if dry:
        out.append(f"{spec['name']}: the GPU is free; it would start once it stays free for {FREE_CHECKS} checks: "
                   f"{' '.join(spec['args'])}")
        return out
    st["free_checks"] = st.get("free_checks", 0) + 1
    if lt(st["free_checks"], FREE_CHECKS):
        return out
    st["free_checks"] = 0
    c = counts.setdefault(f, {"attempts": 0, "quick": 0})
    pid, created = m.start(spec["args"], jpath("logs", spec["name"] + ".log"))
    st["job"] = {"file": f, "name": spec["name"], "pid": pid, "created": created, "started": m.now()}
    out.append(f"{spec['name']}: started, pid {pid} (attempt {c['attempts'] + 1}): {' '.join(spec['args'])}")
    return out


def status(m, st, lines):
    q = queue()
    job = st.get("job")
    return {"time": utc(m.now()), "watcher_pid": os.getpid(), "memory": st.get("memory"),
            "job": job and dict(job, started=utc(job["started"]), hours=round((m.now() - job["started"]) / 3600, 2)),
            "busy": st.get("busy"), "queue": [s["name"] if s else f for f, s in q],
            "held": sorted(os.listdir(jpath("held"))), "failed": sorted(os.listdir(jpath("failed"))),
            "last": lines[-5:] or st.get("last", [])}


def log(m, lines):
    with open(jpath("watcher.log"), "a", encoding="utf-8") as fh:
        for line in lines:
            fh.write(f"[{utc(m.now())}] {line}\n")


def holder(m):
    lock = read_json(jpath("watcher.lock"))
    if lock and lock.get("pid") != os.getpid() and m.alive(lock.get("pid", 0), lock.get("created", 0)):
        return lock["pid"]
    return None


def run(m, once=False, dry=False, every=EVERY_S):
    for d in FOLDERS:
        os.makedirs(jpath(d), exist_ok=True)
    other = holder(m)
    if other and not dry:
        print(f"a watcher is already running (pid {other}); nothing to do")
        return 0
    if not dry:
        write_json(jpath("watcher.lock"), {"pid": os.getpid(), "created": m.created(os.getpid()),
                                           "since": utc(m.now())})
        log(m, [f"watcher started, pid {os.getpid()}"])
    while True:
        st = read_json(jpath("state.json")) or {}
        try:
            lines = step(m, st, dry)
        except Exception:  # keep watching whatever one check hit
            lines = ["check failed: " + traceback.format_exc().strip().splitlines()[-1]]
        if dry:
            for line in lines or ["nothing to do now"]:
                print(line)
            print(json.dumps(status(m, st, lines), indent=1))
            return 0
        if lines:
            st["last"] = (st.get("last", []) + [f"{utc(m.now())} {x}" for x in lines])[-5:]
        write_json(jpath("state.json"), st)
        write_json(jpath("status.json"), status(m, st, lines))
        log(m, lines)
        if once:
            return 0
        time.sleep(every)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--once", action="store_true", help="one check, then exit")
    ap.add_argument("--dry-run", action="store_true", help="change nothing; print what it would do")
    ap.add_argument("--status", action="store_true", help="print status.json and the queue")
    ap.add_argument("--every", type=int, default=EVERY_S, help="seconds between checks")
    a = ap.parse_args()
    if a.status:
        print(json.dumps(read_json(jpath("status.json")), indent=1))
        for d in ("queue", "held", "failed", "done"):
            names = sorted(os.listdir(jpath(d))) if os.path.isdir(jpath(d)) else []
            print(f"{d}: {', '.join(names) or '-'}")
        return 0
    return run(Machine(), once=a.once, dry=a.dry_run, every=a.every)


if __name__ == "__main__":
    sys.exit(main())
