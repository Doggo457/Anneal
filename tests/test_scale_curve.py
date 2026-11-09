"""Tests for scripts/scale_curve.py with a fake lora_scale_check.py: strengths in order, saved work skipped, a pause
passed on as STOP and cleared for the resume, a failing strength, the curve file, and the refusal while loop.py runs.

  python tests/test_scale_curve.py        (or: python -m pytest tests/test_scale_curve.py)
"""

import json
import os
import shutil
import sys
import tempfile
import threading
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import scale_curve as C  # noqa: E402

FAKE = r'''
import argparse, json, os, sys, time
ap = argparse.ArgumentParser()
ap.add_argument("--round", type=int); ap.add_argument("--scale", type=float); ap.add_argument("--n", type=int)
ap.add_argument("--part")
a = ap.parse_args()
name = f"{a.scale:.2f}" if round(a.scale, 2) == a.scale else f"{a.scale:g}"
out = os.path.join(os.environ["FAKE_OUT"], f"r{a.round:03d}_x{name}")
os.makedirs(out, exist_ok=True)
with open(os.path.join(os.environ["FAKE_OUT"], "calls.txt"), "a") as fh:
    fh.write(f"{a.scale} {a.n} {a.part}\n")
if os.environ.get("FAKE_FAIL") == name:
    sys.exit(3)
if os.environ.get("FAKE_WAIT") == name:
    open(os.path.join(os.environ["FAKE_OUT"], "waiting"), "w").close()
    while not os.path.exists(os.path.join(out, "STOP")):
        time.sleep(0.05)
    sys.exit(0)  # stopped cleanly: exits 0, as lora_scale_check does
p = int(40 + 10 * a.scale)
json.dump({"round_scale": 0.25, "gate": {"all": {"n": 60, "part": {"pass": p, "reasoning_mean": 1, "budget_hits": 2},
                                                "base": {"pass": 40}, "full": {"pass": 42},
                                                "part_vs_base": {"wins": 1}, "part_vs_full": {"wins": 2}}},
           "bfcl": {"all": {"part": 90, "base": 92, "part_vs_base": {"wins": 3}}},
           "loss": {"swe": {"part": 0.4, "base": 0.47, "full": 0.44, "part_rel_to_base": -0.1}}},
          open(os.path.join(out, "summary.json"), "w"))
'''


class World:
    def __init__(self):
        self.tmp = tempfile.mkdtemp()
        self.out = os.path.join(self.tmp, "lora_scale")
        os.makedirs(self.out)
        self.check = os.path.join(self.tmp, "fake_check.py")
        open(self.check, "w").write(FAKE)
        self.saved = (C.CHECK, C.OUT_ROOT, C.LOOP_PID, C.EVERY_S, sys.argv, sys.stdout, sys.stderr)
        C.CHECK, C.OUT_ROOT, C.LOOP_PID, C.EVERY_S = self.check, self.out, os.path.join(self.tmp, "loop.pid"), 0.05
        os.environ["FAKE_OUT"] = self.out
        self.run_dir = os.path.join(self.tmp, "run")

    def main(self, *argv):
        sys.argv = ["scale_curve.py", "--run-dir", self.run_dir, *argv]
        try:
            return C.main()
        finally:
            sys.stdout, sys.stderr = self.saved[5], self.saved[6]

    def calls(self):
        p = os.path.join(self.out, "calls.txt")
        return open(p).read().split("\n")[:-1] if os.path.exists(p) else []

    def status(self):
        return json.load(open(os.path.join(self.run_dir, "status.json")))

    def close(self):
        C.CHECK, C.OUT_ROOT, C.LOOP_PID, C.EVERY_S, sys.argv, sys.stdout, sys.stderr = self.saved
        for k in ("FAKE_OUT", "FAKE_FAIL", "FAKE_WAIT"):
            os.environ.pop(k, None)
        shutil.rmtree(self.tmp)


def test_all_strengths_then_curve_and_skip():
    w = World()
    try:
        assert w.main("--round", "4", "--scales", "0.25,0.125,0.5", "--resume") == 0
        assert w.calls() == ["0.25 60 all", "0.125 60 all", "0.5 60 all"]  # in the order given
        st = w.status()
        assert st["state"] == "finished" and st["step"] == 3 and st["max_steps"] == 3
        c = json.load(open(os.path.join(w.run_dir, "curve.json")))
        assert [r["scale"] for r in c["strengths"]] == [0.125, 0.25, 0.5]
        assert [r["gate_all"] for r in c["strengths"]] == [41, 42, 45]
        assert c["reference"]["gate_all_base"] == 40 and c["reference"]["gate_all_full"] == 42
        assert c["reference"]["bfcl_base"] == 92 and c["reference"]["loss_swe_round"] == 0.44
        assert os.path.isdir(os.path.join(w.out, "r004_x0.125"))
        assert w.main("--round", "4", "--scales", "0.25,0.125,0.5") == 0  # all done before: nothing run again
        assert len(w.calls()) == 3
    finally:
        w.close()


def test_pause_passes_stop_on_and_resume_finishes():
    w = World()
    try:
        os.environ["FAKE_WAIT"] = "0.125"
        done = {}
        t = threading.Thread(target=lambda: done.update(rc=w.main("--round", "4", "--scales", "0.25,0.125,0.5")))
        t.start()
        for _ in range(400):
            if os.path.exists(os.path.join(w.out, "waiting")):
                break
            time.sleep(0.05)
        open(os.path.join(w.run_dir, "STOP"), "w").close()  # the Training page's pause
        t.join(30)
        assert not t.is_alive() and done["rc"] == 0
        st = w.status()
        assert st["state"] == "paused" and "Training page" in st["detail"], st
        assert not os.path.exists(os.path.join(w.out, "r004_x0.125", "STOP"))  # cleared for the resume
        assert not os.path.exists(os.path.join(w.run_dir, "done_x0.125.json"))
        assert w.calls() == ["0.25 60 all", "0.125 60 all"]  # 0.5 not started
        os.environ.pop("FAKE_WAIT")
        assert w.main("--round", "4", "--scales", "0.25,0.125,0.5", "--resume") == 0  # STOP removed at start
        assert w.calls()[2:] == ["0.125 60 all", "0.5 60 all"] and w.status()["state"] == "finished"
    finally:
        w.close()


def test_failing_strength_and_refusal_while_loop_runs():
    w = World()
    try:
        os.environ["FAKE_FAIL"] = "0.125"
        assert w.main("--round", "4", "--scales", "0.25,0.125,0.5") == 1
        assert w.status()["state"] == "failed" and w.calls() == ["0.25 60 all", "0.125 60 all"]
        os.environ.pop("FAKE_FAIL")
        open(C.LOOP_PID, "w").write(str(os.getpid()))  # a live pid: the loop "runs"
        assert w.main("--round", "4", "--scales", "0.25,0.125,0.5") == 1
        assert "loop.py is running" in w.status()["detail"] and len(w.calls()) == 2
        try:
            w.main("--round", "4", "--scales", "0.25,1.0")
            raise AssertionError("should have refused")
        except SystemExit as exc:
            assert "between 0 and 1" in str(exc)
    finally:
        w.close()


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
