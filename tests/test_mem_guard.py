"""Tests for anneal/mem_guard.py: two low readings in a row end training as paused (status, STOP, trace, metric), one
low reading does not, a failed reading never stops the guard, and no status write can follow the trip.

  python tests/test_mem_guard.py        (or: python -m pytest tests/test_mem_guard.py)
"""
import json
import os
import sys
import tempfile
import threading
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from anneal import mem_guard  # noqa: E402
from anneal.runtime import Run  # noqa: E402


def readings(*values):
    """read_sys that returns these available-memory values in turn, then repeats the last; an Exception value is
    raised instead."""
    seq = list(values)

    def read():
        v = seq.pop(0) if len(seq) > 1 else seq[0]
        if isinstance(v, Exception):
            raise v
        return {"avail_gib": v, "commit_gib": 60.0}
    return read


def guard(run, read, **kw):
    ended = []
    status_lock_held = []

    def end(code):
        ended.append(code)
        status_lock_held.append(not run._status_lock.acquire(blocking=False))
    g = mem_guard.MemGuard(run, floor_gib=1.0, interval=0.001, read_sys=read, end=end,
                           extra=lambda: {"micro": 5, "tokens": 11280}, **kw)
    return g, ended, status_lock_held


def lines(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(x) for x in fh if x.strip()]


def test_two_low_readings_end_training_as_paused():
    with tempfile.TemporaryDirectory() as d:
        run = Run(d)
        run.write_status(state="training", step=175)
        g, ended, held = guard(run, readings(8.0, 6.0, 0.9, 0.4))
        g.start()
        g.join(5)
        assert not g.is_alive()
        assert ended == [3]
        assert held == [True], "a status write could still land after the trip"
        st = json.load(open(run.status_path, encoding="utf-8"))
        assert st["state"] == "paused" and st["paused_by"] == "mem_guard" and st["step"] == 175
        assert "0.4 GiB" in st["error"]
        assert os.path.exists(run.stop_path)
        trace = lines(g.trace_path)
        trip = [x for x in trace if x.get("event") == "trip"]
        assert len(trip) == 1
        assert trip[0]["reading"]["avail_gib"] == 0.4 and trip[0]["reading"]["tokens"] == 11280
        assert [h["avail_gib"] for h in trip[0]["history"]] == [8.0, 6.0, 0.9, 0.4]
        assert any(x.get("event") == "watch" for x in trace)  # 0.9 is under the watch level (floor + 2)
        m = lines(run.metrics_path)
        assert m[-1]["event"] == "mem_guard_trip" and m[-1]["avail_gib"] == 0.4
        assert g.min_avail == 0.4


def test_one_low_reading_does_not_trip():
    with tempfile.TemporaryDirectory() as d:
        run = Run(d)
        g, ended, _ = guard(run, readings(0.5, 5.0, 0.5, 5.0, 5.0))
        g.start()
        while len(g.history) < 20 and g.is_alive():
            time.sleep(0.001)
        g.close()
        g.join(5)
        assert ended == []
        assert not os.path.exists(run.stop_path)
        assert g.min_avail == 0.5


def test_failed_reading_does_not_stop_the_guard():
    with tempfile.TemporaryDirectory() as d:
        run = Run(d)
        g, ended, _ = guard(run, readings(OSError("counter busy"), 0.2, 0.2))
        g.start()
        g.join(5)
        assert ended == [3] and g.errors == 1
        assert any(x.get("event") == "read_error" for x in lines(g.trace_path))


def test_quiet_readings_are_thinned_but_watch_readings_are_all_kept():
    with tempfile.TemporaryDirectory() as d:
        run = Run(d)
        vals = [10.0] * 30 + [2.5] * 5 + [0.5, 0.5]
        g, ended, _ = guard(run, readings(*vals))
        g.start()
        g.join(5)
        plain = [x for x in lines(g.trace_path) if "avail_gib" in x and "event" not in x]
        assert sum(1 for x in plain if x["avail_gib"] == 10.0) == 3  # every 10th of 30
        assert sum(1 for x in plain if x["avail_gib"] == 2.5) == 5   # under 3 GiB: all of them
        assert ended == [3]


def gpu_readings(*rows):
    """read_sys giving (avail_gib, training's dedicated, shared) in turn, then repeating the last, with fresh GPU
    reading times."""
    seq = list(rows)

    def read():
        a, ded, shr = seq.pop(0) if len(seq) > 1 else seq[0]
        return {"avail_gib": a, "gpu_train_ded_gib": ded, "gpu_train_shr_gib": shr, "gpu_read_t": time.time()}
    return read


def test_gpu_cap_trips_before_windows_runs_out():
    """26 Nov step 176: 42.2 GiB on the GPU with 3.3 GiB free, two readings in a row, ends training as paused."""
    with tempfile.TemporaryDirectory() as d:
        run = Run(d)
        run.write_status(state="training", step=175)
        g, ended, held = guard(run, gpu_readings((7.6, 30.14, 9.9), (4.6, 30.14, 10.42), (3.3, 30.14, 12.08)),
                               gpu_cap_gib=42.0, gpu_free_gib=6.0)
        g.start()
        g.join(5)
        assert ended == [3] and held == [True]
        st = json.load(open(run.status_path, encoding="utf-8"))
        assert st["state"] == "paused" and "GPU memory 42.22 GiB" in st["error"] and "3.3 GiB" in st["error"]
        trip = [x for x in lines(g.trace_path) if x.get("event") == "trip"]
        assert len(trip) == 1 and trip[0]["rule"] == "gpu"
        assert lines(run.metrics_path)[-1]["rule"] == "gpu"


def test_gpu_cap_needs_windows_short_of_memory_and_a_fresh_reading():
    with tempfile.TemporaryDirectory() as d:
        run = Run(d)
        g, ended, _ = guard(run, gpu_readings((8.0, 30.14, 13.0)), gpu_cap_gib=42.0, gpu_free_gib=6.0)
        g.start()
        while len(g.history) < 20 and g.is_alive():
            time.sleep(0.001)
        g.close()
        g.join(5)
        assert ended == [], "plenty of free memory: a high GPU figure alone must not trip"

        def stale():
            return {"avail_gib": 3.0, "gpu_train_ded_gib": 30.14, "gpu_train_shr_gib": 13.0,
                    "gpu_read_t": time.time() - 60}
        g2, ended2, _ = guard(run, stale, gpu_cap_gib=42.0, gpu_free_gib=6.0)
        g2.start()
        while len(g2.history) < 20 and g2.is_alive():
            time.sleep(0.001)
        g2.close()
        g2.join(5)
        assert ended2 == [], "a GPU reading a minute old must not be used"


def test_no_gpu_cap_by_default_in_the_class():
    with tempfile.TemporaryDirectory() as d:
        run = Run(d)
        g, ended, _ = guard(run, gpu_readings((3.0, 30.14, 15.0)))
        g.start()
        while len(g.history) < 20 and g.is_alive():
            time.sleep(0.001)
        g.close()
        g.join(5)
        assert ended == []


def test_zero_floor_means_off():
    with tempfile.TemporaryDirectory() as d:
        assert mem_guard.start(Run(d), 0) is None
        assert mem_guard.start(Run(d), None) is None


def test_status_writes_work_again_when_no_trip():
    with tempfile.TemporaryDirectory() as d:
        run = Run(d)
        g, _, _ = guard(run, readings(9.0))
        g.start()
        g.close()
        g.join(5)
        done = threading.Event()
        threading.Thread(target=lambda: (run.write_status(state="finished"), done.set()), daemon=True).start()
        assert done.wait(5)


def test_reads_this_machine():
    r = mem_guard.read_system()
    assert r["avail_gib"] > 0


def test_train_longrun_starts_and_closes_the_guard():
    src = open(os.path.join(ROOT, "scripts", "train_longrun.py"), encoding="utf-8").read()
    assert 'mem_guard.start(run, cfg.get("mem_guard_free_gib", 2.0)' in src
    assert 'gpu_cap_gib=cfg.get("mem_guard_gpu_gib", 42.0)' in src
    assert src.index("mem_guard.start(") < src.index("Loader.from_pretrained(")  # covers loading and the warm-up
    assert src.index("trainer.train(resume_from_checkpoint=resume)") < src.index("guard.close()")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
