"""Tests for scripts/job_watcher.py with a fake machine (no processes are started)."""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
import job_watcher as W  # noqa: E402


class Fake:
    def __init__(self):
        self.t = 1_000_000.0
        self.procs = []          # (pid, name, cmdline) of other processes
        self.live = {}           # pid -> created
        self.mem = (20.0, 50.0, 74.8)
        self.started = []
        self.next_pid = 100

    def now(self):
        return self.t

    def processes(self):
        return list(self.procs)

    def alive(self, pid, created):
        return self.live.get(pid) == created

    def created(self, pid):
        return self.live.get(pid, 0.0)

    def memory(self):
        return self.mem

    def start(self, args, log_path):
        self.next_pid += 1
        self.live[self.next_pid] = self.t
        self.started.append((self.next_pid, list(args), log_path))
        return self.next_pid, self.t

    def exit(self, pid):
        self.live.pop(pid, None)


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(W, "ROOT", str(tmp_path))
    monkeypatch.setattr(W, "JOBS", str(tmp_path / "results" / "jobs"))
    for d in W.FOLDERS:
        os.makedirs(W.jpath(d))
    return tmp_path, Fake()


def add_job(root, fname, name, done="out/{n}/DONE", stop=True, **kw):
    spec = {"name": name, "args": ["scripts/x.py", name], "done": [done.format(n=name)]}
    if stop:
        os.makedirs(root / "out" / name, exist_ok=True)
        spec["stop_dirs"] = [f"out/{name}"]
    spec.update(kw)
    with open(W.jpath("queue", fname), "w") as fh:
        json.dump(spec, fh)
    return spec


def finish(root, name):
    (root / "out" / name / "DONE").write_text("")


def checks(m, st, n, every=60):
    lines = []
    for _ in range(n):
        lines += W.step(m, st)
        m.t += every
    return lines


def test_starts_first_job_after_three_free_checks(env):
    root, m = env
    add_job(root, "02-b.json", "b")
    add_job(root, "01-a.json", "a")
    st = {}
    checks(m, st, 2)
    assert m.started == []
    checks(m, st, 1)
    assert [s[1][1] for s in m.started] == ["a"]
    assert m.started[0][2] == W.jpath("logs", "a.log")
    assert st["job"]["name"] == "a"


def test_busy_gpu_resets_the_count(env):
    root, m = env
    add_job(root, "01-a.json", "a")
    st = {}
    checks(m, st, 2)
    m.procs = [(55, "llama-server.exe", "llama-server.exe -m x")]
    lines = checks(m, st, 3)
    assert m.started == [] and "busy" in lines[0]
    m.procs = [(56, "pythonw.exe", r"pythonw.exe scratch\sweep_runner2.py")]
    checks(m, st, 3)
    assert m.started == []
    m.procs = [(57, "python.exe", "python.exe other_tool.py"), (58, "comfyui.exe", "lora_scale_check.py")]
    checks(m, st, 3)
    assert len(m.started) == 1


def test_finished_job_moves_to_done_and_next_starts(env):
    root, m = env
    add_job(root, "01-a.json", "a")
    add_job(root, "02-b.json", "b")
    st = {}
    checks(m, st, 3)
    pid = m.started[0][0]
    m.t += 7200
    finish(root, "a")
    m.exit(pid)
    lines = checks(m, st, 1)
    assert "a: finished" in lines[0]
    assert os.path.exists(W.jpath("done", "01-a.json"))
    checks(m, st, 3)
    assert [s[1][1] for s in m.started] == ["a", "b"]


def test_dead_job_restarts_then_fails(env):
    root, m = env
    add_job(root, "01-a.json", "a", max_attempts=2)
    st = {}
    checks(m, st, 3)
    m.t += 3600
    m.exit(m.started[-1][0])
    lines = checks(m, st, 4)
    assert "starts again" in lines[0]
    assert len(m.started) == 2
    m.t += 3600
    m.exit(m.started[-1][0])
    lines = checks(m, st, 4)
    assert "failed" in lines[0]
    assert os.path.exists(W.jpath("failed", "01-a.json"))
    assert len(m.started) == 2


def test_quick_exit_waits_and_does_not_count(env):
    root, m = env
    add_job(root, "01-a.json", "a", max_attempts=1)
    st = {}
    checks(m, st, 3)
    m.exit(m.started[-1][0])
    lines = checks(m, st, 1)
    assert "trying again in 10 minutes" in lines[0]
    checks(m, st, 8)
    assert len(m.started) == 1
    checks(m, st, 4)
    assert len(m.started) == 2
    assert os.path.exists(W.jpath("queue", "01-a.json"))


def test_ram_hard_line_writes_stop_and_holds(env):
    root, m = env
    add_job(root, "01-a.json", "a")
    st = {}
    checks(m, st, 3)
    m.mem = (1.2, 60.0, 74.8)
    lines = checks(m, st, 1)
    assert "RAM hard line" in lines[0]
    assert (root / "out" / "a" / "STOP").exists()
    assert m.live  # nothing killed
    m.mem = (20.0, 50.0, 74.8)
    m.t += 600
    m.exit(m.started[-1][0])
    lines = checks(m, st, 4)
    assert "moved to held" in lines[0]
    assert os.path.exists(W.jpath("held", "01-a.json"))
    assert len(m.started) == 1


def test_commit_line_and_low_points(env):
    root, m = env
    add_job(root, "01-a.json", "a")
    st = {}
    checks(m, st, 3)
    m.mem = (9.0, 74.5, 74.8)
    checks(m, st, 1)
    assert (root / "out" / "a" / "STOP").exists()
    assert st["job"]["low_free_gib"] == 9.0 and st["job"]["high_commit_gb"] == 74.5


def test_already_finished_and_invalid_jobs(env):
    root, m = env
    add_job(root, "01-a.json", "a")
    finish(root, "a")
    with open(W.jpath("queue", "02-bad.json"), "w") as fh:
        fh.write('{"name": "bad"}')
    add_job(root, "03-c.json", "c")
    st = {}
    lines = checks(m, st, 3)
    assert os.path.exists(W.jpath("done", "01-a.json"))
    assert os.path.exists(W.jpath("failed", "02-bad.json"))
    assert [s[1][1] for s in m.started] == ["c"]
    assert any("already finished" in x for x in lines)


def test_stop_present_before_start_holds(env):
    root, m = env
    add_job(root, "01-a.json", "a")
    (root / "out" / "a" / "STOP").write_text("")
    st = {}
    checks(m, st, 3)
    assert m.started == []
    assert os.path.exists(W.jpath("held", "01-a.json"))


def test_dry_run_changes_nothing(env):
    root, m = env
    add_job(root, "01-a.json", "a")
    add_job(root, "02-b.json", "b")
    finish(root, "a")
    st = {}
    lines = W.step(m, st, dry=True)
    assert m.started == []
    assert os.path.exists(W.jpath("queue", "01-a.json"))
    assert any("would move to done" in x for x in lines) and not any("moved to" in x for x in lines)
    assert any("would start" in x for x in lines)


def test_restart_adopts_running_job(env):
    root, m = env
    add_job(root, "01-a.json", "a")
    st = {}
    checks(m, st, 3)
    saved = json.loads(json.dumps(st))
    checks(m, saved, 5)
    assert len(m.started) == 1


def test_run_once_writes_status_and_respects_lock(env, capsys):
    root, m = env
    add_job(root, "01-a.json", "a")
    m.live[4242] = 5.0
    W.write_json(W.jpath("watcher.lock"), {"pid": 4242, "created": 5.0})
    assert W.run(m, once=True) == 0
    assert "already running" in capsys.readouterr().out
    assert not os.path.exists(W.jpath("status.json"))
    m.live.pop(4242)
    assert W.run(m, once=True) == 0
    st = W.read_json(W.jpath("status.json"))
    assert st["queue"] == ["a"] and st["memory"]["free_gib"] == 20.0
    assert "watcher started" in open(W.jpath("watcher.log")).read()


def test_stop_wins_over_done_files(env):
    root, m = env
    add_job(root, "01-a.json", "a")
    add_job(root, "02-b.json", "b")
    st = {}
    checks(m, st, 3)
    m.t += 3600
    (root / "out" / "a" / "STOP").write_text("")
    finish(root, "a")  # a stopped job can still write its summary
    m.exit(m.started[-1][0])
    lines = checks(m, st, 1)
    assert "STOP" in lines[0] and "held" in lines[0]
    assert os.path.exists(W.jpath("held", "01-a.json"))
    (root / "out" / "b" / "STOP").write_text("")
    finish(root, "b")
    checks(m, st, 3)
    assert os.path.exists(W.jpath("held", "02-b.json"))
    assert len(m.started) == 1
