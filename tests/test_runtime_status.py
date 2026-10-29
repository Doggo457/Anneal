"""Tests for anneal/runtime.py Run.write_status: a refused swap (Windows, status.json open elsewhere) is retried,
a lasting refusal still raises, and threads writing at once always leave whole JSON.

  python tests/test_runtime_status.py        (or: python -m pytest tests/test_runtime_status.py)
"""

import json
import os
import sys
import tempfile
import threading
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from anneal import runtime  # noqa: E402

REAL_REPLACE = os.replace


def refusing_replace(times):
    """os.replace that is refused `times` times, then works; returns (fake, calls)."""
    calls = []

    def fake(src, dst):
        calls.append(dst)
        if len(calls) <= times:
            raise PermissionError(5, "Access is denied", dst)
        return REAL_REPLACE(src, dst)
    return fake, calls


def read_status(run):
    with open(run.status_path, encoding="utf-8") as fh:
        return json.load(fh)


def test_refused_swap_is_retried():
    with tempfile.TemporaryDirectory() as d:
        run = runtime.Run(d)
        fake, calls = refusing_replace(3)
        with mock.patch.object(runtime.os, "replace", fake), mock.patch.object(runtime.time, "sleep", lambda s: None):
            run.write_status(state="running", step=7)
        assert len(calls) == 4, calls
        assert read_status(run)["step"] == 7


def test_lasting_refusal_raises():
    with tempfile.TemporaryDirectory() as d:
        run = runtime.Run(d)
        run.write_status(state="running", step=1)
        fake, calls = refusing_replace(10 ** 6)
        with mock.patch.object(runtime.os, "replace", fake), mock.patch.object(runtime.time, "sleep", lambda s: None):
            try:
                run.write_status(state="running", step=2)
            except PermissionError:
                pass
            else:
                raise AssertionError("a refusal that never ends must raise")
        assert len(calls) == runtime.Run.STATUS_REPLACE_TRIES
        assert read_status(run)["step"] == 1  # the last good status stays


def test_other_errors_are_not_retried():
    with tempfile.TemporaryDirectory() as d:
        run = runtime.Run(d)
        calls = []

        def fake(src, dst):
            calls.append(dst)
            raise FileNotFoundError(2, "gone", src)
        with mock.patch.object(runtime.os, "replace", fake):
            try:
                run.write_status(state="running")
            except FileNotFoundError:
                pass
            else:
                raise AssertionError("only PermissionError is retried")
        assert len(calls) == 1


def test_threads_leave_whole_json():
    with tempfile.TemporaryDirectory() as d:
        run = runtime.Run(d)
        errors = []

        def writer(k):
            try:
                for i in range(200):
                    run.write_status(state="running", step=i, writer=k, detail="x" * (k * 50 + i % 7))
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)
        threads = [threading.Thread(target=writer, args=(k,)) for k in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors, errors
        st = read_status(run)
        assert st["step"] == 199 and st["writer"] in range(4)
        assert not os.path.exists(run.status_path + ".tmp")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print("ok", t.__name__)
    print(f"{len(tests)}/{len(tests)} passed")
