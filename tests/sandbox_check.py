"""Acceptance test for anneal/sandbox.py (AppContainer + Job Object): what model-written code can and can't do.

Every probe runs in the sandbox and reports whether its operation succeeded; a denial probe passes when the
operation fails. The whole battery runs once in sequence, then 8 copies of it run in parallel, then 100 trivial runs
measure the per-run overhead. Writes <run dir>\\summary.json and a markdown table to run.log.

  python tests\\sandbox_check.py [--run-dir DIR]
"""

import argparse
import json
import os
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # C:\AI\training
from anneal import sandbox  # noqa: E402
from anneal.runtime import RESULTS_DIR, Run  # noqa: E402

PROBE = r'''
import json, os, sys
def probe():
{body}
try:
    detail = probe()
    res = {{"succeeded": True, "detail": repr(detail)[:300]}}
except BaseException as e:
    res = {{"succeeded": False, "detail": (type(e).__name__ + ": " + str(e))[:300]}}
print("PROBE " + json.dumps(res))
'''


def probe_code(body):
    return PROBE.format(body="\n".join("    " + ln for ln in body.strip("\n").splitlines()))


def sandbox_pythons():
    import psutil
    out = []
    for p in psutil.process_iter(["exe"]):
        try:
            if (p.info["exe"] or "").lower() == sandbox.PYTHON_EXE.lower():
                out.append(p.pid)
        except psutil.Error:
            pass
    return out


def make_probes(paths):
    """(name, what it tries, expectation, body or callable). expectation: "deny" (must fail), "allow" (must
    succeed), "info" (reported only)."""
    P = paths
    return [
        ("read_training_file", f"read {P['training_probe']}", "deny", f"return open({P['training_probe']!r}).read()"),
        ("read_user_profile_file", f"read {P['profile_probe']}", "deny", f"return open({P['profile_probe']!r}).read()"),
        ("list_user_profile", f"list {P['profile']}", "deny", f"return os.listdir({P['profile']!r})[:5]"),
        ("read_portal_file", r"read C:\AI\portal\server.py", "deny", r"return open(r'C:\AI\portal\server.py').read(100)"),
        ("list_portal", r"list C:\AI\portal", "deny", r"return os.listdir(r'C:\AI\portal')"),
        ("read_memory_notes", f"read {P['memory']}", "deny", f"return open({P['memory']!r}).read(100)"),
        ("write_training", r"create C:\AI\training\sandbox_write_probe.txt", "deny",
         r"open(r'C:\AI\training\sandbox_write_probe.txt', 'w').write('x')"),
        ("write_interpreter_tree", r"create a file in the sandbox interpreter folder", "deny",
         f"open({os.path.join(sandbox.PYTHON_DIR, 'write_probe.txt')!r}, 'w').write('x')"),
        ("write_users_public", r"create C:\Users\Public\anneal_write_probe.txt", "deny",
         r"open(r'C:\Users\Public\anneal_write_probe.txt', 'w').write('x')"),
        ("write_user_temp", f"create a file in {P['user_temp']}", "deny",
         f"open({os.path.join(P['user_temp'], 'anneal_write_probe.txt')!r}, 'w').write('x')"),
        ("write_container_folder", r"create a file in the container's own %LOCALAPPDATA% (AC) folder", "deny",
         "open(os.path.join(os.environ['LOCALAPPDATA'], 'probe.txt'), 'w').write('x')"),
        ("write_hkcu", r"create HKCU\Software\AnnealSandboxProbe and set a value", "deny",
         "import winreg\nk = winreg.CreateKey(winreg.HKEY_CURRENT_USER, r'Software\\AnnealSandboxProbe')\n"
         "winreg.SetValueEx(k, 'x', 0, winreg.REG_SZ, 'y')"),
        ("write_hkcu_environment", r"set a value in HKCU\Environment", "deny",
         "import winreg\nk = winreg.OpenKey(winreg.HKEY_CURRENT_USER, 'Environment', 0, winreg.KEY_SET_VALUE)\n"
         "winreg.SetValueEx(k, 'ANNEAL_PROBE', 0, winreg.REG_SZ, 'y')"),
        ("tcp_loopback_8080", "TCP connect 127.0.0.1:8080 (the portal)", "deny",
         "import socket\ns = socket.create_connection(('127.0.0.1', 8080), timeout=5)\nreturn s.getpeername()"),
        ("tcp_internet_443", "TCP connect 1.1.1.1:443", "deny",
         "import socket\ns = socket.create_connection(('1.1.1.1', 443), timeout=5)\nreturn s.getpeername()"),
        ("dns_lookup", "getaddrinfo('python.org')", "deny",
         "import socket\nreturn socket.getaddrinfo('python.org', 443)[:2]"),
        ("dns_lookup_localhost", "getaddrinfo('localhost') (resolved locally, no network)", "info",
         "import socket\nreturn socket.getaddrinfo('localhost', 80)[:1]"),
        ("udp_send", "UDP datagram to 1.1.1.1:53", "deny",
         "import socket\ns = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)\ns.settimeout(3)\n"
         "s.sendto(b'\\x00' * 12, ('1.1.1.1', 53))\nreturn s.recvfrom(512)"),
        ("spawn_cmd", "subprocess.run(['cmd.exe', '/c', 'echo hi'])", "deny",
         "import subprocess\nreturn subprocess.run(['cmd.exe', '/c', 'echo hi'], capture_output=True).stdout"),
        ("spawn_python", "subprocess.run([sys.executable, '-c', 'print(1)'])", "deny",
         "import subprocess\nreturn subprocess.run([sys.executable, '-c', 'print(1)'], capture_output=True).stdout"),
        ("os_system", "os.system('echo hi') returns 0", "deny",
         "rc = os.system('echo hi')\nassert rc == 0, rc\nreturn rc"),
        ("kill_parent", "os.kill(<test process pid>, SIGTERM)", "deny",
         f"import signal\nos.kill({os.getpid()}, signal.SIGTERM)"),
        ("read_windows_hosts", r"read C:\Windows\System32\drivers\etc\hosts (ALL APPLICATION PACKAGES)", "info",
         r"return len(open(r'C:\Windows\System32\drivers\etc\hosts').read())"),
        ("scratch_read_write", "write, read back and list a file in the scratch folder (cwd)", "allow",
         "open('w.txt', 'w').write('hello')\nassert open('w.txt').read() == 'hello'\n"
         "assert open('given.txt').read() == 'from parent'\nimport tempfile\nd = tempfile.mkdtemp()\n"
         "open(os.path.join(d, 't'), 'w').write('t')\nreturn sorted(os.listdir('.')), tempfile.gettempdir() == os.getcwd()"),
        ("import_numpy_evalplus", "import numpy and evalplus.eval in the sandbox interpreter", "allow",
         "import numpy, evalplus.eval\nreturn numpy.__version__"),
    ]


def run_probe(name, body, expect):
    t0 = time.perf_counter()
    r = sandbox.run(["-c", probe_code(body)], timeout_s=30, files={"given.txt": b"from parent"})
    line = next((ln for ln in r.stdout.decode("utf-8", "replace").splitlines() if ln.startswith("PROBE ")), None)
    res = json.loads(line[6:]) if line else {"succeeded": None, "detail": f"no probe output; rc {r.returncode}; "
                                                                          f"stderr {r.stderr[-300:]!r}"}
    ok = {"deny": res["succeeded"] is False, "allow": res["succeeded"] is True, "info": True}[expect]
    return {"name": name, "expect": expect, "succeeded": res["succeeded"], "detail": res["detail"], "pass": ok,
            "wall_s": round(time.perf_counter() - t0, 3)}


def special_checks():
    import psutil
    rows = []
    # print(1+1)
    r = sandbox.run(["-c", "print(1+1)"], timeout_s=20)
    rows.append({"name": "print_1_plus_1", "expect": "allow", "succeeded": r.stdout.strip() == b"2",
                 "detail": f"stdout {r.stdout!r} rc {r.returncode}", "pass": r.stdout.strip() == b"2"
                 and r.returncode == 0, "wall_s": r.wall_s})
    # memory: bytearray(8 GiB) must raise MemoryError inside the 4 GiB job; the machine is unaffected
    avail0 = psutil.virtual_memory().available
    r = sandbox.run(["-c", "try:\n    b = bytearray(8 << 30)\n    print('ALLOCATED')\nexcept MemoryError:\n"
                     "    print('MemoryError')\nb2 = bytearray(1 << 30)\nprint('1GiB ok')"], timeout_s=60)
    avail1 = psutil.virtual_memory().available
    out = r.stdout.decode()
    ok = "MemoryError" in out and "ALLOCATED" not in out and "1GiB ok" in out and not r.timed_out
    rows.append({"name": "memory_8gib", "expect": "deny", "succeeded": "ALLOCATED" in out,
                 "detail": f"stdout {out.strip()!r}; job peak {sandbox.gib(r.peak_job_memory) if hasattr(sandbox, 'gib') else round((r.peak_job_memory or 0) / 2**30, 3)} GiB; "
                           f"RAM available before/after {avail0 / 2**30:.1f}/{avail1 / 2**30:.1f} GiB; {r.wall_s} s",
                 "pass": ok, "wall_s": r.wall_s})
    # memory: 5 GiB (above the 4 GiB job limit, well inside free RAM) must fail too
    r = sandbox.run(["-c", "try:\n    b = bytearray(5 << 30)\n    print('ALLOCATED')\nexcept MemoryError:\n"
                     "    print('MemoryError')"], timeout_s=60)
    rows.append({"name": "memory_5gib", "expect": "deny", "succeeded": b"ALLOCATED" in r.stdout,
                 "detail": f"{r.stdout.decode().strip()}; job peak {(r.peak_job_memory or 0) / 2**30:.2f} GiB",
                 "pass": b"MemoryError" in r.stdout, "wall_s": r.wall_s})
    # memory: a 3 GiB allocation works (the limit is the job's 4 GiB)
    r = sandbox.run(["-c", "try:\n    b = bytearray(3 << 30)\n    print('3GiB ok')\nexcept MemoryError:\n"
                     "    print('3GiB MemoryError')"], timeout_s=60)
    rows.append({"name": "memory_3gib_allowed", "expect": "allow", "succeeded": b"3GiB ok" in r.stdout,
                 "detail": r.stdout.decode().strip(), "pass": b"3GiB ok" in r.stdout, "wall_s": r.wall_s})
    # infinite loop killed at the timeout, nothing left behind
    r = sandbox.run(["-c", "while True:\n    pass"], timeout_s=3)
    left = [r.pid] if r.pid in sandbox_pythons() else []
    rows.append({"name": "infinite_loop_timeout", "expect": "deny", "succeeded": not r.timed_out,
                 "detail": f"timed_out {r.timed_out} after {r.wall_s} s, rc {r.returncode}, sandbox python.exe left: {left}",
                 "pass": r.timed_out and 2.9 < r.wall_s < 6 and not left, "wall_s": r.wall_s})
    # infinite print killed at the timeout; output capped
    r = sandbox.run(["-c", "import sys\nwhile True:\n    sys.stdout.write('x' * 4096 + '\\n')"], timeout_s=3)
    left = [r.pid] if r.pid in sandbox_pythons() else []
    rows.append({"name": "infinite_print_timeout", "expect": "deny", "succeeded": not r.timed_out,
                 "detail": f"timed_out {r.timed_out} after {r.wall_s} s, stdout kept {len(r.stdout):,} bytes "
                           f"(truncated {r.stdout_truncated}), sandbox python.exe left: {left}",
                 "pass": r.timed_out and len(r.stdout) <= sandbox.OUTPUT_CAP and not left, "wall_s": r.wall_s})
    # environment: only the minimal set
    allowed = {"LOCALAPPDATA", "PATH", "PYTHONDONTWRITEBYTECODE", "PYTHONIOENCODING", "PYTHONNOUSERSITE",
               "SYSTEMROOT", "TEMP", "TMP"}
    r = sandbox.run(["-c", "import os, json\nprint(json.dumps(dict(os.environ)))"], timeout_s=20)
    env = json.loads(r.stdout.decode())
    rows.append({"name": "environment_minimal", "expect": "allow", "succeeded": set(env) <= allowed,
                 "detail": json.dumps(env), "pass": set(env) <= allowed and "anneal.codesandbox" in env.get("TEMP", "").lower(),
                 "wall_s": r.wall_s})
    # stdin reaches the child
    r = sandbox.run(["-c", "import sys\nprint(sys.stdin.read()[::-1])"], stdin=b"abc", timeout_s=20)
    rows.append({"name": "stdin_roundtrip", "expect": "allow", "succeeded": r.stdout.strip() == b"cba",
                 "detail": repr(r.stdout), "pass": r.stdout.strip() == b"cba", "wall_s": r.wall_s})
    return rows


def battery(paths):
    rows = special_checks()
    for name, what, expect, body in make_probes(paths):
        row = run_probe(name, body, expect)
        row["what"] = what
        rows.append(row)
    return rows


def table(rows):
    out = ["| check | expected | operation succeeded | pass | detail |", "|---|---|---|---|---|"]
    for r in rows:
        detail = str(r["detail"]).replace("|", "/").replace("\n", " ")[:160]
        out.append(f"| {r['name']} | {r['expect']} | {r['succeeded']} | {'PASS' if r['pass'] else 'FAIL'} | {detail} |")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run-dir", default=os.path.join(RESULTS_DIR, "sandbox-check"))
    ap.add_argument("--parallel", type=int, default=8)
    ap.add_argument("--overhead-runs", type=int, default=100)
    args = ap.parse_args()
    run = Run(args.run_dir)
    run.capture_output()
    run.write_status(state="running")
    sandbox.require()
    profile = os.environ["USERPROFILE"]
    user_temp = os.path.realpath(tempfile.gettempdir())
    paths = {"training_probe": os.path.join(sandbox.TRAINING_ROOT, "sandbox_probe_secret.txt"),
             "profile": profile, "profile_probe": os.path.join(profile, "anneal_sandbox_probe.txt"),
             "user_temp": user_temp,
             "memory": os.environ.get("ANNEAL_PRIVATE_NOTES", os.path.join(profile, "anneal_sandbox_probe.txt"))}
    for k in ("training_probe", "profile_probe"):
        with open(paths[k], "w") as fh:
            fh.write("secret probe text\n")
    summary = {"container_sid": sandbox.container_sid()[1], "python": sandbox.PYTHON_EXE, "paths": paths}
    try:
        print(f"== sandbox check {time.strftime('%Y-%m-%d %H:%M:%S')}  SID {summary['container_sid']}")
        t0 = time.time()
        rows = battery(paths)
        summary["sequential"] = rows
        summary["sequential_seconds"] = round(time.time() - t0, 1)
        print(table(rows))

        print(f"\n== {args.parallel} batteries in parallel")
        t0 = time.time()
        with ThreadPoolExecutor(args.parallel) as ex:
            par = list(ex.map(lambda _: battery(paths), range(args.parallel)))
        summary["parallel_seconds"] = round(time.time() - t0, 1)
        fails = [(i, r["name"], r["detail"]) for i, rs in enumerate(par) for r in rs if not r["pass"]]
        summary["parallel"] = {"copies": args.parallel, "checks": sum(len(rs) for rs in par), "failures": fails,
                               "sandbox_python_left": sandbox_pythons()}
        print(f"parallel: {summary['parallel']['checks']} checks, {len(fails)} failures in "
              f"{summary['parallel_seconds']} s; left-over sandbox python.exe: {summary['parallel']['sandbox_python_left']}")
        for f in fails:
            print("  FAIL", f)

        print(f"\n== overhead: {args.overhead_runs} trivial runs")
        times = []
        for _ in range(args.overhead_runs):
            t = time.perf_counter()
            r = sandbox.run(["-c", "pass"], timeout_s=20)
            times.append(time.perf_counter() - t)
            assert r.returncode == 0, r
        times.sort()
        lock = threading.Lock()
        t0 = time.perf_counter()
        with ThreadPoolExecutor(8) as ex:
            list(ex.map(lambda _: sandbox.run(["-c", "pass"], timeout_s=20), range(args.overhead_runs)))
        par_wall = time.perf_counter() - t0
        r = None
        imp = []
        for _ in range(10):
            t = time.perf_counter()
            r = sandbox.run(["-c", "import numpy, evalplus.eval"], timeout_s=30)
            imp.append(time.perf_counter() - t)
        summary["overhead"] = {
            "sequential_ms_mean": round(1000 * sum(times) / len(times), 1),
            "sequential_ms_median": round(1000 * times[len(times) // 2], 1),
            "sequential_ms_p95": round(1000 * times[int(len(times) * 0.95)], 1),
            "parallel8_ms_per_run": round(1000 * par_wall / args.overhead_runs, 1),
            "import_numpy_evalplus_ms_mean": round(1000 * sum(imp) / len(imp), 1),
        }
        print(json.dumps(summary["overhead"], indent=1))
        all_ok = all(r["pass"] for r in rows) and not fails and not summary["parallel"]["sandbox_python_left"]
        summary["result"] = "PASS" if all_ok else "FAIL"
        print("RESULT:", summary["result"])
        run.write_status(state="finished")
    except Exception as exc:
        import traceback
        traceback.print_exc()
        run.write_status(state="failed", error=f"{type(exc).__name__}: {exc}")
        summary["result"] = f"ERROR {exc}"
    finally:
        # the probe files, and anything a denial probe would have created had it wrongly succeeded
        for path in [paths["training_probe"], paths["profile_probe"],
                     os.path.join(sandbox.TRAINING_ROOT, "sandbox_write_probe.txt"),
                     os.path.join(sandbox.PYTHON_DIR, "write_probe.txt"),
                     os.path.join(os.environ.get("PUBLIC", r"C:\Users\Public"), "anneal_write_probe.txt"),
                     os.path.join(paths["user_temp"], "anneal_write_probe.txt")]:
            if os.path.exists(path):
                print("removing", path)
                os.remove(path)
        with open(os.path.join(run.dir, "summary.json"), "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=1, default=str)
    return 0 if summary.get("result") == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
