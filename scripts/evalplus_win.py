"""EvalPlus (HumanEval+ 164, MBPP+ 378) on Windows: generation against llama-server, then EvalPlus's own sanitizer
and evaluator with model-written code run only in the sandbox (anneal/sandbox.py).

EvalPlus's evaluator can't run on Windows as it is: untrusted_check() forks a multiprocessing.Process whose
time_limit() uses signal.setitimer and whose reliability_guard() uses the resource module. Everything else is
EvalPlus's own code: problem loading, get_groundtruth() (the trusted canonical solutions, run natively in this
process), check_correctness(), the pass@1 for base and base+extra tests and the _eval_results.json. Only two names
in evalplus.evaluate are replaced while it runs:
  ProcessPoolExecutor -> concurrent.futures.ThreadPoolExecutor (the workers only wait on sandboxed processes)
  untrusted_check     -> sandboxed_untrusted_check (same signature and the same verdict mapping)

sandboxed_untrusted_check computes time_limits and the timeout with EvalPlus's own formula, pickles (dataset,
entry_point, code, inputs, expected, time_limits, atol, fast_check, nonce) into the scratch folder and runs CHILD in
the sandbox. CHILD imports evalplus.eval, swaps its time_limit for a threading.Timer + _thread.interrupt_main version
raising TimeoutException, and its reliability_guard for the original called with maximum_memory_bytes=None (the job
object caps memory at 4 GiB), then calls evalplus.eval.unsafe_execute itself. The shared-memory stat / details /
progress values become objects that print "<nonce> D i v", "<nonce> P n" and finally "<nonce> STAT s" to the real
stdout. EvalPlus's timeout starts when the child prints "<nonce> READY" (just before unsafe_execute; before that a
separate start-up allowance applies). No STAT -> timeout; PASS needs every detail True, as in untrusted_check.

Generation (gen_samples): the user message is exactly what EvalPlus 0.3.1's OpenAI backend builds,
  "Please provide a self-contained Python script that solves the following problem in a markdown code block:"
  + "\\n```python\\n" + prompt.strip() + "\\n```"
as the only message (no system message), with chat_template_kwargs enable_thinking=True, temperature 0.6,
top_p 0.95, top_k 20 and min_p 0 (the model's generation_config top_k, written out because llama-server's own defaults
are top_k 40 and min_p 0.05), seed = 100000 * seed + task index, max_tokens 40960. Each task's raw answer content is
saved as "solution" with the reasoning text, token counts, finish reason and seconds (resumable), then EvalPlus's
sanitize(solution, entry_point) writes <name>-sanitized.jsonl as `evalplus.sanitize` would.

  python scripts/evalplus_win.py generate --dataset humaneval --url http://127.0.0.1:8080 --seed 1 --out DIR
  python scripts/evalplus_win.py sanitize --dataset humaneval --samples DIR/humaneval.raw.jsonl
  python scripts/evalplus_win.py evaluate --dataset humaneval --samples DIR/humaneval-sanitized.jsonl
  python scripts/evalplus_win.py selftest [--run-dir DIR]     canonical / null solutions and hand-made edge cases
"""

import argparse
import concurrent.futures
import contextlib
import json
import os
import pickle
import secrets
import sys
import threading
import time
import urllib.error
import urllib.request

TRAINING_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, TRAINING_ROOT)
from anneal import sandbox  # noqa: E402

INSTRUCTION_PREFIX = ("Please provide a self-contained Python script that solves the following problem in a markdown "
                      "code block:")
SAMPLING = {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0.0}
MAX_TOKENS = 40960
STARTUP_ALLOWANCE_S = 60.0   # sandbox start + imports + unpickling, before EvalPlus's own timeout starts
DATASETS = ("humaneval", "mbpp")


def evalplus_message(prompt):
    """EvalPlus 0.3.1 codegen passes task["prompt"].strip() + "\\n"; its OpenAI decoder then builds this."""
    return INSTRUCTION_PREFIX + f"\n```python\n{prompt.strip()}\n```"


def load_problems(dataset):
    from evalplus.data import get_human_eval_plus, get_mbpp_plus
    return get_human_eval_plus() if dataset == "humaneval" else get_mbpp_plus()


def read_jsonl(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


# ================================================================================================== generation

def post(url, path, body, timeout):
    req = urllib.request.Request(url.rstrip("/") + path, data=json.dumps(body).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def count_tokens(url, text):
    if not text:
        return 0
    return len(post(url, "/tokenize", {"content": text, "add_special": False, "parse_special": False}, 120)["tokens"])


def chat_request(url, message, seed, max_tokens=MAX_TOKENS, sampling=None, timeout=6 * 3600):
    """One chat completion with thinking on. Returns the fields we keep for a task."""
    body = {"model": "local", "messages": [{"role": "user", "content": message}], "max_tokens": max_tokens,
            "seed": seed, "chat_template_kwargs": {"enable_thinking": True}, "stream": False}
    body.update(sampling or SAMPLING)
    t0 = time.time()
    data = post(url, "/v1/chat/completions", body, timeout)
    secs = time.time() - t0
    choice = data["choices"][0]
    msg = choice.get("message") or {}
    usage = data.get("usage") or {}
    timings = data.get("timings") or {}
    return {
        "content": msg.get("content") or "",
        "reasoning": msg.get("reasoning_content") or "",
        "finish_reason": choice.get("finish_reason"),
        "completion_tokens": usage.get("completion_tokens"),
        "prompt_tokens": usage.get("prompt_tokens"),
        "seconds": round(secs, 2),
        "timings": {k: timings.get(k) for k in ("prompt_n", "prompt_ms", "predicted_n", "predicted_ms",
                                                "draft_n", "draft_n_accepted") if k in timings},
    }


class Stopped(Exception):
    """generate_tasks() was asked to stop (stop_check); everything finished so far is saved."""


def generate_tasks(jobs, url, out_for, seed, concurrency=4, max_tokens=MAX_TOKENS, sampling=None,
                   reasoning_budget=None, log=print, stop_check=None, on_row=None):
    """One sample per (dataset, task_id) job, all jobs in one pool of `concurrency` requests (so a batch has no gap
    between datasets). out_for: {dataset: raw JSONL path}; resumable (tasks already there are skipped). Returns
    {"rows": rows written this session, "wall_s": session wall time, "generated_tokens": their completion tokens}."""
    problems = {ds: load_problems(ds) for ds in {d for d, _ in jobs}}
    order = {ds: list(p) for ds, p in problems.items()}  # EvalPlus's own task order; the index seeds the request
    done = {ds: {r["task_id"] for r in read_jsonl(out_for[ds])} for ds in problems}
    todo = [(d, t) for d, t in jobs if t not in done[d]]
    for ds in problems:
        os.makedirs(os.path.dirname(os.path.abspath(out_for[ds])), exist_ok=True)
    log(f"generate: {len(jobs)} tasks, {len(jobs) - len(todo)} done, {len(todo)} to do (seed {seed}, "
        f"{concurrency} in flight, max_tokens {max_tokens})")
    lock = threading.Lock()
    t_start = time.time()
    written, tok_total = [], [0]

    def one(job):
        ds, task_id = job
        if stop_check and stop_check():
            return None
        idx = order[ds].index(task_id)
        req_seed = 100000 * seed + idx
        r = None
        for attempt in range(3):
            try:
                r = chat_request(url, evalplus_message(problems[ds][task_id]["prompt"]), req_seed, max_tokens,
                                 sampling)
                break
            except (urllib.error.URLError, ConnectionError, TimeoutError, OSError) as exc:
                if stop_check and stop_check():
                    return None
                log(f"  {task_id}: request failed ({type(exc).__name__}: {exc}); retry {attempt + 1}")
                time.sleep(10)
        if r is None:
            raise RuntimeError(f"{task_id}: the server failed three times")
        row = {"task_id": task_id, "dataset": ds, "index": idx, "seed": req_seed, "solution": r["content"],
               "reasoning": r["reasoning"], "finish_reason": r["finish_reason"],
               "completion_tokens": r["completion_tokens"], "prompt_tokens": r["prompt_tokens"],
               "reasoning_tokens": count_tokens(url, r["reasoning"]),
               "answer_tokens": count_tokens(url, r["content"]), "seconds": r["seconds"], "timings": r["timings"]}
        if reasoning_budget:
            row["budget_hit"] = row["reasoning_tokens"] >= reasoning_budget - 2
        with lock:
            with open(out_for[ds], "a", encoding="utf-8") as fh:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            written.append(row)
            tok_total[0] += row["completion_tokens"] or 0
            el = time.time() - t_start
            log(f"  [{len(written)}/{len(todo)}] {task_id}: {row['completion_tokens']} tokens "
                f"({row['reasoning_tokens']} reasoning), {row['finish_reason']}, {row['seconds']} s; "
                f"{tok_total[0] / el:.1f} tok/s overall")
            if on_row:
                on_row(row, len(written), len(todo), tok_total[0] / el)
        return row

    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as ex:
        list(ex.map(one, todo))
    if stop_check and stop_check():
        raise Stopped()
    return {"rows": written, "wall_s": time.time() - t_start, "generated_tokens": tok_total[0]}


def generate(dataset, url, out_path, seed, task_ids=None, concurrency=4, max_tokens=MAX_TOKENS, sampling=None,
             reasoning_budget=None, log=print, stop_check=None):
    """Generate one sample per task of one dataset (resumable). Returns all rows for those tasks."""
    ids = list(load_problems(dataset)) if task_ids is None else list(task_ids)
    generate_tasks([(dataset, t) for t in ids], url, {dataset: out_path}, seed, concurrency, max_tokens, sampling,
                   reasoning_budget, log, stop_check)
    return [r for r in read_jsonl(out_path) if r["task_id"] in set(ids)]


def sanitize_file(dataset, raw_path, out_path=None):
    """EvalPlus's sanitize(solution, entry_point) on every raw row -> {task_id, solution} JSONL."""
    from evalplus.sanitize import sanitize
    problems = load_problems(dataset)
    out_path = out_path or raw_path.replace(".raw.jsonl", "-sanitized.jsonl")
    rows = read_jsonl(raw_path)
    with open(out_path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps({"task_id": r["task_id"],
                                 "solution": sanitize(r["solution"], entrypoint=problems[r["task_id"]]["entry_point"])},
                                ensure_ascii=False) + "\n")
    return out_path


# ================================================================================================== scoring

CHILD = r'''
import contextlib, os, pickle, signal, sys, threading, _thread

real_out = sys.stdout
with open("payload.pkl", "rb") as fh:
    dataset, entry_point, code, inputs, expected, time_limits, atol, fast_check, nonce = pickle.load(fh)


def emit(msg):
    real_out.write(nonce + " " + msg + "\n")
    real_out.flush()


import evalplus.eval as ev
from evalplus.eval.utils import TimeoutException

_state = {"gen": 0, "armed": False}
_lock = threading.Lock()


def _on_interrupt(signum, frame):
    if _state["armed"]:
        _state["armed"] = False
        raise TimeoutException("Timed out!")


signal.signal(signal.SIGINT, _on_interrupt)


@contextlib.contextmanager
def time_limit(seconds):
    """evalplus.eval.utils.time_limit without setitimer: a timer thread interrupts the main thread."""
    with _lock:
        _state["gen"] += 1
        gen = _state["gen"]
        _state["armed"] = True

    def fire():
        with _lock:
            if _state["gen"] == gen and _state["armed"]:
                _thread.interrupt_main()

    timer = threading.Timer(seconds, fire)
    timer.daemon = True
    timer.start()
    try:
        yield
    finally:
        with _lock:
            _state["armed"] = False
        timer.cancel()


_original_guard = ev.reliability_guard


def reliability_guard(maximum_memory_bytes=None):
    _original_guard(maximum_memory_bytes=None)  # no resource module on Windows: the job object caps memory


ev.time_limit = time_limit
ev.reliability_guard = reliability_guard


class Stat:
    def __init__(self, v):
        self.value = v


class Progress:
    def __init__(self):
        self._v = 0

    @property
    def value(self):
        return self._v

    @value.setter
    def value(self, v):
        self._v = v
        emit("P %d" % v)


class Details(list):
    def __setitem__(self, i, v):
        list.__setitem__(self, i, v)
        emit("D %d %d" % (i, 1 if v else 0))


stat = Stat(ev._UNKNOWN)
details = Details([False] * len(inputs))
progress = Progress()
emit("READY")
ev.unsafe_execute(dataset, entry_point, code, inputs, expected, time_limits, atol, fast_check, stat, details,
                  progress)
emit("STAT %d" % stat.value)
'''


class HarnessError(RuntimeError):
    """The sandboxed checker itself failed (never printed READY): a broken setup, not a wrong solution."""


_stats_lock = threading.Lock()
CHECK_STATS = {"checks": 0, "harness_retries": 0, "timeouts": 0, "startup_s_max": 0.0}


def sandboxed_untrusted_check(dataset, code, inputs, entry_point, expected, atol, ref_time, fast_check=False,
                              min_time_limit=None, gt_time_limit_factor=None):
    """evalplus.eval.untrusted_check with the child process in the sandbox. Same signature, same result."""
    import evalplus.eval as ev
    from evalplus.config import DEFAULT_GT_TIME_LIMIT_FACTOR, DEFAULT_MIN_TIME_LIMIT
    min_time_limit = DEFAULT_MIN_TIME_LIMIT if min_time_limit is None else min_time_limit
    gt_time_limit_factor = DEFAULT_GT_TIME_LIMIT_FACTOR if gt_time_limit_factor is None else gt_time_limit_factor
    time_limits = [max(min_time_limit, gt_time_limit_factor * t) for t in ref_time]
    timeout = min(os.getenv("EVALPLUS_TIMEOUT_PER_TASK", 60), sum(time_limits)) + 1
    if not fast_check:
        timeout += 1  # extra time for data collection
    for attempt in range(2):
        nonce = secrets.token_hex(16)
        payload = pickle.dumps((dataset, entry_point, code, inputs, expected, time_limits, atol, fast_check, nonce),
                               protocol=pickle.HIGHEST_PROTOCOL)
        # untrusted_check joins the process for timeout + 1 seconds
        r = sandbox.run(["child.py"], files={"child.py": CHILD, "payload.pkl": payload}, timeout_s=STARTUP_ALLOWANCE_S,
                        marker=(nonce + " READY").encode(), after_marker_s=timeout + 1, mem_bytes=4 << 30)
        if r.marker_s is not None:
            break
        with _stats_lock:
            CHECK_STATS["harness_retries"] += 1
    else:
        raise HarnessError(f"sandboxed check never started ({entry_point}): rc {r.returncode}, timed_out "
                           f"{r.timed_out}, stderr {r.stderr[-600:]!r}")
    stat_value, progress, det = ev._UNKNOWN, 0, [0] * len(inputs)
    prefix = nonce + " "
    for line in r.stdout.decode("utf-8", "replace").splitlines():
        if not line.startswith(prefix):
            continue
        parts = line[len(prefix):].split()
        if parts[0] == "D" and len(parts) == 3:
            det[int(parts[1])] = int(parts[2])
        elif parts[0] == "P" and len(parts) == 2:
            progress = int(parts[1])
        elif parts[0] == "STAT" and len(parts) == 2:
            stat_value = int(parts[1])
    stat = ev._mapping[stat_value]
    details = det[:progress]
    if not stat:
        stat = ev.TIMEOUT
    if stat == ev.PASS:
        if len(details) != len(inputs) or not all(details):
            stat = ev.FAIL
    with _stats_lock:
        CHECK_STATS["checks"] += 1
        CHECK_STATS["timeouts"] += stat == ev.TIMEOUT
        CHECK_STATS["startup_s_max"] = max(CHECK_STATS["startup_s_max"], r.marker_s)
    return stat, details


@contextlib.contextmanager
def windows_evaluator():
    """Swap the two Unix-only pieces of evalplus.evaluate for the duration."""
    import evalplus.evaluate as ee
    saved = ee.ProcessPoolExecutor, ee.untrusted_check
    ee.ProcessPoolExecutor = concurrent.futures.ThreadPoolExecutor
    ee.untrusted_check = sandboxed_untrusted_check
    try:
        yield ee
    finally:
        ee.ProcessPoolExecutor, ee.untrusted_check = saved


def pass_at_1(result_path):
    """pass@1 (base, base+extra) from an _eval_results.json, computed as evaluate() does for n=1, plus per-task
    statuses."""
    with open(result_path, encoding="utf-8") as fh:
        res = json.load(fh)
    per_task = {}
    for task_id, rs in res["eval"].items():
        r = rs[0]
        per_task[task_id] = {"base": r["base_status"], "plus": r["plus_status"],
                             "base_pass": r["base_status"] == "pass",
                             "plus_pass": r["base_status"] == r["plus_status"] == "pass"}
    n = len(per_task)
    return {"n": n, "base": sum(t["base_pass"] for t in per_task.values()) / n,
            "plus": sum(t["plus_pass"] for t in per_task.values()) / n, "per_task": per_task}


def evaluate(dataset, samples, parallel=8):
    """EvalPlus's evaluate() (all tasks must be in samples) with the sandboxed checker. Returns pass_at_1()."""
    with windows_evaluator() as ee:
        ee.evaluate(dataset, samples=samples, parallel=parallel)
    return pass_at_1(samples.replace(".jsonl", "_eval_results.json"))


def score_subset(dataset, samples, parallel=8):
    """EvalPlus's get_groundtruth + check_correctness for a subset of tasks (evaluate() insists on every task).
    samples: [{"task_id", "solution"}]. Returns {task_id: {"base", "plus", "base_pass", "plus_pass"}}."""
    from evalplus.data import get_human_eval_plus_hash, get_mbpp_plus_hash
    from evalplus.eval._special_oracle import MBPP_OUTPUT_NOT_NONE_TASKS
    with windows_evaluator() as ee:
        problems = load_problems(dataset)
        if dataset == "humaneval":
            expected = ee.get_groundtruth(problems, get_human_eval_plus_hash(), [])
        else:
            expected = ee.get_groundtruth(problems, get_mbpp_plus_hash(), MBPP_OUTPUT_NOT_NONE_TASKS)
        out = {}
        with concurrent.futures.ThreadPoolExecutor(parallel) as ex:
            futs = {ex.submit(ee.check_correctness, dataset, 0, problems[s["task_id"]], s["solution"],
                              expected[s["task_id"]], False, True, s["task_id"]): s["task_id"] for s in samples}
            for f in concurrent.futures.as_completed(futs):
                r = f.result()
                b, p = r["base"][0], r["plus"][0]
                out[futs[f]] = {"base": b, "plus": p, "base_pass": b == "pass", "plus_pass": b == p == "pass"}
    return out


# ================================================================================================== self-test

EDGE_CASES = [
    # (name, dataset, task_id, solution, expected base status, expected plus status)
    ("float within atol (truncate_number + 1e-9)", "humaneval", "HumanEval/2",
     "def truncate_number(number: float) -> float:\n    return number % 1.0 + 1e-9\n", "pass", "pass"),
    ("float outside atol (truncate_number + 1e-3)", "humaneval", "HumanEval/2",
     "def truncate_number(number: float) -> float:\n    return number % 1.0 + 1e-3\n", "fail", "fail"),
    ("find_zero special oracle: different root-finder (Newton)", "humaneval", "HumanEval/32",
     "import math\n\ndef poly(xs, x):\n    return sum(c * math.pow(x, i) for i, c in enumerate(xs))\n\n"
     "def find_zero(xs):\n    x = 0.0\n    for _ in range(200):\n"
     "        d = sum(i * c * x ** (i - 1) for i, c in enumerate(xs) if i)\n"
     "        if d == 0:\n            x += 0.5\n            continue\n        x -= poly(xs, x) / d\n    lo, hi = -1.0, 1.0\n"
     "    while poly(xs, lo) * poly(xs, hi) > 0:\n        lo *= 2.0\n        hi *= 2.0\n"
     "    if abs(poly(xs, x)) < 1e-9:\n        return x\n"
     "    while hi - lo > 1e-10:\n        mid = (lo + hi) / 2\n"
     "        if poly(xs, mid) * poly(xs, lo) > 0:\n            lo = mid\n        else:\n            hi = mid\n"
     "    return lo\n", "pass", "pass"),
    ("find_zero special oracle: returns 0.0", "humaneval", "HumanEval/32",
     "def find_zero(xs):\n    return 0.0\n", "fail", "fail"),
    ("MBPP set-equality oracle: same elements, other order (similar_elements)", "mbpp", "Mbpp/2",
     "def similar_elements(test_tup1, test_tup2):\n    return tuple(sorted(set(test_tup1) & set(test_tup2), reverse=True))\n",
     "pass", "pass"),
    ("MBPP set-equality oracle: wrong elements", "mbpp", "Mbpp/2",
     "def similar_elements(test_tup1, test_tup2):\n    return tuple(set(test_tup1) | set(test_tup2))\n", "fail", "fail"),
    # A Python-level infinite loop is stopped by EvalPlus's own time_limit (TimeoutException inside the test loop),
    # which unsafe_execute counts as a failure; only a call the timer can't interrupt reaches the process timeout.
    ("infinite Python loop (time_limit raises TimeoutException -> fail, as in EvalPlus)", "humaneval", "HumanEval/0",
     "def has_close_elements(numbers, threshold):\n    while True:\n        pass\n", "fail", "fail"),
    ("timeout: one long C-level call the timer can't interrupt", "humaneval", "HumanEval/0",
     "def has_close_elements(numbers, threshold):\n    return sum(range(10 ** 15)) < 0\n", "timeout", "timeout"),
    ("exception", "humaneval", "HumanEval/0",
     "def has_close_elements(numbers, threshold):\n    raise ValueError('no')\n", "fail", "fail"),
    ("uses os.getcwd (disabled by reliability_guard)", "humaneval", "HumanEval/0",
     "import os\n\ndef has_close_elements(numbers, threshold):\n    os.getcwd()\n"
     "    return any(abs(a - b) < threshold for i, a in enumerate(numbers) for b in numbers[i + 1:])\n", "fail", "fail"),
    ("correct, written differently", "humaneval", "HumanEval/0",
     "def has_close_elements(numbers, threshold):\n"
     "    s = sorted(numbers)\n    return any(b - a < threshold for a, b in zip(s, s[1:]))\n", "pass", "pass"),
    ("prints and calls sys.exit at import", "humaneval", "HumanEval/0",
     "import sys\nprint('x' * 100000)\nsys.exit(0)\n", "fail", "fail"),
    ("tries to read a file outside the sandbox", "humaneval", "HumanEval/0",
     "import io\n\ndef has_close_elements(numbers, threshold):\n"
     "    io.open(r'C:\\AI\\portal\\server.py').read()\n    return False\n", "fail", "fail"),
]


def selftest(run_dir, parallel=8, log=print):
    out = {"datasets": {}, "edge_cases": []}
    os.makedirs(run_dir, exist_ok=True)
    for ds in DATASETS:
        problems = load_problems(ds)
        out["datasets"][ds] = {"tasks": len(problems)}
        for kind in ("canonical", "null"):
            path = os.path.join(run_dir, f"{ds}-{kind}.jsonl")
            res_path = path.replace(".jsonl", "_eval_results.json")
            if os.path.exists(res_path):
                os.remove(res_path)  # our own output from an earlier self-test
            with open(path, "w", encoding="utf-8") as fh:
                for task_id, p in problems.items():
                    sol = (p["prompt"] + p["canonical_solution"] if kind == "canonical"
                           else f"def {p['entry_point']}(*args, **kwargs):\n    return None\n")
                    fh.write(json.dumps({"task_id": task_id, "solution": sol}) + "\n")
            t0 = time.time()
            res = evaluate(ds, path, parallel)
            failed = sorted(t for t, v in res["per_task"].items() if not v["plus_pass"]) if kind == "canonical" else \
                sorted(t for t, v in res["per_task"].items() if v["base_pass"] or v["plus_pass"])
            out["datasets"][ds][kind] = {"base": res["base"], "plus": res["plus"], "seconds": round(time.time() - t0, 1),
                                         "exceptions": failed[:40]}
            log(f"{ds} {kind}: pass@1 base {res['base']:.4f}, base+extra {res['plus']:.4f} "
                f"({out['datasets'][ds][kind]['seconds']} s); {'not passing' if kind == 'canonical' else 'passing'}: {failed[:20]}")
    for name, ds, task_id, sol, want_b, want_p in EDGE_CASES:
        r = score_subset(ds, [{"task_id": task_id, "solution": sol}], parallel=1)[task_id]
        ok = r["base"] == want_b and r["plus"] == want_p
        out["edge_cases"].append({"case": name, "task": task_id, "expected": [want_b, want_p],
                                  "got": [r["base"], r["plus"]], "ok": ok})
        log(f"edge case {'OK  ' if ok else 'FAIL'} {name} ({task_id}): expected {want_b}/{want_p}, got {r['base']}/{r['plus']}")
    out["check_stats"] = dict(CHECK_STATS)
    can = all(out["datasets"][d]["canonical"]["base"] == 1.0 and out["datasets"][d]["canonical"]["plus"] == 1.0
              for d in DATASETS)
    out["result"] = "PASS" if can and all(e["ok"] for e in out["edge_cases"]) else "FAIL"
    return out


# ================================================================================================== CLI

def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate")
    g.add_argument("--dataset", choices=DATASETS, required=True)
    g.add_argument("--url", default="http://127.0.0.1:8080")
    g.add_argument("--seed", type=int, default=1)
    g.add_argument("--out", required=True, help="folder; writes <dataset>.raw.jsonl and <dataset>-sanitized.jsonl")
    g.add_argument("--concurrency", type=int, default=4)
    g.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    g.add_argument("--tasks", default=None, help="comma-separated task ids (default: all)")
    s = sub.add_parser("sanitize")
    s.add_argument("--dataset", choices=DATASETS, required=True)
    s.add_argument("--samples", required=True)
    e = sub.add_parser("evaluate")
    e.add_argument("--dataset", choices=DATASETS, required=True)
    e.add_argument("--samples", required=True)
    e.add_argument("--parallel", type=int, default=8)
    t = sub.add_parser("selftest")
    t.add_argument("--run-dir", default=os.path.join(TRAINING_ROOT, "results", "evalplus-selftest"))
    t.add_argument("--parallel", type=int, default=8)
    args = ap.parse_args()
    sandbox.require()
    if args.cmd == "generate":
        raw = os.path.join(args.out, f"{args.dataset}.raw.jsonl")
        generate(args.dataset, args.url, raw, args.seed, args.tasks.split(",") if args.tasks else None,
                 args.concurrency, args.max_tokens)
        print("sanitized ->", sanitize_file(args.dataset, raw))
    elif args.cmd == "sanitize":
        print("sanitized ->", sanitize_file(args.dataset, args.samples))
    elif args.cmd == "evaluate":
        res = evaluate(args.dataset, args.samples, args.parallel)
        print(json.dumps({k: v for k, v in res.items() if k != "per_task"}, indent=1))
    elif args.cmd == "selftest":
        from anneal.runtime import Run
        run = Run(args.run_dir)
        run.capture_output()
        run.write_status(state="running")
        out = selftest(args.run_dir, args.parallel)
        with open(os.path.join(args.run_dir, "summary.json"), "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=1)
        run.write_status(state="finished" if out["result"] == "PASS" else "failed")
        print("RESULT:", out["result"])
        return 0 if out["result"] == "PASS" else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
