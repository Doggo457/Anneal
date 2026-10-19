"""Tests for the improvement loop without a GPU, a model or Windows: the keep rule, the overlap check, scoring,
row bookkeeping, the watchdog's decisions, and a full simulated loop (base, first run, a kept round, a skipped recipe,
a rejected round, a pause and resume in the middle of training) with every subprocess and the model faked.

  python tests/test_loop_logic.py        (or: python -m pytest tests/test_loop_logic.py)
"""

import json
import os
import shutil
import sys
import tempfile
import threading
import types

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

# The real sandbox needs Windows; nothing here runs model code, so a stub stands in for it.
_stub = types.ModuleType("anneal.sandbox")
_stub.require = lambda: None
_stub.run = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("sandbox.run must not be called in tests"))
_stub.TEST_MEM_BYTES = 0
sys.modules["anneal.sandbox"] = _stub
import anneal  # noqa: E402
anneal.sandbox = _stub

from anneal import evalsets as E  # noqa: E402
import loop as L  # noqa: E402
import loop_watchdog as W  # noqa: E402
import first_run as fr  # noqa: E402
import evalplus_win as ew  # noqa: E402
import generate_traces as gt  # noqa: E402


# ============================================================================================ unit tests

def test_sign_test():
    assert E.sign_test_greater(0, 0) == 1.0
    assert abs(E.sign_test_greater(10, 0) - 1 / 1024) < 1e-12
    assert abs(E.sign_test_greater(5, 5) - 0.623046875) < 1e-9  # P(X >= 5), n = 10
    assert E.sign_test_greater(3, 7) > 0.9


def rows(ids, passing, seed=1):
    return [{"id": i, "seed": seed, "pass": i in passing} for i in ids]


def test_compare_and_decide():
    ids = [f"cc/v/{i}" for i in range(20)] + [f"oci/{i}" for i in range(20)]
    nc_ids = [f"mmlupro/{i}" for i in range(10)]
    rules = {"alpha": 0.05, "cc_tolerance": 0.05, "nc_tolerance": 0.05}
    base = rows(ids, set(ids[:20]))
    nc = rows(nc_ids, set(nc_ids[:7])) + rows(nc_ids, set(nc_ids[:7]), seed=2)
    better = rows(ids, set(ids[:30]))                     # 10 OCI wins, no losses
    d = E.decide(base, better, nc, nc, rules)
    assert d["keep"] and d["gate"]["wins"] == 10 and d["gate"]["losses"] == 0, d
    assert d["gate_oci"]["diff"] == 0.5 and d["gate_cc"]["diff"] == 0
    # same score: not kept
    assert not E.decide(base, base, nc, nc, rules)["keep"]
    # small gain, not significant: 3 wins, 1 loss
    small = rows(ids, (set(ids[:20]) - {ids[0]}) | set(ids[20:23]))
    d = E.decide(base, small, nc, nc, rules)
    assert not d["keep"] and "not clear" in d["reasons"][0], d
    # gate up overall but CodeContests down by 0.10 (> tolerance 0.05): rejected
    trade = rows(ids, (set(ids[:20]) - set(ids[:2])) | set(ids[20:35]))
    d = E.decide(base, trade, nc, nc, rules)
    assert not d["keep"] and any("CodeContests" in r for r in d["reasons"]), d
    # non-coding drop beyond tolerance: rejected
    worse_nc = rows(nc_ids, set(nc_ids[:5])) + rows(nc_ids, set(nc_ids[:5]), seed=2)
    d = E.decide(base, better, nc, worse_nc, rules)
    assert not d["keep"] and any("non-coding" in r for r in d["reasons"]), d
    # sample generations broken: rejected
    assert not E.decide(base, better, nc, nc, rules, sanity_ok=False)["keep"]


def test_mc_extract_and_mmlu():
    assert E.mc_extract("so... The answer is (C).") == "C"
    assert E.mc_extract("the answer is B, wait, The answer is (D)") == "D"
    assert E.mc_extract("Answer: E") == "E"
    assert E.mc_extract("no letter") is None
    src = [{"question_id": i, "question": f"q{i}", "options": ["a", "b", "c", "N/A", "N/A"], "answer_index": 2,
            "category": "computer science" if i % 5 == 0 else "law"} for i in range(50)]
    qs = E.mmlu_pro_questions(src, n=10, seed=42)
    assert len(qs) == 10 and all(q["category"] == "law" and q["options"] == ["a", "b", "c"] and q["answer"] == "C"
                                 for q in qs)
    assert qs == E.mmlu_pro_questions(src, n=10, seed=42)  # reproducible
    assert "C. c" in E.mc_message(qs[0])


def test_cc_candidates():
    row = {"description": "Sum two numbers.", "public_tests": {"input": ["1 2\n"], "output": ["3\n"]},
           "private_tests": {"input": ["1 2\n", "5 5\n"], "output": ["3\n", "10\n"]},
           "generated_tests": {"input": [], "output": []},
           "solutions": {"language": [2, 3, 3], "solution": ["c++", "py-a", "py-b"]}, "input_file": "", "output_file": ""}
    filed = dict(row, input_file="input.txt")
    nosol = dict(row, solutions={"language": [2], "solution": ["c++"]})
    c = E.cc_candidates([row, filed, nosol], "valid", "\nSUFFIX")
    assert len(c) == 1 and c[0]["id"] == "cc/valid/0" and c[0]["cases"] == [("1 2\n", "3\n"), ("5 5\n", "10\n")]
    assert c[0]["refs"] == ["py-a", "py-b"] and c[0]["prompt"].endswith("SUFFIX")
    kept = E.harness_check(c, lambda code, cases, t: code == "py-b", lambda *a: False, parallel=2, log=lambda m: None)
    assert len(kept) == 1 and kept[0]["ref_ok"] == 1 and "refs" not in kept[0]


def test_overlap_index():
    boiler = ("The first line contains a single integer t (1 <= t <= 10^4) - the number of test cases. "
              "Each test case consists of one line")
    # real statements run to hundreds of words, so shared boilerplate is a small share of their 13-grams
    a = ("Vasya has an array of n integers and wants to sort it in non-decreasing order by reversing exactly one "
         "contiguous segment. He may choose any segment from position l to position r, reverse the order of the "
         "elements inside it, and leave every other element where it was. Help Vasya decide whether a single "
         "reversal is enough, and if it is, print the smallest pair of positions that works. If the array is "
         "already sorted he must still reverse a segment, so a segment of length one is allowed. " + boiler +
         " with n, followed by a line with the array elements a1 to an separated by spaces.")
    b = ("Petya loves lucky numbers, the positive integers whose decimal representation contains only the digits "
         "four and seven. For example 47, 744 and 4 are lucky while 5, 17 and 467 are not. Petya wants to know how "
         "many lucky numbers lie in a closed range from l to r, and also the sum of the next lucky number for each "
         "integer in that range, where next of x is the smallest lucky number not less than x. " + boiler +
         " with the two integers l and r separated by a space.")
    idx = E.OverlapIndex([("cc/a", a), ("mm/q", "Which organ pumps blood around the human body in mammals?")])
    assert idx.hits(b) == set()
    assert idx.hits("Solve: " + a) == {"cc/a"}
    assert idx.hits("which organ pumps blood around the human body in mammals") == {"mm/q"}  # Jaccard
    assert idx.hits("unrelated text", jaccard_text="") == set()


def test_score_resumes_and_stops():
    with tempfile.TemporaryDirectory() as d:
        items = [{"id": f"oci/{i}", "kind": "oci", "prompt": f"p{i}", "tests": "assert True"} for i in range(8)]
        calls = []

        def chat(url, message, seed, max_tokens):
            calls.append((message, seed))
            return {"content": f"```python\nx = {message!r}\n```", "reasoning": "think", "finish_reason": "stop",
                    "completion_tokens": 10, "seconds": 0.01}
        check = lambda item, content: item["id"].endswith(("0", "2", "4"))  # noqa: E731
        out = os.path.join(d, "s.jsonl")
        stop = {"n": 0}

        def stop_check():
            return len(calls) >= 3 and stop["n"] == 0

        try:
            E.score("u", items, out, 1, 100, check, chat, lambda u, t: 5, 1000, concurrency=1, stop_check=stop_check,
                    log=lambda m: None)
            raise AssertionError("expected Stopped")
        except E.Stopped:
            pass
        done = E.read_jsonl(out)
        assert 1 <= len(done) <= 3
        stop["n"] = 1
        r = E.score("u", items, out, 1, 100, check, chat, lambda u, t: 5, 1000, concurrency=3, log=lambda m: None)
        assert len(r) == 8 and len({x["id"] for x in r}) == 8 and sum(x["pass"] for x in r) == 3
        assert {s for _, s in calls} <= {100000 + i for i in range(8)}  # request seed = 100000 * seed + index
        r2 = E.score("u", items, out, 2, 100, check, chat, lambda u, t: 5, 1000, log=lambda m: None)
        assert len(r2) == 8 and len(E.read_jsonl(out)) == 16


def test_rows_bookkeeping():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "pool.jsonl")
        data = [{"messages": [{"role": "user", "content": f"q{i % 7}"}, {"role": "assistant", "content": f"a{i}"}]}
                for i in range(20)]
        E.write_jsonl(path, data)
        used = {L.row_key(data[0])}  # question q0 used before
        idx, nxt = L.pick_rows(path, 0, 5, used)
        assert idx == [1, 2, 3, 4, 5] and nxt == 6, (idx, nxt)
        idx2, nxt2 = L.pick_rows(path, nxt, 5, used)  # q6 at 6 new; 7..13 repeat q0..q6 -> all used
        assert idx2 == [6] and nxt2 == 20, (idx2, nxt2)
        assert [r["messages"][1]["content"] for r in L.rows_at(path, [6, 1, 3])] == ["a6", "a1", "a3"]
        agent = {"messages": [{"role": "system", "content": "s"}, {"role": "user", "content": "task"},
                              {"role": "assistant", "content": "step1"}, {"role": "tool", "content": "out"},
                              {"role": "assistant", "content": "step2"}]}
        agent2 = dict(agent, messages=agent["messages"][:3])
        assert L.row_key(agent) != L.row_key(agent2)  # each step of a session is its own row


def test_plan_counts_and_init():
    cfg = {"rows_per_round": 100, "sources": [{"name": "a", "share": 0.85}, {"name": "b", "share": 0.05},
                                              {"name": "c", "share": 0.10}, {"name": "f", "share": 0}]}
    st = {"offsets": {"a": 10}}
    c = L.plan_counts(cfg, {}, st, {"a": 300, "c": 60, "f": 100})  # b unusable
    assert c == {"a": 89, "c": 11}, c
    assert L.plan_counts(cfg, {"mix": {"f": 1.0}}, st, {"a": 300}) == {}
    shape = {"lora_r": 16, "lora_alpha": 32, "target_modules": ["q"]}
    assert L.choose_init({}, {"adapter": None}, shape) == "base"
    assert L.choose_init({}, {"adapter": "x", "lora_shape": shape}, shape) == "best"
    assert L.choose_init({}, {"adapter": "x", "lora_shape": shape}, dict(shape, lora_r=32)) == "base_lineage"
    assert L.choose_init({"init": "base_lineage"}, {"adapter": "x", "lora_shape": shape}, shape) == "base_lineage"


def test_safe_delete_stays_inside():
    with tempfile.TemporaryDirectory() as d:
        rd = os.path.join(d, "rounds", "r003")
        os.makedirs(os.path.join(rd, "train", "adapter"))
        open(os.path.join(rd, "train", "adapter", "w.bin"), "wb").write(b"x" * 10)
        outside = os.path.join(d, "keep.txt")
        open(outside, "w").write("keep")
        freed = L.safe_delete(rd, ["train/adapter", "../../keep.txt", "missing"], log=lambda m: None)
        assert freed == 10 and not os.path.exists(os.path.join(rd, "train", "adapter")) and os.path.exists(outside)
    assert L.pid_alive(os.getpid()) and not L.pid_alive(0)


def test_watchdog_decide():
    now = 1_000_000.0
    st = {"restarts": []}
    run = lambda tag, state: {"tag": tag, "state": state}  # noqa: E731
    fin = {"state": "finished"}
    dec = lambda t, fs=fin, ls=None, gate=True, mt=None, s=st: W.decide(t, now, s, fs, ls, gate, mt)  # noqa: E731
    assert dec({"active": "first-run", "runs": []})[0] is None
    assert dec({"runs": [run("first-run", "running")]})[0] is None
    assert dec({"runs": [run("first-run", "paused")]})[0] is None
    assert dec({"runs": [run("first-run", "ended")]})[:2] == ("resume", "first-run")
    assert dec({"runs": [run("first-run", "failed")]})[:2] == ("resume", "first-run")
    many = {"restarts": [{"t": now - 100, "tag": "first-run"}] * 3}
    assert dec({"runs": [run("first-run", "failed")]}, s=many)[0] is None
    assert dec({"runs": [run("first-run", "finished")]}, gate=False)[0] is None
    assert dec({"runs": [run("first-run", "finished")]})[:2] == ("start", "loop")
    assert dec({"runs": [run("first-run", "finished")]}, fs={"state": "running"})[0] is None
    both = lambda s: {"runs": [run("first-run", "finished"), run("loop", s)]}  # noqa: E731
    assert dec(both("running"))[0] is None
    assert dec(both("paused"), ls={"state": "paused"})[0] is None
    assert dec(both("stopped"))[0] is None
    assert dec(both("ended"), ls={"state": "running"})[:2] == ("resume", "loop")
    assert dec(both("failed"), ls={"state": "failed"})[:2] == ("resume", "loop")
    waiting = {"state": "waiting", "config_mtime": 500.0, "updated": now - 60, "detail": "no rows"}
    assert dec(both("finished"), ls=waiting, mt=500.0)[0] is None
    assert dec(both("finished"), ls=waiting, mt=600.0)[0] == "resume"
    assert dec(both("finished"), ls=dict(waiting, updated=now - 7 * 3600), mt=500.0)[0] == "resume"
    assert dec(both("finished"), ls={"state": "finished"})[0] is None


# ============================================================================================ simulated loop

GATE_N, NC_N = 40, 10
PASSING = {None: 20, "first-run": 22, "r002": 30, "r003": 18, "r004": 35}  # gate items 0..k-1 pass


class FakeServer:
    def __init__(self, flags):
        self.lora = flags[flags.index("--lora") + 1] if "--lora" in flags else None
        self.url = "http://fake"
        self.proc = None
        self.log_path = os.devnull

    def alive(self):
        return True

    def stop(self):
        pass


class World:
    """Everything outside loop.py, faked: model answers, tests, subprocesses."""

    def __init__(self, tmp):
        self.tmp = tmp
        self.server = None
        self.calls = []
        self.pause_train_once = {"loop-r003"}
        self.prompt_to_idx = {}

    def model(self):
        lora = self.server.lora if self.server else None
        if lora is None:
            return None
        if "first-run" in lora:
            return "first-run"
        return os.path.basename(lora).split("-")[1]  # loop-r002-F16-LoRA.gguf -> r002

    def chat(self, url, message, seed, max_tokens=None, sampling=None):
        m = self.model()
        if message in self.prompt_to_idx:
            kind, i = self.prompt_to_idx[message]
            if kind == "gate":
                ok = i < PASSING[m]
                return {"content": f"```python\n# {'PASS' if ok else 'FAIL'}\n```", "reasoning": "r",
                        "finish_reason": "stop", "completion_tokens": 50, "seconds": 0.0}
            return {"content": f"The answer is ({'A' if i < 7 else 'B'})", "reasoning": "r", "finish_reason": "stop",
                    "completion_tokens": 20, "seconds": 0.0}
        return {"content": "def f(): pass", "reasoning": "thinking", "finish_reason": "stop", "completion_tokens": 30,
                "seconds": 0.0}  # sample prompts

    def run_child(self, ctl, cmd, log_name, stop_dir=None, monitor=None, interval=30):
        script = os.path.basename(cmd[2])
        arg = lambda flag: cmd[cmd.index(flag) + 1]  # noqa: E731
        self.calls.append((script, cmd))
        if script in ("prepare_dataset.py", "public_source.py"):
            out = arg("--out")
            name = os.path.basename(out)
            n = {"tachibana4_v4pro": 300, "banaxi_v4pro": 20, "math_v4_aops": 60, "agent_sessions": 100}[name]
            E.write_jsonl(os.path.join(out, "train.jsonl"),
                          [{"messages": [{"role": "user", "content": f"{name} q{i}"},
                                         {"role": "assistant", "content": f"<think>t</think> a{i}"}]} for i in range(n)])
            E.write_jsonl(os.path.join(out, "val.jsonl"),
                          [{"messages": [{"role": "user", "content": f"{name} v{i}"},
                                         {"role": "assistant", "content": "<think>t</think> v"}]} for i in range(5)])
            return 0
        if script == "mix_datasets.py":
            out = arg("--out")
            train, val = [], []
            for i, c in enumerate(cmd):
                if c == "--source":
                    name, _, rest = cmd[i + 1].partition("=")
                    folder = rest.rpartition(":")[0]
                    train += [dict(r, source=name) for r in E.read_jsonl(os.path.join(folder, "train.jsonl"))]
                    val += E.read_jsonl(os.path.join(folder, "val.jsonl"))[:2]
            E.write_jsonl(os.path.join(out, "train.jsonl"), train)
            E.write_jsonl(os.path.join(out, "val.jsonl"), val)
            open(os.path.join(out, "data_card.md"), "w").write(f"# Mixed training set\n\nTrain rows: {len(train)}.\n")
            return 0
        if script == "train_longrun.py":
            import yaml
            cfg = yaml.safe_load(open(arg("--config"), encoding="utf-8"))
            rd = arg("--run-dir")
            if cfg["run_name"] in self.pause_train_once:
                self.pause_train_once.discard(cfg["run_name"])
                os.makedirs(os.path.join(rd, "checkpoints", "checkpoint-10"), exist_ok=True)
                json.dump({"state": "paused"}, open(os.path.join(rd, "status.json"), "w"))
                ctl.stop.set()  # as if Pause was pressed
                return 0
            os.makedirs(os.path.join(rd, "adapter"), exist_ok=True)
            os.makedirs(os.path.join(rd, "checkpoints", "checkpoint-20"), exist_ok=True)
            open(os.path.join(rd, "adapter", "adapter_model.safetensors"), "wb").write(b"w" * 100)
            json.dump({"state": "finished"}, open(os.path.join(rd, "status.json"), "w"))
            json.dump({"global_step": 12, "adapter": os.path.join(rd, "adapter"), "rows": len(E.read_jsonl(cfg["train_file"])),
                       "init_adapter": cfg.get("init_adapter")}, open(os.path.join(rd, "summary.json"), "w"))
            return 0
        if script == "convert_lora_to_gguf.py":
            open(arg("--outfile"), "wb").write(b"gguf")
            return 0
        if script == "val_loss.py":
            rd = arg("--run-dir")
            os.makedirs(rd, exist_ok=True)
            json.dump({"all": {"rows": 4, "base": {"answer_token_loss": 1.0, "trained_token_loss": 1.0},
                               "adapter": {"answer_token_loss": 0.9, "trained_token_loss": 0.9}}},
                      open(os.path.join(rd, "summary.json"), "w"))
            return 0
        raise AssertionError(f"unexpected child {cmd}")


def setup_world(tmp):
    w = World(tmp)
    ev = os.path.join(tmp, "data", "eval")
    E.EVAL_DIR = ev
    E.GATE_PATH, E.NONCODING_PATH, E.DECONTAM_EXTRA = (os.path.join(ev, f) for f in
                                                        ("gate.jsonl", "noncoding_100.jsonl", "decontam_extra.jsonl"))
    gate = [{"id": f"cc/valid/{i}", "kind": "cc", "prompt": f"gate prompt {i}", "cases": [["1\n", "1\n"]]} for i in range(20)]
    gate += [{"id": f"oci/{i}", "kind": "oci", "prompt": f"gate prompt {i + 20}", "tests": "assert True"} for i in range(20)]
    nc = [{"id": f"mmlupro/{i}", "category": "law", "question": f"nc question {i}", "options": ["x", "y"], "answer": "A"}
          for i in range(NC_N)]
    E.write_jsonl(E.GATE_PATH, gate)
    E.write_jsonl(E.NONCODING_PATH, nc)
    E.write_jsonl(E.DECONTAM_EXTRA, [{"id": g["id"], "prompt": g["prompt"]} for g in gate])
    for i, g in enumerate(gate):
        w.prompt_to_idx[g["prompt"]] = ("gate", i)
    for i, q in enumerate(nc):
        w.prompt_to_idx[E.mc_message(q)] = ("nc", i)
    # data trained on before the loop (the first run's mix)
    E.write_jsonl(os.path.join(tmp, "data", "processed", "mix_first_2k", "train.jsonl"),
                  [{"messages": [{"role": "user", "content": "tachibana4_v4pro q0"}, {"role": "assistant", "content": "x"}]}])
    # the first run, finished
    fdir = os.path.join(tmp, "results", "first-run")
    os.makedirs(os.path.join(fdir, "stages"))
    os.makedirs(os.path.join(fdir, "convert"))
    gg = os.path.join(fdir, "convert", "first-run-2k-r16-F16-LoRA.gguf")
    open(gg, "wb").write(b"gguf")
    json.dump({"gguf": gg}, open(os.path.join(fdir, "stages", "convert.done.json"), "w"))
    json.dump({"state": "finished"}, open(os.path.join(fdir, "status.json"), "w"))
    os.makedirs(os.path.join(fdir, "score"))
    json.dump({"datasets": {ds: {"per_task": {f"{ds}/{i}": {"base_pass": i % 2 == 0, "plus_pass": i % 3 == 0}
                                              for i in range(10)}} for ds in ew.DATASETS}},
              open(os.path.join(fdir, "score", "base.json"), "w"))
    # config: the real loop.yaml with this world's first-run folder and smaller rounds
    import yaml
    cfg = yaml.safe_load(open(os.path.join(ROOT, "experiments", "loop.yaml"), encoding="utf-8"))
    cfg.update(first_run_dir=fdir, rows_per_round=100, min_rows_per_round=30, min_free_gb=0)
    for src in cfg["sources"]:  # small fake pools: tachibana (300) and math (60) usable; banaxi and agent-session keep theirs
        if src["name"] in ("tachibana4_v4pro", "math_v4_aops"):
            src["min_pool_rows"] = 10
    cpath = os.path.join(tmp, "loop.yaml")
    yaml.safe_dump(cfg, open(cpath, "w", encoding="utf-8"), sort_keys=False)

    # the scripts the loop checks before using them (require_script)
    os.makedirs(os.path.join(tmp, "scripts"))
    for name in ("prepare_dataset.py", "train_longrun.py"):
        shutil.copyfile(os.path.join(ROOT, "scripts", name), os.path.join(tmp, "scripts", name))
    # patch the world into the modules
    L.ROOT = tmp
    L.POOLS = os.path.join(tmp, "data", "processed", "loop")
    L.RESULTS_DIR = os.path.join(tmp, "results")

    def serve(self, flags, name, coopmat=False):
        self.server = FakeServer(flags)
        w.server = self.server
        return self.server

    def unserve(self):
        self.server = None
        w.server = None

    fr.Controller.serve = serve
    fr.Controller.unserve = unserve
    fr.Controller.run_child = lambda self, cmd, log_name, stop_dir=None, monitor=None, interval=30: \
        w.run_child(self, cmd, log_name, stop_dir, monitor, interval)
    ew.chat_request = w.chat
    ew.count_tokens = lambda url, text: len(text) // 4
    gt.run_io_tests = lambda code, cases, timeout: "PASS" in code
    gt.run_assert_tests = lambda code, tests, timeout: "PASS" in code
    ew.load_problems = lambda ds: {f"{ds}/{i}": {} for i in range(10)}
    ew.generate_tasks = lambda *a, **k: {"rows": [], "wall_s": 1, "generated_tokens": 0}
    ew.sanitize_file = lambda ds, raw, out=None: raw
    ew.evaluate = lambda ds, path, parallel=8: {"base": 0.8, "plus": 0.7, "per_task": {
        f"{ds}/{i}": {"base_pass": True, "plus_pass": i % 2 == 0} for i in range(10)}}
    L.Run.capture_output = lambda self, name="run.log": None
    return w, cpath


def run_main(argv):
    saved = sys.argv
    sys.argv = ["loop.py"] + argv
    try:
        return L.main()
    finally:
        sys.argv = saved


def test_simulated_loop():
    tmp = tempfile.mkdtemp(prefix="anneal-loop-test-")
    try:
        w, cpath = setup_world(tmp)
        run_dir = os.path.join(tmp, "results", "loop")
        args = ["--config", cpath, "--run-dir", run_dir, "--max-rounds", "2"]
        # first call: r000, r001, r002 (kept), r003 paused in training
        assert run_main(args) == 0
        st = json.load(open(os.path.join(run_dir, "status.json")))
        assert st["state"] == "paused", st
        state = json.load(open(os.path.join(run_dir, "state.json")))
        assert [r["status"] for r in state["rounds"]] == ["reference", "record", "kept", "running"], state["rounds"]
        plan3 = json.load(open(os.path.join(run_dir, "rounds", "r003", "stages", "plan.done.json")))
        # second call: resumes r003 (same plan), rejects it, then r004
        assert run_main(args) == 0, open(os.path.join(run_dir, "status.json")).read()
        st = json.load(open(os.path.join(run_dir, "status.json")))
        assert st["state"] == "finished", st
        state = json.load(open(os.path.join(run_dir, "state.json")))
        status = {r["round"]: r["status"] for r in state["rounds"]}
        assert status == {0: "reference", 1: "record", 2: "kept", 3: "rejected", 4: "kept"}, state["rounds"]
        assert json.load(open(os.path.join(run_dir, "rounds", "r003", "stages", "plan.done.json"))) == plan3
        # r001 (first run): scored for the record, never the best
        r1 = state["rounds"][1]
        assert r1["gate"] == "22/40" and r1["would_keep"] is False
        # r002 from the base, kept; confirm with a fresh seed
        rd = lambda n: os.path.join(run_dir, "rounds", f"r{n:03d}")  # noqa: E731
        import yaml
        y2 = yaml.safe_load(open(os.path.join(rd(2), "train.yaml")))
        assert "init_adapter" not in y2 and y2["expect_trainable_params"] == 79691776
        # the agent-session recipe had no usable rows (100 < 500): skipped, round 3 used the default recipe
        assert state["recipes_skipped"] == ["agent-sessions"] and "v4pro-1" in state["recipes_done"]
        assert plan3["recipe"]["id"] == "default" and plan3["init"] == "best"
        y3 = yaml.safe_load(open(os.path.join(rd(3), "train.yaml")))
        assert y3["init_adapter"].replace("/", os.sep) == os.path.join(rd(2), "train", "adapter")
        # r003 rejected: adapter, checkpoints and GGUF deleted; scores, plan and report kept
        assert not os.path.exists(os.path.join(rd(3), "train", "adapter"))
        assert not os.path.exists(os.path.join(rd(3), "train", "checkpoints"))
        assert not os.path.exists(os.path.join(rd(3), "convert"))
        assert os.path.exists(os.path.join(rd(3), "report.md")) and os.path.exists(os.path.join(rd(3), "gate.seed1002.jsonl"))
        # r002 kept: adapter and GGUF kept
        assert os.path.exists(os.path.join(rd(2), "train", "adapter")) and os.path.exists(os.path.join(rd(2), "convert"))
        # r004 continued from r002 (the best), compared against r002's fresh-seed rows, and kept
        y4 = yaml.safe_load(open(os.path.join(rd(4), "train.yaml")))
        assert y4["init_adapter"].replace("/", os.sep) == os.path.join(rd(2), "train", "adapter")
        d4 = json.load(open(os.path.join(rd(4), "stages", "decide.done.json")))
        assert d4["keep"] and d4["best_round"] == 2 and d4["gate"]["wins"] == 5 and d4["gate"]["losses"] == 0
        best = state["best"]
        assert best["round"] == 4 and best["gate_seed"] == 1004 and len(best["lineage"]) == 4  # 2 sources x 2 rounds
        # fresh rows every round: no row trained twice, nothing from before the loop
        seen = set()
        for n in (2, 3, 4):
            for r in E.read_jsonl(os.path.join(rd(n), "data", "mix", "train.jsonl")):
                q = r["messages"][0]["content"]
                assert q not in seen and q != "tachibana4_v4pro q0", q
                seen.add(q)
        assert len(seen) == 300  # 89 + 11 per round; banaxi (20 < 100) and agent-session unused
        # milestone EvalPlus only on kept rounds
        assert os.path.exists(os.path.join(rd(2), "stages", "milestone.done.json"))
        assert not os.path.exists(os.path.join(rd(3), "stages", "milestone.done.json"))
        summary = open(os.path.join(run_dir, "summary.md")).read()
        assert summary.count("\n| ") >= 6 and "rejected" in summary
        assert "KEPT" in open(os.path.join(rd(4), "report.md")).read()
        check_child_commands(w.calls)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ============================================================================================ command lines and sources

class _Parsed(Exception):
    pass


def parsed_args(script, argv):
    """Run a script's main() only as far as its argparse, to prove the loop builds a command line it accepts."""
    import argparse
    import importlib
    if "datasets" not in sys.modules:
        try:
            import datasets  # noqa: F401
        except ImportError:
            stub = types.ModuleType("datasets")
            stub.load_dataset = None
            sys.modules["datasets"] = stub
    mod = importlib.import_module(os.path.splitext(script)[0])
    orig, got = argparse.ArgumentParser.parse_args, {}

    def capture(self, args=None, namespace=None):
        got["ns"] = orig(self, args, namespace)
        raise _Parsed

    argparse.ArgumentParser.parse_args = capture
    saved = sys.argv
    sys.argv = [script] + list(argv)
    try:
        mod.main()
    except _Parsed:
        return got["ns"]
    except SystemExit as exc:
        raise AssertionError(f"{script} rejected {argv}: exit {exc.code}")
    finally:
        argparse.ArgumentParser.parse_args = orig
        sys.argv = saved
    raise AssertionError(f"{script}: main() returned before parsing arguments")


def check_child_commands(calls):
    seen = set()
    for script, cmd in calls:
        assert cmd[1] == "-u", cmd
        seen.add(script)
        if script == "convert_lora_to_gguf.py":  # llama.cpp's script: same command as the first run's convert stage
            assert cmd[3:9] == ["--base", L.llama.LORA_BASE_CONFIG, "--outtype", "f16", "--outfile", cmd[8]], cmd
            continue
        ns = parsed_args(script, cmd[3:])
        if script == "public_source.py":
            assert ns.tokenizer and ns.decontaminate and ns.decontaminate_gate and ns.max_tokens == 8192, ns
        if script == "prepare_dataset.py":
            assert ns.tokenizer and ns.decontaminate and ns.decontaminate_gate and ns.max_tokens == 8192, ns
        if script == "train_longrun.py":
            assert ns.config.endswith("train.yaml"), ns
    assert seen == {"prepare_dataset.py", "public_source.py", "mix_datasets.py", "train_longrun.py",
                    "convert_lora_to_gguf.py", "val_loss.py"}, seen


def fake_transformers():
    m = types.ModuleType("transformers")

    class Tok:
        def apply_chat_template(self, msgs, tokenize=True, add_generation_prompt=False, enable_thinking=True):
            return list(range(10 + sum(len(x.get("content") or "") for x in msgs) // 4))

    m.AutoTokenizer = types.SimpleNamespace(from_pretrained=lambda name: Tok())
    return m


def run_public_source(argv):
    import public_source as ps
    saved_argv, saved_tf = sys.argv, sys.modules.get("transformers")
    sys.modules["transformers"] = fake_transformers()
    sys.argv = ["public_source.py"] + argv
    try:
        return ps.main()
    finally:
        sys.argv = saved_argv
        if saved_tf is None:
            sys.modules.pop("transformers", None)
        else:
            sys.modules["transformers"] = saved_tf


def test_public_source_formats():
    with tempfile.TemporaryDirectory() as d:
        gate_text = ("Given an undirected tree with n vertices numbered from one to n, find for every vertex the number "
                     "of vertices at even distance from it and print these counts in order separated by spaces. The "
                     "first line contains n and each of the next n minus one lines contains an edge u v.")
        E.write_jsonl(os.path.join(d, "gate_extra.jsonl"), [{"id": "cc/valid/7", "prompt": gate_text}])
        ok = {"subset": "cot", "license": "CC-BY-4.0"}

        def qa(q, a, trace="Let me think it through.", **cols):
            return dict(ok, messages=[{"role": "user", "content": q},
                                      {"role": "assistant", "content": a, "reasoning_content": trace}], **cols)
        rows = [
            qa("What is the sum of the roots of x^2 - 5x + 6?", "The roots are 2 and 3, so the sum is 5."),
            dict(qa("Same kind of question from another site?", "Answer from a share-alike site."), subset="mse"),
            dict(ok, messages=[{"from": "human", "value": "How many primes are below 20?"},
                               {"from": "gpt", "value": "<think>\n2 3 5 7 11 13 17 19\n</think>\n\nThere are 8."}]),
            dict(ok, messages=[{"role": "user", "content": "No reasoning here?"}, {"role": "assistant", "content": "Yes."}]),
            qa("A different prompt with the same answer", "The roots are 2 and 3, so the sum is 5."),
            qa(gate_text, "Root the tree and count depths by parity; the answer for v is the size of its parity class."),
            dict(ok, messages="not a conversation"),
            dict(ok, messages=[{"role": "user", "content": "Is anyone there?"}]),
            qa("Write a very long essay.", "x" * 20000),
            qa("What is 12 times 12?", "144."),
            qa("What is the derivative of x^3?", "3x^2."),
        ]
        src = os.path.join(d, "math.jsonl")
        E.write_jsonl(src, rows)
        out = os.path.join(d, "out")
        assert run_public_source(["--dataset", "t/math", "--files", "*", "--from-local", src, "--where", "subset=cot",
                                  "--where", "license=CC-BY-4.0", "--require-think", "--tokenizer", "x",
                                  "--max-tokens", "2000", "--decontaminate-gate", os.path.join(d, "gate_extra.jsonl"),
                                  "--val-fraction", "0.2", "--out", out]) == 0
        m = json.load(open(os.path.join(out, "measure.json")))
        c = m["counts"]
        assert (c["conversations"], c["filtered_where"], c["bad_format"], c["no_assistant"], c["require_think"],
                c["duplicate"], c["too_long"], c["contaminated_gate"], c["kept"]) == (11, 1, 1, 1, 1, 1, 1, 1, 4), c
        kept = E.read_jsonl(os.path.join(out, "train.jsonl")) + E.read_jsonl(os.path.join(out, "val.jsonl"))
        assert len(kept) == 4 and len(E.read_jsonl(os.path.join(out, "val.jsonl"))) == 1
        by_q = {r["messages"][0]["content"]: r["messages"] for r in kept}
        a = by_q["What is the sum of the roots of x^2 - 5x + 6?"][1]
        assert a == {"role": "assistant", "content": "<think>\nLet me think it through.\n</think>\n\nThe roots are 2 "
                                                     "and 3, so the sum is 5."}, a
        assert [x["role"] for x in by_q["How many primes are below 20?"]] == ["user", "assistant"]
        assert "contaminated_gate | 1" in open(os.path.join(out, "data_card.md")).read()

        # agent sessions, one row per step
        call = lambda i, name, arg: {"id": f"c{i}", "type": "function",  # noqa: E731
                                     "function": {"name": name, "arguments": json.dumps(arg)}}
        session = [{"role": "system", "content": "You are a coding agent."},
                   {"role": "user", "content": "The test test_parse fails in this repo; fix it."},
                   {"role": "assistant", "content": "", "tool_calls": [call(1, "bash", {"cmd": "pytest -x"})]},
                   {"role": "tool", "tool_call_id": "c1", "content": "FAILED test_parse - KeyError: 'id'"},
                   {"role": "assistant", "content": "", "tool_calls": [call(2, "unknown_tool", {})]},
                   {"role": "tool", "tool_call_id": "c2", "content": "error"},
                   {"role": "assistant", "content": "parse() read row['id'] before checking it exists; it now uses "
                                                    "row.get('id') and the test passes."}]
        short = [{"role": "user", "content": "thanks"}, {"role": "assistant", "content": "ok"}]
        src2 = os.path.join(d, "agent.jsonl")
        E.write_jsonl(src2, [{"messages": session}, {"messages": short}])
        out2 = os.path.join(d, "out2")
        assert run_public_source(["--dataset", "t/agent", "--files", "*", "--from-local", src2, "--turns", "each",
                                  "--drop-tool-name", "unknown_tool", "--min-target-chars", "20", "--tokenizer", "x",
                                  "--val-fraction", "0.5", "--out", out2]) == 0
        c = json.load(open(os.path.join(out2, "measure.json")))["counts"]
        assert (c["candidate_rows"], c["dropped_tool_name"], c["short_target"], c["kept"]) == (4, 1, 1, 2), c
        tr, va = E.read_jsonl(os.path.join(out2, "train.jsonl")), E.read_jsonl(os.path.join(out2, "val.jsonl"))
        assert sorted([len(tr), len(va)]) == [0, 2]  # both steps of the session on the same side
        steps = sorted(tr + va, key=lambda r: len(r["messages"]))
        assert len(steps[0]["messages"]) == 3 and steps[0]["messages"][-1]["tool_calls"][0]["function"]["name"] == "bash"
        assert len(steps[1]["messages"]) == 7 and steps[1]["messages"][3] == {"role": "tool", "content": "FAILED "
                                                                              "test_parse - KeyError: 'id'",
                                                                              "tool_call_id": "c1"}


def test_staged_scripts():
    saved_wd = W.WD_DIR
    try:
        _staged_scripts()
    finally:
        W.WD_DIR = saved_wd


def _staged_scripts():
    with tempfile.TemporaryDirectory() as d:
        W.WD_DIR = os.path.join(d, "results", "watchdog")
        first = os.path.join(d, "results", "first-run")
        os.makedirs(os.path.join(first, "stages"))
        os.makedirs(os.path.join(d, "scripts"))
        os.makedirs(os.path.join(d, "staged"))
        for name in ("prepare_dataset.py", "train_longrun.py"):
            open(os.path.join(d, "scripts", name), "w").write(f"# old {name}\n")
            open(os.path.join(d, "staged", name), "w").write(f"# new {name}\n")
        # the loop refuses the old versions
        saved_root = L.ROOT
        L.ROOT = d
        try:
            for name, marker in (("prepare_dataset.py", "--decontaminate-gate"), ("train_longrun.py", "def load_init_adapter")):
                try:
                    L.require_script(name, marker)
                    raise AssertionError(f"old {name} accepted")
                except L.Waiting as w:
                    assert name in str(w)
        finally:
            L.ROOT = saved_root
        assert W.install_staged(d, first) == []  # first run still generating: nothing replaced
        json.dump({}, open(os.path.join(first, "stages", "generate.done.json"), "w"))
        assert W.install_staged(d, first, dry_run=True) == []
        assert open(os.path.join(d, "scripts", "prepare_dataset.py")).read() == "# old prepare_dataset.py\n"
        assert W.install_staged(d, first) == ["prepare_dataset.py"]  # train_longrun waits for the train stage
        assert open(os.path.join(d, "scripts", "prepare_dataset.py")).read() == "# new prepare_dataset.py\n"
        assert open(os.path.join(d, "scripts", "train_longrun.py")).read() == "# old train_longrun.py\n"
        kept = os.listdir(os.path.join(d, "staged", "replaced"))
        assert len(kept) == 1 and open(os.path.join(d, "staged", "replaced", kept[0])).read() == "# old prepare_dataset.py\n"
        json.dump({}, open(os.path.join(first, "stages", "train.done.json"), "w"))
        assert W.install_staged(d, first) == ["train_longrun.py"]
        assert W.install_staged(d, first) == []  # idempotent
        assert "installed staged/train_longrun.py" in open(os.path.join(W.WD_DIR, "watchdog.log")).read()
        # a waiting loop is resumed as soon as the scripts are in
        training = {"active": None, "runs": [{"tag": "first-run", "state": "finished"}, {"tag": "loop", "state": "finished"}]}
        waiting = {"state": "waiting", "updated": 1000.0, "config_mtime": 50.0, "detail": "old scripts"}
        assert W.decide(training, 1100.0, {"restarts": []}, {"state": "finished"}, waiting, True, 50.0)[0] is None
        a, tag, why = W.decide(training, 1100.0, {"restarts": []}, {"state": "finished"}, waiting, True, 50.0, True)
        assert (a, tag) == ("resume", "loop") and "staged scripts" in why


def main():
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS {name}")
        except Exception:
            import traceback
            failed += 1
            print(f"FAIL {name}")
            traceback.print_exc()
    print(f"{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    threading.excepthook = lambda a: None
    sys.exit(main())
