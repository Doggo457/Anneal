"""Tests for the improvement loop without a GPU, a model or Windows: the keep rule, the overlap check, scoring,
row bookkeeping, the watchdog's decisions, and a full simulated loop with every subprocess and the model faked: base,
first run, kept rounds, a skipped recipe, a pause and resume in the middle of training, rejected rows rolling forward
until the stack is dropped, a fresh-seed confirm that fails, the retrain-vs-continue round, a rank-64 round whose shape
later rounds inherit, the two-pass round when the pools run short, and the wait after it.

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
    c = L.plan_counts(cfg, {}, st, {"a": 300, "c": 60, "f": 100})  # b unusable: out of the mix
    assert c == {"a": 89, "c": 11}, c
    assert L.plan_counts(cfg, {"mix": {"f": 1.0}}, st, {"a": 300}) == {}
    assert L.plan_counts(cfg, {}, st, {"a": 300, "c": 60}, rows=50) == {"a": 45, "c": 5}
    # the ratio is held when a source runs low: the round shrinks instead of c taking a's share
    assert L.plan_counts(cfg, {}, {"offsets": {"a": 290}}, {"a": 300, "c": 600}) == {"a": 10, "c": 1}
    assert L.plan_counts(cfg, {}, {"offsets": {"a": 300}}, {"a": 300, "c": 600}) == {}
    # a source in the same group stands in (b for a), but c still never takes the code share
    grp = {"rows_per_round": 100, "sources": [{"name": "a", "share": 0.85, "group": "code"},
                                              {"name": "b", "share": 0.05, "group": "code"}, {"name": "c", "share": 0.10}]}
    c = L.plan_counts(grp, {}, {"offsets": {"a": 290}}, {"a": 300, "b": 50, "c": 1000})
    assert c == {"a": 10, "b": 49, "c": 7} and sum(c.values()) == 66, c  # 59 code rows : 7 math = the 90:10 ratio
    assert L.plan_counts(grp, {}, {"offsets": {"a": 300, "b": 50}}, {"a": 300, "b": 50, "c": 1000}) == {}
    c = L.plan_counts(grp, {}, {"offsets": {"b": 50}}, {"a": 3000, "b": 50, "c": 1000})  # b used up: a covers it
    assert c == {"a": 90, "c": 10}, c
    assert L.apportion(10, {"x": 1, "y": 1, "z": 1}, {"x": 2, "y": 100, "z": 100}) == {"x": 2, "y": 4, "z": 4}
    assert L.spec_hash({"name": "a", "share": 1, "group": "g"}) == L.spec_hash({"name": "a", "share": 0.5})
    shape = {"lora_r": 16, "lora_alpha": 32, "target_modules": ["q"]}
    assert L.choose_init({}, {"adapter": None}, shape) == "base"
    assert L.choose_init({}, {"adapter": "x", "lora_shape": shape}, shape) == "best"
    assert L.choose_init({}, {"adapter": "x", "lora_shape": shape}, dict(shape, lora_r=32)) == "base_lineage"
    assert L.choose_init({"init": "base_lineage"}, {"adapter": "x", "lora_shape": shape}, shape) == "base_lineage"


def test_recipes_and_shape():
    tcfg = {"train": {"lora_r": 16, "lora_alpha": 32, "target_modules": ["q"], "expect_trainable_params": 7, "lr": 1}}
    r64 = {"lora_r": 64, "lora_alpha": 128, "target_modules": ["q", "z"], "expect_trainable_params": 9}
    assert L.effective_train(tcfg, {}) == tcfg["train"]
    assert L.effective_train(tcfg, {"train": dict(r64, lr=2)}) == dict(r64, lr=2)
    assert L.effective_train(tcfg, {}, {"train_shape": r64}) == dict(r64, lr=1)  # a kept rank-up carries on
    t = L.effective_train(tcfg, {"train": {"lora_r": 32}}, {"train_shape": r64})  # the recipe's own shape wins;
    assert t == {"lora_r": 32, "lora_alpha": 32, "target_modules": ["q"], "lr": 1}, t  # no stale parameter count
    st = {"rounds": [{"status": "kept"}, {"status": "rejected"}], "recipes_done": [], "best": {"init": "base"}}
    assert L.recipe_ready({}, st) and not L.recipe_ready({"after_kept": 2}, st)
    assert L.recipe_ready({"after_kept": 3, "after_dropped_stacks": 1}, dict(st, stacks_dropped=[{}]))
    assert not L.recipe_ready({"when_best_continued": True}, st)
    assert L.recipe_ready({"when_best_continued": True}, dict(st, best={"init": "best"}))
    cfg = {"recipes": [{"id": "later", "after_kept": 2}, {"id": "short", "when_pools_short": True}, {"id": "now"}],
           "default_recipe": {"id": "default"}}
    assert L.pick_recipe(cfg, st)["id"] == "now"
    assert L.pick_recipe(cfg, st, pools_short=True)["id"] == "short"
    assert L.pick_recipe(cfg, dict(st, recipes_done=["now"]))["id"] == "default"
    assert L.takes_stack({}) and L.takes_stack({"train": {"epochs": 2}})
    assert not L.takes_stack({"mix": {"f": 1}}) and not L.takes_stack({"rows": 0, "init": "base_lineage"})


def test_pick_replay():
    lineage = [["a", [0, 1, 2, 3]], ["b", [5, 6]], ["a", [10, 11, 12, 13]]]
    r = L.pick_replay(lineage, 5, seed=3)
    assert r == L.pick_replay(lineage, 5, seed=3) and r != L.pick_replay(lineage, 5, seed=4)  # reproducible per round
    flat = [(n, i) for n, ix in r for i in ix]
    assert len(flat) == 5 == len(set(flat)) and set(flat) <= {(n, i) for n, ix in lineage for i in ix}
    assert [n for n, _ in r] == sorted({n for n, _ in r}, key=["a", "b"].index)  # grouped by source, lineage order
    assert sum(len(ix) for _, ix in L.pick_replay(lineage, 99, seed=1)) == 10 and L.pick_replay(lineage, 0, 1) == []
    assert L.pick_replay([], 5, 1) == []


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
PASSING = {None: 20, "first-run": 22, "r002": 30, "r003": 18, "r004": 35, "r005": 36, "r006": 33, "r007": 40,
           "r008": 30, "r009": 40, "r010": 40, "r011": 40}  # gate items 0..k-1 pass
CONFIRM_PASSING = {"r007": 34}  # on its fresh-seed confirm (seed 1000 + round) r007 does worse than when it was kept


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
                k = PASSING[m]
                if m in CONFIRM_PASSING and seed // 100000 == 1000 + int(m[1:]):
                    k = CONFIRM_PASSING[m]
                ok = i < k
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
            n = {"tachibana4_v4pro": 720, "banaxi_v4pro": 20, "math_v4_aops": 120, "agent_sessions": 100}[name]
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
    # a row the reworded-copy check flagged (scripts/paraphrase_check.py): the loop must never hand it out
    os.makedirs(os.path.join(ev, "paraphrase"))
    with open(os.path.join(ev, "paraphrase", "blocked_keys.txt"), "w", encoding="utf-8") as fh:
        fh.write(L.row_key({"messages": [{"role": "user", "content": "tachibana4_v4pro q5"}, {"role": "assistant", "content": "a"}]}) + "\n")
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
    cfg.update(first_run_dir=fdir, rows_per_round=100, min_rows_per_round=30, min_free_gb=0, stack_max_new_rows=300)
    cfg["rules"]["confirm_net_win"] = True  # exercised here; off in loop.yaml until chosen
    for src in cfg["sources"]:  # small fake pools: tachibana (720) and math (120) usable; banaxi and agent-session keep theirs
        if src["name"] in ("tachibana4_v4pro", "math_v4_aops"):
            src["min_pool_rows"] = 10
    cpath = os.path.join(tmp, "loop.yaml")
    yaml.safe_dump(cfg, open(cpath, "w", encoding="utf-8"), sort_keys=False)

    # the scripts the loop checks before using them (require_script): the versions the loop needs
    os.makedirs(os.path.join(tmp, "scripts"))
    open(os.path.join(tmp, "scripts", "prepare_dataset.py"), "w").write('ap.add_argument("--decontaminate-gate")\n')
    open(os.path.join(tmp, "scripts", "train_longrun.py"), "w").write("def load_init_adapter(model, path, lora): ...\n")
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
        _simulated_loop(tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _simulated_loop(tmp):
    import yaml
    w, cpath = setup_world(tmp)
    run_dir = os.path.join(tmp, "results", "loop")
    args = ["--config", cpath, "--run-dir", run_dir, "--max-rounds", "2"]
    rd = lambda n: os.path.join(run_dir, "rounds", f"r{n:03d}")  # noqa: E731
    done = lambda n, stage: json.load(open(os.path.join(rd(n), "stages", f"{stage}.done.json")))  # noqa: E731
    ty = lambda n: yaml.safe_load(open(os.path.join(rd(n), "train.yaml")))  # noqa: E731
    qs = lambda n: [r["messages"][0]["content"] for r in E.read_jsonl(os.path.join(rd(n), "data", "mix", "train.jsonl"))]  # noqa: E731
    n_rows = lambda parts: sum(len(ix) for _, ix in parts)  # noqa: E731
    # first call: r000, r001, r002 (kept), r003 paused in training
    assert run_main(args) == 0
    st = json.load(open(os.path.join(run_dir, "status.json")))
    assert st["state"] == "paused", st
    state = json.load(open(os.path.join(run_dir, "state.json")))
    assert [r["status"] for r in state["rounds"]] == ["reference", "record", "kept", "running"], state["rounds"]
    plan3 = done(3, "plan")
    # second call: resumes r003 (same plan), rejects it, then r004 takes r003's rows plus fresh ones and is kept
    assert run_main(args) == 0, open(os.path.join(run_dir, "status.json")).read()
    st = json.load(open(os.path.join(run_dir, "status.json")))
    assert st["state"] == "finished", st
    state = json.load(open(os.path.join(run_dir, "state.json")))
    status = {r["round"]: r["status"] for r in state["rounds"]}
    assert status == {0: "reference", 1: "record", 2: "kept", 3: "rejected", 4: "kept"}, state["rounds"]
    assert done(3, "plan") == plan3
    # r001 (first run): scored for the record, never the best
    r1 = state["rounds"][1]
    assert r1["gate"] == "22/40" and r1["would_keep"] is False
    # r002 from the base, kept; confirm with a fresh seed
    y2 = ty(2)
    assert "init_adapter" not in y2 and y2["expect_trainable_params"] == 79691776
    assert done(2, "confirm")["confirmed"] and done(2, "confirm")["vs_previous_best"]["wins"] == 10
    # the agent-session recipe had no usable rows (100 < 500): skipped; round 3 used the default recipe (the twin and rank-64
    # recipes wait for their conditions, two-pass for short pools)
    assert state["recipes_skipped"] == ["agent-sessions"] and state["recipes_done"] == ["v4pro-1", "agent-sessions"]
    assert plan3["recipe"]["id"] == "default" and plan3["init"] == "best"
    assert n_rows(plan3["replay"]) == 25 and plan3["rows"] == 125 and plan3["rolled"] == []  # 25% of 100 fresh rows
    assert ty(3)["init_adapter"].replace("/", os.sep) == os.path.join(rd(2), "train", "adapter")
    # r003 rejected: adapter, checkpoints and GGUF deleted; scores, plan and report kept; its rows roll forward
    assert not os.path.exists(os.path.join(rd(3), "train", "adapter"))
    assert not os.path.exists(os.path.join(rd(3), "train", "checkpoints"))
    assert not os.path.exists(os.path.join(rd(3), "convert"))
    assert os.path.exists(os.path.join(rd(3), "gate.seed1002.jsonl"))
    assert "100 rows roll into the next round (attempt 2)" in open(os.path.join(rd(3), "report.md")).read()
    # r002 kept: adapter and GGUF kept
    assert os.path.exists(os.path.join(rd(2), "train", "adapter")) and os.path.exists(os.path.join(rd(2), "convert"))
    # r004 continued r002 on r003's rows + 100 fresh + 25% replay of the 200 new rows, compared against r002's
    # fresh-seed rows, and kept
    plan4 = done(4, "plan")
    assert plan4["rolled"] == [[n, ix] for n, ix in plan3["slices"].items()] and plan4["stack_attempt"] == 2
    assert plan4["stack_rounds"] == [3, 4] and n_rows(plan4["replay"]) == 50 and plan4["rows"] == 250, plan4
    assert ty(4)["init_adapter"].replace("/", os.sep) == os.path.join(rd(2), "train", "adapter")
    d4 = done(4, "decide")
    assert d4["keep"] and d4["best_round"] == 2 and d4["gate"]["wins"] == 5 and d4["gate"]["losses"] == 0
    best = state["best"]
    assert best["round"] == 4 and best["gate_seed"] == 1004 and best["init"] == "best" and state.get("stack") is None
    assert len(best["lineage"]) == 6 and n_rows(best["lineage"]) == 300  # r002's, r003's and r004's fresh rows
    # fresh rows every round, nothing from before the loop; replay only from the best's rows
    q2, q3, q4 = (set(qs(n)) for n in (2, 3, 4))
    assert [len(qs(n)) for n in (2, 3, 4)] == [100, 125, 250] and all(len(set(qs(n))) == len(qs(n)) for n in (2, 3, 4))
    assert len(q3 & q2) == 25 and (q3 - q2) <= q4 and len(q4 & q2) == 50, (len(q3 & q2), len(q4 & q2))
    fresh = q2 | (q3 - q2) | (q4 - q2)
    assert len(fresh) == 300 and "tachibana4_v4pro q0" not in fresh  # 89 + 11 per round; banaxi and agent-session unused
    assert os.path.exists(os.path.join(rd(2), "stages", "milestone.done.json"))
    assert not os.path.exists(os.path.join(rd(3), "stages", "milestone.done.json"))
    assert "KEPT" in open(os.path.join(rd(4), "report.md")).read()

    # third call: r005 (retrain-vs-continue), r006-r008 (a stack rejected three times, one confirm failing),
    # r009 (rank 64 after the dropped stack), r010 (inherits rank 64)
    assert run_main(["--config", cpath, "--run-dir", run_dir, "--max-rounds", "6"]) == 0
    state = json.load(open(os.path.join(run_dir, "state.json")))
    status = {r["round"]: r["status"] for r in state["rounds"]}
    assert [status[n] for n in range(5, 11)] == ["rejected"] * 4 + ["kept", "rejected"], state["rounds"]
    # r005: every row r004 was built from, retrained from the base, scored on r004's confirm seed
    p5 = done(5, "plan")
    assert p5["recipe"]["id"] == "retrain-vs-continue" and p5["init"] == "base_lineage" and p5["slices"] == {}
    assert p5["lineage"] == best["lineage"] and p5["rows"] == 300 and p5["replay"] == [] and p5["rolled"] == []
    assert "init_adapter" not in ty(5) and set(qs(5)) == q2 | q3 | q4 and len(qs(5)) == 300
    d5 = done(5, "decide")
    assert d5["best_round"] == 4 and d5["gate"]["wins"] == 1 and not d5["keep"]
    assert "Paired comparison" in open(os.path.join(rd(5), "report.md")).read()
    # r006 rejected (stack of 100), r007 kept by the gate but its fresh-seed confirm lost, r008 at the 300-row cap
    p6, p7, p8 = done(6, "plan"), done(7, "plan"), done(8, "plan")
    assert (p6["rows"], p7["rows"], p8["rows"]) == (125, 250, 375) and p8["stack_attempt"] == 3
    assert n_rows(p8["rolled"]) == 200 and p8["stack_rounds"] == [6, 7, 8]
    assert done(7, "decide")["keep"] and not done(7, "confirm")["confirmed"]
    r7 = [r for r in state["rounds"] if r["round"] == 7][0]
    assert any("fresh-seed confirm" in x for x in r7["reasons"]) and not os.path.exists(os.path.join(rd(7), "convert"))
    assert not os.path.exists(os.path.join(rd(7), "stages", "milestone.done.json"))
    assert [(s["rows"], s["attempts"], s["rounds"]) for s in state["stacks_dropped"]] == [(300, 3, [6, 7, 8])]
    # r009: rank 64 on the DeltaNet projections too, retrained from the base on every kept row + 100 fresh rows
    p9, y9 = done(9, "plan"), ty(9)
    assert p9["recipe"]["id"] == "rank64-dn" and p9["init"] == "base_lineage" and p9["rows"] == 400
    assert y9["lora_r"] == 64 and y9["expect_trainable_params"] == 435159040 and "init_adapter" not in y9
    assert "in_proj_qkv" in y9["target_modules"] and "out_proj" in y9["target_modules"]
    best = state["best"]
    assert best["round"] == 9 and best["lora_shape"]["lora_r"] == 64 and n_rows(best["lineage"]) == 400
    # r010: the default recipe continues the rank-64 adapter instead of retraining a rank-16 one from the base
    p10, y10 = done(10, "plan"), ty(10)
    assert p10["recipe"]["id"] == "default" and p10["init"] == "best" and p10["rows"] == 125
    assert y10["lora_r"] == 64 and y10["expect_trainable_params"] == 435159040
    assert y10["init_adapter"].replace("/", os.sep) == os.path.join(rd(9), "train", "adapter")
    assert state["stack"]["rows"] == 100 and state["stack"]["rounds"] == [10]

    # fourth call: too few fresh rows for a round: two passes over r010's rows + the last fresh ones, then wait
    assert run_main(["--config", cpath, "--run-dir", run_dir, "--max-rounds", "3"]) == 0
    st = json.load(open(os.path.join(run_dir, "status.json")))
    assert st["state"] == "waiting" and "unused public rows" in st["detail"], st
    p11, y11 = done(11, "plan"), ty(11)
    assert p11["recipe"]["id"] == "two-pass" and y11["epochs"] == 2 and y11["lora_r"] == 64
    assert y11["init_adapter"].replace("/", os.sep) == os.path.join(rd(9), "train", "adapter")
    n_fresh11 = sum(len(ix) for ix in p11["slices"].values())
    assert 0 < n_fresh11 < 30 and n_rows(p11["rolled"]) == 100 and p11["stack_attempt"] == 2
    state = json.load(open(os.path.join(run_dir, "state.json")))
    assert "two-pass" in state["recipes_done"] and state["stack"]["rows"] == 100 + n_fresh11
    assert state["rounds"][-1]["round"] == 12 and state["rounds"][-1]["status"] == "running"  # waits to be planned
    assert not os.path.exists(os.path.join(rd(12), "stages", "plan.done.json"))
    summary = open(os.path.join(run_dir, "summary.md")).read()
    assert summary.count("\n| ") >= 13 and "Rows waiting to roll into the next round" in summary
    # the blocked row was skipped in every round and never recorded as trained on
    trained = set()
    for n in range(2, 12):
        mixf = os.path.join(rd(n), "data", "mix", "train.jsonl")
        if os.path.exists(mixf):
            trained |= set(qs(n))
    assert len(trained) > 300 and "tachibana4_v4pro q5" not in trained and "tachibana4_v4pro q6" in trained
    used_keys = open(os.path.join(run_dir, "used_keys.txt")).read()
    assert L.row_key({"messages": [{"role": "user", "content": "tachibana4_v4pro q5"}, {"role": "assistant", "content": "a"}]}) not in used_keys
    check_child_commands(w.calls)


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
    # On the machine, prepare_dataset.py and train_longrun.py wait in staged/ until the first run is done with them:
    # check the loop's commands against the versions it will run.
    staged = os.path.join(ROOT, "staged", script)
    if os.path.exists(staged):
        import importlib.util
        spec = importlib.util.spec_from_file_location(os.path.splitext(script)[0] + "_staged", staged)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    else:
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
            for x in msgs:  # like Qwen3.8's template, which takes tool-call arguments only as a mapping
                for c in x.get("tool_calls") or []:
                    if not isinstance(c["function"]["arguments"], dict):
                        raise ValueError(f'Tool call arguments for function "{c["function"]["name"]}" were passed '
                                         f'as a JSON string. Parse them into an object before calling '
                                         f'apply_chat_template.')
                if "TEMPLATE_REFUSES" in (x.get("content") or ""):
                    raise RuntimeError("template refused this row")
            n = 10 + sum(len(x.get("content") or "") for x in msgs) // 4
            return list(range(n)) if tokenize else "t" * n

        def __call__(self, text, add_special_tokens=True):
            return {"input_ids": list(range(len(text)))}

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
        bad_args = [{"role": "user", "content": "List the files in the repository root please."},
                    {"role": "assistant", "content": "", "tool_calls": [
                        {"id": "c9", "type": "function", "function": {"name": "list_dir", "arguments": "{not json"}}]}]
        list_args = [{"role": "user", "content": "List the files in the src folder please."},
                     {"role": "assistant", "content": "", "tool_calls": [
                         {"id": "c8", "type": "function", "function": {"name": "list_dir", "arguments": "[1, 2]"}}]}]
        refused = [{"role": "user", "content": "TEMPLATE_REFUSES this one"},
                   {"role": "assistant", "content": "An answer long enough to pass the length filter."}]
        redacted = [{"role": "user", "content": "Fix the failing build in this project please."},
                    {"role": "assistant", "content": "", "tool_calls": [
                        {"id": "c7", "type": "function", "function": {"name": "bash", "arguments": json.dumps(
                            {"command": "make test", "env": {"CI": "1", "dir": "<directory_path>"}})}}]}]
        src2 = os.path.join(d, "agent.jsonl")
        E.write_jsonl(src2, [{"messages": session}, {"messages": short}, {"messages": bad_args},
                             {"messages": list_args}, {"messages": refused}, {"messages": redacted}])
        out2 = os.path.join(d, "out2")
        assert run_public_source(["--dataset", "t/agent", "--files", "*", "--from-local", src2, "--turns", "each",
                                  "--drop-tool-name", "unknown_tool", "--min-target-chars", "20", "--tokenizer", "x",
                                  "--val-fraction", "0.5", "--out", out2]) == 0
        m2 = json.load(open(os.path.join(out2, "measure.json")))
        c = m2["counts"]
        assert (c["candidate_rows"], c["bad_format"], c["dropped_tool_name"], c["placeholder_args"], c["short_target"],
                c["template_error"], c["kept"]) == (6, 2, 1, 1, 1, 1, 2), c
        assert m2["template_errors"] == [{"error": "RuntimeError: template refused this row", "first_conversation": 4}]
        tr, va = E.read_jsonl(os.path.join(out2, "train.jsonl")), E.read_jsonl(os.path.join(out2, "val.jsonl"))
        assert sorted([len(tr), len(va)]) == [0, 2]  # both steps of the session on the same side
        steps = sorted(tr + va, key=lambda r: len(r["messages"]))
        assert len(steps[0]["messages"]) == 3 and steps[0]["messages"][-1]["tool_calls"][0]["function"] == {
            "name": "bash", "arguments": '{"cmd": "pytest -x"}'}  # stored as a JSON string, rendered as an object
        assert len(steps[1]["messages"]) == 7 and steps[1]["messages"][3] == {"role": "tool", "content": "FAILED "
                                                                              "test_parse - KeyError: 'id'",
                                                                              "tool_call_id": "c1"}
        # the session's last step follows its unknown_tool call: counted, and dropped with --clean-context
        assert m2["sessions_by_redaction"] == {"no_calls": 2, "clean": 0, "mixed": 1, "redacted": 1}, m2
        assert m2["kept_with_redacted_context"] == 1 and c["redacted_context"] == 0
        assert m2["redacted_step_text_chars"] == {"steps": 2, "with_200_plus": 0, "median": 0, "total": 0}
        out3 = os.path.join(d, "out3")
        assert run_public_source(["--dataset", "t/agent", "--files", "*", "--from-local", src2, "--turns", "each",
                                  "--drop-tool-name", "unknown_tool", "--min-target-chars", "20", "--clean-context",
                                  "--tokenizer", "x", "--measure-only", "--out", out3]) == 0
        m3 = json.load(open(os.path.join(out3, "measure.json")))
        assert (m3["counts"]["redacted_context"], m3["counts"]["kept"], m3["kept_with_redacted_context"]) == (1, 1, 0)
        assert m3["kept_by_target_kind"] == {"tool_call": 1}, m3
        assert m2["kept_sessions"] == 1 and m2["kept_target_tools"] == [["bash", 1]], m2
        assert m3["kept_sessions"] == 1 and m3["kept_target_tools"] == [["bash", 1]], m3
        # one block of the file, and targets that would teach a mask
        out5 = os.path.join(d, "out5")
        assert run_public_source(["--dataset", "t/agent", "--files", "*", "--from-local", src2, "--turns", "each",
                                  "--drop-tool-name", "unknown_tool", "--min-target-chars", "20", "--row-range", "0:2",
                                  "--drop-target-regex", r"pytest|\[REDACTED\]", "--tokenizer", "x", "--measure-only",
                                  "--out", out5]) == 0
        c5 = json.load(open(os.path.join(out5, "measure.json")))["counts"]
        assert (c5["conversations"], c5["outside_range"], c5["masked_target"], c5["short_target"], c5["kept"]) == \
            (2, 0, 1, 1, 1), c5
        out6 = os.path.join(d, "out6")
        assert run_public_source(["--dataset", "t/agent", "--files", "*", "--from-local", src2, "--turns", "each",
                                  "--min-target-chars", "20", "--row-range", "4:", "--tokenizer", "x", "--measure-only",
                                  "--out", out6]) == 0
        c6 = json.load(open(os.path.join(out6, "measure.json")))["counts"]
        assert (c6["conversations"], c6["outside_range"], c6["template_error"], c6["placeholder_args"]) == (2, 4, 1, 1), c6
        # where the kept steps sit in their session; a turn the set marks with loss_mask 0 is never a target
        assert m2["kept_target_position"] == {"median_step": 3, "p90_step": 3, "median_share": 1.0, "p90_share": 1.0}, m2
        lm = [{"role": "user", "content": "Fix the flaky test in the scheduler module please."},
              {"role": "assistant", "content": "", "loss_mask": 0, "tool_calls": [call(1, "bash", {"cmd": "pytest -x"})]},
              {"role": "tool", "content": "1 failed", "loss_mask": 0},
              {"role": "assistant", "content": "", "loss_mask": 1, "tool_calls": [call(2, "bash", {"cmd": "git diff"})]}]
        src4 = os.path.join(d, "lossmask.jsonl")
        E.write_jsonl(src4, [{"messages": lm}])
        out7 = os.path.join(d, "out7")
        assert run_public_source(["--dataset", "t/agent", "--files", "*", "--from-local", src4, "--turns", "each",
                                  "--respect-loss-mask", "--tokenizer", "x", "--measure-only", "--out", out7]) == 0
        m7 = json.load(open(os.path.join(out7, "measure.json")))
        assert (m7["counts"]["loss_masked"], m7["counts"]["kept"], m7["kept_target_position"]["median_step"]) == \
            (1, 1, 2), m7


def test_tool_call_rows():
    """Agent rows keep tool-call arguments as JSON strings on disk; everything that renders them through Qwen3.8's
    template (anneal/data.py template_ids / encode_chat, mix_datasets.py token counts) parses them first."""
    from anneal.data import chat_messages, encode_chat
    plain = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello there"}]
    assert chat_messages(plain) is plain
    call = {"id": "c1", "type": "function", "function": {"name": "list_dir", "arguments": '{"path": "src"}'}}
    # the shape load_dataset gives a mixed file: missing optional keys come back as None
    arrow = [{"role": "user", "content": "List src.", "tool_calls": None, "tool_call_id": None, "name": None},
             {"role": "assistant", "content": "", "tool_calls": [call], "tool_call_id": None, "name": None},
             {"role": "tool", "content": "a.py b.py", "tool_calls": None, "tool_call_id": "c1", "name": "list_dir"},
             {"role": "assistant", "content": "src holds a.py and b.py.", "tool_calls": None, "tool_call_id": None,
              "name": None}]
    before = json.dumps(arrow)
    got = chat_messages(arrow)
    assert json.dumps(arrow) == before  # the input is not changed
    assert got[0] == {"role": "user", "content": "List src."}
    assert got[1]["tool_calls"][0] == {"id": "c1", "type": "function",
                                       "function": {"name": "list_dir", "arguments": {"path": "src"}}}
    assert got[2] == {"role": "tool", "content": "a.py b.py", "tool_call_id": "c1", "name": "list_dir"}
    tok = fake_transformers().AutoTokenizer.from_pretrained("x")
    enc = encode_chat(tok, arrow, max_len=1000)
    assert enc is not None and enc["n_trained"] > 0
    bad = [dict(arrow[1], tool_calls=[{"function": {"name": "f", "arguments": "{oops"}}]), arrow[3]]
    try:
        chat_messages(bad)
        raise AssertionError("bad arguments accepted")
    except ValueError as e:
        assert "'f'" in str(e)

    import mix_datasets
    with tempfile.TemporaryDirectory() as d:
        for name, rows in (("agent", [{"messages": arrow[:2]}, {"messages": arrow}]), ("code", [{"messages": plain}])):
            os.makedirs(os.path.join(d, name))
            E.write_jsonl(os.path.join(d, name, "train.jsonl"), rows)
            E.write_jsonl(os.path.join(d, name, "val.jsonl"), [])
        saved_argv, saved_tf = sys.argv, sys.modules.get("transformers")
        sys.modules["transformers"] = fake_transformers()
        sys.argv = ["mix_datasets.py", "--source", f"agent={os.path.join(d, 'agent')}:all", "--source",
                    f"code={os.path.join(d, 'code')}:all", "--tokenizer", "x", "--out", os.path.join(d, "mix")]
        try:
            mix_datasets.main()
        finally:
            sys.argv = saved_argv
            if saved_tf is None:
                sys.modules.pop("transformers", None)
            else:
                sys.modules["transformers"] = saved_tf
        mixed = E.read_jsonl(os.path.join(d, "mix", "train.jsonl"))
        assert len(mixed) == 3
        stored = [c["function"]["arguments"] for r in mixed for m in r["messages"] for c in m.get("tool_calls") or []]
        assert stored == ['{"path": "src"}'] * 2, stored
        assert "| agent |" in open(os.path.join(d, "mix", "data_card.md")).read()


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


def test_build_select_oci():
    import build_evalsets as B
    words = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron pi rho sigma tau".split()

    def text(i):  # 40 distinct-ish words per problem
        return " ".join(f"{w}{i}" for w in words) + " " + " ".join(f"word{i}x{j}" for j in range(21))
    pool = [{"id": f"oci/{i}", "kind": "oci", "raw_prompt": text(i), "prompt": text(i)} for i in range(40)]
    ep = E.OverlapIndex([("evalplus/x", text(3))])              # problem 3 is an EvalPlus near-copy
    harness = lambda batch: [c for c in batch if c["id"] != "oci/5"]  # noqa: E731  (5's reference fails)
    gen_pool = ["Please solve: " + text(i) for i in (0, 1, 7, 12)]   # the first run's shards hold 0, 1, 7 and 12
    scans = []

    def pool_texts():
        scans.append(1)
        return iter(gen_pool)
    kept, n = B.select_oci(pool, 10, ep, harness, pool_texts, chunk=2, margin=0, log=lambda m: None)
    ids = [c["id"] for c in kept]
    assert len(ids) == 10 and not {"oci/0", "oci/1", "oci/3", "oci/5", "oci/7", "oci/12"} & set(ids), ids
    assert ids == [f"oci/{i}" for i in (2, 4, 6, 8, 9, 10, 11, 13, 14, 15)], ids
    assert n["evalplus_overlap"] == 1 and n["first_run_pool_overlap"] == 4 and len(scans) == 2, (n, scans)  # one top-up after the first scan


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
