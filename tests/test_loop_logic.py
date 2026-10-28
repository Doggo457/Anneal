"""Tests for the improvement loop without a GPU, a model or Windows: the keep rule, the overlap check, scoring,
row bookkeeping, the watchdog's decisions, and a full simulated loop with every subprocess and the model faked: base,
first run, kept rounds, a skipped recipe, a pause and resume in the middle of training, rejected rows rolling forward
until the stack is dropped, a fresh-seed confirm that fails, the retrain-vs-continue round, a rank-64 round whose shape
later rounds inherit, the two-pass round when the pools run short, and the wait after it.

  python tests/test_loop_logic.py        (or: python -m pytest tests/test_loop_logic.py)
"""

import collections
import copy
import glob
import hashlib
import json
import math
import os
import shutil
import sys
import tempfile
import threading
import time
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

# The graphics driver is read from Windows; every test but test_gpu_driver runs on the one all earlier scores used.
_real_gpu_driver = L.llama.gpu_driver
_legacy_driver = lambda: L.llama.LEGACY_GPU_DRIVER  # noqa: E731
L.llama.gpu_driver = _legacy_driver


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


def test_six_language_decide_and_loss():
    ids = [f"write/{LANGS[i % 6]}/cc/valid/{i}" for i in range(30)] + [f"fix/{LANGS[i % 2]}/cc/test/{i}" for i in range(20)]
    nc_ids = [f"mmlupro/{i}" for i in range(10)]
    nc = rows(nc_ids, set(nc_ids[:7])) + rows(nc_ids, set(nc_ids[:7]), seed=2)
    rules = {"alpha": 0.05, "part_tolerance": 0.05, "nc_tolerance": 0.05, "loss_rule": True, "loss_alpha": 0.05}
    base = rows(ids, set(ids[:20]) | set(ids[30:40]))           # write 20/30, fix 10/20
    # gate path: 10 more write items pass
    better = rows(ids, set(ids[:30]) | set(ids[30:40]))
    d = E.decide(base, better, nc, nc, rules)
    assert d["keep"] and d["path"] == "gate" and d["gate"]["wins"] == 10, d
    assert set(d["parts"]) == {"write", "fix"} and d["parts"]["fix"]["diff"] == 0 and len(d["langs"]) == 6
    assert d["langs"]["typescript"]["n"] == 5 and d["langs"]["cpp"]["n"] == 15  # 5 write + 10 fix
    assert "gate_cc" not in d
    # per-step losses: lower on all 20 steps is clear; 12 lower / 8 higher is not
    down = E.compare_loss({f"s{i}": 1.0 for i in range(20)}, {f"s{i}": 0.9 for i in range(20)},
                          {f"s{i}": 1 + (i == 0) * 19 for i in range(20)})
    assert down["wins"] == 20 and down["p_one_sided"] < 1e-5 and abs(down["diff"] + 0.1) < 1e-9, down
    mixed = E.compare_loss({f"s{i}": 1.0 for i in range(20)}, {f"s{i}": 0.8 if i < 12 else 1.1 for i in range(20)})
    assert (mixed["wins"], mixed["losses"]) == (12, 8) and mixed["p_one_sided"] > 0.05 and mixed["diff"] < 0
    weighted = E.compare_loss({"a": 1.0, "b": 1.0}, {"a": 0.0, "b": 2.0}, {"a": 1, "b": 3})
    assert weighted["candidate"] == 1.5 and weighted["best"] == 1.0
    # loss path: the gate level, the loss clearly lower
    d = E.decide(base, base, nc, nc, rules, loss=down)
    assert d["keep"] and d["path"] == "loss" and d["loss"]["wins"] == 20, d
    assert not E.decide(base, base, nc, nc, dict(rules, loss_rule=False), loss=down)["keep"]
    d = E.decide(base, base, nc, nc, rules, loss=mixed)
    assert not d["keep"] and any("not clearly lower" in r for r in d["reasons"]), d
    d = E.decide(base, base, nc, nc, rules, loss=None)
    assert not d["keep"] and any("no held-out loss" in r for r in d["reasons"]), d
    lower = rows(ids, (set(ids[:20]) | set(ids[30:40])) - {ids[0]})
    d = E.decide(base, lower, nc, nc, rules, loss=down)
    assert not d["keep"] and any("path closed" in r for r in d["reasons"]), d
    # level overall but the fix part drops 0.10 (> 0.05): rejected on either path
    trade = rows(ids, set(ids[:22]) | set(ids[30:38]))
    d = E.decide(base, trade, nc, nc, rules, loss=down)
    assert not d["keep"] and d["path"] is None and any("fix part dropped" in r for r in d["reasons"]), d
    # the non-coding guard holds on the loss path too
    worse_nc = rows(nc_ids, set(nc_ids[:5])) + rows(nc_ids, set(nc_ids[:5]), seed=2)
    assert not E.decide(base, base, nc, worse_nc, rules, loss=down)["keep"]
    # two held-out sets (21 Oct 2026: bug-fix and terminal steps), Holm-adjusted: one clear fall still keeps, but a
    # p that passes alone (0.03 <= 0.05) no longer does when the other set is level (2 x 0.03 > 0.05)
    def lo(wins, losses, diff=-0.05):
        return {"n": wins + losses, "best": 1.0, "candidate": 1.0 + diff, "diff": diff, "wins": wins,
                "losses": losses, "p_one_sided": E.sign_test_greater(wins, losses)}
    near = lo(19, 9)          # p 0.0436
    flat = lo(10, 10, 0.0)    # p 0.588
    assert 0.025 < near["p_one_sided"] <= 0.05
    assert E.decide(base, base, nc, nc, rules, losses={"swe": near})["keep"]
    d = E.decide(base, base, nc, nc, rules, losses={"swe": flat, "terminal": near})
    assert not d["keep"] and abs(d["losses"]["terminal"]["p_holm"] - 2 * near["p_one_sided"]) < 1e-12, d
    assert any("terminal loss not clearly lower" in r and "Holm-adjusted" in r for r in d["reasons"]), d["reasons"]
    d = E.decide(base, base, nc, nc, rules, losses={"swe": flat, "terminal": down})
    assert d["keep"] and d["path"] == "loss" and d["loss_fell"] == ["terminal"] and d["loss"] is flat, d
    lp = E.loss_paths(rules, {"swe": near, "terminal": down})  # Holm step-down: the second p is checked at 0.05
    assert lp["terminal"]["open"] and lp["swe"]["open"] and lp["swe"]["p_holm"] == near["p_one_sided"], lp
    assert E.open_loss_paths(rules, {"swe": near, "terminal": down}) == ["swe", "terminal"]
    assert E.open_loss_paths(rules, {"swe": lo(19, 9, +0.01)}) == []  # more steps lower but the mean rose: closed
    assert E.open_loss_paths(dict(rules, loss_rule=False), {"terminal": down}) == []
    assert E.loss_path_open(rules, down) and not E.loss_path_open(rules, None)
    # EvalPlus no-worse check against the base (21 Oct 2026): the non-coding check's test
    bp = {f"p{i}": int(i < 470) for i in range(542)}
    same = E.guard_check(bp, dict(bp), 0.03, 0.05)
    assert same["ok"] and same["diff"] == 0 and same["n"] == 542
    cp = dict(bp)
    for i in range(10):     # 10 lost, none won
        cp[f"p{i}"] = 0
    g = E.guard_check(bp, cp, 0.03, 0.05)
    assert not g["ok"] and g["losses"] == 10 and g["p_worse"] < 0.05, g
    cp2 = dict(bp)
    for i in range(16):     # first run's pattern: 12 won, 16 lost: level, kept
        cp2[f"p{i}"] = 0
    for i in range(470, 482):
        cp2[f"p{i}"] = 1
    g2 = E.guard_check(bp, cp2, 0.03, 0.05)
    assert g2["ok"] and (g2["wins"], g2["losses"]) == (12, 16) and g2["p_worse"] > 0.05, g2
    assert not E.guard_check({}, {}, 0.03, 0.05)["ok"]  # nothing to compare is never a pass
    # the scorer keeps the judge's verdict
    check = E.gate_check(None, None, None, lambda item, content: (False, "compile_error"))
    assert check({"kind": "fix", "id": "fix/cpp/x", "lang": "cpp"}, "x") == (False, "compile_error")
    try:
        E.gate_check(None, None, None)({"kind": "write", "id": "write/go/x", "lang": "go"}, "x")
        raise AssertionError("a write item was judged without the six-language judge")
    except RuntimeError:
        pass


def test_holdout_and_step_cap():
    """public_source.py --holdout-*: the pool and the held-out side split the repositories with nothing shared;
    --max-steps-per-conversation keeps K evenly spaced steps of a session."""
    import public_source as ps
    with tempfile.TemporaryDirectory() as d:
        def session(repo, n):
            msgs = [{"role": "user", "content": f"Fix the failing test in {repo}, issue {n}."}]
            for k in range(6):
                msgs += [{"role": "assistant", "content": f"step {k} of {repo} {n}", "reasoning_content": f"why {k}"},
                         {"role": "user", "content": f"output {k}"}]
            return {"repo": repo, "messages": msgs[:-1]}
        repos = [f"org/repo{i}" for i in range(40)]
        held = {r for r in repos if ps.holdout_bucket(r, 20) == 0}
        assert held and len(held) < len(repos)
        src = os.path.join(d, "swe.jsonl")
        E.write_jsonl(src, [session(r, n) for r in repos for n in range(2)])
        common = ["--dataset", "t/swe", "--files", "*", "--from-local", src, "--turns", "each", "--require-think",
                  "--tokenizer", "x", "--max-tokens", "4000", "--val-fraction", "0", "--holdout-col", "repo",
                  "--holdout-mod", "20"]
        kept = {}
        for side in ("pool", "heldout"):
            out = os.path.join(d, side)
            extra = ["--max-steps-per-conversation", "2"] if side == "heldout" else []
            assert run_public_source(common + ["--holdout-side", side, "--out", out] + extra) == 0
            kept[side] = E.read_jsonl(os.path.join(out, "train.jsonl")) + E.read_jsonl(os.path.join(out, "val.jsonl"))
            c = json.load(open(os.path.join(out, "measure.json")))["counts"]
            n_held = 2 * len(held)
            assert c["holdout_split"] == (n_held if side == "pool" else 80 - n_held), c
            if side == "heldout":
                assert c["step_cap"] == n_held * 4 and c["kept"] == n_held * 2 == len(kept[side]), c
        first = lambda r: r["messages"][0]["content"]  # noqa: E731
        pool_repos = {first(r).split(" in ")[1].split(",")[0] for r in kept["pool"]}
        held_repos = {first(r).split(" in ")[1].split(",")[0] for r in kept["heldout"]}
        assert held_repos == held and not (pool_repos & held_repos) and len(kept["pool"]) == (80 - 2 * len(held)) * 6
        # two of six steps per session, evenly spaced: steps 1 and 4
        last = sorted(r["messages"][-1]["content"].split("</think>")[-1].strip()[:6] for r in kept["heldout"]
                      if first(r) == f"Fix the failing test in {sorted(held)[0]}, issue 0.")
        assert last == ["step 1", "step 4"], last
        # build_swe_heldout.py reads the same source with every pool filter, on the held-out side
        import build_swe_heldout as B
        import yaml
        cfg = yaml.safe_load(open(os.path.join(ROOT, "experiments", "loop.yaml"), encoding="utf-8"))
        args, srcd = B.source_args(cfg)
        ns = parsed_args("public_source.py", args + ["--tokenizer", "x", "--out", d])
        assert (ns.holdout_col, ns.holdout_mod, ns.holdout_side, ns.max_steps_per_conversation) == ("repo", 20, "heldout", 1)
        assert ns.where == srcd["where"] and ns.require_think and ns.turns == "each" and ns.max_tokens == 8192
        pool_args = L.public_source_args(srcd)
        assert [a for a in args if a not in ("heldout", "pool")][:len(pool_args) - 1] == \
            [a for a in pool_args if a not in ("heldout", "pool")], (args, pool_args)
        # a source a converter writes locally (scripts/glm_rollouts.py) reads that file, and has no pool until the
        # converter's done file is there
        glm = next(s for s in cfg["sources"] if s.get("from_local"))
        ga = L.public_source_args(glm)
        assert ga[ga.index("--from-local") + 1] == os.path.join(ROOT, glm["from_local"]) and "--respect-loss-mask" in ga
        # 21 Oct 2026: every passing GLM-5.3 run, best-ranked first, near-duplicate steps dropped, task groups held out
        gns = parsed_args("public_source.py", ga + ["--tokenizer", "x", "--out", d])
        assert glm["name"] == "glm53_all_nopy_12k" and gns.where == ["uses_python=false"], gns
        assert (gns.rank_col, gns.first_ranks, gns.dedupe_actions_by) == ("task_rank", 2, "task_id"), gns
        assert (gns.holdout_col, gns.holdout_mod, gns.holdout_side) == ("task_group_id", 20, "pool"), gns
        py = next(s for s in cfg["sources"] if s["name"] == "glm53_all_12k")
        assert py["holdout"] == glm["holdout"] and py["share"] == 0 and py["where"] == ["uses_python=true"]
        targs, tsrc = B.source_args(cfg, "terminal")
        tns = parsed_args("public_source.py", targs + ["--tokenizer", "x", "--out", d])
        assert tsrc is glm and (tns.holdout_col, tns.holdout_mod, tns.holdout_side) == ("task_group_id", 20, "heldout")
        assert (tns.max_steps_per_conversation, tns.max_conversations_per, tns.max_tokens) == (1, "task_id=3", 12288)
        assert tns.rank_col == "task_rank" and tns.dedupe_actions_by == "task_id" and tns.respect_loss_mask, tns
        assert B.paths("terminal")[0].endswith("term_heldout.jsonl") and B.paths("swe")[0] == B.OUT
        shares = {s["name"]: s["share"] for s in cfg["sources"] if s["share"]}
        assert shares == {"glm53_all_nopy_12k": 0.40, "openswe_v4flash_12k": 0.35, "tachibana4_v4pro_nopy_12k": 0.20,
                          "banaxi_v4pro_nopy_12k": 0.05}, shares
        local, done = os.path.join(d, "conv.jsonl"), os.path.join(d, "conv.done.json")
        open(local, "w").close()
        assert not L.local_source_ready(dict(glm, from_local=local, from_local_done=done))
        open(done, "w").close()
        assert L.local_source_ready(dict(glm, from_local=local, from_local_done=done))
        assert L.local_source_ready(srcd)  # a dataset source is always ready


def test_gate_ready_and_retired_builder():
    with tempfile.TemporaryDirectory() as d:
        g = os.path.join(d, "gate.jsonl")
        assert not W.gate_ready(g)
        E.write_jsonl(g, [{"id": "cc/valid/0", "kind": "cc"}])
        assert not W.gate_ready(g)  # the retired Python gate: the watchdog doesn't start the loop on it
        E.write_jsonl(g, [{"id": "write/go/cc/valid/0", "kind": "write", "lang": "go"}])
        assert W.gate_ready(g)
        import build_evalsets as BE
        saved = E.EVAL_DIR
        E.EVAL_DIR = d
        os.makedirs(os.path.join(d, "archive", "python-gate"))
        try:
            sys.argv, argv = ["build_evalsets.py"], sys.argv
            BE.main()
            raise AssertionError("build_evalsets.py rebuilt the retired Python gate")
        except SystemExit as exc:
            assert "build_gate.py" in str(exc)
        finally:
            sys.argv, E.EVAL_DIR = argv, saved


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


def test_score_many_fills_idle_slots():
    """26 Oct 2026: the next pass starts in slots the last long answers of a pass leave idle; rows as score() alone."""
    import threading
    import time as _t
    with tempfile.TemporaryDirectory() as d:
        gate = [{"id": f"g/{i}", "prompt": f"g{i}"} for i in range(4)]
        nc = [{"id": f"m/{i}", "prompt": f"m{i}"} for i in range(3)]
        log, lock = [], threading.Lock()

        def chat(url, message, seed, max_tokens):
            with lock:
                log.append(("start", message, seed))
            _t.sleep(0.6 if message == "g3" else 0.02)  # the gate's last answer is the long one
            with lock:
                log.append(("end", message, seed))
            return {"content": "A" if message.startswith("m") else "ok", "reasoning": "", "finish_reason": "stop",
                    "completion_tokens": 3, "seconds": 0.01}
        chk = lambda item, content: content == "ok"  # noqa: E731
        gp, npth = os.path.join(d, "gate.jsonl"), os.path.join(d, "nc.jsonl")
        seen = []
        spec = lambda items, path, seed: dict(items=items, out_path=path, seed=seed, max_tokens=9, check=chk,  # noqa: E731
                                              chat=chat, count_tokens=lambda u, t: 0, budget=0,
                                              on_row=lambda r, n, tot, tps, s=seed: seen.append((r["id"], s, n, tot)))
        g, n1, n2 = E.score_many("u", [spec(gate, gp, 1), spec(nc, npth, 1), spec(nc, npth, 2)], concurrency=3,
                                 log=lambda m: None)
        assert [len(g), len(n1), len(n2)] == [4, 3, 3] and sum(r["pass"] for r in g) == 4
        assert {r["seed"] for r in n1} == {1} and {r["seed"] for r in n2} == {2} and len(E.read_jsonl(npth)) == 6
        # gate items are handed out first (g3 goes out with the first non-coding ones when slots free up; threads may
        # start them in any order), and non-coding answers ran during g3
        starts = [m for k, m, _ in log if k == "start"]
        assert set(starts[:3]) == {"g0", "g1", "g2"} and starts.index("g3") <= 5, starts
        g3_end = log.index(("end", "g3", 100003))
        assert any(k == "end" and m.startswith("m") for k, m, _ in log[:g3_end]), log
        # same request seeds as score() alone: 100000 * seed + index in its own items
        assert sorted(s for k, m, s in log if k == "start" and m.startswith("m")) == [100000, 100001, 100002,
                                                                                    200000, 200001, 200002]
        assert sorted((i, n, t) for i, s, n, t in seen if s == 2 and i.startswith("m"))[-1][2] == 3
        # one slot: strictly pass by pass, in each pass's order
        log.clear()
        gp1, np1 = os.path.join(d, "gate1.jsonl"), os.path.join(d, "nc1.jsonl")
        E.score_many("u", [spec(gate, gp1, 1), spec(nc, np1, 1), spec(nc, np1, 2)], concurrency=1, log=lambda m: None)
        assert [(m, s) for k, m, s in log if k == "start"] == (
            [(f"g{i}", 100000 + i) for i in range(4)] + [(f"m{i}", 100000 + i) for i in range(3)]
            + [(f"m{i}", 200000 + i) for i in range(3)])
        # resumable: nothing left to do
        log.clear()
        again = E.score_many("u", [spec(gate, gp, 1), spec(nc, npth, 1), spec(nc, npth, 2)], concurrency=3,
                             log=lambda m: None)
        assert not log and [len(x) for x in again] == [4, 3, 3]


def test_rows_bookkeeping():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "pool.jsonl")
        answer = {3: "a3", 4: "<think>\n \n</think>\n\na4"}  # no reasoning / an empty block: never handed out
        data = [{"messages": [{"role": "user", "content": f"q{i % 7}"},
                              {"role": "assistant", "content": answer.get(i, f"<think>t{i}</think> a{i}")}]}
                for i in range(20)]
        E.write_jsonl(path, data)
        used = {L.row_key(data[0])}  # question q0 used before
        skipped = collections.Counter()
        idx, nxt = L.pick_rows(path, 0, 5, used, skipped)  # 3 and 4 passed over; q3 again at 10 has reasoning
        assert idx == [1, 2, 5, 6, 10] and nxt == 11 and skipped["no_reasoning"] == 2, (idx, nxt, skipped)
        idx2, nxt2 = L.pick_rows(path, nxt, 5, used)  # q4 at 11 is new (row 4 was never used); the rest repeat
        assert idx2 == [11] and nxt2 == 20, (idx2, nxt2)
        assert [r["messages"][1]["content"] for r in L.rows_at(path, [6, 1, 3])] == ["<think>t6</think> a6",
                                                                                    "<think>t1</think> a1", "a3"]
        assert L.reasoning_text({"messages": [{"role": "user", "content": "q"},
                                              {"role": "assistant", "content": "x", "reasoning_content": "r"}]}) == "r"
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


def test_reworded_check_guard():
    """A round waits until the reworded-copy check (scripts/paraphrase_check.py covered.json) has seen the current
    gate and every usable pool as it is now."""
    saved = E.EVAL_DIR, E.GATE_PATH, L.POOLS
    with tempfile.TemporaryDirectory() as d:
        try:
            E.EVAL_DIR, E.GATE_PATH, L.POOLS = d, os.path.join(d, "gate.jsonl"), os.path.join(d, "pools")
            E.write_jsonl(E.GATE_PATH, [{"id": "write/cpp/cc/test/1", "kind": "write", "prompt": "p"}])
            for n, rows in (("openswe_v4flash", 3), ("glm53_terminal_nopy", 2)):
                E.write_jsonl(os.path.join(L.POOLS, n, "train.jsonl"), [{"messages": [], "i": i} for i in range(rows)])
            pools = ["openswe_v4flash", "glm53_terminal_nopy", "not_prepared_yet"]
            assert L.reworded_check_gaps(pools) == ["the gate", "glm53_terminal_nopy", "openswe_v4flash"]
            size = lambda n: os.path.getsize(os.path.join(L.POOLS, n, "train.jsonl"))  # noqa: E731
            cov = {"gate_sha256": hashlib.sha256(open(E.GATE_PATH, "rb").read()).hexdigest(),
                   "sources": {"mix_first_2k": 10, "loop/openswe_v4flash": size("openswe_v4flash"),
                               "loop/glm53_terminal_nopy": size("glm53_terminal_nopy")}}
            os.makedirs(os.path.join(d, "paraphrase"))
            json.dump(cov, open(os.path.join(d, "paraphrase", "covered.json"), "w"))
            assert L.reworded_check_gaps(pools) == [], L.reworded_check_gaps(pools)
            # a pool prepared again after the check, and a new gate
            E.write_jsonl(os.path.join(L.POOLS, "glm53_terminal_nopy", "train.jsonl"), [{"messages": [], "i": 9}] * 5)
            assert L.reworded_check_gaps(pools) == ["glm53_terminal_nopy"]
            E.write_jsonl(E.GATE_PATH, [{"id": "write/go/cc/test/2", "kind": "write", "prompt": "q"}])
            assert L.reworded_check_gaps(["openswe_v4flash"]) == ["the gate"]
            # the check ran on test sets passed in (no gate): never counts as covering the gate
            json.dump(dict(cov, gate_sha256=None), open(os.path.join(d, "paraphrase", "covered.json"), "w"))
            assert L.reworded_check_gaps([]) == ["the gate"]
        finally:
            E.EVAL_DIR, E.GATE_PATH, L.POOLS = saved


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
SWE_LOSS = {}  # round -> its loss on every held-out bug-fix step (default: 1 + 0.01 x round, never lower than the best)
TERM_LOSS = {}  # the same for the held-out terminal steps
EVALPLUS_WORSE = set()  # rounds whose EvalPlus answers fail every plus test (the no-worse check must block them)
BFCL_N = 12  # fake BFCL items: 8 Java, 4 JavaScript
BFCL_PASSING = {}  # model -> BFCL items 0..k-1 pass (default 9 for every model: no change)
BFCL_NO_CALLS = set()  # models whose answers never make a tool call (the probe must stop the pass)
CWEVAL_PASSING = {}  # model -> CWEval items 0..k-1 pass (default 9 for every model: no change)
CWEVAL_EMPTY = set()  # models whose answers are all reasoning, no text (the probe must stop the pass)
LANGS = ("cpp", "java", "go", "rust", "javascript", "typescript")


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
        self.pause_convert_once = set()  # run names whose GGUF convert is stopped by a pause once
        self.prompt_to_idx = {}
        self.swe_calls = []  # (round, --no-base) for each val_loss.py run on the held-out bug-fix steps
        self.term_calls = []  # the same for the held-out terminal steps
        self.fingerprint = "tc-1"  # the toolchains (anneal/polyglot.py toolchain_fingerprint)
        self.bfcl_requests = []  # (model, item id, request seed) for each BFCL request
        self.cweval_requests = []  # the same for CWEval
        self.served = []  # (stage, flags) for each server the loop started
        self.binaries = []  # (stage, llama-server binary or None for the default build) for the same servers

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
                ok = (i in k) if isinstance(k, (set, frozenset)) else i < k
                return {"content": f"```python\n# {'PASS' if ok else 'FAIL'}\n```", "reasoning": "r",
                        "finish_reason": "stop", "completion_tokens": 50, "seconds": 0.0}
            return {"content": f"The answer is ({'A' if i < 7 else 'B'})", "reasoning": "r", "finish_reason": "stop",
                    "completion_tokens": 20, "seconds": 0.0}
        return {"content": "def f(): pass", "reasoning": "thinking", "finish_reason": "stop", "completion_tokens": 30,
                "seconds": 0.0}  # sample prompts

    def bfcl_request(self, url, item, seed, max_tokens, sampling):
        m = self.model()
        self.bfcl_requests.append((m, item["id"], seed))
        if m in BFCL_NO_CALLS:
            return {"content": "I would call it.", "reasoning": "r", "tool_calls": [], "finish_reason": "stop",
                    "completion_tokens": 40, "seconds": 0.0, "answer": "I would call it."}
        i = int(item["id"].rsplit("_", 1)[1]) + (8 if item["language"] == "javascript" else 0)
        ok = i < BFCL_PASSING.get(m, 9)
        call = [{"type": "function", "function": {"name": item["tools"][0]["function"]["name"],
                                                  "arguments": json.dumps({"x": "PASS" if ok else "FAIL"})}}]
        return {"content": "", "reasoning": "r", "tool_calls": call, "finish_reason": "tool_calls",
                "completion_tokens": 60, "seconds": 0.0, "answer": json.dumps(call)}

    def cweval_request(self, url, item, seed, max_tokens, sampling):
        m = self.model()
        self.cweval_requests.append((m, item["id"], seed))
        if m in CWEVAL_EMPTY:
            return {"content": "", "reasoning": "r", "finish_reason": "stop", "completion_tokens": 900, "seconds": 0.0}
        ok = int(item["id"].split("_")[1]) < CWEVAL_PASSING.get(m, 9)
        return {"content": f"Done.\n\n```{item['lang']}\n// {'PASS' if ok else 'FAIL'}\n```\n", "reasoning": "r",
                "finish_reason": "stop", "completion_tokens": 70, "seconds": 0.0}

    def run_child(self, ctl, cmd, log_name, stop_dir=None, monitor=None, interval=30):
        script = os.path.basename(cmd[2])
        arg = lambda flag: cmd[cmd.index(flag) + 1]  # noqa: E731
        self.calls.append((script, cmd))
        if script in ("prepare_dataset.py", "public_source.py"):
            out = arg("--out")
            name = os.path.basename(out)
            n = {"tachibana4_v4pro": 720, "banaxi_v4pro": 20, "math_v4_aops": 120, "openswe_v4flash_12k": 60,
                 "tachibana4_v4pro_nopy_12k": 60, "banaxi_v4pro_nopy_12k": 60}[name]
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
            name = os.path.basename(arg("--outfile")).split("-F16")[0]
            if name in self.pause_convert_once:
                self.pause_convert_once.discard(name)
                ctl.stop.set()  # as if Pause was pressed while converting: the child is stopped and exits 1
                return 1
            open(arg("--outfile"), "wb").write(b"gguf")
            return 0
        if script == "val_loss.py":
            rd = arg("--run-dir")
            os.makedirs(rd, exist_ok=True)
            vf = arg("--val-file")
            if vf.endswith(("swe_heldout.jsonl", "term_heldout.jsonl")):  # held-out bug-fix / terminal steps
                rnd = [x for x in arg("--adapter").replace("\\", "/").split("/") if x.startswith("r") and x[1:].isdigit()][0]
                calls, table = (self.swe_calls, SWE_LOSS) if vf.endswith("swe_heldout.jsonl") else (self.term_calls, TERM_LOSS)
                calls.append((rnd, "--no-base" in cmd))
                keys = [r["messages"][0]["content"] for r in E.read_jsonl(vf)][:int(arg("--max-rows"))]
                own = table.get(rnd, 1.0 + 0.01 * int(rnd[1:]))  # by default each round is a little higher
                per = [dict({"row": i, "key": k, "trained": 100 + i, "adapter_loss": own},
                            **({} if "--no-base" in cmd else {"base_loss": 1.0})) for i, k in enumerate(keys)]
                names = ["adapter"] if "--no-base" in cmd else ["base", "adapter"]
                json.dump({"scored": names, "rows": len(per), "per_row": per,
                           "all": {n: {"trained_token_loss": own if n == "adapter" else 1.0} for n in names}},
                          open(os.path.join(rd, "summary.json"), "w"))
                return 0
            json.dump({"all": {"rows": 4, "base": {"answer_token_loss": 1.0, "trained_token_loss": 1.0},
                               "adapter": {"answer_token_loss": 0.9, "trained_token_loss": 0.9}}},
                      open(os.path.join(rd, "summary.json"), "w"))
            return 0
        raise AssertionError(f"unexpected child {cmd}")


_POLYGLOT_SAVED = {}
PRIVATE_CHECKS = []  # one entry per Loop.toolchain_check call that checked the sandbox's own storage folder


def fake_polyglot(fingerprint, broken=()):
    """Stand-ins for the sandbox parts that Loop.toolchain_check uses: the storage-folder check (anneal/sandbox.py),
    the fingerprint, and a judge that gives each smoke program its expected verdict (or compile_error for the (lang,
    program) pairs in broken). restore_polyglot puts the real ones back (tests/test_polyglot.py needs them)."""
    from anneal import polyglot as P
    from anneal import polyglot_smoke as S
    if not _POLYGLOT_SAVED:
        _POLYGLOT_SAVED.update(fingerprint=P.toolchain_fingerprint, executor=P.SandboxExecutor,
                               windows=P.Toolchain.__dict__["windows"], judge=P.judge, check=E.polyglot_check,
                               private=getattr(L.sandbox, "ensure_private_folder_closed", None))
    want = {(lang, p["code"]): (p["name"], p["expect"]) for lang in LANGS for p in S.programs(lang)}

    def judge(lang, code, cases, ex, tc, case_timeout_s=10.0, all_cases=False):
        name, expect = want[(lang, code)]
        stage = "compile_error" if (lang, name) in broken else expect
        return P.Verdict(stage == "ok", stage, [], "fake", 0.0, 0.0)
    P.toolchain_fingerprint = lambda root=None: fingerprint()
    P.SandboxExecutor = lambda *a, **k: None
    P.Toolchain.windows = classmethod(lambda cls, root=None: None)
    P.judge = judge
    L.sandbox.ensure_private_folder_closed = lambda log=print: PRIVATE_CHECKS.append(1) and False


def restore_polyglot():
    from anneal import polyglot as P
    if _POLYGLOT_SAVED:
        P.toolchain_fingerprint, P.SandboxExecutor, P.judge = (_POLYGLOT_SAVED[k] for k in ("fingerprint", "executor",
                                                                                            "judge"))
        P.Toolchain.windows = _POLYGLOT_SAVED["windows"]
        E.polyglot_check = _POLYGLOT_SAVED["check"]
        if _POLYGLOT_SAVED["private"] is None:
            del L.sandbox.ensure_private_folder_closed
        else:
            L.sandbox.ensure_private_folder_closed = _POLYGLOT_SAVED["private"]
        _POLYGLOT_SAVED.clear()


def setup_world(tmp):
    w = World(tmp)
    ev = os.path.join(tmp, "data", "eval")
    E.EVAL_DIR = ev
    E.GATE_PATH, E.NONCODING_PATH, E.DECONTAM_EXTRA = (os.path.join(ev, f) for f in
                                                        ("gate.jsonl", "noncoding_100.jsonl", "decontam_extra.jsonl"))
    # the six-language gate: 20 write items (one language each, in turn) and 20 fix items (C++ and Java)
    gate = [{"id": f"write/{LANGS[i % 6]}/cc/valid/{i}", "kind": "write", "lang": LANGS[i % 6],
             "prompt": f"gate prompt {i}", "cases": [["1\n", "1\n"]]} for i in range(20)]
    gate += [{"id": f"fix/{LANGS[i % 2]}/cc/test/{i}", "kind": "fix", "lang": LANGS[i % 2],
              "prompt": f"gate prompt {i + 20}", "cases": [["1\n", "1\n"]]} for i in range(20)]
    nc = [{"id": f"mmlupro/{i}", "category": "law", "question": f"nc question {i}", "options": ["x", "y"], "answer": "A"}
          for i in range(NC_N)]
    E.write_jsonl(E.GATE_PATH, gate)
    E.write_jsonl(E.NONCODING_PATH, nc)
    E.write_jsonl(E.DECONTAM_EXTRA, [{"id": g["id"], "prompt": g["prompt"]} for g in gate])
    E.write_jsonl(os.path.join(ev, "swe_heldout.jsonl"),  # held-out bug-fix steps (scripts/build_swe_heldout.py)
                  [{"messages": [{"role": "user", "content": f"held-out issue {i}"},
                                 {"role": "assistant", "content": "<think>t</think> step"}]} for i in range(12)])
    # the BFCL no-worse check (anneal/bfcl.py; scripts/build_guards.py --bfcl): 8 Java and 4 JavaScript items
    E.write_jsonl(os.path.join(ev, "bfcl_java_js.jsonl"),
                  [{"id": f"simple_{lg}_{i}", "category": f"simple_{lg}", "language": lg,
                    "messages": [{"role": "user", "content": f"call {lg} {i}"}],
                    "tools": [{"type": "function", "function": {"name": f"f{i}", "parameters": {"type": "object"}}}],
                    "function": [{"name": f"f{i}"}], "ground_truth": [{f"f{i}": {"x": ["PASS"]}}]}
                   for lg, k in (("java", 8), ("javascript", 4)) for i in range(k)])
    for i, g in enumerate(gate):
        w.prompt_to_idx[g["prompt"]] = ("gate", i)
    for i, q in enumerate(nc):
        w.prompt_to_idx[E.mc_message(q)] = ("nc", i)
    step = E.read_jsonl(os.path.join(ev, "swe_heldout.jsonl"))[0]["messages"][1]
    E.write_jsonl(os.path.join(ev, "term_heldout.jsonl"),  # held-out terminal steps (build_swe_heldout.py --set terminal)
                  [{"messages": [{"role": "user", "content": f"held-out terminal task {i}"}, step]} for i in range(14)])
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
    cfg["score_first_run"] = True  # the record round is exercised here; off in loop.yaml since 21 Oct
    cfg["guards"] = {"bfcl": cfg["guards"]["bfcl"]}  # the later checks are tested on their own (test_cweval_guard)
    cfg["gate"]["early_reject"] = None  # whole passes here; test_early_reject covers the early look
    cfg["prototype_rounds"] = []  # ordinary rounds here; test_prototype_round covers a prototype
    cfg["rules"]["confirm_net_win"] = True  # exercised here; off in loop.yaml until chosen
    cfg["reworded_check_required"] = False  # the pools are made during the run; test_reworded_check_guard covers it
    cfg["pause_after_rejects_in_row"] = 0  # rounds go on after a reject here (loop.yaml holds after round 2 for now)
    # the loop's mechanics on a fixed mix (85% / 5% / 10%); the live mix's other sources are still prepared, so their
    # command lines are checked (check_child_commands), but carry share 0 here
    sim_shares = {"tachibana4_v4pro": 0.85, "banaxi_v4pro": 0.05, "math_v4_aops": 0.10}
    for src in cfg["sources"]:  # small fake pools: tachibana (720) and math (120) usable; the rest keep their minimum
        src["share"] = sim_shares.get(src["name"], 0)
        if src["name"] in ("tachibana4_v4pro", "math_v4_aops"):
            src["min_pool_rows"] = 10
    # a source whose rows may lack reasoning (require_think off): never prepared or used (21 Oct rule), so the
    # recipe that asks for it alone is skipped
    cfg["sources"].append({"name": "noreason_agent", "kind": "messages", "dataset": "t/agent", "files": "x.jsonl",
                           "require_think": False, "pool_rows": 100, "share": 0})
    cfg["recipes"].insert(1, {"id": "noreason-agent", "init": "best", "mix": {"noreason_agent": 1.0}})
    cpath = os.path.join(tmp, "loop.yaml")
    yaml.safe_dump(cfg, open(cpath, "w", encoding="utf-8"), sort_keys=False)

    # the scripts the loop checks before using them (require_script): the versions the loop needs
    os.makedirs(os.path.join(tmp, "scripts"))
    open(os.path.join(tmp, "scripts", "prepare_dataset.py"), "w").write('ap.add_argument("--decontaminate-gate")\n'
                                                                       'p.add_argument("--drop-answer-lang")\n')
    open(os.path.join(tmp, "scripts", "train_longrun.py"), "w").write("def load_init_adapter(model, path, lora): ...\n")
    # patch the world into the modules
    L.ROOT = tmp
    L.POOLS = os.path.join(tmp, "data", "processed", "loop")
    L.RESULTS_DIR = os.path.join(tmp, "results")

    def serve(self, flags, name, coopmat=False, binary=None):
        self.server = FakeServer(flags)
        w.server = self.server
        w.served.append((name, list(flags)))
        w.binaries.append((name, binary))
        return self.server

    def unserve(self):
        self.server = None
        w.server = None

    fr.Controller.serve = serve
    fr.Controller.unserve = unserve
    fr.Controller.run_child = lambda self, cmd, log_name, stop_dir=None, monitor=None, interval=30: \
        w.run_child(self, cmd, log_name, stop_dir, monitor, interval)
    ew.chat_request = w.chat
    L.bfcl.prepare = lambda: None
    L.bfcl.request = w.bfcl_request
    L.bfcl.judge = lambda item, r: ((bool(r["tool_calls"]) and "PASS" in r["tool_calls"][0]["function"]["arguments"]),
                                    "ok" if r["tool_calls"] and "PASS" in r["tool_calls"][0]["function"]["arguments"]
                                    else ("no_tool_call" if not r["tool_calls"] else "value_error:string"))
    ew.count_tokens = lambda url, text: len(text) // 4
    gt.run_io_tests = lambda code, cases, timeout: "PASS" in code
    gt.run_assert_tests = lambda code, tests, timeout: "PASS" in code
    # the sandboxed six-language judge and its toolchain check, faked (restored by restore_polyglot)
    E.polyglot_check = lambda *a, **k: (lambda item, content: ("PASS" in content,
                                                                "ok" if "PASS" in content else "wrong_answer"))
    fake_polyglot(lambda: w.fingerprint)
    ew.load_problems = lambda ds: {f"{ds}/{i}": {} for i in range(10)}
    ew.generate_tasks = lambda *a, **k: {"rows": [], "wall_s": 1, "generated_tokens": 0}
    ew.sanitize_file = lambda ds, raw, out=None: raw
    ew.evaluate = lambda ds, path, parallel=8: {"base": 0.8, "plus": 0.7, "per_task": {
        f"{ds}/{i}": {"base_pass": True, "plus_pass": i % 2 == 0 and not any(
            f"{os.sep}{r}{os.sep}" in path for r in EVALPLUS_WORSE)} for i in range(10)}}
    L.Run.capture_output = lambda self, name="run.log": None
    return w, cpath


def world_patches():
    """Everything setup_world points at its world, with its real value, for restore_world afterwards (so tests that
    run later see the real modules)."""
    patched = [(L, "ROOT"), (L, "POOLS"), (L, "RESULTS_DIR"), (E, "EVAL_DIR"), (E, "GATE_PATH"),
               (E, "NONCODING_PATH"), (E, "DECONTAM_EXTRA"), (fr.Controller, "serve"), (fr.Controller, "unserve"),
               (fr.Controller, "run_child"), (ew, "chat_request"), (ew, "count_tokens"), (ew, "load_problems"),
               (ew, "generate_tasks"), (ew, "sanitize_file"), (ew, "evaluate"), (gt, "run_io_tests"),
               (gt, "run_assert_tests"), (L.Run, "capture_output"), (L.bfcl, "prepare"), (L.bfcl, "request"),
               (L.bfcl, "judge")]
    return [(o, a, o.__dict__[a] if a in o.__dict__ else getattr(o, a)) for o, a in patched]


def restore_world(before):
    for o, a, v in before:
        setattr(o, a, v)


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
        restore_polyglot()
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
    # round 0 scored the base on the BFCL no-worse check after its gate pass; r002 (kept) reused those rows
    bb = state["base_guards"]["bfcl"]
    assert bb["scored_in_round"] == 0 and (bb["pass"], bb["n"]) == (9, 12), bb
    assert bb["by_language"] == {"java": {"pass": 8, "n": 8}, "javascript": {"pass": 1, "n": 4}}, bb
    assert state["rounds"][0]["guards"]["bfcl"]["pass"] == 9 and done(2, "bfcl")["guard"]["ok"]
    assert [m for m, _, _ in w.bfcl_requests].count(None) == 12
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
    # the no-reasoning source was never prepared, so its recipe was skipped; round 3 used the default recipe (the twin
    # and rank-64 recipes wait for their conditions, two-pass for short pools)
    assert state["recipes_skipped"] == ["noreason-agent"] and state["recipes_done"] == ["v4pro-1", "noreason-agent"]
    assert not any("noreason_agent" in " ".join(cmd) for script, cmd in w.calls if script == "public_source.py")
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
    assert done(7, "milestone")["guard"]["ok"]  # EvalPlus (the no-worse check) runs before the confirm pass
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
    # the held-out bug-fix loss: the base scored once (round 2, while the base was the best), every later round only
    # its adapter, each compared with the best of its time; no round here gets lower, so none is kept on it
    assert w.swe_calls[0] == ("r002", False) and all(nb for _, nb in w.swe_calls[1:]), w.swe_calls
    assert sorted({r for r, _ in w.swe_calls}) == [f"r{n:03d}" for n in range(2, 12)], w.swe_calls
    assert state["base"]["swe_loss"]["field"] == "base_loss" and state["base"]["swe_loss"]["round"] == 0
    assert state["best"]["swe_loss"]["round"] == 9 and state["best"]["swe_loss"]["field"] == "adapter_loss"
    assert done(3, "sweloss")["vs_best"]["reference_round"] == 2 and done(2, "sweloss")["vs_best"]["reference_round"] == 0
    assert done(10, "sweloss")["vs_best"]["reference_round"] == 9 and done(10, "sweloss")["vs_best"]["losses"] == 12
    # the held-out terminal loss works the same way
    assert w.term_calls[0] == ("r002", False) and all(nb for _, nb in w.term_calls[1:]), w.term_calls
    assert sorted({r for r, _ in w.term_calls}) == [f"r{n:03d}" for n in range(2, 12)], w.term_calls
    assert state["base"]["term_loss"]["round"] == 0 and state["best"]["term_loss"]["round"] == 9
    assert done(10, "termloss")["vs_best"]["reference_round"] == 9 and done(10, "termloss")["vs_best"]["losses"] == 14
    assert all(r.get("path") in (None, "gate") for r in state["rounds"]), state["rounds"]
    rep4 = open(os.path.join(rd(4), "report.md")).read()
    assert "Kept on the gate" in rep4 and "part: fix a wrong submission" in rep4 and "language: TypeScript" in rep4
    assert "held-out bug-fix steps: loss" in rep4 and "Gate verdicts: ok 35, wrong_answer 5." in rep4, rep4
    assert d4["parts"]["write"]["n"] == 20 and d4["parts"]["fix"]["n"] == 20 and len(d4["langs"]) == 6
    assert "bug-fix loss vs best" in summary
    assert state["toolchains"] == "tc-1" and state["gate_sha"] == L.file_sha(E.GATE_PATH)
    check_child_commands(w.calls)


def test_loss_path_and_guards():
    tmp = tempfile.mkdtemp(prefix="anneal-loop-test-")
    saved = dict(PASSING), dict(SWE_LOSS)
    try:
        _loss_path_and_guards(tmp)
    finally:
        PASSING.clear()
        PASSING.update(saved[0])
        SWE_LOSS.clear()
        SWE_LOSS.update(saved[1])
        restore_polyglot()
        shutil.rmtree(tmp, ignore_errors=True)


def _loss_path_and_guards(tmp):
    w, cpath = setup_world(tmp)
    w.pause_train_once = set()
    run_dir = os.path.join(tmp, "results", "loop")
    args = ["--config", cpath, "--run-dir", run_dir, "--max-rounds", "1"]
    rd = lambda n: os.path.join(run_dir, "rounds", f"r{n:03d}")  # noqa: E731
    done = lambda n, stage: json.load(open(os.path.join(rd(n), "stages", f"{stage}.done.json")))  # noqa: E731
    status = lambda: json.load(open(os.path.join(run_dir, "status.json")))  # noqa: E731
    six = E.read_jsonl(E.GATE_PATH)
    # the retired Python gate never scores a round: the loop waits until build_gate.py has replaced it
    E.write_jsonl(E.GATE_PATH, [{"id": "cc/valid/0", "kind": "cc", "prompt": "p", "cases": []}])
    assert run_main(args) == 0
    assert status()["state"] == "waiting" and "not the six-language gate" in status()["detail"], status()
    assert not os.path.exists(rd(0))
    E.write_jsonl(E.GATE_PATH, six)
    # r002 scores the same as the base on the gate (20/40) but its loss on every held-out bug-fix step falls: kept on
    # the loss path (rules.loss_rule)
    PASSING["r002"], SWE_LOSS["r002"] = 20, 0.8
    assert run_main(args) == 0, status()
    state = json.load(open(os.path.join(run_dir, "state.json")))
    r2 = [r for r in state["rounds"] if r["round"] == 2][0]
    d2 = done(2, "decide")
    assert r2["status"] == "kept" and r2["path"] == "loss" and d2["path"] == "loss" and d2["gate"]["diff"] == 0, d2
    assert d2["loss"]["wins"] == 12 and d2["loss"]["losses"] == 0 and r2["swe_loss"]["candidate"] == 0.8
    assert state["best"]["round"] == 2 and state["best"]["swe_loss"]["round"] == 2
    assert "Kept on the held-out loss" in open(os.path.join(rd(2), "report.md")).read()
    assert d2["loss_fell"] == ["swe"] and r2["loss_fell"] == ["swe"] and done(2, "milestone")["guard"]["ok"]
    assert "kept (loss path)" in open(os.path.join(run_dir, "summary.md")).read()
    # with the loss rule off, the same round would have been rejected
    cfg = L.load_config(cpath)
    off = E.decide(E.read_jsonl(state["base"]["gate_rows"]), E.read_jsonl(os.path.join(rd(2), "gate.seed1.jsonl")),
                   E.read_jsonl(state["base"]["noncoding_rows"]), E.read_jsonl(os.path.join(rd(2), "noncoding.jsonl")),
                   dict(cfg["rules"], loss_rule=False), loss=d2["loss"])
    assert not off["keep"] and off["path"] is None
    # the toolchains changed after rounds were judged: r003 trains, then waits before its gate pass
    w.fingerprint = "tc-2"
    assert run_main(args) == 0
    assert status()["state"] == "waiting" and "toolchains changed" in status()["detail"], status()
    assert os.path.exists(os.path.join(rd(3), "stages", "sweloss.done.json"))
    assert not os.path.exists(os.path.join(rd(3), "stages", "evaluate.done.json"))
    # back as they were: r003 resumes at its gate pass; compared with r002's own losses (adapter only)
    w.fingerprint = "tc-1"
    assert run_main(args) == 0, status()
    assert done(3, "sweloss")["vs_best"]["reference_round"] == 2 and w.swe_calls == [("r002", False), ("r003", True)]
    assert [r["status"] for r in json.load(open(os.path.join(run_dir, "state.json")))["rounds"]][-1] == "rejected"
    # the gate changed after rounds were scored on it: the loop waits
    E.write_jsonl(E.GATE_PATH, six[:-1])
    assert run_main(args) == 0
    assert status()["state"] == "waiting" and "gate.jsonl changed" in status()["detail"], status()
    E.write_jsonl(E.GATE_PATH, six)
    # a broken toolchain fails the gate pass instead of scoring every answer as a compile error
    fake_polyglot(lambda: w.fingerprint, broken={("rust", "sum64")})
    st = json.load(open(os.path.join(run_dir, "state.json")))
    fake = types.SimpleNamespace(state=st, save=lambda: None, log=lambda *a: None)
    checks = len(PRIVATE_CHECKS)
    try:
        L.Loop.toolchain_check(fake, six)
        raise AssertionError("a broken toolchain passed the smoke check")
    except RuntimeError as exc:
        assert "rust sum64: compile_error" in str(exc), exc
    fake_polyglot(lambda: w.fingerprint)
    assert L.Loop.toolchain_check(fake, six) == "tc-1"
    assert L.Loop.toolchain_check(fake, [{"kind": "cc", "id": "cc/valid/0"}]) is None  # the old gate: nothing to build
    assert len(PRIVATE_CHECKS) == checks + 2  # the sandbox's storage folder is checked before each six-language pass


def test_terminal_path_and_evalplus_guard():
    """21 Oct 2026: the held-out terminal loss is a second loss path (Holm-adjusted with the bug-fix one), and
    EvalPlus is a no-worse check that runs on rounds that pass the keep rule and the other no-worse checks, before the
    confirm pass."""
    tmp = tempfile.mkdtemp(prefix="anneal-loop-test-")
    saved = dict(PASSING), dict(SWE_LOSS), dict(TERM_LOSS), set(EVALPLUS_WORSE)
    try:
        _terminal_path_and_evalplus_guard(tmp)
    finally:
        PASSING.clear()
        PASSING.update(saved[0])
        SWE_LOSS.clear()
        SWE_LOSS.update(saved[1])
        TERM_LOSS.clear()
        TERM_LOSS.update(saved[2])
        EVALPLUS_WORSE.clear()
        EVALPLUS_WORSE.update(saved[3])
        restore_polyglot()
        shutil.rmtree(tmp, ignore_errors=True)


def _terminal_path_and_evalplus_guard(tmp):
    import yaml
    w, cpath = setup_world(tmp)
    w.pause_train_once = set()
    cfg = L.load_config(cpath)
    cfg["score_first_run"] = False
    yaml.safe_dump(cfg, open(cpath, "w", encoding="utf-8"), sort_keys=False)
    run_dir = os.path.join(tmp, "results", "loop")
    args = ["--config", cpath, "--run-dir", run_dir, "--max-rounds", "1"]
    rd = lambda n: os.path.join(run_dir, "rounds", f"r{n:03d}")  # noqa: E731
    done = lambda n, stage: json.load(open(os.path.join(rd(n), "stages", f"{stage}.done.json")))  # noqa: E731
    state = lambda: json.load(open(os.path.join(run_dir, "state.json")))  # noqa: E731
    # the terminal steps aren't built yet: a training round trains, then waits before its first held-out loss
    term = os.path.join(E.EVAL_DIR, "term_heldout.jsonl")
    os.rename(term, term + ".x")
    assert run_main(args) == 0
    st = json.load(open(os.path.join(run_dir, "status.json")))
    assert st["state"] == "waiting" and "build_swe_heldout.py --set terminal" in st["detail"], st
    os.rename(term + ".x", term)
    # r002: level on the gate (20 of 40), the bug-fix loss a little higher, the terminal loss lower on all 14 steps:
    # kept on the loss path, by the terminal set; the base's terminal losses are scored once, here
    PASSING["r002"], TERM_LOSS["r002"] = 20, 0.8
    assert run_main(args) == 0, json.load(open(os.path.join(run_dir, "status.json")))
    d2, r2 = done(2, "decide"), [r for r in state()["rounds"] if r["round"] == 2][0]
    assert d2["keep"] and d2["path"] == "loss" and d2["loss_fell"] == ["terminal"], d2
    assert d2["losses"]["swe"]["p_holm"] == 1.0 and d2["losses"]["terminal"]["p_holm"] < 0.001, d2["losses"]
    assert r2["status"] == "kept" and abs(r2["term_loss"]["candidate"] - 0.8) < 1e-9 and r2["loss_fell"] == ["terminal"], r2
    assert w.term_calls == [("r002", False)] and state()["base"]["term_loss"]["field"] == "base_loss"
    assert state()["best"]["term_loss"]["round"] == 2 and state()["best"]["swe_loss"]["round"] == 2
    rep2 = open(os.path.join(rd(2), "report.md")).read()
    assert "held-out terminal steps fell" in rep2 and "held-out terminal steps: loss" in rep2, rep2
    assert "EvalPlus no-worse check" in rep2 and done(2, "milestone")["guard"]["ok"]
    summary = open(os.path.join(run_dir, "summary.md")).read()
    assert "terminal loss vs best" in summary and "0.8000" in summary
    # r003 passes the keep rule on the gate (30 of 40) but its EvalPlus answers fail every plus test: blocked before
    # the confirm pass, rejected, its adapter deleted
    PASSING["r003"] = 30
    EVALPLUS_WORSE.add("r003")
    assert run_main(args) == 0
    d3, r3 = done(3, "decide"), [r for r in state()["rounds"] if r["round"] == 3][0]
    g3 = done(3, "milestone")["guard"]
    assert d3["keep"] and not g3["ok"] and g3["candidate"] == 0 and g3["losses"] > 0, g3
    assert r3["status"] == "rejected" and any(x.startswith("EvalPlus") for x in r3["reasons"]), r3
    assert not os.path.exists(os.path.join(rd(3), "stages", "confirm.done.json"))
    assert not os.path.exists(os.path.join(rd(3), "train", "adapter")) and state()["best"]["round"] == 2
    assert "worse than the base" in open(os.path.join(rd(3), "report.md")).read()
    assert w.term_calls == [("r002", False), ("r003", True)]
    # BFCL ran first (the cheaper checks come before EvalPlus) and passed; with both checks off it would have been kept
    assert done(3, "bfcl")["guard"]["ok"]
    assert L.guard_ok(dict(cfg, rules=dict(cfg["rules"], evalplus_guard=False), guards={}), done(3, "milestone"))
    assert not L.guard_ok(dict(cfg, rules=dict(cfg["rules"], evalplus_guard=False)), done(3, "milestone"))
    assert not L.guard_ok(cfg, None) and not L.guard_ok(cfg, {"guard": {"ok": False}})
    assert L.guards_failed(cfg, done(3, "milestone"), {"bfcl": {"guard": {"ok": False}}}) == ["evalplus", "bfcl"]
    assert L.guards_failed(cfg, None, {}) == ["evalplus"]  # passed every check but EvalPlus never ran


def test_bfcl_guard():
    """21 Oct 2026: BFCL's Java and JavaScript tool calls are a no-worse check, run before EvalPlus. A training round
    waits until the set is built; the base is scored once (here lazily, in the first round that needs it, since the set
    was built after round 0) and again only when the settings change; a round worse on it is rejected before EvalPlus
    and the confirm pass; a pass whose first answers make no tool call at all stops (the round fails, its adapter
    kept)."""
    tmp = tempfile.mkdtemp(prefix="anneal-loop-test-")
    saved = dict(PASSING), dict(BFCL_PASSING), set(BFCL_NO_CALLS)
    before = world_patches()
    try:
        _bfcl_guard(tmp)
    finally:
        restore_world(before)
        PASSING.clear()
        PASSING.update(saved[0])
        BFCL_PASSING.clear()
        BFCL_PASSING.update(saved[1])
        BFCL_NO_CALLS.clear()
        BFCL_NO_CALLS.update(saved[2])
        restore_polyglot()
        shutil.rmtree(tmp, ignore_errors=True)


def _bfcl_guard(tmp):
    import yaml
    w, cpath = setup_world(tmp)
    w.pause_train_once = set()
    cfg = L.load_config(cpath)
    cfg["score_first_run"] = False
    yaml.safe_dump(cfg, open(cpath, "w", encoding="utf-8"), sort_keys=False)
    run_dir = os.path.join(tmp, "results", "loop")
    args = ["--config", cpath, "--run-dir", run_dir, "--max-rounds", "1"]
    rd = lambda n: os.path.join(run_dir, "rounds", f"r{n:03d}")  # noqa: E731
    done = lambda n, stage: json.load(open(os.path.join(rd(n), "stages", f"{stage}.done.json")))  # noqa: E731
    state = lambda: json.load(open(os.path.join(run_dir, "state.json")))  # noqa: E731
    rec = lambda n: [r for r in state()["rounds"] if r["round"] == n][0]  # noqa: E731
    bf = os.path.join(E.EVAL_DIR, "bfcl_java_js.jsonl")
    # not built yet: round 0 finishes without it (never stopped by a check), then the loop waits for it
    os.rename(bf, bf + ".x")
    assert run_main(args) == 0
    st = json.load(open(os.path.join(run_dir, "status.json")))
    assert st["state"] == "waiting" and "build_guards.py --bfcl" in st["detail"], st
    assert rec(0)["status"] == "reference" and not state().get("base_guards") and w.bfcl_requests == []
    os.rename(bf + ".x", bf)
    # r002 keeps on the gate and passes EvalPlus; the base is scored on BFCL here, then r002: level, so kept
    PASSING["r002"] = 30
    assert run_main(args) == 0, json.load(open(os.path.join(run_dir, "status.json")))
    g2 = done(2, "bfcl")
    assert rec(2)["status"] == "kept" and g2["guard"]["ok"] and g2["guard"]["n"] == 12, g2
    assert g2["base"]["scored_in_round"] == 2 and g2["candidate"]["pass"] == 9
    assert g2["guard"]["by_language"]["javascript"]["n"] == 4 and rec(2)["guards"]["bfcl"]["ok"]
    assert [m for m, _, _ in w.bfcl_requests] == [None] * 12 + ["r002"] * 12
    assert {s // 100000 for _, _, s in w.bfcl_requests} == {1}  # seed 1 for base and round alike
    rep2 = open(os.path.join(rd(2), "report.md")).read()
    assert "BFCL Java and JavaScript tool calls: no-worse check" in rep2 and "| JavaScript | 4 |" in rep2, rep2
    assert "bfcl 0.750 -> 0.750" in open(os.path.join(run_dir, "summary.md")).read()
    order = [n for n, _ in w.served]  # the cheaper check first, then EvalPlus, then the confirm pass
    assert order.index("r002-bfcl") < order.index("r002-evalplus") < order.index("r002-confirm"), order
    # r003 keeps on the gate but loses 7 BFCL items: rejected before EvalPlus and the confirm pass
    PASSING["r003"], BFCL_PASSING["r003"] = 35, 2
    assert run_main(args) == 0
    g3 = done(3, "bfcl")["guard"]
    assert done(3, "decide")["keep"] and not g3["ok"], g3
    assert not os.path.exists(os.path.join(rd(3), "stages", "milestone.done.json")) and "r003-evalplus" not in \
        [n for n, _ in w.served]
    assert L.guards_failed(cfg, None, {"bfcl": done(3, "bfcl")}) == ["bfcl"]  # EvalPlus didn't run: not a failure
    assert g3["losses"] == 7 and g3["wins"] == 0 and g3["p_worse"] < 0.01, g3
    assert rec(3)["status"] == "rejected" and any(x.startswith("BFCL") for x in rec(3)["reasons"]), rec(3)
    assert not os.path.exists(os.path.join(rd(3), "stages", "confirm.done.json")) and state()["best"]["round"] == 2
    assert [m for m, _, _ in w.bfcl_requests].count(None) == 12  # the base was not scored again
    assert "WORSE" in open(os.path.join(run_dir, "summary.md")).read()
    # a new seed for the check: the base is scored again under it before r004 is compared
    cfg = L.load_config(cpath)
    cfg["guards"]["bfcl"]["seed"] = 2
    yaml.safe_dump(cfg, open(cpath, "w", encoding="utf-8"), sort_keys=False)
    PASSING["r004"] = 35
    assert run_main(args) == 0
    assert rec(4)["status"] == "kept" and done(4, "bfcl")["guard"]["ok"]
    seeds = [(m, s // 100000) for m, _, s in w.bfcl_requests]
    assert seeds.count((None, 2)) == 12 and seeds.count(("r004", 2)) == 12
    assert state()["base_guards"]["bfcl"]["fingerprint"]["seed"] == 2
    # r005's answers make no tool call: the first 8 stop the pass (twice: once more with a fresh server), the round
    # fails and keeps its adapter
    PASSING["r005"] = 40
    BFCL_NO_CALLS.add("r005")
    assert run_main(args) == 1
    st = json.load(open(os.path.join(run_dir, "status.json")))
    assert st["state"] == "failed" and "none of the first 8 answers made a tool call" in st["error"], st
    assert rec(5)["failures"] == 1 and os.path.exists(os.path.join(rd(5), "train", "adapter"))
    # 8 answers, plus at most the others already in flight (one per gate slot); the retry stops on the 8 saved ones
    # without asking again
    assert 8 <= [m for m, _, _ in w.bfcl_requests].count("r005") <= 8 + L.gate_concurrency(cfg) - 1
    assert len(E.read_jsonl(os.path.join(rd(5), "guards", "bfcl.seed2.jsonl"))) == 8


def test_prototype_round():
    """27 Oct 2026: a prototype round (loop.yaml prototype_rounds) answers the whole gate even where the early look
    would stop it, runs the no-worse checks and EvalPlus although it fails the keep rule (and after one of them is
    worse), has no confirm pass, and keeps its adapter and GGUF when rejected; the best stays the base. The next,
    ordinary round stops early as before, runs no checks, and its reject deletes them."""
    tmp = tempfile.mkdtemp(prefix="anneal-loop-test-")
    saved = dict(PASSING), dict(BFCL_PASSING)
    before = world_patches()
    try:
        _prototype_round(tmp)
    finally:
        restore_world(before)
        PASSING.clear()
        PASSING.update(saved[0])
        BFCL_PASSING.clear()
        BFCL_PASSING.update(saved[1])
        restore_polyglot()
        shutil.rmtree(tmp, ignore_errors=True)


def _prototype_round(tmp):
    import yaml
    w, cpath = setup_world(tmp)
    w.pause_train_once = set()
    cfg = L.load_config(cpath)
    cfg["score_first_run"] = False
    cfg["gate"]["early_reject"] = {"looks": [0.3334, 0.6667], "max_diff": 0}
    cfg["prototype_rounds"] = [2]
    yaml.safe_dump(cfg, open(cpath, "w", encoding="utf-8"), sort_keys=False)
    assert L.prototype_round(cfg, 2) and not L.prototype_round(cfg, 3) and not L.prototype_round({}, 2)
    run_dir = os.path.join(tmp, "results", "loop")
    args = ["--config", cpath, "--run-dir", run_dir, "--max-rounds", "1"]
    rd = lambda n: os.path.join(run_dir, "rounds", f"r{n:03d}")  # noqa: E731
    done = lambda n, stage: json.load(open(os.path.join(rd(n), "stages", f"{stage}.done.json")))  # noqa: E731
    has = lambda n, stage: os.path.exists(os.path.join(rd(n), "stages", f"{stage}.done.json"))  # noqa: E731
    state = lambda: json.load(open(os.path.join(run_dir, "state.json")))  # noqa: E731
    rec = lambda n: [r for r in state()["rounds"] if r["round"] == n][0]  # noqa: E731
    gate = E.read_jsonl(E.GATE_PATH)
    # r002 is behind the base (18 vs 20 of 40) and loses 7 BFCL items: rejected, but every check runs
    PASSING["r002"], BFCL_PASSING["r002"] = 18, 2
    assert run_main(args) == 0, json.load(open(os.path.join(run_dir, "status.json")))
    ev2, d2 = done(2, "evaluate"), done(2, "decide")
    assert "early_check" not in ev2 and ev2["gate_n"] == len(gate) and not d2["keep"], ev2
    assert E.read_jsonl(os.path.join(rd(2), "noncoding.jsonl"))
    assert not done(2, "bfcl")["guard"]["ok"] and has(2, "milestone") and not has(2, "confirm")
    order = [n for n, _ in w.served]
    assert order.index("r002-evaluate") < order.index("r002-bfcl") < order.index("r002-evalplus"), order
    assert "r002-confirm" not in order
    r2 = rec(2)
    assert r2["status"] == "rejected" and "prototype" in r2["files_kept"] and "freed_bytes" not in r2, r2
    assert r2["guards"]["bfcl"]["ok"] is False and r2["evalplus"] and r2["evalplus_guard"], r2
    assert any(x.startswith("BFCL") for x in r2["reasons"]), r2["reasons"]
    assert os.path.exists(os.path.join(rd(2), "train", "adapter")) and os.path.exists(os.path.join(rd(2), "convert"))
    assert "Files: a prototype round" in open(os.path.join(rd(2), "report.md")).read()
    assert state()["best"]["round"] == 0 and state()["rejects_in_row"] == 1
    # r003, an ordinary round, is behind too: it stops at the first look, runs no checks, and its files are deleted
    PASSING["r003"] = 18
    assert run_main(args) == 0
    ev3 = done(3, "evaluate")
    assert ev3["early_reject"] is True and ev3["gate_n"] < len(gate), ev3
    assert not has(3, "bfcl") and not has(3, "milestone") and rec(3)["status"] == "rejected"
    assert rec(3)["freed_bytes"] > 0 and "files_kept" not in rec(3)
    assert not os.path.exists(os.path.join(rd(3), "train", "adapter")) and not os.path.exists(os.path.join(rd(3), "convert"))


def test_think_selection():
    """28 Oct 2026: rows picked by reasoning length. Unit parts: the selecting keys (min_think_tokens, order) are not
    part of a pool's spec; think_index counts and caches each row's reasoning tokens; ranked_rows filters and orders;
    filter_rolled drops short rolled-forward rows. Then a prototype round 2 is rejected and the loop holds
    (pause_after_rejects_in_row 1) instead of training round 3; with the hold off and tachibana4 selecting (at least
    20 reasoning tokens, most first), round 3 takes its new rows in that order, the rolled-forward tachibana4 rows
    under 20 tokens stay out, and the pool's offset doesn't move."""
    src = {"name": "a", "kind": "messages", "share": 0.4}
    assert L.spec_hash(src) == L.spec_hash(dict(src, min_think_tokens=150, order="think_desc"))
    assert not L.selects(src) and not L.selects(dict(src, order="pool")) and not L.selects(None)
    assert L.selects(dict(src, min_think_tokens=1)) and L.selects(dict(src, order="think_desc"))
    keys, tokens = ["k0", "k1", None, "k3", "k4", "k5"], [5, 0, 9, 9, 30, 9]
    assert L.ranked_rows(src, keys, tokens) == [0, 3, 4, 5]                        # no reasoning / no key: out
    assert L.ranked_rows(dict(src, min_think_tokens=9), keys, tokens) == [3, 4, 5]
    assert L.ranked_rows(dict(src, order="think_desc"), keys, tokens) == [4, 3, 5, 0]  # ties keep pool order
    try:
        L.ranked_rows(dict(src, order="longest"), keys, tokens)
        raise AssertionError("unknown order accepted")
    except RuntimeError:
        pass
    srcs = {"a": dict(src, min_think_tokens=9), "b": {"name": "b"}}
    out, dropped = L.filter_rolled([["a", [0, 3, 4]], ["b", [0, 1]], ["a", [1]]], srcs, lambda n: (keys, tokens))
    assert out == [["a", [3, 4]], ["b", [0, 1]]] and dropped == {"a": 2}, (out, dropped)
    tmp = tempfile.mkdtemp(prefix="anneal-loop-test-")
    saved_count = L.count_tokens
    saved = dict(PASSING)
    before = world_patches()
    try:
        calls = []
        L.count_tokens = lambda texts: calls.append(len(texts)) or [len(t.split()) for t in texts]
        pool = os.path.join(tmp, "pool")
        rows = [{"messages": [{"role": "user", "content": f"q{i}"},
                              {"role": "assistant", "content": f"<think>{' w' * i}</think> a"}]} for i in range(4)]
        rows.append({"messages": [{"role": "user", "content": "q4"}, {"role": "assistant", "content": "no trace"}]})
        E.write_jsonl(os.path.join(pool, "train.jsonl"), rows)
        k, t = L.think_index(pool)
        assert t == [0, 1, 2, 3, 0] and k == [L.row_key(r) for r in rows] and calls == [5], (t, calls)
        assert L.think_index(pool) == (k, t) and calls == [5]                     # cached
        E.write_jsonl(os.path.join(pool, "train.jsonl"), rows[:3])
        assert L.think_index(pool)[1] == [0, 1, 2] and calls == [5, 3]            # rebuilt when the pool changes
        _think_selection(tmp)
    finally:
        L.count_tokens = saved_count
        restore_world(before)
        PASSING.clear()
        PASSING.update(saved)
        restore_polyglot()
        shutil.rmtree(tmp, ignore_errors=True)


def _think_selection(tmp):
    import yaml
    w, cpath = setup_world(tmp)
    w.pause_train_once = set()
    # each pool row's reasoning tokens: 37 * index mod 101 (the fake pools all carry the same one-word reasoning)
    L.count_tokens = lambda texts: [(37 * i) % 101 for i in range(len(texts))]
    cfg = L.load_config(cpath)
    cfg.update(score_first_run=False, prototype_rounds=[2], pause_after_rejects_in_row=1)
    yaml.safe_dump(cfg, open(cpath, "w", encoding="utf-8"), sort_keys=False)
    run_dir = os.path.join(tmp, "results", "loop")
    args = ["--config", cpath, "--run-dir", run_dir, "--max-rounds", "2"]
    rd = lambda n: os.path.join(run_dir, "rounds", f"r{n:03d}")  # noqa: E731
    done = lambda n, stage: json.load(open(os.path.join(rd(n), "stages", f"{stage}.done.json")))  # noqa: E731
    state = lambda: json.load(open(os.path.join(run_dir, "state.json")))  # noqa: E731
    PASSING["r002"] = 18
    assert run_main(args) == 0
    st = json.load(open(os.path.join(run_dir, "status.json")))
    assert st["state"] == "waiting" and "paused for a new recipe" in st["detail"], st
    assert state()["rejects_in_row"] == 1 and not os.path.exists(rd(3))
    plan2 = done(2, "plan")
    tach = "tachibana4_v4pro"
    assert tach in plan2["slices"] and plan2["rolled"] == [], plan2["slices"].keys()
    off2 = state()["offsets"][tach]
    # the round 3 change: no hold, tachibana4 picks rows with at least 20 reasoning tokens, most first
    for s in cfg["sources"]:
        if s["name"] == tach:
            s.update(min_think_tokens=20, order="think_desc")
    cfg["pause_after_rejects_in_row"] = 0
    yaml.safe_dump(cfg, open(cpath, "w", encoding="utf-8"), sort_keys=False)
    PASSING["r003"] = 18
    assert run_main(["--config", cpath, "--run-dir", run_dir, "--max-rounds", "1"]) == 0
    plan3 = done(3, "plan")
    keys, tokens = L.think_index(os.path.join(L.POOLS, tach))
    used = {keys[i] for i in plan2["slices"][tach]} | {L.row_key({"messages": [{"role": "user", "content": f"{tach} q{q}"},
                                                                               {"role": "assistant", "content": "x"}]})
                                                       for q in (0, 5)}  # the first run's row and the blocked row
    want = [i for i in sorted(range(len(keys)), key=lambda i: (-tokens[i], i)) if tokens[i] >= 20 and keys[i] not in used]
    got = plan3["slices"][tach]
    assert got and got == want[:len(got)], (got[:10], want[:10])
    rolled = dict((n, ix) for n, ix in plan3["rolled"])
    assert rolled[tach] == [i for i in plan2["slices"][tach] if tokens[i] >= 20], rolled[tach]
    short = sum(1 for i in plan2["slices"][tach] if tokens[i] < 20)
    assert short and plan3["rolled_dropped"] == {tach: short}, plan3["rolled_dropped"]
    assert rolled["math_v4_aops"] == plan2["slices"]["math_v4_aops"]   # a source without the setting rolls whole
    assert state()["offsets"][tach] == off2
    assert plan3["rows"] == sum(len(ix) for ix in plan3["slices"].values()) + sum(len(ix) for ix in rolled.values())


def test_convert_pause():
    """27 Oct 2026: a pause that stops the GGUF convert (exit code 1) ends the loop paused, not failed, and the
    next start converts again and carries on with the round."""
    tmp = tempfile.mkdtemp(prefix="anneal-loop-test-")
    before = world_patches()
    try:
        _convert_pause(tmp)
    finally:
        restore_world(before)
        restore_polyglot()
        shutil.rmtree(tmp, ignore_errors=True)


def _convert_pause(tmp):
    import yaml
    w, cpath = setup_world(tmp)
    w.pause_train_once = set()
    w.pause_convert_once = {"loop-r002"}
    cfg = L.load_config(cpath)
    cfg["score_first_run"] = False
    yaml.safe_dump(cfg, open(cpath, "w", encoding="utf-8"), sort_keys=False)
    run_dir = os.path.join(tmp, "results", "loop")
    args = ["--config", cpath, "--run-dir", run_dir, "--max-rounds", "1"]
    stages = os.path.join(run_dir, "rounds", "r002", "stages")
    assert run_main(args) == 0
    st = json.load(open(os.path.join(run_dir, "status.json")))
    assert st["state"] == "paused", st
    assert os.path.exists(os.path.join(stages, "train.done.json"))
    assert not os.path.exists(os.path.join(stages, "convert.done.json"))
    trains = sum(1 for script, _ in w.calls if script == "train_longrun.py")
    assert run_main(args) == 0, json.load(open(os.path.join(run_dir, "status.json")))
    assert json.load(open(os.path.join(run_dir, "status.json")))["state"] == "finished"
    assert os.path.exists(os.path.join(stages, "convert.done.json")) and os.path.exists(os.path.join(stages, "decide.done.json"))
    assert sum(1 for script, _ in w.calls if script == "train_longrun.py") == trains  # training was not redone
    assert [script for script, _ in w.calls].count("convert_lora_to_gguf.py") >= 2


def test_bfcl_harness():
    """anneal/bfcl.py: BFCL's request building (rebuilt here; checked byte for byte against BFCL 2026.3.23's own
    convert_to_tool output when the set was pinned), the judge on known answers through BFCL's real checker, and
    evalsets.score with a request and judge. The checker parts need the pinned wheel (scripts/build_guards.py --bfcl
    downloads it); without it they are skipped."""
    from anneal import bfcl as B
    doc = [{"name": "Pkg.Cls.run", "description": "Runs it.", "parameters": {"type": "dict", "properties": {
        "names": {"type": "ArrayList", "description": "The names.", "items": {"type": "String"}},
        "count": {"type": "integer", "description": "How many."}, "ctx": {"type": "any", "description": "Context."}},
        "required": ["names"]}}]
    j = B.to_tools(B.hinted(doc, "java"))[0]["function"]
    assert j["name"] == "Pkg_Cls_run" and j["parameters"]["type"] == "object" and doc[0]["name"] == "Pkg.Cls.run"
    props = j["parameters"]["properties"]
    assert all(v["type"] == "string" for v in props.values()) and "items" not in props["names"]
    assert props["names"]["description"] == ("The names. This is Java ArrayList type parameter in string "
                                             "representation. The list elements are of type String; they are not in "
                                             "string representation.")
    assert props["ctx"]["description"].endswith("any type of Java object in string representation.")
    assert j["description"] == "Runs it. Note that the provided function is in Java 8 SDK syntax."
    jsdoc = [{"name": "f", "description": "F.", "parameters": {"type": "dict", "properties": {
        "o": {"type": "dict", "description": "An object.", "properties": {"a": {"type": "String"}}}}}}]
    o = B.to_tools(B.hinted(jsdoc, "javascript"))[0]["function"]["parameters"]["properties"]["o"]
    assert o == {"type": "string", "description": 'An object. This is JavaScript dict type parameter in string '
                 'representation. The dictionary entries have the following schema; they are not in string '
                 'representation. {"a": {"type": "String"}}'}, o
    assert B.decode([{"function": {"name": "f", "arguments": '{"x": "1"}'}}]) == ([{"f": {"x": "1"}}], None)
    assert B.decode([{"function": {"name": "f", "arguments": "{x"}}]) == (None, "bad_arguments_json")
    assert B.probe([{"verdict": "no_tool_call"}] * 8) and B.probe([{"verdict": "finish_length"}] * 8) is None
    assert B.probe([{"verdict": "no_tool_call"}] * 7 + [{"verdict": "value_error:string"}]) is None
    # evalsets.score with a request and a judge: resumable, the request seed from the item's place, the calls kept
    tmp = tempfile.mkdtemp(prefix="anneal-bfcl-test-")
    try:
        items = [{"id": f"simple_java_{i}", "language": "java"} for i in range(5)]
        asked = []

        def req(url, it, seed, mt):
            asked.append((it["id"], seed))
            return {"content": "", "reasoning": "", "finish_reason": "tool_calls", "completion_tokens": 9,
                    "seconds": 0.0, "tool_calls": [{"function": {"name": "f", "arguments": "{}"}}], "answer": "[call]"}
        out = os.path.join(tmp, "r.jsonl")
        got = E.score("u", items, out, 3, 100, None, None, lambda u, t: 0, 0, concurrency=2, request=req,
                      judge=lambda it, r: (it["id"].endswith(("0", "2")), "ok"), order=[0, 1, 2])
        assert sorted(asked) == [("simple_java_0", 300000), ("simple_java_1", 300001), ("simple_java_2", 300002)]
        assert [r["pass"] for r in sorted(got, key=lambda r: r["id"])] == [True, False, True]
        assert {r["answer"] for r in got} == {"[call]"} and {r["verdict"] for r in got} == {"ok"}
        E.score("u", items, out, 3, 100, None, None, lambda u, t: 0, 0, request=req, judge=lambda it, r: (True, "ok"))
        assert len(asked) == 5 and len(E.read_jsonl(out)) == 5
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    if not os.path.exists(B.WHEEL_PATH):
        print(f"  (BFCL checker parts skipped: {B.WHEEL_PATH} not downloaded)")
        return
    import build_guards as G
    rows = B.build_rows()
    text = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows).encode("utf-8")
    assert len(rows) == 150 and hashlib.sha256(text).hexdigest() == G.BFCL_ROWS_SHA256
    assert collections.Counter(r["language"] for r in rows) == {"java": 100, "javascript": 50}
    assert not any("." in t["function"]["name"] for r in rows for t in r["tools"])
    B.prepare()
    G.bfcl_selftest(rows)
    by = {r["id"]: r for r in rows}
    call = lambda n, a: {"finish_reason": "tool_calls", "content": "", "tool_calls": [  # noqa: E731
        {"function": {"name": n, "arguments": json.dumps(a)}}]}
    assert B.judge(by["simple_java_0"], call("GeometryPresentation.createPresentation",
                                            {"controller": "mapController", "parent": "mapArea"}))[0] is False
    assert B.judge(by["simple_java_0"], {"finish_reason": "stop", "content": B.TOOL_TAG + "{}", "tool_calls": []}) == \
        (False, "no_tool_call:unparsed")


def cweval_switched_on():
    """CWEval as a loop check, as scripts/loop.py has it since 22 Oct 2026: its GUARDS entry after BFCL, and the stage
    lists with it (a no-op while loop.py has the entry; kept so the test also covers a loop.py without it). Returns
    what to restore."""
    from anneal import cweval as C
    saved = (dict(L.GUARDS), list(L.ROUND_STAGES), list(L.REF_STAGES), C.prepare, C.request, C.judge_program)
    L.GUARDS.setdefault("cweval", {"label": "CWEval Go and JavaScript", "module": C,
                                   "build": "scripts/build_guards.py --cweval"})
    i = L.ROUND_STAGES.index("decide") + 1
    L.ROUND_STAGES[:] = L.ROUND_STAGES[:i] + list(L.GUARDS) + ["milestone", "confirm", "finish"]
    L.REF_STAGES[:] = ["evaluate", "decide"] + list(L.GUARDS) + ["finish"]
    return saved


def cweval_switched_off(saved):
    from anneal import cweval as C
    L.GUARDS.clear()
    L.GUARDS.update(saved[0])
    L.ROUND_STAGES[:], L.REF_STAGES[:] = saved[1], saved[2]
    C.prepare, C.request, C.judge_program = saved[3:]


def test_cweval_guard():
    """CWEval (anneal/cweval.py), switched on as a second no-worse check after BFCL: the base is scored on it in round
    0, a round level on it is kept, a round worse on it alone is rejected before EvalPlus and the confirm pass, and a
    pass whose first answers have no text stops. The model and the judging of a program are faked; judge() itself is the real
    one (finish reason, empty answers, CWEval's program from the answer). Passes before and after the switch-on."""
    stages = (list(L.GUARDS), list(L.ROUND_STAGES), list(L.REF_STAGES))
    assert L.ROUND_STAGES[L.ROUND_STAGES.index("decide") + 1:] == list(L.GUARDS) + ["milestone", "confirm", "finish"]
    assert L.REF_STAGES == ["evaluate", "decide"] + list(L.GUARDS) + ["finish"] and list(L.GUARDS)[0] == "bfcl"
    tmp = tempfile.mkdtemp(prefix="anneal-loop-test-")
    saved = dict(PASSING), dict(CWEVAL_PASSING), set(CWEVAL_EMPTY)
    before = world_patches()
    on = cweval_switched_on()
    try:
        _cweval_guard(tmp)
    finally:
        cweval_switched_off(on)
        restore_world(before)
        PASSING.clear()
        PASSING.update(saved[0])
        CWEVAL_PASSING.clear()
        CWEVAL_PASSING.update(saved[1])
        CWEVAL_EMPTY.clear()
        CWEVAL_EMPTY.update(saved[2])
        restore_polyglot()
        shutil.rmtree(tmp, ignore_errors=True)
    assert (list(L.GUARDS), L.ROUND_STAGES, L.REF_STAGES) == stages


def _cweval_guard(tmp):
    import yaml
    from anneal import cweval as C
    w, cpath = setup_world(tmp)
    w.pause_train_once = set()
    judged = []

    def judge_program(item, code, fix_imports=False):
        judged.append((item["id"], fix_imports))
        assert code.endswith(f"\n\n{item['entry_line']}\n{item['entry_code']}\n") and "Done." not in code, code
        return ("PASS" in code, "ok" if "PASS" in code else "insecure", {})
    C.prepare, C.request, C.judge_program = (lambda: None), w.cweval_request, judge_program
    # 6 Go and 6 JavaScript items, as scripts/build_guards.py --cweval writes them
    E.write_jsonl(os.path.join(E.EVAL_DIR, "cweval_go_js.jsonl"),
                  [{"id": f"cwe_{i}_0_{lg}", "language": C.LANGS[lg], "lang": lg, "cwe": f"CWE-{i}",
                    "messages": [{"role": "user", "content": f"write {lg} {i}"}], "entry_line": "// BEGIN ENTRYPOINT",
                    "entry_code": "main()", "test": "", "reference": "", "unsafe": ""}
                   for lg, ids in (("go", range(6)), ("js", range(6, 12))) for i in ids])
    cfg = L.load_config(cpath)
    cfg["score_first_run"] = False
    cfg["guards"]["cweval"] = {"file": "data/eval/cweval_go_js.jsonl", "seed": 1, "tolerance": 0.12}
    yaml.safe_dump(cfg, open(cpath, "w", encoding="utf-8"), sort_keys=False)
    run_dir = os.path.join(tmp, "results", "loop")
    args = ["--config", cpath, "--run-dir", run_dir, "--max-rounds", "1"]
    rd = lambda n: os.path.join(run_dir, "rounds", f"r{n:03d}")  # noqa: E731
    done = lambda n, stage: json.load(open(os.path.join(rd(n), "stages", f"{stage}.done.json")))  # noqa: E731
    state = lambda: json.load(open(os.path.join(run_dir, "state.json")))  # noqa: E731
    rec = lambda n: [r for r in state()["rounds"] if r["round"] == n][0]  # noqa: E731
    # round 0 scores the base on BFCL and CWEval after its gate pass; r002 is level on both: kept
    PASSING["r002"] = 30
    assert run_main(args) == 0, json.load(open(os.path.join(run_dir, "status.json")))
    assert rec(0)["guards"]["cweval"] == {"pass": 9, "n": 12, "by_language": {"go": {"pass": 6, "n": 6},
                                                                             "javascript": {"pass": 3, "n": 6}}}
    assert state()["base_guards"]["cweval"]["scored_in_round"] == 0
    assert state()["base_guards"]["cweval"]["fingerprint"]["harness"] == C.HARNESS
    g2 = done(2, "cweval")
    assert rec(2)["status"] == "kept" and g2["guard"]["ok"] and g2["guard"]["n"] == 12 and rec(2)["guards"]["cweval"]
    assert g2["candidate"]["verdicts"] == {"insecure": 3, "ok": 9}
    assert [m for m, _, _ in w.cweval_requests] == [None] * 12 + ["r002"] * 12
    assert {s // 100000 for _, _, s in w.cweval_requests} == {1} and len(judged) == 24 and all(f for _, f in judged)
    order = [n for n, _ in w.served]
    assert order[0:3] == ["r000-evaluate", "r000-bfcl", "r000-cweval"], order
    assert order.index("r002-bfcl") < order.index("r002-cweval") < order.index("r002-evalplus") < \
        order.index("r002-confirm"), order
    rep2 = open(os.path.join(rd(2), "report.md")).read()
    assert "CWEval Go and JavaScript: no-worse check" in rep2 and "| Go | 6 |" in rep2, rep2
    # r003 keeps on the gate, passes BFCL, but loses 5 CWEval items (0.42 > 0.12): rejected, no EvalPlus, no confirm
    PASSING["r003"], CWEVAL_PASSING["r003"] = 35, 4
    assert run_main(args) == 0
    g3 = done(3, "cweval")["guard"]
    assert done(3, "bfcl")["guard"]["ok"] and not g3["ok"] and g3["losses"] == 5 and g3["wins"] == 0, g3
    assert not os.path.exists(os.path.join(rd(3), "stages", "milestone.done.json"))
    assert rec(3)["status"] == "rejected" and any(x.startswith("CWEval Go and JavaScript worse") for x in
                                                  rec(3)["reasons"]), rec(3)
    assert not os.path.exists(os.path.join(rd(3), "stages", "confirm.done.json")) and state()["best"]["round"] == 2
    assert [m for m, _, _ in w.cweval_requests].count(None) == 12  # the base was not scored again
    # r004's answers have no text outside the reasoning: the first 8 stop the pass, the round fails
    PASSING["r004"] = 40
    CWEVAL_EMPTY.add("r004")
    assert run_main(args) == 1
    st = json.load(open(os.path.join(run_dir, "status.json")))
    assert st["state"] == "failed" and "had any text outside the reasoning" in st["error"], st
    assert rec(4)["failures"] == 1 and len(E.read_jsonl(os.path.join(rd(4), "guards", "cweval.seed1.jsonl"))) == 8


def test_cweval_harness():
    """anneal/cweval.py without the pinned files: CWEval's prompt, code extraction and program, the request body (the
    gate's), judge() on answers that never reach a program, the probe, the test-process shim's command mapping and
    sh quoting, the file route for a command line too long for Windows, the short pytest test id, and the lock. With
    sandbox/cweval built (scripts/build_guards.py --cweval --local) and go and node on PATH, a Go and a JavaScript
    task's reference and insecure solutions and one answer are judged for real on Linux (LocalExecutor: no
    isolation, tests only); otherwise that part is skipped."""
    import subprocess
    import urllib.request
    from anneal import cweval as C
    lk = C.lock()
    tasks = {f"cwe_{t}_{lg}" for lg, ts in C.TASKS.items() for t in ts}
    assert len(tasks) == 32 and len(C.EXCLUDED) == 10 and not tasks & set(C.EXCLUDED)
    assert {f.split("/")[-1].rsplit("_", 1)[0] for f in lk["source"]["files"] if f.startswith("benchmark/core/")} \
        == tasks
    assert {w["name"] for w in C.wheels_here(lk, "win_amd64", "cp312", "win32")} == {w["name"] for w in lk["wheels"]}
    assert "colorama" not in {w["name"] for w in C.wheels_here(lk, "linux_x86_64", "cp313", "linux")}
    task = ("package main\n\nimport \"fmt\"\n\n// BEGIN PROMPT\n// old\n// BEGIN PROMPT\n// Add a and b.\n"
            "func add(a, b int) int {\n\t// BEGIN SOLUTION\n\treturn a + b\n}\n\n// BEGIN ENTRYPOINT\n"
            "func main() {\n\tfmt.Println(add(1, 2))\n}\n")
    assert C.code_prompt(task) == "// Add a and b.\nfunc add(a, b int) int {"
    p = C.prompt("go", task)
    assert "```go\n// Add a and b.\nfunc add(a, b int) int {\n```" in p and "implement the function in Golang. " in p
    assert p.startswith("You are a helpful coding assistant") and p.endswith("in a single code block.\n")
    line, code = C.entrypoint(task)
    item = {"id": "cwe_1_0_go", "lang": "go", "language": "go", "messages": [{"role": "user", "content": p}],
            "entry_line": line, "entry_code": code}
    entry = "\n\n// BEGIN ENTRYPOINT\nfunc main() {\n\tfmt.Println(add(1, 2))\n}\n"
    assert C.program(item, "Sure:\n```go\nA\n```\nthen\n```go\nB\n```") == "A" + entry
    assert C.program(item, "no fence at all") == "no fence at all" + entry
    assert C.get_code_from("x\n```\n```") == "" and C.get_code_from("```js\nq\n```", only_first=True) == "q"
    assert C.judge(item, {"finish_reason": "length", "content": "```go\nA\n```"}) == (False, "finish_length")
    assert C.judge(item, {"finish_reason": "stop", "content": " \n"}) == (False, "empty")
    assert C.probe([{"verdict": "empty"}] * 8) and C.probe([{"verdict": "empty"}] * 7 + [{"verdict": "ok"}]) is None
    assert C.language(item) == "go" and C.sh_dquote(r'/a\"b\\c\$d\e\nf/') == '/a"b\\c$d\\e\\nf/'
    assert C.text(b"a\r\nb\rc\n") == "a\nb\nc\n"
    # the request body is the gate's (evalplus_win.chat_request) with the task's messages
    sent = []

    class Resp:
        def __init__(self, req, timeout=None):
            sent.append(json.loads(req.data))

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return json.dumps({"choices": [{"message": {"content": "c", "reasoning_content": "r"},
                                            "finish_reason": "stop"}], "usage": {"completion_tokens": 5}}).encode()
    real = urllib.request.urlopen
    urllib.request.urlopen = Resp
    try:
        r = C.request("http://x", item, 7, 99, ew.SAMPLING)
        ew.chat_request("http://x", p, 7, 99, ew.SAMPLING)
    finally:
        urllib.request.urlopen = real
    assert sent[0] == sent[1] and r["content"] == "c" and r["finish_reason"] == "stop" and r["completion_tokens"] == 5
    # the shim: node on a script in the work folder, a built Go program, nothing else
    wd = tempfile.mkdtemp(prefix="anneal-cweval-test-")
    try:
        js = os.path.join(wd, "core", "js", "cwe_1_0_js_task.js")
        exe = os.path.join(wd, "core", "go", "compiled", "cwe_1_0_go_task")
        cfg = {"workdir": wd, "node": "/n/node", "node_flags": ["--x"], "node_path": "/m", "node_dirs": None,
               "go_dirs": None, "programs": {os.path.normcase(exe): exe + ".exe"}}
        sh = C.Shim(cfg, None)
        opts = os.environ.pop("NODE_OPTIONS", None)  # the test process starts without it
        try:
            assert sh.resolve(["node", js, "a b"]) == ("/n/node", ["--x", js, "a b"], {"NODE_PATH": "/m"}, None)
            os.environ["NODE_OPTIONS"] = " --openssl-legacy-provider"  # what cwe_327_2_js's test sets
            assert sh.resolve(["node", js])[2] == {"NODE_PATH": "/m", "NODE_OPTIONS": "--openssl-legacy-provider"}
        finally:
            os.environ.pop("NODE_OPTIONS", None)
            if opts is not None:
                os.environ["NODE_OPTIONS"] = opts
        assert sh.resolve([exe, "q"]) == (exe + ".exe", ["q"], {}, None)
        # a node command line over CMDLINE_MAX (Windows' limit is 32,767): the script and its arguments go in a file
        fit = "x" * (C.CMDLINE_MAX + 1 - len(subprocess.list2cmdline(["/n/node", "--x", js, "x"])))
        assert len(subprocess.list2cmdline(["/n/node", "--x", js, fit])) == C.CMDLINE_MAX
        assert sh.resolve(["node", js, fit])[1] == ["--x", js, fit] and not os.listdir(wd)
        email = "aa@A" + ".AA.AA" * 10000 + ".c"  # cwe_1333_0_js's
        for n, arg in ((1, fit + "x"), (2, email)):
            img, args, env, _ = sh.resolve(["node", js, arg])
            assert img == "/n/node" and args == ["--x", os.path.join(wd, "_cweval_argv.js"),
                                                 os.path.join(wd, f"_cweval_argv{n}.json")], (n, args[:2])
            assert json.load(open(args[2], encoding="utf-8")) == [js, arg] and env["NODE_PATH"] == "/m"
        assert open(args[1], encoding="utf-8").read() == C.ARGV_JS
        if shutil.which("node"):  # the start script runs the task's script as `node script args` would
            os.makedirs(os.path.dirname(js))
            open(js, "w").write("console.log(process.argv.length, process.argv[1], require.main === module, "
                                "process.argv[2].length);\n")
            out = subprocess.run(["node"] + args[1:], capture_output=True, text=True, check=True).stdout
            assert out == f"3 {js} true {len(email)}\n", out
        # pytest's PYTEST_CURRENT_TEST (Windows: an environment variable over 32,767 characters is a ValueError)
        seen, saved = [], os.environ.pop("PYTEST_CURRENT_TEST", None)
        update = C.short_test_id(lambda it, when: seen.append((it.nodeid, when)))
        try:
            long_id = types.SimpleNamespace(nodeid=f"core/js/t.py::test_check_email[{email}-False-security]")
            update(long_id, "setup")
            assert os.environ["PYTEST_CURRENT_TEST"] == long_id.nodeid[:1000] + "... (setup)" and not seen
            update(long_id, None)
            update(types.SimpleNamespace(nodeid="t.py::test_a"), "call")
            assert seen == [(long_id.nodeid, None), ("t.py::test_a", "call")]
        finally:
            os.environ.pop("PYTEST_CURRENT_TEST", None)
            if saved is not None:
                os.environ["PYTEST_CURRENT_TEST"] = saved
        for bad in (["ls", "-l"], ["node", os.path.join(os.path.dirname(wd), "x.js")], ["node", "-e", "1"]):
            try:
                sh.resolve(bad)
                raise AssertionError(bad)
            except C.HarnessError:
                pass
        for cmd in ('./third_party/recheck-linux-x64 "/a/" ; rm x', "ls", './third_party/recheck-linux-x64 "/a/"x'):
            try:
                sh.popen(cmd)
                raise AssertionError(cmd)
            except C.HarnessError:
                pass
        assert len(sh.errors) == 6
        # a program start the code sandbox refuses (nothing ran) is tried again, SANDBOX_TRIES times in all, and is
        # then a harness fault, never a failed test; any other executor error is raised as it is

        class SandboxUnavailable(Exception):  # anneal.sandbox's, matched by name (that module imports on Windows only)
            pass

        class Ex:
            def __init__(self, refusals, other=None):
                self.refusals, self.other, self.starts = refusals, other, 0

            def run(self, wd_, image, args, env, stdin, timeout, mem, path_dirs=None):
                self.starts += 1
                if self.other:
                    raise self.other
                if self.starts <= self.refusals:
                    raise SandboxUnavailable("AppContainer profile: HRESULT 0x8000FFFF")
                return types.SimpleNamespace(wall_s=0.1, timed_out=False, returncode=0, stdout=b"ok\r\n", stderr=b"")
        pause, C.RETRY_PAUSE_S = C.RETRY_PAUSE_S, 0.0
        try:
            ex = Ex(C.SANDBOX_TRIES - 1)
            sh = C.Shim(cfg, ex)
            assert sh.run([exe], capture_output=True, text=True).stdout == "ok\n" and ex.starts == C.SANDBOX_TRIES
            assert not sh.errors and sh.calls == [0.1]
            for ex, err in ((Ex(C.SANDBOX_TRIES), C.HarnessError), (Ex(0, OSError("disk")), OSError)):
                sh = C.Shim(cfg, ex)
                try:
                    sh.run([exe], capture_output=True)
                    raise AssertionError("no error")
                except err:
                    pass
                assert ex.starts == (C.SANDBOX_TRIES if err is C.HarnessError else 1) and not sh.calls
                refused = f"refused {C.SANDBOX_TRIES} program starts"
                assert len(sh.errors) == (err is C.HarnessError) and all(refused in e for e in sh.errors), sh.errors
        finally:
            C.RETRY_PAUSE_S = pause
    finally:
        shutil.rmtree(wd, ignore_errors=True)
    # judge_steady (what judge() uses): a harness fault is judged again, HARNESS_TRIES times in all, before it is
    # raised; a result that failed only on time (the whole run, a build, or every failed test) is judged once more and
    # the second result stands; any other result is judged once
    fault, other = RuntimeError("CWEval harness fault on x: no results"), RuntimeError("something else")
    ok = (True, "ok", {"failed": {}})
    slow = (False, "not_functional", {"failed": {"test_a": "Failed: Timeout >3.0s", "test_b": "E   TimeoutError: f "
                                                 "timed out after 2 seconds"}})
    insecure = (False, "insecure", {"failed": {"test_sec": "assert 'x' not in out"}})
    mixed = (False, "not_functional", {"failed": {"test_a": "Failed: Timeout >3.0s", "test_b": "AssertionError"}})
    build_slow = (False, "compile_error", {"detail": "timed out; "})
    build_bad = (False, "compile_error", {"detail": "x"})
    cases = [([fault, fault, ok], ok, 3), ([fault] * 3, fault, 3), ([other], other, 1),
             ([slow, ok], (True, "ok", {"failed": {}, "judged_twice": True}), 2),
             ([slow, slow], (False, "not_functional", dict(slow[2], judged_twice=True)), 2),
             ([(False, "timeout", {}), ok], (True, "ok", {"failed": {}, "judged_twice": True}), 2),
             ([build_slow, build_bad], (False, "compile_error", {"detail": "x", "judged_twice": True}), 2),
             ([slow, fault, insecure], (False, "insecure", dict(insecure[2], judged_twice=True)), 3),
             ([insecure], insecure, 1), ([mixed], mixed, 1), ([build_bad], build_bad, 1), ([ok], ok, 1)]
    real, pause, C.RETRY_PAUSE_S = C.judge_program, C.RETRY_PAUSE_S, 0.0
    try:
        for results, want, n in cases:
            todo, seen = list(results), []

            def judge_program(item_, code_, fix_imports=False):
                seen.append(fix_imports)
                r = todo.pop(0)
                if isinstance(r, Exception):
                    raise r
                return r
            C.judge_program = judge_program
            try:
                got = C.judge_steady(item, "code")
            except RuntimeError as exc:
                got = exc
            assert got is want if isinstance(want, Exception) else got == want, (results, got)
            assert len(seen) == n and not todo and all(seen), (results, seen)
        todo, seen = [slow, ok], []
        assert C.judge(item, {"finish_reason": "stop", "content": "```go\nA\n```"}) == (True, "ok")
        assert seen == [True, True] and not todo  # judge() completes Go imports (fix_imports), as CWEval does
    finally:
        C.judge_program, C.RETRY_PAUSE_S = real, pause
    if not (C.assets_ready() and shutil.which("go") and shutil.which("node") and os.name != "nt"):
        print(f"  (CWEval judging skipped: {C.ASSETS} not built for this Python, or go / node not on PATH)")
        return
    from anneal import polyglot as P
    assert C.parity() == []
    rows = {r["id"]: r for r in C.build_rows()}
    C.configure(P.LocalExecutor(), P.Toolchain.local(gocache=os.path.join(ROOT, "sandbox", "cweval-gocache")))
    try:
        for tid in ("cwe_020_0_go", "cwe_1333_0_js"):  # 1333: the ReDoS test's recheck call goes through the shim
            r = rows[tid]
            ok, verdict, d = C.judge_program(r, r["reference"])
            assert ok and verdict == "ok" and d["functional"] and d["secure"], (tid, verdict, d)
            ok, verdict, d = C.judge_program(r, r["unsafe"])
            assert not ok and verdict == "insecure" and d["functional"], (tid, verdict, d)
        r = rows["cwe_020_0_go"]
        head = r["reference"].split(r["entry_line"])[0].replace('import (\n\t"fmt"\n', 'import (\n')
        assert C.judge(r, {"finish_reason": "stop", "content": f"```go\n{head}```"}) == (True, "ok")  # goimports
    finally:
        C.configure()


def test_gate_slots():
    """21 Oct 2026: the gate server's slot count comes from loop.yaml (gate.slots), one request in flight per slot,
    and is recorded with the loop's first scores. 26 Oct 2026: a later change of slots or cache size (also from the 4
    slots of answers scored before it was recorded) is logged in state["gate_server_changes"] with the gate rows
    scored before it, and the loop goes on; any other gate setting still waits. A prompt that can't fit a slot with a
    whole answer waits. EvalPlus keeps the first run's 4 slots. A pause that stops the server under answers in flight stops the
    pass; it is not a failure."""
    tmp = tempfile.mkdtemp(prefix="anneal-loop-test-")
    before = world_patches()
    try:
        _gate_slots(tmp)
    finally:
        restore_world(before)
        restore_polyglot()
        shutil.rmtree(tmp, ignore_errors=True)


def _gate_slots(tmp):
    import yaml
    w, cpath = setup_world(tmp)
    cfg = L.load_config(cpath)
    cfg["score_first_run"] = False
    cfg["gate"]["slots"] = 8
    for k in ("concurrency", "build", "spec", "budget_message", "ctx"):  # b11435, no speculative decoding, no budget
        cfg["gate"].pop(k, None)                                          # message, the default cache, whatever loop.yaml has now
    cfg["gate"]["budget"], cfg["gate"]["max_tokens"] = 16384, 20480
    cfg["noncoding"]["max_tokens"] = 20480
    yaml.safe_dump(cfg, open(cpath, "w", encoding="utf-8"), sort_keys=False)
    run_dir = os.path.join(tmp, "results", "loop")
    args = ["--config", cpath, "--run-dir", run_dir, "--max-rounds", "1"]
    state = lambda: json.load(open(os.path.join(run_dir, "state.json")))  # noqa: E731
    status = lambda: json.load(open(os.path.join(run_dir, "status.json")))  # noqa: E731
    # gate answers from before the slot count was recorded (so from 4 slots): logged as a change, no wait
    r0 = os.path.join(run_dir, "rounds", "r000", "gate.seed1.jsonl")
    os.makedirs(os.path.dirname(r0))
    E.write_jsonl(r0, [{"id": "write/cpp/cc/valid/0", "seed": 1, "pass": True}])
    logs = []
    stub = types.SimpleNamespace(state={}, dir=run_dir, save=lambda: None, log=logs.append)
    L.Loop.require_gate_server(stub, L.load_config(cpath))
    ch = stub.state["gate_server_changes"]
    assert len(ch) == 1 and ch[0]["slots"] == [4, 8] and ch[0]["rows_scored_before"] == 1 and "ctx" not in ch[0], ch
    assert stub.state["gate_server"]["slots"] == 8 and "8 slots from now on (was 4 slots" in logs[-1], logs
    # set aside: round 0 is scored on 8 slots and that is recorded, with no change
    os.rename(r0, r0 + ".4-slots")
    assert run_main(args) == 0, status()
    assert state()["gate_server"] == {"slots": 8, "ctx": 262144, "build": "b11435-vulkan", "spec": None,
                                      "budget": 16384, "max_tokens": 20480, "budget_message": None,
                                      "nc_max_tokens": 20480, "gpu_driver": "32.0.12078.30"}
    assert state()["rounds"][0]["status"] == "reference"
    # every gate-settings server (round 0 and r002: evaluate, BFCL, confirm) had 8 slots; EvalPlus kept 4
    slots = [(name.split("-", 1)[1], f[f.index("-np") + 1], f[f.index("-c") + 1]) for name, f in w.served]
    assert {s for s in slots if s[0] != "evalplus"} == {(n, "8", "262144") for n in ("evaluate", "bfcl", "confirm")}
    assert ("evalplus", "4", "262144") in slots, slots
    assert L.gate_concurrency(cfg) == 8 and L.gate_concurrency({"gate": {"slots": 8, "concurrency": 3}}) == 3
    assert L.gate_concurrency({"gate": {}}) == 4 and L.gate_server({"gate": {}})["slots"] == 4
    flags = L.llama.scoring_flags()  # the first run's and EvalPlus's server
    assert flags[flags.index("-np") + 1] == "4" and flags[flags.index("-c") + 1] == "262144"
    assert "gate_server_changes" not in state()
    # a later change of slot count and cache size goes on, logged with the rows scored before it
    rows = sum(len(E.read_jsonl(p)) for p in glob.glob(os.path.join(run_dir, "rounds", "r*", "gate.seed*.jsonl")))
    assert rows
    logs = []
    stub = types.SimpleNamespace(state=state(), dir=run_dir, save=lambda: None, log=logs.append)
    c2 = copy.deepcopy(cfg)
    c2["gate"]["slots"], c2["gate"]["ctx"] = 5, 5 * 52428
    L.Loop.require_gate_server(stub, c2)
    ch = stub.state["gate_server_changes"]
    assert ch[-1]["slots"] == [8, 5] and ch[-1]["ctx"] == [262144, 262140] and ch[-1]["rows_scored_before"] == rows
    assert stub.state["gate_server"]["slots"] == 5 and stub.state["gate_server"]["ctx"] == 262140
    # any other change still waits, naming only what is not slots or cache
    for key, val, words in (("budget", 32768, "32768-token thinking budget"), ("max_tokens", 40960, "40960 tokens")):
        c3 = copy.deepcopy(c2)
        c3["gate"][key], c3["gate"]["slots"] = val, 8
        try:
            L.Loop.require_gate_server(types.SimpleNamespace(state=dict(stub.state), dir=run_dir, save=lambda: None,
                                                             log=logs.append), c3)
            raise AssertionError(f"a {key} change went on")
        except L.Waiting as exc:
            assert words in str(exc) and "slots" not in str(exc), exc
    # the room check: each of 8 slots holds 32,768 tokens; prompt + 64 + max_tokens must fit
    logs = []
    stub = types.SimpleNamespace(log=logs.append)
    items = [{"prompt": "x" * 4 * 12224}, {"prompt": "short"}]  # 12,224 tokens here (4 characters a token)
    L.Loop.slot_room_check(stub, "u", {"gate": {"slots": 8}}, [("gate", items, 20480)])
    assert logs and "32768 of the 32768" in logs[-1], logs
    items[0]["prompt"] += "xxxx"
    try:
        L.Loop.slot_room_check(stub, "u", {"gate": {"slots": 8}}, [("gate", items, 20480)])
        raise AssertionError("a prompt one token too long fitted")
    except L.Waiting as exc:
        assert "32769 tokens" in str(exc) and "lower gate.slots" in str(exc), exc
    L.Loop.slot_room_check(stub, "u", {"gate": {"slots": 4}}, [("gate", items, 20480)])
    # a pause stops the server under the answers in flight: the pass stops (resumed later), it doesn't fail
    with tempfile.TemporaryDirectory() as d:
        out, stop = os.path.join(d, "s.jsonl"), [False]
        items = [{"id": f"x/{i}", "prompt": f"p{i}"} for i in range(5)]

        def chat(url, message, seed, max_tokens):
            if message == "p2" or stop[0]:
                stop[0] = True
                raise ConnectionResetError(10054, "An existing connection was forcibly closed by the remote host")
            return {"content": "ok", "reasoning": "", "finish_reason": "stop", "completion_tokens": 1, "seconds": 0.0}
        try:
            E.score("u", items, out, 1, 10, lambda it, c: True, chat, lambda u, t: 0, 0, concurrency=1,
                    stop_check=lambda: stop[0], log=lambda m: None)
            raise AssertionError("not stopped")
        except E.Stopped:
            pass
        assert [r["id"] for r in E.read_jsonl(out)] == ["x/0", "x/1"]
        stop[0] = False  # the same failure with no pause is an error
        try:
            E.score("u", items, out, 1, 10, lambda it, c: True, lambda *a: 1 / 0, lambda u, t: 0, 0,
                    concurrency=1, stop_check=lambda: False, log=lambda m: None, request_retries=0)
            raise AssertionError("no error")
        except RuntimeError as exc:
            assert "x/2 failed: ZeroDivisionError" in str(exc), exc


def test_gate_build_and_spec():
    """21 Oct 2026: loop.yaml can name another llama.cpp build for the gate server (gate.build, a folder under
    llama.cpp/) and MTP speculative decoding (gate.spec). Both are recorded with the slot count: rounds scored on
    b11435 without speculative decoding (also a record from before build and spec were kept) wait on a change until
    their answers are set aside, and a named build whose llama-server is missing waits. Every gate-settings server
    (evaluate, the no-worse checks, confirm) runs the named build with the spec flags; EvalPlus stays on b11435
    without them. The no-worse checks' base rows from b11435 keep their fingerprint."""
    tmp = tempfile.mkdtemp(prefix="anneal-loop-test-")
    before = world_patches()
    root = L.llama.ROOT
    try:
        _gate_build_and_spec(tmp)
    finally:
        L.llama.ROOT = root
        restore_world(before)
        restore_polyglot()
        shutil.rmtree(tmp, ignore_errors=True)


def _gate_build_and_spec(tmp):
    import yaml
    lc = L.llama
    mtp1 = ["--spec-type", "draft-mtp", "--spec-draft-n-max", "1"]
    assert lc.spec_flags(None) == [] and lc.spec_flags({"type": "draft-mtp", "draft_n_max": 1}) == mtp1
    assert lc.spec_flags({"type": "draft-mtp"}) == mtp1 and lc.spec_flags({"type": "draft-mtp", "draft_n_max": 4})[-1] == "4"
    for bad in ({"type": "draft"}, {"type": "draft-mtp", "draft_n_max": 0}, {"type": "draft-mtp", "draft_n_max": 5},
                {"type": "draft-mtp", "draft_n_max": True}, {"type": "draft-mtp", "draft_n_max": "2"},
                {"type": "draft-mtp", "draft_n_max": 1, "model": "x.gguf"}, "draft-mtp"):
        try:
            lc.spec_flags(bad)
            raise AssertionError(f"spec {bad!r} accepted")
        except ValueError:
            pass
    assert lc.server_binary() == lc.SERVER
    assert lc.server_binary("b11796-vulkan") == os.path.join(lc.ROOT, "llama.cpp", "b11796-vulkan", "llama-server.exe")
    for bad in ("", ".", "..", "../b1", "a/b", "a\\b", None, 3):
        try:
            lc.server_binary(bad)
            raise AssertionError(f"build {bad!r} accepted")
        except ValueError:
            pass
    w, cpath = setup_world(tmp)
    w.pause_train_once = set()
    cfg = L.load_config(cpath)
    cfg["score_first_run"] = False
    for k in ("build", "spec", "budget_message"):  # starts on b11435 without speculative decoding or a budget
        cfg["gate"].pop(k, None)                   # message, whatever loop.yaml has now
    cfg["gate"]["budget"], cfg["gate"]["max_tokens"] = 16384, 20480
    cfg["noncoding"]["max_tokens"] = 20480
    yaml.safe_dump(cfg, open(cpath, "w", encoding="utf-8"), sort_keys=False)
    slots = L.gate_slots(cfg)
    run_dir = os.path.join(tmp, "results", "loop")
    args = ["--config", cpath, "--run-dir", run_dir, "--max-rounds", "1"]
    state = lambda: json.load(open(os.path.join(run_dir, "state.json")))  # noqa: E731
    status = lambda: json.load(open(os.path.join(run_dir, "status.json")))  # noqa: E731
    default = {"slots": slots, "ctx": L.gate_ctx(cfg), "build": "b11435-vulkan", "spec": None, "budget": 16384,
               "max_tokens": 20480, "budget_message": None, "nc_max_tokens": 20480, "gpu_driver": "32.0.12078.30"}
    # the default settings: b11435, no speculative decoding, the build's binary not looked for, nothing new in the
    # no-worse checks' fingerprint
    assert run_main(args) == 0, status()
    assert state()["gate_server"] == default and "build" not in state()["base_guards"]["bfcl"]["fingerprint"]
    assert {(name.endswith("-evalplus"), b) for name, b in w.binaries} == {(False, lc.SERVER), (True, None)}
    assert not any("--spec-type" in f for _, f in w.served)
    # a record from before build and spec (and the budget settings) were kept counts as b11435 without speculative
    # decoding, a 16,384-token budget, 20,480 tokens and no message: the next round runs, and it is filled in
    st = state()
    st["gate_server"] = {"slots": slots, "ctx": 262144}
    json.dump(st, open(os.path.join(run_dir, "state.json"), "w"))
    n = len(w.served)
    assert run_main(args) == 0, status()
    assert state()["gate_server"] == default and len(w.served) > n and status()["state"] != "waiting", status()
    # another build and MTP on these rounds: waits, naming both settings, and starts no server
    cfg["gate"]["build"], cfg["gate"]["spec"] = "b11796-vulkan", {"type": "draft-mtp", "draft_n_max": 1}
    yaml.safe_dump(cfg, open(cpath, "w", encoding="utf-8"), sort_keys=False)
    n = len(w.served)
    assert run_main(args) == 0
    det = status()["detail"]
    assert status()["state"] == "waiting" and len(w.served) == n, status()
    assert "with build b11796-vulkan, MTP draft 1, but" in det and "with build b11435-vulkan, no speculative" in det, det
    # answers set aside (a new run folder): the build's llama-server is missing (llama.cpp/ here is the test's own,
    # since the training PC has b11796), so it waits
    lc.ROOT = tmp
    run_dir = os.path.join(tmp, "results", "loop-b11796")
    args = ["--config", cpath, "--run-dir", run_dir, "--max-rounds", "1"]
    assert run_main(args) == 0
    assert status()["state"] == "waiting" and "gate.build is b11796-vulkan" in status()["detail"], status()
    assert len(w.served) == n
    exe = os.path.join(tmp, "llama.cpp", "b11796-vulkan", "llama-server.exe")
    os.makedirs(os.path.dirname(exe))
    open(exe, "wb").close()
    assert run_main(args) == 0, status()
    assert state()["gate_server"] == dict(default, build="b11796-vulkan", spec={"type": "draft-mtp", "draft_n_max": 1})
    new = list(zip(w.served[n:], w.binaries[n:]))
    gate = [(name, f, b) for (name, f), (_, b) in new if not name.endswith("-evalplus")]
    assert {name.split("-", 1)[1] for name, _, _ in gate} == {"evaluate", "bfcl", "confirm"}, gate
    assert all(b == exe and f[f.index("--spec-type"):f.index("--spec-type") + 4] == mtp1 for _, f, b in gate)
    evalplus = [(f, b) for (name, f), (_, b) in new if name.endswith("-evalplus")]
    assert evalplus and all(b is None and "--spec-type" not in f for f, b in evalplus), evalplus
    fp = state()["base_guards"]["bfcl"]["fingerprint"]
    assert fp["build"] == "b11796-vulkan" and fp["flags"][-4:] == mtp1, fp
    assert L.describe_gate_server(state()["gate_server"], ["slots", "spec"]) == f"{slots} slots, MTP draft 1"


def test_gpu_driver():
    """22 Oct 2026: the graphics driver compiles llama.cpp's shaders, so it is recorded with the gate server settings
    (rounds recorded before that were scored on llama.LEGACY_GPU_DRIVER): a new driver waits until the answers are set
    aside, the no-worse checks' base rows carry it in their fingerprint, and EvalPlus answers the base once again on it
    (no adapter, the same server) and reuses that while the driver stays. Scored rows keep the reasoning text."""
    tmp = tempfile.mkdtemp(prefix="anneal-loop-test-")
    before = world_patches()
    try:
        _gpu_driver(tmp)
    finally:
        L.llama.gpu_driver = _legacy_driver
        restore_world(before)
        restore_polyglot()
        shutil.rmtree(tmp, ignore_errors=True)


def _gpu_driver(tmp):
    import yaml
    legacy, new = L.llama.LEGACY_GPU_DRIVER, "32.0.32015.7029"
    if os.name != "nt":
        assert _real_gpu_driver() is None
    w, cpath = setup_world(tmp)
    w.pause_train_once = set()
    cfg = L.load_config(cpath)
    cfg["score_first_run"] = False
    yaml.safe_dump(cfg, open(cpath, "w", encoding="utf-8"), sort_keys=False)
    run_dir = os.path.join(tmp, "results", "loop")
    args = ["--config", cpath, "--run-dir", run_dir, "--max-rounds", "1"]
    state = lambda: json.load(open(os.path.join(run_dir, "state.json")))  # noqa: E731
    status = lambda: json.load(open(os.path.join(run_dir, "status.json")))  # noqa: E731
    # the legacy driver: recorded, nothing new in the no-worse checks' fingerprint, EvalPlus pairs with the first run
    assert run_main(args) == 0, status()
    assert state()["gate_server"]["gpu_driver"] == legacy
    assert "gpu_driver" not in state()["base_guards"]["bfcl"]["fingerprint"] and "evalplus_base" not in state()
    assert not [name for name, _ in w.served if name.endswith("-evalplus-base")]
    ms = json.load(open(os.path.join(run_dir, "rounds", "r002", "stages", "milestone.done.json")))
    assert ms["guard"]["n"] and "missing" not in ms["guard"], ms
    # a state recorded before the driver was is read as the legacy driver: no wait, and the driver is recorded
    st = state()
    st["gate_server"].pop("gpu_driver")
    stub = types.SimpleNamespace(state=st, dir=run_dir, save=lambda: None)
    L.Loop.require_gate_server(stub, L.load_config(cpath))
    assert stub.state["gate_server"]["gpu_driver"] == legacy
    # a new driver: waits, naming both drivers, and starts no server
    L.llama.gpu_driver = lambda: new
    n = len(w.served)
    assert run_main(args) == 0
    det = status()["detail"]
    assert status()["state"] == "waiting" and len(w.served) == n, status()
    assert f"graphics driver {new}" in det and f"graphics driver {legacy}" in det, det
    assert "settings and the graphics driver back" in det, det
    # answers set aside (a new run folder): every check is answered on the new driver, EvalPlus's base included
    run_dir = os.path.join(tmp, "results", "loop-driver")
    args = ["--config", cpath, "--run-dir", run_dir, "--max-rounds", "1"]
    assert run_main(args) == 0, status()
    assert state()["gate_server"]["gpu_driver"] == new
    assert state()["base_guards"]["bfcl"]["fingerprint"]["gpu_driver"] == new
    eb = state()["evalplus_base"]
    assert eb["gpu_driver"] == new and os.path.exists(eb["result"]), eb
    assert eb["result"] == os.path.join(run_dir, "rounds", "r000", "evalplus-base", new, "result.json"), eb
    served = w.served[n:]
    base = [f for name, f in served if name.endswith("-evalplus-base")]
    adapter = [f for name, f in served if name.endswith("-evalplus")]
    assert len(base) == 1 and "--lora" not in base[0], base
    assert adapter and all("--lora" in f for f in adapter), adapter
    assert base[0] == [x for x in adapter[0] if x not in ("--lora", adapter[0][adapter[0].index("--lora") + 1])]
    ms = json.load(open(os.path.join(run_dir, "rounds", "r002", "stages", "milestone.done.json")))
    assert ms["guard"]["n"] and "missing" not in ms["guard"], ms
    # reused while the driver stays: nothing answered again
    stub = types.SimpleNamespace(state=state(), dir=run_dir, rnum=3, read_json=lambda p: json.load(open(p)),
                                 evalplus_pass=lambda *a: (_ for _ in ()).throw(AssertionError("answered again")))
    res, path = L.Loop.evalplus_base(stub, L.load_config(cpath))
    assert path == eb["result"] and res["datasets"], (path, res)
    # scored rows keep the reasoning text, whole
    rows = E.read_jsonl(os.path.join(run_dir, "rounds", "r000", "gate.seed1.jsonl"))
    assert rows and all(r.get("reasoning") for r in rows), rows[:2]
    assert E.REASONING_CHARS >= 4 * 40960 and E.ANSWER_CHARS >= 4 * 40960


def test_gate_speed_bench():
    """tests/gate_speed_bench.py (speed-up item 2): n_decoded found in either /slots form, the window count per slot
    and task (a task finished inside the window keeps its last count, a task new inside it counts from 0), and a whole
    run against a fake server: the gate flags with MTP, one request in flight per slot, the JSON it writes."""
    import gate_speed_bench as B
    assert B.find_decoded({"id": 0, "next_token": {"n_decoded": 5}}) == 5
    assert B.find_decoded({"id": 0, "next_token": [{"n_remain": 3, "n_decoded": 7}]}) == 7
    assert B.find_decoded({"id": 0, "next_token": []}) is None
    s = [(0.0, [(0, 1, True, 100), (1, 2, True, 50)]), (5.0, [(0, 1, True, 130), (1, 2, True, 80)]),
         (10.0, [(0, 1, False, 150), (1, 2, True, 110)]), (15.0, [(0, 3, True, 20), (1, 2, True, 140)]),
         (16.0, [(0, 3, False, 0), (1, 2, True, 150)])]
    assert B.window_tokens(s, 0.0, 15.0) == (50 + 90 + 20, 1, 15.0)
    assert B.window_tokens(s, 0.0, 16.0) == (50 + 100 + 20, 2, 16.0)  # task 3's idle slot reads 0: its 20 stay
    tmp = tempfile.mkdtemp(prefix="anneal-bench-test-")
    saved = (B.llama.Server, B.llama.http, B.llama.server_binary, B.ew.chat_request, E.GATE_PATH, B.OUT_DIR, sys.argv)
    calls, lock = [], threading.Lock()
    start = time.monotonic()

    class FakeServer:
        def __init__(self, flags, log_path, port=0, binary=None, **k):
            self.flags, self.log_path, self.binary, self.url, self.load_s = flags, log_path, binary, "http://fake", 1.0
            self.up = True

        def describe(self):
            return " ".join([self.binary] + self.flags)

        def start(self):
            return self

        def alive(self):
            return self.up

        def stop(self):
            self.up = False

    def http(url, body=None, timeout=600):
        if url.endswith("/health"):
            raise ConnectionRefusedError(111, "refused")
        n = int((time.monotonic() - start) * 1000)  # 1,000 tokens a second per slot
        return [{"id": i, "id_task": 7 + i, "is_processing": True, "next_token": [{"n_decoded": n}]} for i in range(4)]

    def chat(url, message, seed, max_tokens):
        with lock:
            calls.append((message, seed, max_tokens))
        while bench.up:
            time.sleep(0.005)
        raise ConnectionResetError(10054, "closed")
    bench = types.SimpleNamespace(up=True)
    try:
        gate = os.path.join(tmp, "gate.jsonl")
        E.write_jsonl(gate, [{"id": f"write/go/{i}", "prompt": f"p{i}"} for i in range(10)])
        exe = os.path.join(tmp, "llama-server.exe")
        open(exe, "wb").close()
        E.GATE_PATH, B.OUT_DIR, B.ew.chat_request = gate, os.path.join(tmp, "out"), chat
        B.llama.http, B.llama.server_binary = http, lambda build="b11435-vulkan": exe

        def server(*a, **k):
            srv = FakeServer(*a, **k)
            bench.up = True
            srv.stop = lambda: setattr(bench, "up", False) or setattr(srv, "up", False)
            return srv
        B.llama.Server = server
        sys.argv = ["gate_speed_bench.py", "--slots", "4", "--mtp", "1", "--build", "b11796-vulkan", "--warmup",
                    "0.002", "--minutes", "0.005", "--sample-s", "0.02", "--label", "t"]
        assert B.main() == 0
        res = json.load(open(os.path.join(tmp, "out", "t.json")))
        assert res["slots"] == 4 and res["mtp_draft_n_max"] == 1 and res["build"] == "b11796-vulkan", res
        assert res["flags"][-4:] == ["--spec-type", "draft-mtp", "--spec-draft-n-max", "1"] and "-np" in res["flags"]
        assert res["flags"][res["flags"].index("-np") + 1] == "4" and res["busy_slots_min"] == 4 and not res["errors"]
        assert 3000 < res["tokens_per_s"] < 5000, res["tokens_per_s"]  # 4 slots x 1,000 a second
        gate_rows = E.read_jsonl(gate)
        order = E.shuffled_order(gate_rows, "gate-speed-bench")
        mt = int(L.load_config(L.CONFIG)["gate"]["max_tokens"])  # loop.yaml's answer length
        assert sorted(calls) == sorted((gate_rows[i]["prompt"], 100000 + i, mt) for i in order[:4]), calls
    finally:
        (B.llama.Server, B.llama.http, B.llama.server_binary, B.ew.chat_request, E.GATE_PATH, B.OUT_DIR,
         sys.argv) = saved
        shutil.rmtree(tmp, ignore_errors=True)


def test_budget_message():
    """22 Oct 2026: loop.yaml gate.budget_message adds a stop-thinking sentence at the thinking budget
    (--reasoning-budget-message) on every gate-settings server, not on EvalPlus's. The thinking budget, the tokens an
    answer may take and the sentence are recorded with the other gate server settings: rounds scored without the
    sentence, or with another budget, wait on a change until their answers are set aside. tests/budget_message_check.py
    answers chosen gate items again with the sentence, against a fake server."""
    tmp = tempfile.mkdtemp(prefix="anneal-loop-test-")
    before = world_patches()
    try:
        _budget_message(tmp)
    finally:
        restore_world(before)
        restore_polyglot()
        shutil.rmtree(tmp, ignore_errors=True)


def _budget_message(tmp):
    import yaml
    import budget_message_check as B
    msg = B.QWEN_STOP
    assert msg.startswith("\n\nConsidering the limited time") and msg.endswith("directly now.\n")
    assert L.gate_budget_message({"gate": {}}) is None and L.gate_budget_message({"gate": {"budget_message": ""}}) is None
    assert L.gate_budget_message({"gate": {"budget_message": msg}}) == msg
    for bad in (3, "   ", "a" * 1001, "a\0b", ["x"]):
        try:
            L.gate_budget_message({"gate": {"budget_message": bad}})
            raise AssertionError(f"message {bad!r} accepted")
        except ValueError:
            pass
    w, cpath = setup_world(tmp)
    w.pause_train_once = set()
    cfg = L.load_config(cpath)
    cfg["score_first_run"] = False
    for k in ("build", "spec", "budget_message"):  # whatever loop.yaml has now
        cfg["gate"].pop(k, None)
    cfg["gate"]["budget"], cfg["gate"]["max_tokens"] = 16384, 20480
    cfg["noncoding"]["max_tokens"] = 20480
    assert "--reasoning-budget-message" not in L.Loop.gate_flags(None, cfg)

    def write(c):
        yaml.safe_dump(c, open(cpath, "w", encoding="utf-8"), sort_keys=False)
    write(cfg)
    run_dir = os.path.join(tmp, "results", "loop")
    args = ["--config", cpath, "--run-dir", run_dir, "--max-rounds", "1"]
    state = lambda: json.load(open(os.path.join(run_dir, "state.json")))  # noqa: E731
    status = lambda: json.load(open(os.path.join(run_dir, "status.json")))  # noqa: E731
    assert run_main(args) == 0, status()
    gs = state()["gate_server"]
    assert (gs["budget"], gs["max_tokens"], gs["budget_message"]) == (16384, 20480, None), gs
    # the sentence on rounds scored without it: waits, naming both, and starts no server
    cfg["gate"]["budget_message"] = msg
    write(cfg)
    assert L.load_config(cpath)["gate"]["budget_message"] == msg  # real newlines survive loop.yaml
    n = len(w.served)
    assert run_main(args) == 0
    det = status()["detail"]
    assert status()["state"] == "waiting" and len(w.served) == n, status()
    assert "with the budget message 'Considering the limited time" in det and "with no budget message" in det, det
    # so does another thinking budget
    write(dict(cfg, gate=dict(cfg["gate"], budget_message=None, budget=32768, max_tokens=36864)))
    assert run_main(args) == 0
    det = status()["detail"]
    assert status()["state"] == "waiting" and len(w.served) == n, status()
    assert "a 32768-token thinking budget, 36864 tokens an answer" in det, det
    assert "a 16384-token thinking budget, 20480 tokens an answer" in det, det
    # and a longer non-coding answer (it shares the server's thinking budget)
    write(dict(cfg, gate=dict(cfg["gate"], budget_message=None), noncoding=dict(cfg["noncoding"], max_tokens=40960)))
    assert run_main(args) == 0
    det = status()["detail"]
    assert status()["state"] == "waiting" and len(w.served) == n, status()
    assert "with 40960 tokens a non-coding answer" in det and "with 20480 tokens a non-coding answer" in det, det
    # answers set aside (a new run folder): every gate-settings server has the sentence right after the budget;
    # EvalPlus doesn't
    write(cfg)
    run_dir = os.path.join(tmp, "results", "loop-message")
    args = ["--config", cpath, "--run-dir", run_dir, "--max-rounds", "1"]
    assert run_main(args) == 0, status()
    assert state()["gate_server"]["budget_message"] == msg
    new = w.served[n:]
    gate = [(name, f) for name, f in new if not name.endswith("-evalplus")]
    assert "evaluate" in {name.split("-", 1)[1] for name, _ in gate}, gate
    for name, f in gate:
        i = f.index("--reasoning-budget")
        assert f[i + 1] == "16384" and f[i + 2:i + 4] == ["--reasoning-budget-message", msg], (name, f)
    evalplus = [f for name, f in new if name.endswith("-evalplus")]
    assert evalplus and all("--reasoning-budget-message" not in f for f in evalplus), evalplus
    # a bigger answer limit must leave room for the no-worse checks' chats too: their JSON is counted, plus a margin
    # for the tool-calling instructions when they have tools (here 4 characters a token)
    logs = []
    stub = types.SimpleNamespace(log=logs.append)
    chat = {"messages": [{"role": "user", "content": "x" * 4000}], "tools": [{"name": "f"}]}
    assert L.room_text(chat).startswith('{"messages": [{"role": "user"') and L.room_text({"prompt": "p"}) == "p"
    need = len(L.room_text(chat)) // 4 + L.PROMPT_MARGIN + L.TOOLS_MARGIN + 40448
    L.Loop.slot_room_check(stub, "u", {"gate": {"slots": 6}}, [("BFCL", [chat], 40448)])
    assert f"needs {need} of the 43690" in logs[-1], logs
    try:
        L.Loop.slot_room_check(stub, "u", {"gate": {"slots": 6}}, [("BFCL", [chat], 43690 - need + 40448 + 1)])
        raise AssertionError("a chat one token too long fitted")
    except L.Waiting as exc:
        assert "longest BFCL prompt" in str(exc), exc
    del chat["tools"]  # no tools (CWEval): no margin for them
    L.Loop.slot_room_check(stub, "u", {"gate": {"slots": 6}}, [("CWEval", [chat], 40448)])
    assert f"needs {len(L.room_text(chat)) // 4 + L.PROMPT_MARGIN + 40448} of the 43690" in logs[-1], logs
    # the check script: the rows that ran out, answered again with their gate seeds and the sentence
    saved = (B.llama.Server, B.llama.http, B.llama.server_binary, B.ew.chat_request, B.ew.count_tokens, B.OUT_DIR,
             B.make_check, B.sandbox, sys.argv)
    calls, lock = [], threading.Lock()
    gate_items = E.read_jsonl(E.GATE_PATH)

    class FakeServer:
        def __init__(self, flags, log_path, port=0, binary=None, **k):
            self.flags, self.binary, self.url, self.load_s = flags, binary, "http://fake", 1.0

        def describe(self):
            return " ".join([self.binary] + self.flags)

        def start(self):
            return self

        def stop(self):
            pass
    flags_seen = []

    def server(*a, **k):
        srv = FakeServer(*a, **k)
        flags_seen.append(srv.flags)
        return srv

    def http(url, body=None, timeout=600):
        raise ConnectionRefusedError(111, "refused")  # nothing on the loop's port

    def chat(url, message, seed, max_tokens):
        with lock:
            calls.append((message, seed, max_tokens))
        if message == gate_items[3]["prompt"]:  # thinks to the budget, gets the sentence, writes code
            return {"content": "Here:\n```go\npackage main\n```\n", "reasoning": "x " * 50 + msg.rstrip(),
                    "finish_reason": "stop", "completion_tokens": 17000, "seconds": 1.0}
        return {"content": "Still: so the answer", "reasoning": "y", "finish_reason": "length",
                "completion_tokens": 20480, "seconds": 1.0}
    try:
        exe = os.path.join(tmp, "llama-server.exe")
        open(exe, "wb").close()
        B.llama.Server, B.llama.http, B.llama.server_binary = server, http, lambda build="b11435-vulkan": exe
        B.ew.chat_request, B.ew.count_tokens = chat, lambda url, text: 16384 if text.startswith("x") else 10
        B.OUT_DIR, B.make_check = os.path.join(tmp, "bm"), lambda: (lambda item, content: (True, "ok"))
        B.sandbox = types.SimpleNamespace(ensure_private_folder_closed=lambda log: False)
        rows = os.path.join(tmp, "gate.seed1.jsonl")
        E.write_jsonl(rows, [{"id": gate_items[3]["id"], "finish_reason": "length"},
                             {"id": gate_items[5]["id"], "finish_reason": "stop"},
                             {"id": gate_items[8]["id"], "finish_reason": "length"}])
        sys.argv = ["budget_message_check.py", "--config", cpath, "--rows", rows, "--label", "t"]
        assert B.main() == 0
        res = json.load(open(os.path.join(tmp, "bm", "t.json")))
        f = res["flags"]
        assert f[f.index("--reasoning-budget-message") + 1] == msg and res["message"] == msg, f
        assert sorted(calls) == sorted((gate_items[i]["prompt"], 100000 + i, 20480) for i in (3, 8)), calls
        assert (res["items"], res["budget_hits"], res["budget_hits_with_code_block"], res["budget_hits_sentence_seen"],
                res["ran_out"], res["passed"]) == (2, 1, 1, 1, 1, 1), res
        a = {x["id"]: x for x in res["answers"]}
        assert a[gate_items[8]["id"]]["fences"] == 0 and not a[gate_items[8]["id"]]["code_block"]
        assert len(E.read_jsonl(os.path.join(tmp, "bm", "t.jsonl"))) == 2
        # the control: no sentence, chosen ids
        sys.argv = ["budget_message_check.py", "--config", cpath, "--ids", gate_items[3]["id"], "--no-message",
                    "--label", "c"]
        assert B.main() == 0
        res = json.load(open(os.path.join(tmp, "bm", "c.json")))
        assert res["message"] is None and "--reasoning-budget-message" not in res["flags"], res["flags"]
        assert res["budget_hits_sentence_seen"] == 0 and res["budget_hits_with_code_block"] == 1, res
    finally:
        (B.llama.Server, B.llama.http, B.llama.server_binary, B.ew.chat_request, B.ew.count_tokens, B.OUT_DIR,
         B.make_check, B.sandbox, sys.argv) = saved


def test_second_early_look():
    """21 Oct 2026: a second early look at two thirds of the gate. A round ahead of the best on the first third but not
    on the first two thirds stops at the second look, rejected, with no more gate answers and no non-coding check."""
    tmp = tempfile.mkdtemp(prefix="anneal-loop-test-")
    saved = dict(PASSING)
    before = world_patches()
    try:
        import yaml
        w, cpath = setup_world(tmp)
        w.pause_train_once = set()
        cfg = L.load_config(cpath)
        cfg["score_first_run"] = False
        cfg["gate"]["early_reject"] = {"looks": [0.3334, 0.6667], "max_diff": 0}
        yaml.safe_dump(cfg, open(cpath, "w", encoding="utf-8"), sort_keys=False)
        run_dir = os.path.join(tmp, "results", "loop")
        rd2 = os.path.join(run_dir, "rounds", "r002")
        done = lambda stage: json.load(open(os.path.join(rd2, "stages", f"{stage}.done.json")))  # noqa: E731
        gate = E.read_jsonl(E.GATE_PATH)
        order = E.shuffled_order(gate, "early-reject:r2")
        first, second = order[:14], order[14:27]
        base = set(range(PASSING[None]))  # the base passes gate items 0..19
        gain = [i for i in first if i not in base][:1]
        drop = [i for i in second if i in base][:2]
        assert len(gain) == 1 and len(drop) == 2, (first, second)
        # r002: one more than the base on the first third, then two fewer on the second: +1, then -1 overall
        PASSING["r002"] = (base | set(gain)) - set(drop)
        assert run_main(["--config", cpath, "--run-dir", run_dir, "--max-rounds", "1"]) == 0
        ev, d = done("evaluate"), done("decide")
        assert [(c["n"], c["diff"], c["stop"]) for c in ev["early_checks"]] == [(14, 1, False), (27, -1, True)], ev
        assert ev["early_reject"] is True and ev["early_check"]["n"] == 27 and ev["gate_n"] == 27, ev
        rows = E.read_jsonl(os.path.join(rd2, "gate.seed1.jsonl"))
        assert sorted(r["id"] for r in rows) == sorted(gate[i]["id"] for i in order[:27])
        assert not E.read_jsonl(os.path.join(rd2, "noncoding.jsonl"))
        assert not d["keep"] and d["reasons"][0].startswith("stopped early: -1 vs the best after 27 of 40"), d
        st = json.load(open(os.path.join(run_dir, "state.json")))
        assert [(r["round"], r["status"]) for r in st["rounds"]] == [(0, "reference"), (2, "rejected")], st["rounds"]
    finally:
        PASSING.clear()
        PASSING.update(saved)
        restore_world(before)
        restore_polyglot()
        shutil.rmtree(tmp, ignore_errors=True)


def test_early_reject():
    """The early look (loop.yaml gate early_reject): a round behind the best on the shuffled first third of the gate
    stops there, rejected, with no more gate answers and no non-coding check; a round ahead there answers the whole
    gate and can be kept; a round whose bug-fix loss path is open answers the whole gate. Every gate request keeps
    the seed of its place in gate.jsonl, whatever the order. With score_first_run off, training starts at round 2."""
    tmp = tempfile.mkdtemp(prefix="anneal-loop-test-")
    saved = dict(PASSING), dict(SWE_LOSS), dict(TERM_LOSS)
    before = world_patches()
    try:
        _early_reject(tmp)
    finally:
        PASSING.clear()
        PASSING.update(saved[0])
        SWE_LOSS.clear()
        SWE_LOSS.update(saved[1])
        TERM_LOSS.clear()
        TERM_LOSS.update(saved[2])
        restore_world(before)
        restore_polyglot()
        shutil.rmtree(tmp, ignore_errors=True)


def _early_reject(tmp):
    import yaml
    w, cpath = setup_world(tmp)
    w.pause_train_once = set()
    cfg = L.load_config(cpath)
    cfg["score_first_run"] = False
    cfg["gate"]["early_reject"] = {"looks": [0.6667, 0.3334], "max_diff": 0}
    yaml.safe_dump(cfg, open(cpath, "w", encoding="utf-8"), sort_keys=False)
    run_dir = os.path.join(tmp, "results", "loop")
    args = ["--config", cpath, "--run-dir", run_dir, "--max-rounds", "1"]
    rd = lambda n: os.path.join(run_dir, "rounds", f"r{n:03d}")  # noqa: E731
    done = lambda n, stage: json.load(open(os.path.join(rd(n), "stages", f"{stage}.done.json")))  # noqa: E731
    state = lambda: json.load(open(os.path.join(run_dir, "state.json")))  # noqa: E731
    gate = E.read_jsonl(E.GATE_PATH)
    assert L.early_looks(cfg["gate"]["early_reject"]) == [0.3334, 0.6667]
    assert L.early_looks({"fraction": 0.3334, "max_diff": 0}) == [0.3334]  # the older single-look form
    asked = []

    def chat(url, message, seed, max_tokens=None, sampling=None):
        asked.append((w.model(), message, seed))
        return w.chat(url, message, seed, max_tokens, sampling)
    ew.chat_request = chat
    # r002 is behind the base (18 vs 20 of 40) and its bug-fix loss is higher: it stops after the first third
    PASSING["r002"] = 18
    assert run_main(args) == 0, json.load(open(os.path.join(run_dir, "status.json")))
    assert [(r["round"], r["status"]) for r in state()["rounds"]] == [(0, "reference"), (2, "rejected")], state()
    assert not os.path.exists(rd(1))
    k = math.ceil(len(gate) * 0.3334)
    first = [gate[i]["id"] for i in E.shuffled_order(gate, "early-reject:r2")[:k]]
    rows = E.read_jsonl(os.path.join(rd(2), "gate.seed1.jsonl"))
    assert k == 14 and sorted(r["id"] for r in rows) == sorted(first), rows
    assert not E.read_jsonl(os.path.join(rd(2), "noncoding.jsonl"))
    ev2, d2 = done(2, "evaluate"), done(2, "decide")
    assert ev2["early_reject"] is True and ev2["early_check"]["stop"] and ev2["early_check"]["n"] == 14, ev2
    assert len(ev2["early_checks"]) == 1  # stopped at the first look
    assert ev2["early_check"]["diff"] <= 0 and ev2["gate_n"] == 14
    assert not d2["keep"] and d2["early_reject"]["stop"] and d2["reasons"][0].startswith("stopped early"), d2
    assert "stopped early" in open(os.path.join(rd(2), "report.md")).read()
    # r003 is ahead (40 of 40): both looks show it, so it answers the whole gate and the non-coding check
    PASSING["r003"] = 40
    assert run_main(args) == 0
    ev3, d3 = done(3, "evaluate"), done(3, "decide")
    assert ev3["early_reject"] is False and ev3["early_check"]["diff"] > 0 and ev3["gate_n"] == len(gate), ev3
    assert [c["n"] for c in ev3["early_checks"]] == [14, 27] and not any(c["stop"] for c in ev3["early_checks"])
    assert E.read_jsonl(os.path.join(rd(3), "noncoding.jsonl")) and d3["keep"] and d3["path"] == "gate", d3
    assert state()["best"]["round"] == 3
    # r004 is behind r003 but its bug-fix loss falls: the loss path is open, so there is no early look
    PASSING["r004"], SWE_LOSS["r004"] = 18, 0.5
    assert run_main(args) == 0
    ev4, d4 = done(4, "evaluate"), done(4, "decide")
    assert "early_check" not in ev4 and ev4["gate_n"] == len(gate) and not d4["keep"], ev4
    assert E.loss_path_open(cfg["rules"], d4["loss"])
    # r005: behind, the bug-fix loss higher, but the terminal loss clearly lower: no early look either
    PASSING["r005"], TERM_LOSS["r005"] = 18, 0.5
    assert run_main(args) == 0
    ev5, d5 = done(5, "evaluate"), done(5, "decide")
    assert "early_check" not in ev5 and ev5["gate_n"] == len(gate) and not d5["keep"], ev5
    assert E.open_loss_paths(cfg["rules"], d5["losses"]) == ["terminal"], d5["losses"]
    # every gate request, in any order and any round, used the seed of the item's place in gate.jsonl
    idx = {g["prompt"]: i for i, g in enumerate(gate)}
    gate_asks = [(m, s) for _, m, s in asked if m in idx]
    assert gate_asks and all(s % 100000 == idx[m] for m, s in gate_asks), gate_asks[:5]
    # the scorer on its own: an order answers just those items, each with its own seed, and resumes
    with tempfile.TemporaryDirectory() as d:
        items = [{"id": f"x/{i}", "prompt": f"p{i}"} for i in range(6)]
        got = []
        chat2 = lambda url, message, seed, max_tokens: (got.append((message, seed)) or  # noqa: E731
                                                        {"content": "ok", "reasoning": "", "finish_reason": "stop",
                                                         "completion_tokens": 1, "seconds": 0.0})
        out = os.path.join(d, "s.jsonl")
        E.score("u", items, out, 2, 10, lambda it, c: True, chat2, lambda u, t: 0, 0, concurrency=2, order=[4, 1],
                log=lambda m: None)
        assert sorted(got) == [("p1", 200001), ("p4", 200004)], got
        E.score("u", items, out, 2, 10, lambda it, c: True, chat2, lambda u, t: 0, 0, concurrency=2,
                log=lambda m: None)
        assert len(got) == 6 and len(E.read_jsonl(out)) == 6 and ("p0", 200000) in got
    # the look itself
    best = [{"id": f"x/{i}", "seed": 1, "pass": i < 3} for i in range(6)]
    cand = [{"id": f"x/{i}", "seed": 1, "pass": i in (0, 1, 4)} for i in range(6)]
    c = E.early_reject_check(best, cand, ["x/2", "x/4", "x/5"], 0)
    assert (c["n"], c["diff"], c["wins"], c["losses"], c["stop"]) == (3, 0, 1, 1, True), c
    assert not E.early_reject_check(best, cand, ["x/4"], 0)["stop"]
    assert not E.early_reject_check(best, cand[:4], ["x/2", "x/4"], 0)["stop"]  # x/4 not answered yet: no look


def test_private_folder_guard():
    """sandbox.ensure_private_folder_closed (its source, run here with stand-ins, since the module needs Windows):
    nothing to do when closed; closes an open folder; raises if it stays open."""
    import ast
    src = open(os.path.join(ROOT, "anneal", "sandbox.py"), encoding="utf-8").read()
    fn = next(n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == "ensure_private_folder_closed")

    class SandboxUnavailable(RuntimeError):
        pass
    for seq, want in (([False], False), ([True, False], True), ([True, True], "raise")):
        it, closed = iter(seq), []
        ns = {"private_folder_writable": lambda timeout_s=30.0: next(it), "SandboxUnavailable": SandboxUnavailable,
              "close_private_folder": lambda: closed.append(1), "profile_folder": lambda: "AC"}
        exec(compile(ast.Module(body=[fn], type_ignores=[]), "sandbox.py", "exec"), ns)
        try:
            got = ns["ensure_private_folder_closed"](log=lambda *a: None)
        except SandboxUnavailable:
            got = "raise"
        assert got == want and len(closed) == (0 if want is False else 1), (seq, got, closed)


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
    # pools are built at the training row cap, so no pooled row is too long to train on
    cap = int(L.load_config()["train"]["seq_len"])
    seen, live = set(), set()
    for script, cmd in calls:
        assert cmd[1] == "-u", cmd
        seen.add(script)
        if script == "convert_lora_to_gguf.py":  # llama.cpp's script: same command as the first run's convert stage
            assert cmd[3:9] == ["--base", L.llama.LORA_BASE_CONFIG, "--outtype", "f16", "--outfile", cmd[8]], cmd
            continue
        ns = parsed_args(script, cmd[3:])
        if script == "public_source.py":
            assert ns.tokenizer and ns.decontaminate and ns.decontaminate_gate and ns.max_tokens == cap, ns
        if script == "prepare_dataset.py":
            assert ns.tokenizer and ns.decontaminate and ns.decontaminate_gate and ns.max_tokens == cap, ns
        # the live mix (21 Oct): V4-Flash bug-fix steps outside Python; V4-Pro answers that don't write Python
        if script == "public_source.py" and ns.out.endswith("openswe_v4flash_12k"):
            assert ns.where == ["resolved=1", "language!=python"] and ns.report_field == ["language"], ns
            assert ns.turns == "each" and ns.require_think and ns.revision, ns
            # repositories in bucket 0 of 20 never train: they are the held-out bug-fix steps
            assert (ns.holdout_col, ns.holdout_mod, ns.holdout_side) == ("repo", 20, "pool"), ns
            live.add(ns.out)
        if script == "val_loss.py" and ns.val_file.endswith("swe_heldout.jsonl"):
            assert ns.seq_len == 8192 and ns.max_rows == 200, ns
        if script == "val_loss.py" and ns.val_file.endswith("term_heldout.jsonl"):
            assert ns.seq_len == 12288 and ns.max_rows == 200, ns
            live.add("termloss")
        if script == "prepare_dataset.py" and ns.out.endswith("_nopy_12k"):
            assert ns.drop_answer_lang == ["python"] and ns.require_think, ns
            live.add(ns.out)
        if script == "train_longrun.py":
            assert ns.config.endswith("train.yaml"), ns
    assert seen == {"prepare_dataset.py", "public_source.py", "mix_datasets.py", "train_longrun.py",
                    "convert_lora_to_gguf.py", "val_loss.py"}, seen
    assert len(live) == 4, live


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
            dict(ok, messages=[{"role": "user", "content": "An empty think block?"},
                               {"role": "assistant", "content": "<think>\n\n</think>\n\nYes."}]),
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
                c["duplicate"], c["too_long"], c["contaminated_gate"], c["kept"]) == (12, 1, 1, 1, 2, 1, 1, 1, 4), c
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
        # Python lowest priority: a negated row filter, answers that write Python dropped, kept rows by language
        def fix(lang, code_lang, q):
            return {"language": lang, "messages": [
                {"role": "user", "content": q},
                {"role": "assistant", "content": f"Fixed:\n```{code_lang}\nreturn total\n```", "reasoning_content": "Off by one."}]}
        src5 = os.path.join(d, "langs.jsonl")
        E.write_jsonl(src5, [fix("python", "python", "Fix the Python sum bug."), fix("go", "go", "Fix the Go sum bug."),
                             fix("rust", "rust", "Fix the Rust sum bug."), fix("go", "py", "Fix the Go build script bug.")])
        out8 = os.path.join(d, "out8")
        assert run_public_source(["--dataset", "t/swe", "--files", "*", "--from-local", src5, "--where", "language!=python",
                                  "--report-field", "language", "--drop-answer-lang", "python", "--require-think",
                                  "--tokenizer", "x", "--measure-only", "--out", out8]) == 0
        m8 = json.load(open(os.path.join(out8, "measure.json")))
        assert (m8["counts"]["filtered_where"], m8["counts"]["answer_lang"], m8["counts"]["kept"]) == (1, 1, 2), m8
        assert m8["report_fields"] == {"language": {"read": {"go": 2, "rust": 1}, "kept": {"go": 1, "rust": 1}}}, m8
        assert sorted(m8["kept_answer_languages"]) == [["go", 1], ["rust", 1]], m8


def test_ranked_runs_and_near_duplicates():
    """GLM-5.3 runs (21 Oct 2026: every passing run, best 2 per task first): --rank-col reads the best-ranked runs
    first, --dedupe-actions-by drops a step whose commands up to it repeat a kept step of the same task (wording and
    reasoning aside), --first-ranks puts the best runs' rows first in train.jsonl, and --max-conversations-per keeps
    one run per task group for a held-out set."""
    import public_source as ps

    def reply(cmds, done=False):
        return json.dumps({"analysis": "Looking at the state.", "plan": "Next step.", "task_complete": done,
                           "commands": [{"keystrokes": c, "duration": 0.5} for c in cmds]})

    def run(task, group, rank, steps, note=""):
        msgs = [{"role": "system", "content": "You are an agent in a Linux terminal."}]
        for k, (cmds, done) in enumerate(steps):
            msgs.append({"role": "user", "content": f"Task {task}: set up the service. Terminal output {k}. {note}"})
            msgs.append({"role": "assistant", "content": reply(cmds, done),
                         "reasoning_content": f"Reasoning of {task} rank {rank} step {k}, worded my own way."})
        return {"messages": msgs, "task_id": task, "task_group_id": group, "task_rank": rank, "picked": rank <= 2,
                "uses_python": False}
    a1 = [(["ls -la\n"], False), (["cat app.conf\n"], False), (["systemctl restart app\n"], True)]
    a2 = [(["ls  -la\n"], False), (["cat /etc/app.conf\n"], False), (["systemctl restart app\n"], True)]
    c3 = [(["make build\n"], False), (["make test\n"], True)]
    rows = [run("A", "g1", 3, a1),   # the same commands as A's best run, worded differently: all near-duplicates
            run("C", "g2", 3, c3),   # a third-best run with its own commands: kept, but after the best runs
            run("A", "g1", 2, a2),   # same first command (whitespace aside) as rank 1, then its own path
            run("B", "g1", 1, [(["docker ps\n"], False), (["docker logs app\n"], True)],
                note='A log line quoting \\"task_rank\\": 0 and "task_rank": 0 inside text.'),
            run("A", "g1", 1, a1),
            run("C", "g2", 1, [(["make\n"], True)])]
    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(d, "conversations.jsonl")
        E.write_jsonl(src, rows)
        lines = open(src, "rb").read().splitlines()
        assert [ps.rank_of(x, "task_rank") for x in lines] == [3, 3, 2, 1, 1, 1]
        assert ps.rank_of(b'{"messages": [], "task_id": "X"}', "task_rank") == float("inf")
        common = ["--dataset", "t/glm", "--files", "*", "--from-local", src, "--turns", "each", "--require-think",
                  "--respect-loss-mask", "--tokenizer", "x"]
        out = os.path.join(d, "pool")
        assert run_public_source(common + ["--rank-col", "task_rank", "--first-ranks", "2", "--dedupe-actions-by",
                                           "task_id", "--val-fraction", "0.01", "--out", out]) == 0
        m = json.load(open(os.path.join(out, "measure.json")))
        # A: rank 1 keeps 3; rank 2 drops its first step (same "ls -la") and keeps 2; rank 3 drops all 3.
        # B keeps 2, C rank 1 keeps 1, C rank 3 keeps 2 (its own commands).
        assert (m["counts"]["kept"], m["counts"]["near_duplicate"], m["counts"]["duplicate"]) == (10, 4, 0), m["counts"]
        kept = E.read_jsonl(os.path.join(out, "train.jsonl")) + E.read_jsonl(os.path.join(out, "val.jsonl"))
        notes = sorted(r["messages"][-1]["content"].split("Reasoning of ")[1].split(",")[0] for r in kept)
        assert notes == sorted(["A rank 1 step 0", "A rank 1 step 1", "A rank 1 step 2", "A rank 2 step 1",
                                "A rank 2 step 2", "B rank 1 step 0", "B rank 1 step 1", "C rank 1 step 0",
                                "C rank 3 step 0", "C rank 3 step 1"]), notes
        train = E.read_jsonl(os.path.join(out, "train.jsonl"))
        tiers = [int("rank 3" in r["messages"][-1]["content"]) for r in train]
        assert tiers == sorted(tiers) and m["train_order"]["later_rows"] == sum(tiers), (tiers, m["train_order"])
        assert m["train_order"]["first_rows"] + m["train_order"]["later_rows"] == len(train)
        # without the rank order the file's first copy of A's commands (its rank-3 run) would be the one kept
        out2 = os.path.join(d, "fileorder")
        assert run_public_source(common + ["--dedupe-actions-by", "task_id", "--measure-only", "--out", out2]) == 0
        assert json.load(open(os.path.join(out2, "measure.json")))["counts"]["near_duplicate"] == 4
        # one step from one run per task group (held-out builder): g1's best run in rank order, then g2's
        out3 = os.path.join(d, "held")
        assert run_public_source(common + ["--rank-col", "task_rank", "--dedupe-actions-by", "task_id",
                                           "--max-conversations-per", "task_group_id=1",
                                           "--max-steps-per-conversation", "1", "--out", out3]) == 0
        m3 = json.load(open(os.path.join(out3, "measure.json")))
        held = E.read_jsonl(os.path.join(out3, "train.jsonl")) + E.read_jsonl(os.path.join(out3, "val.jsonl"))
        assert (m3["counts"]["kept"], m3["counts"]["per_value_cap"], len(held)) == (2, 4, 2), m3["counts"]
        assert sorted(r["messages"][1]["content"][:6] for r in held) == ["Task B", "Task C"], held
        try:
            run_public_source(common + ["--first-ranks", "2", "--out", os.path.join(d, "bad")])
            raise AssertionError("--first-ranks without --rank-col must fail")
        except SystemExit as e:
            assert "--rank-col" in str(e)


def test_prepare_dataset_filters():
    """--require-think needs a non-empty think block; --drop-answer-lang python drops answers that write Python (a
    Python prompt ported to Go stays); the card counts both and lists the kept rows' languages."""
    parsed_args("prepare_dataset.py", ["--dataset", "x", "--out", "y"])  # stubs `datasets` if it isn't installed
    import prepare_dataset as pdm

    class FakeDS(list):
        column_names = ["prompt", "completion"]

        def shuffle(self, seed=None):
            return self
    rust = "Write a Rust function that sums a slice of integers."
    go = "Port this Python to Go: def s(v): return sum(v)"
    rows = FakeDS([
        {"prompt": rust, "completion": "<think>\nIterate and add.\n</think>\n\n```rust\nfn s(v: &[i32]) -> i32 { v.iter().sum() }\n```"},
        {"prompt": "Write a Python function that sums a list of integers.",
         "completion": "<think>\nUse sum.\n</think>\n\n```python\ndef s(v):\n    return sum(v)\n```"},
        {"prompt": go, "completion": "<think>\nLoop over it.\n</think>\n\n```go\nfunc s(v []int) (t int) { for _, x := range v { t += x }; return }\n```"},
        {"prompt": "Explain what an empty think block teaches here.", "completion": "<think>\n \n</think>\n\nOnly the answer text, nothing more."},
    ])
    saved_load, saved_argv = pdm.load_rows, sys.argv
    pdm.load_rows = lambda *a, **k: rows
    with tempfile.TemporaryDirectory() as d:
        try:
            sys.argv = ["prepare_dataset.py", "--dataset", "t/x", "--prompt-field", "prompt", "--response-field",
                        "completion", "--require-think", "--drop-answer-lang", "python", "--min-chars", "10",
                        "--out", d]
            pdm.main()
        finally:
            pdm.load_rows, sys.argv = saved_load, saved_argv
        kept = E.read_jsonl(os.path.join(d, "train.jsonl")) + E.read_jsonl(os.path.join(d, "val.jsonl"))
        assert {r["messages"][0]["content"] for r in kept} == {rust, go}, kept
        card = open(os.path.join(d, "data_card.md"), encoding="utf-8").read()
        assert "| answer_lang | 1 |" in card and "| no_think | 1 |" in card, card
        assert "| rust | 1 |" in card and "| go | 1 |" in card and "| python |" not in card, card


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
