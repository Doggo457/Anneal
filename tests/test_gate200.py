"""The fixed 200 (25 Oct 2026): its comparison, per-problem judging, the builder, the re-judge and the switch, on
made-up data (no CodeContests download, no sandbox).

  python tests/test_gate200.py      (or: python -m pytest tests/test_gate200.py)
"""

import hashlib
import json
import os
import shutil
import sys
import tempfile
import types

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
if os.name != "nt" and "anneal.sandbox" not in sys.modules:  # the real sandbox needs Windows; loop.py imports it
    _stub = types.ModuleType("anneal.sandbox")
    _stub.require = lambda: None
    _stub.TEST_MEM_BYTES = 0
    sys.modules["anneal.sandbox"] = _stub
    import anneal  # noqa: E402
    anneal.sandbox = _stub
from anneal import evalsets as E  # noqa: E402
from anneal import polyglot as P  # noqa: E402
import build_gate200 as B  # noqa: E402
import rejudge_gate as R  # noqa: E402


def test_gate_match():
    m = P.gate_match
    assert m("1 2\n3\n", "1 2 3")
    assert m("Yes\nno\nYES", "YES\nNO\nYES")  # YES/NO in any case
    assert not m("YESS", "YES") and not m("1", "1 2")
    assert not m("1000000000001", "1000000000000")  # outputs_match's 1e-6 took this
    assert P.outputs_match("1000000000001", "1000000000000")
    assert not m("9007199254740993", "9007199254740992")  # same float, different integer
    assert not m("0.5000001", "0.5")  # no tolerance unless the statement gives one
    assert m("0.50009", "0.5", 1e-4) and not m("0.5002", "0.5", 1e-4)
    assert m("2000.1", "2000", 1e-4)  # relative above 1
    assert not m("nan", "0.5", 1e-4) and not m("inf", "1e308", 1e-4)
    assert not m("1_0", "10", 1e-4)  # float() reads 1_0 as 10; a judge mustn't
    assert m("2", "2.000001", 1e-6) and not m("abc", "abd", 1e-6)


def fake_checkers(fn):
    mod = types.ModuleType("anneal.gate_checkers")
    mod.CHECKERS = {"1000A": fn}
    mod.module_sha256 = lambda: "v1"
    sys.modules["anneal.gate_checkers"] = mod


def test_case_match():
    assert E.case_match({"cases": []}) is None  # the 495: outputs_match
    exact = E.case_match({"judge": {}})
    assert exact("", "no", "NO") and not exact("", "0.5000001", "0.5")
    tol = E.case_match({"judge": {"tol": 1e-4}})
    assert tol("", "0.50004", "0.5")

    def chk(inp, out, exp):  # any two numbers summing to the input
        a, b = map(int, out.split())
        return a + b == int(inp)
    fake_checkers(chk)
    c = E.case_match({"judge": {"checker": "1000A"}})
    assert c("5", "2 3", "1 4") and c("5", "5 0", "1 4") and not c("5", "2 2", "1 4")
    assert not c("5", "garbage", "1 4")  # a checker that raises counts the output wrong
    assert E.case_match({"judge": {"checker": "1000A", "checker_sha256": "v1"}})("5", "2 3", "1 4")
    try:
        E.case_match({"id": "x", "judge": {"checker": "1000A", "checker_sha256": "v0"}})
        raise AssertionError("a changed checker module must stop the judging")
    except RuntimeError as err:
        assert "changed after x" in str(err)


def test_real_checkers():
    """anneal/gate_checkers.py on hand-made cases (loaded under its own name so the fakes above stay in place)."""
    import hashlib
    import importlib.util
    path = os.path.join(ROOT, "anneal", "gate_checkers.py")
    spec = importlib.util.spec_from_file_location("gate_checkers_real", path)
    G = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(G)
    assert G.module_sha256() == hashlib.sha256(open(path, "rb").read()).hexdigest()
    assert len(G.CHECKERS) == 54 and all(callable(f) for f in G.CHECKERS.values())
    assert not {"1552B", "1555B", "1557A", "1560B", "1575B"} & set(G.CHECKERS)  # plain compare, not a checker

    def checker(cf):  # as evalsets.case_match uses it: an exception is a wrong answer
        def f(i, o, e):
            try:
                return bool(G.CHECKERS[cf](i, o, e))
            except Exception:  # noqa: BLE001
                return False
        return f
    c = checker("1549A")  # 2 <= a < b <= P with P mod a = P mod b
    assert c("2\n17\n5\n", "3 5\n2 4\n", "3 5\n2 4\n") and c("2\n17\n5\n", "2 16\r\n2 4", "3 5\n2 4\n")
    assert not c("2\n17\n5\n", "3 4\n2 4", "3 5\n2 4") and not c("2\n17\n5\n", "03 5\n2 4", "3 5\n2 4")
    assert not c("2\n17\n5\n", "3 5\n2 4 1", "3 5\n2 4") and not c("2\n17\n5\n", "3 5", "3 5\n2 4")
    c = checker("1618A")
    assert c("1\n1 3 4 4 5 7 8\n", "4 1 3", "1 3 4") and not c("1\n1 3 4 4 5 7 8\n", "1 3 3", "1 3 4")
    c = checker("1608A")  # increasing, each not a multiple of the one before
    assert c("2\n1\n3\n", "7\n2 3 5", "1\n2 3 4") and not c("2\n1\n3\n", "7\n2 3 6", "1\n2 3 4")
    c = checker("1622F")  # largest subset of 1..n whose factorials multiply to a square
    assert c("9\n", "7\n1 2 3 5 6 7 8", "7\n1 2 4 5 6 7 8") and c("9\n", "7\n8 7 6 5 4 2 1", "7\n1 2 4 5 6 7 8")
    assert not c("9\n", "6\n1 3 4 6 8 9", "7\n1 2 4 5 6 7 8") and not c("9\n", "7\n1 3 4 5 6 7 8", "7\n1 2 4 5 6 7 8")


def test_judge_with_match():
    """polyglot.judge decides each case with the item's matcher (here a real C++ build, Linux only)."""
    if P.WINDOWS or not shutil.which("g++"):
        print("skip: needs Linux g++")
        return
    tc, ex = P.Toolchain.local(), P.LocalExecutor()
    code = '#include <cstdio>\nint main(){long long n; scanf("%lld",&n); puts(n % 2 ? "yes" : "No");}\n'
    cases = [("3", "YES"), ("4", "NO")]
    assert not P.judge("cpp", code, cases, ex, tc).ok  # outputs_match: letter case counts
    v = P.judge("cpp", code, cases, ex, tc, match=E.case_match({"judge": {}}))
    assert v.ok and v.cases == ["ok", "ok"], v
    fake_checkers(lambda i, o, e: o.strip().lower() == ("yes" if int(i) % 2 else "no") and int(i) != 4)
    v = P.judge("cpp", code, cases, ex, tc, match=E.case_match({"judge": {"checker": "1000A"}}), all_cases=True)
    assert not v.ok and v.cases == ["ok", "wrong_answer"], v


def write_cases(d, name, cases):
    raw = json.dumps({"problem": name, "cases": cases}).encode()
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, name + ".json"), "wb") as fh:
        fh.write(raw)
    return hashlib.sha256(raw).hexdigest()


def test_item_cases():
    tmp = tempfile.mkdtemp()
    saved = E.EVAL_DIR
    try:
        E.EVAL_DIR = tmp
        sha = write_cases(os.path.join(tmp, "g", "cases"), "1000A", [["1", "2"], ["3", "4"]])
        it = {"id": "x", "cases_file": "g/cases/1000A.json", "cases_sha256": sha}
        assert E.item_cases(it) == [("1", "2"), ("3", "4")]
        assert E.item_cases({"cases": [["a", "b"]]}) == [["a", "b"]]
        E._CASES.clear()
        try:
            E.item_cases(dict(it, cases_sha256="0" * 64))
            raise AssertionError("a changed test file must not be used")
        except RuntimeError:
            pass
    finally:
        E.EVAL_DIR = saved
        E._CASES.clear()
        shutil.rmtree(tmp)


def cc_row(name, n_pub, n_priv, n_gen, out=lambda i: str(int(i) * 2)):
    t = lambda a, b: {"input": [str(k) for k in range(a, b)], "output": [out(str(k)) for k in range(a, b)]}  # noqa: E731
    return {"name": name, "public_tests": t(0, n_pub), "private_tests": t(n_pub, n_pub + n_priv),
            "generated_tests": t(n_pub + n_priv, n_pub + n_priv + n_gen)}


def gate_item(kind, lang, split, idx, name, row, n=30):
    cases = []
    for f in ("public_tests", "private_tests", "generated_tests"):
        cases += list(zip(row[f]["input"], row[f]["output"]))
    return {"id": f"{kind}/{lang}/cc/{split}/{idx}", "kind": kind, "lang": lang, "problem": f"cc/{split}/{idx}",
            "name": name, "prompt": "p", "cases": [list(c) for c in cases[:n]]}


def write_validators(d, rules):
    os.makedirs(d, exist_ok=True)
    for cf, body in rules.items():
        with open(os.path.join(d, cf + ".py"), "w") as fh:
            fh.write("def validate(t):\n    n = int(t)\n    " + body + "\n")
    return d


def cpp_ref(bad=()):
    cond = " || ".join(f"n == {b}" for b in bad) or "false"
    return '#include <cstdio>\nint main(){long long n; scanf("%lld",&n); printf("%lld\\n", (' + cond + ') ? 0 : 2 * n);}\n'


def test_build():
    rows = {("test", 0): cc_row("1000_A. Doubles", 2, 10, 60), ("test", 1): cc_row("1000_B2. Hard", 1, 5, 40),
            ("valid", 3): cc_row("999_C. Few", 1, 1, 30)}
    gate = [gate_item("write", "cpp", "test", 0, "1000_A. Doubles", rows[("test", 0)]),
            gate_item("fix", "java", "test", 0, "1000_A. Doubles", rows[("test", 0)]),
            gate_item("write", "go", "test", 1, "1000_B2. Hard", rows[("test", 1)]),
            gate_item("write", "rust", "valid", 3, "999_C. Few", rows[("valid", 3)])]
    assert B.cf_id("1000_B2. Hard") == "1000B2"
    mixed = {"language": [4, 2, 3, 2, 4, 2, 2, 4, 4], "solution": list("abcdefghi")}
    assert B.refs_of({"solutions": mixed}) == [("cpp", "b"), ("cpp", "d"), ("cpp", "f")]
    assert B.refs_of({"solutions": B.trim_refs(mixed)}) == B.refs_of({"solutions": mixed})
    assert B.trim_refs(mixed) == {"language": [4, 2, 2, 4, 2, 4], "solution": list("abdefh")}
    assert B.refs_of({"solutions": {"language": [4, 2, 4], "solution": list("xyz")}}) == [("cpp", "y"), ("java", "x"),
                                                                                            ("java", "z")]
    spec = {"_sha256": "s", "items": [gate[2]["id"], gate[0]["id"], gate[1]["id"]],
            "problems": {"1000B2": {"tol": 1e-6}}}
    saved = E.EVAL_DIR
    tmp = tempfile.mkdtemp()
    quiet = lambda *_: None  # noqa: E731
    try:
        E.EVAL_DIR = tmp
        out = os.path.join(tmp, "gate200")
        vd = write_validators(os.path.join(tmp, "v"), {"1000A": "return n not in (20, 70)", "1000B2": "return True",
                                                       "999C": "return n not in (2, 3, 4)"})
        card = B.build(gate, spec, rows, vd, out, log=quiet)
        items = E.read_jsonl(os.path.join(out, "gate.jsonl"))
        assert [it["id"] for it in items] == [gate[0]["id"], gate[1]["id"], gate[2]["id"]]  # the 495's order
        a = items[0]
        assert "cases" not in a and a["n_cases_full"] == 72 and a["n_cases"] == 70 and a["set_aside"] == [20, 70]
        assert a["judge"] == {} and items[2]["judge"] == {"tol": 1e-6}
        cases = E.item_cases(a)
        assert len(cases) == 70 and ("20", "40") not in cases and ("70", "140") not in cases
        assert items[0]["cases_sha256"] == items[1]["cases_sha256"]  # write and fix share the test file
        assert card["tests"] == 70 + 46 and card["tests_set_aside"] == 2 and card["problems_with_tests_set_aside"] == 1
        assert card["tests_added"] == {} and "hand-checked" not in open(os.path.join(out, "card.md")).read()
        plain_1000a = items[0]["cases_sha256"]

        # 26 Oct 2026: hand-checked tests go after the official ones, through the same validator and judge
        extra = [{"input": "500", "output": "1000", "why": "a wrong method passes every official test"}]
        card = B.build(gate, dict(spec, problems={**spec["problems"], "1000A": {"added_tests": extra}}), rows, vd, out,
                       log=quiet)
        items = E.read_jsonl(os.path.join(out, "gate.jsonl"))
        E._CASES.clear()  # item_cases keeps files read in this process; the rebuild rewrote 1000A.json
        a = items[0]
        assert a["n_cases"] == 71 and a["n_cases_added"] == 1 and a["n_cases_full"] == 72 and a["judge"] == {}
        assert E.item_cases(a)[-1] == ("500", "1000") and E.item_cases(a)[:70] == [tuple(c) for c in json.load(
            open(os.path.join(out, "cases", "1000A.json")))["cases"]][:70]
        assert a["cases_sha256"] != plain_1000a and "n_cases_added" not in items[2]
        assert card["tests"] == 71 + 46 and card["tests_added"] == {"1000A": 1}
        assert "1 hand-checked test added to 1000A" in open(os.path.join(out, "card.md")).read()

        def fails(sp, g=gate, r=rows, v=vd, ex=None, tc=None):
            try:
                B.build(g, dict(sp, _sha256="s"), r, v, out, ex, tc, log=quiet)
            except SystemExit as e:
                return str(e)
            raise AssertionError("should have refused")
        assert "no validator 1000B2.py" in fails({"items": [gate[2]["id"]]}, v=write_validators(
            os.path.join(tmp, "v2"), {"1000A": "return True"}))
        assert "tests left" in fails({"items": [gate[3]["id"]]})
        assert "public or private" in fails({"items": [gate[2]["id"]]}, v=write_validators(
            os.path.join(tmp, "v3"), {"1000B2": "return n != 3"}))
        assert "public or private" in fails({"items": [gate[2]["id"]]}, v=write_validators(
            os.path.join(tmp, "v4"), {"1000B2": "assert n >= 0"}))  # must return True, not fall through
        moved = {**rows, ("test", 1): cc_row("1000_B2. Hard", 1, 5, 40, out=lambda i: i)}  # another revision
        assert "first of" in fails({"items": [gate[2]["id"]]}, r=moved)
        assert "not both" in fails({"items": [gate[0]["id"]], "problems": {"1000A": {"checker": "1000A", "tol": 1e-6}}})
        assert "an added test fails the validator" in fails({"items": [gate[0]["id"]], "problems": {"1000A": {
            "added_tests": [{"input": "70", "output": "140", "why": "w"}]}}})
        assert "not in the 200" in fails({"items": [gate[0]["id"]], "problems": {"1000B2": {
            "added_tests": [{"input": "9", "output": "18", "why": "w"}]}}})
        assert "needs a non-empty" in fails({"items": [gate[0]["id"]], "problems": {"1000A": {
            "added_tests": [{"input": "9", "output": "18"}]}}})
        fake_checkers(lambda i, o, e: int(o) == 2 * int(i) and int(i) != 7)
        assert "rejects expected" in fails({"items": [gate[0]["id"]], "problems": {"1000A": {"checker": "1000A"}}})
        fake_checkers(lambda i, o, e: int(o) == 2 * int(i))
        assert "rejects expected" in fails({"items": [gate[0]["id"]], "problems": {"1000A": {  # a wrong added output
            "checker": "1000A", "added_tests": [{"input": "9", "output": "19", "why": "w"}]}}})
        assert "not tested" in fails({"items": [gate[0]["id"]], "problems": {"1000A": {"checker": "1000A"}}})
        if P.WINDOWS or not shutil.which("g++"):
            print("skip the checker self-test: needs Linux g++")
            return
        tc, ex = P.Toolchain.local(), P.LocalExecutor()
        sols = lambda *codes: {"language": [2] * len(codes), "solution": list(codes)}  # noqa: E731
        two_off = {**rows, ("test", 0): dict(rows[("test", 0)], solutions=sols(cpp_ref(), cpp_ref([40]),
                                                                                  cpp_ref([40, 41])))}
        cspec = {"items": [gate[0]["id"]], "problems": {"1000A": {"checker": "1000A"}}}
        assert "2 or more accepted" in fails(cspec, r=two_off, ex=ex, tc=tc)
        one_off = {**rows, ("test", 0): dict(rows[("test", 0)], solutions=sols(cpp_ref(), cpp_ref([40]), cpp_ref()))}
        card = B.build(gate, dict(cspec, _sha256="s"), one_off, vd, out, ex, tc, log=quiet)
        assert card["checkers"] == ["1000A"] and [r["wrong"] for r in card["per_problem"]["1000A"]["refs"]] == [0, 1, 0]
        assert card["checkers_sha256"] == "v1"
        assert [it["judge"] for it in E.read_jsonl(os.path.join(out, "gate.jsonl"))] == [
            {"checker": "1000A", "checker_sha256": "v1"}]
        assert "fewer than 2" in fails(cspec, r={**rows, ("test", 0): dict(rows[("test", 0)], solutions=sols(cpp_ref()))},
                                       ex=ex, tc=tc)
        rep = B.check(gate, {"problems": {"1000A": {"checker": "1000A"}, "999C": {}}}, one_off, vd, ex, tc, log=quiet)
        assert rep["1000A"]["fit"] and not rep["999C"]["fit"] and "tests left" in rep["999C"]["why"]
    finally:
        E.EVAL_DIR = saved
        E._CASES.clear()
        shutil.rmtree(tmp)


def test_rejudge():
    tmp = tempfile.mkdtemp()
    try:
        items = [{"id": f"write/cpp/cc/test/{k}"} for k in range(4)]
        rows = [{"id": items[0]["id"], "seed": 1, "pass": True, "verdict": "ok", "finish_reason": "stop", "answer": "A"},
                {"id": items[1]["id"], "seed": 1, "pass": False, "verdict": "wrong_answer", "finish_reason": "stop",
                 "answer": "B"},
                {"id": items[2]["id"], "seed": 1, "pass": False, "finish_reason": "length", "answer": "C"},
                {"id": items[0]["id"], "seed": 2, "pass": False, "finish_reason": "stop", "answer": "Z"}]
        seen = []

        def judge(it, content):
            seen.append(content)
            return (content == "B", "ok" if content == "B" else "wrong_answer")
        out = os.path.join(tmp, "r.jsonl")
        s = R.rejudge(items, rows, 1, out, judge, log=lambda *_: None)
        assert sorted(seen) == ["A", "B"]  # unfinished answers aren't run; other seeds ignored
        assert s["missing"] == [items[3]["id"]] and s["old_pass"] == 1 and s["new_pass"] == 1
        assert s["fail_to_pass"] == [items[1]["id"]] and s["pass_to_fail"] == [items[0]["id"]]
        s2 = R.rejudge(items, rows, 1, out, judge, log=lambda *_: None)  # resumable: nothing judged again
        assert len(seen) == 2 and s2["judged"] == 3
    finally:
        shutil.rmtree(tmp)


def test_switch():
    import loop as L
    import switch_gate200 as S
    tmp = tempfile.mkdtemp()
    saved = (E.EVAL_DIR, E.GATE_PATH)
    try:
        E.EVAL_DIR = os.path.join(tmp, "eval")
        E.GATE_PATH = os.path.join(E.EVAL_DIR, "gate.jsonl")
        old = [{"id": f"write/cpp/cc/test/{k}", "kind": "write", "cases": []} for k in range(495)]
        E.write_jsonl(E.GATE_PATH, old)
        g200 = os.path.join(E.EVAL_DIR, "gate200")
        sha = write_cases(os.path.join(g200, "cases"), "1000A", [["1", "2"]])
        new = [dict(old[k], judge={}, cases_file="gate200/cases/1000A.json", cases_sha256=sha) for k in (3, 9)]
        for it in new:
            it.pop("cases")
        E.write_jsonl(os.path.join(g200, "gate.jsonl"), new)
        run = os.path.join(tmp, "loop")
        os.makedirs(os.path.join(run, "rounds", "r000"))
        os.makedirs(os.path.join(run, "rounds", "r001"))
        state = {"gate_sha": "x", "gate_server": {"slots": 5}, "base": {"gate_rows": "g"}, "best": {"round": 0},
                 "offsets": {"glm": 5}, "evalplus_base": {"r": 1},
                 "rounds": [{"round": 0, "status": "reference"}, {"round": 1, "status": "skipped"}]}
        json.dump(state, open(os.path.join(run, "state.json"), "w"))
        with open(os.path.join(run, "loop.pid"), "w") as fh:
            fh.write(str(os.getpid()))
        try:
            S.switch(run, g200, log=lambda *_: None)
            raise AssertionError("must refuse while the loop runs")
        except SystemExit as e:
            assert "running" in str(e)
        os.remove(os.path.join(run, "loop.pid"))
        reuse = os.path.join(tmp, "rejudged.jsonl")
        E.write_jsonl(reuse, [{"id": old[3]["id"], "seed": 1, "pass": True}, {"id": old[4]["id"], "seed": 1, "pass": True},
                              {"id": old[9]["id"], "seed": 2, "pass": True}])
        st = S.switch(run, g200, reuse, 1, log=lambda *_: None)
        assert E.read_jsonl(E.GATE_PATH) == new and len(E.read_jsonl(os.path.join(E.EVAL_DIR, "archive", "gate495",
                                                                                   "gate.jsonl"))) == 495
        assert st["offsets"] == {"glm": 5} and st["evalplus_base"] == {"r": 1} and st["rounds"] == []
        assert "base" not in st and "best" not in st and "gate_sha" not in st and st["gate_server"] == {"slots": 5}
        r0 = E.read_jsonl(os.path.join(run, "rounds", "r000", "gate.seed1.jsonl"))
        assert [r["id"] for r in r0] == [old[3]["id"]]  # only the 200's items, only the loop's seed
        assert os.path.isdir(os.path.join(run, "archive", "gate495", "rounds", "r001"))
        assert json.load(open(os.path.join(run, "state.gate495.json")))["gate_sha"] == "x"
        try:
            S.switch(run, g200, log=lambda *_: None)
            raise AssertionError("must refuse a second switch")
        except SystemExit as e:
            assert "already" in str(e)
        assert L.pid_alive(os.getpid())
    finally:
        E.EVAL_DIR, E.GATE_PATH = saved
        E._CASES.clear()
        shutil.rmtree(tmp)


def test_gate_ctx():
    import loop as L
    cfg = {"gate": {"slots": 8, "ctx": 8 * 44032, "budget": 32768, "max_tokens": 40960}}
    assert L.gate_ctx(cfg) == 352256 and L.gate_ctx({"gate": {}}) == L.llama.SCORING_CTX
    flags = L.Loop.gate_flags(None, cfg)
    assert flags[flags.index("-c") + 1] == "352256" and flags[flags.index("-np") + 1] == "8"
    f0 = L.llama.scoring_flags()
    assert f0[f0.index("-c") + 1] == str(L.llama.SCORING_CTX)  # EvalPlus and the first run unchanged
    assert L.gate_server(cfg)["ctx"] == 352256


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"{name}: ok")
