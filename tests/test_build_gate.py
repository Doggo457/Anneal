"""scripts/build_gate.py end to end on Linux, on a handful of made-up CodeContests problems, with the local C++ and Java
compilers standing in for the Windows sandbox: the Python gate is archived, problems with input files, overlapping
trained rows or no passing C++/Java reference are left out, wrong submissions qualify as fix items only if they pass
every example and fail a later case with a wrong answer or crash (never a timeout, never file I/O or a bare C++ long),
and the gate is filled to --target with fix items and then second-language write items. Skipped without g++ and javac
(on the PC, scripts/setup_toolchains.py --check tests the real sandbox).

  python tests/test_build_gate.py      (or: python -m pytest tests/test_build_gate.py)
"""

import json
import os
import shutil
import sys
import tempfile
import types

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from anneal import evalsets as E  # noqa: E402
from anneal import polyglot as P  # noqa: E402

CPP, JAVA, PY3 = 2, 4, 3

AB_CPP = "#include <cstdio>\nint main(){long long a,b;scanf(\"%lld %lld\",&a,&b);printf(\"%lld\\n\",a+b);}\n"
AB_JAVA = ("import java.util.*;\npublic class Main { public static void main(String[] x) { Scanner s = new Scanner(System.in);"
           " long a = s.nextLong(), b = s.nextLong(); System.out.println(a + b); } }\n")
AB_WRONG_CPP = "#include <cstdio>\nint main(){int a,b;scanf(\"%d %d\",&a,&b);printf(\"%d\\n\",a+b);}\n"  # overflows
AB_WRONG_JAVA = ("import java.util.*;\npublic class Main { public static void main(String[] x) { Scanner s = new "
                 "Scanner(System.in); int a = s.nextInt(), b = s.nextInt(); System.out.println(a + b); } }\n")
MAX_CPP = ("#include <cstdio>\nint main(){int n;scanf(\"%d\",&n);long long m=-(1LL<<62);for(int i=0;i<n;i++){long long v;"
           "scanf(\"%lld\",&v);if(v>m)m=v;}printf(\"%lld\\n\",m);}\n")
MAX_WRONG_CPP = ("#include <cstdio>\nint main(){int n;scanf(\"%d\",&n);long long m=0;for(int i=0;i<n;i++){long long v;"
                 "scanf(\"%lld\",&v);if(v>m)m=v;}printf(\"%lld\\n\",m);}\n")  # 0 start: wrong when all are negative
MAX_LOOP_CPP = ("#include <cstdio>\nint main(){int n;scanf(\"%d\",&n);if(n>3)for(;;){}long long m=-(1LL<<62);for(int i=0;"
                "i<n;i++){long long v;scanf(\"%lld\",&v);if(v>m)m=v;}printf(\"%lld\\n\",m);}\n")  # times out
MAX_FILE_CPP = "#include <cstdio>\nint main(){freopen(\"in.txt\",\"r\",stdin);return 0;}\n"
MAX_LONG_CPP = "#include <cstdio>\nint main(){long x=0;printf(\"%ld\\n\",x);}\n"
SUM_JAVA = ("import java.util.*;\npublic class Main { public static void main(String[] x) { Scanner s = new Scanner(System.in);"
            " int n = s.nextInt(); long t = 0; for (int i = 0; i < n; i++) t += s.nextLong(); System.out.println(t); } }\n")
SUM_WRONG_JAVA = ("public class Main { public static void main(String[] x) { int[] a = new int[1]; "
                  "java.util.Scanner s = new java.util.Scanner(System.in); int n = s.nextInt(); if (n > 2) a[5] = 1; "
                  "long t = 0; for (int i = 0; i < n; i++) t += s.nextLong(); System.out.println(t); } }\n")  # crashes


def tests(pairs):
    return {"input": [i for i, _ in pairs], "output": [o for _, o in pairs]}


def sols(*pairs):
    return {"language": [lang for lang, _ in pairs], "solution": [code for _, code in pairs]}


def problem(name, desc, public, private, good, wrong=(), input_file=""):
    return {"name": name, "description": desc, "public_tests": tests(public), "private_tests": tests(private),
            "generated_tests": tests([]), "solutions": sols(*good), "incorrect_solutions": sols(*wrong),
            "input_file": input_file, "output_file": "", "cf_rating": 1200, "difficulty": 7, "source": 2}


TRAINED_DESC = ("Read a word and print it backwards. The input is one line holding a lowercase word of at most one "
                "hundred letters and the output is the same word with its letters in reverse order on one line.")
ROWS = {
    "valid": [
        problem("ab", "Given two integers a and b, print their sum. Both fit in a signed 64-bit integer.",
                [("1 2\n", "3\n"), ("5 7\n", "12\n")], [("3000000000 3000000000\n", "6000000000\n"), ("0 0\n", "0\n")],
                [(CPP, AB_CPP), (JAVA, AB_JAVA)], [(CPP, AB_WRONG_CPP), (JAVA, AB_WRONG_JAVA)]),
        problem("max", "Given n and then n integers, print the largest of them.",
                [("3\n1 5 2\n", "5\n")], [("2\n-4 -9\n", "-4\n"), ("4\n1 2 3 4\n", "4\n")],
                [(CPP, MAX_CPP)], [(CPP, MAX_FILE_CPP), (CPP, MAX_LONG_CPP), (CPP, MAX_LOOP_CPP), (CPP, MAX_WRONG_CPP)]),
        problem("max2", "Given n and then n integers, print the largest of them. All values are below a billion.",
                [("3\n1 5 2\n", "5\n")], [("4\n1 2 3 4\n", "4\n")], [(CPP, MAX_CPP)], [(CPP, MAX_LOOP_CPP)]),
        problem("files", "Read the numbers from input.txt and write their sum to output.txt.",
                [("1 2\n", "3\n")], [], [(CPP, AB_CPP)], input_file="input.txt"),
    ],
    "test": [
        problem("sum", "Given n and then n integers, print their sum, which fits in a signed 64-bit integer.",
                [("2\n1 2\n", "3\n")], [("3\n1 2 3\n", "6\n"), ("1\n9\n", "9\n")],
                [(PY3, "print(1)"), (JAVA, SUM_JAVA)], [(JAVA, SUM_WRONG_JAVA)]),
        problem("pyonly", "Print the number of vowels in the given lowercase word, on one line by itself.",
                [("abe\n", "2\n")], [], [(PY3, "print(sum(c in 'aeiou' for c in input()))")]),
        problem("reverse", TRAINED_DESC, [("abc\n", "cba\n")], [], [(CPP, AB_CPP)]),
    ],
}


def test_build_gate():
    if os.name == "nt" or not (shutil.which("g++") and shutil.which("javac")):
        print("skip: needs g++ and javac on Linux (the PC runs scripts/setup_toolchains.py --check)")
        return
    import build_evalsets as BE
    import build_gate as B
    tmp = tempfile.mkdtemp(prefix="anneal-gate-test-")
    saved = {"sandbox": sys.modules.get("anneal.sandbox"), "anneal.sandbox": getattr(sys.modules["anneal"], "sandbox", None),
             "E": (E.EVAL_DIR, E.GATE_PATH, E.DECONTAM_EXTRA), "B": (B.ROOT, B.ARCHIVE),
             "BE": (BE.fetch, BE.parquet_rows, BE.evalplus_index),
             "P": (P.SandboxExecutor, P.Toolchain.__dict__["windows"]), "argv": sys.argv}
    try:
        tcdir = os.path.join(tmp, "sandbox", "toolchains")
        os.makedirs(tcdir)
        json.dump({"gcc": {"version": "13", "sha256": "a" * 64, "version_output": "g++ 13"},
                   "jdk": {"version": "21", "sha256": "b" * 64}, "shims": {"node_stdin.js": "c" * 64}},
                  open(os.path.join(tcdir, "lock.json"), "w"))
        stub = types.ModuleType("anneal.sandbox")
        stub.require, stub.TOOLCHAINS_DIR = (lambda: None), tcdir
        sys.modules["anneal.sandbox"] = stub
        sys.modules["anneal"].sandbox = stub
        ev = os.path.join(tmp, "data", "eval")
        E.EVAL_DIR, E.GATE_PATH, E.DECONTAM_EXTRA = ev, os.path.join(ev, "gate.jsonl"), os.path.join(ev, "decontam_extra.jsonl")
        B.ROOT, B.ARCHIVE = tmp, os.path.join(ev, "archive", "python-gate")
        gocache = os.path.join(tmp, "gocache")
        P.SandboxExecutor = P.LocalExecutor
        P.Toolchain.windows = classmethod(lambda cls, root=None: cls.local(gocache=gocache))
        BE.fetch = lambda repo, patterns, revision=None: ([f"{s}-00000-of-00001.parquet" for s in ROWS], revision or "cc-sha")
        BE.parquet_rows = lambda paths, columns=None: (r for p in paths for r in ROWS[os.path.basename(p).split("-")[0]])
        BE.evalplus_index = lambda: E.OverlapIndex([])
        # the retired Python gate and its card; earlier decontamination rows; a trained row that matches "reverse"
        E.write_jsonl(E.GATE_PATH, [{"id": "cc/valid/0", "kind": "cc", "prompt": "p"}])
        json.dump({"cc_revision": "old-cc-sha"}, open(os.path.join(ev, "card.json"), "w"))
        E.write_jsonl(E.DECONTAM_EXTRA, [{"id": "oci/1", "prompt": "old"}])
        E.write_jsonl(os.path.join(tmp, "data", "processed", "loop", "x", "train.jsonl"),
                      [{"messages": [{"role": "user", "content": TRAINED_DESC}, {"role": "assistant", "content": "ok"}]}])

        def run(target):
            sys.argv = ["build_gate.py", "--target", str(target), "--parallel", "2"]
            return B.main()
        # no passing smoke check yet: refuses before touching anything
        try:
            run(8)
            raise AssertionError("built without a smoke check")
        except SystemExit as exc:
            assert "smoke check" in str(exc), exc
        assert E.read_jsonl(E.GATE_PATH)[0]["kind"] == "cc"
        os.makedirs(os.path.join(tmp, "results", "toolchains"))
        json.dump({"mismatches": 0, "results": [{"ok": True}]}, open(os.path.join(tmp, "results", "toolchains", "smoke.json"), "w"))
        run(8)
        gate = E.read_jsonl(E.GATE_PATH)
        card = json.load(open(os.path.join(ev, "card.json")))
        by_id = {it["id"]: it for it in gate}
        # kept: ab (C++ and Java refs), max and max2 (C++), sum (Java); out: files (input file), pyonly (no C++ or
        # Java reference), reverse (trained on)
        assert card["cc_revision"] == "old-cc-sha" and card["cc_candidates"] == 6, card
        assert card["overlap_dropped"] == ["cc/test/2"] and card["no_ref_pass"] == 1, card
        assert card["ref_pass"] == {"cpp": 3, "java": 2}, card
        assert {it["problem"] for it in gate if it["kind"] == "write"} == {"cc/test/0", "cc/valid/0", "cc/valid/1",
                                                                         "cc/valid/2"}
        fixes = {it["problem"]: it for it in gate if it["kind"] == "fix"}
        assert set(fixes) == {"cc/valid/0", "cc/valid/1", "cc/test/0"}, sorted(fixes)  # max2's only one times out
        # max: the file-I/O, bare-long and endless-loop submissions never qualify; the wrong start value does
        assert fixes["cc/valid/1"]["lang"] == "cpp" and fixes["cc/valid/1"]["wrong"]["index"] == 3, fixes["cc/valid/1"]
        assert fixes["cc/test/0"]["lang"] == "java" and fixes["cc/test/0"]["wrong"]["statuses"].get("runtime_error")
        assert MAX_WRONG_CPP.strip() in fixes["cc/valid/1"]["prompt"] and "```cpp" in fixes["cc/valid/1"]["prompt"]
        assert len(gate) == 8 and card["write_items"] == 5 and card["second_language_write_items"] == 1, card
        assert all(it["lang"] in P.LANG_ORDER and it["cases"] for it in gate) and len(by_id) == 8
        assert all(it["prompt"].endswith(P.write_suffix(it["lang"])) for it in gate if it["kind"] == "write")
        assert card["toolchain_fingerprint"] == P.toolchain_fingerprint(tcdir) and card["archived_python_gate"]
        assert E.read_jsonl(os.path.join(B.ARCHIVE, "gate.jsonl"))[0]["kind"] == "cc"
        assert json.load(open(os.path.join(B.ARCHIVE, "card.json")))["cc_revision"] == "old-cc-sha"
        extra = E.read_jsonl(E.DECONTAM_EXTRA)
        assert [r["id"] for r in extra][:1] == ["oci/1"] and len(extra) == 1 + 4, extra
        assert "fix C++" in open(os.path.join(ev, "card.md")).read()
        # every gate item's own reference (or the problem's) is judged right by the gate's judge, every fix item's
        # wrong submission wrong: the gate's answers are checkable
        judge = E.polyglot_check(make_executor=P.LocalExecutor, make_toolchain=lambda: P.Toolchain.local(gocache=gocache))
        ref = {"cc/valid/0": {"cpp": AB_CPP, "java": AB_JAVA}, "cc/valid/1": {"cpp": MAX_CPP},
               "cc/valid/2": {"cpp": MAX_CPP}, "cc/test/0": {"java": SUM_JAVA}}
        wrong = {"cc/valid/0": {"cpp": AB_WRONG_CPP, "java": AB_WRONG_JAVA}, "cc/valid/1": {"cpp": MAX_WRONG_CPP},
                 "cc/test/0": {"java": SUM_WRONG_JAVA}}
        for it in gate:
            code = ref[it["problem"]].get(it["lang"])
            if code:
                assert judge(it, f"```{it['lang']}\n{code}```") == (True, "ok"), it["id"]
            if it["kind"] == "fix":
                ok, stage = judge(it, f"```{it['lang']}\n{wrong[it['problem']][it['lang']]}```")
                assert not ok and stage in ("wrong_answer", "runtime_error"), (it["id"], stage)
        # the six-language gate is never rebuilt over
        try:
            run(8)
            raise AssertionError("rebuilt over the six-language gate")
        except SystemExit as exc:
            assert "already the six-language gate" in str(exc), exc
    finally:
        if saved["sandbox"] is not None:
            sys.modules["anneal.sandbox"] = saved["sandbox"]
        else:
            sys.modules.pop("anneal.sandbox", None)
        if saved["anneal.sandbox"] is not None:
            sys.modules["anneal"].sandbox = saved["anneal.sandbox"]
        E.EVAL_DIR, E.GATE_PATH, E.DECONTAM_EXTRA = saved["E"]
        B.ROOT, B.ARCHIVE = saved["B"]
        BE.fetch, BE.parquet_rows, BE.evalplus_index = saved["BE"]
        P.SandboxExecutor, P.Toolchain.windows = saved["P"]
        sys.argv = saved["argv"]
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    test_build_gate()
    print("test_build_gate: ok")
