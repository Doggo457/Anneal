"""The six-language judge (anneal/polyglot.py) on Linux with the local toolchains: code extraction, Java names, and
every smoke program (anneal/polyglot_smoke.py) getting its expected verdict. Languages whose compiler isn't on PATH
are skipped. The Windows sandbox itself is checked on the PC by scripts/setup_toolchains.py --check.

  python tests/test_polyglot.py      (or: python -m pytest tests/test_polyglot.py)
"""

import os
import shutil
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from anneal import polyglot as P  # noqa: E402
from anneal import polyglot_smoke as S  # noqa: E402

TOOLS = {"cpp": "g++", "java": "javac", "go": "go", "rust": "rustc", "javascript": "node", "typescript": "node"}


def test_extract_code():
    ans = ("The bug is here:\n```cpp\nint x = a[i+1];\n```\nFixed program:\n```c++\n#include <cstdio>\nint main() {\n"
           "  int a, b;\n  scanf(\"%d %d\", &a, &b);\n  printf(\"%d\\n\", a + b);\n}\n```\nDone.")
    code = P.extract_code(ans, "cpp")
    assert code.startswith("#include <cstdio>") and code.rstrip().endswith("}"), code
    # the original shown first, then the fix of about the same length: the last one
    orig, fixed = "x = 1\n" * 10, "x = 2\n" * 10
    assert P.extract_code(f"```go\n{orig}```\nnow\n```go\n{fixed}```", "go") == fixed
    assert P.extract_code("```python\nprint(1)\n```", "rust") == ""  # wrong language: no program
    assert P.extract_code("```\nfn main() {}\n```", "rust") == "fn main() {}\n"  # untagged is accepted
    assert P.extract_code("```ts\nlet a = 1;\n", "typescript") == "let a = 1;\n"  # unclosed last block
    assert P.extract_code("~~~javascript\nconsole.log(1)\n~~~", "javascript") == "console.log(1)\n"
    assert P.code_blocks("````md\n```js\nx\n```\n````") == [("md", "```js\nx\n```\n")]


def test_java_names():
    assert P.java_names("public class Solution { public static void main(String[] a) {} }") == \
        ("Solution.java", "Solution")
    assert P.java_names("class A {}\nclass Main { static class B {} public static void main(String[] a) {} }") == \
        ("Main.java", "Main")
    assert P.java_names("package p.q;\npublic class Main { public static void main(String[] a) {} }") == \
        ("Main.java", "p.q.Main")
    src = 'public class Out { static String s = "class Fake {"; }\nclass Runner { public static void main(String[] a) {} }'
    assert P.java_names(src) == ("Out.java", "Runner")
    assert P.java_names("// class X {\nvoid main() { System.out.println(1); }") == ("Main.java", "Main")
    assert P.java_names("public final class Main {\n  record P(int x) {}\n  public static void main(String[] a) {}\n}") \
        == ("Main.java", "Main")


def test_build_plan_windows_args():
    """Command lines as the PC builds them: rustc's stack flag after "-C debuginfo=0" (not inside it), go build with
    at most 4 compilers at once, node keeping paths as given (the sandbox may not lstat C:\\)."""
    from types import SimpleNamespace
    tc = SimpleNamespace(gxx="g++", javac="javac", java="java", java_home=None, go="go", goroot=None, gocache=None,
                         rustc="rustc", node="node", typescript="tsc.js", shims="shims")
    saved = P.WINDOWS
    P.WINDOWS = True
    try:
        _, build, _ = P.build_plan("rust", "fn main() {}", tc, "wd")
        args = build[0][1]
        pairs = [(args[i], args[i + 1]) for i, a in enumerate(args) if a == "-C"]
        assert [v for _, v in pairs] == ["debuginfo=0", f"link-args=-Wl,--stack,{P.STACK_BYTES}"], args
        assert args[-3:] == ["-o", "main.exe", "main.rs"], args
        _, build, _ = P.build_plan("go", "package main", tc, "wd")
        assert build[0][1][:3] == ["build", "-p", "4"], build
        for lang in ("javascript", "typescript"):
            _, build, run = P.build_plan(lang, "console.log(1)", tc, "wd")
            for args in [run[1]] + [b[1] for b in build]:
                assert args[:2] == P.NODE_PATH_FLAGS, (lang, args)
        _, build, _ = P.build_plan("cpp", "int main(){}", tc, "wd")
        assert f"-Wl,--stack,{P.STACK_BYTES}" in build[0][1]
    finally:
        P.WINDOWS = saved


def test_prompts():
    for lang in P.LANG_ORDER:
        s = P.write_suffix(lang)
        assert f"```{P.FENCE[lang]}" in s and P.NAMES[lang] in s and P.ENVIRONMENT[lang] in s
        f = P.fix_prompt("Add two numbers.", lang, "code here\n")
        assert f.startswith("Add two numbers.") and f"```{P.FENCE[lang]}\ncode here\n```" in f
    assert "long long" in P.ENVIRONMENT["cpp"]


def test_smoke_programs():
    if P.WINDOWS:  # LocalExecutor is Linux-only; on the PC the sandbox runs these (setup_toolchains.py --check)
        print("skip: on Windows, scripts/setup_toolchains.py --check runs the smoke programs in the sandbox")
        return
    gocache = tempfile.mkdtemp(prefix="gocache-")
    try:
        tc, ex = P.Toolchain.local(gocache=gocache), P.LocalExecutor()
        bad, ran = [], 0
        for lang in P.LANG_ORDER:
            if not shutil.which(TOOLS[lang]) or (lang == "typescript" and not os.path.isdir(tc.typescript)):
                print(f"skip {lang}: no local toolchain")
                continue
            for prog in S.programs(lang, windows=False):
                v = P.judge(lang, prog["code"], prog["cases"], ex, tc, case_timeout_s=prog["timeout"], all_cases=True)
                ran += 1
                if v.stage != prog["expect"]:
                    bad.append((lang, prog["name"], prog["expect"], v.stage, v.cases, v.detail[-400:]))
        assert not bad, bad
        assert ran >= 6, ran
    finally:
        shutil.rmtree(gocache, ignore_errors=True)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"{name}: ok")
