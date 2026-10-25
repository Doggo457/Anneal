"""Build and judge stdin/stdout programs in six languages: C++, Java, Go, Rust, JavaScript and TypeScript.

The gate (scripts/build_gate.py, anneal/evalsets.py) asks the model to write or fix a program in one of these
languages; this module pulls the program out of the answer, builds it, runs it on the problem's cases and compares
the output token by token: on the 495-item gate numbers within 1e-6 (as generate_traces.py outputs_match); on the
fixed 200 (25 Oct 2026) with gate_match, or with the problem's own checker where many answers are right.

Where things run. On the training PC (Windows) every build and run goes through anneal/sandbox.py: builds run in the
"Anneal.CodeBuild" AppContainer (read-only access to the toolchains, up to 16 processes, because compilers start
helpers), and the built program runs in "Anneal.CodeSandbox", the container that already runs model-written Python
(no network, no files outside its own work folder, one process, 4 GiB). Each program gets a fresh work folder that
both containers can use and that is deleted afterwards. The compilers never run the code they compile. LocalExecutor
runs the same commands without any isolation; it exists only for the Linux tests of this module.

Toolchains (scripts/setup_toolchains.py installs them under sandbox/toolchains, pinned and checksummed):
  C++         GCC (WinLibs MinGW-w64, UCRT): g++ -std=gnu++20 -O2 -DONLINE_JUDGE -static, 256 MiB stack
  Java        Temurin JDK 25: javac, then java -Xss128m -Xmx2g -XX:+UseSerialGC -DONLINE_JUDGE=true
  Go          Go 1.27: go build, standard library only, cgo off
  Rust        Rust 1.99 (x86_64-pc-windows-gnu): rustc --edition 2021 -O, 256 MiB stack
  JavaScript  Node.js 24, CommonJS or ES modules, with a preload that makes readFileSync(0) and /dev/stdin read all
              of standard input on Windows
  TypeScript  TypeScript 6 transpileModule to CommonJS (no type check), then Node.js 24 as JavaScript
ONLINE_JUDGE is defined as on Codeforces, so solutions that read input.txt only off the judge read standard input here.
"""

import math
import os
import re
import shutil
import subprocess
import tempfile
import time
from collections import namedtuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SHIMS_SRC = os.path.join(ROOT, "anneal", "shims")
WINDOWS = os.name == "nt"

LANG_ORDER = ("cpp", "java", "go", "rust", "javascript", "typescript")
NAMES = {"cpp": "C++", "java": "Java", "go": "Go", "rust": "Rust", "javascript": "JavaScript",
         "typescript": "TypeScript"}
FENCE = {"cpp": "cpp", "java": "java", "go": "go", "rust": "rust", "javascript": "javascript", "typescript": "typescript"}
TAGS = {  # fence tags read as each language (lowercase)
    "cpp": {"cpp", "c++", "cc", "cxx", "hpp", "c"},
    "java": {"java"},
    "go": {"go", "golang"},
    "rust": {"rust", "rs"},
    "javascript": {"javascript", "js", "jsx", "node", "nodejs", "mjs", "cjs"},
    "typescript": {"typescript", "ts", "tsx"},
}
# What the prompt tells the model about the judge, so a correct program isn't failed by the platform.
ENVIRONMENT = {
    "cpp": "It is compiled as GNU C++20 with g++ -O2 (MinGW-w64 on 64-bit Windows, where long is 32 bits: use long "
           "long for 64-bit integers).",
    "java": "It is compiled and run with Java 25, standard library only.",
    "go": "It is built with Go 1.27 as package main, standard library only.",
    "rust": "It is compiled with Rust 1.99, edition 2021, standard library only (no crates).",
    "javascript": "It runs on Node.js 24 with no packages; read the whole input with "
                  "require('fs').readFileSync(0, 'utf8') or the readline module.",
    "typescript": "It is compiled to JavaScript without type checking and run on Node.js 24 with no packages; read "
                  "the whole input with require('fs').readFileSync(0, 'utf8') or the readline module.",
}

BUILD_TIMEOUT_S = 90.0
BUILD_MEM = 3 << 30
RUN_MEM = 4 << 30
CASE_TIMEOUT_S = 10.0
STACK_BYTES = 256 << 20

Verdict = namedtuple("Verdict", "ok stage cases detail build_s run_s")
# stage: ok | no_code | compile_error | runtime_error | timeout | wrong_answer
# cases: one status per case run (ok / wrong_answer / runtime_error / timeout), in order


# ============================================================================================ prompts

def write_suffix(lang):
    tag = FENCE[lang]
    return (f"\n\nWrite a complete {NAMES[lang]} program that reads the input from standard input and writes the "
            f"answer to standard output. {ENVIRONMENT[lang]} Put the whole program in a single ```{tag} code block.")


def fix_prompt(description, lang, code):
    tag = FENCE[lang]
    return (f"{description.strip()}\n\nThis {NAMES[lang]} program was written to solve the problem above. It passes "
            f"the examples but gives wrong answers or fails on other tests:\n\n```{tag}\n{code.rstrip()}\n```\n\n"
            f"Find and fix the bug. {ENVIRONMENT[lang]} Give the complete corrected program, reading standard input "
            f"and writing standard output, in a single ```{tag} code block.")


# ============================================================================================ extraction

_FENCE_RE = re.compile(r"^[ \t]*(`{3,}|~{3,})[ \t]*([^\n`]*)$", re.MULTILINE)


def code_blocks(text):
    """[(tag, body)] of the fenced blocks in text, in order. A block closes at the next fence line of the same
    character and at least the same length with no tag; an unclosed last block runs to the end."""
    out, pos = [], 0
    text = text or ""
    while True:
        m = _FENCE_RE.search(text, pos)
        if not m:
            return out
        fence, info = m.group(1), m.group(2).strip()
        tag = info.split()[0].strip("{}.").lower() if info else ""
        start = m.end() + 1
        close = None
        for c in _FENCE_RE.finditer(text, start):
            if c.group(1)[0] == fence[0] and len(c.group(1)) >= len(fence) and not c.group(2).strip():
                close = c
                break
        if close is None:
            out.append((tag, text[start:]))
            return out
        out.append((tag, text[start:close.start()]))
        pos = close.end()


def extract_code(text, lang):
    """The program in an answer: among the blocks tagged with the language (or, if none is, the untagged blocks),
    the last one at least 60% as long as the longest of them - the final full program, not a quoted snippet or the
    buggy original shown first. "" if the answer has no such block."""
    blocks = code_blocks(text)
    cands = [b for t, b in blocks if t in TAGS[lang]] or [b for t, b in blocks if not t]
    cands = [b for b in cands if b.strip()]
    if not cands:
        return ""
    longest = max(len(b) for b in cands)
    return [b for b in cands if len(b) >= 0.6 * longest][-1]


# ============================================================================================ Java names

def _strip_java(src):
    """Source with comments, string and char literals blanked (same length), so braces and names can be scanned."""
    out, i, n = [], 0, len(src)
    while i < n:
        c = src[i]
        if src.startswith("//", i):
            j = src.find("\n", i)
            j = n if j == -1 else j
            out.append(" " * (j - i))
            i = j
        elif src.startswith("/*", i):
            j = src.find("*/", i + 2)
            j = n if j == -1 else j + 2
            out.append(re.sub(r"[^\n]", " ", src[i:j]))
            i = j
        elif src.startswith('"""', i):
            j = src.find('"""', i + 3)
            j = n if j == -1 else j + 3
            out.append(re.sub(r"[^\n]", " ", src[i:j]))
            i = j
        elif c in "\"'":
            j = i + 1
            while j < n and src[j] != c and src[j] != "\n":
                j += 2 if src[j] == "\\" else 1
            j = min(j + 1, n)
            out.append(" " * (j - i))
            i = j
        else:
            out.append(c)
            i += 1
    return "".join(out)


_TYPE_RE = re.compile(r"\b(class|interface|enum|record)\s+([A-Za-z_$][\w$]*)")
_MAIN_RE = re.compile(r"\bvoid\s+main\s*\(")


def java_names(src):
    """(file name, class to run) for a Java program: the file is named after its public top-level type (javac
    requires it), else Main.java; the class to run is the top-level type whose body holds main, with the package
    prefix. A program without any type declaration (a compact source file) runs as the file's own class."""
    s = _strip_java(src)
    pkg = re.search(r"^\s*package\s+([\w.]+)\s*;", s, re.MULTILINE)
    prefix = pkg.group(1) + "." if pkg else ""
    tops, depth = [], 0  # [name, public, body start, body end]
    pending = None
    for m in re.finditer(r"[{}]|" + _TYPE_RE.pattern, s):
        tok = m.group(0)
        if tok == "{":
            if pending is not None and depth == 0:
                tops.append([pending[0], pending[1], m.end(), len(s)])
                pending = None
            depth += 1
        elif tok == "}":
            depth -= 1
            if depth == 0 and tops and tops[-1][3] == len(s):
                tops[-1][3] = m.start()
        elif depth == 0 and pending is None:
            head = s[max(0, s.rfind("\n", 0, m.start()) + 1):m.start()]
            pending = (m.group(2), bool(re.search(r"\bpublic\b", head)))
    public = [t for t in tops if t[1]]
    file_cls = public[0][0] if public else None
    main_cls = next((t[0] for t in tops if _MAIN_RE.search(s, t[2], t[3])), None)
    if not tops:
        return "Main.java", prefix + "Main"
    file_cls = file_cls or main_cls or "Main"
    return f"{file_cls}.java", prefix + (main_cls or file_cls)


# ============================================================================================ toolchains

class Toolchain:
    """Paths of the compilers and runtimes. windows(): the sandbox layout setup_toolchains.py creates; local(): the
    tools on PATH (tests)."""

    def __init__(self, gxx, javac, java, go, rustc, node, typescript, shims, goroot=None, gocache=None,
                 java_home=None, path_dirs=None):
        self.gxx, self.javac, self.java, self.go, self.rustc = gxx, javac, java, go, rustc
        self.node, self.typescript, self.shims = node, typescript, shims
        self.goroot, self.gocache, self.java_home = goroot, gocache, java_home
        self.path_dirs = path_dirs or {}

    @classmethod
    def windows(cls, root=None):
        from anneal import sandbox
        r = root or sandbox.TOOLCHAINS_DIR
        j = lambda *p: os.path.join(r, *p)  # noqa: E731
        return cls(gxx=j("gcc", "bin", "g++.exe"), javac=j("jdk", "bin", "javac.exe"), java=j("jdk", "bin", "java.exe"),
                   go=j("go", "bin", "go.exe"), rustc=j("rust", "bin", "rustc.exe"), node=j("node", "node.exe"),
                   typescript=j("typescript"), shims=j("shims"), goroot=j("go"), gocache=j("cache", "go-build"),
                   java_home=j("jdk"),
                   path_dirs={"cpp": [j("gcc", "bin")], "java": [j("jdk", "bin")], "go": [j("go", "bin")],
                              "rust": [j("rust", "bin")], "javascript": [j("node")], "typescript": [j("node")]})

    @classmethod
    def local(cls, typescript=None, gocache=None):
        node = shutil.which("node") or "node"
        ts = typescript or os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(node))), "lib",
                                        "node_modules", "typescript")
        return cls(gxx=shutil.which("g++") or "g++", javac=shutil.which("javac") or "javac",
                   java=shutil.which("java") or "java", go=shutil.which("go") or "go",
                   rustc=shutil.which("rustc") or "rustc", node=node, typescript=ts, shims=SHIMS_SRC, gocache=gocache)


def toolchain_fingerprint(root=None):
    """A short hash of the installed toolchains: each archive's sha256 and each shim's, from
    sandbox/toolchains/lock.json (None if it is missing). Gate scores are only comparable under one fingerprint."""
    import hashlib
    import json
    if root is None:
        from anneal import sandbox
        root = sandbox.TOOLCHAINS_DIR
    path = os.path.join(root, "lock.json")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as fh:
        lock = json.load(fh)
    key = {k: v.get("sha256") for k, v in lock.items() if isinstance(v, dict) and "sha256" in v}
    key["shims"] = lock.get("shims")
    return hashlib.sha256(json.dumps(key, sort_keys=True).encode("utf-8")).hexdigest()[:16]


# Node resolves the real path of its main script and of --require files by walking every folder up to the drive
# root; the code sandbox's AppContainer may not lstat C:\ (EPERM), so these keep the paths as given.
NODE_PATH_FLAGS = ["--preserve-symlinks", "--preserve-symlinks-main"]


def build_plan(lang, code, tc, wd_path):
    """(files {name: text}, build steps [(image, args, env)], run (image, args, env)) for one program. Paths in
    args are relative to the work folder, which is the current folder of every step."""
    tmp = os.path.join(wd_path, "tmp")
    common = {"TMPDIR": tmp, "TEMP": tmp, "TMP": tmp}
    exe = os.path.join(wd_path, "main.exe")
    stdin_shim = os.path.join(tc.shims, "node_stdin.js")
    if lang == "cpp":
        args = ["-std=gnu++20", "-O2", "-DONLINE_JUDGE", "-o", "main.exe", "main.cpp"]
        if WINDOWS:
            args[3:3] = ["-static", f"-Wl,--stack,{STACK_BYTES}"]
        return {"main.cpp": code}, [(tc.gxx, args, common)], (exe, [], {})
    if lang == "java":
        fname, cls = java_names(code)
        env = dict(common, **({"JAVA_HOME": tc.java_home} if tc.java_home else {}))
        build = [(tc.javac, ["-J-Xmx1g", "-encoding", "UTF-8", "-nowarn", "-d", ".", fname], env)]
        run = (tc.java, ["-Xss128m", "-Xmx2g", "-XX:+UseSerialGC", "-XX:-UsePerfData", "-DONLINE_JUDGE=true",
                         "-Dfile.encoding=UTF-8", "-Dstdout.encoding=UTF-8", "-cp", ".", cls],
               {"JAVA_HOME": tc.java_home} if tc.java_home else {})
        return {fname: code}, build, run
    if lang == "go":
        env = dict(common, GOPATH=os.path.join(wd_path, "gopath"), GOTMPDIR=tmp, GOTOOLCHAIN="local",
                   GOFLAGS="-buildvcs=false", CGO_ENABLED="0", GOENV="off", GOPROXY="off", GOTELEMETRY="off",
                   GO111MODULE="")
        if tc.goroot:
            env["GOROOT"] = tc.goroot
        env["GOCACHE"] = tc.gocache or os.path.join(wd_path, "gocache")
        # -p 4: a cold build cache compiles the standard library package by package, in parallel; more compilers at
        # once than the build job's process limit fail with "Not enough quota"
        return {"main.go": code}, [(tc.go, ["build", "-p", "4", "-o", "main.exe", "main.go"], env)], (exe, [], {})
    if lang == "rust":
        args = ["--edition", "2021", "-O", "-C", "debuginfo=0", "-o", "main.exe", "main.rs"]
        if WINDOWS:
            args[5:5] = ["-C", f"link-args=-Wl,--stack,{STACK_BYTES}"]  # after "-C debuginfo=0"
        return {"main.rs": code}, [(tc.rustc, args, common)], (exe, [], {})
    if lang == "javascript":
        # .js with ES module syntax runs as a module (Node 24 detects it); CommonJS otherwise
        return {"main.js": code}, [], (tc.node, NODE_PATH_FLAGS + ["--require", stdin_shim, "main.js"], {})
    if lang == "typescript":
        ts_build = [os.path.join(tc.shims, "ts_build.js"), tc.typescript, "main.ts", "main.js"]
        build = [(tc.node, NODE_PATH_FLAGS + ts_build, common)]
        return {"main.ts": code}, build, (tc.node, NODE_PATH_FLAGS + ["--require", stdin_shim, "main.js"], {})
    raise ValueError(f"unknown language {lang!r}")


# ============================================================================================ executors

Run = namedtuple("Run", "returncode stdout stderr timed_out wall_s")


class LocalExecutor:
    """No isolation at all: for the Linux tests of this module only, never for model output on the training PC."""

    def __init__(self, build_timeout_s=BUILD_TIMEOUT_S):
        self.build_timeout_s = build_timeout_s

    def workdir(self):
        return _LocalWorkdir()

    def run(self, wd, image, args, env, stdin=b"", timeout_s=10.0, mem_bytes=RUN_MEM, phase="run", path_dirs=None):
        full_env = dict(os.environ)
        full_env.update({k: v for k, v in env.items() if v != ""})
        for k in [k for k, v in env.items() if v == ""]:
            full_env.pop(k, None)
        t0 = time.perf_counter()
        try:
            p = subprocess.run([image] + list(args), cwd=wd.path, env=full_env, input=stdin, capture_output=True,
                               timeout=timeout_s, preexec_fn=None if WINDOWS else _big_stack)
            return Run(p.returncode, p.stdout, p.stderr, False, time.perf_counter() - t0)
        except subprocess.TimeoutExpired as e:
            return Run(124, e.stdout or b"", e.stderr or b"", True, time.perf_counter() - t0)


def _big_stack():
    """Linux tests: a 512 MiB main-thread stack, as the Windows builds get from their linker flag."""
    import resource
    _, hard = resource.getrlimit(resource.RLIMIT_STACK)
    want = 512 << 20
    resource.setrlimit(resource.RLIMIT_STACK, (want if hard == resource.RLIM_INFINITY else min(want, hard), hard))


class _LocalWorkdir:
    def __init__(self):
        self._td = tempfile.TemporaryDirectory(prefix="polyglot-")
        self.path = self._td.name

    def write(self, name, data):
        p = os.path.join(self.path, name)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "wb") as fh:
            fh.write(data.encode("utf-8") if isinstance(data, str) else data)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self._td.cleanup()


class SandboxExecutor:
    """Builds in the build container, runs in the code sandbox (anneal/sandbox.py). Windows only."""

    def __init__(self):
        from anneal import sandbox
        self.sb = sandbox

    def workdir(self):
        return self.sb.Workdir(containers=(self.sb.BUILD_CONTAINER, self.sb.CONTAINER_NAME))

    def run(self, wd, image, args, env, stdin=b"", timeout_s=10.0, mem_bytes=RUN_MEM, phase="run", path_dirs=None):
        build = phase == "build"
        r = self.sb.spawn(image, list(args), cwd=wd.path, env=env, stdin=stdin, timeout_s=timeout_s,
                          mem_bytes=mem_bytes, max_procs=16 if build else 1,
                          container=self.sb.BUILD_CONTAINER if build else self.sb.CONTAINER_NAME, path_dirs=path_dirs)
        return Run(r.returncode, r.stdout, r.stderr, r.timed_out, r.wall_s)


# ============================================================================================ judging

def outputs_match(got, want, rel=1e-6):
    """Token by token; numbers may differ by 1e-6 relative (scripts/generate_traces.py outputs_match)."""
    g, w = got.split(), want.split()
    if g == w:
        return True
    if len(g) != len(w):
        return False
    for a, b in zip(g, w):
        if a == b:
            continue
        try:
            fa, fb = float(a), float(b)
        except ValueError:
            return False
        if not (math.isfinite(fa) and math.isfinite(fb)) or abs(fa - fb) > rel * max(1.0, abs(fb)):
            return False
    return True


NUMBER = re.compile(r"[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?\Z")
YES_NO = ("yes", "no")


def gate_match(got, want, tol=None):
    """The fixed 200's comparison (25 Oct 2026): token by token and exact, except that YES and NO match in any letter
    case, and, only where the statement allows an error (tol, e.g. 1e-6 for "absolute or relative error does not
    exceed 10^-6"), plain decimal numbers match when |got - want| <= tol * max(1, |want|). Elsewhere numbers must be
    exact: outputs_match's 1e-6 accepts 1000000000001 for 1000000000000, and float() can't tell 64-bit values
    apart."""
    g, w = got.split(), want.split()
    if g == w:
        return True
    if len(g) != len(w):
        return False
    for a, b in zip(g, w):
        if a == b:
            continue
        if b.lower() in YES_NO and a.lower() == b.lower():
            continue
        if tol is None or not NUMBER.match(a) or not NUMBER.match(b):
            return False
        fa, fb = float(a), float(b)
        if not (math.isfinite(fa) and math.isfinite(fb)) or abs(fa - fb) > tol * max(1.0, abs(fb)):
            return False
    return True


def judge(lang, code, cases, executor, tc, case_timeout_s=CASE_TIMEOUT_S, all_cases=False, match=None):
    """Build the program once and run it on each (stdin, expected stdout) case; stops at the first failing case
    unless all_cases. match(stdin, output, expected) -> bool decides a case that ran (default outputs_match).
    -> Verdict; ok only if every case ran and matched."""
    if not (code or "").strip():
        return Verdict(False, "no_code", [], "", 0.0, 0.0)
    path_dirs = tc.path_dirs.get(lang)
    with executor.workdir() as wd:
        files, build, (image, run_args, run_env) = build_plan(lang, code, tc, wd.path)
        os.makedirs(os.path.join(wd.path, "tmp"), exist_ok=True)
        for name, text in files.items():
            wd.write(name, text)
        t0 = time.perf_counter()
        for img, args, env in build:
            r = executor.run(wd, img, args, env, b"", BUILD_TIMEOUT_S, BUILD_MEM, phase="build", path_dirs=path_dirs)
            if r.timed_out or r.returncode != 0:
                detail = ("build timed out; " if r.timed_out else "") + (r.stderr or r.stdout or b"")[-3000:].decode(
                    "utf-8", "replace")
                return Verdict(False, "compile_error", [], detail, round(time.perf_counter() - t0, 3), 0.0)
        build_s = round(time.perf_counter() - t0, 3)
        statuses, detail, t1 = [], "", time.perf_counter()
        for stdin, expected in cases:
            r = executor.run(wd, image, run_args, run_env, stdin.encode("utf-8"), case_timeout_s, RUN_MEM,
                             path_dirs=path_dirs)
            if r.timed_out:
                st = "timeout"
            elif r.returncode != 0:
                st = "runtime_error"
                detail = detail or f"exit {r.returncode}: " + (r.stderr or b"")[-1500:].decode("utf-8", "replace")
            elif (match(stdin, r.stdout.decode("utf-8", "replace"), expected) if match
                  else outputs_match(r.stdout.decode("utf-8", "replace"), expected)):
                st = "ok"
            else:
                st = "wrong_answer"
            statuses.append(st)
            if st != "ok" and not all_cases:
                break
        run_s = round(time.perf_counter() - t1, 3)
    bad = [x for x in statuses if x != "ok"]
    return Verdict(not bad and len(statuses) == len(cases), bad[0] if bad else "ok", statuses, detail, build_s, run_s)
