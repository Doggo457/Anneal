"""NOT USED YET (parked 20 Oct 2026). I decided not to pay for a teacher API, so the loop stays on the public
AI-written sets and nothing imports this module. It is kept, smoke-tested but unfinished (no CLI, no teacher client,
no build step, no unit tests yet), in case that changes. See docs/hf-dataset-search.md sections 3, 7 and 8.

Human-written answers with teacher-written reasoning: training rows whose final answer is always a person's work
(an accepted contest solution, a merged patch, a JUnit test, a review comment, a commit message, a Stack Overflow
answer) and whose <think> trace is written by an allowed teacher model. A driver script (collect -> teacher ->
build) was planned but not written.

Why this shape (docs/hf-dataset-search.md, sections 3 and 7):
  - LoRA training on answers without reasoning took Qwen3-8B's valid-reasoning rate on the trained task to 0% [E5],
    so every row carries a trace.
  - Reasoning written toward a known answer trains worse (78.9% against 95.1% on MATH-500); asking the teacher to
    derive first recovers about two thirds of that, and deleting early statements of the answer 83% [E12]. The
    teacher prompt below asks it to derive first, and check_trace() rejects traces that mention the answer, quote it
    early or copy it.
  - The row's answer is the human text byte for byte (assembled here, checked again in the build step), and for
    contest problems that exact code has passed the problem's tests in our runner before any teacher call is paid for.

Kinds and the user turn the student sees (the answer is always the human's):
  cp       problem statement + the stdin/stdout instruction for the language (Python's is generate_traces.py's, the
           gate's own wording)                                             -> accepted program (Java, Python 3, C++)
  bugfix   issue + the code around each hunk before the fix               -> the merged patch (unified diff)
  tests    a Java class (method under test in full, the rest as signatures) -> the JUnit test class
  review   a diff hunk                                                     -> the reviewer's comment
  commit   a diff                                                          -> the commit message
  explain  a Stack Overflow question                                       -> the accepted (or best-scored) answer

Nothing here makes a network call.
"""

import collections
import datetime
import hashlib
import json
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
from html.parser import HTMLParser
from urllib.parse import urlparse

from anneal.evalsets import IO_CASE_TIMEOUT_S, MAX_CASE_BYTES, MAX_IO_CASES, cc_cases, word_ngrams, word_tokens

PROMPT_VERSION = "hr-1"            # bump when a teacher or judge prompt changes; traces are keyed by it
HUMAN_CUTOFF = "2022-11-30"        # ChatGPT's release: content dated before it is very unlikely to be AI-written
GATE_CF_CONTESTS = (1547, 1624)    # the gate's CodeContests problems are Codeforces contests 1548-1623, +-1 for
                                   # parallel Div. 1 / Div. 2 rounds (docs/hf-dataset-search.md section 5)
LCB_CUTOFF_UNIX = 1785542400       # 2026-08-01 00:00 UTC: LiveCodeBench problems we report start here
EVAL_REPOS = frozenset({           # repositories of benchmarks we may report (SWE-bench Multilingual, Multi-SWE-bench)
    "apache/druid", "apache/lucene", "alibaba/fastjson2", "mockito/mockito", "elastic/logstash"})
TEST_MEM_BYTES = 4 << 30

STDIO_SUFFIX = ("\n\nWrite a complete Python 3 program that reads the input from standard input and writes "
                "the answer to standard output. Put the whole program in a single ```python code block.")
CP_LANGS = {
    "python": {"fence": "python", "name": "Python 3", "suffix": STDIO_SUFFIX},
    "java": {"fence": "java", "name": "Java", "suffix": (
        "\n\nWrite a complete Java program that reads the input from standard input and writes the answer to "
        "standard output. Put the whole program in a single ```java code block.")},
    "cpp": {"fence": "cpp", "name": "C++", "suffix": (
        "\n\nWrite a complete C++17 program that reads the input from standard input and writes the answer to "
        "standard output. Put the whole program in a single ```cpp code block.")},
}
KINDS = {  # words of reasoning asked for: at least / at most (the row's token budget can lower the maximum)
    "cp": {"min_words": 120, "max_words": 1800, "code": True},
    "bugfix": {"min_words": 100, "max_words": 1500, "code": True},
    "tests": {"min_words": 60, "max_words": 800, "code": True},
    "review": {"min_words": 40, "max_words": 450, "code": False},
    "commit": {"min_words": 30, "max_words": 350, "code": False},
    "explain": {"min_words": 60, "max_words": 900, "code": False},
}
LANG_ALIASES = {
    "py": "python", "python3": "python", "python 3": "python", "kt": "kotlin", "kts": "kotlin", "js": "javascript",
    "ts": "typescript", "golang": "go", "rs": "rust", "c++": "cpp", "cc": "cpp", "cxx": "cpp", "cs": "csharp",
    ".cs": "csharp", "c#": "csharp", "c-sharp": "csharp", "rb": "ruby", "sh": "shell", "bash": "shell"}
LANG_NAMES = {
    "python": "Python", "java": "Java", "kotlin": "Kotlin", "javascript": "JavaScript", "typescript": "TypeScript",
    "go": "Go", "rust": "Rust", "cpp": "C++", "c": "C", "csharp": "C#", "ruby": "Ruby", "php": "PHP",
    "scala": "Scala", "swift": "Swift", "dart": "Dart", "shell": "shell"}
EXT_FENCE = {
    ".py": "python", ".java": "java", ".kt": "kotlin", ".kts": "kotlin", ".js": "javascript", ".jsx": "jsx",
    ".mjs": "javascript", ".ts": "typescript", ".tsx": "tsx", ".go": "go", ".rs": "rust", ".c": "c", ".h": "c",
    ".cc": "cpp", ".cpp": "cpp", ".cxx": "cpp", ".hpp": "cpp", ".hh": "cpp", ".cs": "csharp", ".rb": "ruby",
    ".php": "php", ".scala": "scala", ".swift": "swift", ".dart": "dart", ".sh": "bash", ".yml": "yaml",
    ".yaml": "yaml", ".json": "json", ".xml": "xml", ".md": "markdown", ".gradle": "groovy", ".groovy": "groovy",
    ".sql": "sql", ".html": "html", ".css": "css", ".scss": "scss", ".toml": "toml", ".proto": "protobuf"}


# ============================================================================================ small helpers

def sha1(text):
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def lang_key(value):
    v = str(value or "").strip().lower()
    return LANG_ALIASES.get(v, v)


def fence(text, lang=""):
    """A fenced block whose fence is longer than any backtick run inside the text."""
    longest = max((len(m) for m in re.findall(r"`+", text)), default=0)
    f = "`" * max(3, longest + 1)
    return f"{f}{lang}\n{text}\n{f}"


def norm_code(code):
    """CRLF to LF, trailing spaces off each line, blank lines off both ends. Tests run on this exact text, and it is
    what the row trains on."""
    lines = (code or "").replace("\r\n", "\n").replace("\r", "\n").split("\n")
    lines = [ln.rstrip() for ln in lines]
    while lines and not lines[0]:
        lines.pop(0)
    while lines and not lines[-1]:
        lines.pop()
    return "\n".join(lines)


def parse_date(value):
    """datetime.date from the formats these sets use (dd.mm.yyyy hh:mm:ss, yyyy/mm/dd, ISO, Unix seconds or
    milliseconds, datetime objects), else None."""
    if value is None or value == "":
        return None
    if isinstance(value, datetime.datetime):
        return value.date()
    if isinstance(value, datetime.date):
        return value
    if isinstance(value, (int, float)):
        secs = value / 1000 if value > 1e11 else value
        return datetime.datetime.fromtimestamp(secs, tz=datetime.timezone.utc).date()
    s = str(value).strip()
    try:
        return datetime.datetime.fromisoformat(s.replace("Z", "+00:00")).date()
    except ValueError:
        pass
    for fmt in ("%d.%m.%Y %H:%M:%S", "%d.%m.%Y", "%Y/%m/%d %H:%M:%S", "%Y/%m/%d"):
        try:
            return datetime.datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    return None


def as_date(value):
    """--before values: None / "none" / "" -> None, else a date."""
    if value in (None, "", "none", "None"):
        return None
    d = parse_date(value)
    if d is None:
        raise ValueError(f"not a date: {value!r}")
    return d


_LIC_ALLOW = re.compile(
    r"\bmit\b|\bapache\b|\bbsd\b|\b0bsd\b|\bisc\b|\bunlicen[sc]e\b|\bcc0\b|\bzlib\b|\bmpl\b|mozilla public|"
    r"\bbsl-1\.0\b|boost software|\bpsf\b|python software foundation|\bartistic\b|\bwtfpl\b|"
    r"\bcc[- ]by[- ]4\.0\b|creative commons attribution 4\.0", re.I)
_LIC_BLOCK = re.compile(
    r"gpl|general public|\bnc\b|non-?commercial|proprietary|\bunknown\b|^\s*other\s*$|no ?licen[sc]e|\bsspl\b|"
    r"commons clause|\bbusl\b|business source|elastic licen[sc]e|share-?alike|\bsa\b", re.I)


def licence_ok(value, allow_any=False):
    """Permissive licences only (MIT, Apache, BSD, ISC, ...); GPL-family, share-alike, non-commercial, unknown or
    missing licences are refused. allow_any skips the check (the set's own licence then covers the row)."""
    if allow_any:
        return True
    s = str(value or "").strip()
    return bool(s) and not _LIC_BLOCK.search(s) and bool(_LIC_ALLOW.search(s))


def slug(text, n=48):
    s = re.sub(r"[^A-Za-z0-9]+", "-", str(text or "")).strip("-").lower()
    return s[:n] or "x"


class _HTMLText(HTMLParser):
    """Stack Exchange HTML -> Markdown-ish text: <pre> blocks become fenced code (kept verbatim), inline <code>
    becomes `code`, paragraphs and list items keep their breaks, links keep their text."""

    BLOCK = {"p", "div", "blockquote", "ul", "ol", "table", "tr", "h1", "h2", "h3", "h4", "h5", "h6"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts, self.cur, self.pre, self.in_pre, self.in_code = [], [], [], 0, 0

    def handle_starttag(self, tag, attrs):
        if tag == "pre":
            self.in_pre += 1
            self.pre = []
        elif self.in_pre:
            return
        elif tag == "code":
            self.in_code += 1
            self.cur.append("`")
        elif tag == "br":
            self.cur.append("\n")
        elif tag == "li":
            self.cur.append("\n- ")
        elif tag == "hr":
            self.cur.append("\n\n---\n\n")
        elif tag == "img":
            self.cur.append(f"[{dict(attrs).get('alt') or 'image'}]")
        elif tag in self.BLOCK:
            self.cur.append("\n\n")

    def handle_endtag(self, tag):
        if tag == "pre" and self.in_pre:
            self.in_pre -= 1
            self.parts.append(("text", "".join(self.cur)))
            self.cur = []
            self.parts.append(("code", fence("".join(self.pre).strip("\n"))))
        elif self.in_pre:
            return
        elif tag == "code" and self.in_code:
            self.in_code -= 1
            self.cur.append("`")
        elif tag in self.BLOCK:
            self.cur.append("\n\n")

    def handle_data(self, data):
        if self.in_pre:
            self.pre.append(data)
        else:
            self.cur.append(re.sub(r"\s+", " ", data))

    def text(self):
        out = []
        for kind, s in self.parts + [("text", "".join(self.cur))]:
            if kind == "code":
                out.append(s)
                continue
            s = re.sub(r"[ \t]*\n[ \t]*", "\n", s)        # spaces around line breaks
            s = re.sub(r"\n{3,}", "\n\n", s).strip()
            if s:
                out.append(s)
        return "\n\n".join(out)


def html_to_text(markup):
    p = _HTMLText()
    p.feed(markup or "")
    p.close()
    return p.text()


# ============================================================================================ reading files

def iter_records(paths, columns=None, batch_size=32):
    """Dict rows from .parquet (only `columns`, when given, that exist), .jsonl / .json (JSON lines or one array)
    and .gz versions of those."""
    for p in paths:
        if p.endswith(".parquet"):
            import pyarrow.parquet as pq
            pf = pq.ParquetFile(p)
            names = pf.schema_arrow.names
            cols = [c for c in columns if c in names] if columns else None
            for batch in pf.iter_batches(batch_size=batch_size, columns=cols):
                yield from batch.to_pylist()
            continue
        opener = open
        if p.endswith(".gz"):
            import gzip
            opener = gzip.open
        with opener(p, "rt", encoding="utf-8") as fh:
            first = fh.read(1)
            while first and first.isspace():
                first = fh.read(1)
            if first == "[":
                yield from json.loads(first + fh.read())
                continue
            line0 = first + fh.readline()
            if line0.strip():
                yield json.loads(line0)
            for line in fh:
                if line.strip():
                    yield json.loads(line)


# ============================================================================================ loaders

class Opts:
    """Loader options; every field has a default so tests can pass only what they need."""

    def __init__(self, **kw):
        self.languages = None            # set of language keys, None = all
        self.before = None               # datetime.date: keep only content dated before it
        self.per_problem = 1             # cp: accepted solutions per problem and language
        self.max_tries = 3               # cp: solutions tried per problem and language before giving up
        self.max_code_chars = 8000
        self.max_statement_chars = 12000
        self.max_patch_chars = 8000
        self.max_files = 4               # bugfix / commit: files in one change
        self.min_answer_chars = 0        # 0 = the kind's default
        self.max_answer_chars = 6000
        self.exclude_contests = GATE_CF_CONTESTS
        self.lcb_cutoff = LCB_CUTOFF_UNIX
        self.exclude_repos = set(EVAL_REPOS)
        self.sites = ("stackoverflow.com",)
        self.min_score = 2               # explain: the chosen answer's pm_score
        self.require_code = True         # explain: the question or answer has a code block
        self.grep = None                 # compiled regex the user text must match
        self.any_licence = False
        self.seed = 42
        for k, v in kw.items():
            if not hasattr(self, k):
                raise TypeError(f"unknown option {k}")
            setattr(self, k, v)

    def lang_ok(self, lang):
        return not self.languages or lang in self.languages

    def date_ok(self, d):
        return self.before is None or (d is not None and d < self.before)


def _cand(base, **kw):
    """A candidate: an item still missing "answer"/"ref" when it carries "solutions" + "cases" (cp), else complete."""
    meta = dict(base)
    meta.update(kw.pop("meta", {}))
    c = {"tests": {"status": "not_run"}, "hint": None}
    c.update(kw)
    c["meta"] = meta
    return c


def _seeded(seq, *key):
    out = list(dict.fromkeys(seq))  # drop exact duplicates, keep order
    random.Random("/".join(str(k) for k in key)).shuffle(out)
    return out


CC_LANG_IDS = {3: "python", 4: "java", 2: "cpp"}  # code_contests: 1 PYTHON2 (left out), 2 CPP, 3 PYTHON3, 4 JAVA


def load_code_contests(records, o, stats, base):
    """deepmind/code_contests train: a candidate per problem and language, with up to max_tries accepted solutions
    to test and the problem's own tests (public, private, generated; evalsets.cc_cases)."""
    for r in records:
        stats["records"] += 1
        name = str(r.get("name") or "")
        cid = r.get("cf_contest_id")
        if cid and o.exclude_contests and o.exclude_contests[0] <= int(cid) <= o.exclude_contests[1]:
            stats["gate_contest"] += 1
            continue
        if (r.get("input_file") or "").strip() or (r.get("output_file") or "").strip():
            stats["file_io"] += 1
            continue
        desc = (r.get("description") or "").strip()
        if not desc or len(desc) > o.max_statement_chars:
            stats["bad_statement"] += 1
            continue
        cases = cc_cases(r)
        if not cases:
            stats["no_tests"] += 1
            continue
        sols = r.get("solutions") or {}
        by_lang = collections.defaultdict(list)
        for lang_id, code in zip(sols.get("language") or [], sols.get("solution") or []):
            lang = CC_LANG_IDS.get(lang_id)
            if lang and o.lang_ok(lang) and code:
                code = norm_code(code)
                if code and len(code) <= o.max_code_chars:
                    by_lang[lang].append(code)
        if not by_lang:
            stats["no_solution_in_languages"] += 1
            continue
        key = f"{slug(name)}-{sha1(name)[:8]}"
        url = (f"https://codeforces.com/contest/{cid}/problem/{r.get('cf_index')}" if cid and r.get("cf_index")
               else None)
        for lang in sorted(by_lang):
            yield _cand(base, id=f"cc/{key}/{lang}", group=f"cc/{key}", source="code_contests", kind="cp",
                        lang=lang, user=desc + CP_LANGS[lang]["suffix"],
                        solutions=_seeded(by_lang[lang], o.seed, name, lang)[:o.max_tries], cases=cases,
                        meta={"name": name, "url": url, "cf_rating": r.get("cf_rating")})


def cf_lang(name):
    """Codeforces language name -> python / java / cpp, or None (Python 2, Kotlin, C, ...)."""
    n = str(name or "").lower()
    if "java" in n and "javascript" not in n:
        return "java"
    if re.search(r"(python|pypy)\s*3", n):
        return "python"
    if "c++" in n or "g++" in n:
        return "cpp"
    return None


def cf_statement(p):
    """open-r1/codeforces fields -> a statement laid out like code_contests descriptions."""
    parts = [(p.get("description") or "").strip()]
    if (p.get("input_format") or "").strip():
        parts += ["Input", p["input_format"].strip()]
    if (p.get("output_format") or "").strip():
        parts += ["Output", p["output_format"].strip()]
    ex = p.get("examples") or []
    if ex:
        parts.append("Examples" if len(ex) > 1 else "Example")
        for e in ex:
            parts += ["Input", (e.get("input") or "").rstrip("\n"), "Output", (e.get("output") or "").rstrip("\n")]
    if (p.get("note") or "").strip():
        parts += ["Note", p["note"].strip()]
    return "\n\n".join(x for x in parts if x is not None).strip()


def _cases(pairs):
    out, seen = [], set()
    for t in pairs or []:
        i, o_ = t.get("input"), t.get("output")
        if not isinstance(i, str) or not isinstance(o_, str):
            continue
        if len(i) > MAX_CASE_BYTES or len(o_) > MAX_CASE_BYTES or (i, o_) in seen:
            continue
        seen.add((i, o_))
        out.append((i, o_))
        if len(out) >= MAX_IO_CASES:
            break
    return out


def load_codeforces(problems, submissions, o, stats, base, sub_meta=None):
    """open-r1/codeforces (train) joined with open-r1/codeforces-submissions on the problem id. The editorial, when
    there is one, goes to the teacher as a hint (the student never sees it). Submissions dated on or after
    o.before are left out (AI-written code), so are interactive problems, problems needing a special checker,
    problems from the gate's contests and problems from LiveCodeBench's window."""
    subs = collections.defaultdict(lambda: collections.defaultdict(list))
    for s in submissions:
        stats["submissions"] += 1
        if str(s.get("verdict") or "OK") != "OK":
            stats["sub_not_ok"] += 1
            continue
        lang = cf_lang(s.get("programmingLanguage"))
        if not lang or not o.lang_ok(lang):
            stats["sub_language"] += 1
            continue
        if not o.date_ok(parse_date(s.get("creationTimeSeconds"))):
            stats["sub_date"] += 1
            continue
        code = norm_code(s.get("source") or "")
        if not code or len(code) > o.max_code_chars:
            stats["sub_size"] += 1
            continue
        pid = s.get("problem_id") or f"{s.get('contestId')}/{s.get('problem_index')}"
        subs[pid][lang].append((code, s.get("submission_id"), s.get("contestId")))
    sub_meta = sub_meta or {}
    for p in problems:
        stats["records"] += 1
        pid = p.get("id")
        if pid not in subs:
            stats["no_submission"] += 1
            continue
        contests = []
        for x in [pid] + list(p.get("aliases") or []):
            try:
                contests.append(int(str(x).split("/")[0]))
            except ValueError:
                pass
        lo, hi = o.exclude_contests or (1, 0)
        if any(lo <= c <= hi for c in contests):
            stats["gate_contest"] += 1
            continue
        start = p.get("contest_start")
        if start is not None and o.lcb_cutoff and int(start) >= o.lcb_cutoff:
            stats["lcb_window"] += 1
            continue
        if (p.get("interaction_format") or "").strip() or (p.get("input_mode") or "stdio") != "stdio":
            stats["interactive_or_file_io"] += 1
            continue
        if (p.get("generated_checker") or "").strip():
            stats["needs_checker"] += 1
            continue
        if p.get("executable") is False:
            stats["not_executable"] += 1
            continue
        cases = _cases(p.get("official_tests")) if p.get("official_tests_complete") else []
        cases = cases or _cases(p.get("examples"))
        if not cases:
            stats["no_tests"] += 1
            continue
        stmt = cf_statement(p)
        if not stmt or len(stmt) > o.max_statement_chars:
            stats["bad_statement"] += 1
            continue
        hint = (p.get("editorial") or "").strip() or None
        if hint and len(hint) > 8000:
            hint = hint[:8000]
        key = str(pid).replace("/", "-")
        for lang in sorted(subs[pid]):
            pool = subs[pid][lang]
            order = _seeded(range(len(pool)), o.seed, pid, lang)[:o.max_tries]
            yield _cand(base, id=f"cf/{key}/{lang}", group=f"cf/{key}", source="codeforces", kind="cp", lang=lang,
                        user=stmt + CP_LANGS[lang]["suffix"], hint=hint,
                        solutions=[pool[i][0] for i in order],
                        solution_meta=[{"submission_id": pool[i][1],
                                        "url": f"https://codeforces.com/contest/{pool[i][2]}/submission/{pool[i][1]}"}
                                       for i in order],
                        cases=cases,
                        meta=dict(sub_meta, name=p.get("title"), url=f"https://codeforces.com/problemset/problem/{pid}",
                                  rating=p.get("rating"), editorial_in_hint=bool(hint)))
        del subs[pid]


def split_patch(patch):
    """Unified diff -> [{"path", "new_file", "hunks": [(old_start, [before lines])]}]."""
    files, cur, hunk = [], None, None
    for line in (patch or "").split("\n"):
        if line.startswith("diff --git "):
            cur = {"path": None, "new_file": False, "hunks": []}
            files.append(cur)
            hunk = None
            m = re.match(r"diff --git a/(.+?) b/(.+)$", line)
            if m:
                cur["path"] = m.group(2)
            continue
        if cur is None:
            if line.startswith("--- "):
                cur = {"path": None, "new_file": False, "hunks": []}
                files.append(cur)
            else:
                continue
        if line.startswith("--- ") and hunk is None:
            if line[4:].strip() == "/dev/null":
                cur["new_file"] = True
            continue
        if line.startswith("+++ ") and hunk is None:
            p = line[4:].strip()
            if p != "/dev/null":
                cur["path"] = p[2:] if p.startswith("b/") else p
            continue
        m = re.match(r"@@ -(\d+)(?:,\d+)? \+\d+(?:,\d+)? @@", line)
        if m:
            hunk = (int(m.group(1)), [])
            cur["hunks"].append(hunk)
            continue
        if hunk is not None and line[:1] in (" ", "-"):
            hunk[1].append(line[1:])
        elif hunk is not None and line == "":
            hunk[1].append("")
    return [f for f in files if f["path"]]


def before_code(files):
    """The code around each hunk as it was before the change, per file."""
    parts = []
    for f in files:
        ext = os.path.splitext(f["path"])[1].lower()
        if f["new_file"]:
            parts.append(f"`{f['path']}` does not exist yet.")
            continue
        for start, lines in f["hunks"]:
            while lines and lines[-1] == "":
                lines = lines[:-1]
            if lines:
                parts.append(f"`{f['path']}`, from line {start}:\n{fence(chr(10).join(lines), EXT_FENCE.get(ext, ''))}")
    return "\n\n".join(parts)


def load_swe_rebench(records, o, stats, base):
    """nebius/SWE-rebench-V2: issue + the code each hunk touches (rebuilt from the patch) -> the merged patch. Its
    tests need Docker, so rows rely on the set's own validation (FAIL_TO_PASS checked by its harness)."""
    for r in records:
        stats["records"] += 1
        lang = lang_key(r.get("language"))
        if not o.lang_ok(lang):
            stats["language"] += 1
            continue
        repo = str(r.get("repo") or "")
        if repo.lower() in o.exclude_repos:
            stats["eval_repo"] += 1
            continue
        if not licence_ok(r.get("license"), o.any_licence):
            stats["licence"] += 1
            continue
        if not o.date_ok(parse_date(r.get("created_at"))):
            stats["date"] += 1
            continue
        patch = (r.get("patch") or "").replace("\r\n", "\n").strip("\n")
        stmt = (r.get("problem_statement") or "").strip()
        if not patch or len(patch) > o.max_patch_chars:
            stats["patch_size"] += 1
            continue
        files = split_patch(patch)
        if not files or len(files) > o.max_files:
            stats["file_count"] += 1
            continue
        if len(stmt) < 40 or len(stmt) > o.max_statement_chars:
            stats["bad_statement"] += 1
            continue
        if not r.get("FAIL_TO_PASS"):
            stats["no_failing_test"] += 1
            continue
        user = (f"Fix the following issue in the {repo} repository.\n\nIssue:\n{stmt}\n\n"
                f"Code around the change, before the fix:\n\n{before_code(files)}\n\n"
                f"Reply with the fix as a unified diff in a single ```diff code block.")
        iid = str(r.get("instance_id") or sha1(patch)[:12])
        meta = r.get("meta") or {}
        yield _cand(base, id=f"swe/{iid}", group=f"swe/{repo.lower()}", source="swe_rebench", kind="bugfix",
                    lang=lang, user=user, answer=fence(patch, "diff"), ref=patch,
                    tests={"status": "dataset", "note": "FAIL_TO_PASS validated by SWE-rebench"},
                    meta={"repo": repo, "licence": r.get("license"), "date": str(r.get("created_at") or ""),
                          "url": (meta.get("pr_url") if isinstance(meta, dict) else None),
                          "author": (meta.get("pr_author") if isinstance(meta, dict) else None)})


_TEST_CLASS = re.compile(r"^[ \t]*(?:public\s+|final\s+|abstract\s+)*class\s+(\w+)[^\n{]*\{", re.M)


def split_methods2test(source):
    """methods2test fm+fc+c+m+f+t+tc source -> (focal class part, test class head, test class name). The source is
    the focal class (method under test in full, the rest as signatures), then the test class's fields and the
    opening line of the test method; the target finishes that method and the class."""
    at = source.rfind("@Test")
    if at < 0:
        return None
    decls = [m for m in _TEST_CLASS.finditer(source, 0, at)]
    if len(decls) < 2:
        return None
    t = decls[-1]
    focal = source[:t.start()].rstrip()
    head = source[t.start():]
    if not focal or "{" not in focal:
        return None
    return focal, head, t.group(1)


def load_methods2test(records, o, stats, base):
    """andstor/methods2test, config fm+fc+c+m+f+t+tc: a Java class -> the human JUnit test class for its method."""
    for r in records:
        stats["records"] += 1
        src, tgt = r.get("source") or "", r.get("target") or ""
        parts = split_methods2test(src.replace("\r\n", "\n"))
        if not parts or not tgt.strip():
            stats["bad_format"] += 1
            continue
        focal, head, test_cls = parts
        code = norm_code(head.rstrip() + "\n" + tgt.replace("\r\n", "\n").strip("\n"))
        if len(focal) > o.max_code_chars or len(code) > o.max_answer_chars:
            stats["size"] += 1
            continue
        user = (f"Write a JUnit test class `{test_cls}` with a test for the method whose body is shown in full in "
                f"this Java class (its other members are shown as signatures only).\n\n{fence(focal, 'java')}")
        rid = str(r.get("id") or sha1(src)[:12])
        yield _cand(base, id=f"m2t/{rid}", group=f"m2t/{rid.split('_')[0]}", source="methods2test", kind="tests",
                    lang="java", user=user, answer=fence(code, "java"), ref=code, meta={"m2t_id": rid})


_LOW_VALUE_REVIEW = re.compile(r"^\W*(nit|nits?:?\s*\w{0,10}|same( here)?|ditto|done|\+1|lgtm|ok|thanks?\W*)\W*$", re.I)


def load_codereviewer(records, o, stats, base):
    """Microsoft CodeReviewer comment generation (train): a diff hunk -> the reviewer's comment."""
    for r in records:
        stats["records"] += 1
        msg = (r.get("msg") or "").strip()
        patch = (r.get("patch") or "").replace("\r\n", "\n").strip("\n")
        lang = lang_key(r.get("lang"))
        if lang and not o.lang_ok(lang):
            stats["language"] += 1
            continue
        if str(r.get("proj") or "").lower() in o.exclude_repos:
            stats["eval_repo"] += 1
            continue
        y = r.get("y")
        if y is not None and str(y) not in ("1", "True", "true"):
            stats["no_comment_needed"] += 1
            continue
        lo = o.min_answer_chars or 20
        if len(msg) < lo or len(msg) > o.max_answer_chars or _LOW_VALUE_REVIEW.match(msg):
            stats["comment_size"] += 1
            continue
        if not patch or len(patch) > o.max_patch_chars:
            stats["patch_size"] += 1
            continue
        where = f" in the {r['proj']} project" if r.get("proj") else ""
        lname = LANG_NAMES.get(lang, lang or "")
        user = (f"Review this change to {lname + ' ' if lname else ''}code{where} and write your review comment."
                f"\n\n{fence(patch, 'diff')}")
        rid = str(r.get("id") if r.get("id") is not None else sha1(patch + msg)[:12])
        yield _cand(base, id=f"cr/{rid}", group=f"cr/{str(r.get('proj') or rid).lower()}", source="codereviewer",
                    kind="review", lang=lang or None, user=user, answer=msg, ref=msg,
                    meta={"project": r.get("proj")})


_TRAILER = re.compile(r"^(signed-off-by|co-authored-by|change-id|reviewed-by|reviewed-on|tested-by|acked-by|cc|"
                      r"reported-by|suggested-by|helped-by)\s*:", re.I)
_BOT_MSG = re.compile(r"^(merge (pull request|branch|remote-tracking|tag)|bump |update dependency|revert \"|"
                      r"initial commit|\[bot\]|chore\(deps\)|\[skip ci\]$)|dependabot|renovate\[bot\]", re.I)
_UI_MSG = re.compile(r"^(update|create|delete|add|rename)\s+[\w./-]+$", re.I)


def clean_commit_message(msg):
    lines = (msg or "").replace("\r\n", "\n").strip().split("\n")
    while lines and (not lines[-1].strip() or _TRAILER.match(lines[-1].strip())):
        lines.pop()
    return "\n".join(lines).strip()


def render_mods(mods):
    out = []
    for m in mods:
        old, new = m.get("old_path"), m.get("new_path")
        ct = str(m.get("change_type") or "").upper()
        a, b = old or new, new or old
        head = [f"diff --git a/{a} b/{b}"]
        if ct == "ADD":
            head.append("new file")
        elif ct == "DELETE":
            head.append("deleted file")
        elif ct == "RENAME" and old != new:
            head += [f"rename from {old}", f"rename to {new}"]
        out.append("\n".join(head) + "\n" + (m.get("diff") or "").replace("\r\n", "\n").strip("\n"))
    return "\n".join(out)


def load_commitchronicle(records, o, stats, base):
    """JetBrains-Research/commit-chronicle (train): a diff -> its human commit message."""
    for r in records:
        stats["records"] += 1
        lang = lang_key(r.get("language"))
        if not o.lang_ok(lang):
            stats["language"] += 1
            continue
        repo = str(r.get("repo") or "")
        if repo.lower() in o.exclude_repos:
            stats["eval_repo"] += 1
            continue
        if not licence_ok(r.get("license"), o.any_licence):
            stats["licence"] += 1
            continue
        if not o.date_ok(parse_date(r.get("date"))):
            stats["date"] += 1
            continue
        msg = clean_commit_message(r.get("message"))
        lo = o.min_answer_chars or 15
        if len(msg) < lo or len(msg) > min(o.max_answer_chars, 2000) or _BOT_MSG.search(msg) or _UI_MSG.match(msg):
            stats["message"] += 1
            continue
        mods = r.get("mods") or []
        if not mods or len(mods) > o.max_files:
            stats["file_count"] += 1
            continue
        diff = render_mods(mods)
        if len(diff) > o.max_patch_chars:
            stats["patch_size"] += 1
            continue
        user = f"Write a commit message for this change.\n\n{fence(diff, 'diff')}"
        h = str(r.get("hash") or sha1(diff)[:12])
        yield _cand(base, id=f"commit/{h}", group=f"commit/{repo.lower()}", source="commitchronicle", kind="commit",
                    lang=lang, user=user, answer=msg, ref=msg,
                    meta={"repo": repo, "licence": r.get("license"), "date": r.get("date"),
                          "url": f"https://github.com/{repo}/commit/{h}" if repo and r.get("hash") else None})


def load_stackexchange(records, o, stats, base):
    """HuggingFaceH4/stack-exchange-preferences: question -> the accepted answer (else the best-scored one), dated
    before o.before. CC BY-SA: author and links go to provenance for attribution."""
    for r in records:
        stats["records"] += 1
        meta = r.get("metadata") or []
        q_url = meta[0] if len(meta) > 0 else None
        site = meta[1] if len(meta) > 1 else None
        host = urlparse(site or q_url or "").netloc.lower()
        if host and o.sites and not any(host == s or host.endswith("." + s) for s in o.sites):
            stats["site"] += 1
            continue
        if not o.date_ok(parse_date(r.get("date"))):
            stats["date"] += 1
            continue
        answers = [a for a in (r.get("answers") or []) if isinstance(a, dict) and a.get("text")]
        if not answers:
            stats["no_answer"] += 1
            continue
        chosen = (next((a for a in answers if a.get("selected")), None)
                  or max(answers, key=lambda a: a.get("pm_score") or 0))
        if (chosen.get("pm_score") or 0) < o.min_score:
            stats["score"] += 1
            continue
        q = html_to_text(r.get("question") or "")
        a = html_to_text(chosen["text"])
        lo = o.min_answer_chars or 100
        if len(q) < 50 or len(q) > o.max_statement_chars or len(a) < lo or len(a) > o.max_answer_chars:
            stats["size"] += 1
            continue
        if o.require_code and "`" not in q + a:
            stats["no_code"] += 1
            continue
        if o.grep is not None and not o.grep.search(q):
            stats["grep"] += 1
            continue
        qid = str(r.get("qid") or sha1(q)[:12])
        yield _cand(base, id=f"se/{host or 'se'}/{qid}", group=f"se/{host}/{qid}", source="stackexchange",
                    kind="explain", lang=None, user=q, answer=a, ref=a,
                    meta={"url": q_url,
                          "answer_url": f"{site.rstrip('/')}/a/{chosen.get('answer_id')}" if site else None,
                          "author": chosen.get("author"), "author_url": chosen.get("author_profile"),
                          "date": r.get("date"), "accepted": bool(chosen.get("selected"))})


LOADERS = {"code_contests": load_code_contests, "swe_rebench": load_swe_rebench, "methods2test": load_methods2test,
           "codereviewer": load_codereviewer, "commitchronicle": load_commitchronicle,
           "stackexchange": load_stackexchange}  # codeforces has two inputs: load_codeforces

SOURCES = {  # default dataset, files and filters per source
    "code_contests": {"kind": "cp", "dataset": "deepmind/code_contests", "files": "data/train-", "before": None,
                      "languages": "java,python,cpp", "licence": "CC BY 4.0",
                      "columns": ["name", "description", "public_tests", "private_tests", "generated_tests",
                                  "solutions", "cf_contest_id", "cf_index", "cf_rating", "input_file", "output_file"]},
    "codeforces": {"kind": "cp", "dataset": "open-r1/codeforces", "files": "data/train-", "before": HUMAN_CUTOFF,
                   "submissions": ("open-r1/codeforces-submissions", "selected_accepted/"),
                   "languages": "java,python,cpp", "licence": "CC BY 4.0 (problems), ODC-By (submissions)",
                   "columns": ["id", "aliases", "contest_start", "title", "description", "input_format",
                               "output_format", "interaction_format", "note", "examples", "editorial", "rating",
                               "official_tests", "official_tests_complete", "input_mode", "generated_checker",
                               "executable"],
                   "sub_columns": ["submission_id", "source", "contestId", "problem_index", "problem_id",
                                   "programmingLanguage", "verdict", "creationTimeSeconds"]},
    "swe_rebench": {"kind": "bugfix", "dataset": "nebius/SWE-rebench-V2", "files": "data/", "before": HUMAN_CUTOFF,
                    "languages": None, "licence": "CC BY 4.0 (set) + each repository's licence",
                    "columns": ["instance_id", "repo", "language", "license", "created_at", "patch",
                                "problem_statement", "FAIL_TO_PASS", "meta"]},
    "methods2test": {"kind": "tests", "dataset": "andstor/methods2test", "files": "data/fm+fc+c+m+f+t+tc/train-",
                     "before": None, "languages": None, "licence": "MIT", "columns": ["id", "source", "target"]},
    "codereviewer": {"kind": "review", "dataset": "fasterinnerlooper/codereviewer",
                     "files": "generation/gen-train.jsonl", "before": None, "languages": None,
                     "licence": "CC BY 4.0 (Microsoft CodeReviewer)", "columns": None},
    "commitchronicle": {"kind": "commit", "dataset": "JetBrains-Research/commit-chronicle", "files": "data/train-",
                        "before": HUMAN_CUTOFF, "languages": None, "licence": "each repository's licence",
                        "columns": ["hash", "repo", "date", "license", "language", "message", "mods"]},
    "stackexchange": {"kind": "explain", "dataset": "HuggingFaceH4/stack-exchange-preferences",
                      "files": "data/Stackoverflow.com/", "before": HUMAN_CUTOFF, "languages": None,
                      "licence": "CC BY-SA 4.0", "columns": ["qid", "question", "answers", "date", "metadata"]},
}


# ============================================================================================ running human code

def outputs_match(got, want, rel=1e-6):
    """Whitespace-separated tokens equal, numbers within a relative 1e-6 (the rule generate_traces.py uses)."""
    import math
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


def _java_top_classes(code):
    """[(name, start, end)] of top-level classes, skipping comments, strings and char literals."""
    out, depth, i, n, cur = [], 0, 0, len(code), None
    while i < n:
        c = code[i]
        if code.startswith("//", i):
            j = code.find("\n", i)
            i = n if j < 0 else j
            continue
        if code.startswith("/*", i):
            j = code.find("*/", i + 2)
            i = n if j < 0 else j + 2
            continue
        if c in "\"'":
            if code.startswith('"""', i):
                j = code.find('"""', i + 3)
                i = n if j < 0 else j + 3
                continue
            j = i + 1
            while j < n and code[j] != c and code[j] != "\n":
                j += 2 if code[j] == "\\" else 1
            i = j + 1
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0 and cur is not None:
                out.append((cur[0], cur[1], i))
                cur = None
        elif (depth == 0 and cur is None and code.startswith("class", i)
              and (i == 0 or not (code[i - 1].isalnum() or code[i - 1] == "_"))):
            m = re.match(r"class\s+(\w+)", code[i:])
            if m:
                cur = (m.group(1), i)
        i += 1
    return out


def java_names(code):
    """(file name stem, class to run) for a Java program: a public top-level class sets the file name; the class to
    run is the top-level class whose body holds `static void main`."""
    m = re.search(r"\bpublic\s+(?:final\s+|abstract\s+|strictfp\s+)*class\s+(\w+)", code)
    tops = _java_top_classes(code)
    file_stem = None
    if m:
        file_stem = next((name for name, s, e in tops if s <= m.start(1) <= e), m.group(1))
    main = re.search(r"\bstatic\s+(?:final\s+)?(?:public\s+)?void\s+main\s*\(", code)
    run = None
    if main:
        run = next((name for name, s, e in tops if s <= main.start() <= e), None)
    pkg = re.search(r"^\s*package\s+([\w.]+)\s*;", code, re.M)
    run = run or file_stem or "Main"
    return file_stem or "Main", (pkg.group(1) + "." + run if pkg else run)


class Runner:
    """Runs a human solution on a problem's stdin/stdout cases. Python goes through anneal.sandbox when it is set up
    (Windows), else a plain subprocess; Java (javac/java) and C++ (g++) only with native=True and the tools on PATH.
    These are accepted human submissions from public sets, not model output; each run gets a fresh temp folder and
    a timeout (Java 2x, for JVM start-up)."""

    def __init__(self, native=False, timeout_s=IO_CASE_TIMEOUT_S, tmp_root=None):
        self.native, self.timeout_s, self.tmp_root = native, timeout_s, tmp_root
        self.sandbox = None
        if os.name == "nt":
            try:
                from anneal import sandbox
                if sandbox.available():
                    self.sandbox = sandbox
            except Exception:  # noqa: BLE001
                self.sandbox = None
        self.javac, self.java, self.gxx = shutil.which("javac"), shutil.which("java"), shutil.which("g++")

    def can_run(self, lang):
        if lang == "python":
            return True
        if lang == "java":
            return bool(self.native and self.javac and self.java)
        if lang == "cpp":
            return bool(self.native and self.gxx)
        return False

    def describe(self):
        return {"python": "sandbox" if self.sandbox else f"subprocess ({os.path.basename(sys.executable)})",
                "java": "javac/java" if self.can_run("java") else "off",
                "cpp": "g++ -O2 -std=gnu++17" if self.can_run("cpp") else "off"}

    @staticmethod
    def _limit_memory():  # POSIX only, for the Python fallback
        try:
            import resource
            resource.setrlimit(resource.RLIMIT_AS, (TEST_MEM_BYTES, TEST_MEM_BYTES))
        except Exception:  # noqa: BLE001
            pass

    def _run(self, argv, stdin, cwd, timeout, limit=False):
        try:
            r = subprocess.run(argv, input=stdin.encode("utf-8"), stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                               cwd=cwd, timeout=timeout,
                               preexec_fn=self._limit_memory if (limit and os.name == "posix") else None)
        except subprocess.TimeoutExpired:
            return "timeout", ""
        if r.returncode != 0:
            return "runtime_error", ""
        return "ok", r.stdout.decode("utf-8", "replace")

    def check(self, lang, code, cases):
        """(True, "passed") or (False, reason): compile_error, runtime_error, timeout, wrong_answer, cannot_run."""
        if not self.can_run(lang):
            return False, "cannot_run"
        if lang == "python" and self.sandbox is not None:
            for stdin, want in cases:
                r = self.sandbox.run(["candidate.py"], stdin=stdin.encode("utf-8"), files={"candidate.py": code + "\n"},
                                     timeout_s=self.timeout_s, mem_bytes=TEST_MEM_BYTES)
                if r.timed_out:
                    return False, "timeout"
                if r.returncode != 0:
                    return False, "runtime_error"
                if not outputs_match(r.stdout.decode("utf-8", "replace"), want):
                    return False, "wrong_answer"
            return True, "passed"
        with tempfile.TemporaryDirectory(prefix="hr-", dir=self.tmp_root) as d:
            timeout, limit = self.timeout_s, False
            if lang == "python":
                with open(os.path.join(d, "main.py"), "w", encoding="utf-8") as fh:
                    fh.write(code + "\n")
                argv, limit = [sys.executable, "-I", "-X", "utf8", "main.py"], True
            elif lang == "java":
                stem, main = java_names(code)
                with open(os.path.join(d, stem + ".java"), "w", encoding="utf-8") as fh:
                    fh.write(code + "\n")
                try:
                    c = subprocess.run([self.javac, "-encoding", "UTF-8", "-nowarn", "-d", "out", stem + ".java"],
                                       cwd=d, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=120)
                except subprocess.TimeoutExpired:
                    return False, "compile_error"
                if c.returncode != 0:
                    return False, "compile_error"
                argv, timeout = [self.java, "-Xmx2g", "-Xss256m", "-XX:+UseSerialGC", "-cp", "out", main], 2 * timeout
            else:
                with open(os.path.join(d, "main.cpp"), "w", encoding="utf-8") as fh:
                    fh.write(code + "\n")
                exe = os.path.join(d, "prog.exe" if os.name == "nt" else "prog")
                try:
                    c = subprocess.run([self.gxx, "-O2", "-std=gnu++17", "-pipe", "-o", exe, "main.cpp"], cwd=d,
                                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=180)
                except subprocess.TimeoutExpired:
                    return False, "compile_error"
                if c.returncode != 0:
                    return False, "compile_error"
                argv = [exe]
            for stdin, want in cases:
                status, out = self._run(argv, stdin, d, timeout, limit)
                if status != "ok":
                    return False, status
                if not outputs_match(out, want):
                    return False, "wrong_answer"
        return True, "passed"


def resolve_cp(cand, runner, run_tests=True):
    """A cp candidate -> (items, reason): the first of its solutions (seeded order) that passes every case becomes
    the item's answer. run_tests=False trusts the set's own verdict (accepted solutions) and takes the first."""
    tried = []
    for k, code in enumerate(cand["solutions"]):
        if run_tests:
            ok, why = runner.check(cand["lang"], code, cand["cases"])
            tried.append(why)
            if why == "cannot_run":
                return [], "cannot_run"
            if not ok:
                continue
            tests = {"status": "passed", "cases": len(cand["cases"]), "tried": k + 1}
        else:
            tests = {"status": "dataset", "note": "accepted solution, not re-run"}
        item = {k2: v for k2, v in cand.items() if k2 not in ("solutions", "cases", "solution_meta")}
        item["ref"] = code
        item["answer"] = fence(code, CP_LANGS[cand["lang"]]["fence"])
        item["tests"] = tests
        if cand.get("solution_meta"):
            item["meta"] = dict(item["meta"], **cand["solution_meta"][k])
        return [item], "ok"
    return [], ("failed_tests:" + ",".join(sorted(set(tried)))) if tried else "no_solution"


# ============================================================================================ teacher prompts

TEACHER_SYSTEM = """You write the private thinking a strong software engineer goes through before answering a task.

You get a <task> and the <final_answer> a human expert gave to it. The final answer will be shown right after your thinking, unchanged. Write the thinking that leads from the task to that answer, as if you were working it out yourself for the first time.

Rules:
1. Derive, don't recite. Start from the task: work out what is really being asked, try ideas, test them against the details and examples in the task, and notice and fix your own mistakes. Settle on the specific approach and decisions the final answer uses only towards the end, and only because your reasoning leads there.
2. While you think, the final answer does not exist yet. Never mention or hint at a given, provided, reference, existing, human or accepted solution, answer, code, patch, test or comment, and never state the answer before you have derived it.
3. Do not write the final answer out. No code blocks; at most a short expression or a single line of code at a time. The final answer follows your thinking.
4. Agree with every decision in the final answer: the same approach, data structures, edge cases, output format and points made. If the final answer makes an unusual choice, reach it for a reason.
5. Write in the first person, in plain text, as private notes. Inline `names` are fine; no headings, no code blocks, no greeting, no closing summary, nothing about these rules.
6. If an <editorial> is given, it is a human explanation of the intended solution: follow its idea in your own words and never mention it.
7. If the final answer does not actually do what the task needs, or is too unclear to reason towards, reply with exactly "REJECT: " and one short reason, and nothing else.

Reply with the thinking only."""

KIND_GUIDE = {
    "cp": ("This task is a competitive programming problem, and the final answer is a complete program in the language "
           "the task asks for, reading standard input. Good thinking here: restate what is asked in your own words, read "
           "the constraints and work out what time complexity they allow, find the key observation (try small cases and "
           "the samples), pick the algorithm and check it by hand on a sample, consider edge cases (smallest sizes, equal "
           "values, overflow, large input), and finally plan the program's structure and input/output handling the way "
           "the final program does it."),
    "bugfix": ("This task is a bug report with the code around the change, and the final answer is the patch that fixed "
               "it. Good thinking here: work out the expected and the actual behaviour, trace through the shown code to "
               "where they diverge, find the root cause rather than the symptom, weigh which fix is safest and what else "
               "it could affect, and settle on the change the patch makes, file by file."),
    "tests": ("This task is a Java class whose method under test is shown in full, and the final answer is a JUnit test "
              "class for that method. Good thinking here: work out what the method does and its contract, choose the "
              "behaviour worth checking (normal case, boundaries, errors), decide inputs, expected results and any setup "
              "the class needs, and settle on the test name, cases and assertions the final test uses."),
    "review": ("This task is a change (a diff hunk) under code review, and the final answer is the reviewer's comment on "
               "it. Good thinking here: read what the change does and why it might have been made, look for problems "
               "(correctness, edge cases, naming, readability, performance, API use, consistency with the surrounding "
               "code), weigh which one matters most, and settle on the point the comment makes and its tone."),
    "commit": ("This task is a code change (a diff), and the final answer is its commit message. Good thinking here: work "
               "out what changed in each file, find the purpose that ties the changes together, decide what a reader of "
               "the history needs to know, and settle on the wording and conventions the message uses."),
    "explain": ("This task is a programming question from Stack Overflow, and the final answer is the accepted (or best "
                "voted) answer. Good thinking here: work out what the asker is trying to do and where their understanding "
                "or code goes wrong, reason about the underlying cause, consider possible fixes and their trade-offs, and "
                "settle on the explanation and solution the answer gives."),
}


def teacher_messages(item):
    """System prompt (static per kind, so the API's prefix cache covers it) + the item."""
    parts = [f"<task>\n{item['user']}\n</task>"]
    if item.get("hint"):
        parts.append(f"<editorial>\n{item['hint']}\n</editorial>")
    parts.append(f"<final_answer>\n{item['answer']}\n</final_answer>")
    parts.append(f"Write the thinking now, at most {item['max_words']} words.")
    return [{"role": "system", "content": TEACHER_SYSTEM + "\n\n" + KIND_GUIDE[item["kind"]]},
            {"role": "user", "content": "\n\n".join(parts)}]


JUDGE_SYSTEM = "You check training data for a reasoning model. Reply with one JSON object and nothing else."


def judge_messages(item, trace):
    user = (f"<task>\n{item['user']}\n</task>\n\n<thinking>\n{trace}\n</thinking>\n\n"
            f"<final_answer>\n{item['answer']}\n</final_answer>\n\n"
            "The thinking was written to come right before the final answer. Answer three questions.\n"
            "consistent: does the thinking arrive at the approach and key decisions the final answer uses, with nothing "
            "in it contradicting the final answer?\n"
            "leaks: does the thinking read as if it had already seen the final answer (it refers to a given, provided or "
            "reference answer, or states the answer's specifics before working towards them)?\n"
            "sound: is the thinking free of clear technical errors?\n"
            'Reply as {"consistent": true or false, "leaks": true or false, "sound": true or false, '
            '"issue": "one short sentence"}')
    return [{"role": "system", "content": JUDGE_SYSTEM}, {"role": "user", "content": user}]


def parse_judge(text):
    """The judge's JSON verdict, or None if it can't be read."""
    s = (text or "").strip()
    a, b = s.find("{"), s.rfind("}")
    if a < 0 or b <= a:
        return None
    try:
        v = json.loads(s[a:b + 1])
    except ValueError:
        return None
    return v if isinstance(v, dict) else None


def judge_ok(v):
    return isinstance(v, dict) and v.get("consistent") is True and v.get("leaks") is False and v.get("sound") is True


# ============================================================================================ checking a trace

_NOUNS = r"(solution|answer|submission|patch|diff|fix|test class|unit test|review comment|commit message|reply)"
_ANCHOR = re.compile(
    r"\b(the|a|this|that|our)\s+(given|provided|reference|supplied|attached|human|accepted|official|expert'?s?|"
    r"author'?s)\s+" + _NOUNS + r"\b|"
    r"\breference\s+(solution|answer|code|implementation|program|patch)\b|"
    r"\b(answer|solution|code|patch)\s+(above|below)\b|"
    r"\byou\s+(provided|gave|supplied)\b|\bwe\s+(were|are)\s+given\s+(the|a)\s+" + _NOUNS + r"|"
    r"\b(i|i'm|i am)\s+(given|shown|provided)\s+(the|a|an)\s+" + _NOUNS, re.I)
_ANCHOR_CODE = re.compile(  # a cp task shows no code, so "the given code" can only mean the answer
    r"\b(the|this|that)\s+(given|provided|reference|supplied|attached|accepted|existing|final)\s+"
    r"(code|program|implementation)\b", re.I)
_EDITORIAL = re.compile(r"\beditorial\b|\btutorial\b", re.I)
_META = re.compile(r"\b(word limit|these rules|the rules say|as instructed|my instructions|the instructions say|"
                   r"the prompt says)\b|final_answer|</?task>", re.I)
_AI = re.compile(r"\bas an ai\b|\blanguage model\b|\bi cannot (help|assist)\b", re.I)


def answer_overlap(ref, trace, user="", n=12):
    """(share of the answer's word 12-grams found in the trace, position of the first one as a fraction of the
    trace, or None). 12-grams the task itself contains don't count (a patch's context lines, code an answer quotes
    from the question). Short answers (under 2n tokens) aren't scored."""
    a = word_tokens(ref)
    if len(a) < 2 * n:
        return 0.0, None
    grams = word_ngrams(a, n) - word_ngrams(word_tokens(user), n)
    if not grams:
        return 0.0, None
    t = word_tokens(trace)
    hits, first = set(), None
    for i in range(len(t) - n + 1):
        g = tuple(t[i:i + n])
        if g in grams:
            hits.add(g)
            if first is None:
                first = i / max(1, len(t))
    return len(hits) / len(grams), first


def copied_lines(ref, trace, user="", min_chars=25):
    """Non-trivial answer lines (min_chars+ after stripping, not in the task) that appear verbatim in the trace."""
    lines = {ln.strip() for ln in ref.split("\n") if len(ln.strip()) >= min_chars and ln.strip() not in user}
    return sum(1 for ln in lines if ln in trace)


def check_trace(item, trace, finish_reason="stop"):
    """Reasons to reject a teacher trace (empty list = keep): teacher_rejected, truncated, think_tags, too_short,
    too_long, code_block, mentions_answer, meta, ai_disclaimer, early_answer, copies_answer."""
    t = (trace or "").strip()
    if t.upper().startswith("REJECT"):
        return ["teacher_rejected"]
    if finish_reason not in (None, "stop"):
        return ["truncated"]
    kind = KINDS[item["kind"]]
    reasons = []
    if "<think>" in t or "</think>" in t:
        reasons.append("think_tags")
    words = len(t.split())
    if words < kind["min_words"]:
        reasons.append("too_short")
    if words > 1.25 * item.get("max_words", kind["max_words"]) + 20:
        reasons.append("too_long")
    if "```" in t:
        reasons.append("code_block")
    if (_ANCHOR.search(t) or (item["kind"] == "cp" and _ANCHOR_CODE.search(t))
            or (item.get("hint") and _EDITORIAL.search(t))):
        reasons.append("mentions_answer")
    if _META.search(t):
        reasons.append("meta")
    if _AI.search(t):
        reasons.append("ai_disclaimer")
    share, first = answer_overlap(item["ref"], t, item["user"])
    if first is not None and first < 0.5:
        reasons.append("early_answer")
    if share > 0.25 or (kind["code"] and copied_lines(item["ref"], t, item["user"]) >= 3):
        reasons.append("copies_answer")
    return reasons


# ============================================================================================ rows

def assistant_content(trace, answer):
    return f"<think>\n{trace.strip()}\n</think>\n\n{answer}"


def make_row(item, trace):
    return {"messages": [{"role": "user", "content": item["user"]},
                         {"role": "assistant", "content": assistant_content(trace, item["answer"])}]}


def row_answer(row):
    """The text after the think block: must equal the item's human answer."""
    c = row["messages"][-1]["content"]
    head, sep, tail = c.partition("\n</think>\n\n")
    return tail if sep and head.startswith("<think>\n") else None


def answer_is_human(item, row):
    """The row ends on the human answer byte for byte, and for code kinds that answer is exactly the fenced human
    code the tests ran on."""
    ans = row_answer(row)
    if ans != item["answer"]:
        return False
    if item["kind"] == "cp":
        return ans == fence(item["ref"], CP_LANGS[item["lang"]]["fence"])
    return item["ref"] in ans


def item_key(item, teacher, model):
    """What a trace was written for: same key = same prompt, same teacher, same item."""
    body = item["user"] + "\x00" + item["answer"] + "\x00" + (item.get("hint") or "") + "\x00" + str(item.get("max_words"))
    return sha1(f"{teacher}|{model}|{PROMPT_VERSION}|{item['id']}|{sha1(body)}")


def estimate_tokens(text):
    """Conservative token estimate without a tokenizer: 3 characters per token (code often runs near 3, prose near 4)."""
    return (len(text) + 2) // 3


def trace_word_budget(item, fixed_tokens, max_tokens):
    """Words of reasoning that still fit: the row's token cap minus the user turn and answer, at 1.7 tokens per word
    (prose with identifiers), capped by the kind's maximum. 0 if not even 1.5x the kind's minimum fits."""
    kind = KINDS[item["kind"]]
    words = min(kind["max_words"], int((max_tokens - fixed_tokens - 48) / 1.7))
    return words if words >= 1.5 * kind["min_words"] else 0
