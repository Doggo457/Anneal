"""Build the loop's six-language gate once (CPU only; safe to run next to scoring or training).

  python scripts/build_gate.py [--target 495] [--parallel 4]

data/eval/gate.jsonl gets two kinds of items, none in Python:
  write   a DeepMind CodeContests valid/test problem (stdin/stdout) to solve in C++, Java, Go, Rust, JavaScript or
          TypeScript. Each problem gets one language, in rotation over a seeded shuffle.
  fix     a real wrong C++ or Java submission to the same problem (CodeContests incorrect_solutions) to debug. A
          submission qualifies only if, in our sandbox, it compiles, passes every example case, gets a wrong answer
          or crashes on another case and never times out (a bug, not slowness), uses no file input/output, and, in
          C++, no plain `long` (32 bits on Windows, so the bug would be the platform's). The fix language must also
          have a reference solution that passes, so a correct fix can pass in time.
A problem stays only if one of its own C++ or Java reference solutions passes our runner on all chosen cases (so a
fail means the answer, not the harness), it reads no input or output files, and it overlaps neither EvalPlus nor any
row trained on or pooled so far (data/processed/**/train.jsonl and val.jsonl). Fix items fill the gate up to
--target; if there are too few, problems get a write item in a second language.

The Python gate (CodeContests + OpenCodeInstruct; retired by the language rule of 21 Oct 2026) moves to
data/eval/archive/python-gate/ before anything is built, so the loop never starts on it; scores on the new gate
are not comparable with it. decontam_extra.jsonl keeps every earlier problem and gains the new ones, for
decontaminating later training data. The card (data/eval/card.md, card.json) lists every count and the toolchain
versions (sandbox/toolchains/lock.json).
"""

import argparse
import json
import os
import random
import re
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from anneal import evalsets as E  # noqa: E402
from anneal import polyglot as P  # noqa: E402

ARCHIVE = os.path.join(E.EVAL_DIR, "archive", "python-gate")
CC_REPO = "deepmind/code_contests"
CC_COLUMNS = ["name", "description", "public_tests", "private_tests", "generated_tests", "solutions",
              "incorrect_solutions", "input_file", "output_file", "cf_rating", "difficulty", "source"]
CC_LANG = {2: "cpp", 4: "java"}
REFS_PER_LANG = 3
FIX_TRIES_PER_LANG = 4
MAX_FIX_CHARS, MAX_FIX_LINES = 6000, 200
SEED = 20261021
BARE_LONG = re.compile(r"(?<!long\s)\blong\b(?!\s+(?:long|double)\b)")
FILE_IO = {"cpp": re.compile(r"\b(?:freopen|fopen|ifstream|ofstream|fstream)\b"),
           "java": re.compile(r"\b(?:FileInputStream|FileReader|FileOutputStream|FileWriter|RandomAccessFile)\b"
                              r"|\bnew\s+File\s*\(")}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def cases_and_public(row, max_cases=E.MAX_IO_CASES):
    """Like evalsets.cc_cases (public, private, generated; duplicates and giant cases out), plus how many of the
    first cases are the public examples."""
    out, seen, n_public = [], set(), 0
    for field in ("public_tests", "private_tests", "generated_tests"):
        t = row.get(field) or {}
        for i, o in zip(t.get("input") or [], t.get("output") or []):
            if not isinstance(i, str) or not isinstance(o, str):
                continue
            if len(i) > E.MAX_CASE_BYTES or len(o) > E.MAX_CASE_BYTES or (i, o) in seen:
                continue
            seen.add((i, o))
            out.append((i, o))
            n_public += field == "public_tests"
            if len(out) >= max_cases:
                return out, n_public
    return out, n_public


def solutions(field, lang):
    sols = field or {}
    return [s for code, s in zip(sols.get("language") or [], sols.get("solution") or []) if CC_LANG.get(code) == lang]


def fix_ok(lang, code):
    """A wrong submission worth debugging here: short, no file I/O, and in C++ no plain long."""
    if len(code) > MAX_FIX_CHARS or code.count("\n") > MAX_FIX_LINES or FILE_IO[lang].search(code):
        return False
    return not (lang == "cpp" and BARE_LONG.search(code))


def qualifies(statuses, n_public):
    """Passes every example, fails another case with a wrong answer or a crash, never times out."""
    if not statuses or n_public < 1 or len(statuses) <= n_public or "timeout" in statuses:
        return False
    return all(s == "ok" for s in statuses[:n_public]) and any(s in ("wrong_answer", "runtime_error")
                                                                for s in statuses[n_public:])


def archive_python_gate():
    if not os.path.exists(E.GATE_PATH):
        return False
    first = (E.read_jsonl(E.GATE_PATH) or [{}])[0]
    if first.get("kind") in ("write", "fix"):
        raise SystemExit(f"{E.GATE_PATH} is already the six-language gate. It must not change once rounds are scored "
                         "on it.")
    os.makedirs(ARCHIVE, exist_ok=True)
    for name in ("gate.jsonl", "card.json", "card.md"):
        src = os.path.join(E.EVAL_DIR, name)
        if os.path.exists(src):
            shutil.move(src, os.path.join(ARCHIVE, name))
    log(f"the Python gate and its card moved to {ARCHIVE}")
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--target", type=int, default=495)
    ap.add_argument("--parallel", type=int, default=4, help="programs judged at once (each JVM may take 2 GB)")
    ap.add_argument("--revision", default=None, help="CodeContests revision (default: the one the Python gate used)")
    args = ap.parse_args()
    import build_evalsets as BE
    from anneal import sandbox
    sandbox.require()
    ex, tc = P.SandboxExecutor(), P.Toolchain.windows()
    t0 = time.time()
    card = {"started": time.strftime("%Y-%m-%d %H:%M:%S"), "target": args.target}
    old_card = json.load(open(os.path.join(E.EVAL_DIR, "card.json"), encoding="utf-8")) \
        if os.path.exists(os.path.join(E.EVAL_DIR, "card.json")) else {}
    if os.path.exists(os.path.join(ARCHIVE, "card.json")):
        old_card = json.load(open(os.path.join(ARCHIVE, "card.json"), encoding="utf-8"))
    lock_path = os.path.join(sandbox.TOOLCHAINS_DIR, "lock.json")
    if not os.path.exists(lock_path):
        raise SystemExit(f"{lock_path} is missing: run scripts/setup_toolchains.py, then --check, first")
    lock = json.load(open(lock_path, encoding="utf-8"))
    smoke_path = os.path.join(ROOT, "results", "toolchains", "smoke.json")
    smoke = json.load(open(smoke_path, encoding="utf-8")) if os.path.exists(smoke_path) else None
    if not smoke or smoke.get("mismatches") or not smoke.get("results"):
        raise SystemExit(f"{smoke_path}: the smoke check has not passed; run scripts/setup_toolchains.py --check first")
    card["archived_python_gate"] = archive_python_gate()
    card["toolchains"] = {k: v.get("version_output") or v.get("version") for k, v in lock.items()
                          if isinstance(v, dict) and "version" in v}
    card["toolchain_fingerprint"] = P.toolchain_fingerprint()

    # ---- CodeContests valid + test
    rev = args.revision or old_card.get("cc_revision")
    paths, sha = BE.fetch(CC_REPO, ["data/valid-", "data/test-"], revision=rev)
    probs = []
    for split in ("valid", "test"):
        rows = list(BE.parquet_rows([p for p in paths if os.path.basename(p).startswith(split + "-")], CC_COLUMNS))
        card[f"cc_{split}_rows"] = len(rows)
        for i, r in enumerate(rows):
            desc = (r.get("description") or "").strip()
            if (r.get("input_file") or "").strip() or (r.get("output_file") or "").strip() or not desc:
                continue
            cases, n_public = cases_and_public(r)
            if not cases:
                continue
            probs.append({"id": f"cc/{split}/{i}", "name": r.get("name"), "description": desc, "cases": cases,
                          "n_public": n_public, "cf_rating": r.get("cf_rating"), "difficulty": r.get("difficulty"),
                          "source": r.get("source"),
                          "refs": {lang: solutions(r.get("solutions"), lang)[:REFS_PER_LANG] for lang in ("cpp", "java")},
                          "wrong": {lang: solutions(r.get("incorrect_solutions"), lang) for lang in ("cpp", "java")}})
    card["cc_revision"], card["cc_candidates"] = sha, len(probs)
    log(f"CodeContests @ {sha}: {len(probs)} problems with stdin/stdout cases and no input/output files")

    # ---- overlap: EvalPlus, everything trained on or pooled so far
    ep = BE.evalplus_index()
    gidx = E.OverlapIndex([(p["id"], p["description"]) for p in probs])
    dropped, n_rows = set(), 0
    for dirpath, _, files in sorted(os.walk(os.path.join(ROOT, "data", "processed"))):
        for f in sorted(files):
            if f not in ("train.jsonl", "val.jsonl"):
                continue
            with open(os.path.join(dirpath, f), encoding="utf-8") as fh:  # streamed: pools can be large
                for line in fh:
                    if not line.strip():
                        continue
                    m = json.loads(line).get("messages") or []
                    users = "\n".join(str(x.get("content") or "") for x in m if x.get("role") == "user")
                    last = str(m[-1].get("content") or "") if m else ""
                    dropped |= gidx.hits(users + "\n" + last, jaccard_text=users)
                    n_rows += 1
    dropped |= {p["id"] for p in probs if ep.hits(p["description"])}
    probs = [p for p in probs if p["id"] not in dropped]
    card["overlap_dropped"] = sorted(dropped)
    card["overlap_rows_checked"] = n_rows
    log(f"overlap: {len(dropped)} problems dropped ({n_rows} trained or pooled rows checked); {len(probs)} left")

    # ---- references: which languages have a reference that passes our runner
    def check_refs(p):
        ok = {}
        for lang in ("cpp", "java"):
            for j, code in enumerate(p["refs"][lang]):
                v = P.judge(lang, code, p["cases"], ex, tc)
                if v.ok:
                    ok[lang] = j
                    break
        return ok

    with ThreadPoolExecutor(args.parallel) as pool:
        for p, ok in zip(probs, pool.map(check_refs, probs)):
            p["ref_ok"] = ok
    card["ref_pass"] = {lang: sum(lang in p["ref_ok"] for p in probs) for lang in ("cpp", "java")}
    card["no_ref_pass"] = sum(not p["ref_ok"] for p in probs)
    probs = [p for p in probs if p["ref_ok"]]
    log(f"references: {len(probs)} problems with a passing C++ or Java reference ({card['ref_pass']})")

    # ---- fix candidates: per problem and language, the first wrong submission that qualifies
    def find_fixes(p):
        found, tried = {}, {}
        for lang in ("cpp", "java"):
            if lang not in p["ref_ok"]:
                continue
            cands = [(k, c) for k, c in enumerate(p["wrong"][lang]) if fix_ok(lang, c)]
            random.Random(f"{SEED}/{p['id']}/{lang}").shuffle(cands)
            tried[lang] = 0
            for k, code in cands[:FIX_TRIES_PER_LANG]:
                tried[lang] += 1
                v = P.judge(lang, code, p["cases"], ex, tc, all_cases=True)
                if v.stage != "compile_error" and qualifies(v.cases, p["n_public"]):
                    found[lang] = {"index": k, "code": code, "statuses": {s: v.cases.count(s) for s in set(v.cases)}}
                    break
        return found, tried

    order = sorted(probs, key=lambda p: p["id"])
    random.Random(SEED).shuffle(order)
    with ThreadPoolExecutor(args.parallel) as pool:
        for p, (found, tried) in zip(order, pool.map(find_fixes, order)):
            p["fixes"], p["fix_tried"] = found, tried
    card["fix_tried"] = {lang: sum(p["fix_tried"].get(lang, 0) for p in order) for lang in ("cpp", "java")}
    card["fix_found"] = {lang: sum(lang in p["fixes"] for p in order) for lang in ("cpp", "java")}
    log(f"fix candidates: {card['fix_found']} problems with a qualifying wrong submission ({card['fix_tried']} tried)")

    # ---- the gate: one write item per problem, fix items up to the target, then second-language write items
    items = []

    def base(p):
        return {"problem": p["id"], "name": p["name"], "cases": p["cases"], "cf_rating": p["cf_rating"],
                "difficulty": p["difficulty"], "source": p["source"], "ref_langs": sorted(p["ref_ok"])}

    for i, p in enumerate(order):
        lang = P.LANG_ORDER[i % len(P.LANG_ORDER)]
        items.append(dict(base(p), id=f"write/{lang}/{p['id']}", kind="write", lang=lang,
                          prompt=p["description"] + P.write_suffix(lang)))
    n_fix = max(0, args.target - len(items))
    per_lang = {"cpp": 0, "java": 0}
    for p in order:
        if sum(per_lang.values()) >= n_fix:
            break
        avail = [lang for lang in ("cpp", "java") if lang in p["fixes"]]
        if not avail:
            continue
        lang = min(avail, key=lambda x: (per_lang[x], x))
        f = p["fixes"][lang]
        per_lang[lang] += 1
        items.append(dict(base(p), id=f"fix/{lang}/{p['id']}", kind="fix", lang=lang,
                          prompt=P.fix_prompt(p["description"], lang, f["code"]),
                          wrong={"index": f["index"], "statuses": f["statuses"]}))
    extra = 0
    for i, p in enumerate(order):
        if len(items) >= args.target:
            break
        lang = P.LANG_ORDER[(i + 3) % len(P.LANG_ORDER)]
        items.append(dict(base(p), id=f"write/{lang}/{p['id']}", kind="write", lang=lang,
                          prompt=p["description"] + P.write_suffix(lang)))
        extra += 1
    ids = [it["id"] for it in items]
    assert len(ids) == len(set(ids)), "duplicate gate ids"
    card.update(gate_size=len(items), write_items=sum(it["kind"] == "write" for it in items),
                fix_items=dict(per_lang), second_language_write_items=extra,
                by_language={lang: sum(it["lang"] == lang for it in items) for lang in P.LANG_ORDER},
                by_kind_language={f"{it['kind']}/{it['lang']}": 0 for it in items})
    for it in items:
        card["by_kind_language"][f"{it['kind']}/{it['lang']}"] += 1
    log(f"gate: {len(items)} items ({card['write_items']} write, {per_lang} fix; {extra} second-language write)")

    # ---- write: the decontamination list (old + new problems), the card, and the gate last (the watchdog starts the
    # loop once gate.jsonl exists)
    extra_rows = E.read_jsonl(E.DECONTAM_EXTRA)
    have = {r.get("id") for r in extra_rows}
    added = [{"id": p["id"], "prompt": p["description"]} for p in order if p["id"] not in have]
    E.write_jsonl(E.DECONTAM_EXTRA, extra_rows + added)
    card.update(decontam_added=len(added), decontam_total=len(extra_rows) + len(added),
                seconds=round(time.time() - t0))
    with open(os.path.join(E.EVAL_DIR, "card.json"), "w", encoding="utf-8") as fh:
        json.dump(card, fh, indent=1)
    L = ["# The loop's gate: six languages, write and fix", "",
         f"Built {card['started']} by scripts/build_gate.py in {card['seconds']} s. No Python: Python is the lowest "
         "priority (21 Oct 2026), so the Python gate is archived in data/eval/archive/python-gate/.", "",
         "| Step | Count |", "|---|---|",
         f"| CodeContests `{CC_REPO}` @ `{sha}` valid + test rows | {card['cc_valid_rows'] + card['cc_test_rows']} |",
         f"| with stdin/stdout cases, no input/output files | {card['cc_candidates']} |",
         f"| dropped for overlap with EvalPlus or {card['overlap_rows_checked']} trained or pooled rows | "
         f"{len(card['overlap_dropped'])} |",
         f"| no C++ or Java reference passes our runner | {card['no_ref_pass']} |",
         f"| kept problems (a reference passes: C++ {card['ref_pass']['cpp']}, Java {card['ref_pass']['java']}) | "
         f"{len(order)} |",
         f"| wrong submissions tried (C++ / Java) | {card['fix_tried']['cpp']} / {card['fix_tried']['java']} |",
         f"| problems with a qualifying wrong submission (C++ / Java) | {card['fix_found']['cpp']} / "
         f"{card['fix_found']['java']} |", "",
         f"Gate: {len(items)} items: {card['write_items']} write ({extra} of them a problem's second language), "
         f"fix C++ {per_lang['cpp']}, fix Java {per_lang['java']}.", "",
         "| Language | Write | Fix | All |", "|---|---|---|---|"]
    for lang in P.LANG_ORDER:
        w = card["by_kind_language"].get(f"write/{lang}", 0)
        f = card["by_kind_language"].get(f"fix/{lang}", 0)
        L.append(f"| {P.NAMES[lang]} | {w} | {f} | {w + f} |")
    L += ["", "Toolchains (sandbox/toolchains/lock.json): " + "; ".join(f"{k}: {v}" for k, v in card["toolchains"].items())
          + ".", "",
          "Judging (anneal/polyglot.py): the last full program in the answer's code blocks for the item's language is "
          "built in the build container and run in the code sandbox on up to 30 cases (examples first), 10 s each; "
          "output compared token by token, numbers within 1e-6.",
          f"decontam_extra.jsonl: {len(added)} new problems added, {card['decontam_total']} in all."]
    with open(os.path.join(E.EVAL_DIR, "card.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    E.write_jsonl(E.GATE_PATH, items)
    log(f"wrote {E.GATE_PATH}, {E.DECONTAM_EXTRA} (+{len(added)}), card.md")


if __name__ == "__main__":
    main()
