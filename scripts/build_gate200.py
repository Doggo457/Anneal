"""Build the fixed 200 from the 495-item gate, or check problems before the draw (CPU only; runs next to scoring).

  python scripts/build_gate200.py --spec SPEC.json --validators DIR [--out data/eval/gate200]
  python scripts/build_gate200.py --check SPEC.json --validators DIR --report REPORT.json

My rule (25 Oct 2026): the base answers all 495, then a fixed 200 of them is the gate, drawn by a rule that never
looks at the base's answers (rating bands weighted to the harder problems, language and write/fix shares kept, only
problems that can be tested), and the checker is fixed before the next run.

Every problem is judged on its full CodeContests test set: public, private and generated tests (duplicates and
cases over 256 KB out, as build_gate.py), in that order. Each problem needs an input validator, DIR/CF.py (CF the
Codeforces id, e.g. 1575B) with validate(input_text) that returns True only for an input within the statement's
limits. A public or private test (Codeforces' own) that fails it means the validator is wrong: the problem is refused.
A generated test that fails it is set aside: CodeContests made those by changing real inputs, and some break the
limits (values of -1, n = 6 with five numbers), where a correct program may do anything.

SPEC (JSON): {"items": [the 200 item ids], "problems": {CF: {"tol": 1e-4} or {"checker": CF}}}. tol: the error the
statement allows on real numbers. checker: anneal/gate_checkers.py's checker for a problem where more than one output
is right. Problems not listed are compared exactly (YES/NO in any letter case: anneal/evalsets.case_match).

A problem may also carry "added_tests": [{"input", "output", "why"}], hand-checked tests for a wrong method that passes
every official test (26 Oct 2026: 1615H, where merging the worst-violating groups passes all 204 valid official tests
but misses the minimum on a 6-reindeer input). They go after the official tests, so the 495's cases still come first;
each must pass the validator (else the problem is refused), the judge's self-test and, with a checker, the accepted
solutions, like any other test. The card counts them.

A problem is fit when, after setting tests aside, at least 30 remain, its judge accepts the expected output of every
test, and, with a checker, no test has 2 or more of its accepted solutions (up to 3: C++ first, then Java, as the
eligibility table) rejected by the checker (one rejected solution is that solution's bug: the other two back the
test). --check reports this for every problem in SPEC (no items needed) without writing a gate: a checker problem is
drawn only if it is fit. The build refuses unless every problem is fit and every item's first cases are exactly its
495 cases (same CodeContests revision and order). It writes OUT/gate.jsonl (each item keeps its prompt, id and order
from the 495, so the base's stored answers can be judged again), OUT/cases/CF.json (the kept tests, pinned by a
checksum in each item) and OUT/card.json, card.md. scripts/switch_gate200.py puts it in use.
"""

import argparse
import collections
import hashlib
import importlib.util
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from anneal import evalsets as E  # noqa: E402

MIN_TESTS = 30
REFS = 3
SPEC_KEYS = {"tol", "checker"}
ADDED_KEY = "added_tests"
CC_COLUMNS = ["name", "public_tests", "private_tests", "generated_tests", "solutions"]
CC_LANG = {2: "cpp", 4: "java"}


def cf_id(name):
    """'1575_B. Building an Amusement Park' -> '1575B'."""
    m = re.match(r"(\d+)_([A-Z][0-9]?)\.", name or "")
    if not m:
        raise ValueError(f"not a Codeforces problem name: {name!r}")
    return m.group(1) + m.group(2)


def full_cases(row):
    """[(stdin, expected, generated)]: public, private, then generated tests; duplicates and cases over
    E.MAX_CASE_BYTES out (build_gate.cases_and_public's rule, with no count limit)."""
    out, seen = [], set()
    for field in ("public_tests", "private_tests", "generated_tests"):
        t = row.get(field) or {}
        for i, o in zip(t.get("input") or [], t.get("output") or []):
            if not isinstance(i, str) or not isinstance(o, str):
                continue
            if len(i) > E.MAX_CASE_BYTES or len(o) > E.MAX_CASE_BYTES or (i, o) in seen:
                continue
            seen.add((i, o))
            out.append((i, o, field == "generated_tests"))
    return out


def refs_of(row):
    """Up to REFS accepted solutions: C++ first, then Java."""
    sols = row.get("solutions") or {}
    pairs = [(CC_LANG.get(code), s) for code, s in zip(sols.get("language") or [], sols.get("solution") or [])]
    return ([("cpp", s) for lang, s in pairs if lang == "cpp"] + [("java", s) for lang, s in pairs if lang == "java"])[:REFS]


def load_validator(vdir, cf):
    path = os.path.join(vdir, f"{cf}.py")
    if not os.path.exists(path):
        return None, None
    raw = open(path, "rb").read()
    spec = importlib.util.spec_from_file_location(f"gate200_validator_{cf}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.validate, hashlib.sha256(raw).hexdigest()


def is_valid(validate, text):
    try:
        return validate(text) is True
    except Exception:  # noqa: BLE001  (an input the validator can't parse breaks the format)
        return False


def check_problem(cf, row, jspec, validate, ex=None, tc=None, added=()):
    """-> (report, kept cases). report["fit"] says whether the problem can be in the 200 (module docstring). added:
    the problem's hand-checked (input, output) tests, kept after the official ones."""
    full = full_cases(row)
    flags = [is_valid(validate, i) for i, _, _ in full]
    official_bad = [k for k, (_, _, gen) in enumerate(full) if not gen and not flags[k]]
    aside = [k for k, (_, _, gen) in enumerate(full) if gen and not flags[k]]
    added_bad = [k for k, (i, _) in enumerate(added) if not is_valid(validate, i)]
    drop = set(aside)
    kept = [(i, o) for k, (i, o, _) in enumerate(full) if k not in drop] + [(i, o) for i, o in added]
    match = E.case_match({"judge": jspec})
    expected_rejected = [k for k, (i, o) in enumerate(kept) if not match(i, o, o)]
    rep = {"tests": len(full), "official": sum(not g for _, _, g in full), "set_aside": aside, "kept": len(kept),
           "official_invalid": official_bad[:20], "expected_rejected": expected_rejected[:20]}
    if added:
        rep.update(added=len(added), added_invalid=added_bad)
    if jspec.get("checker") and ex is not None and not official_bad and not added_bad and not expected_rejected:
        from anneal import polyglot as P
        per_test, refs = collections.Counter(), []
        for lang, code in refs_of(row):
            v = P.judge(lang, code, kept, ex, tc, case_timeout_s=E.IO_CASE_TIMEOUT_S, all_cases=True, match=match)
            st = v.cases if v.stage != "compile_error" else []
            wrong = [k for k, s in enumerate(st) if s == "wrong_answer"]
            per_test.update(wrong)
            refs.append({"lang": lang, "stage": v.stage, "wrong": len(wrong),
                         "other": sum(s not in ("ok", "wrong_answer") for s in st)})
        rep["refs"] = refs
        rep["tests_two_refs_rejected"] = sorted(k for k, c in per_test.items() if c >= 2)[:20]
        rep["refs_ran"] = sum(r["stage"] != "compile_error" for r in refs)
    why = ("public or private tests fail the validator" if official_bad
           else "an added test fails the validator" if added_bad
           else f"{len(kept)} tests left (under {MIN_TESTS})" if len(kept) < MIN_TESTS
           else "the judge rejects expected outputs" if expected_rejected
           else "checker not tested on the accepted solutions (no executor)" if jspec.get("checker") and ex is None
           else "fewer than 2 accepted solutions ran" if jspec.get("checker") and rep.get("refs_ran", 0) < 2
           else "2 or more accepted solutions rejected on a test" if rep.get("tests_two_refs_rejected")
           else None)
    rep["fit"], rep["why"] = why is None, why
    return rep, kept


def load_cc(revision, splits):
    """{(split, index): row} for the CodeContests rows the gate's problems come from, at the gate's revision."""
    import build_evalsets as BE
    import build_gate as BG
    paths, sha = BE.fetch(BG.CC_REPO, [f"data/{s}-" for s in sorted(splits)], revision=revision)
    if sha != revision:
        raise SystemExit(f"CodeContests resolved to {sha}, not the gate's {revision}")
    out = {}
    for split in sorted(splits):
        files = [p for p in paths if os.path.basename(p).startswith(split + "-")]
        for i, r in enumerate(BE.parquet_rows(files, CC_COLUMNS)):
            out[(split, i)] = dict(r, solutions=trim_refs(r.get("solutions") or {}))
    return out


def trim_refs(sols):
    """The first REFS C++ and the first REFS Java solutions, in their order: all refs_of can pick (keeps memory low)."""
    out, seen = {"language": [], "solution": []}, collections.Counter()
    for code, s in zip(sols.get("language") or [], sols.get("solution") or []):
        if code in CC_LANG and seen[code] < REFS:
            seen[code] += 1
            out["language"].append(code)
            out["solution"].append(s)
    return out


def read_problems(spec):
    probs = spec.get("problems") or {}
    for cf, p in probs.items():
        if set(p) - SPEC_KEYS - {ADDED_KEY}:
            raise SystemExit(f"{cf}: unknown spec keys {sorted(set(p) - SPEC_KEYS)}")
        if p.get("tol") is not None and not (isinstance(p["tol"], float) and 0 < p["tol"] < 1):
            raise SystemExit(f"{cf}: tol must be a number between 0 and 1, got {p['tol']!r}")
        if p.get("tol") is not None and p.get("checker"):
            raise SystemExit(f"{cf}: a problem has a tolerance or a checker, not both")
    if any(p.get("checker") for p in probs.values()):
        from anneal import gate_checkers
        absent = sorted(p["checker"] for p in probs.values() if p.get("checker") and p["checker"] not in
                        gate_checkers.CHECKERS)
        if absent:
            raise SystemExit(f"no checker in anneal/gate_checkers.py for {absent}")
    return {cf: {k: p[k] for k in SPEC_KEYS if p.get(k) is not None} for cf, p in probs.items()}


def read_added(spec):
    """{CF: [(input, output)]} from each problem's "added_tests" (module docstring)."""
    out = {}
    for cf, p in (spec.get("problems") or {}).items():
        tests = p.get(ADDED_KEY) or []
        for t in tests:
            if not (isinstance(t, dict) and set(t) == {"input", "output", "why"} and
                    all(isinstance(t[k], str) and t[k].strip() for k in t)):
                raise SystemExit(f"{cf}: each added test needs a non-empty input, output and why, got {t!r}")
        if tests:
            out[cf] = [(t["input"], t["output"]) for t in tests]
    return out


def problems_of(gate):
    """{CF: [gate items on it]} in the 495's order."""
    out = {}
    for it in gate:
        out.setdefault(cf_id(it["name"]), []).append(it)
    return out


def row_for(cf, its, cc_rows):
    split, idx = its[0]["problem"].split("/")[1:]
    row = cc_rows[(split, int(idx))]
    if cf_id(row["name"]) != cf:
        raise SystemExit(f"{its[0]['problem']} is {row['name']!r} in CodeContests, not {cf}")
    return row


def run_checks(cfs, by_cf, probs, cc_rows, vdir, ex, tc, parallel, log, added=None):
    """{CF: (report, kept)} for each problem in cfs, `parallel` at a time."""
    added = added or {}

    def one(cf):
        validate, vsha = load_validator(vdir, cf)
        if validate is None:
            return cf, ({"fit": False, "why": f"no validator {cf}.py"}, [])
        rep, kept = check_problem(cf, row_for(cf, by_cf[cf], cc_rows), probs.get(cf, {}), validate, ex, tc,
                                  added.get(cf, ()))
        rep["validator_sha256"] = vsha
        log(f"{cf}: {'fit' if rep['fit'] else 'NOT fit: ' + rep['why']}; {rep['kept']} of {rep['tests']} tests kept, "
            f"{len(rep['set_aside'])} set aside" + (f"; refs {rep['refs']}" if "refs" in rep else ""))
        return cf, (rep, kept)
    with ThreadPoolExecutor(max(1, parallel)) as pool:
        return dict(pool.map(one, sorted(cfs)))


def check(gate, spec, cc_rows, vdir, ex, tc, parallel=2, log=print):
    """--check: a fitness report for every problem in spec["problems"]."""
    probs, by_cf = read_problems(spec), problems_of(gate)
    unknown = sorted(cf for cf in probs if cf not in by_cf)
    if unknown:
        raise SystemExit(f"not in the gate: {unknown}")
    res = run_checks(set(probs), by_cf, probs, cc_rows, vdir, ex, tc, parallel, log, read_added(spec))
    return {cf: rep for cf, (rep, _) in res.items()}


def build(gate, spec, cc_rows, vdir, out_dir, ex=None, tc=None, parallel=2, log=print):
    """Writes the fixed 200 (module docstring) and returns its card."""
    ids = spec["items"]
    if len(ids) != len(set(ids)):
        raise SystemExit("the spec repeats an item")
    in_gate = {it["id"] for it in gate}
    missing = [i for i in ids if i not in in_gate]
    if missing:
        raise SystemExit(f"{len(missing)} spec items aren't in the gate: {missing[:5]}")
    probs, added = read_problems(spec), read_added(spec)
    want = set(ids)
    chosen = [it for it in gate if it["id"] in want]  # the 495's order: item i keeps its place among the 200
    by_cf = problems_of(chosen)
    stray = sorted(cf for cf in added if cf not in by_cf)
    if stray:
        raise SystemExit(f"added tests for problems not in the 200: {stray}")
    for cf, its in by_cf.items():
        row = row_for(cf, its, cc_rows)
        full = full_cases(row)
        for it in its:
            have = [tuple(c) for c in it["cases"]]
            if [(i, o) for i, o, _ in full[:len(have)]] != have:
                raise SystemExit(f"{it['id']}: its 495 cases aren't the first of {cf}'s full test set (revision or "
                                 "order differs)")
    res = run_checks(set(by_cf), by_cf, probs, cc_rows, vdir, ex, tc, parallel, log, added)
    unfit = sorted(f"{cf}: {rep['why']}" for cf, (rep, _) in res.items() if not rep["fit"])
    if unfit:
        raise SystemExit("not built; unfit problems:\n  " + "\n  ".join(unfit))
    ck_sha = None
    if any(p.get("checker") for p in probs.values()):
        from anneal import gate_checkers
        ck_sha = gate_checkers.module_sha256()
    os.makedirs(os.path.join(out_dir, "cases"), exist_ok=True)
    rel_dir = os.path.relpath(os.path.join(out_dir, "cases"), E.EVAL_DIR).replace(os.sep, "/")
    items, card_p = [], {}
    for cf in sorted(by_cf):
        rep, kept = res[cf]
        raw = json.dumps({"problem": cf, "cases": [list(c) for c in kept]}, ensure_ascii=False,
                         separators=(",", ":")).encode("utf-8")
        with open(os.path.join(out_dir, "cases", f"{cf}.json"), "wb") as fh:
            fh.write(raw)
        sha, jspec = hashlib.sha256(raw).hexdigest(), probs.get(cf, {})
        card_p[cf] = dict(rep, bytes=len(raw), judge=jspec or {"exact": True})
        if jspec.get("checker"):
            jspec = dict(jspec, checker_sha256=ck_sha)  # pins anneal/gate_checkers.py (evalsets.case_match)
        for it in by_cf[cf]:
            new = {k: v for k, v in it.items() if k != "cases"}
            new.update(cases_file=f"{rel_dir}/{cf}.json", cases_sha256=sha, n_cases=len(kept),
                       n_cases_full=rep["tests"], set_aside=rep["set_aside"], judge=jspec)
            if rep.get("added"):
                new["n_cases_added"] = rep["added"]  # the last n_cases_added of n_cases are hand-checked
            items.append(new)
    order = {it["id"]: k for k, it in enumerate(gate)}
    items.sort(key=lambda it: order[it["id"]])
    E.write_jsonl(os.path.join(out_dir, "gate.jsonl"), items)
    by = collections.Counter
    card = {"built": time.strftime("%Y-%m-%d %H:%M:%S"), "items": len(items), "problems": len(card_p),
            "spec_sha256": spec.get("_sha256"), "by_kind_language": dict(by(f"{it['kind']}/{it['lang']}" for it in items)),
            "tests": sum(v["kept"] for v in card_p.values()),
            "tests_set_aside": sum(len(v["set_aside"]) for v in card_p.values()),
            "problems_with_tests_set_aside": sum(bool(v["set_aside"]) for v in card_p.values()),
            "tests_added": {cf: v["added"] for cf, v in card_p.items() if v.get("added")},
            "checkers": sorted(cf for cf, v in card_p.items() if "checker" in v["judge"]), "checkers_sha256": ck_sha,
            "tolerance": {cf: v["judge"]["tol"] for cf, v in card_p.items() if "tol" in v["judge"]},
            "per_problem": card_p}
    with open(os.path.join(out_dir, "card.json"), "w", encoding="utf-8") as fh:
        json.dump(card, fh, indent=1)
    L = ["# The fixed 200", "",
         f"Built {card['built']} by scripts/build_gate200.py from the 495-item gate: {len(items)} items on "
         f"{len(card_p)} problems, judged on their full CodeContests test sets: {card['tests']} tests, after "
         f"{card['tests_set_aside']} generated tests on {card['problems_with_tests_set_aside']} problems were set aside "
         "because their input breaks the statement's limits (each problem's input validator)"
         + ("" if not card["tests_added"] else "; the total includes " + ", ".join(
             f"{n} hand-checked test{'s' * (n > 1)} added to {cf}" for cf, n in sorted(card["tests_added"].items()))
            + " (a wrong method passed every official test there)") + ".", "",
         "| Kind / language | Items |", "|---|---|"]
    L += [f"| {k} | {v} |" for k, v in sorted(card["by_kind_language"].items())]
    L += ["", "Judging: output token by token; YES and NO in any letter case; numbers exact except where the statement "
          "allows an error (" + (", ".join(f"{k} {v:g}" for k, v in sorted(card["tolerance"].items())) or "none")
          + "); a validity checker where more than one output is right (" + (", ".join(card["checkers"]) or "none")
          + "). 10 s a test, as before."]
    with open(os.path.join(out_dir, "card.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    log(f"wrote {out_dir}: {len(items)} items, {len(card_p)} problems, {card['tests']} tests "
        f"({card['tests_set_aside']} set aside)")
    return card


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--spec", help="build the 200 from this spec")
    ap.add_argument("--check", help="report on every problem in this spec instead of building")
    ap.add_argument("--validators", required=True, help="folder of CF.py input validators")
    ap.add_argument("--out", default=os.path.join(E.EVAL_DIR, "gate200"))
    ap.add_argument("--report", default=None, help="--check: where the report goes (JSON)")
    ap.add_argument("--parallel", type=int, default=2, help="problems checked at once (each JVM may take 2 GB)")
    args = ap.parse_args()
    if bool(args.spec) == bool(args.check):
        raise SystemExit("give --spec or --check")
    path = args.spec or args.check
    raw = open(path, "rb").read()
    spec = json.loads(raw)
    spec["_sha256"] = hashlib.sha256(raw).hexdigest()
    gate = E.read_jsonl(E.GATE_PATH)
    if len(gate) != 495 or {it.get("kind") for it in gate} != {"write", "fix"}:
        raise SystemExit(f"{E.GATE_PATH} isn't the 495-item gate ({len(gate)} items)")
    eval_card = json.load(open(os.path.join(E.EVAL_DIR, "card.json"), encoding="utf-8"))
    by_cf = problems_of(gate)
    cfs = set(spec.get("problems") or {}) if args.check else {cf_id(it["name"]) for it in gate if it["id"] in
                                                               set(spec["items"])}
    splits = {its[0]["problem"].split("/")[1] for cf, its in by_cf.items() if cf in cfs}
    cc_rows = load_cc(eval_card["cc_revision"], splits)
    from anneal import polyglot as P
    from anneal import sandbox
    sandbox.require()
    ex, tc = P.SandboxExecutor(), P.Toolchain.windows()
    if args.check:
        rep = check(gate, spec, cc_rows, args.validators, ex, tc, args.parallel)
        out = args.report or os.path.splitext(args.check)[0] + ".report.json"
        with open(out, "w", encoding="utf-8") as fh:
            json.dump(rep, fh, indent=1)
        fit = sorted(cf for cf, r in rep.items() if r["fit"])
        print(f"{len(fit)} of {len(rep)} problems fit: {','.join(fit)}\nreport: {out}")
    else:
        build(gate, spec, cc_rows, args.validators, args.out, ex, tc, args.parallel)


if __name__ == "__main__":
    main()
