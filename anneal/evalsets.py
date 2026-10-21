"""Held-out sets for the keep-if-better loop: the gate and the non-coding check, how they are built and scored.

The gate decides whether a new adapter is kept. It is never trained on and never reported, so choosing adapters on
it doesn't inflate the reported numbers (EvalPlus, LiveCodeBench). Since 21 Oct 2026 it has no Python
(scripts/build_gate.py, anneal/polyglot.py):
  write  a DeepMind CodeContests valid/test stdin/stdout problem to solve in C++, Java, Go, Rust, JavaScript or
         TypeScript (one language per problem, in rotation);
  fix    a real wrong C++ or Java submission to the same problems to debug (passes the examples, fails another case).
A problem stays only if one of its own C++ or Java reference solutions passes our sandboxed runner on the chosen
cases, so a fail means the answer, not the harness. The earlier Python gate (kinds cc and oci: CodeContests with
Python 3 references, and 300 OpenCodeInstruct assert problems) is archived in data/eval/archive/python-gate/; its
kinds are still scored here for old results. Problems that overlap any training data used so far (13-gram or
Jaccard > 0.8) are left out, and every later training source is decontaminated against the gate
(decontam_extra.jsonl).

The non-coding check: 100 MMLU-Pro test questions outside computer science (random.Random(42)), multiple choice
with up to 10 options, scored by the letter after "The answer is".

Scoring: one chat request per problem with thinking on (temperature 0.6, top_p 0.95, top_k 20, min_p 0, request seed
100000 * seed + problem index). A write or fix answer's program (anneal/polyglot.py extract_code) is built and run in
the sandbox; an old cc/oci answer's first Python block runs in the sandbox. Results are JSONL, one row per problem
and seed (with the verdict: ok, no_code, compile_error, runtime_error, timeout, wrong_answer), resumable.

The keep rule (decide) also has a second path (chosen 21 Oct 2026): a round whose gate score holds level
(not lower) is kept when its loss on held-out bug-fix steps (data/eval/swe_heldout.jsonl, repositories no round
trains on) falls, by a one-sided sign test over the steps.
"""

import bisect
import collections
import hashlib
import json
import math
import os
import re
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVAL_DIR = os.path.join(ROOT, "data", "eval")
GATE_PATH = os.path.join(EVAL_DIR, "gate.jsonl")
NONCODING_PATH = os.path.join(EVAL_DIR, "noncoding_100.jsonl")
DECONTAM_EXTRA = os.path.join(EVAL_DIR, "decontam_extra.jsonl")

IO_CASE_TIMEOUT_S = 10      # per stdin/stdout case, as generate_traces.py
ASSERT_TIMEOUT_S = 20       # per assert run, as generate_traces.py
MAX_IO_CASES = 30
MAX_CASE_BYTES = 256 << 10  # skip giant generated inputs/outputs
MC_LETTERS = "ABCDEFGHIJ"
MC_INSTRUCTION = ("Answer the following multiple choice question. Think step by step, then finish your answer with "
                  "\"The answer is (X)\" where X is the letter of the correct option.")


def read_jsonl(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def write_jsonl(path, rows):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


# ============================================================================================ overlap

_TOK = re.compile(r"[a-z0-9_]+")  # the same word tokens as prepare_dataset.py


def word_tokens(text):
    return _TOK.findall((text or "").lower())


def word_ngrams(toks, n):
    return {tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)}


class OverlapIndex:
    """Which held-out problems a text overlaps. A text overlaps problem P if it contains at least `frac` of P's
    distinct 13-grams (lowercase word tokens), or its token set has Jaccard similarity above `jaccard` with P's.
    A share of P's 13-grams rather than any single shared 13-gram, because competitive-programming statements share
    boilerplate ("the first line contains a single integer t ... the number of test cases") that an any-match rule
    would flag in thousands of unrelated problems. items: [(id, text)]."""

    def __init__(self, items, n=13, frac=0.3, jaccard=0.8):
        self.n, self.frac, self.jaccard = n, frac, jaccard
        self.index = collections.defaultdict(list)
        self.ids, self.size, sets = [], [], []
        for k, (iid, text) in enumerate(items):
            toks = word_tokens(text)
            grams = word_ngrams(toks, n)
            for g in grams:
                self.index[g].append(k)
            self.ids.append(iid)
            self.size.append(len(grams))
            s = frozenset(toks)
            if s:
                sets.append((len(s), k, s))
        sets.sort(key=lambda t: (t[0], t[1]))
        self.sets = sets
        self.set_sizes = [t[0] for t in sets]

    def __len__(self):
        return len(self.ids)

    def hits(self, text, jaccard_text=None):
        """Ids of the problems `text` overlaps. jaccard_text (default: text) is what the Jaccard rule compares,
        e.g. only the prompt of a training row while the 13-gram rule reads prompt and response."""
        counts = collections.Counter()
        for g in word_ngrams(word_tokens(text), self.n):
            for k in self.index.get(g, ()):
                counts[k] += 1
        out = {self.ids[k] for k, c in counts.items() if self.size[k] and c / self.size[k] >= self.frac}
        a = frozenset(word_tokens(text if jaccard_text is None else jaccard_text))
        if a:
            # J(a, b) > t needs t*|a| <= |b| <= |a|/t, so only that size band can match.
            i = bisect.bisect_left(self.set_sizes, self.jaccard * len(a))
            hi = len(a) / self.jaccard
            while i < len(self.sets) and self.sets[i][0] <= hi:
                size, k, b = self.sets[i]
                inter = len(a & b)
                if inter / (len(a) + size - inter) > self.jaccard:
                    out.add(self.ids[k])
                i += 1
        return out

    @classmethod
    def from_jsonl(cls, path, **kw):
        """From decontam_extra.jsonl (or any JSONL with an "id" and a "prompt", "question" or "text" field)."""
        items = []
        for i, r in enumerate(read_jsonl(path)):
            text = next((r[k] for k in ("prompt", "question", "text") if r.get(k)), None)
            if text:
                items.append((r.get("id", str(i)), text))
        return cls(items, **kw)


# ============================================================================================ building

def cc_cases(row, max_cases=MAX_IO_CASES):
    """Public, then private, then generated stdin/stdout cases, duplicates and giant cases dropped."""
    out, seen = [], set()
    for field in ("public_tests", "private_tests", "generated_tests"):
        t = row.get(field) or {}
        for i, o in zip(t.get("input") or [], t.get("output") or []):
            if not isinstance(i, str) or not isinstance(o, str):
                continue
            if len(i) > MAX_CASE_BYTES or len(o) > MAX_CASE_BYTES or (i, o) in seen:
                continue
            seen.add((i, o))
            out.append((i, o))
            if len(out) >= max_cases:
                return out
    return out


def cc_python3_solutions(row, limit=3):
    sols = row.get("solutions") or {}
    return [s for lang, s in zip(sols.get("language") or [], sols.get("solution") or []) if lang == 3][:limit]


def cc_candidates(rows, split, stdio_suffix):
    """CodeContests rows -> gate candidates (not yet harness-checked)."""
    out = []
    for i, r in enumerate(rows):
        if (r.get("input_file") or "").strip() or (r.get("output_file") or "").strip():
            continue
        cases = cc_cases(r)
        refs = cc_python3_solutions(r)
        desc = (r.get("description") or "").strip()
        if not cases or not refs or not desc:
            continue
        out.append({"id": f"cc/{split}/{i}", "kind": "cc", "name": r.get("name"), "prompt": desc + stdio_suffix,
                    "raw_prompt": desc, "cases": cases, "refs": refs, "cf_rating": r.get("cf_rating"),
                    "difficulty": r.get("difficulty"), "source": r.get("source")})
    return out


def harness_check(cands, run_io_tests, run_assert_tests, parallel=4, log=print):
    """Keep candidates whose own reference solution passes our runner. Adds "ref_ok" (index of the passing
    reference) and drops "refs"."""
    def one(c):
        if c["kind"] == "cc":
            for j, code in enumerate(c["refs"]):
                if run_io_tests(code, c["cases"], IO_CASE_TIMEOUT_S):
                    return j
            return None
        return 0 if run_assert_tests(c["ref"], c["tests"], ASSERT_TIMEOUT_S) else None

    kept = []
    with ThreadPoolExecutor(parallel) as ex:
        for c, j in zip(cands, ex.map(one, cands)):
            if j is not None:
                c = dict(c, ref_ok=j)
                c.pop("refs", None)
                c.pop("ref", None)
                kept.append(c)
    log(f"harness check: {len(kept)} of {len(cands)} keep (a reference solution passes)")
    return kept


def mmlu_pro_questions(rows, n=100, seed=42, exclude=("computer science",)):
    import random
    pool = [r for r in rows if str(r.get("category", "")).lower() not in exclude]
    pick = random.Random(seed).sample(pool, n)
    out = []
    for r in pick:
        opts = list(r["options"])
        while opts and str(opts[-1]).strip() == "N/A":  # MMLU-Pro pads short option lists with "N/A"
            opts.pop()
        idx = int(r["answer_index"])
        if idx >= len(opts):
            raise ValueError(f"MMLU-Pro {r.get('question_id')}: answer {idx} beyond {len(opts)} options")
        out.append({"id": f"mmlupro/{r['question_id']}", "category": r["category"], "question": r["question"],
                    "options": opts, "answer": MC_LETTERS[idx]})
    return out


def mc_message(q):
    lines = [MC_INSTRUCTION, "", f"Question: {q['question'].strip()}", "Options:"]
    lines += [f"{MC_LETTERS[i]}. {str(o).strip()}" for i, o in enumerate(q["options"])]
    return "\n".join(lines)


_ANS_RE = re.compile(r"answer is[:\s]*\(?\s*([A-J])\s*\)?", re.IGNORECASE)
_ANS_RE2 = re.compile(r"[Aa]nswer:\s*\(?([A-J])\)?")


def mc_extract(text):
    m = _ANS_RE.findall(text or "")
    if m:
        return m[-1].upper()
    m = _ANS_RE2.findall(text or "")
    return m[-1].upper() if m else None


# ============================================================================================ scoring

class Stopped(Exception):
    pass


def score(url, items, out_path, seed, max_tokens, check, chat, count_tokens, budget, concurrency=4,
          stop_check=None, log=print, on_row=None, request_retries=1, order=None, request=None, judge=None):
    """One answer per item for this seed, checked by check(item, content) -> bool. Resumable: items already in
    out_path for this seed are skipped. chat(url, message, seed, max_tokens) -> evalplus_win.chat_request dict.
    order: indexes into items to answer, in that order (default all, in list order); an item's request seed comes
    from its index in items either way, so it gets the same seed whatever the order. Returns the rows of this seed
    (old and new). For items that are not one user message (a tool-calling check, anneal/bfcl.py):
    request(url, item, seed, max_tokens) stands in for chat and judge(item, response) -> (passed, verdict) for
    check, and sees the whole response (finish reason, tool calls)."""
    done = {(r["id"], r["seed"]) for r in read_jsonl(out_path)}
    want = range(len(items)) if order is None else order
    todo = [(i, items[i]) for i in want if (items[i]["id"], seed) not in done]
    log(f"score: {len(want)} items" + ("" if order is None else f" (of {len(items)}, in a set order)")
        + f", seed {seed}, {len(want) - len(todo)} done, {len(todo)} to do")
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    lock = threading.Lock()
    t0, n_done, tok = time.time(), [0], [0]

    def one(i, it):
        if stop_check and stop_check():
            raise Stopped()
        message = None if request else it["prompt"] if "prompt" in it else mc_message(it)
        for attempt in range(1 + request_retries):
            try:
                r = (request(url, it, 100000 * seed + i, max_tokens) if request
                     else chat(url, message, 100000 * seed + i, max_tokens))
                break
            except Exception as exc:  # noqa: BLE001
                if (stop_check and stop_check()) or attempt == request_retries:
                    raise RuntimeError(f"request for {it['id']} failed: {type(exc).__name__}: {exc}") from exc
                time.sleep(5)
        n_reason = count_tokens(url, r["reasoning"]) if r["reasoning"] else 0
        verdict = None
        if judge:
            ok, verdict = judge(it, r)
            ok = bool(ok)
        elif r["finish_reason"] == "stop" and r["content"]:
            res = check(it, r["content"])
            ok, verdict = (bool(res[0]), res[1]) if isinstance(res, tuple) else (bool(res), None)
        else:
            ok = False
        row = {"id": it["id"], "seed": seed, "pass": ok, "finish_reason": r["finish_reason"],
                "completion_tokens": r["completion_tokens"], "reasoning_tokens": n_reason,
                "budget_hit": bool(budget) and n_reason >= 0.97 * budget, "seconds": r["seconds"],
                "answer": (r.get("answer", r["content"]) or "")[:20000]}
        if verdict is not None:
            row["verdict"] = verdict
        return row

    with ThreadPoolExecutor(max(1, concurrency)) as pool, open(out_path, "a", encoding="utf-8") as fh:
        pending = {}
        it_todo = iter(todo)
        exhausted = False
        while True:
            while not exhausted and len(pending) < concurrency:
                try:
                    i, it = next(it_todo)
                except StopIteration:
                    exhausted = True
                    break
                pending[pool.submit(one, i, it)] = it["id"]
            if not pending:
                break
            finished, _ = wait(pending, return_when=FIRST_COMPLETED)
            for fut in finished:
                pending.pop(fut)
                try:
                    row = fut.result()
                except Stopped:
                    exhausted = True
                    continue
                except Exception:
                    for f in pending:
                        f.cancel()
                    raise
                with lock:
                    fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                    fh.flush()
                    n_done[0] += 1
                    tok[0] += row["completion_tokens"] or 0
                if on_row:
                    on_row(row, n_done[0], len(todo), tok[0] / max(time.time() - t0, 1e-9))
    if stop_check and stop_check():
        raise Stopped()
    return [r for r in read_jsonl(out_path) if r["seed"] == seed]


def gate_check(run_io_tests, run_assert_tests, extract_code, polyglot_judge=None):
    """check(item, content) for the gate's items: write and fix items through polyglot_judge(item, content) ->
    (ok, verdict) (polyglot_check), the old Python kinds through the generate_traces.py runners."""
    def check(item, content):
        if item["kind"] in ("write", "fix"):
            if polyglot_judge is None:
                raise RuntimeError(f"{item['id']}: a six-language gate item needs a polyglot judge")
            return polyglot_judge(item, content)
        code = extract_code(content)
        if item["kind"] == "cc":
            return run_io_tests(code, item["cases"], IO_CASE_TIMEOUT_S)
        return run_assert_tests(code, item["tests"], ASSERT_TIMEOUT_S)
    return check


def polyglot_check(make_executor=None, make_toolchain=None, max_parallel=2):
    """(item, content) -> (ok, verdict) for write and fix items: the answer's program in the item's language, built
    and run in the sandbox (anneal/polyglot.py). The executor and toolchain are made on first use. At most
    max_parallel programs build or run at once: judging takes seconds against minutes of generation, and it keeps a
    few JVMs or compilers from stacking up in RAM next to the model server."""
    from anneal import polyglot as P
    state, lock, slots = {}, threading.Lock(), threading.BoundedSemaphore(max(1, max_parallel))

    def judge(item, content):
        with lock:
            if not state:
                state["ex"] = (make_executor or P.SandboxExecutor)()
                state["tc"] = (make_toolchain or P.Toolchain.windows)()
        code = P.extract_code(content, item["lang"])
        with slots:
            v = P.judge(item["lang"], code, item["cases"], state["ex"], state["tc"], case_timeout_s=IO_CASE_TIMEOUT_S)
        return v.ok, v.stage
    return judge


def mc_check(item, content):
    return mc_extract(content) == item["answer"]


# ============================================================================================ decision

def sign_test_greater(wins, losses):
    """One-sided exact sign test: P(X >= wins) for X ~ Binomial(wins + losses, 1/2)."""
    n = wins + losses
    if n == 0:
        return 1.0
    return sum(math.comb(n, k) for k in range(wins, n + 1)) / 2 ** n


def per_problem(rows):
    """{id: mean pass over seeds} from score rows."""
    acc = {}
    for r in rows:
        acc.setdefault(r["id"], []).append(1.0 if r["pass"] else 0.0)
    return {k: sum(v) / len(v) for k, v in acc.items()}


def compare(best_rows, cand_rows, kinds=None, langs=None):
    """Paired comparison on the problems both have. With one seed a problem is a win, a loss or a tie; with several,
    the per-problem mean decides the sign. kinds / langs keep ids whose first / second part (write/cpp/...) match."""
    b, c = per_problem(best_rows), per_problem(cand_rows)
    ids = sorted(set(b) & set(c))
    if kinds is not None:
        ids = [i for i in ids if i.split("/")[0] in kinds]
    if langs is not None:
        ids = [i for i in ids if i.split("/")[0] in ("write", "fix") and i.split("/")[1] in langs]
    wins = sum(1 for i in ids if c[i] > b[i])
    losses = sum(1 for i in ids if c[i] < b[i])
    n = len(ids)
    return {"n": n, "best": sum(b[i] for i in ids) / n if n else None,
            "candidate": sum(c[i] for i in ids) / n if n else None,
            "diff": (sum(c[i] - b[i] for i in ids) / n) if n else None,
            "wins": wins, "losses": losses, "p_one_sided": sign_test_greater(wins, losses)}


def mc_accuracy(rows):
    return sum(1 for r in rows if r["pass"]) / len(rows) if rows else None


LOSS_LABELS = {"swe": "bug-fix", "terminal": "terminal"}


def loss_paths(rules, losses):
    """The held-out loss paths of the keep rule (decide), Holm-adjusted across the paths that have a comparison
    (21 Oct 2026: bug-fix steps and terminal steps), so adding a path doesn't raise the chance of a false keep:
    {name: {p_one_sided, p_holm, open}}. A path is open when loss_rule is on, the candidate's mean loss is lower and
    its Holm-adjusted one-sided sign-test p is <= loss_alpha. With one path, p_holm is its own p."""
    alpha = rules.get("loss_alpha", rules["alpha"])
    live = sorted(((k, v) for k, v in (losses or {}).items() if v and v.get("n") and v.get("diff") is not None),
                  key=lambda kv: (kv[1]["p_one_sided"], kv[0]))
    out, prev = {}, 0.0
    for j, (k, v) in enumerate(live):
        prev = max(prev, min(1.0, (len(live) - j) * v["p_one_sided"]))
        out[k] = {"p_one_sided": v["p_one_sided"], "p_holm": prev,
                  "open": bool(rules.get("loss_rule")) and v["diff"] < 0 and prev <= alpha}
    return out


def open_loss_paths(rules, losses):
    """Names of the open loss paths (loss_paths), in a fixed order."""
    return sorted(k for k, v in loss_paths(rules, losses).items() if v["open"])


def loss_path_open(rules, loss):
    """Whether the bug-fix loss path alone is open (the rule before the terminal path was added)."""
    return bool(open_loss_paths(rules, {"swe": loss}))


def shuffled_order(items, salt):
    """A fixed shuffle of range(len(items)) from salt and the item ids (same salt, same order on every machine)."""
    return sorted(range(len(items)), key=lambda i: hashlib.sha256(f"{salt}:{items[i]['id']}".encode()).hexdigest())


def early_reject_check(best_rows, cand_rows, ids, max_diff):
    """The interim look of the early reject (loop.yaml gate early_reject): on the gate problems ids, how many more the
    candidate solved than the best (diff, a count), and stop = diff <= max_diff. It only ever rejects early, so it
    can't make a keep more likely than the full test; it skips the rest of the gate pass for rounds that are behind
    or level on the first part."""
    b, c = per_problem(best_rows), per_problem(cand_rows)
    common = [i for i in ids if i in b and i in c]
    diff = sum(c[i] - b[i] for i in common)
    return {"n": len(common), "of": len(ids), "diff": diff, "max_diff": max_diff,
            "wins": sum(1 for i in common if c[i] > b[i]), "losses": sum(1 for i in common if c[i] < b[i]),
            "stop": len(common) == len(ids) and diff <= max_diff}


def guard_check(base_pass, cand_pass, tolerance, alpha):
    """A no-worse check against the base ({problem: 1 passed / 0 failed}, paired per problem): worse when the
    candidate's pass rate is lower by more than tolerance, or it fails more of the problems that changed than chance
    allows (one-sided sign test on its losses, p <= alpha). The same test as the non-coding check (decide)."""
    ids = sorted(set(base_pass) & set(cand_pass))
    wins = sum(1 for i in ids if cand_pass[i] > base_pass[i])
    losses = sum(1 for i in ids if cand_pass[i] < base_pass[i])
    b = sum(base_pass[i] for i in ids) / len(ids) if ids else None
    c = sum(cand_pass[i] for i in ids) / len(ids) if ids else None
    diff = (c - b) if ids else None
    p_worse = sign_test_greater(losses, wins)
    worse = bool(ids) and (diff < -tolerance or p_worse <= alpha)
    return {"n": len(ids), "base": b, "candidate": c, "diff": diff, "wins": wins, "losses": losses,
            "p_worse": p_worse, "tolerance": tolerance, "alpha": alpha, "ok": bool(ids) and not worse}


def compare_loss(best, cand, weights=None):
    """Paired comparison of per-row losses on the same held-out rows ({row: loss}; lower is better), weighted by
    weights ({row: trained tokens}) for the means. A row is a win if the candidate's loss is lower. One-sided sign
    test: is the candidate lower on more rows than chance?"""
    rows = sorted(set(best) & set(cand))
    w = {r: (weights or {}).get(r, 1) for r in rows}
    tw = sum(w.values())
    b = sum(best[r] * w[r] for r in rows) / tw if tw else None
    c = sum(cand[r] * w[r] for r in rows) / tw if tw else None
    wins = sum(1 for r in rows if cand[r] < best[r])
    losses = sum(1 for r in rows if cand[r] > best[r])
    return {"n": len(rows), "best": b, "candidate": c, "diff": (c - b) if rows else None,
            "rel_diff": ((c - b) / b) if rows and b else None, "wins": wins, "losses": losses,
            "p_one_sided": sign_test_greater(wins, losses)}


PART_NAMES = {"write": "write part", "fix": "fix part", "cc": "CodeContests part"}


def decide(best_gate, cand_gate, base_nc, cand_nc, rules, sanity_ok=True, loss=None, losses=None):
    """The keep rule. Keep the candidate only if
      EITHER its gate score (paired per problem against the best so far) is higher and a one-sided exact sign test on
        the problems that changed gives p <= alpha,
      OR (rules loss_rule, 21 Oct 2026) its gate score is not lower AND its loss on held-out steps (losses:
        {"swe": bug-fix steps, "terminal": terminal steps}, each compare_loss against the best; loss alone means the
        bug-fix steps) is lower on more steps than chance, Holm-adjusted across the paths, p <= loss_alpha
        (loss_paths);
    and in both cases all hold:
      - each guarded part (write, fix; cc in the old Python gate) is not worse than the best by more than
        part_tolerance (cc_tolerance in older configs), and not significantly worse (one-sided sign test on its
        losses, p <= alpha);
      - the non-coding check (per question, mean over seeds, paired against the BASE model so drift can't build
        up over rounds) is not worse by more than nc_tolerance, and not significantly worse;
      - the sample generations thought, answered and stopped cleanly.
    Returns a dict with keep, path ("gate" / "loss" / None), the reasons for a reject, and every number: per part
    and, for write and fix items, per language (for the record)."""
    allp = compare(best_gate, cand_gate)
    ids = set(per_problem(best_gate)) & set(per_problem(cand_gate))
    kinds = sorted({i.split("/")[0] for i in ids})
    parts = {k: compare(best_gate, cand_gate, kinds={k}) for k in kinds}
    langs = sorted({i.split("/")[1] for i in ids if i.split("/")[0] in ("write", "fix")})
    by_lang = {lg: compare(best_gate, cand_gate, langs={lg}) for lg in langs}
    nc = compare(base_nc, cand_nc)
    alpha = rules["alpha"]
    tol = rules.get("part_tolerance", rules.get("cc_tolerance", 0.05))
    reasons = []
    if not sanity_ok:
        reasons.append("sample generations did not think, answer and stop cleanly")
    gate_gain = bool(allp["n"]) and allp["diff"] is not None and allp["diff"] > 0 and allp["p_one_sided"] <= alpha
    gate_level = bool(allp["n"]) and allp["diff"] is not None and allp["diff"] >= 0
    loss_alpha = rules.get("loss_alpha", alpha)
    if losses is None:
        losses = {"swe": loss} if loss is not None else {}
    losses = {k: v for k, v in losses.items() if v is not None}
    lp = loss_paths(dict(rules, loss_rule=True), losses)
    fell = sorted(k for k, v in lp.items() if v["open"])
    path = "gate" if gate_gain else ("loss" if rules.get("loss_rule") and gate_level and fell else None)
    if path is None:
        if allp["n"] == 0 or allp["diff"] is None or allp["diff"] <= 0:
            reasons.append(f"gate not higher (diff {allp['diff']})")
        else:
            reasons.append(f"gate gain not clear: one-sided sign test p {allp['p_one_sided']:.4f} > {alpha}")
        if rules.get("loss_rule"):
            if not gate_level:
                reasons.append("held-out loss path closed: the gate score is lower")
            elif not lp:
                reasons.append("held-out loss paths: no held-out loss for this round")
            for k in (sorted(lp) if gate_level else []):  # with the gate level, no path fell here
                lo = losses[k]
                reasons.append(f"{LOSS_LABELS.get(k, k)} loss not clearly lower: {lo['best']:.4f} -> "
                               f"{lo['candidate']:.4f}, {lo['wins']} steps lower / {lo['losses']} higher, p "
                               f"{lo['p_one_sided']:.4f}, Holm-adjusted {lp[k]['p_holm']:.4f} (needs <= {loss_alpha})")
    guarded = [(PART_NAMES[k], parts[k], tol) for k in kinds if k in PART_NAMES]
    for name, c, t in guarded + [("non-coding check", nc, rules["nc_tolerance"])]:
        if not c["n"]:
            continue
        p_worse = sign_test_greater(c["losses"], c["wins"])
        c["p_worse"] = p_worse
        if c["diff"] < -t:
            reasons.append(f"{name} dropped {c['diff']:+.4f} (tolerance {t})")
        elif p_worse <= alpha:
            reasons.append(f"{name} significantly worse: {c['losses']} losses vs {c['wins']} wins, p {p_worse:.4f}")
    out = {"keep": not reasons, "path": path if not reasons else None, "reasons": reasons, "gate": allp,
           "parts": parts, "langs": by_lang, "noncoding": nc, "loss": losses.get("swe"),
           "losses": {k: dict(losses[k], **lp.get(k, {})) for k in losses}, "loss_fell": fell, "rules": dict(rules)}
    for k in ("cc", "oci"):  # the old Python gate's report fields
        if k in parts:
            out[f"gate_{k}"] = parts[k]
    return out
