"""Held-out sets for the keep-if-better loop: the gate and the non-coding check, how they are built and scored.

The gate decides whether a new adapter is kept. It is never trained on and never reported, so choosing adapters on
it doesn't inflate the reported numbers (EvalPlus, LiveCodeBench):
  cc   DeepMind CodeContests valid (117) + test (165): stdin/stdout problems. A problem stays only if one of its own
       Python 3 reference solutions passes our sandboxed runner on the chosen cases (so a fail means the answer, not
       the harness) and it has no input/output files. Prompt = description + the stdin/stdout instruction
       generate_traces.py uses.
  oci  300 OpenCodeInstruct prompts from a shard no training run reads (train-00025), average_test_score 1.0, assert
       tests, reference solution passing in the sandbox, decontaminated against EvalPlus.
Problems that overlap any training data used so far (13-gram or Jaccard > 0.8, prepare_dataset.Decontaminator) are
left out, and every later training source is decontaminated against the gate (decontam_extra.jsonl).

The non-coding check: 100 MMLU-Pro test questions outside computer science (random.Random(42)), multiple choice
with up to 10 options, scored by the letter after "The answer is".

Scoring: one chat request per problem with thinking on (temperature 0.6, top_p 0.95, top_k 20, min_p 0, request seed
100000 * seed + problem index), the answer's first Python block run in the sandbox. Results are JSONL, one row per
problem and seed, resumable.
"""

import bisect
import collections
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
          stop_check=None, log=print, on_row=None, request_retries=1):
    """One answer per item for this seed, checked by check(item, content) -> bool. Resumable: items already in
    out_path for this seed are skipped. chat(url, message, seed, max_tokens) -> evalplus_win.chat_request dict.
    Returns the rows of this seed (old and new)."""
    done = {(r["id"], r["seed"]) for r in read_jsonl(out_path)}
    todo = [(i, it) for i, it in enumerate(items) if (it["id"], seed) not in done]
    log(f"score: {len(items)} items, seed {seed}, {len(items) - len(todo)} done, {len(todo)} to do")
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    lock = threading.Lock()
    t0, n_done, tok = time.time(), [0], [0]

    def one(i, it):
        if stop_check and stop_check():
            raise Stopped()
        message = it["prompt"] if "prompt" in it else mc_message(it)
        for attempt in range(1 + request_retries):
            try:
                r = chat(url, message, 100000 * seed + i, max_tokens)
                break
            except Exception as exc:  # noqa: BLE001
                if (stop_check and stop_check()) or attempt == request_retries:
                    raise RuntimeError(f"request for {it['id']} failed: {type(exc).__name__}: {exc}") from exc
                time.sleep(5)
        n_reason = count_tokens(url, r["reasoning"]) if r["reasoning"] else 0
        ok = bool(check(it, r["content"])) if r["finish_reason"] == "stop" and r["content"] else False
        return {"id": it["id"], "seed": seed, "pass": ok, "finish_reason": r["finish_reason"],
                "completion_tokens": r["completion_tokens"], "reasoning_tokens": n_reason,
                "budget_hit": bool(budget) and n_reason >= 0.97 * budget, "seconds": r["seconds"],
                "answer": (r["content"] or "")[:20000]}

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


def gate_check(run_io_tests, run_assert_tests, extract_code):
    def check(item, content):
        code = extract_code(content)
        if item["kind"] == "cc":
            return run_io_tests(code, item["cases"], IO_CASE_TIMEOUT_S)
        return run_assert_tests(code, item["tests"], ASSERT_TIMEOUT_S)
    return check


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


def compare(best_rows, cand_rows, kinds=None):
    """Paired comparison on the problems both have. With one seed a problem is a win, a loss or a tie; with several,
    the per-problem mean decides the sign."""
    b, c = per_problem(best_rows), per_problem(cand_rows)
    ids = sorted(set(b) & set(c))
    if kinds is not None:
        ids = [i for i in ids if i.split("/")[0] in kinds]
    wins = sum(1 for i in ids if c[i] > b[i])
    losses = sum(1 for i in ids if c[i] < b[i])
    n = len(ids)
    return {"n": n, "best": sum(b[i] for i in ids) / n if n else None,
            "candidate": sum(c[i] for i in ids) / n if n else None,
            "diff": (sum(c[i] - b[i] for i in ids) / n) if n else None,
            "wins": wins, "losses": losses, "p_one_sided": sign_test_greater(wins, losses)}


def mc_accuracy(rows):
    return sum(1 for r in rows if r["pass"]) / len(rows) if rows else None


def decide(best_gate, cand_gate, base_nc, cand_nc, rules, sanity_ok=True):
    """The keep rule. Keep the candidate only if all hold:
      - its gate score (CodeContests + OpenCodeInstruct, paired per problem against the best so far) is higher, and
        a one-sided exact sign test on the problems that changed gives p <= alpha;
      - the CodeContests part alone is not worse than the best by more than cc_tolerance, and not significantly
        worse (one-sided sign test on its losses, p <= alpha);
      - the non-coding check (per question, mean over seeds, paired against the BASE model so drift can't build
        up over rounds) is not worse by more than nc_tolerance, and not significantly worse;
      - the sample generations thought, answered and stopped cleanly.
    rules: alpha, cc_tolerance, nc_tolerance. Returns a dict with keep, the reasons for a reject, and every number."""
    allp = compare(best_gate, cand_gate)
    cc = compare(best_gate, cand_gate, kinds={"cc"})
    oci = compare(best_gate, cand_gate, kinds={"oci"})
    nc = compare(base_nc, cand_nc)
    alpha = rules["alpha"]
    reasons = []
    if not sanity_ok:
        reasons.append("sample generations did not think, answer and stop cleanly")
    if allp["n"] == 0 or allp["diff"] is None or allp["diff"] <= 0:
        reasons.append(f"gate not higher (diff {allp['diff']})")
    elif allp["p_one_sided"] > alpha:
        reasons.append(f"gate gain not clear: one-sided sign test p {allp['p_one_sided']:.4f} > {alpha}")
    for name, c, tol in (("CodeContests part", cc, rules["cc_tolerance"]), ("non-coding check", nc, rules["nc_tolerance"])):
        if not c["n"]:
            continue
        p_worse = sign_test_greater(c["losses"], c["wins"])
        c["p_worse"] = p_worse
        if c["diff"] < -tol:
            reasons.append(f"{name} dropped {c['diff']:+.4f} (tolerance {tol})")
        elif p_worse <= alpha:
            reasons.append(f"{name} significantly worse: {c['losses']} losses vs {c['wins']} wins, p {p_worse:.4f}")
    return {"keep": not reasons, "reasons": reasons, "gate": allp, "gate_cc": cc, "gate_oci": oci,
            "noncoding": nc, "rules": dict(rules)}
