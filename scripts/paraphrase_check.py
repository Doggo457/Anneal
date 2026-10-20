"""Reworded-copy check: flag training rows whose question is a reworded copy of a test problem.

The 13-gram and Jaccard check (prepare_dataset.py --decontaminate, anneal/evalsets.py OverlapIndex) only catches
near-verbatim copies. On reworded HumanEval a 10-gram check scored F1 0 and embedding search 0.938 (Yang et al.,
"Rethinking Benchmark and Contamination for Language Models with Rephrased Samples", arXiv:2311.04850). This is that
paper's embedding stage, without its judge model (my choice, 20 Oct 2026: no second model judges the data).
Embeddings come from FastFlowLM's EmbeddingGemma-300M on the NPU, through any OpenAI-compatible /v1/embeddings server.

Method
  1. Embed every test problem (HumanEval+ and MBPP+ from evalplus, the gate's CodeContests and OpenCodeInstruct
     problems, the MMLU-Pro check, and any --test file) and the question (first user message) of every training row:
     data/processed/*/train.jsonl (the first run) and data/processed/loop/*/train.jsonl (the loop's pools).
  2. A cut-off per test set, the higher of
       t_pos  the 5th percentile of the cosine between each HumanEval prompt and its published GPT-4 rewording
              (lm-sys/llm-decontaminator, 164 pairs, pinned), so 95% of known rewordings score at or above it;
       t_neg  the 95th percentile of each test problem's cosine to its nearest *other* problem in the same set
              (distinct problems by construction), so a row is flagged only when it is closer to a test problem than
              different problems in that set usually are to each other. Sets under 20 problems use t_pos alone.
  3. A training question is flagged when it is among a test problem's --k nearest and at or above that set's cut-off.
     Every row with that question is flagged.

Outputs (data/eval/paraphrase/, never committed: they hold row text)
  blocked_keys.txt  row keys (anneal/rowkeys.py) the loop never hands out (scripts/loop.py blocked_keys)
  flags.jsonl       one line per flagged row: source, row index, key, test set, test id, cosine, cut-off
  pairs.jsonl       the flagged pairs with both texts, for reading
  report.json, card.md   calibration, cut-offs, counts, speed, and the training-speed A/B; copied without any row
                    text to results/paraphrase_check/
  stops.jsonl       one line per stopped attempt (exit code, reason, training-speed A/B so far), also copied there
  cache_<tag>.npz   embeddings (float16), so a stopped run resumes where it stopped (_b<N> when --max-bytes isn't 4000)

Runs beside training, so it stops itself (state kept, rerun to resume):
  exit 3  free RAM under --min-free-gb (default 3)
  exit 4  the training run's tokens_per_second (--watch-metrics) fell more than --max-slowdown (default 2%) below its
          mean over the last 12 steps before this run started (mean of all steps since the start, once there are 4)
  exit 5  the embedder failed its self-check: rewordings must rank above unrelated pairs (AUC >= 0.90) and the same
          text must embed the same twice. FastFlowLM's EmbeddingGemma output is wrong in 1.0.4 on Windows (issue #732)
          and non-discriminative from 0.9.45 to 1.0.8 (issue #789), so use FastFlowLM 0.9.43.
  exit 6  the embeddings server can't be reached
  exit 2  bad input (missing test sets, no training rows, calibration file checksum mismatch)

  python scripts/paraphrase_check.py --url http://127.0.0.1:52625 --model embed-gemma:300m \\
      --tag flm0.9.43-embedgemma300m --watch-metrics results/first-run/train/metrics.jsonl
"""
import argparse
import glob
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.request

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from anneal import evalsets as E  # noqa: E402
from anneal.rowkeys import question_text, row_key  # noqa: E402

CALIB_COMMIT = "931834aa0990d9558a06c5d34daf4ad3852a72ae"  # lm-sys/llm-decontaminator, Apache-2.0
CALIB_FILES = {  # name: (path in that repo, sha256)
    "original": ("data/test/HumanEval.jsonl", "3b8e13687aaa1762a7aa593a11048563420dfc58d5553c7fa6edd9b2ebe533da"),
    "reworded": ("data/rephrase/humaneval_python.jsonl", "e704c0868cda18a0008c16baa5d23f55807d15c82550af8273c3ec44c13428e0"),
}
CALIB_URL = "https://raw.githubusercontent.com/lm-sys/llm-decontaminator/{commit}/{path}"
POS_PERCENTILE = 5      # t_pos: 95% of known rewordings at or above it
NEG_PERCENTILE = 95     # t_neg: above 95% of nearest-different-problem similarities
MIN_SET_FOR_NEG = 20
MIN_AUC = 0.90
MIN_REPEAT_COS = 0.999
PROBES = ["Write a function that returns the sum of two integers.",
          "Given an array of n integers, print the length of its longest strictly increasing subsequence.",
          "Which of the following best describes the role of the judiciary in a constitutional democracy?",
          "def is_prime(n):\n    \"\"\"Return True if n is a prime number.\"\"\""]
EXIT_BAD_INPUT, EXIT_LOW_RAM, EXIT_SLOWED, EXIT_BAD_EMBEDDER, EXIT_NO_SERVER = 2, 3, 4, 5, 6


class Stop(Exception):
    def __init__(self, code, msg):
        super().__init__(msg)
        self.code = code


def log(msg):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def text_hash(text):
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


# ============================================================================================ inputs

def cut_prompt(code):
    """A HumanEval-style program cut to its prompt: everything up to the end of the first docstring after the first
    `def` (the calibration files hold whole programs, our test sets hold prompts). None if it has no such docstring."""
    i = code.find("def ")
    if i < 0:
        return None
    starts = [p for p in (code.find('"""', i), code.find("'''", i)) if p >= 0]
    if not starts:
        return None
    q = min(starts)
    close = code.find(code[q:q + 3], q + 3)
    return code[:close + 3].strip() if close >= 0 else None


def load_calibration(cache_dir, offline_dir=None):
    """[(original prompt, reworded prompt)] for the 164 HumanEval problems, downloaded once at the pinned commit and
    checked against the pinned sha256."""
    texts = {}
    for name, (path, sha) in CALIB_FILES.items():
        local = os.path.join(offline_dir or cache_dir, os.path.basename(path))
        if not os.path.exists(local):
            if offline_dir:
                raise Stop(EXIT_BAD_INPUT, f"calibration file {local} is missing")
            os.makedirs(cache_dir, exist_ok=True)
            url = CALIB_URL.format(commit=CALIB_COMMIT, path=path)
            log(f"downloading {url}")
            with urllib.request.urlopen(url, timeout=120) as r:
                data = r.read()
            with open(local + ".tmp", "wb") as fh:
                fh.write(data)
            os.replace(local + ".tmp", local)
        data = open(local, "rb").read()
        got = hashlib.sha256(data).hexdigest()
        if got != sha:
            raise Stop(EXIT_BAD_INPUT, f"{local}: sha256 {got} is not the pinned {sha}; delete it to download again")
        texts[name] = [json.loads(line)["text"] for line in data.decode("utf-8").splitlines() if line.strip()]
    if len(texts["original"]) != len(texts["reworded"]):
        raise Stop(EXIT_BAD_INPUT, "calibration files differ in length")
    # Some rewordings are empty or moved the docstring into comments (18 of 164 at the pinned commit): those pairs
    # can't be cut to a prompt without keeping the solution, so they are left out.
    pairs = [(cut_prompt(a), cut_prompt(b)) for a, b in zip(texts["original"], texts["reworded"])]
    pairs = [(a, b) for a, b in pairs if a and b]
    if len(pairs) < 100:
        raise Stop(EXIT_BAD_INPUT, f"only {len(pairs)} calibration pairs could be cut to prompts")
    return pairs


def load_test_sets(extra=(), use_evalplus=True):
    """{set name: [(id, text)]}: HumanEval+, MBPP+, the gate (CodeContests and OpenCodeInstruct parts), the MMLU-Pro
    check, plus NAME=PATH files (JSONL with "id" and "prompt", "question" or "text")."""
    sets = {}
    if use_evalplus:
        try:
            from evalplus.data import get_human_eval_plus, get_mbpp_plus
        except ImportError:
            raise Stop(EXIT_BAD_INPUT, "evalplus is not installed in this Python (pip install -r requirements.txt)")
        sets["humaneval+"] = [(k, v["prompt"]) for k, v in get_human_eval_plus().items()]
        sets["mbpp+"] = [(k, v["prompt"]) for k, v in get_mbpp_plus().items()]
    gate = E.read_jsonl(E.GATE_PATH)
    nc = E.read_jsonl(E.NONCODING_PATH)
    if not gate or not nc:
        raise Stop(EXIT_BAD_INPUT, f"{E.GATE_PATH} or {E.NONCODING_PATH} is missing: run scripts/build_evalsets.py first")
    for kind in sorted({g["kind"] for g in gate}):
        sets[f"gate-{kind}"] = [(g["id"], g["prompt"]) for g in gate if g["kind"] == kind]
    sets["mmlu-pro"] = [(q["id"], q["question"]) for q in nc]
    for spec in extra:
        name, _, path = spec.partition("=")
        rows = E.read_jsonl(path)
        if not name or not rows:
            raise Stop(EXIT_BAD_INPUT, f"--test {spec}: expected NAME=PATH to a non-empty JSONL file")
        sets[name] = [(str(r.get("id", i)), next(str(r[k]) for k in ("prompt", "question", "text") if k in r))
                      for i, r in enumerate(rows)]
    return {n: [(i, t) for i, t in items if t.strip()] for n, items in sets.items()}


def discover_train(root):
    """[(source name, path)]: the first run's sets and the loop's pools. Folders with a dot in their name are set-aside
    copies (e.g. loop/agent_sessions.placeholders-20oct, rows filtered out of a pool) and are skipped."""
    proc = os.path.join(root, "data", "processed")
    found = [(os.path.basename(os.path.dirname(p)), p) for p in sorted(glob.glob(os.path.join(proc, "*", "train.jsonl")))]
    found += [("loop/" + os.path.basename(os.path.dirname(p)), p)
              for p in sorted(glob.glob(os.path.join(proc, "loop", "*", "train.jsonl")))]
    return [(n, p) for n, p in found if "." not in n.split("/")[-1]]


def load_train(sources):
    """{source: [(row index, row key, question hash)]} and {question hash: question}. Row indices count non-empty
    lines, as the loop's pick_rows does."""
    rows, texts = {}, {}
    for name, path in sources:
        items = []
        with open(path, encoding="utf-8") as fh:
            idx = 0
            for line in fh:
                if not line.strip():
                    continue
                r = json.loads(line)
                q = question_text(r).strip()
                if q:
                    h = text_hash(q)
                    texts.setdefault(h, q)
                    items.append((idx, row_key(r), h))
                idx += 1
        rows[name] = items
        log(f"training rows: {name}: {idx} rows, {len(items)} with a question ({path})")
    return rows, texts


# ============================================================================================ embedder + cache

class Embedder:
    """OpenAI-compatible POST /v1/embeddings. Returns L2-normalised float32 rows."""

    def __init__(self, url, model, batch=16, timeout=900, max_bytes=4000, retries=4):
        url = url.rstrip("/")
        self.url = (url[:-3] if url.endswith("/v1") else url) + "/v1/embeddings"
        self.model, self.batch, self.timeout, self.max_bytes, self.retries = model, batch, timeout, max_bytes, retries
        self.texts = self.chars = 0
        self.seconds = 0.0

    def cut(self, text):
        """At most max_bytes of UTF-8. Every token covers at least one byte (Gemma's tokenizer falls back to bytes), so
        a server with a window of max_bytes + a few tokens can never get an input that is too long."""
        b = text.encode("utf-8")
        return text if len(b) <= self.max_bytes else b[:self.max_bytes].decode("utf-8", errors="ignore")

    def _post(self, texts):
        body = json.dumps({"model": self.model, "input": [self.cut(t) for t in texts]}).encode("utf-8")
        req = urllib.request.Request(self.url, data=body, headers={"Content-Type": "application/json"})
        delay = 5
        for attempt in range(self.retries + 1):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    raw = r.read()
                try:
                    return json.loads(raw.decode("utf-8"))
                except ValueError:
                    raise Stop(EXIT_BAD_EMBEDDER, f"{self.url} returned something that isn't JSON: {raw[:200]!r}")
            except (urllib.error.URLError, ConnectionError, TimeoutError, OSError) as exc:
                http = isinstance(exc, urllib.error.HTTPError)
                if http and (exc.code < 500 or attempt == self.retries):  # it answered, so it isn't unreachable
                    raise Stop(EXIT_BAD_EMBEDDER, f"{self.url}: HTTP {exc.code}: {exc.read()[:300]!r}")
                if attempt == self.retries:
                    raise Stop(EXIT_NO_SERVER, f"{self.url} unreachable after {self.retries + 1} tries: {exc}")
                log(f"embeddings request failed ({exc}); retrying in {delay} s")
                time.sleep(delay)
                delay *= 2

    def embed(self, texts):
        out = []
        for s in range(0, len(texts), self.batch):
            chunk = texts[s:s + self.batch]
            t0 = time.time()
            resp = self._post(chunk)
            self.seconds += time.time() - t0
            data = resp.get("data") if isinstance(resp, dict) else None
            if not data or len(data) != len(chunk):
                err = resp.get("error") if isinstance(resp, dict) else None
                raise Stop(EXIT_BAD_EMBEDDER, f"embeddings server returned no vectors ({err or 'no embedding model loaded'}"
                                              "; FastFlowLM needs `flm serve <chat model> --embed 1`)")
            vecs = np.array([d["embedding"] for d in sorted(data, key=lambda d: d.get("index", 0))], dtype=np.float32)
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            if vecs.ndim != 2 or not np.all(np.isfinite(vecs)) or np.any(norms == 0):
                raise Stop(EXIT_BAD_EMBEDDER, "embeddings server returned empty, zero or non-finite vectors")
            out.append(vecs / norms)
            self.texts += len(chunk)
            self.chars += sum(len(self.cut(t)) for t in chunk)
        return np.concatenate(out) if out else np.zeros((0, 0), np.float32)


class Cache:
    """Embeddings by text hash, saved atomically as float16 so a stopped run resumes."""

    def __init__(self, path):
        self.path, self.vec, self.dirty = path, {}, 0
        if os.path.exists(path):
            with np.load(path, allow_pickle=False) as z:
                for k, v in zip(z["keys"], z["vecs"]):
                    self.vec[str(k)] = v
            log(f"cache: {len(self.vec)} embeddings from {path}")

    def add(self, hashes, vecs):
        for h, v in zip(hashes, vecs):
            self.vec[h] = v.astype(np.float16)
        self.dirty += len(hashes)

    def save(self):
        if not self.dirty:
            return
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        keys = list(self.vec)
        tmp = self.path + ".tmp"
        with open(tmp, "wb") as fh:
            np.savez(fh, keys=np.array(keys), vecs=np.stack([self.vec[k] for k in keys]))
        os.replace(tmp, self.path)
        self.dirty = 0

    def matrix(self, hashes):
        m = np.stack([self.vec[h] for h in hashes]).astype(np.float32)
        return m / np.linalg.norm(m, axis=1, keepdims=True)


# ============================================================================================ guards

def free_ram_gb():
    try:
        import psutil
        return psutil.virtual_memory().available / 2 ** 30
    except ImportError:
        pass
    if os.name == "nt":
        import ctypes

        class MS(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong)] + \
                       [(n, ctypes.c_ulonglong) for n in ("tp", "ap", "tf", "af", "tv", "av", "ae")]
        st = MS()
        st.dwLength = ctypes.sizeof(MS)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st))
        return st.ap / 2 ** 30
    with open("/proc/meminfo") as fh:
        for line in fh:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 2 ** 20
    return float("inf")


class SlowdownGuard:
    """Training speed from a run's metrics.jsonl ("t", "tokens_per_second" per step). Baseline: the last `base_steps`
    steps logged in the `window_s` before the start. check() stops once the mean of ALL steps since the start (at
    least `min_steps` of them) is more than max_slowdown below the baseline.

    Why these numbers: on the first run (167 steps of ~290 s, 20 Oct) tokens_per_second has a 1.5% coefficient of
    variation with no step-to-step correlation. Replayed on those steps, a 6-step baseline against a sliding 3-step mean
    stops a run that changes nothing 19% of the time within 12 steps; 12 steps against the running mean of 4 or more
    stops 0.7% of the time and still catches a 3% slowdown 97-100% of the time."""

    def __init__(self, path, max_slowdown=0.02, base_steps=12, window_s=7200, min_steps=4, now=None):
        self.path, self.max_slowdown, self.min_steps = path, max_slowdown, min_steps
        self.start = time.time() if now is None else now
        self.offset, self.before, self.during = 0, [], []
        self._read()
        base = [r for r in self.before if r["t"] >= self.start - window_s][-base_steps:]
        if len(base) < 3:
            raise Stop(EXIT_BAD_INPUT, f"{path}: fewer than 3 training steps logged in the {window_s / 3600:g} h before "
                                       "now, so there is no baseline speed; leave out --watch-metrics if training "
                                       "isn't running")
        self.base = [r["tokens_per_second"] for r in base]
        self.baseline = float(np.mean(self.base))
        log(f"training speed baseline: {self.baseline:.1f} tok/s over {len(self.base)} steps")

    def _read(self):
        if not os.path.exists(self.path):
            return
        with open(self.path, "rb") as fh:  # bytes: Windows writes \r\n, so text-mode offsets would drift
            fh.seek(self.offset)
            chunk = fh.read()
        end = chunk.rfind(b"\n") + 1  # only whole lines
        self.offset += end
        for line in chunk[:end].splitlines():
            try:
                r = json.loads(line.decode("utf-8", errors="replace"))
            except ValueError:
                continue
            if not isinstance(r, dict):
                continue
            if isinstance(r.get("tokens_per_second"), (int, float)) and isinstance(r.get("t"), (int, float)):
                (self.before if r["t"] <= self.start else self.during).append(r)

    def check(self):
        self._read()
        if len(self.during) >= self.min_steps:
            m = float(np.mean([r["tokens_per_second"] for r in self.during]))
            if m < self.baseline * (1 - self.max_slowdown):
                raise Stop(EXIT_SLOWED, f"training slowed to {m:.1f} tok/s (mean of the {len(self.during)} steps since "
                                        f"the start) from {self.baseline:.1f}: over {self.max_slowdown:.0%}")

    def summary(self):
        d = [r["tokens_per_second"] for r in self.during]
        out = {"metrics": self.path, "baseline_tok_s": round(self.baseline, 2), "baseline_steps": len(self.base),
               "during_steps": len(d), "max_slowdown": self.max_slowdown}
        if d:
            mb, md = self.baseline, float(np.mean(d))
            out.update(during_tok_s=round(md, 2), slowdown_pct=round(100 * (1 - md / mb), 2))
            if len(d) >= 2:  # normal-approximation 95% interval on the difference of the two means
                se = float(np.sqrt(np.var(self.base, ddof=1) / len(self.base) + np.var(d, ddof=1) / len(d)))
                out["slowdown_ci95_pct"] = [round(100 * (mb - md - 1.96 * se) / mb, 2),
                                            round(100 * (mb - md + 1.96 * se) / mb, 2)]
        return out


class Guards:
    def __init__(self, min_free_gb, slowdown=None):
        self.min_free_gb, self.slowdown = min_free_gb, slowdown
        self.min_seen = float("inf")

    def check(self):
        free = free_ram_gb()
        self.min_seen = min(self.min_seen, free)
        if free < self.min_free_gb:
            raise Stop(EXIT_LOW_RAM, f"free RAM {free:.1f} GB is under {self.min_free_gb} GB")
        if self.slowdown:
            self.slowdown.check()


# ============================================================================================ maths

def auc(pos, neg):
    """P(a positive scores above a negative), ties half."""
    pos, neg = np.asarray(pos, np.float64), np.sort(np.asarray(neg, np.float64))
    below = np.searchsorted(neg, pos, side="left")
    equal = np.searchsorted(neg, pos, side="right") - below
    return float((below + 0.5 * equal).sum() / (len(pos) * len(neg)))


def topk(T, X, k, chunk=8192):
    """For each row of T: the k rows of X with the highest dot product, as (values, indices), best first."""
    m = T.shape[0]
    bv = np.full((m, 0), -np.inf, np.float32)
    bi = np.zeros((m, 0), np.int64)
    for s0 in range(0, X.shape[0], chunk):
        sims = T @ X[s0:s0 + chunk].T
        kk = min(k, sims.shape[1])
        part = np.argpartition(-sims, kk - 1, axis=1)[:, :kk]
        bv = np.concatenate([bv, np.take_along_axis(sims, part, 1)], 1)
        bi = np.concatenate([bi, part + s0], 1)
        if bv.shape[1] > k:
            keep = np.argpartition(-bv, k - 1, axis=1)[:, :k]
            bv, bi = np.take_along_axis(bv, keep, 1), np.take_along_axis(bi, keep, 1)
    order = np.argsort(-bv, axis=1)
    return np.take_along_axis(bv, order, 1), np.take_along_axis(bi, order, 1)


def calibrate(orig, reworded):
    """Self-check and t_pos from the calibration pairs (rows L2-normalised and aligned)."""
    sims = orig @ reworded.T
    pos = np.diag(sims).copy()
    neg = sims[~np.eye(len(pos), dtype=bool)]
    return {"pairs": int(len(pos)), "auc": round(auc(pos, neg), 4),
            "top1_retrieval": round(float(np.mean(np.argmax(sims, axis=1) == np.arange(len(pos)))), 4),
            "pos_median": round(float(np.median(pos)), 4), "neg_median": round(float(np.median(neg)), 4),
            "t_pos": round(float(np.percentile(pos, POS_PERCENTILE)), 4), "_pos": pos, "_neg": neg}


def set_threshold(T, t_pos):
    """(cut-off, t_neg) for one test set."""
    if T.shape[0] < MIN_SET_FOR_NEG:
        return t_pos, None
    s = T @ T.T
    np.fill_diagonal(s, -np.inf)
    t_neg = float(np.percentile(s.max(axis=1), NEG_PERCENTILE))
    return max(t_pos, t_neg), round(t_neg, 4)


# ============================================================================================ run

def embed_missing(emb, cache, hashes, texts, guards, label):
    todo = [h for h in dict.fromkeys(hashes) if h not in cache.vec]
    if not todo:
        log(f"{label}: all {len(set(hashes))} embeddings cached")
        return
    log(f"{label}: embedding {len(todo)} texts ({len(set(hashes)) - len(todo)} cached)")
    t0, last = time.time(), time.time()
    step = emb.batch
    for s in range(0, len(todo), step):
        guards.check()
        chunk = todo[s:s + step]
        cache.add(chunk, emb.embed([texts[h] for h in chunk]))
        if cache.dirty >= 512:
            cache.save()
        if time.time() - last >= 60 or s + step >= len(todo):
            done = min(s + step, len(todo))
            rate = done / max(time.time() - t0, 1e-9)
            log(f"{label}: {done}/{len(todo)} ({rate:.2f} texts/s, about {(len(todo) - done) / max(rate, 1e-9) / 60:.0f} "
                f"min left, free RAM {free_ram_gb():.1f} GB)")
            last = time.time()
    cache.save()


def run(args, emb=None, calib_pairs=None, test_sets=None, sources=None):
    """The whole check. A Stop (a guard, a bad embedder, bad input) first appends one line to stops.jsonl with what
    was measured so far, including the training-speed A/B, then is raised again."""
    out_dir = args.out_dir or os.path.join(E.EVAL_DIR, "paraphrase")
    os.makedirs(out_dir, exist_ok=True)
    started = time.time()
    report = {"started": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(started)), "tag": args.tag,
              "url": args.url, "model": args.model, "k": args.k, "max_bytes": args.max_bytes,
              "pos_percentile": POS_PERCENTILE,
              "neg_percentile": NEG_PERCENTILE, "calibration": f"lm-sys/llm-decontaminator@{CALIB_COMMIT}"}
    ctx = {"emb": emb}
    try:
        return _run(args, out_dir, started, report, ctx, calib_pairs, test_sets, sources)
    except Stop as stop:
        record_stop(stop, report, ctx, started, out_dir, args.results_dir)
        raise


def record_stop(stop, report, ctx, started, out_dir, results_dir):
    """One JSON line per stopped attempt, appended (a rerun keeps earlier stops). No row text."""
    emb, guards, slowdown = ctx.get("emb"), ctx.get("guards"), ctx.get("slowdown")
    secs = getattr(emb, "seconds", 0) or 0
    rec = {k: report[k] for k in ("started", "tag", "url", "model")}
    rec.update(stopped=time.strftime("%Y-%m-%d %H:%M:%S"), exit=stop.code, reason=str(stop),
               minutes=round((time.time() - started) / 60, 1), self_check=report.get("self_check"),
               training_speed=slowdown.summary() if slowdown else None,
               min_free_ram_gb=round(guards.min_seen, 2) if guards and np.isfinite(guards.min_seen) else None,
               embedder_speed={"texts": getattr(emb, "texts", 0), "seconds": round(secs, 1),
                               "texts_per_s": round(emb.texts / secs, 3) if secs else None})
    for d in filter(None, [out_dir, results_dir]):
        try:
            os.makedirs(d, exist_ok=True)
            with open(os.path.join(d, "stops.jsonl"), "a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec) + "\n")
        except OSError as exc:  # recording must never hide the reason for the stop
            log(f"could not write {d}/stops.jsonl: {exc}")


def _run(args, out_dir, started, report, ctx, calib_pairs, test_sets, sources):
    calib_pairs = calib_pairs if calib_pairs is not None else load_calibration(os.path.join(out_dir, "calib"), args.calib_dir)
    test_sets = test_sets if test_sets is not None else load_test_sets(args.test)
    sources = sources if sources is not None else ([tuple(s.split("=", 1)) for s in args.train] or discover_train(ROOT))
    if not sources:
        raise Stop(EXIT_BAD_INPUT, "no training rows found (data/processed/*/train.jsonl)")
    train_rows, train_texts = load_train(sources)
    for n, items in test_sets.items():
        log(f"test set {n}: {len(items)} problems")

    emb = ctx["emb"] = ctx["emb"] or Embedder(args.url, args.model, batch=args.batch, max_bytes=args.max_bytes)
    slowdown = ctx["slowdown"] = SlowdownGuard(args.watch_metrics, args.max_slowdown) if args.watch_metrics else None
    guards = ctx["guards"] = Guards(args.min_free_gb, slowdown)
    guards.check()
    # the cut length is part of the cache's identity: a text cut differently embeds differently
    cache = Cache(os.path.join(out_dir, f"cache_{args.tag}.npz" if args.max_bytes == 4000 else
                               f"cache_{args.tag}_b{args.max_bytes}.npz"))

    try:
        # ---- self-check: same text twice, and against the probes this cache was built with
        probes = emb.embed(PROBES)
        again = emb.embed(PROBES)
        rep = float(np.min(np.sum(probes * again, axis=1)))
        if rep < MIN_REPEAT_COS:
            raise Stop(EXIT_BAD_EMBEDDER, f"the same text embedded twice gave cosine {rep:.4f} (< {MIN_REPEAT_COS})")
        ph = ["probe:" + text_hash(p) for p in PROBES]
        if all(h in cache.vec for h in ph):
            old = cache.matrix(ph)
            drift = float(np.min(np.sum(old * probes, axis=1)))
            if drift < MIN_REPEAT_COS:
                raise Stop(EXIT_BAD_EMBEDDER, f"the cache was made by a different embedder (probe cosine {drift:.4f}); "
                                              "use a new --tag")
        else:
            cache.add(ph, probes)

        # ---- calibration
        c_texts = {}
        c_hashes = []
        for a, b in calib_pairs:
            for t in (a, b):
                h = text_hash(t)
                c_texts[h] = t
                c_hashes.append(h)
        embed_missing(emb, cache, c_hashes, c_texts, guards, "calibration")
        cal = calibrate(cache.matrix(c_hashes[0::2]), cache.matrix(c_hashes[1::2]))
        report["self_check"] = {"repeat_cos_min": round(rep, 5), **{k: v for k, v in cal.items() if not k.startswith("_")}}
        log(f"self-check: AUC {cal['auc']}, top-1 retrieval {cal['top1_retrieval']}, reworded median {cal['pos_median']} "
            f"vs unrelated {cal['neg_median']}, t_pos {cal['t_pos']}")
        if cal["auc"] < MIN_AUC:
            raise Stop(EXIT_BAD_EMBEDDER, f"embedder fails the self-check: AUC {cal['auc']} < {MIN_AUC} on reworded "
                                          "HumanEval (FastFlowLM 1.0.4 and 0.9.45-1.0.8 return wrong EmbeddingGemma "
                                          "vectors, issues #732 and #789: use 0.9.43)")
        t_pos = cal["t_pos"]

        # ---- embed tests, then training questions
        t_texts, t_hash = {}, {}
        for n, items in test_sets.items():
            t_hash[n] = []
            for _, t in items:
                h = text_hash(t)
                t_texts[h] = t
                t_hash[n].append(h)
        embed_missing(emb, cache, [h for hs in t_hash.values() for h in hs], t_texts, guards, "test problems")
        u_hashes = list(train_texts)
        embed_missing(emb, cache, u_hashes, train_texts, guards, "training questions")
    finally:
        cache.save()

    # ---- search
    X = cache.matrix(u_hashes)
    rows_by_hash = {}
    for src, items in train_rows.items():
        for idx, key, h in items:
            rows_by_hash.setdefault(h, []).append((src, idx, key))
    flags, pairs, per_set = [], [], {}
    for n, items in test_sets.items():
        T = cache.matrix(t_hash[n])
        thr, t_neg = set_threshold(T, t_pos)
        vals, idx = topk(T, X, args.k)
        hit_items = 0
        for qi, (tid, ttext) in enumerate(items):
            hit = False
            for v, j in zip(vals[qi], idx[qi]):
                if v < thr:
                    break
                hit = True
                h = u_hashes[j]
                for src, ridx, key in rows_by_hash[h]:
                    flags.append({"source": src, "row": ridx, "key": key, "test_set": n, "test_id": tid,
                                  "cosine": round(float(v), 4), "cutoff": round(thr, 4)})
                pairs.append({"test_set": n, "test_id": tid, "cosine": round(float(v), 4), "test": ttext,
                              "train": train_texts[h][:4000], "rows": [[s, i] for s, i, _ in rows_by_hash[h]]})
            hit_items += hit
        recall = float(np.mean(cal["_pos"] >= thr))
        per_set[n] = {"problems": len(items), "t_neg": t_neg, "cutoff": round(thr, 4),
                      "reworded_recall_at_cutoff": round(recall, 4),
                      "problems_with_a_flagged_row": hit_items,
                      "flagged_test_ids": sorted({f["test_id"] for f in flags if f["test_set"] == n})}
        log(f"{n}: cut-off {thr:.4f} (t_pos {t_pos}, t_neg {t_neg}), recall on rewordings {recall:.1%}, "
            f"{hit_items}/{len(items)} problems with a flagged training row")

    per_source = {}
    for src, items in train_rows.items():
        keys = {(f["row"]) for f in flags if f["source"] == src}
        per_source[src] = {"rows_checked": len(items), "rows_flagged": len(keys)}
    blocked = sorted({f["key"] for f in flags if f["key"]})

    E.write_jsonl(os.path.join(out_dir, "flags.jsonl"), flags)
    E.write_jsonl(os.path.join(out_dir, "pairs.jsonl"), pairs)
    tmp = os.path.join(out_dir, "blocked_keys.txt.tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write("".join(k + "\n" for k in blocked))
    os.replace(tmp, os.path.join(out_dir, "blocked_keys.txt"))

    report.update(per_set=per_set, per_source=per_source, blocked_keys=len(blocked), flags=len(flags),
                  unique_training_questions=len(u_hashes),
                  embedder_speed={"texts": emb.texts, "seconds": round(emb.seconds, 1),
                                  "texts_per_s": round(emb.texts / emb.seconds, 3) if emb.seconds else None,
                                  "chars_per_s": round(emb.chars / emb.seconds, 1) if emb.seconds else None},
                  min_free_ram_gb=round(guards.min_seen, 2), minutes=round((time.time() - started) / 60, 1),
                  training_speed=slowdown.summary() if slowdown else None,
                  started_epoch=round(started, 1), finished_epoch=round(time.time(), 1))
    write_outputs(report, out_dir, args.results_dir)
    log(f"done: {len(flags)} flagged rows, {len(blocked)} keys blocked -> {out_dir}")
    return report


def write_outputs(report, out_dir, results_dir):
    lines = [f"# Reworded-copy check ({report['started']})", "",
             f"Embedder `{report['model']}` at {report['url']} (tag `{report['tag']}`), k = {report['k']}. "
             f"Calibration: {report['calibration']}, HumanEval prompts against their GPT-4 rewordings.", ""]
    sc = report.get("self_check") or {}
    if sc:
        lines += [f"Self-check: AUC {sc['auc']}, top-1 retrieval {sc['top1_retrieval']}, reworded median cosine "
                  f"{sc['pos_median']} against unrelated {sc['neg_median']}, same text twice {sc['repeat_cos_min']}. "
                  f"t_pos (5th percentile of reworded pairs) = {sc['t_pos']}.", ""]
    lines += ["| Test set | Problems | t_neg | Cut-off | Recall on rewordings | Problems with a flagged row |",
              "| --- | --- | --- | --- | --- | --- |"]
    for n, s in report["per_set"].items():
        lines.append(f"| {n} | {s['problems']} | {s['t_neg']} | {s['cutoff']} | {s['reworded_recall_at_cutoff']:.1%} | "
                     f"{s['problems_with_a_flagged_row']} |")
    flagged = [(n, s["flagged_test_ids"]) for n, s in report["per_set"].items() if s["flagged_test_ids"]]
    if flagged:
        lines += ["", "Test problems with a flagged training row (report EvalPlus with and without these):"]
        lines += [f"- {n}: {', '.join(ids)}" for n, ids in flagged]
    lines += ["", "| Training source | Rows checked | Rows flagged |", "| --- | --- | --- |"]
    for n, s in report["per_source"].items():
        lines.append(f"| {n} | {s['rows_checked']} | {s['rows_flagged']} |")
    sp = report["embedder_speed"]
    lines += ["", f"{report['blocked_keys']} row keys blocked for the loop. Embedding speed: {sp['texts_per_s']} texts/s "
                  f"({sp['texts']} texts in {sp['seconds']} s). Lowest free RAM seen: {report['min_free_ram_gb']} GB. "
                  f"Took {report['minutes']} min."]
    ts = report.get("training_speed")
    if ts:
        lines += ["", f"Training speed A/B ({ts['metrics']}): {ts['baseline_tok_s']} tok/s over {ts['baseline_steps']} steps "
                      f"before, {ts.get('during_tok_s')} tok/s over {ts['during_steps']} steps during"
                      + (f" ({ts['slowdown_pct']}% slower)." if "slowdown_pct" in ts else ".")]
    card = "\n".join(lines) + "\n"
    for d in filter(None, [out_dir, results_dir]):
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "report.json"), "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=1)
        with open(os.path.join(d, "card.md"), "w", encoding="utf-8") as fh:
            fh.write(card)


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:52625", help="embeddings server (FastFlowLM's default port)")
    ap.add_argument("--model", default="embed-gemma:300m")
    ap.add_argument("--tag", default="flm0.9.43-embedgemma300m",
                    help="names the embedding cache: change it whenever the embedder or its version changes")
    ap.add_argument("--k", type=int, default=5, help="nearest training questions considered per test problem")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--max-bytes", type=int, default=4000,
                    help="each text is cut to this many bytes of UTF-8: about 1,000-1,600 tokens, inside EmbeddingGemma's "
                         "2,048-token window, and never more than this many tokens; part of the cache's identity")
    ap.add_argument("--min-free-gb", type=float, default=3.0)
    ap.add_argument("--watch-metrics", default=None, help="a running training run's metrics.jsonl (the 2%% rule)")
    ap.add_argument("--max-slowdown", type=float, default=0.02)
    ap.add_argument("--train", action="append", default=[], help="NAME=PATH (default: data/processed/*/train.jsonl "
                                                                  "and data/processed/loop/*/train.jsonl)")
    ap.add_argument("--test", action="append", default=[], help="extra test set NAME=PATH (JSONL: id + prompt)")
    ap.add_argument("--calib-dir", default=None, help="folder already holding the two calibration files")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--results-dir", default=os.path.join(ROOT, "results", "paraphrase_check"))
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    try:
        import psutil
        p = psutil.Process()
        p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS if os.name == "nt" else 10)
    except Exception:  # priority is a courtesy, not a requirement
        pass
    try:
        run(args)
        return 0
    except Stop as s:
        log(f"stopped (exit {s.code}): {s}")
        return s.code


if __name__ == "__main__":
    sys.exit(main())
