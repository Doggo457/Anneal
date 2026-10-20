"""Tests for scripts/paraphrase_check.py without an NPU, a model or the network: a local HTTP server stands in for
FastFlowLM's /v1/embeddings with a word-hashing embedder (rewordings share words, so they score high), plus broken
servers like FastFlowLM's known bugs (random and collapsed vectors).

  python tests/test_paraphrase_check.py        (or: python -m pytest tests/test_paraphrase_check.py)
"""
import hashlib
import json
import os
import re
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import paraphrase_check as P  # noqa: E402
from anneal import evalsets as E  # noqa: E402
from anneal.rowkeys import question_text, row_key  # noqa: E402

DIM = 512
WORDS = [f"w{i}" for i in range(2000)]  # synthetic vocabulary: unrelated problems share almost no words


def word_vec(text):
    v = np.zeros(DIM, np.float32)
    for w in re.findall(r"[a-z0-9]+", text.lower()):
        v[int(hashlib.md5(w.encode()).hexdigest(), 16) % DIM] += 1.0
    return v if v.any() else np.ones(DIM, np.float32)


def problem(seed, n=12):
    rng = np.random.default_rng(seed)
    return " ".join(rng.choice(WORDS, n, replace=False))


def reword(text, keep=0.75, seed=0):
    """Keep most words, swap the rest: what a rewording looks like to a bag-of-words embedder."""
    rng = np.random.default_rng(seed + 1000)
    return " ".join(w if rng.random() < keep else str(rng.choice(WORDS)) for w in text.split())


class Server:
    """mode: "good" (word hashing), "random" (new random vector per call, like a broken kernel), "collapsed"
    (nearly the same vector for every text, like FastFlowLM issue #789)."""

    def __init__(self, mode="good"):
        self.mode, self.calls, self.texts = mode, 0, 0
        outer = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                assert self.path == "/v1/embeddings" and body["model"] == "embed-gemma:300m"
                outer.calls += 1
                outer.texts += len(body["input"])
                data = [{"object": "embedding", "index": i, "embedding": outer.vec(t).tolist()}
                        for i, t in enumerate(body["input"])]
                out = json.dumps({"object": "list", "data": data}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(out)))
                self.end_headers()
                self.wfile.write(out)

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def vec(self, text):
        if self.mode == "random":
            return np.random.default_rng().normal(size=DIM).astype(np.float32)
        if self.mode == "collapsed":
            return np.ones(DIM, np.float32) + 1e-3 * word_vec(text)
        return word_vec(text)

    def close(self):
        self.httpd.shutdown()


def calib_pairs(n=60):
    return [(problem(5000 + i), reword(problem(5000 + i), seed=i)) for i in range(n)]


def make_world(tmp):
    """Test sets (25 problems, so t_neg applies), a first-run set with an exact copy of problem 0, a loop pool with a
    rewording of problem 1, a rewording that is too loose to count, and unrelated rows."""
    tests = {"humaneval+": [(f"HumanEval/{i}", problem(i)) for i in range(25)],
             "mmlu-pro": [(f"mmlupro/{i}", problem(100 + i)) for i in range(25)]}
    row = lambda q, a="a": {"messages": [{"role": "user", "content": q}, {"role": "assistant", "content": a}]}  # noqa: E731
    first = [row(problem(200 + i)) for i in range(30)] + [row(tests["humaneval+"][0][1])]
    pool = [row(problem(300 + i)) for i in range(40)]
    pool.insert(7, row(reword(tests["humaneval+"][1][1], keep=0.9, seed=7)))
    pool.insert(20, row(reword(tests["mmlu-pro"][3][1], keep=0.2, seed=8)))   # mostly different words
    pool.append(row(tests["humaneval+"][1][1] + "\n", "b"))                       # whitespace variant, another answer
    paths = {"mix_first_2k": os.path.join(tmp, "data", "processed", "mix_first_2k", "train.jsonl"),
             "loop/tachibana4_v4pro": os.path.join(tmp, "data", "processed", "loop", "tachibana4_v4pro", "train.jsonl")}
    E.write_jsonl(paths["mix_first_2k"], first)
    E.write_jsonl(paths["loop/tachibana4_v4pro"], pool)
    return tests, list(paths.items()), first, pool


def args_for(tmp, url, **kw):
    a = P.parse_args(["--url", url, "--tag", "test", "--out-dir", os.path.join(tmp, "out"),
                      "--results-dir", os.path.join(tmp, "results"), "--batch", "8", "--min-free-gb", "0"])
    for k, v in kw.items():
        setattr(a, k, v)
    return a


# ============================================================================================ tests

def test_cut_prompt():
    prog = 'from typing import List\n\n\ndef f(x: List[int]) -> int:\n    """ Sum.\n    >>> f([1])\n    1\n    """\n    return sum(x)\n'
    assert P.cut_prompt(prog).endswith('"""') and "return sum" not in P.cut_prompt(prog)
    assert P.cut_prompt("def g(a):\n    '''Doc.'''\n    return a").endswith("'''Doc.'''")
    assert P.cut_prompt("# no docstring\ndef h():\n    return 1") is None and P.cut_prompt("") is None


def test_auc_and_topk():
    assert P.auc([0.9, 0.8], [0.1, 0.2]) == 1.0 and P.auc([0.1], [0.9]) == 0.0 and P.auc([0.5], [0.5]) == 0.5
    rng = np.random.default_rng(0)
    T, X = rng.normal(size=(7, 16)).astype(np.float32), rng.normal(size=(1000, 16)).astype(np.float32)
    v, i = P.topk(T, X, 5, chunk=97)
    want = np.argsort(-(T @ X.T), axis=1)[:, :5]
    assert (i == want).all() and np.allclose(v, np.take_along_axis(T @ X.T, want, 1))
    v1, i1 = P.topk(T, X[:3], 5)  # fewer rows than k
    assert i1.shape == (7, 3)


def test_question_text():
    assert question_text({"messages": [{"role": "system", "content": "s"}, {"role": "user", "content": "q"}]}) == "q"
    assert question_text({"messages": [{"role": "user", "content": [{"type": "text", "text": "a"},
                                                                     {"type": "image", "url": "x"}]}]}) == "a"
    assert question_text({"messages": [{"role": "assistant", "content": "x"}]}) == ""


def test_discover_train_skips_set_aside_folders():
    with tempfile.TemporaryDirectory() as tmp:
        for d in ("mix_first_2k", "opencodereasoning.old", "loop/tachibana4_v4pro", "loop/agent_sessions",
                  "loop/agent_sessions.placeholders-20oct"):
            E.write_jsonl(os.path.join(tmp, "data", "processed", *d.split("/"), "train.jsonl"), [{"messages": []}])
        names = [n for n, _ in P.discover_train(tmp)]
        assert names == ["mix_first_2k", "loop/agent_sessions", "loop/tachibana4_v4pro"], names


def test_end_to_end_and_resume():
    srv = Server("good")
    with tempfile.TemporaryDirectory() as tmp:
        try:
            tests, sources, first, pool = make_world(tmp)
            rep = P.run(args_for(tmp, srv.url), calib_pairs=calib_pairs(), test_sets=tests, sources=sources)
            out = os.path.join(tmp, "out")
            flags = E.read_jsonl(os.path.join(out, "flags.jsonl"))
            got = {(f["source"], f["row"], f["test_id"]) for f in flags}
            assert ("mix_first_2k", 30, "HumanEval/0") in got, got           # exact copy in the first run
            assert ("loop/tachibana4_v4pro", 7, "HumanEval/1") in got, got   # rewording in a pool
            assert ("loop/tachibana4_v4pro", len(pool) - 1, "HumanEval/1") in got, got  # same question, other answer
            assert not any(f["row"] == 20 and f["source"] == "loop/tachibana4_v4pro" for f in flags), flags
            assert len({(s, r) for s, r, _ in got}) == 3, got                # nothing unrelated
            blocked = open(os.path.join(out, "blocked_keys.txt")).read().split()
            assert sorted(blocked) == sorted({row_key(first[30]), row_key(pool[7]), row_key(pool[-1])})
            assert rep["self_check"]["auc"] > 0.95 and rep["per_set"]["humaneval+"]["t_neg"] is not None
            assert rep["per_set"]["humaneval+"]["flagged_test_ids"] == ["HumanEval/0", "HumanEval/1"]
            assert rep["per_source"]["loop/tachibana4_v4pro"] == {"rows_checked": len(pool), "rows_flagged": 2}
            # the committed copy holds no row text
            for f in ("report.json", "card.md"):
                text = open(os.path.join(tmp, "results", f), encoding="utf-8").read()
                assert problem(1) not in text and problem(200) not in text and "HumanEval/1" in text, f
            pairs = E.read_jsonl(os.path.join(out, "pairs.jsonl"))
            assert any(p["test_id"] == "HumanEval/1" and p["train"].startswith(reword(problem(1), keep=0.9, seed=7)[:20])
                       for p in pairs)
            # a rerun embeds only the probes (twice): everything else comes from the cache
            before = srv.texts
            rep2 = P.run(args_for(tmp, srv.url), calib_pairs=calib_pairs(), test_sets=tests, sources=sources)
            assert srv.texts - before == 2 * len(P.PROBES) and rep2["flags"] == rep["flags"]
        finally:
            srv.close()


def test_broken_embedders_stop():
    for mode in ("random", "collapsed"):
        srv = Server(mode)
        with tempfile.TemporaryDirectory() as tmp:
            try:
                tests, sources, _, _ = make_world(tmp)
                try:
                    P.run(args_for(tmp, srv.url), calib_pairs=calib_pairs(), test_sets=tests, sources=sources)
                    raise AssertionError(f"{mode}: should have stopped")
                except P.Stop as s:
                    assert s.code == P.EXIT_BAD_EMBEDDER, (mode, s.code, str(s))
                assert not os.path.exists(os.path.join(tmp, "out", "blocked_keys.txt"))
            finally:
                srv.close()


def test_cache_from_another_embedder_refused():
    srv = Server("good")
    with tempfile.TemporaryDirectory() as tmp:
        try:
            tests, sources, _, _ = make_world(tmp)
            P.run(args_for(tmp, srv.url), calib_pairs=calib_pairs(), test_sets=tests, sources=sources)
            srv.vec = lambda t: word_vec(t[::-1])  # a different embedder behind the same tag
            try:
                P.run(args_for(tmp, srv.url), calib_pairs=calib_pairs(), test_sets=tests, sources=sources)
                raise AssertionError("should have refused the cache")
            except P.Stop as s:
                assert s.code == P.EXIT_BAD_EMBEDDER and "different embedder" in str(s)
        finally:
            srv.close()


def test_low_ram_stops_then_resumes():
    srv = Server("good")
    real = P.free_ram_gb
    with tempfile.TemporaryDirectory() as tmp:
        try:
            tests, sources, _, _ = make_world(tmp)
            calls = {"n": 0}

            def fake():
                calls["n"] += 1
                return 10.0 if calls["n"] < 15 else 1.0
            P.free_ram_gb = fake
            try:
                P.run(args_for(tmp, srv.url, min_free_gb=3.0), calib_pairs=calib_pairs(), test_sets=tests,
                      sources=sources)
                raise AssertionError("should have stopped")
            except P.Stop as s:
                assert s.code == P.EXIT_LOW_RAM, s.code
            rec = json.loads(open(os.path.join(tmp, "results", "stops.jsonl"), encoding="utf-8").read())
            assert rec["exit"] == P.EXIT_LOW_RAM and rec["min_free_ram_gb"] == 1.0 and rec["embedder_speed"]["texts"], rec
            cache = P.Cache(os.path.join(tmp, "out", "cache_test.npz"))
            partial = len(cache.vec)
            assert partial > len(P.PROBES)  # what was embedded before the stop was saved
            P.free_ram_gb = lambda: 10.0
            before = srv.texts
            rep = P.run(args_for(tmp, srv.url, min_free_gb=3.0), calib_pairs=calib_pairs(), test_sets=tests,
                        sources=sources)
            assert rep["flags"] >= 3 and srv.texts - before < rep["unique_training_questions"] + 200
        finally:
            P.free_ram_gb = real
            srv.close()


def test_slowdown_guard():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "metrics.jsonl")
        now = time.time()
        with open(path, "wb") as fh:  # Windows line endings, as the training run writes them there
            for i in range(6):
                fh.write(json.dumps({"step": i, "t": now - 600 + 60 * i, "tokens_per_second": 100.0}).encode() + b"\r\n")
            fh.write(json.dumps({"event": "train_begin", "t": now - 700}).encode() + b"\r\n")
        g = P.SlowdownGuard(path, max_slowdown=0.02, now=now)
        assert g.baseline == 100.0 and len(g.base) == 6

        def add(*speeds):
            with open(path, "ab") as fh:
                for s in speeds:
                    fh.write(json.dumps({"t": time.time() + 1, "tokens_per_second": s}).encode() + b"\r\n")
        add(99.0, 98.5, 99.0)
        g.check()  # fewer than 4 steps since the start: no decision yet
        with open(path, "ab") as fh:
            fh.write(b'{"t": 1, "tokens_per_')  # a half-written line: unreadable once the next line is glued on
        add(95.0, 95.0)  # the first 95 is lost to the glued line; mean of 99, 98.5, 99, 95 = 97.9 (> 2% down)
        try:
            g.check()
            raise AssertionError("should have stopped")
        except P.Stop as s:
            assert s.code == P.EXIT_SLOWED and "slowed" in str(s) and "4 steps" in str(s), str(s)
        sm = g.summary()
        assert sm["during_steps"] == 4 and abs(sm["slowdown_pct"] - 2.125) < 0.01, sm
        lo, hi = sm["slowdown_ci95_pct"]
        assert lo < 2.125 < hi, sm

        # the running mean decides, not the last few steps: 1.4% down keeps going, one slow step more tips it
        path2 = os.path.join(tmp, "metrics2.jsonl")
        with open(path2, "wb") as fh:
            for i in range(12):
                fh.write(json.dumps({"t": now - 3000 + 240 * i, "tokens_per_second": 100.0}).encode() + b"\r\n")
            fh.write(json.dumps({"t": now - 9000, "tokens_per_second": 50.0}).encode() + b"\r\n")  # too old
        g2 = P.SlowdownGuard(path2, max_slowdown=0.02, now=now)
        assert g2.baseline == 100.0 and len(g2.base) == 12
        with open(path2, "ab") as fh:
            for v in (99.5, 98.5, 98.5, 98.0):
                fh.write(json.dumps({"t": now + 1, "tokens_per_second": v}).encode() + b"\r\n")
        g2.check()
        with open(path2, "ab") as fh:
            fh.write(json.dumps({"t": now + 2, "tokens_per_second": 94.0}).encode() + b"\r\n")
        try:
            g2.check()
            raise AssertionError("should have stopped")
        except P.Stop as s:
            assert s.code == P.EXIT_SLOWED and "5 steps" in str(s), str(s)
        try:
            P.SlowdownGuard(os.path.join(tmp, "missing.jsonl"))
            raise AssertionError("no baseline should stop")
        except P.Stop as s:
            assert s.code == P.EXIT_BAD_INPUT


def test_slowdown_stop_is_recorded():
    """The 2% rule stops a real run: its A/B numbers must survive in stops.jsonl (local and results copies)."""
    srv = Server("good")
    with tempfile.TemporaryDirectory() as tmp:
        try:
            tests, sources, _, _ = make_world(tmp)
            path = os.path.join(tmp, "metrics.jsonl")
            now = time.time()
            with open(path, "wb") as fh:
                for i in range(6):
                    fh.write(json.dumps({"t": now - 600 + 60 * i, "tokens_per_second": 100.0}).encode() + b"\r\n")
                for i in range(4):  # steps logged after the start, 10% slower
                    fh.write(json.dumps({"t": now + 1000 + i, "tokens_per_second": 90.0}).encode() + b"\r\n")
            try:
                P.run(args_for(tmp, srv.url, watch_metrics=path), calib_pairs=calib_pairs(), test_sets=tests,
                      sources=sources)
                raise AssertionError("should have stopped")
            except P.Stop as s:
                assert s.code == P.EXIT_SLOWED, s.code
            for d in ("out", "results"):
                recs = [json.loads(x) for x in open(os.path.join(tmp, d, "stops.jsonl"), encoding="utf-8")]
                assert len(recs) == 1 and recs[0]["exit"] == P.EXIT_SLOWED, recs
                ts = recs[0]["training_speed"]
                assert ts["baseline_tok_s"] == 100.0 and ts["during_tok_s"] == 90.0 and ts["slowdown_pct"] == 10.0, ts
                assert recs[0]["min_free_ram_gb"] is not None and "training slowed" in recs[0]["reason"]
            assert not os.path.exists(os.path.join(tmp, "out", "report.json"))
        finally:
            srv.close()


def test_unreachable_server():
    e = P.Embedder("http://127.0.0.1:9", "embed-gemma:300m", retries=0, timeout=2)
    try:
        e.embed(["x"])
        raise AssertionError("should have stopped")
    except P.Stop as s:
        assert s.code == P.EXIT_NO_SERVER


def test_long_texts_cut_and_server_errors():
    """Texts are cut to --max-bytes of UTF-8 (EmbeddingGemma reads 2,048 tokens); a server that keeps failing with HTTP 500
    (llama-server's "input is too large") is a bad embedder, not an unreachable one."""
    seen = []

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen.extend(len(t) for t in body["input"])
            if any(len(t) > 50 for t in body["input"]):
                out, code = b'{"error": {"message": "input is too large to process"}}', 500
            else:
                out, code = json.dumps({"data": [{"index": i, "embedding": [1.0, 0.0]} for i, _ in
                                                  enumerate(body["input"])]}).encode(), 200
            self.send_response(code)
            self.send_header("Content-Length", str(len(out)))
            self.end_headers()
            self.wfile.write(out)
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{httpd.server_address[1]}"
    try:
        e = P.Embedder(url, "m", max_bytes=40)
        e.embed(["x" * 500, "short"])
        assert seen == [40, 5], seen
        cut = e.cut("é" * 30)  # 2 bytes a character: 20 whole characters, never half of one
        assert cut == "é" * 20 and len(cut.encode("utf-8")) == 40, cut
        try:
            P.Embedder(url, "m", max_bytes=500, retries=1).embed(["y" * 500])
            raise AssertionError("should have stopped")
        except P.Stop as s:
            assert s.code == P.EXIT_BAD_EMBEDDER and "too large" in str(s), (s.code, str(s))
        assert P.parse_args([]).max_bytes == 4000
    finally:
        httpd.shutdown()


def test_calibration_file_pinned():
    with tempfile.TemporaryDirectory() as d:
        for _, (path, _) in P.CALIB_FILES.items():
            open(os.path.join(d, os.path.basename(path)), "w").write('{"text": "def f():\\n    \\"\\"\\"x\\"\\"\\""}\n')
        try:
            P.load_calibration(d, offline_dir=d)
            raise AssertionError("checksum mismatch should stop")
        except P.Stop as s:
            assert s.code == P.EXIT_BAD_INPUT and "sha256" in str(s)


def main():
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS {name}")
        except Exception:
            import traceback
            failed += 1
            print(f"FAIL {name}")
            traceback.print_exc()
    print(f"{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
