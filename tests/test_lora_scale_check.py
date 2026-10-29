"""Tests for scripts/lora_scale_check.py without a GPU, a model or Windows: the draw, the scale probe, both parts and
the summary against a fake llama-server (scale-dependent answers), a fake loop folder and a fake val_loss.py.

  python tests/test_lora_scale_check.py        (or: python -m pytest tests/test_lora_scale_check.py)
"""

import contextlib
import hashlib
import json
import math
import os
import shutil
import sys
import tempfile
import threading
import types
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

_stub = types.ModuleType("anneal.sandbox")
_stub.require = lambda: None
_stub.ensure_private_folder_closed = lambda log: None
_stub.run = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("sandbox.run must not be called in tests"))
_stub.TEST_MEM_BYTES = 0
sys.modules.setdefault("anneal.sandbox", _stub)
import anneal  # noqa: E402
anneal.sandbox = sys.modules["anneal.sandbox"]

import lora_scale_check as S  # noqa: E402
from anneal import evalsets as E  # noqa: E402

L = S.L
REAL_CONFIG = L.CONFIG


# ============================================================================================ fake llama-server

def h(*parts):
    return int(hashlib.sha256(":".join(map(str, parts)).encode()).hexdigest()[:8], 16)


class FakeServer:
    """llama.Server's interface over an HTTP server whose answers depend on the LoRA scale: reasoning gets longer and
    passes come back as the scale falls. ignore_scale: POST /lora-adapters is accepted but changes nothing."""
    ignore_scale = False
    instances = []

    def __init__(self, flags, log_path, port=18090, binary=None, **kw):
        self.flags, self.scale, self.applied = list(flags), 1.0, 1.0
        assert "--lora" in self.flags
        FakeServer.instances.append(self)
        srv = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def send(self, obj):
                b = json.dumps(obj).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(b)))
                self.end_headers()
                self.wfile.write(b)

            def do_GET(self):
                if self.path == "/health":
                    return self.send({"status": "ok"})
                if self.path == "/lora-adapters":
                    return self.send([{"id": 0, "path": "x.gguf", "scale": srv.scale}])
                self.send_error(404)

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])) or b"{}")
                s = srv.applied
                if self.path == "/lora-adapters":
                    srv.scale = float(body[0]["scale"])
                    if not FakeServer.ignore_scale:
                        srv.applied = srv.scale
                    return self.send({"success": True})
                if self.path == "/tokenize":
                    return self.send({"tokens": body["content"].split()})
                if self.path == "/completion":
                    probs = [{"id": 5, "token": "a", "logprob": -1.0 + 0.5 * s, "bytes": [97],
                              "top_logprobs": [{"id": 5, "token": "a", "logprob": -1.0 + 0.5 * s}]}] * 4
                    return self.send({"content": "aaaa" if s > 0.7 else "aaab", "completion_probabilities": probs})
                if self.path == "/v1/chat/completions":
                    seed, prompt = body["seed"], body["messages"][0]["content"]
                    n_reason = int((2000 + h(prompt) % 3000) * (1.6 - s))  # x0.6 at scale 1 -> x1.6 at scale 0
                    ok = h(prompt, seed) % 100 < 50 + 30 * (1 - s)
                    return self.send({"choices": [{"message": {"content": "OK" if ok else "NO",
                                                               "reasoning_content": "w " * n_reason},
                                                   "finish_reason": "stop"}],
                                      "usage": {"completion_tokens": n_reason + 50}})
                self.send_error(404)

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        self.load_s = 0.0

    def describe(self):
        return " ".join(self.flags)

    def __enter__(self):
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        return self

    def __exit__(self, *a):
        self.httpd.shutdown()
        self.httpd.server_close()


def fake_val_loss(cmd, cwd=None, stdout=None, stderr=None, env=None):
    """scripts/val_loss.py --no-base: adapter loss = base - 0.1 x (lora_alpha / 32), base from the row key."""
    a, rest = {}, list(cmd[3:])
    while rest:
        k = rest.pop(0)
        a[k] = True if k == "--no-base" else rest.pop(0)
    assert a.get("--no-base") is True and a["--max-rows"] == "200" and cmd[2].endswith("val_loss.py")
    ac = json.load(open(os.path.join(a["--adapter"], "adapter_config.json")))
    rows = [json.loads(x) for x in open(a["--val-file"])]
    per = [{"key": r["key"], "trained": 100,
            "adapter_loss": 1.0 + (h(r["key"]) % 50) / 100 - 0.1 * ac["lora_alpha"] / 32} for r in rows]
    os.makedirs(a["--run-dir"], exist_ok=True)
    json.dump({"scored": ["adapter"], "per_row": per}, open(os.path.join(a["--run-dir"], "summary.json"), "w"))
    return types.SimpleNamespace(returncode=0)


# ============================================================================================ a fake loop folder

def make_world(tmp, n_write=23, n_fix=17, rejects=1):
    gate = [{"id": f"{k}/{lang}/cc/test/{i}", "kind": k, "lang": lang, "prompt": f"problem {k} {i}"}
            for k, n in (("write", n_write), ("fix", n_fix)) for i in range(n)
            for lang in [("cpp", "java", "go")[i % 3]]]
    gpath = os.path.join(tmp, "gate.jsonl")
    E.write_jsonl(gpath, gate)
    E.GATE_PATH = gpath
    S.LOOP_DIR = os.path.join(tmp, "loop")
    S.OUT_ROOT = os.path.join(tmp, "research")
    r2 = os.path.join(S.LOOP_DIR, "rounds", "r002")
    os.makedirs(os.path.join(r2, "stages"))
    os.makedirs(os.path.join(r2, "train", "adapter"))
    json.dump({"r": 16, "lora_alpha": 32, "alpha_pattern": {}, "use_rslora": False},
              open(os.path.join(r2, "train", "adapter", "adapter_config.json"), "w"))
    open(os.path.join(r2, "train", "adapter", "adapter_model.safetensors"), "wb").write(os.urandom(4096))
    open(os.path.join(r2, "r002.gguf"), "wb").write(b"gguf")

    def answers(scale, path):  # the fake server's rule, written as stored rows
        rows = []
        for i, it in enumerate(gate):
            seed = 100000 + i
            n = int((2000 + h(it["prompt"]) % 3000) * (1.6 - scale))
            rows.append({"id": it["id"], "seed": 1, "pass": h(it["prompt"], seed) % 100 < 50 + 30 * (1 - scale),
                         "finish_reason": "stop", "completion_tokens": n + 50, "reasoning_tokens": n,
                         "budget_hit": False})
        E.write_jsonl(path, rows)

    answers(0.0, os.path.join(S.LOOP_DIR, "rounds", "r000", "gate.seed1.jsonl"))
    answers(1.0, os.path.join(r2, "gate.seed1.jsonl"))
    cfg = L.load_config(REAL_CONFIG)
    cfg["pause_after_rejects_in_row"] = 1
    state = {"gate_sha": L.file_sha(gpath), "gate_server": L.gate_server(cfg), "toolchains": "fp",
             "rejects_in_row": rejects,
             "base": {"gate_rows": os.path.join(S.LOOP_DIR, "rounds", "r000", "gate.seed1.jsonl"), "gate_seed": 1}}
    for kind, hh in L.HELDOUT.items():
        f = os.path.join(tmp, f"{kind}_heldout.jsonl")
        E.write_jsonl(f, [{"key": f"{kind}-{i}"} for i in range(30)])
        cfg[hh["key"]] = dict(cfg.get(hh["key"]) or {}, file=f)
        summ = os.path.join(r2, hh["stage"], "summary.json")
        os.makedirs(os.path.dirname(summ))
        per = [{"key": f"{kind}-{i}", "trained": 100, "base_loss": 1.0 + (h(f"{kind}-{i}") % 50) / 100,
                "adapter_loss": 1.0 + (h(f"{kind}-{i}") % 50) / 100 - 0.1} for i in range(30)]
        json.dump({"per_row": per}, open(summ, "w"))
        json.dump({"sha": L.file_sha(f), "summary": summ},
                  open(os.path.join(r2, "stages", f"{hh['stage']}.done.json"), "w"))
        state["base"][hh["key"]] = {"path": summ, "field": "base_loss", "round": 0}
    json.dump({"gguf": os.path.join(r2, "r002.gguf")}, open(os.path.join(r2, "stages", "convert.done.json"), "w"))
    json.dump({"gate_rows": os.path.join(r2, "gate.seed1.jsonl"), "gate_seed": 1},
              open(os.path.join(r2, "stages", "evaluate.done.json"), "w"))
    cpath = os.path.join(tmp, "loop.yaml")
    import yaml
    yaml.safe_dump(cfg, open(cpath, "w"))
    L.CONFIG = cpath
    json.dump(state, open(os.path.join(S.LOOP_DIR, "state.json"), "w"))
    return gate


@contextlib.contextmanager
def world(**kw):
    """A fake loop folder with every outside dependency faked; everything patched is put back afterwards."""
    saved = [(S, "llama_running"), (S, "toolchain_check"), (S, "subprocess"), (S, "LOOP_DIR"), (S, "OUT_ROOT"),
             (S.llama, "Server"), (L.llama, "gpu_driver"), (L.Loop, "gate_checker"), (L, "CONFIG"), (E, "GATE_PATH"),
             (sys, "argv")]
    old = [(o, k, getattr(o, k)) for o, k in saved]
    tmp = tempfile.mkdtemp()
    try:
        S.llama_running = lambda: False
        S.toolchain_check = lambda gate, state: "fp"
        S.llama.Server = FakeServer
        S.subprocess = types.SimpleNamespace(run=fake_val_loss, list2cmdline=lambda c: " ".join(c), STDOUT=None)
        L.llama.gpu_driver = lambda: L.llama.LEGACY_GPU_DRIVER
        L.Loop.gate_checker = lambda self: (lambda item, content: content == "OK")
        FakeServer.ignore_scale = False
        yield make_world(tmp, **kw)
    finally:
        for o, k, v in old:
            setattr(o, k, v)
        FakeServer.ignore_scale = False
        shutil.rmtree(tmp)


def run(*argv):
    sys.argv = ["lora_scale_check.py", *argv]
    return S.main()


# ============================================================================================ tests

def test_quotas_and_draw():
    assert S.quotas({"write": 115, "fix": 85}, 60) == {"write": 35, "fix": 25}
    assert S.quotas({"a": 1, "b": 1}, 1) == {"a": 1, "b": 0}
    gate = [{"id": f"w{i}", "kind": "write"} for i in range(115)] + [{"id": f"f{i}", "kind": "fix"} for i in range(85)]
    d = S.draw(gate, 60)
    assert len(d) == 60 == len(set(d)) and sum(gate[i]["kind"] == "fix" for i in d) == 30
    try:
        S.draw(gate, 180)
        raise AssertionError("90 fix problems drawn from 85")
    except ValueError:
        pass
    assert d == S.draw(list(gate), 60)  # fixed
    assert {gate[i]["id"] for i in d} == {gate[::-1][i]["id"] for i in S.draw(gate[::-1], 60)}


def test_probe_diff():
    def probe(*toks):
        return [{"text_sha": "x", "steps": [{"tok": t, "nums": [v, v - 1]} for t, v in toks]}]
    assert S.probe_diff(probe((1, -0.1), (2, -0.2)), probe((1, -0.1), (2, -0.2))) == 0
    assert abs(S.probe_diff(probe((1, -0.1), (2, -0.2)), probe((1, -0.13), (2, -0.2))) - 0.03) < 1e-9
    # parted at the second token: that token still counts, later ones don't
    assert abs(S.probe_diff(probe((1, -0.1), (2, -0.2), (3, -5.0)), probe((1, -0.1), (7, -0.25), (3, -0.1)))
               - 0.05) < 1e-9
    assert S.probe_diff([{"text_sha": "a", "steps": []}], [{"text_sha": "b", "steps": []}]) == math.inf
    assert S.probe_diff([{"text_sha": "a", "steps": []}], [{"text_sha": "a", "steps": []}]) == 0


def test_full_run_and_resume():
    with world() as gate:
        assert run("--n", "20", "--plan") == 0
        assert run("--n", "20") == 0
        out = os.path.join(S.OUT_ROOT, "r002_x0.50")
        rows = E.read_jsonl(os.path.join(out, "gate.seed1.jsonl"))
        assert len(rows) == 20 and len({r["id"] for r in rows}) == 20
        sel = json.load(open(os.path.join(out, "selection.json")))["ids"]
        assert {r["id"] for r in rows} == set(sel)
        # every answer came at scale 0.5: the stored rows match the fake server's rule at 0.5 exactly
        idx = {it["id"]: i for i, it in enumerate(gate)}
        for r in rows:
            n = int((2000 + h(gate[idx[r["id"]]]["prompt"]) % 3000) * 1.1)
            assert r["reasoning_tokens"] == n, (r, n)
        summ = json.load(open(os.path.join(out, "summary.json")))
        g = summ["gate"]["all"]
        assert g["n"] == 20 and abs(g["thinking_gap_closed"] - 0.5) < 0.01, g  # x1.1: halfway from x0.6 to x1.6
        assert g["base"]["reasoning_mean"] > g["part"]["reasoning_mean"] > g["full"]["reasoning_mean"]
        for kind in L.HELDOUT:
            s = summ["loss"][kind]
            assert s["rows"] == 30 and abs(s["gain_kept"] - 0.5) < 1e-6, s
        ac = json.load(open(os.path.join(out, "adapter", "adapter_config.json")))
        assert ac["lora_alpha"] == 16 and isinstance(ac["lora_alpha"], int)
        src = json.load(open(os.path.join(S.LOOP_DIR, "rounds", "r002", "train", "adapter", "adapter_config.json")))
        assert src["lora_alpha"] == 32  # the round's own adapter untouched
        probe = [json.load(open(os.path.join(out, f))) for f in os.listdir(out) if f.startswith("probe-")]
        assert len(probe) == 1 and probe[0]["ok"] and probe[0]["scale_now"] == {"0": 0.5}
        # a second run answers nothing again and starts no server
        FakeServer.instances.clear()
        assert run("--n", "20") == 0
        assert not FakeServer.instances and len(E.read_jsonl(os.path.join(out, "gate.seed1.jsonl"))) == 20


def test_probe_fails_when_scale_does_nothing():
    with world():
        FakeServer.ignore_scale = True
        try:
            run("--n", "10", "--part", "gate")
            raise AssertionError("should have stopped")
        except SystemExit as exc:
            assert "scale probe failed" in str(exc), exc
        assert not E.read_jsonl(os.path.join(S.OUT_ROOT, "r002_x0.50", "gate.seed1.jsonl"))


def test_refuses_when_loop_would_train():
    with world(rejects=0):
        for part in ("loss", "gate"):
            try:
                run("--n", "10", "--part", part)
                raise AssertionError("should have refused")
            except SystemExit as exc:
                assert "would train a round" in str(exc), exc
        assert not os.path.exists(os.path.join(S.OUT_ROOT, "r002_x0.50", "adapter"))


def test_refuses_other_gate_server():
    with world():
        path = os.path.join(S.LOOP_DIR, "state.json")
        with open(path) as fh:
            st = json.load(fh)
        st["gate_server"]["budget"] = 16384
        with open(path, "w") as fh:
            json.dump(st, fh)
        try:
            run("--n", "10", "--part", "gate")
            raise AssertionError("should have refused")
        except SystemExit as exc:
            assert "gate server settings differ" in str(exc), exc


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
