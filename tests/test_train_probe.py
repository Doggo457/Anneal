"""Tests for the training speed checks (anneal/train_probe.py, scripts/compare_train_runs.py, train_longrun.py --set).
The attention wrapper tests need torch (any build, CPU is enough) and are skipped without it.

  python tests/test_train_probe.py        (or: python -m pytest tests/test_train_probe.py)
"""
import json
import os
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from anneal import train_probe as T  # noqa: E402
import compare_train_runs as C  # noqa: E402

try:
    import torch  # noqa: F401
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False


def test_categorize_kernel_names():
    cases = {
        "Cijk_Alik_Bljk_BBS_BH_MT128x128x32": "matmul",
        "aten::addmm": "matmul",
        "kDequantizeBlockwise": "bnb_dequant",
        "chunk_gated_delta_rule_fwd_kernel_h": "deltanet_fla",
        "aten::_scaled_dot_product_flash_attention": "sdpa_attention",
        "attn_fwd": "sdpa_attention",
        "aten::conv1d": "short_conv",
        "Memcpy HtoD (Pageable -> Device)": "offload_copy",
        "_unsloth_cross_entropy_forward": "cross_entropy",
        "aten::mul": "other",
    }
    for name, want in cases.items():
        assert T.categorize(name) == want, (name, T.categorize(name), want)


def test_summarize_events_shares():
    s = T.summarize_events([("Cijk_gemm", 600.0, 1.0, 3), ("chunk_gated_delta_x", 300.0, 1.0, 1),
                            ("aten::mul", 100.0, 1.0, 9)])
    assert s["basis"] == "self_device_time"
    assert s["share"] == {"matmul": 0.6, "deltanet_fla": 0.3, "other": 0.1}
    cpu = T.summarize_events([("aten::addmm", 0.0, 30.0, 1), ("aten::mul", 0.0, 10.0, 1)])
    assert cpu["basis"].startswith("self_cpu_time") and cpu["share"]["matmul"] == 0.75


def test_install_sdpa_off_wraps_nothing():
    assert T.install_sdpa(False, False) is None


def _sdpa_case():
    import torch
    g = torch.Generator().manual_seed(0)
    b, h, s, d = 1, 4, 16, 8
    q = torch.randn(b, s, h, d, generator=g).transpose(1, 2)   # a transposed view, as the model hands it over
    k = torch.randn(b, s, h, d, generator=g).transpose(1, 2)
    v = torch.randn(b, s, h, d, generator=g).transpose(1, 2)
    return q, k, v


def test_sdpa_probe_and_contiguous_same_values():
    if not HAVE_TORCH:
        print("  skipped: no torch")
        return
    import torch
    import torch.nn.functional as F
    original = F.scaled_dot_product_attention
    q, k, v = _sdpa_case()
    want = original(q, k, v, is_causal=True)
    logs = []
    try:
        rep = T.install_sdpa(contiguous=True, probe=True, log=logs.append)
        got = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
        wrapped_doc = F.scaled_dot_product_attention.__doc__
        wrapped_name = F.scaled_dot_product_attention.__name__
        again = F.scaled_dot_product_attention(q.contiguous(), k.contiguous(), v.contiguous(), is_causal=True)
    finally:
        F.scaled_dot_product_attention = original
        torch.nn.functional.scaled_dot_product_attention = original
    assert torch.allclose(got, want, atol=1e-6, rtol=1e-5)
    assert wrapped_doc == original.__doc__ and wrapped_name == original.__name__
    d = rep.as_dict()
    assert d["calls"] == 2 and d["calls_with_a_copy"] == 1
    assert d["first_call"]["query"]["contiguous"] is False
    assert d["first_call"]["query"]["shape"] == [1, 4, 16, 8]
    assert d["first_call"]["kwargs"] == {"is_causal": True}
    assert again.shape == got.shape
    assert any("first call" in x for x in logs)


def test_sdpa_wrapper_keeps_doc_and_name():
    """Unsloth checks "enable_gqa" in F.scaled_dot_product_attention.__doc__ at import, so the wrapper must carry the
    original's docstring and name. Runs on a stand-in torch module, so it needs no torch install."""
    import types
    keys = ("torch", "torch.nn", "torch.nn.functional")
    saved = {k: sys.modules.get(k) for k in keys}

    def sdpa(query, key, value, attn_mask=None, enable_gqa=False):
        """scaled_dot_product_attention(query, key, value, ..., enable_gqa=False) -> Tensor"""
        return query

    torch = types.ModuleType("torch")
    torch.nn = types.ModuleType("torch.nn")
    torch.nn.functional = types.ModuleType("torch.nn.functional")
    torch.nn.functional.scaled_dot_product_attention = sdpa
    torch.is_grad_enabled = lambda: False
    try:
        sys.modules.update({"torch": torch, "torch.nn": torch.nn, "torch.nn.functional": torch.nn.functional})
        T.install_sdpa(contiguous=False, probe=True, log=lambda *_: None)
        w = torch.nn.functional.scaled_dot_product_attention
        assert w is not sdpa and w._anneal_wrapped and w.__wrapped__ is sdpa
        assert w.__doc__ == sdpa.__doc__ and "enable_gqa" in w.__doc__
        assert w.__name__ == "sdpa" and w.__module__ == sdpa.__module__
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v


class _FakeTensor:
    """Just enough of a tensor for the probe: shape, stride, contiguity, dtype, device."""

    def __init__(self, shape, contiguous=True):
        self.shape, self._contig = tuple(shape), contiguous
        self.dtype, self.device = "torch.bfloat16", "cuda:0"

    def stride(self):
        out, acc = [], 1
        for n in reversed(self.shape):
            out.append(acc)
            acc *= n
        st = list(reversed(out))
        if not self._contig:  # a (B, L, H, D) buffer seen as (B, H, L, D)
            st[-3], st[-2] = st[-2], st[-3]
        return tuple(st)

    def is_contiguous(self):
        return self._contig

    def contiguous(self):
        return _FakeTensor(self.shape, True)

    def dim(self):
        return len(self.shape)


def _with_fake_torch(fn):
    import types
    keys = ("torch", "torch.nn", "torch.nn.functional")
    saved = {k: sys.modules.get(k) for k in keys}
    seen = []

    def sdpa(query, key, value, attn_mask=None, is_causal=False, enable_gqa=False):
        """scaled_dot_product_attention(query, key, value, ..., enable_gqa=False) -> Tensor"""
        seen.append((query.is_contiguous(), key.is_contiguous(), value.is_contiguous()))
        return query

    torch = types.ModuleType("torch")
    torch.nn = types.ModuleType("torch.nn")
    torch.nn.functional = types.ModuleType("torch.nn.functional")
    torch.nn.functional.scaled_dot_product_attention = sdpa
    torch.is_grad_enabled = lambda: True
    try:
        sys.modules.update({"torch": torch, "torch.nn": torch.nn, "torch.nn.functional": torch.nn.functional})
        return fn(torch, sdpa, seen)
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v


def test_sdpa_probe_records_the_model_calls_not_the_import_self_test():
    """Unsloth's import runs a 2-head, 16-token attention call first; the model's own calls (thousands of tokens,
    possibly strided views) must be what decides whether the contiguous-copy test runs."""
    def body(torch, sdpa, seen):
        logs = []
        rep = T.install_sdpa(contiguous=False, probe=True, log=logs.append)
        w = torch.nn.functional.scaled_dot_product_attention
        w(_FakeTensor([1, 2, 16, 64]), _FakeTensor([1, 2, 16, 64]), _FakeTensor([1, 2, 16, 64]), is_causal=True)
        assert rep.long_strided() is None and rep.as_dict()["first_long_call"] is None
        for _ in range(3):
            w(_FakeTensor([1, 24, 4096, 256], False), _FakeTensor([1, 4, 4096, 256], False),
              _FakeTensor([1, 4, 4096, 256], False), is_causal=True, enable_gqa=True)
        w(_FakeTensor([1, 24, 300, 256]), _FakeTensor([1, 4, 300, 256]), _FakeTensor([1, 4, 300, 256]), is_causal=True,
          enable_gqa=True)
        d = rep.as_dict()
        assert d["calls"] == 5 and d["first_call"]["query"]["shape"] == [1, 2, 16, 64]
        assert d["first_long_call"]["query"]["shape"] == [1, 24, 4096, 256] and d["long_strided"] is True
        assert d["first_long_call"]["kwargs"] == {"is_causal": True, "enable_gqa": True}
        assert d["long_layouts"] == {
            "q:strided k:strided v:strided heads 24/4 dim 256 bfloat16 enable_gqa,is_causal": 3,
            "q:contiguous k:contiguous v:contiguous heads 24/4 dim 256 bfloat16 enable_gqa,is_causal": 1}, d
        assert sum(1 for x in logs if "first call with" in x) == 1
        assert seen[1] == (False, False, False)  # probe only: the inputs reach the original unchanged
        json.dumps(d)

    _with_fake_torch(body)

    def copy(torch, sdpa, seen):
        rep = T.install_sdpa(contiguous=True, probe=False, log=lambda *_: None)
        w = torch.nn.functional.scaled_dot_product_attention
        w(_FakeTensor([1, 24, 4096, 256], False), _FakeTensor([1, 4, 4096, 256]), _FakeTensor([1, 4, 4096, 256]))
        assert seen == [(True, True, True)] and rep.copied_calls == 1 and rep.first_long is None

    _with_fake_torch(copy)


def test_sdpa_double_wrap_refused():
    if not HAVE_TORCH:
        print("  skipped: no torch")
        return
    import torch.nn.functional as F
    original = F.scaled_dot_product_attention
    try:
        T.install_sdpa(probe=True, log=lambda *_: None)
        try:
            T.install_sdpa(probe=True, log=lambda *_: None)
            raise AssertionError("second wrap allowed")
        except RuntimeError:
            pass
    finally:
        F.scaled_dot_product_attention = original


def test_prefer_blas_sets_the_switch_before_torch():
    """On Windows builds torch.backends.cuda.preferred_blas_library(x) changes nothing, so the choice goes through
    TORCH_BLAS_PREFER_HIPBLASLT before torch is imported, and is refused once torch is loaded."""
    env = {}
    assert T.prefer_blas(None, env, {}) is None and env == {}
    assert T.prefer_blas("hipBLASLt", env, {}) == "Cublaslt" and env == {"TORCH_BLAS_PREFER_HIPBLASLT": "1"}
    assert T.prefer_blas("hipblas", env, {}) == "Cublas" and env == {"TORCH_BLAS_PREFER_HIPBLASLT": "0"}
    for bad, modules in (("tensile", {}), ("hipblaslt", {"torch": object()})):
        try:
            T.prefer_blas(bad, {}, modules)
            raise AssertionError(f"{bad} accepted")
        except (ValueError, RuntimeError):
            pass


def test_blas_report_stops_when_the_library_was_not_switched():
    """A void A/B (asked for hipBLASLt, ran on hipBLAS) must stop the run, not report a speed."""
    import types

    def body(torch, sdpa, seen):
        state = {"lib": "_BlasBackend.Cublas"}
        torch.version = types.SimpleNamespace(hip="7.14")
        torch.backends = types.SimpleNamespace(cuda=types.SimpleNamespace(preferred_blas_library=lambda: state["lib"]))
        out = T.blas_report(None, None, log=lambda *_: None)
        assert out["in_use"] == "_BlasBackend.Cublas" and out["requested"] is None
        try:
            T.blas_report("hipblaslt", "Cublaslt", log=lambda *_: None)
            raise AssertionError("library mismatch not caught")
        except RuntimeError:
            pass
        state["lib"] = "_BlasBackend.Cublaslt"
        assert T.blas_report("hipblaslt", "Cublaslt", log=lambda *_: None)["in_use"].endswith("Cublaslt")
        try:
            T.blas_report("hipblas", "Cublas", log=lambda *_: None)
            raise AssertionError("Cublaslt accepted as Cublas")
        except RuntimeError:
            pass

    _with_fake_torch(body)


def _write_run(folder, secs, losses, tokens=24000, reserved=27.5):
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, "metrics.jsonl"), "w", encoding="utf-8") as f:
        f.write(json.dumps({"event": "train_begin", "max_steps": len(secs)}) + "\n")
        for i, (s, l) in enumerate(zip(secs, losses), 1):
            f.write(json.dumps({"step": i, "seconds": s, "tokens": tokens, "loss": l, "grad_norm": 0.5,
                                "peak_reserved_gib": reserved}) + "\n")
            f.write(json.dumps({"event": "mem", "step": i, "hip_free_gib": 3.0}) + "\n")
    with open(os.path.join(folder, "summary.json"), "w", encoding="utf-8") as f:
        json.dump({"peak_reserved_gib": reserved}, f)


def test_compare_runs_rule():
    with tempfile.TemporaryDirectory() as tmp:
        base, rep, fast, drift, big = (os.path.join(tmp, n) for n in ("a", "a2", "b", "c", "d"))
        _write_run(base, [400, 260, 262, 258, 261, 259], [1.0, 0.9, 0.8, 0.85, 0.7, 0.75])
        _write_run(rep, [395, 261, 261, 259, 260, 260], [1.0, 0.9, 0.8, 0.85, 0.7, 0.75])
        _write_run(fast, [390, 240, 241, 239, 240, 240], [1.002, 0.901, 0.801, 0.851, 0.7005, 0.7504])
        _write_run(drift, [390, 230, 230, 230, 230, 230], [1.0, 0.9, 0.85, 0.85, 0.7, 0.75])
        _write_run(big, [390, 230, 230, 230, 230, 230], [1.0, 0.9, 0.8, 0.85, 0.7, 0.75], reserved=30.2)
        b = C.read_run(base)
        assert b["timed_steps"] == [2, 3, 4, 5, 6] and b["seconds_per_step"] == 260.0
        assert b["tokens_per_second"] == round(24000 * 5 / 1300, 2)
        res = {os.path.basename(r): C.compare(b, C.read_run(r)) for r in (rep, fast, drift, big)}
        assert not res["a2"]["passes"] and res["a2"]["checks"]["loss_within_tol"]   # noise floor: no gain
        assert res["b"]["passes"] and res["b"]["speed_ratio"] > 1.05
        assert not res["c"]["passes"] and not res["c"]["checks"]["loss_within_tol"]  # 0.85 vs 0.8 = 6% drift
        assert not res["d"]["passes"] and not res["d"]["checks"]["memory_ok"]
        out = os.path.join(tmp, "cmp.json")
        assert C.main([base, rep, fast, "--json", out]) == 0
        with open(out, encoding="utf-8") as f:
            assert [c["passes"] for c in json.load(f)["comparisons"]] == [False, True]


def test_train_longrun_set_override_parses():
    # --set must reach the config as YAML values; checked by the source rather than by a GPU run.
    src = open(os.path.join(ROOT, "scripts", "train_longrun.py"), encoding="utf-8").read()
    assert 'p.add_argument("--set", action="append"' in src
    assert "cfg[key.strip()] = yaml.safe_load(value)" in src
    assert src.index("train_probe.install_sdpa(") < src.index("import unsloth  # noqa")
    r = subprocess.run([sys.executable, "-c", "import yaml; print(repr(yaml.safe_load('true')), "
                        "repr(yaml.safe_load('hipblaslt')), repr(yaml.safe_load('2')))"],
                       capture_output=True, text=True)
    assert r.stdout.strip() == "True 'hipblaslt' 2", r.stdout


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
