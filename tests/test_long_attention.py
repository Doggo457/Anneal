"""anneal/long_attention.py: the chunked path gives exactly causal attention and its gradients, short rows keep
transformers' own call, and the registration reaches the attention interface."""
import os
import sys

import pytest

torch = pytest.importorskip("torch")
F = torch.nn.functional
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from anneal import long_attention as LA  # noqa: E402


def _inputs(T=301, S=None, hq=6, hkv=2, d=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    S = S or T
    q = torch.randn(1, hq, T, d, generator=g, dtype=torch.float64, requires_grad=True)
    k = torch.randn(1, hkv, S, d, generator=g, dtype=torch.float64, requires_grad=True)
    v = torch.randn(1, hkv, S, d, generator=g, dtype=torch.float64, requires_grad=True)
    do = torch.randn(1, hq, T, d, generator=g, dtype=torch.float64)
    return q, k, v, do


def _reference(q, k, v, mask=None, scale=None):
    n = q.shape[1] // k.shape[1]
    kk, vv = k.repeat_interleave(n, dim=1), v.repeat_interleave(n, dim=1)
    if mask is None:
        return F.scaled_dot_product_attention(q, kk, vv, is_causal=True, scale=scale)
    return F.scaled_dot_product_attention(q, kk, vv, attn_mask=mask, scale=scale)


def _grads(fn, q, k, v, do):
    for t in (q, k, v):
        t.grad = None
    out = fn()
    out.backward(do)
    return out.detach(), [t.grad.clone() for t in (q, k, v)]


@pytest.mark.parametrize("chunk", [1, 64, 100, 301, 1024])
def test_chunked_matches_causal_attention_and_grads(chunk):
    q, k, v, do = _inputs()
    scale = 0.3
    o1, g1 = _grads(lambda: LA.chunked_attention(q, k, v, None, scale, chunk), q, k, v, do)
    o2, g2 = _grads(lambda: _reference(q, k, v, scale=scale), q, k, v, do)
    assert torch.allclose(o1, o2, atol=1e-12)
    for a, b in zip(g1, g2):
        assert torch.allclose(a, b, atol=1e-10)


def test_chunked_with_explicit_mask_and_cached_keys():
    # 40 cached key positions before 100 queries, plus padding on the first 7 keys
    q, k, v, do = _inputs(T=100, S=140)
    S, T = 140, 100
    allowed = torch.ones(T, S, dtype=torch.bool).tril(diagonal=S - T)
    allowed[:, :7] = False
    mask = allowed[None, None]
    o1, g1 = _grads(lambda: LA.chunked_attention(q, k, v, mask, None, 32), q, k, v, do)
    o2, g2 = _grads(lambda: _reference(q, k, v, mask=mask), q, k, v, do)
    assert torch.allclose(o1, o2, atol=1e-12)
    for a, b in zip(g1, g2):
        assert torch.allclose(a, b, atol=1e-10)


def test_no_grad_path_skips_checkpoint():
    q, k, v, _ = _inputs(T=50)
    with torch.no_grad():
        out = LA.chunked_attention(q, k, v, None, None, 16)
    assert torch.allclose(out, _reference(q, k, v).detach(), atol=1e-12)


def test_registered_function_routes_by_length():
    pytest.importorskip("transformers")
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    class M(torch.nn.Module):
        num_key_value_groups = 3
        is_causal = True

    try:
        msg = LA.apply("chunked", chunk=16, min_len=64)
        assert "chunked" in msg
        fn = ALL_ATTENTION_FUNCTIONS.get_interface("sdpa", None)
        assert fn is LA.long_sdpa_forward
        mod = M().train()
        before = dict(LA.STATE)
        q, k, v, _ = _inputs(T=40)                         # short row: transformers' own call
        out, _ = fn(mod, q, k, v, None, scaling=0.25)
        assert LA.STATE["long_calls"] == before["long_calls"]
        assert torch.allclose(out, _reference(q, k, v, scale=0.25).transpose(1, 2), atol=1e-12)
        q, k, v, _ = _inputs(T=130)                        # long row: chunked
        out, _ = fn(mod, q, k, v, None, scaling=0.25)
        assert LA.STATE["long_calls"] == before["long_calls"] + 1
        assert out.shape == (1, 130, 6, 16)
        assert torch.allclose(out, _reference(q, k, v, scale=0.25).transpose(1, 2), atol=1e-12)
        LA.verify(130)
        with pytest.raises(RuntimeError):
            fn(mod, q, k, v, None, dropout=0.1, scaling=0.25)
    finally:
        from transformers.modeling_utils import AttentionInterface
        AttentionInterface.register("sdpa", LA._ORIGINAL["sdpa"])
        ALL_ATTENTION_FUNCTIONS["sdpa"] = LA._ORIGINAL["sdpa"]
        LA.STATE.update(mode="sdpa", long_calls=0, calls=0)


def test_verify_raises_when_long_path_never_ran():
    LA.STATE.update(mode="chunked", min_len=100, calls=0, long_calls=0)
    try:
        with pytest.raises(RuntimeError):
            LA.verify()  # attention never dispatched through the interface
        LA.STATE.update(calls=16)
        LA.verify()
        with pytest.raises(RuntimeError):
            LA.verify(500)
        LA.verify(80)  # no row long enough: nothing to prove
        LA.STATE.update(mode="sdpa", calls=0)
        LA.verify(500)  # unchanged mode: nothing to prove
    finally:
        LA.STATE.update(mode="sdpa", calls=0, long_calls=0)


def test_check_model_needs_sdpa_layers():
    class Cfg:
        _attn_implementation = "eager"

    class FakeAttention(torch.nn.Module):
        num_key_value_groups = 6

        def __init__(self):
            super().__init__()
            self.config = Cfg()

    model = torch.nn.Sequential(FakeAttention())
    LA.STATE.update(mode="chunked")
    try:
        with pytest.raises(RuntimeError):
            LA.check_model(model)
        Cfg._attn_implementation = "sdpa"
        assert LA.check_model(model) == {"attention_layers": 1, "implementations": {"sdpa": 1}}
    finally:
        LA.STATE.update(mode="sdpa")


def test_bad_mode():
    with pytest.raises(ValueError):
        LA.apply("flash9")
