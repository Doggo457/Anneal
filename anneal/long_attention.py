"""Long-row attention for Qwen3.8's 16 full-attention layers, opt-in for training (train config `attention`).

transformers runs those layers through its attention interface under the name "sdpa". When PyTorch has no fused
SDPA kernel for the GPU, SDPA takes its math path, which materialises the whole score matrix: 24 heads x L^2 values,
12 GiB per layer at 16,384 tokens in bf16 (more in fp32). On the 32 GB EVO-X2 that spills into shared memory and the
16,384-token dry run crashed in its first forward (21 Oct 2026); 12,288 fitted.

Modes:
  sdpa     transformers' own SDPA call, unchanged.
  chunked  exact causal attention over `chunk` query rows at a time. Each chunk is recomputed in backward
           (torch.utils.checkpoint), so only one chunk's scores exist at once: 24 x chunk x L values (0.75 GiB in bf16
           at chunk 1,024 and L 16,384). Same softmax per row as the full matrix; nothing is approximated.
  fla      FLA's Triton parallel_attn (flash-style, memory linear in L), used only where tests/attn_check.py shows it
           matching the math reference on this GPU. Calls with an explicit mask fall back to chunked.

Rows of at most `min_len` tokens keep transformers' own call, so rows the first run could train behave as before.

apply() registers the function under "sdpa" in transformers' AttentionInterface (class-wide, so every model file that
dispatches through it uses it), check_model() confirms the attention layers dispatch through "sdpa", and report()
counts calls so the trainer can prove the long path ran (verify()).
"""

import torch
import torch.nn.functional as F

STATE = {"mode": "sdpa", "chunk": 1024, "min_len": 0, "calls": 0, "long_calls": 0, "fla_calls": 0,
         "max_len_seen": 0, "registered": False}
_ORIGINAL = {}


def _repeat_kv(x, n):
    if n == 1:
        return x
    b, h, t, d = x.shape
    return x[:, :, None].expand(b, h, n, t, d).reshape(b, h * n, t, d)


def _part(q, k, v, mask, scale):
    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=scale)


def chunked_attention(query, key, value, attention_mask=None, scaling=None, chunk=1024):
    """Causal attention a chunk of query rows at a time. query [B, HQ, T, D]; key, value [B, HKV, S, D] with S >= T
    (the last T key positions line up with the queries); attention_mask None (plain causal) or a bool/float mask
    broadcastable to [B, HQ, T, S] in SDPA's convention. Returns [B, HQ, T, D]."""
    from torch.utils.checkpoint import checkpoint

    n_rep = query.shape[1] // key.shape[1]
    key, value = _repeat_kv(key, n_rep), _repeat_kv(value, n_rep)
    T, S = query.shape[2], key.shape[2]
    off = S - T
    grad = torch.is_grad_enabled() and any(t.requires_grad for t in (query, key, value))
    outs = []
    for s in range(0, T, chunk):
        e = min(T, s + chunk)
        if attention_mask is None:
            m = torch.ones(e - s, off + e, dtype=torch.bool, device=query.device).tril(diagonal=off + s)
        else:
            m = attention_mask[..., s:e, :off + e]
        args = (query[:, :, s:e], key[:, :, :off + e], value[:, :, :off + e], m, scaling)
        outs.append(checkpoint(_part, *args, use_reentrant=False) if grad else _part(*args))
    return torch.cat(outs, dim=2)


def long_sdpa_forward(module, query, key, value, attention_mask, dropout=0.0, scaling=None, is_causal=None,
                      **kwargs):
    """Drop-in for transformers' sdpa_attention_forward: returns (attn_output [B, T, HQ, D], None)."""
    T = query.shape[2]
    counting = not torch.compiler.is_compiling()
    if counting:
        STATE["calls"] += 1
        STATE["max_len_seen"] = max(STATE["max_len_seen"], T)
    if STATE["mode"] == "sdpa" or T <= STATE["min_len"] or T == 1:
        return _ORIGINAL["sdpa"](module, query, key, value, attention_mask, dropout=dropout, scaling=scaling,
                                 is_causal=is_causal, **kwargs)
    if dropout and module.training:
        raise RuntimeError("anneal/long_attention.py does not do attention dropout")
    if counting:
        STATE["long_calls"] += 1
    if STATE["mode"] == "fla" and attention_mask is None and key.shape[2] == T:
        from fla.ops.attn import parallel_attn
        if counting:
            STATE["fla_calls"] += 1
        out = parallel_attn(query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2), scale=scaling)
        return out.contiguous(), None
    out = chunked_attention(query, key, value, attention_mask, scaling, STATE["chunk"])
    return out.transpose(1, 2).contiguous(), None


def apply(mode="sdpa", chunk=1024, min_len=0):
    """mode: sdpa | chunked | fla. Call before the first forward."""
    if mode not in ("sdpa", "chunked", "fla"):
        raise ValueError(f"attention mode {mode!r}: use sdpa, chunked or fla")
    STATE.update(mode=mode, chunk=int(chunk), min_len=int(min_len))
    if mode == "sdpa":
        return "attention: transformers' SDPA call (unchanged)"
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, AttentionInterface
    if "sdpa" not in _ORIGINAL:
        _ORIGINAL["sdpa"] = ALL_ATTENTION_FUNCTIONS["sdpa"]
    AttentionInterface.register("sdpa", long_sdpa_forward)
    ALL_ATTENTION_FUNCTIONS["sdpa"] = long_sdpa_forward
    STATE["registered"] = True
    if mode == "fla":
        from fla.ops.attn import parallel_attn  # noqa: F401  (fail at start, not mid-run)
    return (f"attention: {mode} for rows over {STATE['min_len']} tokens"
            + (f" ({STATE['chunk']} query rows per chunk)" if mode == "chunked" else "")
            + " (anneal/long_attention.py)")


def check_model(model):
    """The full-attention layers and the implementation they dispatch through. Raises if a long mode is on and a
    layer wouldn't reach it."""
    impls, n = {}, 0
    for name, mod in model.named_modules():
        cfg = getattr(mod, "config", None)
        if cfg is None or not hasattr(mod, "num_key_value_groups") or not type(mod).__name__.endswith("Attention"):
            continue
        n += 1
        impl = getattr(cfg, "_attn_implementation", None)
        impls[impl] = impls.get(impl, 0) + 1
    rep = {"attention_layers": n, "implementations": impls}
    if STATE["mode"] != "sdpa" and (n == 0 or set(impls) != {"sdpa"}):
        raise RuntimeError(f"attention mode {STATE['mode']} needs every attention layer on 'sdpa', found {rep}")
    return rep


def verify(longest_row=None):
    """After a forward pass: with a long mode on, the attention layers must have called long_sdpa_forward, and after
    a row longer than min_len (longest_row) the long path must have run."""
    if STATE["mode"] == "sdpa":
        return
    if STATE["calls"] == 0:
        raise RuntimeError(f"attention mode {STATE['mode']} is on but the attention layers never called it: the model "
                           "doesn't dispatch through transformers' attention interface")
    if longest_row is not None and longest_row > STATE["min_len"] and STATE["long_calls"] == 0:
        raise RuntimeError(f"attention mode {STATE['mode']} is on but no attention call took the long path "
                           f"(calls {STATE['calls']}, longest row {longest_row})")


def report():
    return dict(STATE)
