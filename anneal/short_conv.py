"""Shift-and-add version of the GatedDeltaNet short convolution, opt-in for training (--short-conv shift).

Every Qwen3.5/3.8 GatedDeltaNet layer runs a causal depthwise conv1d (kernel 4, no bias, SiLU after it) over
q, k and v together (10240 channels on Qwen3.8-27B). transformers calls causal_conv1d_fn for it, which on
ROCm is its torch fallback: F.conv1d(groups=channels) on a (B, C, T) view, which also forces a transposed
copy of the input on the way in and of the output on the way out. With kernel width 4 the convolution is
just four shifted multiply-adds along time:

    y[b, t, c] = sum_k w[c, k] * x[b, t - (K - 1) + k, c]      (x = 0 before t = 0)

which this module does directly in the (B, T, C) layout the projection already produced, with its own
backward (dx is the same shifted sum with the taps mirrored, dw[c, k] a per-channel dot product over b, t).
Partial sums are kept in fp32 and rounded once, like the reference conv.

apply() swaps the module-level causal_conv1d_fn in transformers' qwen3_5 modules and in Unsloth's compiled
copy (unsloth_compiled_cache), where the GatedDeltaNet forward looks it up at call time. The cached
single-token decode path (causal_conv1d_update) is left alone.
"""

import sys

import torch
import torch.nn.functional as F

STATE = {"mode": "torch", "modules": [], "calls": 0}


CHUNK = 2048  # time steps per fp32 work block: keeps the fp32 temporaries to ~84 MB per tensor at 10240 channels


def _forward(x, w):
    """x (B, T, C), w (C, K) -> y (B, T, C) in x's dtype; fp32 partial sums."""
    T, H = x.shape[1], w.shape[1] - 1
    wf = w.float()
    y = torch.empty_like(x)
    for t0 in range(0, T, CHUNK):
        t1 = min(T, t0 + CHUNK)
        a = max(0, t0 - H)
        xs = x[:, a:t1].float()                      # xs[i] = x[a + i]
        acc = xs[:, t0 - a:] * wf[:, H]              # tap K-1 reads x[t]
        for k in range(H):
            s = H - k                                # tap k reads x[t - s]
            lo = max(t0, s)
            if lo < t1:
                acc[:, lo - t0:].addcmul_(xs[:, lo - s - a:t1 - s - a], wf[:, k])
        y[:, t0:t1] = acc
    return y


def _backward(x, w, dy, need_dx, need_dw):
    T, C, H = x.shape[1], x.shape[2], w.shape[1] - 1
    wf = w.float()
    dx = torch.empty_like(x) if need_dx else None
    dw = torch.zeros(C, H + 1, device=w.device, dtype=torch.float32) if need_dw else None
    for t0 in range(0, T, CHUNK):
        t1 = min(T, t0 + CHUNK)
        e = min(T, t1 + H)
        dys = dy[:, t0:e].float()                    # dys[i] = dy[t0 + i]
        if need_dx:
            # dx[u] = sum_k w[k] * dy[u + s], for u + s < T
            acc = dys[:, :t1 - t0] * wf[:, H]
            for k in range(H):
                s = H - k
                hi = min(t1, T - s)
                if hi > t0:
                    acc[:, :hi - t0].addcmul_(dys[:, s:hi - t0 + s], wf[:, k])
            dx[:, t0:t1] = acc
        if need_dw:
            # dw[k] = sum_t x[t - s] * dy[t]
            a = max(0, t0 - H)
            xs = x[:, a:t1].float()
            for k in range(H + 1):
                s = H - k
                lo = max(t0, s)
                if lo < t1:
                    dw[:, k] += (xs[:, lo - s - a:t1 - s - a] * dys[:, lo - t0:t1 - t0]).sum(dim=(0, 1))
    return dx, (dw.to(w.dtype) if need_dw else None)


class ShiftConv(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w):
        ctx.save_for_backward(x, w)
        return _forward(x, w)

    @staticmethod
    def backward(ctx, dy):
        x, w = ctx.saved_tensors
        return _backward(x, w, dy, ctx.needs_input_grad[0], ctx.needs_input_grad[1])


def shift_conv1d(x_btc, weight, bias=None, activation=None):
    """Causal depthwise conv on x (B, T, C) with weight (C, K); returns (B, T, C)."""
    y = ShiftConv.apply(x_btc, weight)
    if bias is not None:
        y = y + bias
    if activation in ("silu", "swish"):
        y = F.silu(y)
    elif activation is not None:
        from transformers.activations import ACT2FN
        y = ACT2FN[activation](y)
    return y


def causal_conv1d_fn(hidden_states, weight, bias=None, activation=None, **kwargs):
    """Drop-in for transformers' causal_conv1d_fn: hidden_states (B, C, T) -> (B, C, T).

    The GatedDeltaNet forward hands in in_proj_qkv's (B, T, C) output transposed and transposes the result
    straight back, so working on the transposed view costs no copies in either direction.
    """
    if not torch.compiler.is_compiling():
        STATE["calls"] += 1
    x = hidden_states.transpose(1, 2)
    return shift_conv1d(x.to(weight.dtype), weight, bias, activation).to(hidden_states.dtype).transpose(1, 2)


def reference_conv1d(hidden_states, weight, bias=None, activation=None):
    """transformers' torch fallback, unchanged: hidden_states (B, C, T)."""
    D, T = hidden_states.shape[1], hidden_states.shape[2]
    out = F.conv1d(hidden_states.to(weight.dtype), weight.unsqueeze(1), bias, padding=weight.shape[-1] - 1,
                   groups=D)[:, :, :T]
    if activation is not None:
        from transformers.activations import ACT2FN
        out = ACT2FN[activation](out)
    return out.to(hidden_states.dtype)


def apply(mode="torch"):
    """mode: torch (leave transformers' conv alone) | shift. Call after importing unsloth and before the model's
    forward runs; modules imported later (Unsloth's compiled copy is written at load) are patched by
    patch_loaded(), which the loaders below call again after from_pretrained."""
    STATE["mode"] = mode
    if mode == "torch":
        return "GatedDeltaNet short conv: transformers' F.conv1d fallback (unchanged)"
    patch_loaded()
    return "GatedDeltaNet short conv: shift-and-add (anneal/short_conv.py)"


def patch_loaded():
    if STATE["mode"] != "shift":
        return []
    done = []
    for name, mod in list(sys.modules.items()):
        if mod is None or not (name.endswith("modeling_qwen3_5") or name.endswith("compiled_module_qwen3_5")
                               or name.endswith("modeling_qwen3_5_moe") or name.endswith("compiled_module_qwen3_5_moe")):
            continue
        if getattr(mod, "causal_conv1d_fn", None) is not None and mod.causal_conv1d_fn is not causal_conv1d_fn:
            mod.causal_conv1d_fn = causal_conv1d_fn
            done.append(name)
    STATE["modules"] = sorted(set(STATE["modules"]) | set(done))
    return done


def report():
    return dict(STATE)
