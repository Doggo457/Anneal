"""Check the value-head order of every per-head GatedDeltaNet tensor in the HF checkpoint against the GGUF.

llama.cpp stores the 48 value heads tiled (position r*16+k) while the HF layout groups them by key head
(k*3+r); transformers expands q/k with repeat_interleave, which needs every per-value-head tensor grouped.
For each tensor this prints the cosine similarity with the GGUF under "HF is grouped" and "HF is tiled":
a tensor whose best match is "tiled" disagrees with the rest of the layer.

Usage: python gdn_head_order.py <model.safetensors> <model.gguf> [layer ...]
"""

import math
import sys

import numpy as np
import torch
from gguf import GGUFReader
from gguf.quants import dequantize
from safetensors import safe_open

sys.path.insert(0, __file__.rsplit("\\", 1)[0])
from compare_weights import hf_tensor, stats  # noqa: E402

K, R, D = 16, 3, 128           # key heads, value heads per key head, head dim
KEY = K * D                    # q and k sizes in in_proj_qkv / conv channels
G2T = [(h % K) * R + (h // K) for h in range(K * R)]  # tiled position -> grouped head


def ref(g, name):
    t = g[name]
    return torch.from_numpy(np.ascontiguousarray(dequantize(t.data, t.tensor_type))).float()


def heads(x, order, dim, size):
    """Reorder heads of `size` along `dim`."""
    shape = list(x.shape)
    n = shape[dim] // size
    y = x.reshape(shape[:dim] + [n, size] + shape[dim + 1:])
    y = y.index_select(dim, torch.tensor(order))
    return y.reshape(shape)


def report(label, hf, gg, dim, size, transform=None):
    grouped = heads(hf, G2T, dim, size)                  # HF grouped -> tiled, compare with GGUF
    as_is = hf                                           # HF already tiled
    if transform:
        grouped, as_is = transform(grouped), transform(as_is)
    cg, ct = stats(grouped, gg)[0], stats(as_is, gg)[0]
    verdict = "grouped (consistent)" if cg > ct else "TILED (inconsistent with z/out_proj)"
    print(f"  {label:34s} cos if HF grouped {cg:7.4f} | if HF tiled {ct:7.4f}  -> {verdict}")


def main(st_path, gguf_path, layers):
    f = safe_open(st_path, "pt")
    keys = set(f.keys())
    g = {t.name: t for t in GGUFReader(gguf_path, "r").tensors}
    for n in layers:
        p = f"model.language_model.layers.{n}.linear_attn."
        if p + "A_log" not in keys:
            print(f"layer {n}: full attention, skipped")
            continue
        print(f"layer {n}")
        z, _ = hf_tensor(f, keys, p + "in_proj_z.weight", "cuda")
        report("in_proj_z  vs attn_gate", z, ref(g, f"blk.{n}.attn_gate.weight"), 0, D)
        for hf_n, gg_n in (("in_proj_a.weight", "ssm_alpha.weight"), ("in_proj_b.weight", "ssm_beta.weight")):
            w, _ = hf_tensor(f, keys, p + hf_n, "cuda")
            report(f"{hf_n:10s} vs {gg_n}", w, ref(g, f"blk.{n}.{gg_n}"), 0, 1)
        a = f.get_tensor(p + "A_log").float()
        report("A_log      vs ssm_a (=-exp)", a, ref(g, f"blk.{n}.ssm_a"), 0, 1, transform=lambda x: -torch.exp(x))
        dt = f.get_tensor(p + "dt_bias").float()
        report("dt_bias    vs ssm_dt.bias", dt, ref(g, f"blk.{n}.ssm_dt.bias"), 0, 1)
        qkv, _ = hf_tensor(f, keys, p + "in_proj_qkv.weight", "cuda")
        gq = ref(g, f"blk.{n}.attn_qkv.weight")
        print(f"  {'in_proj_qkv q+k rows':34s} cos {stats(qkv[:2 * KEY], gq[:2 * KEY])[0]:7.4f}")
        report("in_proj_qkv v rows", qkv[2 * KEY:], gq[2 * KEY:], 0, D)
        conv = f.get_tensor(p + "conv1d.weight").float().reshape(-1, f.get_tensor(p + "conv1d.weight").shape[-1])
        gc = ref(g, f"blk.{n}.ssm_conv1d.weight")
        print(f"  {'conv1d q+k channels':34s} cos {stats(conv[:2 * KEY], gc[:2 * KEY])[0]:7.4f}")
        report("conv1d v channels", conv[2 * KEY:], gc[2 * KEY:], 0, D)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], [int(x) for x in sys.argv[3:]] or [0, 1, 2, 4])
