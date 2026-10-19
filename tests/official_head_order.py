"""Compare the DeltaNet value-head order of the official Qwen/Qwen3.8-27B weights with the Unsloth bnb-4bit copy.

Reads layer `n`'s per-value-head tensors from an official shard (bf16) and from the Unsloth checkpoint, and
says whether they are in the same order, or one is the grouped -> tiled reordering of the other (the mapping of
llama.cpp's _reorder_v_heads: grouped [G0_v0..v{r-1}, G1_v0..] -> tiled [G0_v0, G1_v0, .., G0_v1, ..]).

Usage: python official_head_order.py <official shard.safetensors> <unsloth model.safetensors> [layer]
"""

import sys

import torch
from safetensors import safe_open

sys.path.insert(0, __file__.rsplit("\\", 1)[0])
from compare_weights import hf_tensor  # noqa: E402

K, R = 16, 3  # key heads, value heads per key head


def to_tiled(x, head_dim):
    """llama.cpp's _reorder_v_heads on dim 0: grouped (by key head) -> tiled."""
    shape = list(x.shape)
    y = x.reshape([K, R, head_dim] + shape[1:])
    return y.transpose(0, 1).contiguous().reshape(shape)


def to_grouped(x, head_dim):
    shape = list(x.shape)
    y = x.reshape([R, K, head_dim] + shape[1:])
    return y.transpose(0, 1).contiguous().reshape(shape)


def close(a, b, tol):
    return a.shape == b.shape and float((a.float() - b.float()).abs().max()) <= tol


def main(official, unsloth, n):
    fo, fu = safe_open(official, "pt"), safe_open(unsloth, "pt")
    ku = set(fu.keys())
    p = f"model.language_model.layers.{n}.linear_attn."
    rows = [("A_log", 1, 1e-6), ("dt_bias", 1, 1e-6), ("in_proj_a.weight", 1, 1e-2), ("in_proj_b.weight", 1, 1e-2)]
    print(f"layer {n}: official {official.rsplit(chr(92), 1)[-1]} vs Unsloth")
    verdicts = {}
    for name, hd, tol in rows:
        o = fo.get_tensor(p + name).float()
        u = (hf_tensor(fu, ku, p + name, "cuda")[0] if (p + name + ".quant_state.bitsandbytes__nf4") in ku
             else fu.get_tensor(p + name)).float().cpu()
        same = close(o, u, tol)
        u_is_tiled = close(to_tiled(o, hd), u, tol)     # Unsloth = official reordered grouped -> tiled
        u_is_grouped = close(to_grouped(o, hd), u, tol)  # Unsloth = official reordered tiled -> grouped
        verdict = "SAME ORDER" if same else "Unsloth = tiled(official)" if u_is_tiled else \
            "Unsloth = grouped(official)" if u_is_grouped else "no simple relation"
        verdicts[name] = verdict
        print(f"  {name:18s} {verdict:28s} max|o-u| as stored {float((o - u).abs().max()):.4g}  "
              f"official first 6: {[round(float(x), 4) for x in o.flatten()[:6]]}")
        print(f"  {'':18s} {'':28s} Unsloth first 6:  {[round(float(x), 4) for x in u.flatten()[:6]]}")
    print("verdict:", verdicts)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 0)
