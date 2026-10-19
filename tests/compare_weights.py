"""Compare the bnb-4bit Hugging Face checkpoint with Ollama's GGUF of the same model, tensor by tensor.

Both are 4-bit copies of the same original weights, so matching tensors should agree closely (cosine
similarity ~0.99, relative error ~10%). It also dequantizes each NF4 tensor with bitsandbytes on the GPU and
on the CPU, which should agree exactly. Everything is read-only.

Usage: python compare_weights.py <model.safetensors> <model.gguf> [layer ...]
"""

import json
import sys

import numpy as np
import torch
from gguf import GGUFReader
from gguf.quants import dequantize
from safetensors import safe_open

import bitsandbytes as bnb
from bitsandbytes.functional import QuantState

# HF name (within model.language_model.layers.N) -> GGUF name (within blk.N); FFN and projections that the
# GGUF converter stores unpermuted.
PAIRS = [("mlp.gate_proj.weight", "ffn_gate.weight"), ("mlp.up_proj.weight", "ffn_up.weight"),
         ("mlp.down_proj.weight", "ffn_down.weight"), ("self_attn.o_proj.weight", "attn_output.weight"),
         ("self_attn.v_proj.weight", "attn_v.weight"), ("linear_attn.out_proj.weight", "ssm_out.weight"),
         ("linear_attn.in_proj_z.weight", "attn_gate.weight")]
GLOBAL = [("model.language_model.embed_tokens.weight", "token_embd.weight"), ("lm_head.weight", "output.weight")]
SUFFIXES = ("absmax", "quant_map", "nested_absmax", "nested_quant_map", "quant_state.bitsandbytes__nf4")


def hf_tensor(f, keys, name, device):
    w = f.get_tensor(name)
    if name + ".quant_state.bitsandbytes__nf4" not in keys:
        return w.float(), "bf16"
    qs = {s: f.get_tensor(f"{name}.{s}") for s in SUFFIXES if f"{name}.{s}" in keys}
    state = QuantState.from_dict(qs, device=torch.device(device))
    return bnb.functional.dequantize_4bit(w.to(device), state).float().cpu(), "nf4"


def stats(a, b):
    a, b = a.double().flatten(), b.double().flatten()
    cos = float(torch.dot(a, b) / (a.norm() * b.norm()))
    rel = float((a - b).norm() / b.norm())
    return cos, rel


def main(st_path, gguf_path, layers):
    f = safe_open(st_path, "pt")
    keys = set(f.keys())
    g = GGUFReader(gguf_path, "r")
    gt = {t.name: t for t in g.tensors}
    rows = []

    def compare(hf_name, gg_name):
        if hf_name not in keys or gg_name not in gt:
            return
        t = gt[gg_name]
        ref = torch.from_numpy(np.ascontiguousarray(dequantize(t.data, t.tensor_type))).float()
        gpu, kind = hf_tensor(f, keys, hf_name, "cuda")
        row = {"hf": hf_name.replace("model.language_model.", ""), "gguf": f"{gg_name} {t.tensor_type.name}",
               "kind": kind, "shape": list(gpu.shape)}
        if list(ref.shape) != list(gpu.shape):
            row["shape_mismatch"] = list(ref.shape)
            rows.append(row)
            print(json.dumps(row))
            return
        row["cos_vs_gguf"], row["rel_vs_gguf"] = [round(x, 4) for x in stats(gpu, ref)]
        if kind == "nf4":
            try:
                cpu, _ = hf_tensor(f, keys, hf_name, "cpu")
                row["cos_gpu_vs_cpu"], row["rel_gpu_vs_cpu"] = [round(x, 6) for x in stats(gpu, cpu)]
            except Exception as exc:
                row["cpu_dequant_error"] = f"{type(exc).__name__}: {exc}"[:160]
        rows.append(row)
        print(json.dumps(row))

    for hf_name, gg_name in GLOBAL:
        compare(hf_name, gg_name)
    for n in layers:
        for hf_suffix, gg_suffix in PAIRS:
            compare(f"model.language_model.layers.{n}.{hf_suffix}", f"blk.{n}.{gg_suffix}")
    bad = [r for r in rows if r.get("cos_vs_gguf", 1) < 0.95 or "shape_mismatch" in r]
    print(f"== {len(rows)} tensors compared, {len(bad)} disagree with the GGUF (cos < 0.95 or shape)")
    return rows


if __name__ == "__main__":
    layers = [int(x) for x in sys.argv[3:]] or [0, 1, 2, 3, 31, 63]
    main(sys.argv[1], sys.argv[2], layers)
