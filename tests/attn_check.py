"""Which softmax-attention kernels work on this GPU, and how much memory each needs at long rows.

Qwen3.8-27B has 16 full-attention layers (24 query heads, 4 key/value heads, head dim 256). If PyTorch's fused SDPA
kernels are missing for gfx1151, attention falls back to the math path, which materialises the full score matrix:
24 x L^2 x 2 bytes = 12 GiB at 16,384 tokens for one layer, which this 32 GB GPU can't hold on top of the model.
This check runs one attention layer forward + backward at Qwen3.8's shapes for each candidate and reports success,
peak GPU memory above the inputs, the lowest free system RAM (the GPU's shared memory comes out of RAM on this
machine) and time; candidates whose answer differs from the math reference at 4,096 tokens are flagged.

Candidates: SDPA as PyTorch picks (default; with enable_gqa it is what transformers' training path calls, with no mask
and head dim 256) and restricted to each backend (flash, efficient, math; math only up to 8,192), with GQA by
enable_gqa and by repeated key/value heads (what transformers does when it can't use enable_gqa); FLA's Triton parallel_attn (bundled
with Unsloth); and a chunked SDPA (1,024 query rows at a time, each chunk recomputed in backward). Every case runs in
its own child process, so a GPU fault ends only that case. The SDPA cases run twice: as installed, and with
TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1.

Usage: python tests/attn_check.py [--lengths 4096,8192,12288,16384] [--out results/attn-check]
"""
import argparse
import json
import os
import subprocess
import sys
import threading
import time

HQ, HKV, D = 24, 4, 256
CHUNK = 1024


def ram_floor(stop, out):
    import psutil
    low = psutil.virtual_memory().available
    while not stop.is_set():
        low = min(low, psutil.virtual_memory().available)
        time.sleep(0.05)
    out["ram_free_min_gib"] = round(low / 2**30, 2)


def chunked_sdpa(q, k, v, scale):
    """Causal attention, CHUNK query rows at a time; each chunk is recomputed in backward, so only one chunk's scores
    exist at once. q: [B, HQ, L, D], k/v: [B, HQ, L, D] (heads already repeated)."""
    import torch
    import torch.nn.functional as F
    from torch.utils.checkpoint import checkpoint

    L = q.shape[2]
    outs = []
    for s in range(0, L, CHUNK):
        e = min(L, s + CHUNK)
        mask = torch.ones(e - s, e, dtype=torch.bool, device=q.device).tril(diagonal=s)

        def part(qc, kc, vc, m=mask):
            return F.scaled_dot_product_attention(qc, kc, vc, attn_mask=m, scale=scale)

        outs.append(checkpoint(part, q[:, :, s:e], k[:, :, :e], v[:, :, :e], use_reentrant=False))
    return torch.cat(outs, dim=2)


def find_parallel_attn():
    """FLA's parallel_attn: from the fla package, or from wherever the Gated DeltaNet kernels the model uses come from
    (a vendored copy has the same ops/ layout)."""
    import importlib
    try:
        return importlib.import_module("fla.ops.attn").parallel_attn
    except ImportError:
        pass
    import unsloth  # noqa: F401  (loads the patched model code)
    from transformers.models.qwen3_5 import modeling_qwen3_5 as m
    src = getattr(getattr(m, "chunk_gated_delta_rule", None), "__module__", None) or ""
    if ".ops." not in src:
        raise ImportError(f"no fla package, and chunk_gated_delta_rule comes from {src or 'nowhere'}")
    return importlib.import_module(src.split(".ops.")[0] + ".ops.attn").parallel_attn


def child(case, L):
    import torch
    import torch.nn.functional as F

    dev = "cuda"
    torch.manual_seed(0)
    scale = D ** -0.5
    q = torch.randn(1, HQ, L, D, device=dev, dtype=torch.bfloat16)
    k = torch.randn(1, HKV, L, D, device=dev, dtype=torch.bfloat16)
    v = torch.randn(1, HKV, L, D, device=dev, dtype=torch.bfloat16)
    do = torch.randn(1, HQ, L, D, device=dev, dtype=torch.bfloat16)

    def run(kind, q, k, v):
        n = q.shape[2]
        q, k, v = (t.detach().clone().requires_grad_(True) for t in (q, k, v))
        if kind.startswith("sdpa-"):
            import contextlib

            from torch.nn.attention import SDPBackend, sdpa_kernel
            backend = {"flash": SDPBackend.FLASH_ATTENTION, "efficient": SDPBackend.EFFICIENT_ATTENTION,
                       "math": SDPBackend.MATH, "default": None}[kind.split("-")[1]]
            with sdpa_kernel([backend]) if backend is not None else contextlib.nullcontext():
                if kind.endswith("-gqa"):
                    o = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=scale, enable_gqa=True)
                else:
                    kk, vv = (t.repeat_interleave(HQ // HKV, dim=1) for t in (k, v))
                    o = F.scaled_dot_product_attention(q, kk, vv, is_causal=True, scale=scale)
        elif kind == "fla-parallel":
            parallel_attn = find_parallel_attn()
            o = parallel_attn(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), scale=scale).transpose(1, 2)
        elif kind == "chunked":
            kk, vv = (t.repeat_interleave(HQ // HKV, dim=1) for t in (k, v))
            o = chunked_sdpa(q, kk, vv, scale)
        else:
            raise ValueError(kind)
        o.backward(do[:, :, :n])
        return [t.float() for t in (o.detach(), q.grad, k.grad, v.grad)]

    res = {"case": case, "L": L, "torch": torch.__version__, "hip": torch.version.hip,
           "aotriton_experimental": os.environ.get("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL")}
    stop, ram = threading.Event(), {}
    th = threading.Thread(target=ram_floor, args=(stop, ram), daemon=True)
    try:
        run(case, q[:, :, :256], k[:, :, :256], v[:, :, :256])  # warm-up: kernel compile/tuning outside the timing
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        th.start()
        t0 = time.time()
        got = run(case, q, k, v)
        torch.cuda.synchronize()
        res["seconds"] = round(time.time() - t0, 3)
        res["peak_extra_gib"] = round((torch.cuda.max_memory_allocated() - base) / 2**30, 3)
        res["ok"] = all(bool(torch.isfinite(t).all()) for t in got)
        if L <= 4096:
            ref = run("sdpa-math", q, k, v)
            res["max_rel_err"] = [round(((a - b).abs().max() / b.abs().max().clamp_min(1e-6)).item(), 5)
                                  for a, b in zip(got, ref)]
            res["matches_math"] = max(res["max_rel_err"]) < 0.02
    except Exception as exc:  # a kernel that isn't built for this GPU raises here
        res["ok"] = False
        res["error"] = f"{type(exc).__name__}: {exc}"[:400]
    finally:
        stop.set()
        if th.is_alive():
            th.join()
        res.update(ram)
    print("RESULT " + json.dumps(res), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--lengths", default="4096,8192,12288,16384")
    ap.add_argument("--out", default=os.path.join("results", "attn-check"))
    ap.add_argument("--child", nargs=2, metavar=("CASE", "L"))
    args = ap.parse_args()
    if args.child:
        child(args.child[0], int(args.child[1]))
        return 0
    os.makedirs(args.out, exist_ok=True)
    lengths = [int(x) for x in args.lengths.split(",")]
    sdpa = ["sdpa-default-gqa", "sdpa-default", "sdpa-efficient-gqa", "sdpa-efficient", "sdpa-flash-gqa",
            "sdpa-flash", "sdpa-math"]
    plan = [(c, env) for env in ("", "1") for c in sdpa] + [("fla-parallel", ""), ("chunked", "")]
    path = os.path.join(args.out, "results.jsonl")
    failed, last = set(), {}  # last: (case, env) -> (L, peak_extra_gib) of its previous length
    with open(path, "a", encoding="utf-8") as fh:
        for L in lengths:
            for case, env in plan:
                if case == "sdpa-math" and L > 8192:
                    continue  # needs the full score matrix: the failure this check is about
                if (case, env) in failed:
                    continue  # failed at a shorter length already
                prev = last.get((case, env))
                if prev and prev[1] * (L / prev[0]) ** 2 > 14:
                    print(json.dumps({"case": case, "aotriton_experimental": env or None, "L": L,
                                      "skipped": f"its memory grew like L^2 up to {prev[0]}: about "
                                                 f"{prev[1] * (L / prev[0]) ** 2:.1f} GiB at {L}"}), flush=True)
                    continue
                e = dict(os.environ)
                e.pop("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", None)
                if env:
                    e["TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"] = env
                t0 = time.time()
                p = subprocess.run([sys.executable, "-u", os.path.abspath(__file__), "--child", case, str(L)],
                                   env=e, capture_output=True, text=True, timeout=1800)
                line = next((x for x in p.stdout.splitlines() if x.startswith("RESULT ")), None)
                res = json.loads(line[7:]) if line else {"case": case, "L": L, "ok": False,
                                                         "aotriton_experimental": env or None,
                                                         "error": f"child exit {p.returncode}: "
                                                                  + (p.stderr or p.stdout)[-400:]}
                res["wall_seconds"] = round(time.time() - t0, 1)
                if not res.get("ok"):
                    failed.add((case, env))
                elif res.get("peak_extra_gib") is not None:
                    last[(case, env)] = (L, res["peak_extra_gib"])
                fh.write(json.dumps(res) + "\n")
                fh.flush()
                print(json.dumps({k: res.get(k) for k in ("case", "aotriton_experimental", "L", "ok", "seconds",
                                                          "peak_extra_gib", "ram_free_min_gib", "matches_math",
                                                          "error")}), flush=True)
    print(f"written {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
