"""Check the shift-and-add GatedDeltaNet short conv (anneal/short_conv.py) against PyTorch's F.conv1d: forward and backward numerics, speed and memory on this GPU.

Shapes follow Qwen3.8-27B: 10240 channels (q, k, v), kernel width 4, no bias, SiLU, bf16, and the input is
handed over the way the model does it: in_proj_qkv's (B, T, C) output viewed as (B, C, T). Both versions
are compared with a float64 CPU reference of the same function; the result is a match when the shift
version's error is no larger than F.conv1d's own bf16 error (and ~1e-6 in fp32).

Timing covers the convolution plus getting the result back to a contiguous (B, T, C), which is what the
GatedDeltaNet forward does next (transpose, split, reshape). "torch_contiguous" runs the same F.conv1d on a
contiguous (B, C, T) copy of the input, to separate the cost of the layout from the convolution itself. The per-step estimate multiplies by 48
linear-attention layers and (forward + recomputed forward + backward), as Unsloth's gradient checkpointing
runs it.

Usage (the Training page adds --run-dir): short_conv_check.py [--lengths 4096,8192]
"""

import argparse
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from anneal.runtime import RESULTS_DIR, Run  # noqa: E402

C, K, LAYERS = 10240, 4, 48


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--lengths", default="4096,8192")
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--tag", default="short_conv_check")
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--resume", action="store_true", help="accepted for the Training page; ignored")
    return ap.parse_args()


def make(T, dtype, device, seed=0, channels=C):
    import torch
    g = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(1, T, channels, generator=g)          # in_proj_qkv output layout (B, T, C)
    w = torch.randn(channels, K, generator=g) * 0.3
    dy = torch.randn(1, T, channels, generator=g) * 0.01   # upstream grad in the (B, T, C) layout
    return x.to(device, dtype), w.to(device, dtype), dy.to(device, dtype)


def run_fn(fn, x_btc, w, dy_btc):
    """Forward + backward through fn on the (B, C, T) view; returns y, dx, dw in (B, T, C) / (C, K)."""
    x = x_btc.detach().clone().requires_grad_(True)
    ww = w.detach().clone().requires_grad_(True)
    y = fn(x.transpose(1, 2), ww, None, "silu").transpose(1, 2)
    y.backward(dy_btc)
    return y.detach(), x.grad.detach(), ww.grad.detach()


def err(a, ref):
    a, ref = a.double().cpu(), ref.double().cpu()
    return {"max_rel": float((a - ref).abs().max() / ref.abs().max().clamp_min(1e-30)),
            "norm_rel": float((a - ref).norm() / ref.norm().clamp_min(1e-30))}


def timed(f, reps):
    import torch
    for _ in range(2):
        f()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        f()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / reps * 1000


def main():
    args = parse_args()
    run = Run(args.run_dir or os.path.join(RESULTS_DIR, args.tag))
    run.capture_output()
    run.write_status(state="running")
    print(f"== short_conv_check.py {time.strftime('%Y-%m-%d %H:%M:%S')}  args: {vars(args)}")
    try:
        out = check(args, run)
    except Exception as exc:
        import traceback
        traceback.print_exc()
        run.write_status(state="failed", error=f"{type(exc).__name__}: {exc}")
        return 1
    with open(os.path.join(run.dir, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    run.write_status(state="finished")
    print("RESULT:", "MATCH" if out["match"] else "MISMATCH")
    return 0


def check(args, run):
    import torch
    from anneal import short_conv as sc

    print("torch", torch.__version__, "hip", torch.version.hip, torch.cuda.get_device_name(0))
    out = {"torch": torch.__version__, "device": torch.cuda.get_device_name(0), "numerics": [], "timing": []}
    match = True

    # Numerics: edge lengths (shorter than the kernel, chunk boundaries) on fewer channels, then full size.
    cases = [(1, 256), (3, 256), (5, 256), (2047, 512), (2049, 512), (4099, 512)] + \
            [(int(t), C) for t in args.lengths.split(",")]
    for T, ch in cases:
        x, w, dy = make(T, torch.float64, "cpu", seed=T, channels=ch)
        ref = run_fn(sc.reference_conv1d, x, w, dy)
        row = {"T": T, "channels": ch}
        for dtype in (torch.float32, torch.bfloat16):
            xd, wd, dyd = (t.to("cuda", dtype) for t in (x, w, dy))
            got = {"torch": run_fn(sc.reference_conv1d, xd, wd, dyd), "shift": run_fn(sc.causal_conv1d_fn, xd, wd, dyd)}
            dn = str(dtype).split(".")[-1]
            for impl, res in got.items():
                row[f"{impl}_{dn}"] = {name: err(a, r) for name, a, r in zip(("y", "dx", "dw"), res, ref)}
            # Pass: the shift version is within 1.5x of F.conv1d's own error in bf16, and tight in fp32.
            for name in ("y", "dx", "dw"):
                e_s, e_t = row[f"shift_{dn}"][name]["norm_rel"], row[f"torch_{dn}"][name]["norm_rel"]
                ok = e_s < 1e-5 if dtype == torch.float32 else e_s <= max(1.5 * e_t, 1e-3)
                match &= ok
                if not ok:
                    print(f"  MISMATCH T={T} {dn} {name}: shift {e_s:.2e} vs torch {e_t:.2e}")
        out["numerics"].append(row)
        print(f"T={T:5d} ch={ch:5d} norm-rel err vs fp64 | bf16 torch y/dx/dw "
              + " / ".join(f"{row['torch_bfloat16'][n]['norm_rel']:.1e}" for n in ("y", "dx", "dw"))
              + " | bf16 shift " + " / ".join(f"{row['shift_bfloat16'][n]['norm_rel']:.1e}" for n in ("y", "dx", "dw"))
              + " | fp32 shift " + " / ".join(f"{row['shift_float32'][n]['norm_rel']:.1e}" for n in ("y", "dx", "dw")))
        del x, w, dy, ref, got
        run.write_status(state="running", phase="numerics", T=T)

    # Speed and memory, bf16, the model's layout.
    for T in (int(t) for t in args.lengths.split(",")):
        x, w, dy = make(T, torch.bfloat16, "cuda", seed=1)
        xr = x.detach().requires_grad_(True)
        wr = w.detach().requires_grad_(True)
        row = {"T": T}
        def torch_contiguous(h, w_, b_, act):
            # F.conv1d on a contiguous (B, C, T) copy: separates the layout cost from the convolution itself
            return sc.reference_conv1d(h.contiguous(), w_, b_, act)

        for impl, fn in (("torch", sc.reference_conv1d), ("torch_contiguous", torch_contiguous),
                         ("shift", sc.causal_conv1d_fn)):
            def fwd():
                with torch.no_grad():
                    return fn(x.transpose(1, 2), w, None, "silu").transpose(1, 2).contiguous()

            def fwd_bwd():
                y = fn(xr.transpose(1, 2), wr, None, "silu").transpose(1, 2).contiguous()
                y.backward(dy)
                xr.grad = None
                wr.grad = None

            f_ms, fb_ms = timed(fwd, args.reps), timed(fwd_bwd, args.reps)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            base = torch.cuda.memory_allocated()
            fwd_bwd()
            torch.cuda.synchronize()
            peak = (torch.cuda.max_memory_allocated() - base) / 2**20
            row[impl] = {"fwd_ms": round(f_ms, 2), "fwd_bwd_ms": round(fb_ms, 2), "bwd_ms": round(fb_ms - f_ms, 2),
                         "peak_extra_mib": round(peak, 1),
                         "per_step_s": round(LAYERS * (2 * f_ms + (fb_ms - f_ms)) / 1000, 2)}
        row["per_step_saving_s"] = round(row["torch"]["per_step_s"] - row["shift"]["per_step_s"], 2)
        out["timing"].append(row)
        print(f"T={T}: torch on a contiguous copy: fwd {row['torch_contiguous']['fwd_ms']} ms, fwd+bwd "
              f"{row['torch_contiguous']['fwd_bwd_ms']} ms")
        print(f"T={T}: torch fwd {row['torch']['fwd_ms']} ms, fwd+bwd {row['torch']['fwd_bwd_ms']} ms, "
              f"+{row['torch']['peak_extra_mib']} MiB | shift fwd {row['shift']['fwd_ms']} ms, "
              f"fwd+bwd {row['shift']['fwd_bwd_ms']} ms, +{row['shift']['peak_extra_mib']} MiB | "
              f"x48 layers x (2 fwd + bwd): {row['torch']['per_step_s']} s vs {row['shift']['per_step_s']} s per step")
        del x, w, dy, xr, wr
    out["match"] = bool(match)
    return out


if __name__ == "__main__":
    sys.exit(main())
