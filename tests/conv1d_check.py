"""Check Unsloth's Triton causal_conv1d (GatedDeltaNet short conv) against the torch reference on this GPU.

Unsloth only enables it on CUDA; this verifies forward/backward numerics and speed on ROCm before we opt in.
Shapes follow Qwen3.8-27B: conv_dim = 2*16*128 + 48*128 = 10240 channels, kernel width 4, SiLU, bf16.
"""

import time

import unsloth  # noqa: F401  (unsloth_zoo refuses to import on its own)
import torch

from unsloth_zoo.temporary_patches import gdn_causal_conv1d as g


def run(fn, x, w, b, dy):
    x = x.detach().requires_grad_(True)
    w = w.detach().requires_grad_(True)
    b = b.detach().requires_grad_(True)
    y = fn(x, w, b, "silu")
    y.backward(dy)
    return y.detach().float(), x.grad.float(), w.grad.float(), b.grad.float()


def timed(f, n=10):
    for _ in range(2):
        f()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n):
        f()
    torch.cuda.synchronize()
    return (time.time() - t0) / n * 1000


def rel(a, b):
    return ((a - b).abs().max() / b.abs().max().clamp_min(1e-6)).item()


if __name__ == "__main__":
    torch.manual_seed(0)
    print("torch", torch.__version__, "hip", torch.version.hip, torch.cuda.get_device_name(0))
    ok = True
    for T in (4096, 8192):
        for dtype in (torch.bfloat16, torch.float32):
            x = torch.randn(1, 10240, T, device="cuda", dtype=dtype)
            w = torch.randn(10240, 4, device="cuda", dtype=dtype) * 0.5
            b = torch.randn(10240, device="cuda", dtype=dtype) * 0.1
            dy = torch.randn(1, 10240, T, device="cuda", dtype=dtype) * 0.01
            fast = run(g.triton_causal_conv1d, x, w, b, dy)
            ref = run(g.causal_conv1d_reference, x, w, b, dy)
            errs = [rel(f, r) for f, r in zip(fast, ref)]
            tol = 2e-2 if dtype == torch.bfloat16 else 1e-4
            passed = all(e < tol for e in errs) and not any(torch.isnan(t).any().item() for t in fast)
            ok &= passed
            with torch.no_grad():
                f_fwd = timed(lambda: g.triton_causal_conv1d(x, w, b, "silu"))
                r_fwd = timed(lambda: g.causal_conv1d_reference(x, w, b, "silu"))
            f_all = timed(lambda: run(g.triton_causal_conv1d, x, w, b, dy))
            r_all = timed(lambda: run(g.causal_conv1d_reference, x, w, b, dy))
            print(f"T={T} {str(dtype).split('.')[-1]:8s} rel.err y/dx/dw/db = "
                  + " / ".join(f"{e:.1e}" for e in errs) + f"  {'OK' if passed else 'MISMATCH'}"
                  + f" | fwd triton {f_fwd:.1f} ms torch {r_fwd:.1f} ms"
                  + f" | fwd+bwd triton {f_all:.1f} ms torch {r_all:.1f} ms")
    print("RESULT:", "MATCH" if ok else "MISMATCH")
