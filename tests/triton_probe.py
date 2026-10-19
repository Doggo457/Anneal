"""Smallest possible check that Triton can compile and run a kernel on this GPU (Windows + ROCm)."""
import time

import torch
import triton
import triton.language as tl


@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    tl.store(out_ptr + offs, tl.load(x_ptr + offs, mask=mask) + tl.load(y_ptr + offs, mask=mask), mask=mask)


if __name__ == "__main__":
    print("torch", torch.__version__, "hip", torch.version.hip, "| triton", triton.__version__)
    print("triton target:", triton.runtime.driver.active.get_current_target())
    n = 1 << 20
    x = torch.randn(n, device="cuda")
    y = torch.randn(n, device="cuda")
    out = torch.empty_like(x)
    t0 = time.time()
    add_kernel[(triton.cdiv(n, 1024),)](x, y, out, n, BLOCK=1024)
    torch.cuda.synchronize()
    print("first launch (compile) %.1fs" % (time.time() - t0))
    print("max error", (out - (x + y)).abs().max().item())
