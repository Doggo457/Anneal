"""Report what this machine can actually do for the project.

Run it first on the EVO-X2 and paste the output into docs/Hardware-Notes.md.
Also works on any PC (it just reports what's missing).
"""
import os
import platform
import shutil
import subprocess
import sys
import time


def section(title: str) -> None:
    print(f"\n== {title} ==")


section("System")
print(platform.platform())
print("python", sys.version.split()[0])
try:
    import psutil

    vm = psutil.virtual_memory()
    print(f"RAM total {vm.total / 2**30:.1f} GiB, available {vm.available / 2**30:.1f} GiB")
except ImportError:
    print("psutil not installed (pip install -r requirements.txt)")
du = shutil.disk_usage(os.getcwd())
print(f"Disk at {os.getcwd()}: {du.free / 2**30:.0f} GiB free of {du.total / 2**30:.0f} GiB")

section("GPU via PyTorch")
try:
    import torch

    print("torch", torch.__version__, "| hip", getattr(torch.version, "hip", None), "| cuda", torch.version.cuda)
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            free, total = torch.cuda.mem_get_info(i)
            arch = getattr(p, "gcnArchName", "?")
            print(f"[{i}] {p.name} arch={arch} total={total / 2**30:.1f} GiB free={free / 2**30:.1f} GiB")
        # A quick bf16 matmul so you know compute works, not just enumeration.
        x = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
        torch.cuda.synchronize()
        t = time.time()
        for _ in range(10):
            _ = x @ x
        torch.cuda.synchronize()
        dt = time.time() - t
        print(f"bf16 4096^2 matmul x10: {dt:.2f}s  (~{10 * 2 * 4096**3 / dt / 1e12:.1f} TFLOPS)")
    else:
        print("torch sees no GPU. On ROCm check: rocminfo lists gfx1151, you are in the render+video groups,")
        print("and the torch wheel is a ROCm build (torch.version.hip should not be None).")
except ImportError:
    print("torch not installed (see setup/02-python-env-and-unsloth.sh)")
except Exception as e:  # noqa: BLE001
    print(f"torch present but GPU test failed: {type(e).__name__}: {e}")

section("Libraries")
for m in ["transformers", "peft", "trl", "bitsandbytes", "datasets", "evalplus", "huggingface_hub"]:
    try:
        mod = __import__(m)
        print(f"{m:16s} {getattr(mod, '__version__', '?')}")
    except Exception as e:  # noqa: BLE001
        print(f"{m:16s} missing ({type(e).__name__})")
# Unsloth patches transformers on import and can be slow; test it last.
try:
    import unsloth  # noqa: F401

    print(f"{'unsloth':16s} {getattr(unsloth, '__version__', 'ok')}")
except Exception as e:  # noqa: BLE001
    print(f"{'unsloth':16s} missing or broken ({type(e).__name__}: {str(e)[:80]})")

section("Tools on PATH")
for t in ["rocminfo", "rocm-smi", "llama-server", "llama-bench", "llama-quantize", "llama-cli"]:
    print(f"{t:16s} {shutil.which(t) or '-'}")
if shutil.which("rocminfo"):
    out = subprocess.run(["rocminfo"], capture_output=True, text=True, check=False).stdout
    targets = sorted({line.split()[-1] for line in out.splitlines() if "Name:" in line and "gfx" in line})
    print("gfx targets:", targets or "none found")

section("Next")
print("Paste the numbers above into docs/Hardware-Notes.md, then run:")
print("  python scripts/memory_estimate.py --preset qwen3-32b")
