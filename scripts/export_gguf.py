"""Convert a merged bf16 model folder to GGUF and quantise it with llama.cpp's tools.

Two steps: convert_hf_to_gguf.py (bf16 -> f16 GGUF, same size as the source) then llama-quantize
(f16 -> Q4_K_M, roughly a quarter of the size). Both need llama.cpp cloned and built (setup/03).

Usage:
  python scripts/export_gguf.py models/<run>/merged-bf16 --llama-cpp ~/src/llama.cpp
  python scripts/export_gguf.py models/<run>/merged-bf16 --llama-cpp ~/src/llama.cpp --quant Q8_0 Q4_K_M

Alternative: Unsloth can do this in one call from the training script,
  model.save_pretrained_gguf(out_dir, tok, quantization_method="q4_k_m")
but doing it explicitly keeps the steps visible and easier to debug.
"""
import argparse
import os
import shutil
import subprocess
import sys


def run(cmd):
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("merged_dir")
    p.add_argument("--llama-cpp", required=True, help="path to the llama.cpp checkout")
    p.add_argument("--quant", nargs="+", default=["Q4_K_M"])
    p.add_argument("--out", default=None, help="output folder (default: sibling 'gguf' of merged_dir)")
    p.add_argument("--keep-f16", action="store_true", help="keep the intermediate f16 GGUF")
    args = p.parse_args()

    llama = os.path.expanduser(args.llama_cpp)
    convert = os.path.join(llama, "convert_hf_to_gguf.py")
    quantize = shutil.which("llama-quantize") or next(
        (c for c in (os.path.join(llama, b, "bin", "llama-quantize") for b in ("build-vulkan", "build-rocm", "build"))
         if os.path.exists(c)), None)
    if not os.path.exists(convert) or not quantize:
        sys.exit("convert_hf_to_gguf.py or llama-quantize not found; run setup/03-llama-cpp.sh")

    out = args.out or os.path.join(os.path.dirname(args.merged_dir.rstrip("/\\")), "gguf")
    os.makedirs(out, exist_ok=True)
    name = os.path.basename(os.path.dirname(args.merged_dir.rstrip("/\\"))) or "model"
    f16 = os.path.join(out, f"{name}-f16.gguf")

    run([sys.executable, convert, args.merged_dir, "--outtype", "f16", "--outfile", f16])
    for q in args.quant:
        run([quantize, f16, os.path.join(out, f"{name}-{q}.gguf"), q])
    if not args.keep_f16:
        os.remove(f16)
        print("removed", f16)

    print("\nVerify before deleting anything else:")
    print(f"  llama-cli -m {os.path.join(out, name + '-' + args.quant[0] + '.gguf')} -ngl 999 -p 'def fib(n):' -n 64")
    print(f"Then: python scripts/cleanup_intermediates.py --dry-run   (to drop {args.merged_dir})")


if __name__ == "__main__":
    main()
