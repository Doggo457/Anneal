"""Find the big intermediate files under models/ that can go once a GGUF exists.

Dry-run by default: it only lists. Nothing is deleted without --yes, and it never touches
adapter/ folders (the actual fine-tuning result, which is small) or GGUF files.

What it considers deletable:
  models/<run>/merged-bf16/      when models/<run>/gguf/*.gguf exists
  models/<run>/checkpoints/      when models/<run>/adapter/ exists (adapter is the final state)
  models/<repo__name>/*.safetensors   only with --bases, i.e. you explicitly ask to drop downloaded base weights

Usage:
  python scripts/cleanup_intermediates.py --dry-run
  python scripts/cleanup_intermediates.py --yes
  python scripts/cleanup_intermediates.py --bases --dry-run
"""
import argparse
import os
import shutil


def dir_size(path):
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models-dir", default="models")
    p.add_argument("--dry-run", action="store_true", help="list only (default behaviour anyway)")
    p.add_argument("--yes", action="store_true", help="actually delete")
    p.add_argument("--bases", action="store_true", help="also offer downloaded base-model weights")
    args = p.parse_args()

    candidates = []
    for run in sorted(os.listdir(args.models_dir)) if os.path.isdir(args.models_dir) else []:
        rp = os.path.join(args.models_dir, run)
        if not os.path.isdir(rp):
            continue
        gguf_dir = os.path.join(rp, "gguf")
        has_gguf = os.path.isdir(gguf_dir) and any(f.endswith(".gguf") for f in os.listdir(gguf_dir))
        merged = os.path.join(rp, "merged-bf16")
        if os.path.isdir(merged) and has_gguf:
            candidates.append((merged, "merged bf16, GGUF exists"))
        ckpt = os.path.join(rp, "checkpoints")
        if os.path.isdir(ckpt) and os.path.isdir(os.path.join(rp, "adapter")):
            candidates.append((ckpt, "training checkpoints, final adapter saved"))
        if args.bases and "__" in run:
            weights = [f for f in os.listdir(rp) if f.endswith((".safetensors", ".bin"))]
            if weights:
                candidates.append((rp, f"downloaded base weights ({len(weights)} shards)"))

    if not candidates:
        print("Nothing to clean.")
        return

    total = 0
    for path, why in candidates:
        size = dir_size(path)
        total += size
        print(f"{size / 2**30:7.1f} GiB  {path}   [{why}]")
    print(f"{total / 2**30:7.1f} GiB  total reclaimable")

    if not args.yes:
        print("\nDry run. Re-run with --yes to delete. Verify the GGUF generates text first.")
        return
    for path, _ in candidates:
        shutil.rmtree(path)
        print("deleted", path)


if __name__ == "__main__":
    main()
