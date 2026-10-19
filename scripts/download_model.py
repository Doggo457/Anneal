"""Download a model from Hugging Face into models/, but check the size against free disk first.

A 32B model in bf16 is ~65GB. This script sums the file sizes before starting so you never
discover a full disk at 90% of a download.

Usage:
  python scripts/download_model.py Qwen/Qwen3-32B
  python scripts/download_model.py Qwen/Qwen3-32B --exclude "*.gguf" "original/*"
  python scripts/download_model.py unsloth/Qwen3-32B-GGUF --include "*Q4_K_M*"

Needs: pip install huggingface_hub, and `huggingface-cli login` for gated models.
"""
import argparse
import os
import shutil
import sys
from fnmatch import fnmatch

from huggingface_hub import HfApi, snapshot_download


def main():
    p = argparse.ArgumentParser()
    p.add_argument("repo_id")
    p.add_argument("--dest", default="models", help="parent folder (default: models/)")
    p.add_argument("--include", nargs="*", default=None, help="only files matching these globs")
    p.add_argument("--exclude", nargs="*", default=["*.gguf", "original/*", "*.msgpack", "*.h5"],
                   help="skip files matching these globs (default skips GGUF/original/TF/Flax dupes)")
    p.add_argument("--yes", action="store_true", help="skip the confirmation prompt")
    p.add_argument("--headroom-gb", type=float, default=20, help="free space to keep after download")
    args = p.parse_args()

    api = HfApi()
    info = api.model_info(args.repo_id, files_metadata=True)
    files = []
    for s in info.siblings:
        name = s.rfilename
        if args.include and not any(fnmatch(name, g) for g in args.include):
            continue
        if args.exclude and any(fnmatch(name, g) for g in args.exclude):
            continue
        files.append((name, s.size or 0))

    total = sum(size for _, size in files)
    dest = os.path.join(args.dest, args.repo_id.replace("/", "__"))
    free = shutil.disk_usage(args.dest if os.path.isdir(args.dest) else ".").free

    print(f"{args.repo_id}: {len(files)} files, {total / 2**30:.1f} GiB")
    for name, size in sorted(files, key=lambda x: -x[1])[:8]:
        print(f"  {size / 2**30:6.1f} GiB  {name}")
    if len(files) > 8:
        print(f"  ... and {len(files) - 8} more")
    print(f"Free disk: {free / 2**30:.0f} GiB. After download: {(free - total) / 2**30:.0f} GiB.")

    if free - total < args.headroom_gb * 2**30:
        print(f"Refusing: less than {args.headroom_gb} GiB would remain. Run scripts/cleanup_intermediates.py first.")
        sys.exit(1)
    if not args.yes and input("Proceed? [y/N] ").strip().lower() != "y":
        sys.exit(0)

    path = snapshot_download(
        args.repo_id,
        local_dir=dest,
        allow_patterns=args.include,
        ignore_patterns=args.exclude,
    )
    print(f"Done: {path}")
    print("Remember to delete the bf16 copy once you have a GGUF you've verified (cleanup_intermediates.py).")


if __name__ == "__main__":
    main()
