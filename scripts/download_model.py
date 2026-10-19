"""Download a Hugging Face model repo into C:\\AI\\training\\models\\hf (HF_HOME, set by the venv), resumably.

Meant to run detached: progress goes to a log file every 60 s instead of a progress bar, and a finished
download writes DONE (or FAILED) as the last log line.

Usage: python download_model.py <repo_id> [--revision SHA] [--log PATH]
"""

import argparse
import glob
import os
import threading
import time
import traceback

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

from huggingface_hub import HfApi, hf_hub_download  # noqa: E402


def log(fh, msg):
    fh.write(time.strftime("%Y-%m-%d %H:%M:%S ") + msg + "\n")
    fh.flush()


def watch(fh, repo_dir, total, stop):
    """Log bytes on disk (finished + .incomplete files) for the repo every minute."""
    last, t_last = None, time.time()
    while not stop.wait(60):
        done = sum(os.path.getsize(p) for p in glob.glob(os.path.join(repo_dir, "**", "*"), recursive=True)
                   if os.path.isfile(p))
        rate = "" if last is None else f", {(done - last) / (time.time() - t_last) / 1e6:.1f} MB/s"
        log(fh, f"progress {done / 1e9:.2f} / {total / 1e9:.2f} GB ({100 * done / max(total, 1):.1f}%){rate}")
        last, t_last = done, time.time()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("repo_id")
    ap.add_argument("--revision", default="main")
    ap.add_argument("--log", default=None)
    args = ap.parse_args()
    log_path = args.log or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs",
                                        "download-" + args.repo_id.replace("/", "--") + ".log")
    with open(log_path, "a", encoding="utf-8") as fh:
        log(fh, f"start {args.repo_id}@{args.revision} pid={os.getpid()} HF_HOME={os.environ.get('HF_HOME')} "
                f"HF_HUB_DISABLE_XET={os.environ.get('HF_HUB_DISABLE_XET')}")
        stop = threading.Event()
        try:
            info = HfApi().model_info(args.repo_id, revision=args.revision, files_metadata=True)
            files = [(s.rfilename, s.size or 0) for s in info.siblings]
            total = sum(size for _, size in files)
            log(fh, f"{len(files)} files, {total / 1e9:.2f} GB, revision {info.sha}")
            repo_dir = os.path.join(os.environ.get("HF_HOME", ""), "hub", "models--" + args.repo_id.replace("/", "--"))
            threading.Thread(target=watch, args=(fh, repo_dir, total, stop), daemon=True).start()
            # Small files first, so the config and tokenizer are usable while the weights download.
            for name, size in sorted(files, key=lambda f: f[1]):
                for attempt in range(1, 6):
                    try:
                        path = hf_hub_download(args.repo_id, name, revision=info.sha)
                        log(fh, f"ok {name} ({size / 1e9:.2f} GB) -> {path}")
                        break
                    except Exception as exc:
                        log(fh, f"retry {attempt}/5 {name}: {exc}")
                        time.sleep(min(300, 15 * 2 ** attempt))
                else:
                    raise RuntimeError(f"giving up on {name}")
            stop.set()
            log(fh, "DONE")
        except Exception:
            stop.set()
            log(fh, "FAILED\n" + traceback.format_exc())
            raise


if __name__ == "__main__":
    main()
