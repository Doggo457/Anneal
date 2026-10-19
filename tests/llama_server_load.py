"""Check that a llama-server build loads a GGUF on the GPU and generates: start it on localhost, wait for
/health, run one short completion, report speed, stop it.

Usage: python llama_server_load.py <llama-server.exe> <model.gguf> [--ctx 8192] [--port 18090] [--log PATH]
"""

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request


def get(url, timeout=5):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.status, json.loads(r.read() or b"{}")


def post(url, body, timeout=600):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("server")
    ap.add_argument("model")
    ap.add_argument("--ctx", type=int, default=8192)
    ap.add_argument("--port", type=int, default=18090)
    ap.add_argument("--log", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                                  "logs", "llama-server-load.log"))
    args = ap.parse_args()
    base = f"http://127.0.0.1:{args.port}"
    cmd = [args.server, "-m", args.model, "--host", "127.0.0.1", "--port", str(args.port), "-c", str(args.ctx),
           "-ngl", "999", "-np", "1", "--no-webui"]
    print("cmd:", " ".join(cmd))
    with open(args.log, "w", encoding="utf-8", errors="replace") as logf:
        proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                creationflags=0x08000000)  # CREATE_NO_WINDOW
        t0 = time.time()
        ok = False
        try:
            while time.time() - t0 < 600:
                if proc.poll() is not None:
                    print(f"server exited early with code {proc.returncode}")
                    break
                try:
                    status, body = get(base + "/health")
                    if status == 200 and body.get("status") == "ok":
                        ok = True
                        break
                except Exception:
                    pass
                time.sleep(2)
            load_s = time.time() - t0
            if ok:
                print(f"healthy after {load_s:.1f} s")
                props = get(base + "/props")[1]
                print("model:", props.get("model_path"), "| n_ctx:", (props.get("default_generation_settings") or {}).get("n_ctx"))
                out = post(base + "/completion", {"prompt": "<|im_start|>user\nWrite a Python function that returns "
                                                            "the n-th Fibonacci number.<|im_end|>\n<|im_start|>assistant\n",
                                                  "n_predict": 128, "temperature": 0})
                t = out.get("timings", {})
                print(f"prompt {t.get('prompt_n')} tok at {t.get('prompt_per_second', 0):.1f} tok/s, "
                      f"generated {t.get('predicted_n')} tok at {t.get('predicted_per_second', 0):.1f} tok/s")
                print("sample:", out.get("content", "")[:300].replace("\n", "\\n"))
        finally:
            proc.terminate()
            try:
                proc.wait(30)
            except subprocess.TimeoutExpired:
                proc.kill()
    with open(args.log, encoding="utf-8", errors="replace") as fh:
        lines = fh.read().splitlines()
    keys = ("version", "build", "ggml_hip", "ROCm", "HIP", "device", "arch", "model type", "file type", "offload",
            "nextn", "mtp", "error", "failed", "warning", "llama_model_load", "buffer size", "n_ctx")
    print("---- log highlights")
    for ln in lines:
        if any(k.lower() in ln.lower() for k in keys):
            print("  " + ln[:220])
    print("RESULT:", "LOADS" if ok else "DOES NOT LOAD")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
