"""Measure MTP speculative decoding in llama-server (Vulkan) over Ollama's qwen3.8:27b Q4_K_M: tokens/s with and without it, and whether greedy output stays the same.

Each configuration is a fresh llama-server on the same base GGUF; only the speculative flags differ:
  base       no speculative decoding
  mtp        --spec-type draft-mtp -md <Unsloth's MTP/mtp-Qwen3.8-27B-Q4_0.gguf> (default --spec-draft-n-max 3)
  mtp-n1/2/4 the same with --spec-draft-n-max 1 / 2 / 4
  builtin    --spec-type draft-mtp with no -md: the MTP block (blk.64 nextn) already inside Ollama's GGUF
Every configuration runs the same prompts greedily (temperature 0, fixed token IDs from the HF tokenizer) after
one warm-up request, and records generation speed from the server's own timings, wall time, the draft
acceptance it logs, and the generated token IDs, which are compared with the base run.

Usage (the Training page adds --run-dir):
  mtp_spec_bench.py [--configs base,mtp,mtp-n1,mtp-n2,mtp-n4,builtin] [--n-predict 256]
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))
from anneal.runtime import RESULTS_DIR, Run  # noqa: E402
from parity_llama_hf import SERVER, http, ollama_gguf, _json_default  # noqa: E402

MTP_GGUF = os.path.join(ROOT, "models", "unsloth-Qwen3.8-27B-GGUF-MTP", "mtp-Qwen3.8-27B-Q4_0.gguf")


def chat(text):
    return f"<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n"


PROMPTS = [
    ("code-chat", chat("Write a Python function that parses an ISO-8601 duration such as 'P3DT4H5M' into a number "
                       "of seconds, with a few pytest tests.")),
    ("explain-chat", chat("Explain the difference between optimistic and pessimistic locking in a database, with "
                          "a short example of each.")),
    ("refactor-chat", chat("Rewrite this JavaScript with async/await:\n\nfunction load(id, cb) {\n  fetch('/api/items/' "
                           "+ id).then(r => r.json()).then(data => cb(null, data)).catch(err => cb(err));\n}")),
    ("bash-chat", chat("Write a bash one-liner that counts the lines in every .py file under the current directory.")),
    ("code-raw", 'def merge_intervals(intervals):\n    """Merge overlapping [start, end] intervals and return them '
                 'sorted."""\n'),
    ("prose-raw", "The main difference between a process and a thread is"),
]

CONFIGS = {
    "base": [],
    "mtp": ["--spec-type", "draft-mtp", "-md", MTP_GGUF],
    "mtp-n1": ["--spec-type", "draft-mtp", "-md", MTP_GGUF, "--spec-draft-n-max", "1"],
    "mtp-n2": ["--spec-type", "draft-mtp", "-md", MTP_GGUF, "--spec-draft-n-max", "2"],
    "mtp-n4": ["--spec-type", "draft-mtp", "-md", MTP_GGUF, "--spec-draft-n-max", "4"],
    "builtin": ["--spec-type", "draft-mtp"],
}


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--configs", default="base,mtp,mtp-n1,mtp-n2,mtp-n4,builtin")
    ap.add_argument("--n-predict", type=int, default=256)
    ap.add_argument("--ctx", type=int, default=8192)
    ap.add_argument("--port", type=int, default=18094)
    ap.add_argument("--model", default="unsloth/Qwen3.8-27B-unsloth-bnb-4bit", help="tokenizer for the prompt IDs")
    ap.add_argument("--tag", default="mtp_spec_bench")
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--resume", action="store_true", help="accepted for the Training page; starts over")
    return ap.parse_args()


def start_server(args, run, name, extra):
    gguf = ollama_gguf()
    # Context checkpoints (a readback of the recurrent state, ~5 s each here) are off in every configuration so
    # they don't hide the generation speed; speculative rollback uses its own on-GPU state snapshots.
    cmd = [SERVER, "-m", gguf, "--host", "127.0.0.1", "--port", str(args.port), "-c", str(args.ctx), "-ngl", "999",
           "-np", "1", "--no-webui", "--ctx-checkpoints", "0", "--cache-ram", "0"] + extra
    log_path = os.path.join(run.dir, f"llama-server-{name}.log")
    print(f"[{name}] {subprocess.list2cmdline(cmd)}")
    logf = open(log_path, "w", encoding="utf-8", errors="replace")
    proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                            creationflags=0x08000000)
    t0 = time.time()
    while True:
        if proc.poll() is not None:
            logf.close()
            return None, log_path, time.time() - t0
        try:
            if http(f"http://127.0.0.1:{args.port}/health", timeout=5).get("status") == "ok":
                return (proc, logf), log_path, time.time() - t0
        except Exception:
            pass
        if time.time() - t0 > 900:
            proc.kill()
            logf.close()
            return None, log_path, time.time() - t0
        time.sleep(1)


def stop_server(handle):
    proc, logf = handle
    proc.terminate()
    try:
        proc.wait(30)
    except subprocess.TimeoutExpired:
        proc.kill()
    logf.close()


def acceptance_from_log(path):
    with open(path, encoding="utf-8", errors="replace") as fh:
        text = fh.read()
    rows = re.findall(r"task (\d+) \| draft acceptance = ([0-9.]+) \(\s*(\d+) accepted /\s*(\d+) generated\), "
                      r"mean len =\s*([0-9.]+)", text)
    return [{"task": int(t), "rate": float(r), "accepted": int(a), "drafted": int(g), "mean_len": float(m)}
            for t, r, a, g, m in rows]


def first_divergence(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def main():
    args = parse_args()
    run = Run(args.run_dir or os.path.join(RESULTS_DIR, args.tag))
    run.capture_output()
    names = [c for c in args.configs.split(",") if c]
    run.write_status(state="running", step=0, max_steps=len(names))
    print(f"== mtp_spec_bench.py {time.strftime('%Y-%m-%d %H:%M:%S')}  args: {vars(args)}")
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.model)
        prompts = [(n, tok(p, add_special_tokens=False)["input_ids"]) for n, p in PROMPTS]
        results = {}
        for ci, name in enumerate(names):
            run.write_status(state="running", step=ci, max_steps=len(names), phase=name)
            handle, log_path, load_s = start_server(args, run, name, CONFIGS[name])
            if handle is None:
                with open(log_path, encoding="utf-8", errors="replace") as fh:
                    tail = fh.read().splitlines()[-15:]
                print(f"[{name}] llama-server didn't start; log tail:\n  " + "\n  ".join(tail))
                results[name] = {"error": "server did not start", "log_tail": tail}
                continue
            base = f"http://127.0.0.1:{args.port}"
            try:
                print(f"[{name}] healthy after {load_s:.1f} s")
                http(base + "/completion", {"prompt": prompts[0][1], "n_predict": 16, "temperature": 0,
                                            "cache_prompt": False})  # warm-up
                rows = []
                for pname, ids in prompts:
                    t0 = time.time()
                    r = http(base + "/completion", {"prompt": ids, "n_predict": args.n_predict, "temperature": 0,
                                                    "cache_prompt": False, "return_tokens": True}, timeout=1800)
                    wall = time.time() - t0
                    tim = r.get("timings") or {}
                    rows.append({"prompt": pname, "tokens": r.get("tokens") or [], "text": r.get("content", ""),
                                 "wall_s": round(wall, 2), "timings": tim})
                    print(f"[{name}] {pname:13s} {tim.get('predicted_n')} tok at {tim.get('predicted_per_second', 0):6.2f} "
                          f"tok/s (wall {wall:5.1f} s) draft {tim.get('draft_n')}/{tim.get('draft_n_accepted')}")
            finally:
                stop_server(handle)
            gen_n = sum(x["timings"].get("predicted_n", 0) for x in rows)
            gen_ms = sum(x["timings"].get("predicted_ms", 0) for x in rows)
            results[name] = {"args": CONFIGS[name], "load_s": round(load_s, 1), "rows": rows,
                             "tok_per_s": round(gen_n / (gen_ms / 1000), 2) if gen_ms else None,
                             "wall_s": round(sum(x["wall_s"] for x in rows), 1),
                             "acceptance": acceptance_from_log(log_path)}
            acc = results[name]["acceptance"]
            if acc:
                a, g = sum(x["accepted"] for x in acc), sum(x["drafted"] for x in acc)
                results[name]["acceptance_total"] = {"accepted": a, "drafted": g, "rate": round(a / g, 4) if g else None}
            print(f"[{name}] overall {results[name]['tok_per_s']} tok/s, acceptance "
                  f"{results[name].get('acceptance_total')}")
            run.metric(step=ci + 1, config=name, tok_per_s=results[name]["tok_per_s"])

        # Compare every configuration's greedy output with the base run.
        summary = {"n_predict": args.n_predict, "prompts": [p[0] for p in PROMPTS], "configs": {}}
        ref = results.get("base")
        for name, res in results.items():
            if "rows" not in res:
                summary["configs"][name] = res
                continue
            s = {"tok_per_s": res["tok_per_s"], "wall_s": res["wall_s"], "load_s": res["load_s"],
                 "per_prompt_tok_per_s": {x["prompt"]: round(x["timings"].get("predicted_per_second", 0), 2)
                                          for x in res["rows"]},
                 "acceptance": res.get("acceptance_total")}
            if ref and "rows" in ref:
                s["speedup_vs_base"] = round(res["tok_per_s"] / ref["tok_per_s"], 3)
                s["identical_to_base"] = {x["prompt"]: x["tokens"] == y["tokens"]
                                          for x, y in zip(res["rows"], ref["rows"])}
                s["first_divergence_from_base"] = {x["prompt"]: first_divergence(x["tokens"], y["tokens"])
                                                   for x, y in zip(res["rows"], ref["rows"])}
            summary["configs"][name] = s
        with open(os.path.join(run.dir, "results.json"), "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=1, default=_json_default)
        with open(os.path.join(run.dir, "summary.json"), "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=1, default=_json_default)
        print("== summary")
        print(json.dumps(summary, indent=1, default=_json_default))
        run.write_status(state="finished", step=len(names))
        print("RESULT: OK")
        return 0
    except Exception as exc:
        import traceback
        traceback.print_exc()
        run.write_status(state="failed", error=f"{type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
