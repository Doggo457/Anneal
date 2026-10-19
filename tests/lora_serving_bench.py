"""Serve Qwen3.8-27B plus a GGUF LoRA in llama-server (Vulkan): generation speed with and without MTP speculative decoding, and how fast a batch of problems finishes with parallel slots.

Every configuration is a fresh llama-server on Ollama's qwen3.8:27b Q4_K_M GGUF with context checkpoints off
(each costs ~5 s here; results/parity_findings.md) and Vulkan cooperative-matrix kernels off
(GGML_VK_DISABLE_COOPMAT=1; with them, llama-server b11435 dies on any prompt batch over 8 tokens once a LoRA is
loaded, results/lora-server-check); only the flags below differ. Greedy (temperature 0), fixed
token IDs from the HF tokenizer, one warm-up request per server, speed from the server's own timings.

Speed (-np 1, -c 8192; the 6 prompts of tests/mtp_spec_bench.py, --n-predict tokens each):
  nolora        no adapter: the reference that shows what the adapter changes
  lora          --lora <adapter.gguf>
  lora-mtp      --lora ... --spec-type draft-mtp (the MTP block inside Ollama's GGUF, no -md)
  lora-mtp-md4  --lora ... --spec-type draft-mtp -md <Unsloth's MTP GGUF> --spec-draft-n-max 4
Batch (a batch of --batch coding problems in the chat template, --n-predict tokens each, -c 16384):
  seq           -np 1, one request at a time
  seq-mtp       -np 1 + --spec-type draft-mtp, one at a time
  np4           -np 4, 4 requests in flight
  np4-mtp       -np 4 + --spec-type draft-mtp, 4 in flight (if llama-server accepts both)
The batch figure is wall time from the first request to the last answer, and aggregate tok/s = all generated
tokens / that wall time.

Usage (the Training page adds --run-dir):
  lora_serving_bench.py --lora C:\\AI\\training\\results\\smoke-27b-stock\\adapter-gguf\\<file>.gguf
                        [--configs nolora,lora,lora-mtp,lora-mtp-md4,seq,seq-mtp,np4,np4-mtp]
"""

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))
from anneal.runtime import RESULTS_DIR, Run  # noqa: E402
from mtp_spec_bench import MTP_GGUF, PROMPTS, acceptance_from_log, first_divergence  # noqa: E402
from parity_llama_hf import SERVER, _json_default, http, ollama_gguf  # noqa: E402

PROBLEMS = [
    "Write a Python function that merges two sorted lists into one sorted list.",
    "Write a JavaScript function that debounces another function.",
    "Write a SQL query that finds duplicate email addresses in a users(id, email) table.",
    "Write a bash script that renames every .jpeg file in the current folder to .jpg.",
    "Write a Python class implementing an LRU cache with get and put methods.",
    "Write a Go function that reverses a slice of ints in place.",
    "Write a Rust function that counts word frequencies in a string and returns a HashMap.",
    "Write a Python function that checks whether a string is a valid IPv4 address.",
    "Write a C function that returns the length of a null-terminated string without using strlen.",
    "Write a TypeScript type guard that checks whether a value is a non-empty array of strings.",
    "Write a Python generator that yields the Fibonacci sequence.",
    "Write a Python snippet that extracts all hex colour codes such as #fff or #a1b2c3 from CSS text.",
]


def configs(lora):
    L = ["--lora", lora]
    return {
        "nolora": {"np": 1, "ctx": 8192, "flags": [], "part": "speed"},
        "lora": {"np": 1, "ctx": 8192, "flags": L, "part": "speed"},
        "lora-mtp": {"np": 1, "ctx": 8192, "flags": L + ["--spec-type", "draft-mtp"], "part": "speed"},
        "lora-mtp-md4": {"np": 1, "ctx": 8192, "part": "speed",
                         "flags": L + ["--spec-type", "draft-mtp", "-md", MTP_GGUF, "--spec-draft-n-max", "4"]},
        "seq": {"np": 1, "ctx": 16384, "flags": L, "part": "batch"},
        "seq-mtp": {"np": 1, "ctx": 16384, "flags": L + ["--spec-type", "draft-mtp"], "part": "batch"},
        "np4": {"np": 4, "ctx": 16384, "flags": L, "part": "batch"},
        "np4-mtp": {"np": 4, "ctx": 16384, "flags": L + ["--spec-type", "draft-mtp"], "part": "batch"},
    }


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--lora", required=True, help="GGUF LoRA adapter (convert_lora_to_gguf.py output)")
    ap.add_argument("--configs", default="nolora,lora,lora-mtp,lora-mtp-md4,seq,seq-mtp,np4,np4-mtp")
    ap.add_argument("--n-predict", type=int, default=256)
    ap.add_argument("--batch", type=int, default=12, help="problems in the batch part (from a fixed list of 12)")
    ap.add_argument("--vk-coopmat", choices=["on", "off"], default="off",
                    help="Vulkan cooperative-matrix kernels for every server (off = GGML_VK_DISABLE_COOPMAT=1, the "
                         "default: llama-server b11435 with a LoRA dies on prompt batches over 8 tokens otherwise, "
                         "results/lora-server-check; off for the no-LoRA reference too, so only the adapter differs)")
    ap.add_argument("--port", type=int, default=18095)
    ap.add_argument("--model", default="unsloth/Qwen3.8-27B-unsloth-bnb-4bit", help="tokenizer for the prompt IDs")
    ap.add_argument("--tag", default="lora_serving_bench")
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--resume", action="store_true", help="accepted for the Training page; starts over")
    return ap.parse_args()


def start_server(args, run, name, cfg):
    cmd = [SERVER, "-m", ollama_gguf(), "--host", "127.0.0.1", "--port", str(args.port), "-c", str(cfg["ctx"]),
           "-ngl", "999", "-np", str(cfg["np"]), "--no-webui", "--ctx-checkpoints", "0", "--cache-ram", "0"]
    cmd += cfg["flags"]
    log_path = os.path.join(run.dir, f"llama-server-{name}.log")
    print(f"[{name}] {subprocess.list2cmdline(cmd)}")
    logf = open(log_path, "w", encoding="utf-8", errors="replace")
    env = dict(os.environ, GGML_VK_DISABLE_COOPMAT="1") if args.vk_coopmat == "off" else None
    proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, env=env,
                            creationflags=0x08000000)
    t0 = time.time()
    while time.time() - t0 < 900:
        if proc.poll() is not None:
            logf.close()
            return None, log_path, time.time() - t0
        try:
            if http(f"http://127.0.0.1:{args.port}/health", timeout=5).get("status") == "ok":
                return (proc, logf), log_path, time.time() - t0
        except Exception:
            pass
        time.sleep(1)
    proc.kill()
    logf.close()
    return None, log_path, time.time() - t0


def stop_server(handle):
    proc, logf = handle
    proc.terminate()
    try:
        proc.wait(30)
    except subprocess.TimeoutExpired:
        proc.kill()
    logf.close()


def log_lines(path, keys):
    with open(path, encoding="utf-8", errors="replace") as fh:
        return [ln.rstrip()[:240] for ln in fh if any(k in ln.lower() for k in keys)][:80]


def complete(base, ids, n_predict):
    t0 = time.time()
    r = http(base + "/completion", {"prompt": ids, "n_predict": n_predict, "temperature": 0, "cache_prompt": False,
                                    "return_tokens": True}, timeout=3600)
    t1 = time.time()
    return {"tokens": r.get("tokens") or [], "text": r.get("content", ""), "timings": r.get("timings") or {},
            "start": t0, "end": t1, "wall_s": round(t1 - t0, 2)}


def run_speed(args, base, prompts, name):
    rows = []
    for pname, ids in prompts:
        r = complete(base, ids, args.n_predict)
        r["prompt"] = pname
        rows.append(r)
        t = r["timings"]
        print(f"[{name}] {pname:13s} {t.get('predicted_n')} tok at {t.get('predicted_per_second', 0):6.2f} tok/s "
              f"(wall {r['wall_s']:5.1f} s) draft {t.get('draft_n')}/{t.get('draft_n_accepted')}")
    gen_n = sum(x["timings"].get("predicted_n", 0) for x in rows)
    gen_ms = sum(x["timings"].get("predicted_ms", 0) for x in rows)
    return {"rows": rows, "tok_per_s": round(gen_n / (gen_ms / 1000), 2) if gen_ms else None,
            "wall_s": round(sum(x["wall_s"] for x in rows), 1)}


def run_batch(args, base, problems, name, parallel):
    lock, done = threading.Lock(), []

    def one(i):
        r = complete(base, problems[i], args.n_predict)
        r["problem"] = i
        with lock:
            done.append(i)
            print(f"[{name}] problem {i:2d} done: {r['timings'].get('predicted_n')} tok, "
                  f"{r['timings'].get('predicted_per_second', 0):.2f} tok/s in its slot, wall {r['wall_s']} s")
        return r

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=parallel) as ex:
        rows = list(ex.map(one, range(len(problems))))
    wall = time.time() - t0
    gen_n = sum(x["timings"].get("predicted_n", 0) for x in rows)
    return {"rows": rows, "batch_wall_s": round(wall, 1), "generated_tokens": gen_n,
            "aggregate_tok_per_s": round(gen_n / wall, 2),
            "mean_slot_tok_per_s": round(sum(x["timings"].get("predicted_per_second", 0) for x in rows) / len(rows), 2)}


def main():
    args = parse_args()
    run = Run(args.run_dir or os.path.join(RESULTS_DIR, args.tag))
    run.capture_output()
    allcfg = configs(args.lora)
    names = [c for c in args.configs.split(",") if c]
    run.write_status(state="running", step=0, max_steps=len(names))
    print(f"== lora_serving_bench.py {time.strftime('%Y-%m-%d %H:%M:%S')}  args: {vars(args)}")
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.model)
        prompts = [(n, tok(p, add_special_tokens=False)["input_ids"]) for n, p in PROMPTS]
        problems = []
        for p in PROBLEMS[:args.batch]:
            text = tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)
            problems.append(tok(text, add_special_tokens=False)["input_ids"])
        results = {}
        for ci, name in enumerate(names):
            cfg = allcfg[name]
            run.write_status(state="running", step=ci, max_steps=len(names), phase=name)
            handle, log_path, load_s = start_server(args, run, name, cfg)
            if handle is None:
                with open(log_path, encoding="utf-8", errors="replace") as fh:
                    tail = fh.read().splitlines()[-20:]
                print(f"[{name}] llama-server didn't start; log tail:\n  " + "\n  ".join(tail))
                results[name] = {"cfg": cfg, "error": "server did not start", "log_tail": tail}
                run.metric(step=ci + 1, config=name, error="server did not start")
                continue
            base = f"http://127.0.0.1:{args.port}"
            try:
                print(f"[{name}] healthy after {load_s:.1f} s")
                http(base + "/completion", {"prompt": prompts[0][1], "n_predict": 16, "temperature": 0,
                                            "cache_prompt": False})  # warm-up
                try:
                    lora_state = http(base + "/lora-adapters", timeout=10)
                except Exception as exc:
                    lora_state = f"error: {exc}"
                print(f"[{name}] /lora-adapters: {lora_state}")
                if cfg["part"] == "speed":
                    res = run_speed(args, base, prompts, name)
                else:
                    res = run_batch(args, base, problems, name, cfg["np"])
            except Exception as exc:
                import traceback
                traceback.print_exc()
                res = {"error": f"{type(exc).__name__}: {exc}"}
            finally:
                stop_server(handle)
            res.update(cfg=cfg, load_s=round(load_s, 1), lora_adapters=lora_state,
                       acceptance=acceptance_from_log(log_path),
                       log_notes=log_lines(log_path, ("lora", "adapter", "spec", "draft", "mtp", "n_parallel",
                                                      "error", "warn", "not supported")))
            acc = res["acceptance"]
            if acc:
                a, g = sum(x["accepted"] for x in acc), sum(x["drafted"] for x in acc)
                res["acceptance_total"] = {"accepted": a, "drafted": g, "rate": round(a / g, 4) if g else None}
            results[name] = res
            print(f"[{name}] tok/s {res.get('tok_per_s')} | batch wall {res.get('batch_wall_s')} s, aggregate "
                  f"{res.get('aggregate_tok_per_s')} tok/s | acceptance {res.get('acceptance_total')}")
            run.metric(step=ci + 1, config=name, tok_per_s=res.get("tok_per_s"),
                       batch_wall_s=res.get("batch_wall_s"), aggregate_tok_per_s=res.get("aggregate_tok_per_s"))

        summary = {"lora": args.lora, "n_predict": args.n_predict, "vk_coopmat": args.vk_coopmat, "speed": {},
               "batch": {}}
        ref_no, ref_lora, ref_seq = results.get("nolora"), results.get("lora"), results.get("seq")
        for name, res in results.items():
            part = allcfg[name]["part"]
            if "rows" not in res:
                summary[part][name] = {k: res.get(k) for k in ("error", "log_tail", "log_notes")}
                continue
            if part == "speed":
                s = {"tok_per_s": res["tok_per_s"], "wall_s": res["wall_s"], "load_s": res["load_s"],
                     "acceptance": res.get("acceptance_total"),
                     "per_prompt_tok_per_s": {x["prompt"]: round(x["timings"].get("predicted_per_second", 0), 2)
                                              for x in res["rows"]}}
                for ref_name, ref in (("nolora", ref_no), ("lora", ref_lora)):
                    if ref and "rows" in ref and ref is not res:
                        s[f"first_divergence_from_{ref_name}"] = {
                            x["prompt"]: first_divergence(x["tokens"], y["tokens"])
                            for x, y in zip(res["rows"], ref["rows"])}
                if ref_lora and "rows" in ref_lora and ref_lora is not res:
                    s["speedup_vs_lora_no_mtp"] = round(res["tok_per_s"] / ref_lora["tok_per_s"], 3)
            else:
                s = {k: res[k] for k in ("batch_wall_s", "generated_tokens", "aggregate_tok_per_s",
                                          "mean_slot_tok_per_s", "load_s")}
                s["acceptance"] = res.get("acceptance_total")
                if ref_seq and "rows" in ref_seq and ref_seq is not res:
                    s["speedup_vs_seq"] = round(ref_seq["batch_wall_s"] / res["batch_wall_s"], 3)
                    s["first_divergence_from_seq"] = [first_divergence(x["tokens"], y["tokens"])
                                                      for x, y in zip(res["rows"], ref_seq["rows"])]
            summary[part][name] = s
        done = {n: s["batch_wall_s"] for n, s in summary["batch"].items() if "batch_wall_s" in s}
        if done:
            summary["fastest_batch"] = min(done, key=done.get)
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
