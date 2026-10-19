"""Find which llama-server (Vulkan) setups can process multi-token prompts with a GGUF LoRA loaded: a fresh server per variant, then prompts of growing length.

parity_llama_hf.py with --lora scored 2,047 positions fine (each request processed a 1-token prompt, the rest
cached) and then the server died without a log line on the first uncached 25-token prompt. Each variant here
starts llama-server on Ollama's qwen3.8:27b Q4_K_M with the given flags and sends, with no prompt cache, prompts
of 1, 2, 8, 32 and 128 tokens (n_predict 8, greedy), recording which succeed, the server's exit code if it dies,
and the end of its log. Variants (names usable in --variants):
  nolora      no adapter (control)
  f16         --lora <F16 GGUF>
  f32         --lora <F32 GGUF>
  f16-ub1     --lora <F16> -ub 1 -b 1 (prompts processed one token per batch)
  f16-ub8     --lora <F16> -ub 8
  f16-fa-off  --lora <F16> -fa off
  f16-nofuse  --lora <F16> with GGML_VK_DISABLE_FUSION=1 (no fused Vulkan ops)
  f16-nocoopmat --lora <F16> with GGML_VK_DISABLE_COOPMAT=1 (no cooperative-matrix matmul kernels)
  f16-verbose --lora <F16> -lv 4 (most verbose log)

Usage (the Training page adds --run-dir):
  lora_server_check.py --lora-f16 <F16 GGUF> --lora-f32 <F32 GGUF> [--variants nolora,f16,f32,...]
"""

import argparse
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))
from anneal.runtime import RESULTS_DIR, Run  # noqa: E402
from parity_llama_hf import SERVER, _json_default, http, ollama_gguf  # noqa: E402

LENGTHS = [1, 2, 8, 32, 128]


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--lora-f16", required=True)
    ap.add_argument("--lora-f32", default=None)
    ap.add_argument("--variants", default="nolora,f16,f32,f16-ub1,f16-ub8,f16-fa-off,f16-nofuse,f16-nocoopmat,f16-verbose")
    ap.add_argument("--port", type=int, default=18096)
    ap.add_argument("--model", default="unsloth/Qwen3.8-27B-unsloth-bnb-4bit", help="tokenizer for the prompt IDs")
    ap.add_argument("--tag", default="lora_server_check")
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--resume", action="store_true", help="accepted for the Training page; starts over")
    return ap.parse_args()


def variants(args):
    f16, f32 = ["--lora", args.lora_f16], (["--lora", args.lora_f32] if args.lora_f32 else None)
    return {
        "nolora": ([], {}),
        "f16": (f16, {}),
        "f32": (f32, {}),
        "f16-ub1": (f16 + ["-ub", "1", "-b", "1"], {}),
        "f16-ub8": (f16 + ["-ub", "8"], {}),
        "f16-fa-off": (f16 + ["-fa", "off"], {}),
        "f16-nofuse": (f16, {"GGML_VK_DISABLE_FUSION": "1"}),
        "f16-nocoopmat": (f16, {"GGML_VK_DISABLE_COOPMAT": "1"}),
        "f16-verbose": (f16 + ["-lv", "4"], {}),
    }


def main():
    args = parse_args()
    run = Run(args.run_dir or os.path.join(RESULTS_DIR, args.tag))
    run.capture_output()
    names = [v for v in args.variants.split(",") if v]
    run.write_status(state="running", step=0, max_steps=len(names))
    print(f"== lora_server_check.py {time.strftime('%Y-%m-%d %H:%M:%S')}  args: {vars(args)}")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    text = open(os.path.join(ROOT, "tests", "parity_text.txt"), encoding="utf-8").read()
    ids = tok(text, add_special_tokens=False)["input_ids"]
    allv, out = variants(args), {}
    base = f"http://127.0.0.1:{args.port}"
    for vi, name in enumerate(names):
        flags, env_extra = allv[name]
        if flags is None:
            out[name] = {"skipped": "no F32 GGUF given"}
            continue
        run.write_status(state="running", step=vi, max_steps=len(names), phase=name)
        cmd = [SERVER, "-m", ollama_gguf(), "--host", "127.0.0.1", "--port", str(args.port), "-c", "4096", "-ngl", "999",
               "-np", "1", "--no-webui", "--ctx-checkpoints", "0", "--cache-ram", "0"] + flags
        env = dict(os.environ, **env_extra)
        log_path = os.path.join(run.dir, f"llama-server-{name}.log")
        print(f"[{name}] {'env ' + str(env_extra) + ' ' if env_extra else ''}{subprocess.list2cmdline(cmd)}")
        logf = open(log_path, "w", encoding="utf-8", errors="replace")
        proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, env=env,
                                creationflags=0x08000000)
        res = {"flags": flags, "env": env_extra, "prompts": []}
        try:
            t0 = time.time()
            while time.time() - t0 < 600:
                if proc.poll() is not None:
                    break
                try:
                    if http(base + "/health", timeout=5).get("status") == "ok":
                        break
                except Exception:
                    pass
                time.sleep(1)
            res["load_s"] = round(time.time() - t0, 1)
            if proc.poll() is None:
                try:
                    res["lora_adapters"] = http(base + "/lora-adapters", timeout=10)
                except Exception as exc:
                    res["lora_adapters"] = f"error: {exc}"
                for n in LENGTHS:
                    t1 = time.time()
                    try:
                        r = http(base + "/completion", {"prompt": ids[:n], "n_predict": 8, "temperature": 0,
                                                        "cache_prompt": False, "return_tokens": True}, timeout=600)
                        tim = r.get("timings") or {}
                        res["prompts"].append({"prompt_tokens": n, "ok": True, "tokens": r.get("tokens"),
                                               "prompt_ms": tim.get("prompt_ms"),
                                               "gen_tok_per_s": tim.get("predicted_per_second"),
                                               "seconds": round(time.time() - t1, 2)})
                        print(f"[{name}] prompt {n:4d} tokens: ok, prompt {tim.get('prompt_ms', 0):.0f} ms, "
                              f"gen {tim.get('predicted_per_second', 0):.2f} tok/s, tokens {r.get('tokens')}")
                    except Exception as exc:
                        time.sleep(2)
                        res["prompts"].append({"prompt_tokens": n, "ok": False, "error": f"{type(exc).__name__}: {exc}",
                                               "server_exit_code": proc.poll()})
                        print(f"[{name}] prompt {n:4d} tokens: FAILED ({type(exc).__name__}: {exc}); server exit code "
                              f"{proc.poll()}")
                        break
        finally:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(30)
                except subprocess.TimeoutExpired:
                    proc.kill()
            res["exit_code"] = proc.returncode
            logf.close()
        with open(log_path, encoding="utf-8", errors="replace") as fh:
            lines = fh.read().splitlines()
        res["log_lora_lines"] = [ln[:240] for ln in lines if "lora" in ln.lower() or "adapter" in ln.lower()][:40]
        res["log_tail"] = [ln[:240] for ln in lines[-25:]]
        res["max_ok_prompt_tokens"] = max([p["prompt_tokens"] for p in res["prompts"] if p["ok"]], default=0)
        out[name] = res
        print(f"[{name}] longest prompt that worked: {res['max_ok_prompt_tokens']}, exit code {res['exit_code']}")
        print(f"[{name}] log lines about the LoRA: {res['log_lora_lines'][:6]}")
        print(f"[{name}] log tail:\n  " + "\n  ".join(res["log_tail"][-8:]))
        run.metric(step=vi + 1, variant=name, max_ok_prompt_tokens=res["max_ok_prompt_tokens"])
    summ = {n: {k: r.get(k) for k in ("max_ok_prompt_tokens", "exit_code", "load_s", "lora_adapters", "skipped")}
            for n, r in out.items()}
    # Same greedy tokens as the control at each length show whether the adapter is applied at all.
    ctrl = out.get("nolora", {}).get("prompts", [])
    for n, r in out.items():
        if n != "nolora" and r.get("prompts"):
            summ[n]["differs_from_nolora"] = [p["tokens"] != c["tokens"] for p, c in zip(r["prompts"], ctrl)
                                              if p.get("ok") and c.get("ok")]
    with open(os.path.join(run.dir, "results.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, default=_json_default)
    with open(os.path.join(run.dir, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summ, fh, indent=1, default=_json_default)
    print("== summary")
    print(json.dumps(summ, indent=1, default=_json_default))
    run.write_status(state="finished", step=len(names))
    print("RESULT: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
