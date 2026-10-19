"""Step D checks on the base GGUF in llama-server (no LoRA), with the scoring server's flags.

1. Template: llama-server's /apply-template for coding prompts with enable_thinking=True must equal, character for
   character, the HF tokenizer's apply_chat_template(add_generation_prompt=True, enable_thinking=True) from the
   training snapshot (and the token IDs must match). If not, the check is repeated with --chat-template-file
   <snapshot>\\chat_template.jinja.
2. reasoning_content and content come back separated (--reasoning-format deepseek).
3. Reasoning budget: does b11435 enforce it? Per request (reasoning_budget_tokens=64) and as the server flag
   (--reasoning-budget 64). Raw output (reasoning_format none) shows exactly what is forced at the end.

  python tests\\template_budget_check.py [--run-dir DIR]     (through the Training page: the GPU is needed)
"""

import argparse
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from anneal import llama  # noqa: E402
from anneal.runtime import RESULTS_DIR, Run  # noqa: E402

OCI_PROMPT = ("You are given a list of integers `nums` and an integer `k`. Write a function `top_k_frequent(nums, k)` "
              "that returns the `k` most frequent elements, ties broken by the smaller value first.\n\n"
              "**Sample Input:**\n```\nnums = [1, 1, 1, 2, 2, 3]\nk = 2\n```\n**Sample Output:**\n```\n[1, 2]\n```")


def prompts():
    from evalplus_win import evalplus_message, load_problems
    he, mb = load_problems("humaneval"), load_problems("mbpp")
    return [("HumanEval/0", evalplus_message(he["HumanEval/0"]["prompt"])),
            ("HumanEval/32", evalplus_message(he["HumanEval/32"]["prompt"])),
            ("Mbpp/2", evalplus_message(mb["Mbpp/2"]["prompt"])),
            ("OCI-style", OCI_PROMPT),
            ("unicode", "Écris une fonction Python qui inverse une chaîne — avec des émojis 🙂 et des tabulations\t.")]


def first_diff(a, b):
    n = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
    return {"index": n, "server": a[max(0, n - 40):n + 40], "hf": b[max(0, n - 40):n + 40]} if a != b else None


def template_check(srv, tok):
    rows = []
    for name, text in prompts():
        msgs = [{"role": "user", "content": text}]
        for think in (True, False):
            server = llama.http(srv.url + "/apply-template",
                                {"messages": msgs, "chat_template_kwargs": {"enable_thinking": think}})["prompt"]
            hf = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=think)
            s_ids = llama.http(srv.url + "/tokenize", {"content": server, "add_special": False,
                                                       "parse_special": True})["tokens"]
            h_ids = tok(hf, add_special_tokens=False)["input_ids"]
            rows.append({"prompt": name, "enable_thinking": think, "chars_equal": server == hf,
                         "token_ids_equal": s_ids == h_ids, "chars": len(hf), "tokens": len(h_ids),
                         "diff": first_diff(server, hf), "server_tail": server[-80:]})
            print(f"  {name:13s} thinking={think!s:5s} chars equal {server == hf}, ids equal {s_ids == h_ids} "
                  f"({len(h_ids)} tokens)" + (f"  DIFF {rows[-1]['diff']}" if server != hf else ""))
    return rows


def chat(srv, text, max_tokens, **extra):
    body = {"messages": [{"role": "user", "content": text}], "max_tokens": max_tokens, "temperature": 0.6,
            "top_p": 0.95, "top_k": 20, "min_p": 0.0, "seed": 1234, "chat_template_kwargs": {"enable_thinking": True}}
    body.update(extra)
    t0 = time.time()
    d = llama.http(srv.url + "/v1/chat/completions", body, timeout=3600)
    msg = d["choices"][0]["message"]
    reasoning, content = msg.get("reasoning_content") or "", msg.get("content") or ""
    count = lambda s: len(llama.http(srv.url + "/tokenize", {"content": s, "add_special": False})["tokens"]) if s else 0  # noqa: E731
    return {"finish_reason": d["choices"][0].get("finish_reason"), "completion_tokens": d["usage"]["completion_tokens"],
            "reasoning_tokens": count(reasoning), "content_tokens": count(content), "seconds": round(time.time() - t0, 1),
            "reasoning_head": reasoning[:300], "reasoning_tail": reasoning[-300:], "content_head": content[:400],
            "content_has_think_tag": "<think>" in content or "</think>" in content}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run-dir", default=os.path.join(RESULTS_DIR, "template-budget-check"))
    ap.add_argument("--port", type=int, default=18091)
    ap.add_argument("--resume", action="store_true", help="accepted for the Training page; starts over")
    args = ap.parse_args()
    run = Run(args.run_dir)
    run.capture_output()
    run.write_status(state="running", step=0, max_steps=4)
    summary = {}
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(llama.SNAPSHOT)
    try:
        template_file = None
        with llama.Server(llama.scoring_flags(), os.path.join(run.dir, "llama-server.log"), args.port) as srv:
            summary["server"] = {"command": srv.describe(), "load_s": srv.load_s}
            print("== 1. template (scoring server, model's own template)\n", srv.describe())
            summary["template_gguf"] = template_check(srv, tok)
            ok = all(r["chars_equal"] and r["token_ids_equal"] for r in summary["template_gguf"])
            run.write_status(state="running", step=1, max_steps=4)
            print("== 2. reasoning/content separation")
            summary["separation"] = chat(srv, "What is 17 * 23? Reply with only the number.", 4096)
            print("  ", json.dumps(summary["separation"])[:600])
            run.write_status(state="running", step=2, max_steps=4)
            print("== 3a. per-request reasoning_budget_tokens=64")
            he0 = prompts()[0][1]
            summary["budget_request_64"] = chat(srv, he0, 4096, reasoning_budget_tokens=64)
            summary["budget_request_64_raw"] = chat(srv, he0, 4096, reasoning_budget_tokens=64, reasoning_format="none")
            summary["budget_request_none"] = chat(srv, he0, 768)
            for k in ("budget_request_64", "budget_request_64_raw", "budget_request_none"):
                print(f"  {k}: ", json.dumps(summary[k])[:700])
        run.write_status(state="running", step=3, max_steps=4)
        print("== 3b. server flag --reasoning-budget 64")
        with llama.Server(llama.scoring_flags(extra=["--reasoning-budget", "64"]),
                          os.path.join(run.dir, "llama-server-budget64.log"), args.port) as srv:
            summary["budget_flag_64"] = chat(srv, prompts()[0][1], 4096)
            summary["budget_flag_64_override_off"] = chat(srv, prompts()[0][1], 768, reasoning_budget_tokens=-1)
            print("  budget_flag_64:", json.dumps(summary["budget_flag_64"])[:700])
        if not ok:
            template_file = os.path.join(llama.SNAPSHOT, "chat_template.jinja")
            print("== 1b. template mismatch: recheck with --chat-template-file", template_file)
            with llama.Server(llama.scoring_flags(), os.path.join(run.dir, "llama-server-hf-template.log"), args.port,
                              chat_template_file=template_file) as srv:
                summary["template_hf_file"] = template_check(srv, tok)
                summary["separation_hf_file"] = chat(srv, "What is 17 * 23? Reply with only the number.", 4096)
                summary["budget_request_64_hf_file"] = chat(srv, prompts()[0][1], 4096, reasoning_budget_tokens=64)
        b = summary["budget_request_64"]
        summary["verdict"] = {
            "template_matches_gguf_template": ok,
            "chat_template_file_needed": not ok,
            "template_matches_with_hf_file": (all(r["chars_equal"] and r["token_ids_equal"]
                                                  for r in summary["template_hf_file"]) if not ok else None),
            "separated": bool(summary["separation"]["reasoning_tokens"]) and bool(summary["separation"]["content_tokens"])
                         and not summary["separation"]["content_has_think_tag"],
            "budget_enforced_per_request": b["reasoning_tokens"] <= 64 + 8,
            "budget_enforced_by_flag": summary["budget_flag_64"]["reasoning_tokens"] <= 64 + 8,
        }
        print("== verdict", json.dumps(summary["verdict"], indent=1))
        run.write_status(state="finished", step=4, max_steps=4)
    except Exception as exc:
        import traceback
        traceback.print_exc()
        run.write_status(state="failed", error=f"{type(exc).__name__}: {exc}")
        summary["error"] = str(exc)
    with open(os.path.join(run.dir, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=1)
    return 0 if "verdict" in summary else 1


if __name__ == "__main__":
    sys.exit(main())
