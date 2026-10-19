"""Generate EvalPlus samples for HumanEval+ / MBPP+ from a local OpenAI-compatible server (llama-server),
N requests at a time, thinking ON. Writes the JSONL that `evalplus.sanitize` and
`evalplus.evaluate --samples` consume, plus per-task token and timing stats, so the first 20 problems
tell you how long the full set will take instead of guessing.

  python scripts/gen_samples.py --dataset humaneval --limit 20                  # calibrate: tokens/problem, tok/s, ETA
  python scripts/gen_samples.py --dataset humaneval --out results/<name>/humaneval_samples.jsonl
  evalplus.sanitize --samples results/<name>/humaneval_samples.jsonl
  evalplus.evaluate --dataset humaneval --samples results/<name>/humaneval_samples-sanitized.jsonl

`scripts/run_eval.sh` runs exactly that sequence. The server decides thinking and the reasoning budget
(--reasoning-format deepseek --reasoning-budget N); this script just asks for completions and records
what came back. --workers should match the server's -np.
"""
import argparse
import json
import os
import re
import statistics
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

# Same instruction EvalPlus uses for chat models, so sanitize/evaluate see the format they expect.
INSTRUCTION = ("Please provide a self-contained Python script that solves the following problem "
               "in a markdown code block:\n```\n{prompt}\n```")


def load_tasks(dataset):
    if dataset == "humaneval":
        from evalplus.data import get_human_eval_plus
        return get_human_eval_plus()
    from evalplus.data import get_mbpp_plus
    return get_mbpp_plus()


def extract_code(text):
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, re.S)
    return max(blocks, key=len) if blocks else text


def call(url, model, prompt, args, seed):
    body = {
        "model": model,
        "messages": [{"role": "user", "content": INSTRUCTION.format(prompt=prompt)}],
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        "seed": seed,
        "chat_template_kwargs": {"enable_thinking": True},
    }
    req = urllib.request.Request(f"{url}/v1/chat/completions", data=json.dumps(body).encode("utf-8"),
                                 headers={"Content-Type": "application/json", "Authorization": "Bearer none"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=args.timeout) as r:
        data = json.loads(r.read().decode("utf-8"))
    secs = time.time() - t0
    choice = data["choices"][0]
    msg = choice["message"]
    content = msg.get("content") or ""
    reasoning = msg.get("reasoning_content") or msg.get("reasoning") or ""
    if not reasoning and "</think>" in content:  # server without --reasoning-format
        head, _, content = content.partition("</think>")
        reasoning = head.replace("<think>", "")
    usage = data.get("usage") or {}
    return {
        "content": content.strip(),
        "reasoning_chars": len(reasoning),
        "completion_tokens": usage.get("completion_tokens"),
        "prompt_tokens": usage.get("prompt_tokens"),
        "finish_reason": choice.get("finish_reason"),
        "seconds": round(secs, 1),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", choices=["humaneval", "mbpp"], required=True)
    p.add_argument("--url", default="http://127.0.0.1:8080")
    p.add_argument("--model", default="local")
    p.add_argument("--workers", type=int, default=1, help="concurrent requests; match llama-server -np")
    p.add_argument("--max-tokens", type=int, default=40960, help="reasoning budget + room for the answer")
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--limit", type=int, default=0, help="only the first N tasks (0 = all); use 20 to calibrate")
    p.add_argument("--timeout", type=int, default=7200)
    p.add_argument("--out", default=None, help="default results/calibration/<dataset>_samples.jsonl")
    p.add_argument("--resume", action="store_true", help="skip task_ids already in --out")
    args = p.parse_args()

    tasks = load_tasks(args.dataset)
    ids = list(tasks.keys())
    total_in_set = len(ids)
    if args.limit:
        ids = ids[:args.limit]
    out = args.out or f"results/calibration/{args.dataset}_samples.jsonl"
    stats_path = out.replace(".jsonl", "_stats.jsonl")
    os.makedirs(os.path.dirname(out), exist_ok=True)

    done = set()
    if args.resume and os.path.exists(out):
        with open(out, encoding="utf-8") as f:
            done = {json.loads(line)["task_id"] for line in f if line.strip()}
    todo = [t for t in ids if t not in done]
    print(f"{args.dataset}: {len(todo)} tasks to run ({len(done)} already done), {args.workers} worker(s)")

    lock = threading.Lock()
    tokens, seconds = [], []
    t_start = time.time()
    with open(out, "a", encoding="utf-8") as fo, open(stats_path, "a", encoding="utf-8") as fs, \
            ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(call, args.url, args.model, tasks[t]["prompt"], args, args.seed): t for t in todo}
        for n, fut in enumerate(as_completed(futures), 1):
            tid = futures[fut]
            try:
                r = fut.result()
            except Exception as e:  # noqa: BLE001
                print(f"  {tid}: FAILED {type(e).__name__}: {str(e)[:120]}")
                continue
            with lock:
                fo.write(json.dumps({"task_id": tid, "solution": extract_code(r["content"])}) + "\n")
                fo.flush()
                fs.write(json.dumps({"task_id": tid, **{k: v for k, v in r.items() if k != "content"}}) + "\n")
                fs.flush()
                if r["completion_tokens"]:
                    tokens.append(r["completion_tokens"])
                seconds.append(r["seconds"])
            flag = "" if r["finish_reason"] == "stop" else f"  [{r['finish_reason']}]"
            print(f"  {n:4d}/{len(todo)} {tid:14s} {r['completion_tokens'] or '?':>6} tok  {r['seconds']:6.0f}s{flag}")

    wall = time.time() - t_start
    if not tokens:
        print("No completed tasks with usage data.")
        return
    mean_tok = statistics.mean(tokens)
    agg_tps = sum(tokens) / wall
    print("\n== calibration ==")
    print(f"tasks: {len(tokens)}   tokens/problem: mean {mean_tok:.0f}, median {statistics.median(tokens):.0f}, "
          f"max {max(tokens)}   aggregate {agg_tps:.1f} tok/s   wall {wall / 60:.1f} min")
    remaining = total_in_set - len(done) - len(tokens)
    eta_h = remaining * mean_tok / agg_tps / 3600
    print(f"projected for the remaining {remaining} tasks in {args.dataset}: {eta_h:.1f} h at this rate")
    both = (164 + 378) * mean_tok / agg_tps / 3600
    print(f"projected full HumanEval+ + MBPP+ pass (542 tasks): {both:.1f} h at this rate")
    print(f"samples -> {out}\nnext: evalplus.sanitize --samples {out}")


if __name__ == "__main__":
    main()
