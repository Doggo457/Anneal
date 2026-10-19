"""Explain where MTP speculative decoding changed the greedy output: at each first divergence found by mtp_spec_bench.py, ask the plain model (no speculative decoding) for its top-5 next tokens.

If the base run's token and the speculative run's token are the plain model's top two with nearly equal
probability, the divergence is a near-tie broken differently by batched verification (several tokens per
forward pass) versus one-token decoding, not a draft error: speculative decoding only ever emits tokens the
target model itself picked at verification.

Usage (the Training page adds --run-dir): mtp_divergence_check.py --bench results/mtp-spec-bench
"""

import argparse
import json
import math
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))
from anneal.runtime import RESULTS_DIR, Run  # noqa: E402
from mtp_spec_bench import PROMPTS, start_server, stop_server  # noqa: E402
from parity_llama_hf import _json_default, http  # noqa: E402


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--bench", default=os.path.join(RESULTS_DIR, "mtp-spec-bench"))
    ap.add_argument("--model", default="unsloth/Qwen3.8-27B-unsloth-bnb-4bit")
    ap.add_argument("--ctx", type=int, default=8192)
    ap.add_argument("--port", type=int, default=18095)
    ap.add_argument("--tag", default="mtp_divergence_check")
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--resume", action="store_true", help="accepted for the Training page; ignored")
    return ap.parse_args()


def main():
    args = parse_args()
    run = Run(args.run_dir or os.path.join(RESULTS_DIR, args.tag))
    run.capture_output()
    run.write_status(state="running")
    print(f"== mtp_divergence_check.py {time.strftime('%Y-%m-%d %H:%M:%S')}  args: {vars(args)}")
    with open(os.path.join(args.bench, "results.json"), encoding="utf-8") as fh:
        res = json.load(fh)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    prompt_ids = {n: tok(p, add_special_tokens=False)["input_ids"] for n, p in PROMPTS}
    base = {r["prompt"]: r["tokens"] for r in res["base"]["rows"]}

    handle, log_path, load_s = start_server(args, run, "base", [])
    if handle is None:
        run.write_status(state="failed", error="llama-server did not start")
        return 1
    out = []
    try:
        url = f"http://127.0.0.1:{args.port}/completion"
        seen = {}
        for cfg, r in res.items():
            if cfg == "base" or "rows" not in r:
                continue
            for row in r["rows"]:
                p, toks = row["prompt"], row["tokens"]
                ref = base[p]
                d = next((i for i, (x, y) in enumerate(zip(toks, ref)) if x != y), None)
                if d is None:
                    continue
                key = (p, d, toks[d])
                if key not in seen:
                    rr = http(url, {"prompt": prompt_ids[p] + ref[:d], "n_predict": 1, "n_probs": 20,
                                    "temperature": 0, "cache_prompt": False, "post_sampling_probs": False})
                    tops = rr["completion_probabilities"][0]["top_logprobs"]
                    probs = {e["id"]: math.exp(e["logprob"]) for e in tops}
                    seen[key] = {"top5": [(tok.decode([e["id"]]), round(math.exp(e["logprob"]), 4)) for e in tops[:5]],
                                 "p_base_token": probs.get(ref[d]), "p_spec_token": probs.get(toks[d]),
                                 "spec_token_rank": next((i for i, e in enumerate(tops) if e["id"] == toks[d]), None)}
                info = seen[key]
                gap = (info["p_base_token"] or 0) - (info["p_spec_token"] or 0)
                out.append({"config": cfg, "prompt": p, "position": d, "base_token": tok.decode([ref[d]]),
                            "spec_token": tok.decode([toks[d]]), "p_gap": round(gap, 4), **info})
                print(f"{cfg:8s} {p:13s} @{d:3d}: base {tok.decode([ref[d]])!r} p={info['p_base_token']} | spec "
                      f"{tok.decode([toks[d]])!r} p={info['p_spec_token']} (rank {info['spec_token_rank']}) | "
                      f"top5 {info['top5']}")
    finally:
        stop_server(handle)
    gaps = [o["p_gap"] for o in out]
    summ = {"divergences": out, "n": len(out), "max_p_gap": max(gaps) if gaps else None,
            "spec_token_was_rank_1_or_0": sum(1 for o in out if o["spec_token_rank"] in (0, 1))}
    with open(os.path.join(run.dir, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summ, fh, indent=1, default=_json_default)
    print(f"== {len(out)} divergences; spec token in the plain model's top 2 at {summ['spec_token_was_rank_1_or_0']}; "
          f"largest probability gap {summ['max_p_gap']}")
    run.write_status(state="finished")
    print("RESULT: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
