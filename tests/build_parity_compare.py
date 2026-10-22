"""Compare two llama.cpp builds on the parity text: the llama.cpp side of tests/parity_llama_hf.py (--phase llama), run
once per build into its own folder, with the same --tokens and --lora (or none).

  python tests/parity_llama_hf.py --phase llama --tokens 2048 --no-ppl --run-dir DIR_A [--lora LORA_GGUF]
  python tests/parity_llama_hf.py --phase llama --tokens 2048 --no-ppl --run-dir DIR_B --build b11796-vulkan [--lora ...]
  python tests/build_parity_compare.py DIR_A DIR_B [--name llama]

Reports both mean losses over the scored tokens (from the forced tokens' log-probabilities, the raw logits), the
difference, top-1 agreement (the same most likely token at a position), the mean absolute log-probability difference,
and whether the greedy continuations match. The bar set beforehand for a build switch (speed-up item 2, 21 Oct 2026):
mean loss within 0.01 nats and top-1 agreement of at least 99%. Writes DIR_B/build_parity.json.
"""

import argparse
import json
import math
import os
import sys

LOSS_BAR = 0.01
TOP1_BAR = 0.99


def compare(a_dir, b_dir, name="llama"):
    import numpy as np
    a = np.load(os.path.join(a_dir, f"{name}_scores.npz"))
    b = np.load(os.path.join(b_dir, f"{name}_scores.npz"))
    with open(os.path.join(a_dir, "tokens.json"), encoding="utf-8") as fh:
        ids_a = json.load(fh)["ids"]
    with open(os.path.join(b_dir, "tokens.json"), encoding="utf-8") as fh:
        ids_b = json.load(fh)["ids"]
    if ids_a != ids_b or a["actual_lp"].shape != b["actual_lp"].shape:
        raise SystemExit("the two runs scored different token IDs: use the same --tokens and text")
    la, lb = a["actual_lp"].astype(np.float64), b["actual_lp"].astype(np.float64)
    ok = ~(np.isnan(la) | np.isnan(lb))
    loss_a, loss_b = float(-la[ok].mean()), float(-lb[ok].mean())
    top1 = float((a["top_ids"][:, 0] == b["top_ids"][:, 0]).mean())
    with open(os.path.join(a_dir, f"{name}.json"), encoding="utf-8") as fh:
        ja = json.load(fh)
    with open(os.path.join(b_dir, f"{name}.json"), encoding="utf-8") as fh:
        jb = json.load(fh)
    greedy = [{"prompt": x.get("name"), "same": x.get("text") == y.get("text")}
              for x, y in zip(ja.get("greedy") or [], jb.get("greedy") or [])]
    res = {"a": a_dir, "b": b_dir, "server_a": ja.get("server"), "server_b": jb.get("server"), "lora": jb.get("lora"),
           "positions": int(ok.sum()), "mean_loss_a": round(loss_a, 5), "mean_loss_b": round(loss_b, 5),
           "loss_diff": round(loss_b - loss_a, 5), "top1_agreement": round(top1, 5),
           "mean_abs_logprob_diff": round(float(np.abs(la[ok] - lb[ok]).mean()), 5),
           "max_abs_logprob_diff": round(float(np.abs(la[ok] - lb[ok]).max()), 5), "greedy_same": greedy,
           "bar": {"loss_within": LOSS_BAR, "top1_at_least": TOP1_BAR}}
    res["passes"] = abs(res["loss_diff"]) <= LOSS_BAR and top1 >= TOP1_BAR and math.isfinite(loss_b)
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--name", default="llama")
    args = ap.parse_args()
    res = compare(args.a, args.b, args.name)
    with open(os.path.join(args.b, "build_parity.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    print(f"mean loss {res['mean_loss_a']} -> {res['mean_loss_b']} ({res['loss_diff']:+.5f}), top-1 agreement "
          f"{res['top1_agreement']:.4f} over {res['positions']} positions, mean |logprob diff| "
          f"{res['mean_abs_logprob_diff']}, greedy same {[g['same'] for g in res['greedy_same']]}: "
          + ("PASSES" if res["passes"] else "FAILS") + f" the bar (loss within {LOSS_BAR}, top-1 >= {TOP1_BAR})")
    return 0 if res["passes"] else 1


if __name__ == "__main__":
    sys.exit(main())
