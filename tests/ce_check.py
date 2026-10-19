"""Does Unsloth's fused cross entropy with a fixed chunk budget (ce_target_gb) give the training loss the plain,
unfused computation gives? Recomputes logged training losses of a train_longrun.py dry run independently.

Step 1's loss was computed with the fresh adapter (B = 0, i.e. the base model) and step N+1's with the weights saved in
checkpoint-N, so both are recomputed here on the same rows: decoder hidden states, then the LM head in chunks of
positions and torch's cross entropy (no Unsloth fused loss), summed over trained tokens of the step's micro-batches
and divided by their count (Trainer's num_items_in_batch normalisation).

  ce_check.py --dry-run results\\resume-check-27b-memdebug\\dryrun-27b --config experiments\\dryrun-27b-memdebug.yaml
              --steps "1:6,3;3:10,15" [--run-dir DIR]
"""

import argparse
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from anneal.runtime import RESULTS_DIR, Run  # noqa: E402

os.environ.setdefault("UNSLOTH_BLOCK_COMPILE", "0")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dry-run", required=True, help="train_longrun.py run folder (metrics.jsonl, checkpoints)")
    ap.add_argument("--config", required=True)
    ap.add_argument("--steps", default="1:6,3;3:10,15", help="step:row,row;... (rows = dataset indices)")
    ap.add_argument("--chunk", type=int, default=1024)
    ap.add_argument("--run-dir", default=os.path.join(RESULTS_DIR, "ce-check"))
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    run = Run(args.run_dir)
    run.capture_output()
    run.write_status(state="running")
    try:
        out = check(args)
        with open(os.path.join(run.dir, "summary.json"), "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=1)
        print("RESULT:", out["result"])
        run.write_status(state="finished" if out["result"] == "PASS" else "failed")
        return 0 if out["result"] == "PASS" else 1
    except Exception as exc:
        import traceback
        traceback.print_exc()
        run.write_status(state="failed", error=str(exc))
        return 1


def check(args):
    import yaml
    with open(args.config, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    import unsloth  # noqa: F401
    from unsloth import FastLanguageModel

    from anneal import gdn_heads, short_conv
    print(gdn_heads.apply(cfg["base_model"], "stock"))
    print(short_conv.apply("shift"))
    import torch
    from datasets import load_dataset
    from peft import PeftModel

    from anneal.data import IGNORE_INDEX
    from train_longrun import tokenize_rows

    logged = {}
    with open(os.path.join(args.dry_run, "metrics.jsonl"), encoding="utf-8") as fh:
        for line in fh:
            m = json.loads(line)
            if "loss" in m:
                logged[m["step"]] = m["loss"]
    plan = []
    for part in args.steps.split(";"):
        step, rows = part.split(":")
        plan.append((int(step), [int(r) for r in rows.split(",")]))

    t0 = time.time()
    model, tokenizer = FastLanguageModel.from_pretrained(model_name=cfg["base_model"], max_seq_length=int(cfg["seq_len"]),
                                                         load_in_4bit=True, dtype=torch.bfloat16)
    short_conv.patch_loaded()
    tok = getattr(tokenizer, "tokenizer", tokenizer)
    raw = load_dataset("json", data_files={"train": cfg["train_file"]})["train"]
    raw = raw.select(range(min(int(cfg.get("max_samples") or len(raw)), len(raw))))
    rows, _, _ = tokenize_rows(raw, tok, cfg.get("chat_template_kwargs") or {}, int(cfg["seq_len"]), True)
    print(f"loaded in {time.time() - t0:.0f} s; {len(rows)} rows")

    def step_loss(m, idxs):
        inner = m.get_base_model() if hasattr(m, "get_base_model") else m
        decoder, head = inner.model, inner.lm_head
        nll_sum, n_tok, per_row = 0.0, 0, []
        for i in idxs:
            r = rows[i]
            ids = torch.tensor([r["input_ids"]], device="cuda")
            labels = torch.tensor(r["labels"][1:], device="cuda")
            with torch.no_grad():
                h = decoder(input_ids=ids).last_hidden_state[0, :-1]
                parts = []
                for s in range(0, h.shape[0], args.chunk):
                    logits = head(h[s:s + args.chunk]).float()
                    parts.append(torch.nn.functional.cross_entropy(
                        logits, labels[s:s + logits.shape[0]], ignore_index=IGNORE_INDEX, reduction="sum"))
            s_ = float(torch.stack(parts).sum())
            n = int((labels != IGNORE_INDEX).sum())
            nll_sum += s_
            n_tok += n
            per_row.append({"row": i, "tokens": len(r["input_ids"]), "trained": n, "loss": s_ / n})
        return nll_sum / n_tok, per_row

    out = {"steps": []}
    peft = None
    for step, idxs in plan:
        if step == 1:
            loss, per_row = step_loss(model, idxs)
            src = "base (fresh adapter, B = 0)"
        else:
            ck = os.path.join(args.dry_run, "checkpoints", f"checkpoint-{step - 1}")
            peft = PeftModel.from_pretrained(model, ck) if peft is None else peft
            peft.eval()
            loss, per_row = step_loss(peft, idxs)
            src = f"adapter from {ck}"
        diff = abs(loss - logged[step])
        out["steps"].append({"step": step, "rows": idxs, "weights": src, "logged_loss": logged[step],
                             "unfused_loss": loss, "abs_diff": diff, "per_row": per_row})
        print(f"step {step} ({src}): logged {logged[step]:.5f}, unfused {loss:.5f}, |diff| {diff:.5f}; {per_row}")
    out["result"] = "PASS" if all(s["abs_diff"] < 0.005 for s in out["steps"]) else "FAIL"
    return out


if __name__ == "__main__":
    sys.exit(main())
