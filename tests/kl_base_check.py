"""Before a round trains with kl_think_beta: does switching the adapter off give the base model, and how far did an
adapter move the reasoning? On the real model (Unsloth, 4-bit, the loop's train settings), no training.

For a few rows of a training mix (the longest trained turns of each source):
  1. base hidden states with a fresh LoRA (B = 0), no gradients;
  2. a saved adapter loaded into the LoRA (train_longrun.load_init_adapter): hidden states with it on, and with it
     switched off (PeftModel.disable_adapter, as anneal/kl_loss.py does). Off must equal step 1 (within 1% of the
     adapter's own change); on must differ;
  3. per row: KL(base || adapter) per reasoning token, cross entropy per trained token for base and adapter, and the
     mean probability each gives the close tag inside the reasoning (where the teacher has not closed yet), so a
     higher number means more ready to stop thinking.
Counts and numbers only; no text leaves the machine.

  kl_base_check.py --config results\\loop\\rounds\\r002\\train.yaml --adapter results\\loop\\rounds\\r002\\train\\adapter
                   --rows 2 [--run-dir DIR]
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
    ap.add_argument("--config", required=True, help="a train.yaml (base_model, seq_len, LoRA shape, train_file)")
    ap.add_argument("--adapter", required=True, help="saved adapter folder to compare with the base")
    ap.add_argument("--rows", type=int, default=2, help="rows per source (longest trained turns)")
    ap.add_argument("--max-tokens", type=int, default=8192, help="skip rows longer than this")
    ap.add_argument("--run-dir", default=os.path.join(RESULTS_DIR, "kl-base-check"))
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
    print(gdn_heads.apply(cfg["base_model"], cfg.get("gdn_heads", "stock")))
    print(short_conv.apply(cfg.get("short_conv", "shift")))
    import torch
    from datasets import load_dataset

    from anneal import kl_loss
    from train_longrun import load_init_adapter, tokenize_rows

    t0 = time.time()
    model, tokenizer = FastLanguageModel.from_pretrained(model_name=cfg["base_model"], max_seq_length=int(cfg["seq_len"]),
                                                         load_in_4bit=True, dtype=torch.bfloat16)
    short_conv.patch_loaded()
    lora = dict(r=int(cfg["lora_r"]), lora_alpha=int(cfg["lora_alpha"]), lora_dropout=float(cfg.get("lora_dropout", 0.0)),
                target_modules=cfg["target_modules"], bias="none")
    model = FastLanguageModel.get_peft_model(model, use_gradient_checkpointing="unsloth",
                                             random_state=int(cfg.get("seed", 42)), **lora)
    model.train()  # as in training: the base pass in kl_loss runs in train mode
    tok = getattr(tokenizer, "tokenizer", tokenizer)
    close_id = tok.convert_tokens_to_ids("</think>")
    raw = load_dataset("json", data_files={"train": cfg["train_file"]})["train"]
    rows, _, _ = tokenize_rows(raw, tok, cfg.get("chat_template_kwargs") or {}, int(cfg["seq_len"]), True)
    sources = [r.get("source") for r in raw]
    by_src = {}
    for i, r in enumerate(rows):
        if len(r["input_ids"]) <= args.max_tokens:
            by_src.setdefault(r.get("source") or "?", []).append(i)
    pick = []
    for src, ix in sorted(by_src.items()):
        ix.sort(key=lambda i: -rows[i]["n_trained"])
        pick += ix[:args.rows]
    print(f"loaded in {time.time() - t0:.0f} s; {len(rows)} rows; checking {len(pick)}: "
          f"{ {s: min(len(v), args.rows) for s, v in by_src.items()} }; sources in file {len(set(sources))}")
    inner = model.get_base_model()
    decoder, head = inner.model, inner.lm_head

    def hidden(ids):
        with torch.no_grad(), kl_loss.no_checkpointing(decoder):
            return decoder(input_ids=ids, use_cache=False).last_hidden_state[0, :-1]

    def stats(h_b, h_a, ids, labels):
        y = labels[0, 1:]
        think = kl_loss.think_targets(ids, labels, close_id)[0]
        return row_stats(head, h_b, h_a, y, think, close_id, ignore_index=kl_loss.IGNORE_INDEX)

    base_h = {}
    for i in pick:
        ids = torch.tensor([rows[i]["input_ids"]], device="cuda")
        base_h[i] = hidden(ids)
    report = load_init_adapter(model, args.adapter, lora)
    out = {"adapter": args.adapter, "init_report": report, "close_id": close_id, "rows": []}
    worst_off, min_on = 0.0, None
    for i in pick:
        r = rows[i]
        ids, labels = torch.tensor([r["input_ids"]], device="cuda"), torch.tensor([r["labels"]], device="cuda")
        h_on = hidden(ids)
        with model.disable_adapter():
            h_off = hidden(ids)
        off = float((h_off.float() - base_h[i].float()).abs().max())
        on = float((h_on.float() - base_h[i].float()).abs().max())
        worst_off = max(worst_off, off)
        min_on = on if min_on is None else min(min_on, on)
        ce_b, ce_a, kl, pc_b, pc_a, n_think = stats(base_h[i], h_on, ids, labels)
        row = {"row": i, "source": r.get("source"), "tokens": len(r["input_ids"]), "trained": r["n_trained"],
               "think": n_think, "off_vs_base_max_abs": off, "on_vs_base_max_abs": on,
               "kl_base_adapter_per_think_token": round(kl, 5), "ce_base": round(ce_b, 5), "ce_adapter": round(ce_a, 5),
               "p_close_inside_base": pc_b, "p_close_inside_adapter": pc_a}
        out["rows"].append(row)
        print(row)
    out["max_off_vs_base"] = worst_off
    out["min_on_vs_base"] = min_on
    # identical kernels on identical inputs give identical bits; allow 1% of the adapter's own change for
    # run-to-run noise, and say which it was
    out["off_exact"] = worst_off == 0.0
    out["result"] = "PASS" if (min_on or 0) > 0 and worst_off <= 0.01 * min_on else "FAIL"
    return out


def row_stats(head, h_b, h_a, y, think, close_id, chunk=256, ignore_index=-100):
    """One row's numbers from base and adapter hidden states (h_b, h_a: [T, H], aligned with targets y: [T]), a few
    hundred positions at a time so no full-vocabulary tensor for the whole row is ever held (29 Oct 2026: holding
    both rows' log-probs and the KL's elementwise tensor ran a 43.9 GiB GPU out of memory on a long row).
    Returns (ce_base, ce_adapter) per trained token, KL(base || adapter) per reasoning token, the mean probability
    base and adapter give the close tag inside the reasoning (None if the row never is), and the reasoning count."""
    import torch
    import torch.nn.functional as F
    trained = y != ignore_index
    inside = think & (y != close_id)
    ce_b = ce_a = kl = pc_b = pc_a = 0.0
    with torch.no_grad():
        for s in range(0, int(y.shape[0]), chunk):
            tr, th, ins = trained[s:s + chunk], think[s:s + chunk], inside[s:s + chunk]
            if not bool(tr.any()) and not bool(th.any()):
                continue
            lb = F.log_softmax(head(h_b[s:s + chunk]).float(), -1)
            la = F.log_softmax(head(h_a[s:s + chunk]).float(), -1)
            if bool(tr.any()):
                yy = y[s:s + chunk][tr].unsqueeze(1)
                ce_b -= float(lb[tr].gather(1, yy).sum())
                ce_a -= float(la[tr].gather(1, yy).sum())
            if bool(th.any()):
                kl += float(F.kl_div(la[th], lb[th], log_target=True, reduction="sum"))
            if bool(ins.any()):
                pc_b += float(lb[ins, close_id].exp().sum())
                pc_a += float(la[ins, close_id].exp().sum())
            del lb, la
    n_tr, n_th, n_in = int(trained.sum()), int(think.sum()), int(inside.sum())
    return (ce_b / max(n_tr, 1), ce_a / max(n_tr, 1), kl / max(n_th, 1),
            pc_b / n_in if n_in else None, pc_a / n_in if n_in else None, n_th)


if __name__ == "__main__":
    sys.exit(main())
