"""Check the base model as loaded for training: loss on held-out chats (no LoRA) and the RoPE frequencies.

The held-out rows are CodeAlpaca-20k rows 10000 onwards, as many as fill --len x --chunks tokens (the same rows
every run of this script has used). Default (--loss assistant): each row is scored on its own, on the
assistant's answer only, with the same mask as training (anneal/data.py: labels -100 before the final
assistant turn), and the loss and perplexity are reported per row and as a mean. --loss full is the old
measurement: the rows joined into one text, cut into --chunks chunks of --len tokens, every token of the second
half of each chunk scored (including the system prompt Qwen3.8's template adds to every row, which the real
model doesn't predict; see results/parity_findings.md). The joined text is saved as eval_text.txt either way so
llama.cpp can score the GGUF on it.

A much higher loss than expected on load points at a loading problem (for example rotary inv_freq buffers left
uninitialised by transformers v5's meta-device loading, which Unsloth warns gives "5-11x higher training loss").

Usage (the Training page adds --run-dir):
  check_base_loss.py --model unsloth/Qwen3.8-27B-unsloth-bnb-4bit [--plain] [--len 2048] [--chunks 4] [--loss full]
--plain loads with transformers + bitsandbytes only (no Unsloth patches), to tell the two apart.
"""

import argparse
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # C:\AI\training
from anneal.runtime import RESULTS_DIR, Run, gib  # noqa: E402


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", default="unsloth/Qwen3.8-27B-unsloth-bnb-4bit")
    ap.add_argument("--dataset", default="sahil2801/CodeAlpaca-20k")
    ap.add_argument("--len", type=int, default=2048)
    ap.add_argument("--chunks", type=int, default=4)
    ap.add_argument("--plain", action="store_true", help="transformers + bitsandbytes only, no Unsloth")
    ap.add_argument("--attn", choices=["sdpa", "eager"], default="sdpa",
                    help="attention implementation (with --plain): eager rules out the fused ROCm SDPA kernels")
    ap.add_argument("--gdn-heads", choices=["stock", "auto", "interleave", "tile"], default="stock",
                    help="GatedDeltaNet value-head pairing (anneal/gdn_heads.py): stock (default; also auto, "
                         "interleave) is transformers' own, value head j -> key head j // 3, and matches llama.cpp; "
                         "tile (j -> j %% 16) is the old, wrong patch, only to reproduce runs made before 18 Oct 2026")
    ap.add_argument("--gate", choices=["sigmoid", "silu"], default="sigmoid",
                    help="attention output gate (with --plain): transformers 5.17 always uses sigmoid and ignores "
                         "the config's output_gate_type; silu tests what output_gate_type='swish' implies")
    ap.add_argument("--loss", choices=["assistant", "full"], default="assistant",
                    help="assistant: each row alone, answer tokens only (the training mask); full: the old chunked "
                         "whole-text loss")
    ap.add_argument("--no-thinking", action="store_true",
                    help="render rows with the template's enable_thinking=False (no system prompt; the generation "
                         "prompt already closes the think block), for --loss assistant")
    ap.add_argument("--tag", default="check_base_loss")
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--resume", action="store_true", help="accepted for the Training page; ignored")
    return ap.parse_args()


def eval_rows(tokenizer, name, n_tokens):
    """Held-out chats (rows 10000+, never used by the other tests) as message lists, as many as fill n_tokens + 64
    tokens of ChatML text, and that text joined."""
    from datasets import load_dataset

    from anneal.data import codealpaca_messages

    data = load_dataset(name, split="train[10000:12000]")
    rows, parts, total = [], [], 0
    for i, r in enumerate(data):
        messages = codealpaca_messages(r)
        text = tokenizer.apply_chat_template(messages, tokenize=False)
        rows.append({"dataset_row": 10000 + i, "messages": messages})
        parts.append(text)
        total += len(tokenizer(text, add_special_tokens=False)["input_ids"])
        if total >= n_tokens + 64:
            break
    return rows, "".join(parts)


def eval_text(tokenizer, name, n_tokens):
    return eval_rows(tokenizer, name, n_tokens)[1]


def score_rows(model, tok, rows, max_len, run, template_kwargs=None):
    """Each row alone, loss on its trained (unmasked) tokens; also on the answer text only (after </think>, before
    <|im_end|>), the subset results/parity_findings.md calls "assistant answers"."""
    import torch

    from anneal.data import IGNORE_INDEX, MaskStats, encode_chat, print_sample

    think_end = tok.convert_tokens_to_ids("</think>")
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    stats = MaskStats()
    out, sample = [], None
    for k, r in enumerate(rows):
        e = encode_chat(tok, r["messages"], max_len=max_len, stats=stats, template_kwargs=template_kwargs)
        if e is None:
            out.append({"dataset_row": r["dataset_row"], "dropped": "longer than --len"})
            continue
        if sample is None:
            sample = print_sample(tok, e, f"sample row {r['dataset_row']} (loss mask)")
        ids = torch.tensor([e["input_ids"]], device="cuda")
        labels = torch.tensor(e["labels"][1:], device="cuda")
        with torch.no_grad():
            lp = torch.log_softmax(model(input_ids=ids).logits[0, :-1].float(), dim=-1)
        nll = -lp.gather(1, ids[0, 1:].unsqueeze(1)).squeeze(1)
        mask = labels != IGNORE_INDEX
        tgt = e["input_ids"][1:]
        start = next((i for i, t in enumerate(tgt) if t == think_end and mask[i]), None)
        end = next((i for i in range(len(tgt) - 1, -1, -1) if tgt[i] == im_end), len(tgt))
        answer = torch.zeros_like(mask)
        if start is not None:
            answer[start + 1:end] = True
        else:  # no think block in the trained span (enable_thinking=False): the answer is all of it before <|im_end|>
            answer[:end] = mask[:end]
        row = {"dataset_row": r["dataset_row"], "tokens": len(e["input_ids"]), "trained": int(mask.sum()),
               "nll_sum": float(nll[mask].sum()), "loss": float(nll[mask].mean()),
               "answer_tokens": int(answer.sum()),
               "answer_nll_sum": float(nll[answer].sum()) if answer.any() else 0.0,
               "loss_by_position": {"think_close_3": None if start is None
                                    else [round(float(x), 4) for x in nll[mask][:3]],
                                    "im_end": round(float(nll[end]), 4) if end < len(tgt) else None,
                                    "final_newline": round(float(nll[-1]), 4) if mask[-1] else None}}
        row["ppl"] = math.exp(row["loss"])
        row["answer_loss"] = row["answer_nll_sum"] / row["answer_tokens"] if row["answer_tokens"] else None
        out.append(row)
        ans = "n/a" if row["answer_loss"] is None else f"{row['answer_loss']:.4f}"
        print(f"row {r['dataset_row']}: {row['trained']:4d} trained / {row['tokens']:4d} tokens  "
              f"loss {row['loss']:.4f} (ppl {row['ppl']:.3f})  answer text only {ans}")
        run.metric(step=k + 1, loss=row["loss"], dataset_row=r["dataset_row"], trained=row["trained"])
        del lp
    return out, stats, sample


def rope_report(model):
    """Compare every rotary inv_freq buffer with 1 / base^(2i/dim) from the config."""
    import torch

    cfg = getattr(model.config, "text_config", None) or model.config
    params = getattr(cfg, "rope_parameters", None) or {}
    base = params.get("rope_theta") or getattr(cfg, "rope_theta", None)
    head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
    partial = params.get("partial_rotary_factor") or getattr(cfg, "partial_rotary_factor", 1.0) or 1.0
    dim = int(head_dim * partial)
    out = {"rope_theta": base, "head_dim": head_dim, "partial_rotary_factor": partial, "rotary_dim": dim,
           "rope_type": params.get("rope_type"), "modules": []}
    if not base:
        return out
    expected = 1.0 / (float(base) ** (torch.arange(0, dim, 2, dtype=torch.float64) / dim))
    for name, m in model.named_modules():
        inv = getattr(m, "inv_freq", None)
        if not isinstance(inv, torch.Tensor):
            continue
        got = inv.detach().double().cpu()
        entry = {"module": name, "class": type(m).__name__, "shape": list(got.shape), "dtype": str(inv.dtype),
                 "finite": bool(torch.isfinite(got).all())}
        if got.shape == expected.shape:
            entry["max_rel_err"] = float(((got - expected).abs() / expected).max())
        entry["first"] = [float(x) for x in got[:3]]
        out["modules"].append(entry)
    out["expected_first"] = [float(x) for x in expected[:3]]
    return out


def main():
    args = parse_args()
    run = Run(args.run_dir or os.path.join(RESULTS_DIR, args.tag))
    run.capture_output()
    run.write_status(state="starting", model=args.model, script=os.path.abspath(__file__))
    print(f"== check_base_loss.py {time.strftime('%Y-%m-%d %H:%M:%S')}  args: {vars(args)}")
    try:
        return check(args, run)
    except Exception as exc:
        import traceback
        traceback.print_exc()
        run.write_status(state="failed", error=f"{type(exc).__name__}: {exc}")
        return 1


def check(args, run):
    if not args.plain:
        import unsloth  # noqa: F401
        from unsloth import FastLanguageModel
    import torch

    from anneal import gdn_heads
    print(gdn_heads.apply(args.model, args.gdn_heads))

    run.write_status(state="loading")
    t0 = time.time()
    if args.plain:
        from transformers import AutoModelForImageTextToText, AutoTokenizer
        if args.gate == "silu":
            # Qwen3_5Attention gates its output with torch.sigmoid(gate) (the only torch.sigmoid call in the
            # module); route that one call to SiLU, i.e. gate * sigmoid(gate).
            import transformers.models.qwen3_5.modeling_qwen3_5 as mq

            class _Torch:
                def __getattr__(self, name):
                    return getattr(torch, name)

                @staticmethod
                def sigmoid(x):
                    return torch.nn.functional.silu(x)
            mq.torch = _Torch()
            print("attention output gate: SiLU (swish) instead of sigmoid")
        model = AutoModelForImageTextToText.from_pretrained(args.model, dtype=torch.bfloat16, device_map={"": 0},
                                                            attn_implementation=args.attn)
        print(f"attention implementation: {args.attn}")
        tokenizer = AutoTokenizer.from_pretrained(args.model)
    else:
        model, tokenizer = FastLanguageModel.from_pretrained(model_name=args.model, max_seq_length=args.len,
                                                             load_in_4bit=True, dtype=torch.bfloat16)
        FastLanguageModel.for_inference(model)
    tok = getattr(tokenizer, "tokenizer", tokenizer)
    model.eval()
    out = {"args": vars(args), "loader": "transformers+bnb" if args.plain else "unsloth",
           "load_seconds": round(time.time() - t0, 1), "allocated_gib": gib(torch.cuda.memory_allocated())}
    print(f"loaded in {out['load_seconds']} s ({type(model).__name__}), {out['allocated_gib']} GiB")

    out["rope"] = rope_report(model)
    print("RoPE:", json.dumps(out["rope"], indent=1))

    rows, text = eval_rows(tok, args.dataset, args.len * args.chunks)
    with open(os.path.join(run.dir, "eval_text.txt"), "w", encoding="utf-8") as fh:
        fh.write(text)
    if args.loss == "assistant":
        kw = {"enable_thinking": False} if args.no_thinking else None
        out["template_kwargs"] = kw
        return finish_rows(args, run, out, *score_rows(model, tok, rows, args.len, run, kw))
    ids = tok(text, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
    losses = []
    with torch.no_grad():
        for c in range(args.chunks):
            chunk = ids[c * args.len:(c + 1) * args.len]
            if len(chunk) < args.len:
                break
            x = chunk.unsqueeze(0).cuda()
            t1 = time.time()
            logits = model(input_ids=x).logits.float()
            # Score the second half of each chunk, like llama.cpp's perplexity tool.
            half = args.len // 2
            lp = torch.log_softmax(logits[0, half - 1:-1], dim=-1)
            nll = -lp.gather(1, x[0, half:].unsqueeze(1)).squeeze(1)
            losses.append(float(nll.mean()))
            print(f"chunk {c}: loss {losses[-1]:.4f} (ppl {math.exp(losses[-1]):.2f})  {time.time() - t1:.1f} s")
            run.metric(step=c + 1, loss=losses[-1])
    out["losses"] = losses
    out["mean_loss"] = sum(losses) / len(losses) if losses else None
    out["ppl"] = math.exp(out["mean_loss"]) if losses else None
    out["loss"] = "full (chunked whole text, second half of each chunk)"
    out["gdn_heads"] = gdn_heads.report()
    print("== summary")
    print(json.dumps(out, indent=2, default=str))
    with open(os.path.join(run.dir, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, default=str)
    run.write_status(state="finished")
    print("RESULT:", "OK" if losses else "NO CHUNKS")
    return 0


def finish_rows(args, run, out, rows, stats, sample):
    from anneal import gdn_heads

    scored = [r for r in rows if "loss" in r]
    n_tok = sum(r["trained"] for r in scored)
    n_ans = sum(r["answer_tokens"] for r in scored)
    out["loss"] = "assistant (each row alone, labels -100 before the final assistant turn; anneal/data.py)"
    out["rows"] = rows
    out["rows_scored"] = len(scored)
    out["trained_tokens"] = n_tok
    # Token-weighted: every scored token counts once (the training loss over these rows as one batch).
    out["mean_loss"] = sum(r["nll_sum"] for r in scored) / n_tok if n_tok else None
    out["ppl"] = math.exp(out["mean_loss"]) if n_tok else None
    # Row-weighted: the plain mean of the per-row losses (what a batch-size-1 trainer logs on average).
    out["mean_row_loss"] = sum(r["loss"] for r in scored) / len(scored) if scored else None
    out["mean_row_ppl"] = sum(r["ppl"] for r in scored) / len(scored) if scored else None
    ans = sum(r["answer_nll_sum"] for r in scored) / n_ans if n_ans else None
    out["answer_only"] = {"tokens": n_ans, "mean_loss": ans, "ppl": math.exp(ans) if n_ans else None}
    out["mask_stats"] = stats.as_dict()
    out["sample_row"] = sample
    out["gdn_heads"] = gdn_heads.report()
    print("== per row")
    for r in rows:
        print(f"  {r['dataset_row']}: " + (f"{r['trained']} trained tokens, loss {r['loss']:.4f}, ppl {r['ppl']:.3f}"
                                         if "loss" in r else r.get("dropped", "?")))
    print(f"== mean over {len(scored)} rows: token-weighted loss {out['mean_loss']:.4f} (ppl {out['ppl']:.3f}) on "
          f"{n_tok} tokens; row mean {out['mean_row_loss']:.4f}; answer text only {ans:.4f} "
          f"(ppl {math.exp(ans):.3f}) on {n_ans} tokens")
    print("== summary")
    print(json.dumps(out, indent=2, default=str))
    with open(os.path.join(run.dir, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, default=str)
    run.write_status(state="finished")
    print("RESULT:", "OK" if scored else "NO ROWS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
