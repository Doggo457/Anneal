"""Check that training and testing compute the same model: score one fixed token sequence with llama.cpp (Ollama's GGUF) and transformers (the Unsloth bnb-4bit weights), with both GatedDeltaNet head pairings.

The text (tests\\parity_text.txt, mixed prose and code) is tokenised once with the HF tokenizer and the same
token IDs go to both engines:
  llama   llama-server (Vulkan) on Ollama's qwen3.8:27b Q4_K_M GGUF. One /completion request with the first
          token as prompt generates the rest, each token forced by a grammar of token IDs (<[id]>), with n_probs K;
          the reported probabilities come from the raw logits, before any sampler. (Growing prefixes with
          n_predict 1 give the same numbers but cost ~5 s per call here; the first few positions are scored that way
          as a cross-check.) Also llama-perplexity on the
          check_base_loss eval text, so its numbers line up with the earlier transformers runs.
  hf      transformers + Unsloth (the training loader) on the bnb-4bit weights, one child process per pairing:
          stock (transformers' own pairing, value head j -> key head j // 3; the default) and, on request, tile
          (j -> j % 16, the anneal/gdn_heads.py patch, known wrong; --modes stock,tile).
          One forward pass over the same IDs; per-token loss, top-1 agreement and KL(llama || hf) against
          llama's top-K distribution (the rest of the vocabulary as one extra bucket).
Both also continue the prompts greedily for 64 tokens. One model is on the GPU at a time.

With a trained LoRA adapter: --adapter <PEFT folder> loads it over the bnb-4bit base on the transformers side, and
--lora <GGUF LoRA> passes it to llama-server (--lora) on the llama.cpp side, so the two trained models are
compared on the same IDs. --llama-base-too also scores llama.cpp without the LoRA (llama_base.json), to show what
the adapter changes there. The report adds "assistant_trained": the tokens the training mask scores in each
assistant turn (after <think> up to and including <|im_end|> and its newline; anneal/data.py).

Usage (the Training page adds --run-dir):
  parity_llama_hf.py [--tokens 1024] [--n-probs 1000] [--modes stock[,tile]] [--adapter DIR --lora GGUF]
"""

import argparse
import json
import math
import os
import subprocess
import sys
import time
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # C:\AI\training
sys.path.insert(0, ROOT)
from anneal.runtime import RESULTS_DIR, Run, gib  # noqa: E402

SERVER = os.path.join(ROOT, "llama.cpp", "b11435-vulkan", "llama-server.exe")
PERPLEXITY = os.path.join(ROOT, "llama.cpp", "b11435-vulkan", "llama-perplexity.exe")
OLLAMA_MANIFEST = os.path.expanduser(r"~\.ollama\models\manifests\registry.ollama.ai\library\qwen3.8\27b")
EVAL_TEXT = os.path.join(RESULTS_DIR, "baseloss-27b-unsloth-fixed", "eval_text.txt")
PROMPTS = [
    ("code", 'def merge_intervals(intervals):\n    """Merge overlapping [start, end] intervals and return them sorted."""\n'),
    ("prose", "The main difference between a process and a thread is"),
    ("chat", "<|im_start|>user\nWrite a bash one-liner that counts the lines in every .py file under the current "
             "directory.<|im_end|>\n<|im_start|>assistant\n"),
]

TEMPLATE_PROMPT = "Write a Python function that returns the n-th Fibonacci number iteratively."


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", default="unsloth/Qwen3.8-27B-unsloth-bnb-4bit")
    ap.add_argument("--gguf", default=None, help="default: Ollama's qwen3.8:27b model blob")
    ap.add_argument("--text", default=os.path.join(ROOT, "tests", "parity_text.txt"))
    ap.add_argument("--tokens", type=int, default=1024, help="score the first N tokens of the text")
    ap.add_argument("--n-probs", type=int, default=1000, help="top-K tokens llama-server returns per position")
    ap.add_argument("--new-tokens", type=int, default=64)
    ap.add_argument("--modes", default="stock",
                    help="comma-separated pairings to score with transformers: stock (correct) and/or tile (the old, "
                         "wrong patch); the 18 Oct 2026 runs used tile,interleave (interleave = stock)")
    ap.add_argument("--phase", choices=["all", "llama", "hf", "report"], default="all")
    ap.add_argument("--gdn-heads", choices=["stock", "auto", "interleave", "tile"], default="stock",
                    help="for --phase hf: stock (transformers' own pairing, correct) or tile (old, wrong patch)")
    ap.add_argument("--port", type=int, default=18093)
    ap.add_argument("--prefix-check", type=int, default=6, help="positions also scored by growing prefixes")
    ap.add_argument("--special-from", type=int, default=248044,
                    help="token IDs from here up are special (<|endoftext|>, <|im_start|>, <|im_end|>, <think> ...)")
    ap.add_argument("--no-ppl", action="store_true", help="skip llama-perplexity on the eval text")
    ap.add_argument("--adapter", default=None, help="PEFT LoRA folder to load over the HF base (transformers side)")
    ap.add_argument("--lora", default=None, help="GGUF LoRA for llama-server --lora (llama.cpp side)")
    ap.add_argument("--llama-base-too", action="store_true",
                    help="also score llama.cpp without --lora (llama_base.json), to show the adapter's effect")
    ap.add_argument("--vk-coopmat", choices=["auto", "on", "off"], default="auto",
                    help="Vulkan cooperative-matrix kernels for llama-server: auto turns them off "
                         "(GGML_VK_DISABLE_COOPMAT=1) for every server of a run with --lora, since llama-server b11435 "
                         "with a LoRA dies on prompt batches over 8 tokens otherwise (results/lora-server-check)")
    ap.add_argument("--tag", default="parity_llama_hf")
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--resume", action="store_true", help="accepted for the Training page; starts over")
    return ap.parse_args()


def _json_default(o):
    return o.item() if hasattr(o, "item") else str(o)


def ollama_gguf():
    with open(OLLAMA_MANIFEST, encoding="utf-8") as fh:
        man = json.load(fh)
    digest = next(lay["digest"] for lay in man["layers"] if lay["mediaType"] == "application/vnd.ollama.image.model")
    models = OLLAMA_MANIFEST[:OLLAMA_MANIFEST.lower().index(os.sep + "manifests" + os.sep)]
    return os.path.join(models, "blobs", digest.replace(":", "-"))


def tokens_and_prompts(args, run):
    """Tokenise once with the HF tokenizer and keep the IDs in the run folder; both engines read them from there."""
    path = os.path.join(run.dir, "tokens.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    with open(args.text, encoding="utf-8") as fh:
        text = fh.read()
    ids = tok(text, add_special_tokens=False)["input_ids"]
    prompts = [{"name": n, "text": p, "ids": tok(p, add_special_tokens=False)["input_ids"]} for n, p in PROMPTS]
    tpl = tok.apply_chat_template([{"role": "user", "content": TEMPLATE_PROMPT}], tokenize=False,
                                  add_generation_prompt=True)
    prompts.append({"name": "chat-template", "text": tpl, "ids": tok(tpl, add_special_tokens=False)["input_ids"]})
    data = {"model": args.model, "text_file": args.text, "text_tokens": len(ids), "ids": ids[:args.tokens],
            "prompts": prompts}
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, default=_json_default)
    print(f"tokenised {args.text}: {len(ids)} tokens, scoring the first {len(data['ids'])}")
    return data


# ---------------------------------------------------------------------------------------------- llama.cpp

def http(url, body=None, timeout=600):
    req = urllib.request.Request(url, data=None if body is None else json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read() or b"{}")


def entry_lp(e):
    return e["logprob"] if "logprob" in e else math.log(max(e.get("prob", 0.0), 1e-45))


def run_llama(args, run, data, lora=None, name="llama"):
    """Score the IDs with llama-server (with --lora if given); writes <name>.json and <name>_scores.npz."""
    import numpy as np

    gguf = args.gguf or ollama_gguf()
    base = f"http://127.0.0.1:{args.port}"
    # No context checkpoints or RAM prompt cache: scoring never rolls back, and each checkpoint (a copy of the
    # recurrent state read back from the GPU) costs ~5 s per request here.
    cmd = [SERVER, "-m", gguf, "--host", "127.0.0.1", "--port", str(args.port), "-c", "4096", "-ngl", "999",
           "-np", "1", "--no-webui", "--ctx-checkpoints", "0", "--cache-ram", "0"] + (["--lora", lora] if lora else [])
    no_coopmat = args.vk_coopmat == "off" or (args.vk_coopmat == "auto" and bool(args.lora))
    env = dict(os.environ, GGML_VK_DISABLE_COOPMAT="1") if no_coopmat else None
    print(f"llama-server ({name}{', GGML_VK_DISABLE_COOPMAT=1' if no_coopmat else ''}):", subprocess.list2cmdline(cmd))
    logf = open(os.path.join(run.dir, f"{name}-server.log" if name != "llama" else "llama-server.log"), "w",
                encoding="utf-8", errors="replace")
    proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, env=env,
                            creationflags=0x08000000)
    out = {"gguf": gguf, "server": SERVER, "lora": lora, "vk_coopmat_disabled": no_coopmat}
    try:
        t0 = time.time()
        while True:
            if proc.poll() is not None:
                raise RuntimeError(f"llama-server exited with code {proc.returncode} (see llama-server.log)")
            try:
                if http(base + "/health", timeout=5).get("status") == "ok":
                    break
            except Exception:
                pass
            if time.time() - t0 > 900:
                raise RuntimeError("llama-server didn't become healthy in 15 minutes")
            time.sleep(2)
        out["load_seconds"] = round(time.time() - t0, 1)
        props = http(base + "/props")
        out["n_ctx"] = (props.get("default_generation_settings") or {}).get("n_ctx")
        print(f"llama-server healthy after {out['load_seconds']} s, n_ctx {out['n_ctx']}")

        ids = data["ids"]
        # llama.cpp's own tokenisation of the same text, for the record only (the scored IDs are the HF ones)
        with open(args.text, encoding="utf-8") as fh:
            lt = http(base + "/tokenize", {"content": fh.read(), "add_special": False}).get("tokens", [])
        out["llama_tokenizer_same_ids"] = lt[:len(ids)] == ids
        print(f"llama.cpp tokenizer gives the same first {len(ids)} IDs: {out['llama_tokenizer_same_ids']}")

        K = args.n_probs
        n = len(ids) - 1
        top_ids = np.zeros((n, K), dtype=np.int32)
        top_lp = np.full((n, K), -np.inf, dtype=np.float32)
        actual_lp = np.full(n, np.nan, dtype=np.float32)

        # Forced generation: the first token is the prompt and the rest are generated, each forced by a grammar of
        # token IDs (<[id]>). With post_sampling_probs false the reported probabilities come from the raw logits,
        # so the grammar only picks the token; it doesn't change the numbers. llama.cpp's grammar sampler masks
        # end-of-generation tokens until the grammar is complete, so special tokens (<|im_end|> etc. in chat text)
        # are scored with a one-token request each (forced with logit_bias) and the stretches between them with
        # one forced generation each.
        t1 = time.time()
        run.write_status(state="running", phase="llama scoring (forced generation)")
        def special(t):
            return t >= args.special_from

        pos, requests, got_all, forced_check, fallbacks = 1, 0, [], [], 0

        def store(cps, start, m):
            for t, cp in enumerate(cps[:m]):
                tops = cp.get("top_logprobs") or cp.get("probs") or []
                row = start - 1 + t
                for j, e in enumerate(tops[:K]):
                    top_ids[row, j] = e["id"]
                    top_lp[row, j] = entry_lp(e)
                hit = [entry_lp(e) for e in tops if e["id"] == ids[start + t]]
                if cp.get("id") == ids[start + t]:
                    actual_lp[row] = entry_lp(cp)
                    if hit:
                        forced_check.append(abs(hit[0] - actual_lp[row]))
                elif hit:
                    actual_lp[row] = hit[0]

        while pos < len(ids):
            if special(ids[pos]):
                r = http(base + "/completion", {"prompt": ids[:pos], "n_predict": 1, "n_probs": K, "temperature": 0,
                                                "cache_prompt": True, "post_sampling_probs": False,
                                                "logit_bias": [[ids[pos], 1000.0]], "return_tokens": True})
                requests += 1
                fallbacks += 1
                store(r.get("completion_probabilities") or [], pos, 1)
                got_all.append(ids[pos] if (r.get("completion_probabilities") or [{}])[0].get("id") == ids[pos] else -1)
                pos += 1
                continue
            e = next((j for j in range(pos, len(ids)) if special(ids[j])), len(ids))
            grammar = "root ::= " + " ".join(f"<[{t}]>" for t in ids[pos:e])
            r = http(base + "/completion", {"prompt": ids[:pos], "n_predict": e - pos, "n_probs": K, "temperature": 0,
                                            "grammar": grammar, "cache_prompt": True, "post_sampling_probs": False,
                                            "return_tokens": True}, timeout=7200)
            requests += 1
            cps = r.get("completion_probabilities") or []
            got = (r.get("tokens") or [])[:e - pos]
            store(cps, pos, min(len(cps), e - pos))
            got_all += got + [-1] * (e - pos - len(got))
            pos = e
            if requests % 10 == 0:
                run.write_status(state="running", phase="llama scoring (forced generation)", step=pos, max_steps=n)
        out["score_seconds"] = round(time.time() - t1, 1)
        out["score_requests"] = requests
        out["score_fallbacks"] = fallbacks
        out["forced_tokens_match"] = got_all == ids[1:]
        tim = r.get("timings") or {}
        out["score_timings"] = {k: tim.get(k) for k in ("prompt_n", "predicted_n", "predicted_per_second")}
        out["forced_logprob_vs_top_list_max_abs_diff"] = float(max(forced_check)) if forced_check else None
        out["actual_in_topk"] = len(forced_check)
        out["missing_actual"] = int(np.isnan(actual_lp).sum())
        out["mean_loss"] = float(-np.nanmean(actual_lp))
        print(f"llama.cpp forced generation: {n} positions in {out['score_seconds']} s ({requests} requests, "
              f"{fallbacks} one-token fallbacks), tokens match the HF IDs: {out['forced_tokens_match']}, "
              f"missing {out['missing_actual']}; mean loss {out['mean_loss']:.4f} (ppl {math.exp(out['mean_loss']):.3f}); "
              f"real-token logprob vs its top-K entry max diff {out['forced_logprob_vs_top_list_max_abs_diff']}")

        # Cross-check the first positions the slow way: growing prefixes through the prompt path, n_predict 1.
        xc = []
        for i in range(1, 1 + min(args.prefix_check, n)):
            rr = http(base + "/completion", {"prompt": ids[:i], "n_predict": 1, "n_probs": 20, "temperature": 0,
                                             "cache_prompt": True, "post_sampling_probs": False,
                                             "logit_bias": [[ids[i], 1000.0]], "return_tokens": True})
            cp = rr["completion_probabilities"][0]
            tops = cp.get("top_logprobs") or cp.get("probs") or []
            xc.append({"pos": i, "prefix_lp": entry_lp(cp) if cp.get("id") == ids[i] else None,
                       "forced_lp": float(actual_lp[i - 1]), "prefix_top1": tops[0]["id"] if tops else None,
                       "forced_top1": int(top_ids[i - 1, 0])})
        diffs = [abs(x["prefix_lp"] - x["forced_lp"]) for x in xc if x["prefix_lp"] is not None]
        out["prefix_cross_check"] = {"positions": xc, "max_abs_logprob_diff": max(diffs) if diffs else None,
                                     "top1_same": all(x["prefix_top1"] == x["forced_top1"] for x in xc)}
        print(f"prefix cross-check on the first {len(xc)} positions: max |logprob diff| "
              f"{out['prefix_cross_check']['max_abs_logprob_diff']}, top-1 same: {out['prefix_cross_check']['top1_same']}")
        np.savez_compressed(os.path.join(run.dir, f"{name}_scores.npz"), top_ids=top_ids, top_lp=top_lp,
                            actual_lp=actual_lp)

        gens = []
        for p in data["prompts"]:
            t2 = time.time()
            r = http(base + "/completion", {"prompt": p["ids"], "n_predict": args.new_tokens, "temperature": 0,
                                            "cache_prompt": False, "return_tokens": True})
            tim = r.get("timings") or {}
            gens.append({"name": p["name"], "tokens": r.get("tokens") or [], "text": r.get("content", ""),
                         "seconds": round(time.time() - t2, 1), "tok_per_s": tim.get("predicted_per_second")})
            print(f"  greedy [{p['name']}] {len(gens[-1]['tokens'])} tokens at {tim.get('predicted_per_second', 0):.1f} tok/s: "
                  f"{gens[-1]['text'][:160]!r}")
        out["greedy"] = gens
    finally:
        proc.terminate()
        try:
            proc.wait(30)
        except subprocess.TimeoutExpired:
            proc.kill()
        logf.close()

    if name == "llama" and not args.no_ppl and os.path.exists(EVAL_TEXT):
        # Same text, context and scored half as check_base_loss.py (4 chunks x 2048, second half of each).
        cmd = [PERPLEXITY, "-m", gguf, "-f", EVAL_TEXT, "-c", "2048", "--chunks", "4", "-ngl", "999", "-b", "2048"]
        print("llama-perplexity:", subprocess.list2cmdline(cmd))
        t3 = time.time()
        res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace",
                             creationflags=0x08000000)
        with open(os.path.join(run.dir, "llama-perplexity.log"), "w", encoding="utf-8") as fh:
            fh.write(res.stdout + res.stderr)
        text = res.stdout + res.stderr
        import re
        chunks = [float(x) for x in re.findall(r"\[\d+\]([0-9.]+)", text)]
        est = re.search(r"Final estimate: PPL = ([0-9.]+)", text)
        out["eval_text_ppl"] = {"running_ppl_per_chunk": chunks, "final_ppl": float(est.group(1)) if est else None,
                                "final_loss": math.log(float(est.group(1))) if est else None,
                                "seconds": round(time.time() - t3, 1), "exit_code": res.returncode}
        print(f"llama-perplexity on the check_base_loss eval text: {out['eval_text_ppl']}")
    with open(os.path.join(run.dir, f"{name}.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, default=_json_default)
    return out


# ---------------------------------------------------------------------------------------------- transformers

def run_hf(args, run, data):
    import unsloth  # noqa: F401
    from unsloth import FastLanguageModel
    import numpy as np
    import torch

    from anneal import gdn_heads
    print(gdn_heads.apply(args.model, args.gdn_heads))
    t0 = time.time()
    model, tokenizer = FastLanguageModel.from_pretrained(model_name=args.model, max_seq_length=4096,
                                                         load_in_4bit=True, dtype=torch.bfloat16)
    lora_params = None
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
        lora_params = sum(p.numel() for n, p in model.named_parameters() if "lora_" in n)
        print(f"adapter {args.adapter}: {lora_params:,} LoRA parameters")
    FastLanguageModel.for_inference(model)
    model.eval()
    tok = getattr(tokenizer, "tokenizer", tokenizer)
    out = {"mode": args.gdn_heads, "adapter": args.adapter, "lora_params": lora_params,
           "load_seconds": round(time.time() - t0, 1), "allocated_gib": gib(torch.cuda.memory_allocated())}
    print(f"loaded in {out['load_seconds']} s, {out['allocated_gib']} GiB")

    ll = np.load(os.path.join(run.dir, "llama_scores.npz"))
    ids = torch.tensor(data["ids"], device="cuda").unsqueeze(0)
    with torch.no_grad():
        t1 = time.time()
        logits = model(input_ids=ids).logits[0, :-1].float()
        out["forward_seconds"] = round(time.time() - t1, 1)
        lp = torch.log_softmax(logits, dim=-1)
        del logits
        tgt = ids[0, 1:]
        actual = lp.gather(1, tgt.unsqueeze(1)).squeeze(1)
        top1 = lp.argmax(dim=-1)
        l_ids = torch.from_numpy(ll["top_ids"].astype(np.int64)).cuda()
        l_lp = torch.from_numpy(ll["top_lp"]).cuda()
        p = l_lp.exp()
        q_top = lp.gather(1, l_ids)
        kl_top = (p * (l_lp - q_top)).nan_to_num(0.0).sum(1)
        p_rest = (1 - p.sum(1)).clamp_min(0)
        q_rest = (1 - q_top.exp().sum(1)).clamp_min(1e-30)
        kl_rest = torch.where(p_rest > 1e-12, p_rest * (p_rest.clamp_min(1e-30).log() - q_rest.log()),
                              torch.zeros_like(p_rest))
        kl = (kl_top + kl_rest).clamp_min(0)
        agree = (top1 == l_ids[:, 0])
    out.update(mean_loss=float(-actual.mean()), top1_agreement=float(agree.float().mean()),
               mean_kl=float(kl.mean()), median_kl=float(kl.median()), p95_kl=float(kl.quantile(0.95)),
               llama_topk_mass_mean=float(p.sum(1).mean()))
    print(f"[{args.gdn_heads}] mean loss {out['mean_loss']:.4f}, top-1 agreement with llama.cpp "
          f"{out['top1_agreement']:.4f}, mean KL(llama||hf) {out['mean_kl']:.4f} (median {out['median_kl']:.4f})")
    np.savez_compressed(os.path.join(run.dir, f"hf_{args.gdn_heads}.npz"), actual_lp=actual.cpu().numpy(),
                        top1=top1.cpu().numpy().astype(np.int32), kl=kl.cpu().numpy())
    del lp
    torch.cuda.empty_cache()

    # Causality check: the first half scored on its own must give the same numbers as inside the full sequence.
    h = len(data["ids"]) // 2
    with torch.no_grad():
        lp_h = torch.log_softmax(model(input_ids=ids[:, :h]).logits[0, :-1].float(), dim=-1)
        actual_h = lp_h.gather(1, ids[0, 1:h].unsqueeze(1)).squeeze(1)
        del lp_h
    d = (actual_h - actual[:h - 1]).abs()
    out["causal_check"] = {"first_tokens": h, "max_abs_logprob_diff": float(d.max()),
                           "mean_loss_inside_full": float(-actual[:h - 1].mean()),
                           "mean_loss_alone": float(-actual_h.mean())}
    print(f"[{args.gdn_heads}] causality check, first {h} tokens alone vs inside the full sequence: "
          f"loss {out['causal_check']['mean_loss_alone']:.4f} vs {out['causal_check']['mean_loss_inside_full']:.4f}, "
          f"max |logprob diff| {out['causal_check']['max_abs_logprob_diff']:.4f}")
    torch.cuda.empty_cache()

    gens = []
    for pr in data["prompts"]:
        x = torch.tensor([pr["ids"]], device="cuda")
        t2 = time.time()
        try:
            with torch.no_grad():
                g = model.generate(input_ids=x, attention_mask=torch.ones_like(x), max_new_tokens=args.new_tokens,
                                   do_sample=False, temperature=None, top_p=None, top_k=None)
        except Exception as exc:
            import traceback
            traceback.print_exc()
            gens.append({"name": pr["name"], "tokens": [], "text": "", "error": f"{type(exc).__name__}: {exc}"})
            continue
        new = g[0, x.shape[1]:].tolist()
        dt = time.time() - t2
        gens.append({"name": pr["name"], "tokens": new, "text": tok.decode(new), "seconds": round(dt, 1),
                     "tok_per_s": round(len(new) / dt, 2)})
        print(f"  greedy [{pr['name']}] {len(new)} tokens in {dt:.1f} s: {gens[-1]['text'][:160]!r}")
    out["greedy"] = gens
    out["gdn_heads"] = gdn_heads.report()
    with open(os.path.join(run.dir, f"hf_{args.gdn_heads}.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, default=_json_default)
    return out


# ---------------------------------------------------------------------------------------------- report

def chat_roles(tok, ids):
    """Label each token of ChatML text: system / user / assistant_think / assistant_answer / header / special.
    Plain text (no <|im_start|>) is all "text"."""
    im_start = tok.convert_tokens_to_ids("<|im_start|>")
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    think_end = tok.convert_tokens_to_ids("</think>")
    if im_start not in ids:
        return ["text"] * len(ids)
    out, role, answer = [], "text", False
    for t, tid in enumerate(ids):
        if tid == im_start:
            out.append("special")
            role, answer = "header", False
            continue
        if role == "header":
            out.append("header")
            role = tok.decode([tid]).strip() or "other"
            continue
        if tid == im_end:
            out.append("special")
            role = "between"
            continue
        if tid == think_end:
            out.append("special")
            answer = True
            continue
        if role == "assistant":
            out.append("assistant_answer" if answer else "assistant_think")
        elif tid >= 248044:
            out.append("special")
        else:
            out.append(role)
    return out


def assistant_trained_mask(tok, ids):
    """True at each token the training mask (anneal/data.py) scores in an assistant turn: after
    "<|im_start|>assistant\\n<think>" up to and including its <|im_end|> and the newline after it."""
    im_start = tok.convert_tokens_to_ids("<|im_start|>")
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    think = tok.convert_tokens_to_ids("<think>")
    out, i = [False] * len(ids), 0
    while i < len(ids):
        if ids[i] == im_start and i + 1 < len(ids) and tok.decode([ids[i + 1]]).strip() == "assistant":
            j = i + 2
            while j < len(ids) and ids[j] != think and ids[j] != im_end:
                j += 1
            if j < len(ids) and ids[j] == think:
                k = j + 1
                while k < len(ids) and ids[k] != im_end:
                    out[k] = True
                    k += 1
                if k < len(ids):
                    out[k] = True
                    if k + 1 < len(ids) and tok.decode([ids[k + 1]]) == "\n":
                        out[k + 1] = True
                i = k
        i += 1
    return out


def first_divergence(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return min(len(a), len(b)) if len(a) != len(b) else None


def report(args, run, data):
    import numpy as np
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    with open(os.path.join(run.dir, "llama.json"), encoding="utf-8") as fh:
        ll = json.load(fh)
    lls = np.load(os.path.join(run.dir, "llama_scores.npz"))
    modes = [m for m in args.modes.split(",") if os.path.exists(os.path.join(run.dir, f"hf_{m}.json"))]
    hf = {}
    for m in modes:
        with open(os.path.join(run.dir, f"hf_{m}.json"), encoding="utf-8") as fh:
            hf[m] = json.load(fh)
        hf[m]["arrays"] = np.load(os.path.join(run.dir, f"hf_{m}.npz"))
    ids = data["ids"]
    n = len(ids) - 1
    l_loss = -lls["actual_lp"]
    ok = ~np.isnan(l_loss)  # positions where llama.cpp's probability of the real token is known
    import csv
    with open(os.path.join(run.dir, "per_token.csv"), "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["pos", "target_id", "target", "role", "llama_loss"] + [f"{m}_loss" for m in modes]
                   + [f"{m}_kl" for m in modes] + ["llama_top1"] + [f"{m}_top1" for m in modes])
        roles = chat_roles(tok, ids)
        for t in range(n):
            w.writerow([t + 1, ids[t + 1], tok.decode([ids[t + 1]]), roles[t + 1], f"{l_loss[t]:.5f}"]
                       + [f"{-hf[m]['arrays']['actual_lp'][t]:.5f}" for m in modes]
                       + [f"{hf[m]['arrays']['kl'][t]:.5f}" for m in modes]
                       + [int(lls["top_ids"][t, 0])] + [int(hf[m]["arrays"]["top1"][t]) for m in modes])

    def quarters(x):
        q = len(x) // 4
        return [round(float(np.nanmean(x[i * q:(i + 1) * q])), 4) for i in range(4)]

    summ = {"tokens_scored": n, "text_tokens": data["text_tokens"], "n_probs": args.n_probs,
            "llama": {"mean_loss": round(float(np.nanmean(l_loss)), 4), "ppl": round(math.exp(float(np.nanmean(l_loss))), 3),
                      "missing_actual": int((~ok).sum()),
                      "loss_by_quarter": quarters(l_loss),
                      "loss_second_half": round(float(np.nanmean(l_loss[len(l_loss) // 2:])), 4),
                      "tokenizer_same_ids": ll.get("llama_tokenizer_same_ids"),
                      "forced_tokens_match": ll.get("forced_tokens_match"), "score_seconds": ll.get("score_seconds"),
                      "lora": ll.get("lora"), "vk_coopmat_disabled": ll.get("vk_coopmat_disabled"),
                      "prefix_cross_check": (ll.get("prefix_cross_check") or {}).get("max_abs_logprob_diff"),
                      "eval_text_ppl": ll.get("eval_text_ppl")},
            "hf": {}, "greedy": []}
    for m in modes:
        a = hf[m]["arrays"]
        loss = -a["actual_lp"]
        summ["hf"][m] = {k: hf[m][k] for k in ("mean_loss", "top1_agreement", "mean_kl", "median_kl", "p95_kl",
                                                "llama_topk_mass_mean", "load_seconds", "forward_seconds")}
        summ["hf"][m]["ppl"] = round(math.exp(hf[m]["mean_loss"]), 3)
        summ["hf"][m]["loss_by_quarter"] = quarters(loss)
        summ["hf"][m]["loss_second_half"] = round(float(loss[len(loss) // 2:].mean()), 4)
        summ["hf"][m]["causal_check"] = hf[m].get("causal_check")
        summ["hf"][m]["mean_abs_loss_diff_vs_llama"] = round(float(np.abs(loss - l_loss)[ok].mean()), 4)
        summ["hf"][m]["loss_corr_vs_llama"] = round(float(np.corrcoef(loss[ok], l_loss[ok])[0, 1]), 4)
        summ["hf"][m]["gdn_patch_calls"] = (hf[m].get("gdn_heads") or {}).get("calls")
        summ["hf"][m]["gdn_heads"] = {k: (hf[m].get("gdn_heads") or {}).get(k) for k in ("mode", "patched", "pairing")}
    roles = np.array(chat_roles(tok, ids)[1:])
    trained = np.array(assistant_trained_mask(tok, ids)[1:])
    if trained.any():
        row = {"tokens": int(trained.sum()), "llama": round(float(np.nanmean(l_loss[trained])), 4)}
        for m in modes:
            row[m] = round(float((-hf[m]["arrays"]["actual_lp"])[trained].mean()), 4)
            row[f"{m}_kl"] = round(float(hf[m]["arrays"]["kl"][trained].mean()), 4)
            row[f"{m}_top1_agreement"] = round(float((hf[m]["arrays"]["top1"][trained]
                                                      == lls["top_ids"][trained, 0]).mean()), 4)
        summ["assistant_trained"] = row
    if (roles != "text").any():
        summ["loss_by_role"] = {}
        for r in sorted(set(roles.tolist())):
            ix = roles == r
            row = {"tokens": int(ix.sum()), "llama": round(float(np.nanmean(l_loss[ix])), 4)}
            for m in modes:
                row[m] = round(float((-hf[m]["arrays"]["actual_lp"])[ix].mean()), 4)
                row[f"{m}_kl"] = round(float(hf[m]["arrays"]["kl"][ix].mean()), 4)
            summ["loss_by_role"][r] = row
    for i, p in enumerate(data["prompts"]):
        g = {"prompt": p["name"], "prompt_text": p["text"], "llama": ll["greedy"][i]["text"],
             "llama_tok_per_s": ll["greedy"][i].get("tok_per_s")}
        for m in modes:
            hg = hf[m]["greedy"][i]
            g[m] = hg["text"]
            g[f"{m}_first_divergence_from_llama"] = first_divergence(ll["greedy"][i]["tokens"], hg["tokens"])
        summ["greedy"].append(g)
    base_path = os.path.join(run.dir, "llama_base_scores.npz")
    if os.path.exists(base_path):
        with open(os.path.join(run.dir, "llama_base.json"), encoding="utf-8") as fh:
            lb = json.load(fh)
        lbs = np.load(base_path)
        b_loss = -lbs["actual_lp"]
        eff = {"lora": ll.get("lora"),
               "mean_loss_base": round(float(np.nanmean(b_loss)), 4), "mean_loss_lora": summ["llama"]["mean_loss"],
               "top1_same_as_base": round(float((lbs["top_ids"][:, 0] == lls["top_ids"][:, 0]).mean()), 4),
               "mean_abs_logprob_change": round(float(np.nanmean(np.abs(b_loss - l_loss))), 4),
               "positions_changed_over_0.01": int((np.abs(b_loss - l_loss) > 0.01).sum()), "positions": n}
        if trained.any():
            eff["assistant_trained_loss_base"] = round(float(np.nanmean(b_loss[trained])), 4)
            eff["assistant_trained_loss_lora"] = round(float(np.nanmean(l_loss[trained])), 4)
        if "loss_by_role" in summ:
            for r, row in summ["loss_by_role"].items():
                row["llama_base"] = round(float(np.nanmean(b_loss[roles == r])), 4)
        eff["greedy"] = []
        for i, p in enumerate(data["prompts"]):
            a_t, b_t = ll["greedy"][i]["tokens"], lb["greedy"][i]["tokens"]
            eff["greedy"].append({"prompt": p["name"], "base": lb["greedy"][i]["text"], "lora": ll["greedy"][i]["text"],
                                  "first_divergence": first_divergence(a_t, b_t),
                                  "base_tok_per_s": lb["greedy"][i].get("tok_per_s"),
                                  "lora_tok_per_s": ll["greedy"][i].get("tok_per_s")})
        summ["llama_adapter_effect"] = eff
    if len(modes) == 2:
        a, b = modes
        closer = a if hf[a]["mean_kl"] < hf[b]["mean_kl"] else b
        ratio = max(hf[a]["mean_kl"], hf[b]["mean_kl"]) / max(min(hf[a]["mean_kl"], hf[b]["mean_kl"]), 1e-9)
        summ["verdict"] = (f"llama.cpp is closer to transformers with --gdn-heads {closer} "
                           f"(mean KL {hf[closer]['mean_kl']:.4f} vs {hf[a if closer == b else b]['mean_kl']:.4f}, "
                           f"{ratio:.1f}x)")
    print("== summary")
    print(json.dumps(summ, indent=1, default=_json_default))
    with open(os.path.join(run.dir, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summ, fh, indent=1, default=_json_default)
    return summ


# ---------------------------------------------------------------------------------------------- driver

def child(args, phase, extra):
    cmd = [sys.executable, "-u", os.path.abspath(__file__), "--phase", phase, "--run-dir", args.run_dir,
           "--model", args.model, "--tokens", str(args.tokens), "--n-probs", str(args.n_probs),
           "--new-tokens", str(args.new_tokens), "--modes", args.modes] + extra
    cmd += ["--adapter", args.adapter] if args.adapter else []
    print("== child:", subprocess.list2cmdline(cmd))
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                            text=True, encoding="utf-8", errors="replace", cwd=ROOT)
    for line in proc.stdout:
        print("   | " + line.rstrip("\n"))
    return proc.wait()


def main():
    args = parse_args()
    args.run_dir = args.run_dir or os.path.join(RESULTS_DIR, args.tag)
    run = Run(args.run_dir)
    if args.phase != "all":
        data = tokens_and_prompts(args, run)
        if args.phase == "llama":
            run_llama(args, run, data, lora=args.lora)
        else:
            {"hf": run_hf, "report": report}[args.phase](args, run, data)
        return 0
    run.capture_output()
    modes = args.modes.split(",")
    run.write_status(state="running", phase="tokenise", step=0, max_steps=len(modes) + 2)
    print(f"== parity_llama_hf.py {time.strftime('%Y-%m-%d %H:%M:%S')}  args: {vars(args)}")
    try:
        data = tokens_and_prompts(args, run)
        run.write_status(state="running", phase="llama.cpp", step=0)
        out = run_llama(args, run, data, lora=args.lora)
        run.metric(step=1, phase="llama", loss=out["mean_loss"])
        if args.llama_base_too:
            run.write_status(state="running", phase="llama.cpp without the LoRA", step=0)
            ob = run_llama(args, run, data, lora=None, name="llama_base")
            run.metric(step=1, phase="llama_base", loss=ob["mean_loss"])
        for i, m in enumerate(modes):
            run.write_status(state="running", phase=f"transformers {m}", step=i + 1, max_steps=len(modes) + 2)
            code = child(args, "hf", ["--gdn-heads", m])
            if code != 0:
                raise RuntimeError(f"transformers child ({m}) exited with {code}")
            with open(os.path.join(run.dir, f"hf_{m}.json"), encoding="utf-8") as fh:
                h = json.load(fh)
            run.metric(step=i + 2, phase=f"hf_{m}", loss=h["mean_loss"], kl=h["mean_kl"], top1=h["top1_agreement"])
        report(args, run, data)
        run.write_status(state="finished", step=len(modes) + 2)
        print("RESULT: OK")
        return 0
    except Exception as exc:
        import traceback
        traceback.print_exc()
        run.write_status(state="failed", error=f"{type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
