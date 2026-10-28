"""CPU checks of the own loss and the KL-to-base term (anneal/kl_loss.py, scripts/train_longrun.py kl_think_beta)
with a tiny random Qwen3 model and a byte-level tokenizer built here (no GPU, no download):

  1. think_targets marks the trained reasoning and its close tag, nothing else.
  2. With a fresh adapter (B = 0) the KL term is exactly 0 and the cross entropy equals the model's own loss.
  3. With a trained-looking adapter, the chunked loss and its gradients equal a direct full-logits computation.
  4. disable_adapter gives the base model's hidden states.
  5. train_longrun.py (engine hf) for 3 steps: the own loss at beta 0 logs the same losses and gradient norms as the
     model's own loss; at beta 1 step 1 is identical (KL 0 at B = 0) and later steps log a positive KL.

  python tests/test_kl_loss.py
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import torch  # noqa: E402

from anneal import kl_loss  # noqa: E402

TEMPLATE = (
    "{%- for m in messages -%}"
    "{{- '<|im_start|>' + m['role'] + '\\n' -}}"
    "{%- if m['role'] == 'assistant' -%}"
    "{%- if enable_thinking is defined and enable_thinking is false -%}"
    "{{- '<think>\\n\\n</think>\\n\\n' + m['content'] -}}"
    "{%- else -%}"
    "{{- '<think>\\n' + (m['reasoning_content'] if m['reasoning_content'] is defined else '') + '\\n</think>\\n\\n'"
    " + m['content'] -}}"
    "{%- endif -%}"
    "{%- else -%}{{- m['content'] -}}{%- endif -%}"
    "{{- '<|im_end|>\\n' -}}"
    "{%- endfor -%}"
    "{%- if add_generation_prompt -%}"
    "{{- '<|im_start|>assistant\\n' -}}"
    "{%- if enable_thinking is defined and enable_thinking is false -%}{{- '<think>\\n\\n</think>\\n\\n' -}}"
    "{%- else -%}{{- '<think>\\n' -}}{%- endif -%}"
    "{%- endif -%}")
LORA = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def tiny_tokenizer():
    from tokenizers import AddedToken, Tokenizer, decoders, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast
    vocab = {c: i for i, c in enumerate(sorted(pre_tokenizers.ByteLevel.alphabet()))}
    tk = Tokenizer(models.BPE(vocab=vocab, merges=[]))
    tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)
    tk.decoder = decoders.ByteLevel()
    tk.add_special_tokens([AddedToken(t, special=True, normalized=False)
                           for t in ("<|im_start|>", "<|im_end|>", "<|endoftext|>")])
    tk.add_tokens([AddedToken(t, special=False, normalized=False) for t in ("<think>", "</think>")])
    tok = PreTrainedTokenizerFast(tokenizer_object=tk, eos_token="<|im_end|>", pad_token="<|endoftext|>")
    tok.chat_template = TEMPLATE
    return tok


def tiny_model(tok, seed=0):
    from transformers import Qwen3Config, Qwen3ForCausalLM
    torch.manual_seed(seed)
    cfg = Qwen3Config(vocab_size=len(tok), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                      num_attention_heads=2, num_key_value_heads=1, head_dim=16, max_position_embeddings=1024,
                      tie_word_embeddings=False, pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
    return Qwen3ForCausalLM(cfg)


def with_lora(model, r=4):
    from peft import LoraConfig, get_peft_model
    return get_peft_model(model, LoraConfig(task_type="CAUSAL_LM", r=r, lora_alpha=2 * r, lora_dropout=0.0,
                                            target_modules=LORA))


def rows(n=24, seed=1):
    import random
    rnd = random.Random(seed)
    words = ["alpha", "beta", "gamma", "delta", "loop", "index", "fix", "test", "array", "sum"]
    out = []
    for i in range(n):
        trace = " ".join(rnd.choice(words) for _ in range(rnd.randint(3, 30)))
        answer = " ".join(rnd.choice(words) for _ in range(rnd.randint(2, 10)))
        out.append({"messages": [{"role": "user", "content": f"task {i}: " + " ".join(rnd.choice(words) for _ in range(5))},
                                 {"role": "assistant", "content": f"<think>\n{trace}\n</think>\n\n{answer}"}]})
    return out


def encoded(tok, row):
    from train_longrun import tokenize_rows
    out, too_long, _ = tokenize_rows([row], tok, {"enable_thinking": True}, 4096, True)
    assert not too_long
    return out[0]


def reference_loss(peft, ids, labels, beta, close_id):
    """Full logits, no chunks: sum CE over trained targets + beta * sum KL(base || tuned) over reasoning targets."""
    inner = peft.get_base_model()
    logits = inner(input_ids=ids, use_cache=False).logits[:, :-1].float()
    y = labels[:, 1:]
    trained = y != kl_loss.IGNORE_INDEX
    ce = torch.nn.functional.cross_entropy(logits[trained], y[trained], reduction="sum")
    think = kl_loss.think_targets(ids, labels, close_id)
    with torch.no_grad(), peft.disable_adapter():
        base = inner(input_ids=ids, use_cache=False).logits[:, :-1].float()
    lq = torch.log_softmax(base[think], -1)
    lp = torch.log_softmax(logits[think], -1)
    kl = (lq.exp() * (lq - lp)).sum()
    return ce + beta * kl, ce, kl


def test_think_targets():
    close = 9
    ids = torch.tensor([[1, 2, 3, 4, 5, 9, 6, 7, 9, 8]])
    labels = torch.tensor([[-100, -100, -100, 4, 5, 9, 6, 7, 9, 8]])
    m = kl_loss.think_targets(ids, labels, close)
    # targets are labels[1:]: positions 3..5 of the row (the reasoning 4, 5 and the first close tag) -> 2..4
    assert m.tolist() == [[False, False, True, True, True, False, False, False, False]], m.tolist()
    # a close tag in the prompt doesn't count; no trained close tag -> nothing
    assert not kl_loss.think_targets(torch.tensor([[9, 1, 2]]), torch.tensor([[-100, 1, 2]]), close).any()
    assert not kl_loss.think_targets(ids, torch.full_like(labels, -100), close).any()
    tok = tiny_tokenizer()
    r = encoded(tok, rows(1)[0])
    ids, labels = torch.tensor([r["input_ids"]]), torch.tensor([r["labels"]])
    close_id = tok.convert_tokens_to_ids("</think>")
    m = kl_loss.think_targets(ids, labels, close_id)[0]
    pos = m.nonzero().flatten().tolist()
    first = next(i for i, x in enumerate(r["labels"]) if x != -100)
    assert pos == list(range(first - 1, pos[-1] + 1)) and r["input_ids"][pos[-1] + 1] == close_id
    reasoning = tok.decode(r["input_ids"][first:pos[-1] + 1])
    trace = rows(1)[0]["messages"][1]["content"].split("\n")[1]
    assert reasoning == trace + "\n", repr(reasoning)


def test_loss_matches_reference():
    tok = tiny_tokenizer()
    close_id = tok.convert_tokens_to_ids("</think>")
    peft = with_lora(tiny_model(tok))
    r = encoded(tok, rows(3)[2])
    ids, labels = torch.tensor([r["input_ids"]]), torch.tensor([r["labels"]])
    inputs = {"input_ids": ids, "labels": labels, "attention_mask": torch.ones_like(ids)}
    # fresh adapter: tuned == base, so KL is 0; CE = the model's own (mean) loss x trained tokens
    loss, st = kl_loss.own_loss(peft, inputs, 1.0, close_id, 1 << 20, lambda n: n)
    own = peft(input_ids=ids, labels=labels).loss
    assert st["kl_sum"] == 0.0 and st["think"] > 0, st
    assert abs(float(loss) - float(own)) < 1e-5, (float(loss), float(own))
    # a trained-looking adapter: chunked loss and gradients = the full-logits reference
    torch.manual_seed(3)
    with torch.no_grad():
        for n, p in peft.named_parameters():
            if "lora_B" in n:
                p.normal_(0, 0.5)
    params = [p for p in peft.parameters() if p.requires_grad]
    for budget in (1, 1 << 30):  # 16-position chunks, then one chunk
        loss, st = kl_loss.own_loss(peft, inputs, 0.7, close_id, budget, lambda n: 5.0)
        g1 = torch.autograd.grad(loss, params)
        ref, ce, kl = reference_loss(peft, ids, labels, 0.7, close_id)
        g2 = torch.autograd.grad(ref / 5.0, params)
        assert st["kl_sum"] > 1e-3 and abs(st["kl_sum"] - float(kl)) < 1e-4 * max(1, float(kl)), (st, float(kl))
        assert abs(float(loss) - float(ref) / 5.0) < 1e-5 * max(1, float(ref)), (float(loss), float(ref))
        for a, b in zip(g1, g2):
            assert torch.allclose(a, b, rtol=1e-4, atol=1e-6), float((a - b).abs().max())


def test_disable_adapter_is_base():
    tok = tiny_tokenizer()
    plain = tiny_model(tok, seed=5)
    peft = with_lora(tiny_model(tok, seed=5))
    with torch.no_grad():
        for n, p in peft.named_parameters():
            if "lora_B" in n:
                p.normal_(0, 0.5)
    ids = torch.tensor([encoded(tok, rows(2)[1])["input_ids"]])
    with torch.no_grad():
        h0 = plain.model(input_ids=ids, use_cache=False).last_hidden_state
        on = peft.get_base_model().model(input_ids=ids, use_cache=False).last_hidden_state
        with peft.disable_adapter():
            off = peft.get_base_model().model(input_ids=ids, use_cache=False).last_hidden_state
    assert torch.equal(off, h0) and not torch.allclose(on, h0)


def run_train(tmp, name, model_dir, train_file, **over):
    import yaml
    cfg = {"run_name": name, "base_model": model_dir, "engine": "hf", "load_in_4bit": False, "bf16": False,
           "seq_len": 1024, "lora_r": 4, "lora_alpha": 8, "lora_dropout": 0.0, "target_modules": LORA,
           "train_file": train_file, "batch_size": 1, "grad_accum": 4, "lr": 5e-2, "max_steps": 3,
           "warmup_steps": 0, "schedule": "wsd", "decay_fraction": 0.34, "completion_only": True,
           "chat_template_kwargs": {"enable_thinking": True}, "save_steps": 100, "eval_steps": 0, "seed": 42,
           "optim": "adamw_torch", "gradient_checkpointing": False, "logging_steps": 1}
    cfg.update(over)
    path = os.path.join(tmp, f"{name}.yaml")
    yaml.safe_dump(cfg, open(path, "w", encoding="utf-8"))
    run_dir = os.path.join(tmp, name)
    # the run contract's callback times steps with torch.cuda.synchronize, which needs a GPU: no-ops here. TRL 1.15's
    # fused LM head is a Triton kernel that needs a GPU too, so the stock runs use transformers' Trainer.compute_loss:
    # the model's own loss, summed over trained tokens and divided by the accumulated batch's count
    boot = ("import runpy, sys, torch, transformers, trl.trainer.sft_trainer as st\n"
            "for f in ('synchronize', 'max_memory_reserved', 'max_memory_allocated'):\n"
            "    setattr(torch.cuda, f, lambda *a, **k: 0)\n"
            "st.add_fused_lm_head = lambda *a, **k: None\n"
            "st.SFTTrainer.compute_loss = transformers.Trainer.compute_loss\n"
            f"sys.argv = ['train_longrun.py', '--config', {path!r}, '--run-dir', {run_dir!r}]\n"
            f"runpy.run_path({os.path.join(ROOT, 'scripts', 'train_longrun.py')!r}, run_name='__main__')\n")
    p = subprocess.run([sys.executable, "-c", boot], capture_output=True, text=True, cwd=ROOT,
                       env=dict(os.environ, CUDA_VISIBLE_DEVICES=""))
    if p.returncode != 0:
        raise AssertionError(f"{name}: exit {p.returncode}\n{p.stdout[-3000:]}\n{p.stderr[-3000:]}")
    metrics = [json.loads(x) for x in open(os.path.join(run_dir, "metrics.jsonl"), encoding="utf-8")]
    steps = {m["step"]: m for m in metrics if "loss" in m and "step" in m and "event" not in m}
    own = {m["step"]: m for m in metrics if m.get("event") == "own_loss"}
    summary = json.load(open(os.path.join(run_dir, "summary.json"), encoding="utf-8"))
    return steps, own, summary


def test_train_longrun_end_to_end():
    tmp = tempfile.mkdtemp(prefix="anneal-kl-test-")
    try:
        tok = tiny_tokenizer()
        model_dir = os.path.join(tmp, "model")
        tiny_model(tok).save_pretrained(model_dir)
        tok.save_pretrained(model_dir)
        train_file = os.path.join(tmp, "train.jsonl")
        with open(train_file, "w", encoding="utf-8") as fh:
            for r in rows(24):
                fh.write(json.dumps(r) + "\n")
        stock, no_own, s0 = run_train(tmp, "stock", model_dir, train_file)
        own0, ev0, s1 = run_train(tmp, "own0", model_dir, train_file, own_loss=True)
        kl1, ev1, s2 = run_train(tmp, "kl1", model_dir, train_file, kl_think_beta=1.0, gradient_checkpointing=True)
        assert sorted(stock) == sorted(own0) == sorted(kl1) == [1, 2, 3], (stock, own0, kl1)
        assert not no_own and s0["own_loss"] is None
        for k in (1, 2, 3):
            a, b = stock[k], own0[k]
            assert abs(a["loss"] - b["loss"]) <= 1e-4 * max(1, abs(a["loss"])), (k, a, b)
            assert abs(a["grad_norm"] - b["grad_norm"]) <= 1e-3 * max(1, a["grad_norm"]), (k, a, b)
        assert abs(kl1[1]["loss"] - stock[1]["loss"]) <= 1e-4 * max(1, stock[1]["loss"]), (kl1[1], stock[1])
        assert abs(kl1[1]["grad_norm"] - stock[1]["grad_norm"]) <= 1e-3 * max(1, stock[1]["grad_norm"])
        assert ev1[1]["kl_per_think_token"] == 0.0 and ev1[2]["kl_per_think_token"] > 0 and ev1[3]["kl_per_think_token"] > 0, ev1
        assert ev0[2]["kl_per_think_token"] is None and ev0[1]["think_tokens"] == 0
        info = s2["own_loss"]
        assert info["rows_with_think"] == info["rows"] == 24 and info["kl_think_beta"] == 1.0, info
        assert s1["own_loss"]["kl_think_beta"] == 0.0
        print("end to end:", {k: (stock[k]["loss"], own0[k]["loss"], kl1[k]["loss"], ev1[k]["kl_per_think_token"])
                              for k in (1, 2, 3)})
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS {name}")
        except Exception:
            import traceback
            failed += 1
            print(f"FAIL {name}")
            traceback.print_exc()
    print(f"{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
