"""Chat rows for training and evaluation, with assistant-only loss labels.

encode_chat() turns one conversation (a list of {"role", "content"} messages ending with an assistant turn) into
input_ids and labels, with labels = -100 over everything before the final assistant turn: the system prompt
Qwen3.8's template adds to every row ("Reasoning effort is set to xhigh..."), the user text, earlier turns and
the template tokens. The model is only scored on what it is asked to produce.

How the boundary is found (the same method as train_longrun.py):
  full   = apply_chat_template(messages, add_generation_prompt=False)       the whole conversation
  prompt = apply_chat_template(messages[:-1], add_generation_prompt=True)   what the model sees before answering
and the longest common token prefix of the two is masked. On Qwen3.8 the generation prompt ends in "<think>\\n"
while a finished assistant turn starts "<think>\\n\\n</think>\\n\\n", so the prompt is never an exact prefix there:
the first "\\n" vs "\\n\\n" token differs and the trained part starts at "\\n\\n</think>\\n\\n". Rows where the prompt
isn't an exact prefix are counted (MaskStats.prompt_not_prefix) so a template that diverges earlier shows up.

That first trained "\\n\\n" is a token the base model doesn't expect after "<think>" (it starts reasoning there):
16.8 nats on average on CodeAlpaca rows (results\\baseloss-27b-stock-assistant), so with data that has no
reasoning the LoRA is also taught to close the think block at once. Qwen3.8's template can render the
non-thinking form instead (template_kwargs={"enable_thinking": False}: no system prompt, and the generation
prompt is "<think>\\n\\n</think>\\n\\n", an exact prefix, so only the answer is trained).

Rows longer than max_len are dropped, never truncated: a truncated row loses the end of its answer, or all of it.
loss="full" gives labels = input_ids (every token scored), only for comparing with the old runs.
"""

IGNORE_INDEX = -100


def codealpaca_messages(row):
    """One CodeAlpaca-20k row as a user/assistant pair."""
    prompt = row["instruction"] + (("\n\n" + row["input"]) if row.get("input") else "")
    return [{"role": "user", "content": prompt}, {"role": "assistant", "content": row["output"]}]


def template_ids(tokenizer, messages, add_generation_prompt=False, **template_kwargs):
    """Token IDs of the chat template (transformers 5 returns a BatchEncoding here, Unsloth's patch a list).
    template_kwargs go to the template, e.g. enable_thinking=False."""
    out = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=add_generation_prompt,
                                        **template_kwargs)
    if hasattr(out, "keys"):
        out = out["input_ids"]
    return list(out)


class MaskStats:
    """Counts over a dataset build: rows seen, kept, dropped as too long, prompt-not-prefix, token totals."""

    def __init__(self):
        self.rows = self.kept = self.dropped_too_long = self.prompt_not_prefix = 0
        self.tokens_total = self.tokens_trained = 0

    def as_dict(self):
        d = dict(vars(self))
        d["trained_fraction"] = round(self.tokens_trained / self.tokens_total, 4) if self.tokens_total else None
        return d


def encode_chat(tokenizer, messages, max_len=None, loss="assistant", stats=None, template_kwargs=None):
    """-> {"input_ids", "labels", "n_trained", "prompt_is_prefix"}, or None if longer than max_len (dropped).
    template_kwargs (e.g. {"enable_thinking": False}) go to both renderings."""
    if loss not in ("assistant", "full"):
        raise ValueError(f"loss must be 'assistant' or 'full', not {loss!r}")
    if not messages or messages[-1]["role"] != "assistant":
        raise ValueError("the conversation must end with an assistant turn")
    stats = stats if stats is not None else MaskStats()
    stats.rows += 1
    kw = template_kwargs or {}
    full = template_ids(tokenizer, messages, add_generation_prompt=False, **kw)
    prompt = template_ids(tokenizer, messages[:-1], add_generation_prompt=True, **kw)
    n = 0
    for a, b in zip(full, prompt):
        if a != b:
            break
        n += 1
    exact = n == len(prompt)
    if not exact:
        stats.prompt_not_prefix += 1
    if max_len is not None and len(full) > max_len:
        stats.dropped_too_long += 1
        return None
    labels = list(full) if loss == "full" else [IGNORE_INDEX] * n + full[n:]
    # The model shifts labels (position i predicts token i + 1), so position 0 is never predicted.
    trained = sum(1 for x in labels[1:] if x != IGNORE_INDEX)
    stats.kept += 1
    stats.tokens_total += len(full)
    stats.tokens_trained += trained
    return {"input_ids": full, "labels": labels, "n_trained": trained, "prompt_is_prefix": exact}


def sample_report(tokenizer, row, context_chars=120):
    """What one encoded row trains on, to prove the mask: token counts, the decoded unmasked text, the masked text
    just before it, and the first trained tokens one by one."""
    ids, labels = row["input_ids"], row["labels"]
    trained_pos = [i for i in range(1, len(labels)) if labels[i] != IGNORE_INDEX]
    first = trained_pos[0] if trained_pos else len(ids)
    contiguous = trained_pos == list(range(first, first + len(trained_pos)))
    return {
        "tokens_total": len(ids),
        "tokens_trained": len(trained_pos),
        "tokens_masked": len(ids) - len(trained_pos),
        "first_trained_position": first,
        "trained_span_contiguous": contiguous,
        "masked_text_tail": tokenizer.decode(ids[:first])[-context_chars:],
        "first_trained_tokens": [tokenizer.decode([t]) for t in ids[first:first + 8]],
        "unmasked_text": tokenizer.decode([ids[i] for i in trained_pos]),
    }


def print_sample(tokenizer, row, label="sample row"):
    """Log sample_report() readably; returns the report for summary.json."""
    rep = sample_report(tokenizer, row)
    print(f"== {label}: {rep['tokens_trained']} trained / {rep['tokens_total']} total tokens "
          f"({rep['tokens_masked']} masked), trained span starts at position {rep['first_trained_position']}, "
          f"contiguous: {rep['trained_span_contiguous']}")
    print(f"   masked text just before it: ...{rep['masked_text_tail']!r}")
    print(f"   first trained tokens: {rep['first_trained_tokens']}")
    print("   unmasked (trained) text:")
    print("   >>>" + rep["unmasked_text"].replace("\n", "\n   >>>"))
    return rep


def pack_pairs(tokenizer, pairs, max_len, n_rows):
    """Greedily join user/assistant pairs into multi-turn conversations whose full rendering is at most max_len
    tokens (measured exactly, system prompt included), until n_rows are full. A pair that alone is longer than
    max_len is skipped. Returns a list of message lists."""
    rows, cur, cur_len = [], [], 0
    for pair in pairs:
        cand = cur + pair
        n = len(template_ids(tokenizer, cand))
        if n <= max_len:
            cur, cur_len = cand, n
            continue
        if cur:
            rows.append(cur)
            if len(rows) >= n_rows:
                return rows
        single = len(template_ids(tokenizer, pair))
        cur, cur_len = (pair, single) if single <= max_len else ([], 0)
    if cur and len(rows) < n_rows:
        rows.append(cur)
    return rows
