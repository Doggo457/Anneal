# Stock head pairing, assistant-only loss, the base re-score and the 27B smoke test

18–19 October 2026. These checks follow on from `parity_findings.md`. Testing the trained adapter in llama.cpp is in `lora_testing_findings.md`.

## 1. Stock head pairing is the default

- `anneal/gdn_heads.py`:
  - `--gdn-heads auto` (and the old name `interleave`) now mean **stock**: transformers' own pairing, value head j → key head j // 3, no patch, for every model.
  - The tile patch (j → j % 16) is kept only as the explicit `--gdn-heads tile`, for reproducing runs made before 18 Oct 2026.
- The default is `stock` in `scripts/check_base_loss.py`, `scripts/measure_memory.py`, `tests/smoke_small.py`, `tests/parity_llama_hf.py` (also `--modes stock`) and `tests/adapter_generate.py`.
- Every `summary.json` records `gdn_heads`: the mode asked for, the mode used, whether the patch was installed, and the pairing in words.

## 2. Assistant-only loss (`anneal/data.py`)

**Method**, `encode_chat()`:
- Render the full conversation with `apply_chat_template(add_generation_prompt=False)`.
- Render `messages[:-1]` with `add_generation_prompt=True`.
- Set labels to -100 over the longest common token prefix, and count rows where the prompt isn't an exact prefix.
- Drop rows over the length limit; nothing is truncated.
- `--loss full` (whole-row loss) is still available, but only when asked for.

**Proof the trainer sees it**:
- `smoke_small.py` and `measure_memory.py` collate one real row through the trainer's own collator (TRL `DataCollatorForLanguageModeling`). They check its labels equal the helper's, and abort if not. The check passed in every run.
- Unsloth logs `num_items_in_batch` = the trained-token count.
- `anneal/runtime.CountingCollator` records exact tokens and trained tokens per step in `metrics.jsonl`.

**What Qwen3.8's template does to the mask**:
- The generation prompt ends `<|im_start|>assistant\n<think>\n`, but a finished turn is `<think>\n\n</think>\n\n<answer>`.
- The first differing token is `\n` vs `\n\n`, so the prompt is never an exact prefix: 61 of 61 eval rows are flagged.
- So the trained span is `\n\n</think>\n\n` + answer + `<|im_end|>` + `\n`.
- On Qwen3.5-0.8B the prompt is an exact prefix (0 rows flagged).

**On packed multi-turn chats only a sliver is trained**, because only the final turn counts:
- smoke data: 12–132 tokens per ~2.8K-token row, 2.1% of tokens.
- measure_memory's 8K rows: 0.96%.

## 3. Base model, stock pairing, assistant tokens (`baseloss-27b-stock-assistant`)

The same 61 held-out rows as every earlier run of check_base_loss (CodeAlpaca 10000–10060; the joined text is byte-identical to the old `eval_text.txt`). Each row is scored on its own with the training mask.

| measure | tokens | loss | ppl |
|---|---|---|---|
| trained tokens, token-weighted mean | 3,654 | **1.042** | **2.834** |
| mean of the 61 per-row losses | | 1.366 | (mean of per-row ppl 7.13) |
| answer text only (after `</think>`, before `<|im_end|>`) | 3,410 | 0.809 | 2.246 |

The answer-only figure matches `parity_findings.md` (llama.cpp 0.808, stock transformers 0.812). Per-row values are in `baseloss-27b-stock-assistant/summary.json` (`rows`).

**The first trained token costs 16.8 nats.**
- That is the `\n\n` right after `<think>`: mean 16.77, max 20.5. The base model starts reasoning there rather than closing an empty think block.
- It is 1,023 of the 3,806 nats. The other template tokens are cheap: `</think>` 0.002, the `\n\n` after it 0.0001, `<|im_end|>` 0.40, the final `\n` 0.0.

**Non-thinking form** (`--no-thinking`, i.e. `enable_thinking=False`; `baseloss-27b-stock-assistant-nothink`):
- No system prompt, and the generation prompt is already `<think>\n\n</think>\n\n`. It is an exact prefix (0 of 61 rows flagged), so only answer + `<|im_end|>\n` are trained.
- Loss **0.806** (ppl 2.24) on 3,471 tokens. Answer text only 0.829, against 0.824 for the same tokens in the thinking render.

## 4. 27B smoke test, stock pairing (`smoke-27b-stock`)

Same settings and data as `smoke-27b`:
- 30 steps, the same 7 packed chats (2675 / 2822 / 2938 tokens).
- LoRA r16 on MLP + full-attention layers.
- Unsloth gradient checkpointing, bf16, adamw_8bit, lr 2e-4.

Now with stock pairing, shift conv and assistant-only loss. Run through the Training page with `--save-adapter`.

| | smoke-27b (tile, full-row loss, F.conv1d) | smoke-27b-stock |
|---|---|---|
| loss first / last / lowest | 1.604 / 0.673 / 0.512 | 0.892 / 0.0016 / 0.00043 (step 26) |
| NaN | no | no |
| s/step (steps 3–30) | 33.46 | **27.43** |
| tokens/s, all tokens | 84.4 | **103.0** (exact per-step counts) |
| tokens/s, trained tokens | (all) | 2.1 |
| first step | 182.2 s | 76.3 s (Triton autotune cache warm) |
| peak reserved / allocated (PyTorch) | 25.14 / 22.67 GiB | 24.55 / 22.66 GiB |
| peak GPU seen by Windows (training process) | | 30.47 GiB (26.73 dedicated + 3.75 shared) |
| peak system RAM used / process RSS | | 13.2 / 5.84 GiB |

- The near-zero loss is memorisation, not quality: 410 trained tokens in total, each row seen 4–5 times.
- The two loss columns are not comparable: different pairing, different loss.

**Base + adapter from disk** (`tests/adapter_generate.py`, 64 greedy tokens, 2 coding prompts): no loops in either mode (no repeated 8-grams, 96–100% distinct 4-grams).

- **Default prompt** (`smoke-27b-stock-generate`): the model still reasons first ("We need answer user: …").
  - The palindrome text first differs from the base at token 31. The SQL text is identical to the base over all 64 tokens.
  - Reason: the generation prompt already contains `<think>\n`, so the `\n\n</think>` the adapter learned to emit can never come up. The context the trained answers follow never appears at inference.
- **`--no-thinking` prompt** (`smoke-27b-stock-generate-nothink`): code straight away.
  - Palindrome: a typed function with a docstring, identical to the base.
  - SQL: a more spread-out layout; it first differs from the base at token 4.

## Recommendations for real training

1. **Render answers-only data in the non-thinking form.** For data without reasoning traces (CodeAlpaca and most code-instruction sets), use `enable_thinking=False` (`template_kwargs` in `anneal/data.py`) for training, evaluation and serving.
   - Otherwise the first trained token is a 17-nat "skip thinking" token.
   - And the learned answers sit behind a context that the default generation prompt never produces.
   - Data with reasoning traces in the think block avoids the mismatch too.
2. **Use one conversation per row, or mask every assistant turn**, rather than final-turn-only masking of packed multi-turn chats. Final-turn-only trains on 1–2% of the tokens processed.
