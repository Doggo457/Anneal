# Do training and testing compute the same model?

18 October 2026. Script: `tests/parity_llama_hf.py`, run through the Training page. Read-only: no weights were changed.

## Setup

- **Testing side**: llama.cpp b11435 (Vulkan) `llama-server` on Ollama's `qwen3.8:27b` (Q4_K_M GGUF, blob `sha256-f5f1dd89…`).
- **Training side**: transformers 5.17 + Unsloth 2026.10.6 on `unsloth/Qwen3.8-27B-unsloth-bnb-4bit` (NF4), run twice:
  - `--gdn-heads tile`: the `anneal/gdn_heads.py` patch, value head j → key head j % 16.
  - `--gdn-heads interleave`: transformers' own pairing, j → j // 3.
- **Same token IDs in both**: the text is tokenised once with the HF tokenizer and those IDs go to llama-server.
  - llama-server scores them with forced generation: the first token is the prompt, every later token is forced by a token-ID grammar (`<[id]>`), with `n_probs` 1000.
  - The reported probabilities come from the raw logits, before any sampler.
  - The forced tokens came back identical to the HF IDs.
  - Special tokens (`<|im_end|>` etc.) can't be forced by llama.cpp's grammar, so each is scored with a one-token request.
  - Cross-check on the first 6 positions, scored again as growing prompt prefixes: max |Δ logprob| 0.012 on the raw text.
  - llama.cpp's own tokenizer gives the same IDs anyway.
- **Metrics**:
  - KL is KL(llama.cpp ‖ transformers), computed over llama.cpp's top-1000 tokens plus one bucket for the rest of the vocabulary. The top-1000 hold 99.87% of the probability mass on average.
  - One model is on the GPU at a time.
- **Inherent gap**: the two sides use different 4-bit quantisations of the same weights (Q4_K_M vs NF4), so a small KL is expected even when the computation is identical.

## Result 1: mixed prose and code (`tests/parity_text.txt`, 1,023 scored tokens). Run: `results/parity-llama-hf-4`

| | mean loss | ppl | top-1 agreement with llama.cpp | mean KL | median KL | loss corr. with llama.cpp |
|---|---|---|---|---|---|---|
| llama.cpp (GGUF) | **0.519** | 1.68 | | | | |
| transformers, interleave (stock) | **0.515** | 1.67 | **95.0%** | **0.020** | 0.0008 | 0.98 |
| transformers, tile (gdn_heads patch) | 3.040 | 20.9 | 45.6% | 2.399 | 1.580 | 0.51 |

Loss by quarter of the text:
- llama.cpp: 1.16 / 0.08 / 0.45 / 0.40
- interleave: 1.17 / 0.10 / 0.42 / 0.38
- tile: 4.45 / 1.49 / 3.45 / 2.78

### Greedy continuations, 64 tokens

**Code** (`def merge_intervals(intervals):` plus docstring):
- llama.cpp: `if not intervals: return []` / `# Sort intervals by their start time` / `intervals.sort(key=lambda x: x[0])` / `merged = [intervals[0]]` / `for current in intervals[1:]: …`
- interleave: same start, then `# Sort by start time; if starts are equal…` / `intervals.sort(key=lambda x: (x[0], x[1]))` / `merged = [intervals[0]]` / `for start, end in …`. First divergence at token 13.
- tile: `if not intervals: return` / `intervals: return` / `if not intervals: return`, repeated. Diverges at token 8.

**Prose** ("The main difference between a process and a thread is"):
- llama.cpp: "that a process is an independent program running in its own memory space, while a thread is a lightweight execution unit within a process that shares the process's memory…"
- interleave: "that a process is an independent program with its own memory space, while a thread is a lightweight execution unit within a process that shares the same memory space…" First divergence at token 7.
- tile: "a process and a thread is a thread is a thread is a thread is …". Diverges at token 0.

**Chat** (a bash one-liner request):
- All three start `<think>` "The user wants a bash one-liner that counts…".
- llama.cpp and interleave both arrive at `find . -name "*.py" -exec wc -l {} +`. Interleave first diverges at token 27.
- tile repeats the request back. Diverges at token 12.

## Result 2: the earlier check_base_loss eval text (CodeAlpaca chats in Qwen3.8's chat template, first 2,047 tokens). Run: `results/parity-evaltext-2048-c`

| | mean loss | top-1 agreement | mean KL | median KL |
|---|---|---|---|---|
| llama.cpp | 6.680 | | | |
| interleave | 6.160 | 90.2% | 0.386 | 0.005 |
| tile | 1.308 | 42.4% | 3.912 | 2.300 |

These reproduce the earlier check_base_loss numbers exactly. Scoring the second half of the same 2,048 tokens:
- interleave: 6.80 (earlier chunk 0: 6.80)
- tile: 1.256 (earlier chunk 0: 1.256)

So the earlier measurements were right, but the conclusion drawn from them ("stock is broken, tile fixes it") was not.

**llama.cpp gives the same "bad" loss on this text as stock transformers.** Loss by chat role:

| tokens | n | llama.cpp | interleave | tile |
|---|---|---|---|---|
| assistant answers | 711 | **0.808** | **0.812** (KL 0.024) | 1.797 (KL 1.21) |
| system prompt | 684 | 15.17 | 13.55 | 0.39 |
| user turns | 375 | 6.32 | 6.27 | 2.72 |
| role headers (`system`, `user`, `assistant` after `<|im_start|>`) | 53 | 5.27 | 6.98 | 0.98 |

Why the system and user losses are so high:
- Qwen3.8's chat template adds the same system prompt to every row: "Reasoning effort is set to xhigh. Please think carefully…". That is a third of the tokens.
- The real model does not predict system or user text. Inside those turns it puts most of its probability on `<|im_end|>`:
  - 0.98 after " Please".
  - 1.0 in the middle of "[1, 2, 3, 4".
  - 0.84 instead of `assistant` after `<|im_start|>`.
- So it pays 6–15 nats on each such token. The mis-paired (tile) model is a worse, less "chat-shaped" model: it copies the repeated system prompt from context and scores it at 0.39 nats.
- On the assistant answers, the part an SFT loss is computed on, the real model is 0.81 and tile is 1.80.

## Conclusion

- **llama.cpp matches transformers' stock pairing (j → j // 3).** This is consistent with the code:
  - llama.cpp's converter (`conversion/qwen.py`, `_reorder_v_heads`) permutes every value-indexed tensor from grouped to tiled order.
  - The GatedDeltaNet kernels (CPU `ops.cpp` `iq1 = iv1 % neq1`; Vulkan `gated_delta_net.comp` `iq1 = head_id % neq1`), or `ggml_repeat_4d` when the fused op is off, broadcast q/k tiled.
  - Together these reproduce `repeat_interleave`. There was no contradiction to resolve: both engines compute the same model, and that model is the correct one.
- **The `anneal/gdn_heads.py` tile patch is wrong and should not be used.** Every 27B run so far used it (`--gdn-heads auto` tiles Qwen3.8): smoke-27b, the mem-27b-8k runs, baseloss-*-fixed. Their speed and memory numbers still stand, since the pairing doesn't change the work done, but their losses describe a broken model.
- I haven't changed any script defaults (this task was read-only). Correct results need `--gdn-heads interleave`. `auto` should stop tiling Qwen3.8.
- **For training and evaluation**:
  - Judge base loss on assistant tokens (or raw text), not on whole chat rows.
  - Train with assistant-only loss. Otherwise roughly half the gradient goes into teaching the model to predict the template's fixed system prompt and the user's words.

## Side findings

- **llama-perplexity is not comparable on chat text.** It tokenises special tokens as plain text, so `<|im_start|>` becomes `<`, `|`, `im`, … That is where the earlier "ppl ≈ 5.6" on the eval text came from.
- **llama-server context checkpoints are slow here.** Each one reads the recurrent state back from the GPU, and per-request prompt time fell from about 5 s to about 80 ms with `--ctx-checkpoints 0 --cache-ram 0`. Speculative decoding on this hybrid model may hit the same cost (task 3).
- **Small causality check.** On the chat text, scoring the first 1,024 tokens on their own instead of inside the 2,048-token sequence moves the mean loss only from 5.52 to 5.48 in stock transformers, but the largest single change is 11.4 nats (tile: 1.2).
  - I didn't save which positions moved. The likely cause is the many tokens with log-probs of −15 to −30 on this text, where small numeric differences show up large, but I haven't confirmed it.
  - llama.cpp's own two paths (prompt vs generation) also differ by 0.5 on the first 6 tokens of this text, against 0.012 on the raw text.
  - The comparison wasn't run on the raw text.
