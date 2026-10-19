# Testing the trained model in llama.cpp (GGUF LoRA)

19 October 2026. Adapter: `results/smoke-27b-stock/adapter` (the 30-step 27B smoke test: stock head pairing, shift conv, assistant-only loss; LoRA r16 / alpha 16 on the MLP of all 64 layers and q/k/v/o of the 16 full-attention layers, 79,691,776 parameters).

Scripts:
- `tests/lora_server_check.py` (`results/lora-server-check`)
- `tests/parity_llama_hf.py --adapter --lora --llama-base-too` (`results/parity-lora-smoke-27b-stock-b`)
- `tests/lora_serving_bench.py` (`results/lora-serving-bench`)

All three were run through the Training page.

This adapter was trained on only 410 tokens: the final answers of 7 packed chats, each seen 4–5 times, ending at loss 0.0016. So it is a plumbing test, not a useful model. The point here is that the testing path computes the same trained model as the training path, and how fast it serves.

## Conversion

- **Tool**: llama.cpp b11435's own `convert_lora_to_gguf.py`. It comes from the source tag zip (`llama.cpp/_dl/llama.cpp-b11435-src.zip`, sha256 `e0eea087…0574`), extracted to `llama.cpp/b11435-src`.
- **The bnb config is refused**: `--base` pointed at the bnb-4bit snapshot stops with "Quant method is not yet supported: 'bitsandbytes'". This happens even though a LoRA conversion reads no base weights.
- **Fix**: `--base` is `models/lora-base-config/Qwen3.8-27B-unsloth-bnb-4bit-8aa5f05d`. That is the trained-on snapshot's `config.json` with only `quantization_config` removed; every architecture field is unchanged.
- **Command**:

      convert_lora_to_gguf.py --base models\lora-base-config\Qwen3.8-27B-unsloth-bnb-4bit-8aa5f05d --outtype f16
          --outfile results\smoke-27b-stock\adapter-gguf\smoke-27b-stock-F16-LoRA.gguf results\smoke-27b-stock\adapter

- **Output**:
  - 512 tensors: 64 layers × 3 MLP + 16 layers × 4 attention, as A/B pairs.
  - 159.4 MB, `general.architecture = qwen35`, `adapter.lora.alpha = 16`.
  - `attn_q` keeps its query+gate shape (lora_b 16 × 12288).
  - Log: `logs/convert_lora_smoke-27b-stock.log`.
- **f32 copy**: also written (`…-F32-LoRA.gguf`, 318.8 MB), for the crash check below.
- **DeltaNet heads**: no linear-attention (DeltaNet) tensor is targeted, so the converter's DeltaNet V-head reorder never applies.

## llama-server with a LoRA dies on prompt batches over 8 tokens (Vulkan)

The first parity run (`results/parity-lora-smoke-27b-stock`) scored all 2,047 positions with the LoRA loaded. Every one of those requests processed a 1-token prompt, because the rest was cached. The server then died on the first uncached 25-token prompt, leaving no log line. `tests/lora_server_check.py` starts a fresh server per variant and sends uncached prompts of 1, 2, 8, 32 and 128 tokens.

| variant | longest prompt that worked | server exit | 128-token prompt time |
|---|---|---|---|
| no LoRA (control) | 128 | (stopped) | 428 ms |
| `--lora` F16 | 8 | 0xC0000409 at 32 | |
| `--lora` F32 | 8 | 0xC0000409 at 32 | |
| F16, `GGML_VK_DISABLE_FUSION=1` | 8 | 0xC0000409 at 32 | |
| F16, `-lv 4` | 8 | 0xC0000409 at 32, no message | |
| F16, `-ub 8` | 128 | (stopped) | 3,350 ms |
| **F16, `GGML_VK_DISABLE_COOPMAT=1`** | **128** | (stopped) | **973 ms** |
| F16, `-fa off` | 0 | 0xC0000409 during warm-up | |
| F16, `-ub 1 -b 1` | 0 | warm-up error (`get_sub_batch` can't fit 2 tokens in a batch of 1; a separate config problem) | |

What the table shows:
- **Where it breaks**: the crash starts above 8 tokens per batch. That is where ggml-vulkan switches from matrix-vector to matrix-matrix kernels (decode and short prompts are unaffected).
- **What fixes it**: turning off the cooperative-matrix kernels. Neither the adapter's precision (F16 or F32) nor op fusion matters.
- **The adapter is loaded and applied**: `/lora-adapters` lists it at scale 1.0, the verbose log shows it loading, and output at 32 tokens differs from the no-LoRA control.
- **Not identified**: which kernel aborts. 0xC0000409 is a fail-fast, and nothing is printed even at `-lv 4`.

**Workaround used below: `GGML_VK_DISABLE_COOPMAT=1` for llama-server whenever a LoRA is loaded.** `parity_llama_hf.py --vk-coopmat auto` and `lora_serving_bench.py --vk-coopmat off` set it by default. It is also set for the no-LoRA reference servers, so the comparisons differ only in the adapter.

Cost of the workaround:
- Generation: none. No-LoRA decode is 12.67 tok/s with it, against 12.69 tok/s without it in `mtp_findings.md`.
- Prompt processing: mixed. These figures compare the no-LoRA reference here with `mtp-spec-bench`, on the same prompts.
  - Prompts of 10–29 tokens run 17–24% faster with cooperative matrices off.
  - The 45- and 57-token prompts run 28–31% slower: 127 vs 176 and 146 vs 210 tok/s.
  - In `lora-server-check`, a 128-token prompt took 973 ms with the LoRA and cooperative matrices off, against 428 ms with neither.

## a. Do llama.cpp and transformers compute the same trained model?

Text and pairing:
- Same 2,047 token IDs as `parity-evaltext-2048-c`: the first 2,048 tokens of the held-out CodeAlpaca chats (rows 10000+) in Qwen3.8's chat template.
- These rows were not trained on; the smoke test used rows 0–199.
- Stock head pairing, no patch.

The two sides:
- **transformers**: Unsloth bnb-4bit (NF4) plus the PEFT adapter.
- **llama.cpp**: Ollama's Q4_K_M plus the F16 GGUF LoRA.

"Assistant (training mask)" means the tokens the training loss scores in each assistant turn: after `<think>` up to and including `<|im_end|>` and its newline (779 tokens).

| tokens | n | llama.cpp + LoRA | transformers + adapter | mean KL | top-1 agreement |
|---|---|---|---|---|---|
| assistant (training mask) | 779 | **0.728** | **0.756** | **0.038** | **96.0%** |
| assistant answers (after `</think>`) | 711 | 0.747 | 0.779 | 0.025 | |
| all tokens | 2,047 | 0.902 | 0.931 | 0.156 (median 0.0019) | 93.7% |

Without any adapter, on the same IDs (`parity-evaltext-2048-c`, transformers stock vs llama.cpp):
- Assistant (training mask): KL 0.036, top-1 95.5%.
- Assistant answers: KL 0.025.

**With the adapter the two engines agree exactly as closely as without it.** That is KL 0.038 vs 0.036 on assistant tokens, 0.025 vs 0.025 on answers. So the GGUF LoRA is applied the same way as the PEFT adapter. The remaining gap is the one between the two 4-bit quantisations (Q4_K_M vs NF4).

Other checks:
- The forced tokens came back identical to the HF IDs, and the llama.cpp tokenizer gives the same IDs.
- Greedy 64-token continuations, transformers + adapter vs llama.cpp + LoRA:
  - Prose: identical over all 64 tokens.
  - Code: first divergence at token 13. Chat: 27. Chat-template prompt: 59.
  - The base parity run had similar divergence points: 13 / 7 / 27.

### Does the adapter change llama.cpp's output?

Yes. The same server settings were run with and without `--lora`.

- **Assistant (training mask) loss**: 0.826 → 0.728. Answers alone: 0.810 → 0.747.
- **Positions moved**: 1,463 of 2,047 positions change by more than 0.01 nats, and top-1 is the same as the base at only 56.0% of positions.
- **Greedy 64 tokens**:
  - Prose diverges at token 7: base "running in its own memory space", LoRA "with its own memory space".
  - The chat-template prompt diverges at token 20.
  - The raw code and chat prompts are unchanged over 64 tokens.
- **Speed run**: the 6 prompts diverge from no-LoRA at tokens 7–167 (below).
- **Untrained tokens move most**:
  - System prompt 15.20 → 0.46, user turns 6.30 → 2.30, role headers 4.97 → 1.26.
  - The base model predicts `<|im_end|>` inside system/user text instead of the text itself (`parity_findings.md`). The overfit adapter stops doing that, and copies the repeated system prompt from context instead.
  - This is a behaviour change, not a better model, and the reason whole-text loss must not be used to judge an adapter.

## b. Generation speed with the LoRA loaded (-np 1)

Same 6 prompts and settings as `mtp_findings.md`: greedy, 256 tokens each, after a warm-up. Speed is the server's own timings. Cooperative matrices are off in every row.

| configuration | gen tok/s | vs LoRA, no MTP | draft acceptance | wall, 6 × 256 tokens |
|---|---|---|---|---|
| no LoRA (reference) | 12.67 | 1.036x | | 123.2 s |
| `--lora` | **12.23** | 1.00x | | 127.7 s |
| `--lora --spec-type draft-mtp` (built-in MTP block, no `-md`) | **22.64** | 1.85x | 68.4% (1,035 / 1,513) | 70.0 s |
| `--lora --spec-type draft-mtp -md <Unsloth MTP GGUF> --spec-draft-n-max 4` | **24.25** | 1.98x | 59.8% (1,084 / 1,814) | 65.5 s |

- **`--lora` and `--spec-type draft-mtp` work together**, both with the built-in MTP block and with `-md`.
- **The LoRA stays applied under MTP**:
  - MTP output equals the LoRA's plain greedy output over all 256 tokens on 5 of 6 prompts. Explain-chat drifts at token 192, the same kind of near-tie drift as `mtp_findings.md`.
  - All configurations with the LoRA diverge from the no-LoRA output at the same tokens (7–167).
- **The LoRA costs 3.5% of decode speed** (12.67 → 12.23 tok/s). MTP speedups are as without a LoRA:
  - Built-in block: 1.85x here vs 1.87x before.
  - `-md` with draft 4: 1.98x vs 2.00x.
  - Acceptance is unchanged: 68.4% vs 68.6%, and 59.8% vs 59.7%.
- **Notes**:
  - The MTP draft block (`blk.64`) has no LoRA tensors, so the draft runs the base model's MTP head. Its acceptance is unchanged on this lightly trained adapter.
  - An adapter that changes the model more could lower acceptance. That is not measured here.

## c. Batch throughput for benchmarking

Setup:
- 12 coding problems, each a user turn in the model's chat template with the generation prompt.
- Greedy, up to 256 tokens each; about 2,730 tokens are generated in total, since some answers end earlier.
- `-c 16384`, LoRA loaded, cooperative matrices off.
- "Batch wall" runs from the first request to the last answer. Aggregate = generated tokens / batch wall.

| configuration | batch wall | aggregate tok/s | per-slot tok/s | vs sequential | draft acceptance |
|---|---|---|---|---|---|
| `-np 1`, one request at a time | 232.2 s | 11.78 | 12.15 | 1.00x | |
| `-np 1` + draft-mtp, one at a time | 134.0 s | 20.44 | 22.16 | 1.73x | 63.9% |
| `-np 4`, 4 requests in flight | 110.4 s | 24.72 | 7.20 | 2.10x | |
| **`-np 4` + draft-mtp, 4 in flight** | **108.3 s** | **25.27** | 7.51 | **2.14x** | 66.8% |

- **llama-server accepts `-np 4` with `--spec-type draft-mtp`** (and `--lora`) at the same time.
- **Fastest batch: `-np 4` + MTP, 108.3 s.** But it is only 2% ahead of `-np 4` without MTP (110.4 s).
  - With 4 sequences decoding together the GPU is already doing matrix-matrix work. Verifying drafts on top of that buys almost nothing.
  - On one sequence, MTP alone gives 1.73x.
- **Batched output isn't bit-identical to sequential.**
  - Against one-at-a-time greedy, `-np 4` changes 6 of 12 answers, first at tokens 28–218. `-np 4` + MTP changes 6, at tokens 28–97. `-np 1` + MTP changes 4, at tokens 28–163.
  - The cause is the same as MTP's drift: different kernels for different batch shapes flip near-ties.
  - For exact-match evaluations, fix one serving setup and use it for every model compared.

## Conclusions

1. **The testing path computes the trained model.**
   - The training adapter converted with b11435's `convert_lora_to_gguf.py` and loaded with `--lora` over Ollama's Q4_K_M agrees with transformers + PEFT on the bnb weights as closely as the two bases agree with each other: assistant-token KL 0.038, top-1 96%.
   - Conversion needs a config-only `--base` without `quantization_config`.
2. **Set `GGML_VK_DISABLE_COOPMAT=1` whenever llama-server b11435 (Vulkan) loads a LoRA on this GPU.** Without it, the server dies on the first prompt batch over 8 tokens. Generation speed is unaffected; prompt processing gets slower on long prompts.
3. **Interactive or single-stream use**: `--lora --spec-type draft-mtp` gives 22.6 tok/s with the built-in MTP block, or 24.3 with `-md` and `--spec-draft-n-max 4`. That is up from 12.2 tok/s.
4. **Benchmark batches**: `-np 4` finishes a batch about 2.1x faster than one-at-a-time. Adding MTP to `-np 4` gains only 2% more (108 s vs 110 s for 12 problems).
   - `-np 4` without MTP is the simpler choice at about the same speed.
   - `-np 1` with MTP (1.73x) is the choice when one request at a time is needed.
   - Keep the serving setup fixed across compared models, because batched and speculative greedy output drift at near-ties.
