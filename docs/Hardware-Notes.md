# Hardware notes: GMKtec EVO-X2

Spec: AMD Ryzen AI Max+ 395 (16 cores / 32 threads, Radeon 8060S iGPU, ROCm target `gfx1151`), 64GB LPDDR5X-8000 unified memory, 1TB NVMe. Rated bandwidth 256 GB/s; community benchmarks measure about 215 GB/s in practice.

## Usable memory (fill in on day one)

| Setting | Expected | Measured | Date |
|---|---|---|---|
| Windows: dedicated pool (BIOS UMA or Adrenalin slider, max on this unit) | 48GB | | |
| Windows: can PyTorch/ROCm allocate beyond the dedicated pool? | unknown, assume no | | |
| Linux: GTT limit set for the GPU | 52GB conservative, 56GB aggressive | | |
| `torch.cuda.mem_get_info()` total | | | |
| `llama-bench` tokens/s, 32B Q4 dense | ~10 | | |
| `llama-bench` tokens/s, 30B-A3B MoE | 70 to 100 | | |
| Free disk after OS and tools | ~850GB | | |

What the numbers mean. The 64GB is one physical pool; the hardware hides none of it from the GPU. The limits are driver policy:

- **Windows** divides it into a "dedicated" pool (the BIOS UMA setting or the Adrenalin VRAM slider; this unit allows up to 48GB) and "shared" memory handed to the GPU on demand. For inference the dedicated figure is a speed boundary, not a ceiling: a test on a 128GB EVO-X2 ran a 90GB model with 2GB dedicated, but throughput was ~40% higher when the model fit inside the dedicated pool. For training, whether PyTorch/ROCm can use the shared pool is unverified; assume 48GB until measured.
- **Linux** has no split. AMD's guidance is to keep the BIOS carve-out at 0.5GB and let the GPU take system memory through GTT (`amdgpu.gttsize` and `ttm.pages_limit`, or AMD's `amd-ttm` tool), with no speed penalty on unified memory. AMD's example leaves ~20% for the OS; on 64GB that is 52GB for the GPU, 56GB if you push it. See `setup/01-linux-and-rocm.md`.

## Speed rule of thumb

Token generation is memory-bandwidth-bound: tokens/s ≈ bandwidth ÷ bytes read per token.

| Model | Bytes per token | Expected tokens/s |
|---|---|---|
| Dense 32B, Q4 | ~19GB | ~11 |
| Dense 70B, Q4 | ~40GB | ~5 |
| MoE 30B-A3B, Q4 | ~2GB (only active experts) | 70 to 100 |
| MoE 120B-A5B, MXFP4 | ~3.5GB | 30 to 55 |

Prompt processing (prefill) is compute-bound and much slower than a discrete GPU, roughly 340 tokens/s on a 120B MoE in community tests. ROCm beats Vulkan at prefill on long contexts; Vulkan is fine for generation and works with zero driver wrangling. Start on Vulkan, move to ROCm when you need long-context evaluation.

## QLoRA memory rule of thumb

To be calibrated against real runs. `scripts/memory_estimate.py` does the arithmetic.

Frozen 4-bit weights ≈ parameters × 0.55 to 0.6 GB per billion. On top of that:

- LoRA adapter, its gradients, and Adam optimiser state: small. Rank 16 on a 32B model is roughly 1 to 1.5GB total.
- Activations: the variable one. With gradient checkpointing, batch 1, expect roughly 2.5 to 3GB per 1k tokens of sequence on a 32B model, and about double that on a 70B.
- Output logits: vocabulary × sequence × 4 bytes. About 0.6GB at 1k tokens with a 150k vocabulary.
- Headroom for the OS, driver, and fragmentation: keep 4 to 6GB free.

| Model | Weights (4-bit) | Estimated total, seq 1024, batch 1 | Fits 48GB (Windows)? | Fits ~56GB (Linux)? |
|---|---|---|---|---|
| gpt-oss-20b | ~11GB | ~17GB | Yes, room for seq 4096+ | Yes |
| Qwen3-32B | ~17GB | ~26GB | Yes, room for seq 2048 to 4096 | Yes |
| Llama-3.3-70B | ~36GB | ~49GB | No (over by ~1GB before anything goes wrong) | Marginal, ~6GB spare. Seq ≤ 1024, rank ≤ 16 |
| Qwen3.8-27B (same shape as 3.6), **8-bit LoRA** | ~28GB | ~37GB at seq 1024, ~42GB at seq 2048 | Yes at 1024, tight at 2048 | Yes |
| Qwen3.8-27B, 4-bit QLoRA | ~14GB | ~24GB | Yes | Yes (Unsloth warns 4-bit is lossy on this family) |
| Qwen3.8-27B, bf16 LoRA | ~54GB | ~65GB | No | No |
| Any 30B–35B mixture-of-experts model, 4-bit or 8-bit LoRA | ~18GB / ~36GB if the experts were quantised | ~25GB / ~43GB on paper; **~66–70GB in practice**, because bitsandbytes leaves the fused expert tensors in bf16 | No | No |

The MoE row is why the project uses a dense model, and why `memory_estimate.py` prints a second "experts in bf16" total for MoE presets. (llama.cpp's GGUF quantisation does cover experts, so MoE models run fine for inference; it is training that fails.)

These are the numbers `scripts/memory_estimate.py` prints with its default constants (10% overhead on 4-bit weights, 5GB system headroom). Real runs on QLoRA hardware often land 10 to 20% above estimates like these, which is why the 70B row says marginal even on Linux. The 70B row is the whole reason to prefer Linux: the Windows cap eats exactly the margin that run needs.

## Qwen3.8-27B: published and measured numbers (collected 19 September 2026)

These are other people's measurements and the repo's actual file sizes, not the estimator. Replace the "Measured here" column with your own numbers from `scripts/measure_memory.py` on day one.

| What | Figure | Source | Measured here |
|---|---|---|---|
| bf16 weights on disk | ~55GB (18 safetensors shards, 2.1–4.0GB each) | `Qwen/Qwen3.8-27B` repo listing | |
| bf16 weights in GPU memory, inference | 56GB | Unsloth Qwen3.8 page | |
| 8-bit weights, inference | 31GB | Unsloth Qwen3.8 page | |
| 4-bit weights, inference | 16–19GB | Unsloth Qwen3.8 page | |
| 16-bit LoRA training, 27B class | 64GB, "absolute minimum" | Unsloth requirements table | |
| bf16 LoRA training, Qwen3.5-27B | 56GB | Unsloth Qwen3.5 fine-tuning guide | |
| 4-bit QLoRA training, 27B class | 22GB minimum; ~19GB peak in one reported run on a 48GB card | Unsloth requirements table; MindStudio guide | |
| 8-bit LoRA training | not published anywhere. Derived: 31GB weights + the ~6GB overhead QLoRA shows (22 − 16) ≈ 37GB at short context | derived | |
| GPU memory budget on this box | Windows: 48GB dedicated (BIOS max on this unit), training beyond it unverified. Linux: 52–56GB via GTT | GMKtec BIOS documentation; AMD ROCm Strix Halo guide; EVO-X2 Windows test | |

Sequence length matters as much as precision once thinking mode is on (traces run thousands of tokens). Estimator totals for training, batch 1, rank 16:

| Precision | seq 2048 | seq 3072 | seq 4096 | seq 6144 |
|---|---|---|---|---|
| 8-bit LoRA | 41GB, fits both | 45GB, tight on Windows | 49GB, over on Windows, tight on Linux | 58GB, no |
| 4-bit QLoRA | 27GB | 31GB | 35GB, fits both with room | 44GB, Linux |

The project trains at 4096. That means Linux with 8-bit if the day-one measurement confirms it, otherwise 4-bit.

Reading it: bf16 LoRA needs 56–64GB and the weights alone are 55GB, against 48–56GB usable. It is over by 8–16GB on Linux and 16–24GB on Windows before the first token of context. 8-bit puts the weights at 31GB and the whole step around 37–41GB, inside both budgets at seq 1024. 4-bit is comfortable everywhere. There is a separate `Qwen/Qwen3.8-27B-FP8` repo (~28GB); it is meant for GPUs with FP8 hardware (Hopper, MI300) and on this chip it would be expanded back to bf16 on load, so it does not help.

## Context length: two different numbers

**Inference and evaluation run at the model's full native context, 262,144 tokens.** It is affordable because of the architecture: only 16 of the 64 layers are full attention (the other 48 are linear attention with a small fixed-size state), with 4 KV heads of dimension 256. That is 64KB of KV cache per token.

| Setting | Weights | KV cache at 262k | Total | Fits |
|---|---|---|---|---|
| Q8_0, KV f16 | 28GB | 16GB | ~46GB | Yes |
| Q8_0, KV q8_0 (`-ctk q8_0 -ctv q8_0`) | 28GB | 8GB | ~38GB | Yes, the default in `run_eval.sh` |
| Q4_K_M, KV q4_0 | 16GB | 4GB | ~22GB | Yes |

What to measure on day one: prompt-processing time for a 100k-token prompt. Prefill is compute-bound and this chip is slow at it; expect a full 262k fill to take tens of minutes. Fine for the personal task set, irrelevant for HumanEval+/MBPP+ whose prompts are a few hundred tokens.

**Training sequence length is a different knob.** It is the length of one training example, and it does not change the model's context window: LoRA leaves the positional machinery alone, so the fine-tuned model keeps 262k whatever length it was trained at. The training length only needs to cover the traces. It is bounded by activation memory, which grows with sequence length on any machine:

| Training example length | 8-bit | 4-bit |
|---|---|---|
| 4,096 | ~49GB, Linux only, tight | ~35GB, fits both |
| 8,192 | ~68GB, no | ~54GB, Linux only, marginal |
| 16,384 | ~113GB, no | ~99GB, no |

(The estimator's attention term assumes materialised attention scores; with flash attention and the model's linear-attention layers the real numbers at 8k are somewhat lower. Measure before ruling 4-bit at 8k out.) Traces longer than the training length are dropped, not truncated. One risk to watch: fine-tuning only on 4k-token examples can slightly dull behaviour at very long contexts, so the forgetting check includes a long-context probe (`Dataset-Candidates.md`).

## Measured inference speed: Qwen3.8-27B on Ryzen AI Max+ 395 (sourced, September 2026)

| Quant | Size | Generation, plain | Generation with MTP | Prefill | Source |
|---|---|---|---|---|---|
| Q8_K_XL (Unsloth) | 31.5GB | 7.2 tok/s | 15.3 tok/s, 66% draft acceptance | 262 tok/s on an 11k prompt, 134 on short prompts | MikeVeerman/qwen38-27-Strix-Halo-bench, EVO-X2-class box, Vulkan RADV, Ubuntu, llama.cpp b9867 |
| Q6_K_XL (Unsloth) | 25.9GB | 8.4 | 18.7 | | same |
| ROCmFP8 | 26.3GB | 7.7 | | 382 tok/s at 512 tokens, ~245 at 32k (TTFT ~130 s) | julianmb/q38rocm, 128GB unit, ROCm |
| Q4_K_M | 15.9GB | 12.3 | 24.5 (AMD, MTP=4, Windows, Vulkan) | | julianmb; AMD day-0 blog |
| ROCmFP4 + tuned MTP | 13.6GB | 14.0 | 30 to 36, greedy sampling only | | julianmb |

Sanity check: 31.5GB × 7.2 tok/s ≈ 227GB/s, so the plain numbers are the memory bus. Only two things move them: MTP (multi-token prediction heads the model ships with; llama.cpp uses them as a speculative draft, `--spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-p-min 0.0` on build b9867) and batching several streams. Two measured catches with MTP: acceptance falls as temperature rises (at 0.8 it collapsed to ~25% and the gain vanished; 66% is the realistic middle at Qwen's 0.6), and with 4 parallel requests MTP made throughput worse (0.78×). So it is MTP with one stream, or parallel slots without MTP, not both. `run_eval.sh` does the former by default. The one relevant thing nobody has measured on this chip is the aggregate throughput of 2 to 8 plain batched streams for a dense 27B; `NP=4 MTP=0` on the calibration run measures it.

## What a full evaluation pass costs (derived, calibrate before trusting)

542 problems: HumanEval+ 164, MBPP+ 378. Tokens per problem in thinking mode is the unknown: nobody publishes it for these sets. `LIMIT=20 scripts/run_eval.sh calib <gguf>` measures it and prints the projection. Until then, the table assumes 1,000 / 2,000 / 3,000 tokens per problem.

| Effective tok/s | 1k tokens/problem | 2k | 3k |
|---|---|---|---|
| 7.2 (Q8, plain, one stream) | 21 h | 42 h | 63 h |
| 15 (Q8 + MTP) | 10 h | 20 h | 30 h |
| 22 (Q4_K_M + MTP) | 7 h | 14 h | 21 h |

Whatever quantisation and settings the calibration run picks are then fixed for the baseline and every fine-tune.

## Windows vs Linux

**Windows.** LM Studio with the Vulkan runtime gives working inference in under an hour, and inference can use memory beyond the 48GB dedicated pool (slower once it spills). Unsloth documents native Windows ROCm support. What is not known is whether training can use anything beyond the dedicated 48GB, and every ROCm troubleshooting thread assumes Linux.

**Linux (Fedora 42 or Ubuntu 24.04 LTS).** One pool, no dedicated/shared split, 52–56GB for the GPU at full speed through GTT. The community benchmark tooling (the strix-halo toolboxes) targets it, and AMD's own Strix Halo optimisation page is written for it.

Recommendation: Linux on the EVO-X2 for training, keep Windows on this PC. Thinking-mode runs at seq 4096 in 8-bit estimate at ~49GB, which is over the Windows dedicated pool and inside the Linux GTT budget, so the OS choice and the precision choice are the same decision. Inference-only work can happen on either. Write the decision in `Decision-Log.md` when made.

## Storage discipline

One 32B experiment, worst case: ~65GB bf16 download + ~65GB merged copy + ~20GB GGUF ≈ 150GB peak.

Keep permanently: the LoRA adapter (small) and the quantised GGUF. Delete the merged bf16 as soon as the GGUF is verified with a test generation. Delete the bf16 base only when you are finished with that base model entirely. `scripts/cleanup_intermediates.py --dry-run` lists candidates.

## Power and thermals

About 120W sustained under LLM load, fans audible. Set the performance power mode in the BIOS or GMKtec's control app before long training runs. Watch for throttling with `rocm-smi` or `sensors`. If tokens/s drops over a long run, it is thermal, not software.
