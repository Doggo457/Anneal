# Base model

## The constraint (19 September 2026)

The project agreement fixes the work to **one base model, 20B to 40B parameters, trained on the owned 64GB EVO-X2**. One model means one: the before/after comparison is that model's untouched weights against its fine-tuned weights, on the same benchmarks, at the same quantisation, in the same settings. No second model as a reference, no inference-only comparisons, no rented compute.

## The model: Qwen3.8-27B

| | Qwen3.8-27B |
|---|---|
| Type | Dense: 64 layers, hidden 5120, vocab 248,320, hybrid linear/full attention. Multimodal checkpoint; only the text side is trained |
| Licence | Apache-2.0 |
| Published | August 2026 (`Qwen/Qwen3.8-27B`) |
| Context | 262k native. Thinking on by default, switchable per request |
| Coding scores (model card) | Terminal-Bench 2.1 73.0, SWE-bench Pro 61.7, LiveCodeBench v6 90.3 |
| bf16 weights on disk | 55.6GB across 18 shards |
| 8-bit LoRA training | ~37GB at seq 1024, ~41GB at seq 2048. Derived from published pieces; measure on day one |
| 4-bit QLoRA training | 22GB minimum (Unsloth's table); ~19GB measured by others |
| bf16 LoRA training | 56–64GB (Unsloth). Does not fit on this box |

**Why a dense model.** Quantised training (8-bit or 4-bit) is the only way a 20B–40B model fits in 64GB, and quantised training depends on bitsandbytes shrinking every weight. bitsandbytes cannot shrink the fused expert tensors that mixture-of-experts models use, so a 30B–35B MoE loads at ~66GB regardless of the bit width you ask for, and fails. A dense model has no such layer, so its memory numbers are real. Details in `Hardware-Notes.md`.

**Why 8-bit first.** Unsloth warns that 4-bit is lossier than usual on the Qwen3.5 family. 8-bit halves the weights to 31GB with negligible loss and still fits. 4-bit is the fallback if 8-bit misbehaves on ROCm, or if 8-bit does not fit at the 4096-token sequences thinking mode needs (estimate 49GB, Linux only and tight; 4-bit at 4096 is ~35GB). `load_in_4bit: true` in the template, nothing else changes. `measure_memory.py` on day one decides it.

**Headroom.** The model already scores highly on public coding benchmarks, so a generic "make it better at coding" fine-tune is unlikely to move HumanEval+ and may hurt it. The measurable question is adaptation to a specific codebase and its conventions, scored on the personal task set, with HumanEval+/MBPP+ as the "did it get worse" guard. That is what the brief's own test set is for.

Template: `experiments/TEMPLATE-qwen3.8-27b-8bit.yaml`.

## Fallbacks, only if Qwen3.8-27B fails on ROCm

Same-shape earlier releases first: `Qwen/Qwen3.6-27B` (April 2026), then `Qwen/Qwen3.5-27B`. Change `base_model` only; every memory number carries over. If the whole family is the problem, other dense models in range: `Qwen/Qwen2.5-Coder-32B-Instruct` in 4-bit (the most-trodden 32B QLoRA path, not covered by the Qwen3.5 4-bit warning), `Qwen/Qwen3-32B`, `mistralai/Devstral-Small` (24B). Any switch is a decisions-log entry and restarts the baselines; the project still ends with one model.
