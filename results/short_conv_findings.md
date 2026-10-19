# Shift-and-add GatedDeltaNet short convolution

18 October 2026. Code: `anneal/short_conv.py`. Checks: `tests/short_conv_check.py` (`results/short-conv-check`, `short-conv-check-2`). Training step: `scripts/measure_memory.py --short-conv shift` (`results/mem-27b-8k-shiftconv`).

## What it replaces

Every GatedDeltaNet layer runs a causal depthwise conv1d over q, k and v together:
- 10,240 channels, kernel 4, no bias, SiLU.
- transformers calls `causal_conv1d_fn` for it. On ROCm that is its PyTorch fallback, `F.conv1d(groups=channels)` on a (B, C, T) view of the (B, T, C) projection output.

With width 4 the convolution is four shifted multiply-adds along time:

    y[t, c] = Σ_k w[c, k] · x[t − 3 + k, c]

How `anneal/short_conv.py` does it:
- Works directly in (B, T, C), with its own backward:
  - dx is the same shifted sum with the taps mirrored.
  - dw[c, k] is a per-channel dot product over (b, t).
- Keeps partial sums in fp32, in 2,048-step blocks.
- Swaps the module-level `causal_conv1d_fn` in transformers' `modeling_qwen3_5` and in Unsloth's `unsloth_compiled_module_qwen3_5`. The single-token decode path is untouched.

## Correctness

Both implementations were compared with a float64 CPU reference of the same function: conv + SiLU, forward and both gradients. Normwise relative error:

| T × channels | bf16 F.conv1d y / dx / dw | bf16 shift y / dx / dw | fp32 shift y / dx / dw |
|---|---|---|---|
| 1, 3, 5 × 256 (shorter than the kernel) | 4.0e-3 / 3.5e-3 / 2.9e-3 … | identical | ≤ 7e-8 |
| 2047, 2049, 4099 × 512 (block edges) | 3.7e-3 / 3.9e-3 / 3.6e-3 | 3.7e-3 / 3.6e-3 / 3.5e-3 | ≤ 1.3e-7 |
| 4096 × 10240 | 3.7e-3 / 3.9e-3 / 3.6e-3 | 3.7e-3 / 3.6e-3 / 3.6e-3 | ≤ 1.3e-7 |
| 8192 × 10240 | 3.7e-3 / 4.0e-3 / 3.6e-3 | 3.7e-3 / 3.6e-3 / 3.6e-3 | ≤ 1.3e-7 |

- In fp32 the shift version equals the exact result (about 1e-7).
- In bf16 its error is the same as PyTorch's own: identical for y, slightly lower for dx.
- The 27B training losses with it match the baseline at every step to within 0.005 (table below).

## Speed: the convolution alone (bf16, 10,240 channels, conv + getting back to contiguous (B, T, C))

| T | F.conv1d fwd | F.conv1d fwd+bwd | same on a contiguous copy | shift fwd | shift fwd+bwd |
|---|---|---|---|---|---|
| 4096 | 174 ms | 399 ms | 196 / 417 ms | 11.7 ms | 36.8 ms |
| 8192 | 570 ms | 1,323 ms | 585 / 1,279 ms | 23.2 ms | 73.2 ms |

- The cost is in MIOpen's depthwise convolution itself, not in the strided layout.
- Peak extra memory during forward+backward at 8K: F.conv1d 1,280 MiB, shift 1,121 MiB.
- Across 48 layers × (forward + recomputed forward + backward) that predicts about 91 s vs 4.6 s per 8K step.

## Speed: real 27B training steps

Same settings as `mem-27b-8k-fixed`: Unsloth 4-bit, LoRA r=16 on all projections, Unsloth gradient checkpointing, 5 rows × 8,192 tokens, adamw_8bit. Run through the Training page.

| | mem-27b-8k-fixed (F.conv1d) | mem-27b-8k-shiftconv (shift) |
|---|---|---|
| step seconds | 185.8, 133.9, 134.6, 135.0, 134.0 | 128.9, 84.4, 84.5, 84.5, 84.5 |
| steady s/step | **134.55** | **84.52** (−37%, 1.59x) |
| tokens/s (steady) | 60.9 | 96.9 |
| losses | 1.1984, 1.6223, 1.2585, 1.0619, 1.5162 | 1.1981, 1.6272, 1.2579, 1.0605, 1.5113 |
| peak reserved / allocated (torch) | 26.20 / 24.43 GiB | 25.21 / 24.43 GiB |
| peak GPU seen by Windows (dedicated + shared) | 37.7 GiB (29.4 + 8.3) | 36.0 GiB (27.7 + 8.3) |
| peak system RAM used | 17.7 GiB | 16.8 GiB |
| wall time for 5 steps | 725.5 s | 469.1 s |

- Steps vary by under 1 s within a run, so the 50 s difference is far outside the noise.
- The saving is smaller than the stand-alone estimate (about 50 s rather than 86 s), but clearly measurable.
- The patch was confirmed active: 480 calls = 48 layers × 5 steps × (forward + recompute).
- Both runs used `--gdn-heads auto` (the tile pairing, now known to be wrong; see `parity_findings.md`), so the comparison is like-for-like. The pairing doesn't change the amount of work.

## Decision

Kept. `--short-conv shift` is now the default in `scripts/measure_memory.py` and `tests/smoke_small.py`. `--short-conv torch` restores transformers' F.conv1d.

At 8K rows, a step that took 134.6 s now takes 84.5 s. For a fixed time budget that is about 1.6x more training tokens.
