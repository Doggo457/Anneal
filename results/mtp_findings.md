# MTP speculative decoding for Qwen3.8-27B in llama.cpp

18 October 2026. Scripts: `tests/mtp_spec_bench.py` (`results/mtp-spec-bench`) and `tests/mtp_divergence_check.py` (`results/mtp-divergence-check`). Both were run through the Training page.

## Setup

- **Draft file**: `unsloth/Qwen3.8-27B-GGUF`, `MTP/mtp-Qwen3.8-27B-Q4_0.gguf`.
  - Repo revision `4ca72078`; 1,369,590,656 bytes; sha256 `50d9ce5a6da3…501c6e`, which matches Hugging Face.
  - Saved in `C:\AI\training\models\unsloth-Qwen3.8-27B-GGUF-MTP\`.
  - It is an MTP-only GGUF: 18 tensors, namely the `blk.64` nextn block plus its own Q3_K token embedding and output head (2.97B parameters, 3.66 bits/weight).
- **Base model**: Ollama's `qwen3.8:27b` Q4_K_M blob. That file already contains an MTP block of its own (`blk.64.nextn.*`), which llama.cpp ignores unless asked.
- **Server**: llama.cpp b11435, Vulkan, `llama-server -c 8192 -ngl 999 -np 1 --ctx-checkpoints 0 --cache-ram 0`. A fresh server per configuration; only the speculative flags differ.
  - Context checkpoints are off everywhere because each one costs about 5 s on this machine (see `parity_findings.md`).
  - Speculative rollback uses its own on-GPU recurrent-state snapshots (`n_rs_seq` = draft length).
- **Workload**:
  - 6 prompts: 4 chat requests (code, explanation, a JS refactor, a bash one-liner) and 2 raw completions (code, prose).
  - Fixed token IDs, greedy (temperature 0), 256 new tokens each, after one warm-up request.
- **Speed measure**: the server's own generation timings (`predicted_n / predicted_ms`). Prompt processing was the same in every configuration (30–210 tok/s depending on prompt length).

## Results

| configuration | flags | gen tok/s | speedup | drafted | accepted | acceptance | wall time for 6 × 256 tokens |
|---|---|---|---|---|---|---|---|
| base | none | **12.69** | 1.00x | | | | 123.0 s |
| MTP, draft 1 | `--spec-type draft-mtp -md mtp.gguf --spec-draft-n-max 1` | 19.70 | 1.55x | 821 | 707 | 86.1% | 80.0 s |
| MTP, draft 2 | `… --spec-draft-n-max 2` | 23.53 | 1.85x | 1,174 | 938 | 79.9% | 67.2 s |
| **MTP, draft 3 (default)** | `--spec-type draft-mtp -md mtp.gguf` | **24.14** | **1.90x** | 1,524 | 1,015 | 66.6% | 65.6 s |
| MTP, draft 4 | `… --spec-draft-n-max 4` | 25.41 | 2.00x | 1,798 | 1,074 | 59.7% | 62.4 s |
| built-in MTP block of the Ollama GGUF | `--spec-type draft-mtp` (no `-md`) | 23.72 | 1.87x | 1,496 | 1,026 | 68.6% | 66.8 s |

Per prompt with the default MTP, in tok/s: code-chat 21.9, explain 21.6, refactor 26.1, bash 27.3, raw code 26.8, raw prose 22.0. Code gains the most and free-form prose the least, as expected from draft acceptance (about 79% vs about 57% per prompt).

Unsloth's separate file and the block already inside Ollama's GGUF perform about the same: 24.14 vs 23.72 tok/s, acceptance 66.6% vs 68.6%. With the built-in block, no extra file or `-md` is needed.

## Does the greedy output stay the same?

- **Not bit-identical**, but only through near-ties.
  - With the default MTP, 1 of 6 prompts (raw code) is identical over all 256 tokens. The other five first diverge at tokens 86–170. Other draft lengths are similar.
  - Divergence points repeat across draft lengths: code-chat at 87 and bash-chat at 161 in every configuration.
- `mtp_divergence_check.py` re-asked the plain model (no speculative decoding) for its top-5 at each of the 26 first-divergence points (all configurations). **At all 26**:
  - The speculative run's token is one of the plain model's top two.
  - The probability gap is at most 0.033. Examples: " hours" 0.468 vs " Hours" 0.475; "But" 0.4047 vs "The" 0.4046; " clean" 0.439 vs " useful" 0.423.
  - At 8 of them, the plain model re-scored by prompt processing ranks the speculative run's token first.
- **Why**: speculative decoding only emits tokens the target model chose during verification. Verification runs several tokens per forward pass (matrix-matrix kernels, the chunked GatedDeltaNet path), while plain decoding runs one (matrix-vector, the autoregressive path). The logits differ in the last bits, which flips argmax at near-ties. After the first flip the texts go their own ways, and all versions read equally well.

## Conclusion

- MTP gives about **1.9x faster generation** (12.7 → 24.1 tok/s) on this machine with the default settings.
- `--spec-draft-n-max 4` gives 2.0x (25.4 tok/s), with diminishing returns as acceptance falls.
- The draft costs only a 1.4 GB file, and Ollama's own GGUF already carries an equally good MTP block.
- Output quality is unchanged: every difference from plain greedy decoding is a near-tie the model itself rates as about 50/50.
- For reproducible evaluation runs (exact-match baselines), use plain decoding. For interactive or long generations, MTP is a free speedup.
- I did not measure the extra GPU memory for the draft context (llama-server doesn't log buffer sizes at its default verbosity). The draft weights are 1.37 GB, or about 0.3 GB for the built-in block, against 16.8 GB for the base model.
