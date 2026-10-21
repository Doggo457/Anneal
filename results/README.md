# results/

Tracked in git (small text files), except `samples/` which holds raw generations and is ignored.

One folder per evaluated model, named to pair up: `qwen3-32b_base/` vs `qwen3-32b_r16-magicoder5k/`.
Each contains `run_info.txt`, `humaneval.log`, `mbpp.log`, and EvalPlus's own result JSONs.

The summary table lives in `experiments/LOG.md`. This folder is the evidence behind it.

## Shared inputs

Runs that used the same input keep one copy of it, in the first run of each group. The other runs' summaries still name their own copy's path on the training machine, where every copy stays.

| Input | Kept in | Same file in |
|---|---|---|
| `eval_text.txt` (the 61 held-out CodeAlpaca rows, joined) | `baseloss-27b-unsloth-fixed/` | every other `baseloss-27b-*/` run |
| `tokens.json` | `parity-llama-hf/` | `parity-llama-hf-2/`, `-3/`, `-4/` |
| `tokens.json` | `parity-evaltext-2048/` | `parity-evaltext-2048-b/`, `-c/` |
| `tokens.json` | `parity-lora-smoke-27b-stock/` | `parity-lora-smoke-27b-stock-b/` |
