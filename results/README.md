# results/

Tracked in git (small text files), except `samples/` which holds raw generations and is ignored.

One folder per evaluated model, named to pair up: `qwen3-32b_base/` vs `qwen3-32b_r16-magicoder5k/`.
Each contains `run_info.txt`, `humaneval.log`, `mbpp.log`, and EvalPlus's own result JSONs.

The summary table lives in `experiments/LOG.md`. This folder is the evidence behind it.
