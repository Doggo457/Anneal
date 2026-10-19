# Experiment log

Add a row before a run starts (with the question), and fill the results after. Scores are pass@1 from `scripts/run_eval.sh` (HumanEval+ and MBPP+ are the "plus" numbers; keep the base numbers too). "Own" is the personal task set score out of its maximum. "Forget" is the small non-coding check.

## This week's question
_Write it here on Monday._

## Calibration (first 20 problems, `LIMIT=20 run_eval.sh calib ...`)

| Date | Quant | MTP / NP | Mean tokens per problem | Aggregate tok/s | Projected hours, 542 problems | Chosen? |
|---|---|---|---|---|---|---|
| | Q8_0 | MTP, NP=1 | | | | |
| | Q8_0 | no MTP, NP=4 | | | | |
| | Q4_K_M | MTP, NP=1 | | | | |

## Baselines (the untouched base model, no training)

Run once per quantisation and thinking mode you will evaluate fine-tunes in. Same settings for base and fine-tuned, always. Reasoning budget 32,768, total 40,960.

| Date | Model | Quant | Thinking | Seed | HumanEval / + | MBPP / + | Own | Forget | tok/s | Notes |
|---|---|---|---|---|---|---|---|---|---|---|
| | Qwen3.8-27B base | Q8_0 | on | 1 | | | | | | temp 0.6, top-p 0.95 |
| | Qwen3.8-27B base | Q8_0 | on | 2 | | | | | | |
| | Qwen3.8-27B base | Q8_0 | on | 3 | | | | | | mean and spread of the three anchor everything |

## Memory measurements (`scripts/measure_memory.py`)

| Date | Precision | Seq | After load GiB | After LoRA GiB | Peak step GiB | Fits? | Notes |
|---|---|---|---|---|---|---|---|
| | 8-bit | 4096 | | | | | decides the project's precision |
| | 8-bit | 3072 | | | | | only if 4096 failed |
| | 4-bit | 4096 | | | | | |
| | bf16 | 1024 | | | | | expected to fail; record how |

## Fine-tuning runs

| Date | Run (YAML) | Changed vs previous | Samples | Steps | Hours | Peak GiB | HumanEval+ | MBPP+ | Own | Forget | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| | 2026-XX-XX_qwen38-27b_..._smoke | pipeline test | 200 | 30 | | | n/a | n/a | n/a | n/a | did train → merge → export → eval work? |
| | | | | | | | | | | | |

## Things that broke and how they were fixed

| Date | Symptom | Cause | Fix | Versions |
|---|---|---|---|---|
| | | | | |
