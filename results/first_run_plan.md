# First real LoRA run: plan, setup and choices

19 October 2026. Controller: `scripts/first_run.py`, launched from the Training page as run `first-run`, with run folder `results/first-run`.

## Goal

- **Data**: 500 self-generated, test-passing rows (the base model's own answers to OpenCodeInstruct prompts) + 1,500 OpenCodeReasoning split_0 rows. Planned as 1,000 + 1,000; changed on 19 Oct, see "Change: 500 self rows" below.
- **Training**: 4-bit QLoRA on Qwen3.8-27B (`first-run-2k-r16`).
- **Scoring**: EvalPlus HumanEval+ (164) + MBPP+ (378) for base and adapter, one seed each.

## Stages (each resumable, each leaves `results/first-run/stages/<stage>.done.json`)

1. **calibrate**
   - 20 problems (10 HumanEval+, 10 MBPP+, `random.Random(42).sample`), seed 1, scoring settings.
   - Setup A: `-np 4`. Setup B: `-np 4 --spec-type draft-mtp`, the built-in MTP block. This is the only MTP configuration measured with `-np 4` (`lora_testing_findings.md`).
   - B is chosen only if it is at least 3% faster in total generated tok/s.
2. **generate**
   - Server: base GGUF on the chosen setup, `-c 65536` (16k per slot), `--reasoning-budget 6144`, Vulkan coopmat on (no LoRA).
   - `generate_traces.py`: OpenCodeInstruct shards 00000 and 00049, min test score 1.0, decontaminated, tests in the sandbox, 4 in flight, until 1,020 rows are kept.
   - Then `prepare_dataset.py --from-jsonl … --require-think --max-tokens 8192 --decontaminate`.
3. **mix**: `mix_datasets.py` with self 1,000 and ocr 1,000, into `data/processed/mix_first_2k`.
4. **train**: `train_longrun.py`, `experiments/2026-10-19_first-run.yaml`, in `results/first-run/train`.
5. **convert**: b11435 `convert_lora_to_gguf.py --base models/lora-base-config/... --outtype f16`.
6. **sanity**
   - `val_loss.py`: answer-token loss on the mix's val rows, base vs adapter, on the same 4-bit model.
   - 3 sample generations with `--lora`: they must think, answer and stop, with no loops.
7. **score**: base seed 1, then adapter seed 1, all 542 problems, identical scoring server, EvalPlus's evaluator with the sandboxed checker.
8. **report**: `results/first-run/report.md`.

**Scoring server** (`anneal/llama.py`, identical for both models):
- `GGML_VK_DISABLE_COOPMAT=1`, `-c 262144 -np 4 -ctk q8_0 -ctv q8_0 -fa on -ngl 999 --ctx-checkpoints 0 --cache-ram 0 --reasoning-format deepseek --reasoning-budget 32768`, plus the setup-B flag if chosen.
- Sampling: temperature 0.6, top_p 0.95, top_k 20, min_p 0. The same values are set as server defaults and in every request.
- 40,960 tokens per answer; the adapter is loaded with `--lora`.

## Status on 19 Oct 2026 at 06:10 local (05:10 UTC)

- **Run**: Training page run `first-run`, `scripts/first_run.py` (pid 6612), folder `C:\AI\training\results\first-run`, started 03:50 local.
- **Calibration done** (`stages/calibrate.done.json`):
  - Setup A: 29.16 tok/s, completion tokens mean 877 / median 488 / max 3,013, 20/20 base, 18/20 base+extra.
  - Setup B (MTP): 27.55 tok/s (0.945× A), draft acceptance 61.6%.
  - **A chosen**. Projected 4.6 h per 542-problem pass.
- **Generation running** since 04:13 local:
  - At 06:03: 50 kept of 93 attempts (53.8%). Rejected: tests_failed 27, budget_hit 15, length_or_empty 1.
  - 31.2 generated tok/s, about 27 kept rows/h.
  - 18 of 20 tests_failed prompts have reference solutions that pass in the sandbox, so those are model misses.
  - `tests/oci_reference_check.py`: 97.5% of the first 200 reference solutions pass, so the runner is sound.

**ETA per stage** (local time, UTC+1):

| stage | estimate | basis | done around |
|---|---|---|---|
| generate | ~36 h (30–45) | 970 rows at ~27 kept/h | 20 Oct ~18:00 |
| prepare self + mix | ~10 min | | 20 Oct ~18:15 |
| train | ~17–18 h | ~5.86 M tokens (OCR 4.46 M + self ~1.4 M) at ~98 tok/s, plus ~2 min warm-up per start and checkpoints | 21 Oct ~12:00 |
| convert | ~2 min | tested on a dry-run adapter | 21 Oct ~12:05 |
| sanity | ~40 min | val loss on ~40 rows (base + adapter) and 3 samples | 21 Oct ~12:45 |
| score | ~9.5–10 h | 4.6 h base + ~4.8 h adapter (LoRA costs ~3.5% decode); longer if hard problems think longer | 21 Oct ~22:30 |
| report | ~1 min | builder tested on synthetic scores | 21 Oct ~22:30 |

## Data so far

- **`data/processed/ocr_first`**: 1,000 train / 20 val, from 3,116 rows examined.
  - Dropped: 1,949 over 8,192 tokens, 144 duplicates, 2 too short, 1 by 13-gram decontamination.
  - Kept-row length: median 4,502 tokens, p90 7,386.
  - The 8,192-token cap keeps 75% of EASY problems but only 4% of VERY_HARD (`data_card.md`).
  - I read 20 rows. All are stdio competitive-programming tasks with R1-style traces (1.6k–27k characters, "Okay, let's…") and one final ```python block. Two prompts are near-empty AIZU stubs ("UnionFind (random)", "Example … Output Cat"), so those answers are guesses.
  - The R1 trace voice is far wordier than the base model's own terse traces ("We need answer user's…"). The mix therefore also shifts thinking style.
- **OpenCodeInstruct**: the shards are cached (200,000 rows, 60,162 with average_test_score 1.0).

## Checks before the run

- **Sandbox**: `results/sandbox_findings.md`.
- **EvalPlus harness**: `results/evalplus_windows_findings.md`. Canonical MBPP+ scores 100%. HumanEval+ scores 163/164 because of the evalplus 0.3.1 find_zero bug.
- **Template, separation and budget**: same file. The template matches without `--chat-template-file`, and the budget is enforced by the server.
- **Resume** (`results/resume-check`, Qwen3.5-0.8B):
  - 6 steps straight vs `--stop-at-step 3` → paused with checkpoint-3 → resumed to 6.
  - Steps 4–6 have bit-identical losses (0.8192, 0.8033, 0.8640) and learning rates.
- **27B dry run** (`results/resume-check-27b`): see below.

## Change: 500 self rows (19 Oct 2026, 19:41 UTC, my decision)

- Self-generation stops at 520 kept rows (500 train + 10 val after `prepare_dataset.py`, with room for rows the 8,192-token check drops) instead of 1,020. At about 27 kept rows an hour this saves roughly a day of generation.
- The other 500 rows are a second OpenCodeReasoning split_0 draw, `data/processed/ocr_first_b`: the same three shards and filters as `ocr_first`, with `--seed 43 --max-samples 515` and `--decontaminate-gate data/eval/ocr_first_plus_gate.jsonl`. That gate file is the loop's held-out problems (`decontam_extra.jsonl`) plus every `ocr_first` prompt, so no new row repeats or nearly repeats an `ocr_first` question or a held-out problem. `scripts/first_run_ocr_b.py` builds the gate file and checks the result.
- The mix is now `self:500 + ocr:1000 + ocr_b:500` into `data/processed/mix_first_2k`, still 2,000 rows. OpenCodeReasoning rows are longer than the self rows, so training takes a few hours longer than with 1,000 + 1,000.
- Applied by pausing the run on the Training page, patching `scripts/first_run.py` (sha-checked, backup in `backups/first_run-before-500`) and resuming; generation then stops by itself at 520.

## Choices made on this machine

1. **The trace moves to `reasoning_content` before encoding.**
   - Qwen3.8's chat template renders reasoning only from `message.reasoning_content`; it never splits `<think>…</think>` out of the content.
   - Rows from `prepare_dataset.py` / `generate_traces.py` keep the trace in the content, so unsplit they render as `<think>\n\n</think>\n\n<think>\nTRACE\n</think>\n\nANSWER`.
   - `train_longrun.py` (`split_trace`) moves the trace first. The row then renders `<think>\nTRACE\n</think>\n\nANSWER`, and the generation prompt `…<think>\n` is an exact prefix (0 misaligned rows out of 50 OCR rows checked).
   - Training covers the trace, the answer and `<|im_end|>\n`, about 90% of each OCR row's tokens.
2. **No `--config split_0` for OpenCodeReasoning.**
   - datasets 4.8.5 rejects it with `--data-files`: the card declares a `split_0` split, while a file subset builds `train` (ExpectedMoreSplitsError).
   - Without `--config`, the same three shards load the same 56,787 rows. `prepare_dataset.py` stays byte-identical to the cloud copy.
3. **OpenCodeInstruct is read as local parquet.**
   - It is loaded with `--dataset parquet --data-files <the two cached shard files>`, because datasets 4.8.5 rejects 2 of 50 shards against the card's declared split size (NonMatchingSplitsSizesError).
   - The rows are the same, and `generate_traces.py` is unchanged apart from the sandbox.
4. **`generate_traces.py` changes.** Only the two test runners changed: `anneal.sandbox.run` with 20 s per assert run, 10 s per stdio case, 4 GiB and a fresh scratch folder each. `--run-tests` is refused without the sandbox.
   - The sandbox has numpy but not pandas or pytest. Prompts whose tests need those fail and are rejected, which lowers the keep rate a little.
5. **The generation server gets the scoring sampler as defaults** (`--top-k 20 --min-p 0`, plus temperature and top_p). `generate_traces.py` sends only temperature and top_p, and llama-server's own defaults would be top_k 40 / min_p 0.05.
6. **`train_longrun.py` changes**:
   - The portal run contract: `--run-dir`, `run.log`, `metrics.jsonl`, `status.json`, `summary.json`, STOP in the run folder, `--resume`.
   - The smoke-27b-stock model setup: stock heads, shift conv, bf16, Unsloth gradient checkpointing, `UNSLOTH_BLOCK_COMPILE=0`.
   - `encode_chat` labels with the collated-label check, and a vision-tower parameter check.
   - `expect_trainable_params`, and `--stop-at-step` for tests.
   - The allocator cache is released after every micro-batch (below).
   - WSD, resume, the SHA-256 pin, milestones/branches and drop-never-truncate are unchanged.
7. **Training memory on variable-length rows** (27B dry runs, `results/resume-check-27b*`, `results/ce-check`). Three settings in `experiments/2026-10-19_first-run.yaml`, each tested.
   - **What failed.** The first 3-step dry run (16 OCR rows, grad_accum 2) failed at step 2 on a 7,891-token row with Unsloth's "No or negligible GPU memory available for fused cross entropy".
     - Unsloth sizes its fused-CE chunks from half of `torch.cuda.mem_get_info()` free memory.
     - On this GPU HIP counts the 32 GiB dedicated plus 11.9 GiB of the shared pool (43.9 GiB), and that counter read exactly 0.
     - Neither releasing the allocator cache after every micro-batch nor pageable offload buffers (`UNSLOTH_DISABLE_PINNED_MEMORY=1`) changed this. Both are still on; they cost nothing measurable (step 1: 65.6 vs 65.9 tok/s).
   - **What the memory probe showed** (`mem_debug`, memory logged right before the loss in every forward):
     - Torch's reservation stays at 22–25 GiB.
     - The process's Windows shared GPU memory goes 3.4 → 4.3 → 12.0 GiB as rows of 6,019, 2,569 and then 7,891 tokens pass, and never comes back down.
     - With the longest rows first (6 rows of 8,135–8,188 tokens), it sits flat at 8.3 GiB and HIP free stays at 2.0–3.3 GiB, as in the old fixed-length 8k runs.
     - So something outside torch's allocator is sized by the longest row seen so far and is not given back when a longer row comes.
     - No other process was holding GPU memory: no llama-server, no Ollama model, ComfyUI 0.17 GB.
   - **Fix 1: `ce_target_gb: 2`** (`UNSLOTH_CE_LOSS_TARGET_GB`). A fixed 2 GB chunk budget for the fused CE, instead of half of a free counter that can read 0.
     - Checked against an unfused recomputation (decoder hidden states, LM head in chunks, `torch.nn.functional.cross_entropy`) of logged training losses: step 1 (fresh adapter) 0.61516 vs 0.61516, step 3 (checkpoint-2 weights, 6,620 + 7,087-token rows) 0.59899 vs 0.59899, |diff| < 1e-5 (`tests/ce_check.py`).
   - **Fix 2: `warm_up_longest_row: true`.** At `on_train_begin`, one `Trainer.training_step` on the longest training row, then the gradients are dropped (no optimizer step; dropout 0).
     - The first attempt, a plain autocast forward outside the trainer, autotuned a GatedDeltaNet kernel for a key training never uses and the process aborted in `c10_cuda_check`. Going through `training_step` uses the exact training path.
     - With the warm-up, steps 1–3 give bit-identical losses to the run without it (0.6151649, 0.6087741, 0.5989859). Windows shared memory stays flat at 8.26 GiB on every row (12.2 GiB without it), and HIP free never drops below 1.8 GiB before the loss.
     - The warm-up takes about 2 minutes per (re)start.
   - **Worst case.** The 6 longest rows of the set (8,135–8,188 tokens) run, pause and resume with a peak of 36.4 GiB as Windows counts it (28.1 dedicated + 8.3 shared).
   - **Throughput** in these dry runs: 66–98 tok/s per step (mean about 80). About 9 M tokens in the 2,000-row mix therefore means roughly 30–38 h of training.
8. **Calibration's setup B** is `--spec-type draft-mtp` with the built-in block, the only MTP setting measured with `-np 4`.
