# Project plan

Written 19 September 2026. Machine ordered, not yet arrived. Assumes roughly October 2026 to May 2027. Put the real faculty deadlines (interim report, presentation, submission) at the top of this file the moment you know them and pull phases earlier to fit.

## Phase 0: before the hardware arrives (now)

Goal: arrive at day one with the reading done and provisional decisions made.

- Read the LoRA and QLoRA papers properly. Skim Code Llama and Magicoder for how people build code training sets.
- Shortlist a dataset and decide thinking mode. Write the leaning and the reasons in `Decisions-Log.md`.
- Python environment on this PC. Run `scripts/prepare_dataset.py` on a small sample and read twenty real examples. You will learn more from that than from any paper about what "instruction data" is.
- Draft the personal evaluation set: 20 to 50 realistic tasks (see `Dataset-Candidates.md`). Do this before seeing any model output so it isn't biased by what models find easy.
- Run `scripts/memory_estimate.py --preset qwen3.8-27b` at 8-bit and 4-bit and keep the printout.

Exit: `Decisions-Log.md` has a provisional dataset and OS choice with reasons.

## Phase 1: bring-up (weeks 1 to 3 after arrival)

- Install the OS, ROCm, PyTorch, Unsloth, llama.cpp and EvalPlus (`setup/`).
- Run `scripts/check_env.py`. Copy the real numbers into `Hardware-Notes.md`: usable GPU memory, bandwidth from `llama-bench`, free disk.
- Run `scripts/measure_memory.py` on Qwen3.8-27B at 8-bit and 4-bit at seq 4096 (and 3072 if 8-bit fails), and once at bf16 expecting it to fail. Those numbers decide the precision for the whole project, replace every estimate in this folder, and go in the dissertation.
- Confirm `llama-server --reasoning-format deepseek` splits the trace from the answer on this build (one request, check the JSON has `reasoning_content`). `run_eval.sh` and `generate_traces.py` depend on it.
- Start `llama-server` at the full context (`-c 262144 -ctk q8_0 -ctv q8_0`), note the memory it reports for the KV cache, and time a 100k-token prompt. That sets the budget for repository-scale tasks in the personal evaluation set.
- If the machine starts on Windows: run `measure_memory.py` at 8-bit / seq 4096 with the dedicated pool at 48GB. If it succeeds above 48GB, training can use Windows' shared memory and the OS decision reopens; if it fails, that is the measured reason for Linux. Either way, record it.
- Pull a Qwen3.8-27B GGUF (Q8_0; Unsloth and Bartowski publish them) for inference benchmarks and the baseline evaluation. No conversion needed.
- Inference benchmarks: tokens per second, Vulkan vs ROCm, prompt processing speed.
- Load the model, generate a completion, and run one HumanEval+ problem end to end.

Exit: the full inference path works and the memory ceiling is measured, not assumed.

## Phase 2: baselines (weeks 3 to 6)

- Calibrate first: `LIMIT=20 scripts/run_eval.sh calib <gguf>` on the base model. It prints tokens per problem, aggregate tok/s, and the projected hours for the full 542 problems. Run it once with MTP (default) and once with `NP=4 MTP=0`; keep whichever is faster. Measured speeds for this model on this chip are in `Hardware-Notes.md` (Q8 with MTP ~15 tok/s, Q4 with MTP ~24 tok/s); the projection depends on trace length, which only the calibration run knows.
- Full HumanEval+ and MBPP+ runs on the untouched base model, thinking on, reasoning budget 32k, at the quantisation the fine-tuned exports will use, with `run_eval.sh` (temperature 0.6, top-p 0.95, fixed seed). Three seeds for the baseline, since it anchors every later comparison.
- Run the personal task set and the forgetting check on it, same settings.
- Write a draft "baseline" section while the numbers are fresh.

Exit: baseline numbers with spread recorded for every evaluation the fine-tunes will be scored on.

## Phase 3: data (weeks 6 to 10)

- Thinking is on, so every row needs a `<think>` trace. Start `scripts/generate_traces.py` early in this phase (it runs overnight for days) and, in parallel, pull and length-filter `nvidia/OpenCodeReasoning`. Verify licences. Record them.
- Clean, deduplicate, require traces, drop rows over the 4096-token budget, and check for contamination against the benchmark prompts (`prepare_dataset.py --require-think --max-chars 14000 --decontaminate`). Record how many rows each filter removed.
- Hold out a validation split.
- Decide the training mix: self-generated only, OpenCodeReasoning only, or hybrid. One choice for the first real run; the others are later experiments.

Exit: `data/processed/train.jsonl` and `val.jsonl` where every row has a trace, plus a short written data card.

## Phase 4: first fine-tune (weeks 10 to 14, over the winter break)

- Smoke test on the training target (`TEMPLATE-qwen3.8-27b-8bit.yaml` by default): 200 samples, 30 steps. Proves train, merge, export and evaluate work as one pipeline. Fix everything that breaks here, this is where the ROCm surprises will show up. If 8-bit misbehaves on ROCm, the same template with `load_in_4bit: true` is the fallback.
- First real run: 2k to 3k trace rows, rank 16, sequence length 4096 (8-bit on Linux if `measure_memory.py` confirmed it fits, else 4-bit). Evaluate with `run_eval.sh`, one seed. Compare to the baseline.

Exit: one fine-tuned model and a before/after table, however small the difference.

## Phase 5: iterate (January to March)

- One variable per run: rank, target modules, data mix, epochs, sequence length.
- Re-run the small non-coding check after each run to watch for forgetting.
- Interim report or progress presentation, whatever the faculty requires.

Exit: four to eight logged runs and a clear story about what mattered.

## Phase 6: stretch (March to April)

Same model, pushed further. Pick one or two of these, not all:
- Longer context training (seq 2048 or 4096 on Linux) so the model sees whole files, not fragments.
- A second dataset mix: coding data plus a small slice of general instructions, or reasoning-trace data with thinking on.
- Higher rank (32 or 64) or adapters on a different set of modules, if the Phase 5 runs suggest the fine-tune is capacity-limited.
- A larger personal task set (100+ tasks) to tighten the error bars on the headline result.

Exit: one result the Phase 5 runs could not have produced, or a measured reason the attempt did not fit. Both are fine outcomes for the write-up.

## Phase 7: write-up (April to May)

- Clean this repository into the reproducibility package.
- Final evaluation tables. Dissertation.

## Risks and responses

| Risk | How you'll notice | What to do |
|---|---|---|
| ROCm or Unsloth breaks on gfx1151 | Import errors, NaN loss, hangs, absurd speeds | Pin versions that work and log them. Try Unsloth Studio, or plain PEFT + bitsandbytes. Check the strix-halo toolboxes repo for known-good builds |
| Training too slow | A run projects past 24 hours | Shorter sequences, fewer samples, smaller model. Gradient accumulation instead of batch size |
| Out of memory | OOM at load or first step | Lower sequence length, rank, or model size. Confirm BIOS UMA and Linux GTT settings. Linux over Windows |
| No measurable improvement | Delta inside run-to-run noise | Check the chat template matches the model. More data, higher rank. Evaluate with multiple seeds before concluding |
| Disk full | Download or checkpoint fails | `cleanup_intermediates.py`. External NVMe |
| Benchmark contamination | Suspiciously large jump on one benchmark only | Run the decontamination check. Add newer problems (LiveCodeBench-style) the training data can't have seen |
| Scope creep | Wanting a second model "just for comparison" in February | Re-read the agreement: one model. Compare configurations of the same model instead |

## Weekly routine

- Monday: write one question this week should answer at the top of `experiments/LOG.md`.
- Every run: YAML and LOG row before it starts, results row after.
- Friday: one paragraph in `Decisions-Log.md`, what was learned and what changed.
