# Dataset shortlist

Rule: read the dataset card and its licence field yourself before using anything, then record the licence in `Decision-Log.md`. Apache, MIT and CC-BY are straightforward for academic use. Non-commercial licences (CC-BY-NC) are usable for a dissertation but must be stated explicitly.

## Instruction-tuning candidates (code)

| Dataset (Hugging Face id) | Rows | What it is | Licence (verify on the card) |
|---|---|---|---|
| ise-uiuc/Magicoder-OSS-Instruct-75K | 75k | Problem/solution pairs generated from real open-source snippets (the OSS-Instruct paper) | MIT |
| m-a-p/CodeFeedback-Filtered-Instruction | ~157k | Filtered multi-source code instructions with an emphasis on quality | Apache-2.0 |
| glaiveai/glaive-code-assistant-v3 | ~950k | Synthetic code Q&A across many languages | Apache-2.0 |
| nvidia/OpenCodeInstruct | ~5M | Large synthetic set, many rows carry unit tests | CC-BY-4.0 |
| nvidia/OpenCodeReasoning | ~735k | Competitive-programming problems with reasoning traces | CC-BY-4.0 |
| bigcode/self-oss-instruct-sc2-exec-filter-50k | 50k | Execution-filtered self-instruct data from StarCoder2 | check the card |

Start small. 5k to 10k clean, well-formatted examples beats 500k noisy ones on this hardware, and a run finishes in hours rather than days. Pick one source first; add a second only if the first plateaus.

`scripts/prepare_dataset.py` defaults to Magicoder-OSS-Instruct-75K because its rows have a clean `problem` / `solution` shape and a permissive licence. Change `--dataset` and the field names for the others.

## Contamination check (before any training run)

Benchmark problems leak into public training sets constantly. Before training:

1. Export the HumanEval+, MBPP+ prompts from EvalPlus (`prepare_dataset.py --decontaminate` does this if EvalPlus is installed).
2. Drop any training row whose problem text shares a 13-token window with a benchmark prompt, or whose token-set Jaccard similarity is above 0.8.
3. Record how many rows were removed. That number goes in the dissertation.

If you skip this and a benchmark score jumps, nobody will believe the result, including you.

## Personal evaluation set (held out, never trained on)

20 to 50 tasks that look like real work. The Discord bot repository on this PC is a good source because it is your own code, so no public model has trained on it:

- "Add a per-user cooldown to this command" — give the file, expect a diff.
- "This handler double-pays the winner if two people click at once, fix it" — reintroduce a real bug from git history.
- "Write a test for this function" — pick functions with clear inputs and outputs.
- "Explain what this module does and one thing you'd change" — checks reading comprehension, not just generation.

Score each with a simple rubric: compiles or runs, passes the test, matches the codebase style, no invented APIs. Three points per task, total out of 150 max. Write the tasks before you look at any model output so the set isn't biased toward what models find easy.

Because evaluation runs at the model's full 262k context, include repository-scale tasks: "here is the whole `src/` tree (60–120k tokens), add feature X touching three files", "find the bug somewhere in this codebase". These are the tasks a local assistant is actually for, and no public benchmark measures them. Budget prompt-processing time for them (tens of minutes each at this box's prefill speed).

## Thinking mode is ON (decided 19 September 2026), so the data must carry traces

Qwen3.8-27B thinks by default, answering with a `<think>...</think>` block first, and its benchmark scores are measured that way. Training it on plain question/answer pairs with thinking enabled teaches it to skip the thinking it was RL-tuned to do, and scores drop. So every training row needs a trace, and `prepare_dataset.py --require-think` refuses rows without one.

Where traces come from:

- **Ready-made: `nvidia/OpenCodeReasoning`** (CC-BY-4.0, ~735k problems with long reasoning traces). Competitive-programming flavour, traces often 3k–8k tokens. Check the column names on the card, then `--prompt-field input --response-field output --require-think --max-chars 14000`. Expect the length filter to drop a lot; that is fine.
- **Self-generated with `scripts/generate_traces.py`** (recommended for the codebase-adaptation angle). Run the base model with thinking on over coding prompts, keep its own trace and answer, and with `--run-tests` keep only answers whose code passes the prompt's unit tests (`nvidia/OpenCodeInstruct` ships tests). On-policy data in the model's own style, with a measurable quality filter. Slow: roughly a day per 300 samples at this box's speed, so run it overnight in chunks with `--resume`.
- **Hybrid:** a few hundred self-generated rows on the task types you care about plus a filtered slice of OpenCodeReasoning for volume.

Length rule: the training sequence length is 4096 (Linux, 8-bit; or 4-bit). Rows longer than that are **dropped, not truncated**, a cut-off trace teaches unfinished thinking. `--max-chars 14000` is the approximate line.

Evaluation follows the same rule: the baseline and every fine-tune are scored with thinking on, the same sampling settings (temperature 0.6, top-p 0.95, fixed seeds) and the same generation budget. `scripts/run_eval.sh` sets all of this. Mixing modes between baseline and fine-tune is the fastest way to a meaningless result.

## Forgetting check

Keep a tiny non-coding set, 100 questions from an MMLU-style multiple-choice set or a GSM8K subset, and re-run it after each fine-tune. A drop of more than a couple of points means the training mix is too narrow; add a small slice (5 to 10%) of general instruction data and re-run.

Add a **long-context probe** to the same check: three or four tasks that need a fact or a function buried 64k–128k tokens into a long input. Training examples are 4k–8k tokens, and a fine-tune that only ever sees short examples can dull long-context behaviour even though the context window itself is untouched. If the probe drops, mix in a few long examples or lower the learning rate.
