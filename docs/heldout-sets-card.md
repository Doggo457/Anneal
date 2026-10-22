# Held-out sets for the improvement loop

The loop keeps a round only if it does better on these sets, and not worse on the no-worse checks. They are built
once and never change while rounds are scored on them. The data files stay on the training machine and are not in
this repository.

## The gate: six languages, write and fix

Built 2026-10-21 05:24:57 (UK time) by scripts/build_gate.py in 2277 s. No Python: Python is the lowest priority
(21 Oct 2026), so the earlier Python gate (CodeContests + OpenCodeInstruct, built 19 Oct) is archived in
data/eval/archive/python-gate/ and scores on it are not comparable with this one.

| Step | Count |
|---|---|
| CodeContests `deepmind/code_contests` @ `802411c3010cb00d1b05bad57ca77365a3c699d6` valid + test rows | 282 |
| with stdin/stdout cases, no input/output files | 282 |
| dropped for overlap with EvalPlus or 21035 trained or pooled rows | 1 |
| no C++ or Java reference passes our runner | 16 |
| kept problems (a reference passes: C++ 258, Java 223) | 265 |
| wrong submissions tried (C++ / Java) | 601 / 362 |
| problems with a qualifying wrong submission (C++ / Java) | 189 / 145 |

Gate: 495 items: 286 write (21 of them a problem's second language), fix C++ 105, fix Java 104.

| Language | Write | Fix | All |
|---|---|---|---|
| C++ | 48 | 105 | 153 |
| Java | 47 | 104 | 151 |
| Go | 47 | 0 | 47 |
| Rust | 48 | 0 | 48 |
| JavaScript | 48 | 0 | 48 |
| TypeScript | 48 | 0 | 48 |

A write item asks for a complete program in one language. A fix item shows a real wrong C++ or Java submission to the
same problem and asks for the corrected program. A wrong submission qualifies only if, in our sandbox, it compiles,
passes every example case, gets a wrong answer or crashes on another case and never times out, uses no file input or
output, and (in C++) uses no plain `long`.

Toolchains (sandbox/toolchains/lock.json): node: v24.22.0; typescript: TypeScript 6.0.3; rust: rustc 1.99.1 (58cf8f9c3
2026-10-13); go: go1.27.3 windows/amd64; jdk: javac 25.0.4.1; gcc: g++ (MinGW-W64 x86_64-ucrt-posix-seh) 16.2.0.
Toolchain fingerprint c95ebf95513f3d97.

Judging (anneal/polyglot.py): the last full program in the answer's code blocks for the item's language is built in
the build container and run in the code sandbox on up to 30 cases (examples first), 10 s each; output compared token
by token, numbers within 1e-6.

decontam_extra.jsonl (problems later training data is checked against): 72 new problems added, 667 in all.

## Held-out bug-fix steps (the loss rule)

Built 2026-10-21 06:05:20 (UK time) by scripts/build_swe_heldout.py from `nvidia/Open-SWE-Traces` @
`1db674173119d71df45c22007af99e214600711a`, with every filter of the openswe_v4flash pool (DeepSeek V4-Flash-0731
runs that resolved their issue, outside Python, each step with the teacher's reasoning, at most 8,192 tokens,
decontaminated against EvalPlus and the gate), but only repositories whose name hashes to bucket 0 of 20. No round
ever trains on those repositories.

| Step | Count |
|---|---|
| runs in the dataset | 50295 |
| left out by the pool's filters (unresolved, or in Python) | 45620 |
| in the repositories rounds train on | 4482 |
| held-out runs | 193 |
| their steps (candidate rows) | 7053 |
| no teacher reasoning | 1749 |
| duplicate | 85 |
| over 8,192 tokens | 4690 |
| more than one step from the same run | 352 |
| kept (one step per run) | 177 |

Kept runs by language: TypeScript 69, JavaScript 45, Go 32, Java 28, Rust 18, PHP 1 (of 193 runs read).
Tokens per step: min 2512, median 5688, p90 7793, max 8183; 2,907,375 in all. 177 is every held-out run with a step
that passes the filters, so the set is smaller than the 200 planned.

## Held-out terminal steps (the loss rule)

Built 2026-10-21 22:18:43 (UK time) by `scripts/build_swe_heldout.py --set terminal` from
`open-athena/recursive-task-synthesis-glm-5.3-rollouts` @ `dd6f34cbcb8596112c6d12bed53b5a576973b8f7` (GLM-5.3
Terminus-2 runs on Linux terminal tasks, CC-BY-4.0), with every filter of the glm53_all_nopy_12k pool (test-passed
runs, the teacher's reasoning, parsed replies only, near-duplicate steps dropped, decontaminated against EvalPlus and
the gate, rows up to 12,288 tokens), but only task groups whose id hashes to bucket 0 of 20. No round ever trains on
those task groups. The six-language gate can't see terminal skill, so a round can also be kept on a lower loss here.

| Step | Count |
|---|---|
| test-passed runs in the dataset | 12124 |
| runs that run or write Python | 6584 |
| non-Python runs in the task groups rounds train on | 5240 |
| held-out non-Python runs | 300 |
| beyond the best 3 runs of a task | 80 |
| their steps (candidate rows) | 1551 |
| replies Terminus-2 couldn't parse (context only) | 18 |
| near duplicate | 5 |
| over 12,288 tokens | 549 |
| more than one step from the same run | 759 |
| kept (one step per run) | 220 |
| in the set (a seeded draw) | 200 |

Tokens per step of the 220 kept, before the draw: min 1056, median 4772, p90 10591, max 12288; 5,262,263 in all.

## Non-coding check

Built 2026-10-19 by scripts/build_evalsets.py and unchanged: `TIGER-Lab/MMLU-Pro` @
`b189ec765aa7ed75c8acfea42df31fdae71f97be` test (12032 rows), 100 questions outside computer science,
random.Random(42). Categories: {"philosophy": 7, "law": 12, "business": 7, "chemistry": 15, "psychology": 6,
"engineering": 10, "physics": 12, "economics": 5, "math": 5, "biology": 3, "history": 5, "other": 5, "health": 8}.

## BFCL Java and JavaScript tool calls (no-worse check)

Built by scripts/build_guards.py --bfcl from the Berkeley Function Calling Leaderboard's own package, bfcl-eval
2026.3.23 on PyPI (wheel sha256 `3bb6dfa5f0c68ad403c9ec50b00db2bb3b4cc9b38ab1ff33f48fe30d853d3a0a`; the same code as
`ShishirPatil/gorilla` @ `6ea57973c7a6097fd7c5915698c54c17c5b1b6c8`). Code and data are Apache-2.0. Not the Hugging Face
copy: it holds only the V3 files, which still expect `queue` in `simple_javascript_41`, fixed to `queue_1` on 12 Dec
2025.

| Category | Items |
|---|---|
| simple_java (BFCL V4) | 100 |
| simple_javascript (BFCL V4) | 50 |

Each item is one question, one function and one expected call. These two are BFCL's only Java and JavaScript
categories. The request is the question as the only message and the function as a tool, built as BFCL builds it for
OpenAI-compatible servers: BFCL's language hint on every description, every parameter a string, and dots in names
turned into underscores. The rebuilt requests match BFCL's own convert_to_tool output for all 150 items; the file's
sha256 is `f69b618c3639563895c25942b16c101ca941ee5a10e48e567dd23452f09537ea`.

Answers come from the six-language test's server settings with thinking on, one per item (seed 1). llama-server
parses the call, and BFCL's own AST checker judges it against the raw function doc (anneal/bfcl.py). An answer with no
call, a cut-off answer or arguments that aren't JSON fail.

Known checker limits apply to the base and every round alike. A Java `String[]` with quoted elements fails (java 35,
78, 83, 85, 86, 89, 90, 91), and an ArrayList of mixed types can't pass (java 36, 65). Scores are not comparable with
the public leaderboard, which uses other quantisation, sampling and limits, and averages in Python items.

The loop's pools were prepared before this check, so they were not decontaminated against it. scripts/build_guards.py
reports how many pool rows have a first user message that overlaps a BFCL question by the rule below.

## CWEval Go and JavaScript secure coding (no-worse check)

Built by scripts/build_guards.py --cweval from CWEval (`Co1lin/CWEval` @ `e9a2a124c8c53679b6d8d27adfd2f6c40e7576d7`,
Apache-2.0; Peng et al., "CWEval: Outcome-driven Evaluation on Functionality and Security of LLM Code Generation",
LLM4Code 2025). Every file it uses (CWEval's own files, the Go modules, npm packages, the recheck binary and the
test-side Python wheels) is pinned by sha256 in anneal/cweval_lock.json and fetched once; nothing is fetched while
judging. The rows file's sha256 is `d4f3580e86ca2c9394f9c9ff96d36acc862d0cbd7d4690baed81e44b6954d20b`.

| Language | Tasks | Left out |
|---|---|---|
| Go | 14 | 078_0 (runs `ls`), 732_2 (POSIX permission bits), 943_0 (sqlite needs cgo), 326_1 (reference sometimes over the 3 s limit), 022_0 (its reference's symlink lookup can't run in the sandbox) |
| JavaScript | 18 | 078_0, 732_2, 943_0 (native sqlite addon), 326_0 (reference too slow for the 3 s limit), 502_0 (its security test can't fail with the js-yaml CWEval installs) |

Each task asks for one function with CWEval's own "direct" prompt. The answer's first code block (or the whole answer)
gets the task's entry point appended, as CWEval does; Go code goes through goimports. The answer passes only if every
functionality test and every security test of the task passes, each within 3 s (CWEval's func-sec@1). An answer that
didn't finish fails. Go is built in the build container and every program the tests start runs in the code sandbox;
CWEval's pinned pytest files run in a separate Python with only the pinned test libraries, and may start only the
task's own program (or recheck, for the ReDoS task). A sandbox that refuses to start a program is retried and never
counts as a failed test, and an answer that fails only on time is judged once more, for base and adapter alike. Before
it was used, the check passed its self-test on the training PC on 22 Oct 2026: all 32 references pass and all 32
insecure solutions fail security. No training row in the loop's 8 pools is an exact or near copy of a task (the
overlap rules below), checked the same day.

Answers come from the six-language test's server settings with thinking on, one per task (seed 1), like BFCL. With
32 tasks it only catches large drops: a round is blocked when it is more than 12 points (about 4 tasks) below the base,
or significantly below by a sign test.

## Overlap rules

Exact and near copies (anneal/evalsets.py OverlapIndex): a text overlaps a problem if it holds at least 30% of the
problem's 13-grams or its word set has Jaccard > 0.8 with the problem's.

Reworded copies (scripts/paraphrase_check.py, embeddings on the NPU): every training question is compared with each
gate problem's statement; rows closer to a problem than that set's calibrated cut-off are blocked. No round trains
until this check has covered the current gate and every usable pool (experiments/loop.yaml reworded_check_required).
