# Held-out sets for the improvement loop

Built 2026-10-19 16:55:37 by scripts/build_evalsets.py in 777 s.

| Set | Source | Kept |
|---|---|---|
| gate: CodeContests | `deepmind/code_contests` @ `802411c3010cb00d1b05bad57ca77365a3c699d6`, valid 117 + test 165 rows; 217 with stdin/stdout cases and a Python 3 reference; 195 whose reference passes | 195 |
| gate: OpenCodeInstruct | `nvidia/OpenCodeInstruct` @ `8f3ba5bafe4d6e8db46082cf7ae6741bc370604d` `data/train-00025-of-00050.parquet`, 28248 rows with score 1.0 and assert tests; 448 examined, 0 overlap EvalPlus, 134 overlap a prompt in the first run's generation shards (data/train-00000-of-00050.parquet, data/train-00049-of-00050.parquet) | 300 |
| non-coding | `TIGER-Lab/MMLU-Pro` @ `b189ec765aa7ed75c8acfea42df31fdae71f97be` test (12032 rows), outside computer science, random.Random(42) | 100 |

Gate problems dropped for overlap with EvalPlus or data trained on so far: 0.
Non-coding categories: {"philosophy": 7, "law": 12, "business": 7, "chemistry": 15, "psychology": 6, "engineering": 10, "physics": 12, "economics": 5, "math": 5, "biology": 3, "history": 5, "other": 5, "health": 8}.

Overlap rule (anneal/evalsets.py OverlapIndex): a text overlaps a problem if it holds at least 30% of the problem's 13-grams or its word set has Jaccard > 0.8 with the problem's.

Check after the build (read-only, same OverlapIndex over the 495 gate problems): the 390 self rows the first run had kept by then overlap 0 gate problems, and all 200,000 prompts of OpenCodeInstruct shards 00 and 49 (every row, not only score 1.0) overlap 0 gate problems.

The gate files themselves (data/eval/gate.jsonl, noncoding_100.jsonl, decontam_extra.jsonl) stay on the training machine and are not in this repository.
