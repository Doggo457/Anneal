# data/

Git-ignored except this file.

- `raw/` — anything downloaded as-is (dataset dumps, benchmark exports). Never edited by hand.
- `processed/` — output of `scripts/prepare_dataset.py`: `train.jsonl`, `val.jsonl`, `data_card.md`.
- `own-eval/` — the personal evaluation task set. Written by hand, never used for training. Keep a `tasks.md` (task, expected behaviour, rubric) and one folder per task with the input files.

The data card is part of the dissertation: it records source, licence, filters, and how many rows each filter removed.
