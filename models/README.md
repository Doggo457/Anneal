# models/

Git-ignored except this file. Layout:

- `<org>__<name>/` — downloaded base models (`scripts/download_model.py`). Delete when finished with that base.
- `<run_name>/adapter/` — the LoRA adapter from a training run. Small. Keep forever.
- `<run_name>/checkpoints/` — intermediate trainer checkpoints. Delete after the adapter is saved.
- `<run_name>/merged-bf16/` — base + adapter merged at full precision. Delete once the GGUF is verified.
- `<run_name>/gguf/` — quantised exports for llama.cpp. Keep the one you evaluated.
- `<run_name>/train_summary.json` — config, wall time, peak memory, final eval loss.

`python scripts/cleanup_intermediates.py --dry-run` lists what can go.
