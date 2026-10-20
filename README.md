# Anneal

Honours dissertation project (University of Stirling): train a 4-bit QLoRA adapter for Qwen3.8-27B on public and self-generated coding data, and measure properly whether it makes the model better at software development. Everything runs on one machine, a GMKtec EVO-X2 (Ryzen AI Max+ 395, 64 GB unified memory, 32 GB given to the GPU), natively on Windows.

`Decision-Log.md` is the dated record of every decision and measured result, newest at the bottom. Start there for the current state.

## Folder map

| Path | What lives there |
|---|---|
| `anneal/` | Package the scripts import: no-admin AppContainer sandbox for model-written code, shift-and-add DeltaNet short conv, runtime and data helpers, llama-server helpers |
| `scripts/` | The pipeline: trace generation, dataset prep and mixing, training (`train_longrun.py`), the Windows EvalPlus port (`evalplus_win.py`), the first-run controller (`first_run.py`), the keep-if-better improvement loop (`loop.py`, its held-out sets `build_evalsets.py`, public chat-format sources `public_source.py`, the restart watchdog `loop_watchdog.py` and `loop_status.py`), validation loss, memory measurement |
| `experiments/` | One YAML per run, plus `LOG.md` |
| `tests/` | The check scripts behind each finding (parity with llama.cpp, short conv, MTP, sandbox, resume, LoRA serving) |
| `results/` | `*_findings.md` write-ups and the summaries and metrics behind them. Weights, checkpoints and raw logs stay on the machine |
| `docs/` | Project brief, plan, hardware notes, model and dataset shortlists, reading list, and the card for the held-out evaluation sets |
| `setup/` | How the EVO-X2 is set up for training on Windows, and how the loop restarts itself (`loop-watchdog.md`). `superseded/` holds the pre-arrival Linux notes |
| `dissertation/` | Dissertation outline (Word) and its figures |
| `data/`, `models/` | Git-ignored except their READMEs |

## Where things run

The repo mirrors `C:\AI\training` on the EVO-X2, which has its own `.venv`. Training loads `unsloth/Qwen3.8-27B-unsloth-bnb-4bit`; testing uses the Ollama Q4_K_M GGUF in llama-server (Vulkan build) with the adapter converted to a GGUF LoRA. `setup/windows-training-setup.md` has versions, layout and the measured numbers.

`scripts/check_env.py`, `train_qlora.py`, `export_gguf.py`, `run_eval.sh`, `gen_samples.py`, `memory_estimate.py` and `cleanup_intermediates.py` are the scaffolding written before the machine arrived. They are not on the machine; the runs use `train_longrun.py`, `first_run.py` and `evalplus_win.py`.

## House rules

- Every training run gets a YAML in `experiments/` before it starts.
- Every decision and measured result gets a dated row in `Decision-Log.md`.
- Never look at held-out test tasks while building training data.
- Thinking mode is on everywhere: training rows carry `<think>` traces, and baseline and fine-tunes are evaluated with the same sampling settings and seeds.
- Inference and evaluation run at the full 262k context. Training examples are up to 8,192 tokens; that is a different knob and does not shrink the context window.
- One variable per experiment.
- No model weights, datasets or secrets in git.
