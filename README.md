# Anneal

Workspace for the dissertation project: fine-tuning an open-weight coding model on a GMKtec EVO-X2 (Ryzen AI Max+ 395, 64GB unified memory, 1TB NVMe) and measuring the before/after difference properly.

Everything here was scaffolded before the machine arrived. Nothing under `scripts/` has been run on the real hardware yet. Treat each script as a starting point to validate and fix, not a finished tool. Where a command depends on a version that will have moved by the time you install (ROCm, Unsloth), the file says so and points at the page to check.

## Folder map

| Folder | What lives there |
|---|---|
| `docs/` | Project brief, year plan, hardware notes, model and dataset shortlists, reading list, decisions log |
| `setup/` | Install steps for the EVO-X2: Linux + ROCm path, plus a Windows quick start for inference on day one |
| `scripts/` | Environment check, memory estimator, model download, dataset prep, QLoRA training template, GGUF export, evaluation runner, cleanup |
| `experiments/` | One YAML per training run + `LOG.md`, the lab notebook |
| `data/`, `models/`, `results/` | Large files. Git-ignored except their READMEs |

## Start here

**Before the machine arrives**
1. Read `docs/Plan.md`, then start `docs/Reading-List.md` with the LoRA and QLoRA papers.
2. Make a Hugging Face account and an access token. Accept the licences on the candidate models so downloads work later.
3. Read `docs/Model-Candidates.md` and `docs/Dataset-Candidates.md`. Write your leaning in `docs/Decisions-Log.md`.
4. On this PC, set up a Python environment and look at real training data:

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python scripts/prepare_dataset.py --max-samples 200
```

5. Decide Windows vs Linux for the EVO-X2 (see `docs/Hardware-Notes.md`). Log the decision.

**When it arrives**
1. Follow `setup/01-linux-and-rocm.md`, or `setup/windows-quickstart.md` if you want inference working within the hour.
2. Run `python scripts/check_env.py` and copy the numbers into `docs/Hardware-Notes.md`.
3. Pull the model. The bf16 safetensors are what training loads; a published Q8_0 GGUF is the quickest way to benchmark and baseline it:

```
python scripts/download_model.py Qwen/Qwen3.8-27B                                      # 55.6GB bf16, the one base model
python scripts/download_model.py unsloth/Qwen3.8-27B-GGUF --include "*Q8_0*"           # baseline + inference speed (check the repo name)
```

   Baseline it with `scripts/run_eval.sh` before any training, in the thinking mode you will train with.
4. Run `python scripts/train_qlora.py --config experiments/TEMPLATE-qwen3.8-27b-8bit.yaml --smoke-test` to prove train, merge, export and eval work end to end before any real run.

## House rules

- Every training run gets a YAML in `experiments/` and a row in `experiments/LOG.md` before it starts.
- Delete full-precision weights once a verified GGUF exists. `scripts/cleanup_intermediates.py --dry-run` shows what would go. 1TB disappears fast.
- Log decisions and the reasons in `docs/Decisions-Log.md`. The dissertation write-up is mostly this file, expanded.
- Never look at held-out test tasks while building training data.
- Thinking mode is ON everywhere: training rows carry `<think>` traces, and baseline and fine-tunes are evaluated with the same sampling settings and seeds. Never compare a thinking run against a non-thinking one.
- Inference and evaluation run at the model's full 262k context (`run_eval.sh` does this). Training examples are 4k tokens; that is a different knob and does not shrink the context window.
- One variable per experiment. If two things changed, you learned nothing.
