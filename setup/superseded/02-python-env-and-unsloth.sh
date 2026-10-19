#!/usr/bin/env bash
# Python environment with ROCm PyTorch + Unsloth. Run on the EVO-X2 (Linux) after ROCm is installed.
#
# READ BEFORE RUNNING. Two lines here will be out of date by the time you use them:
#   1. The PyTorch ROCm wheel index (rocm7.0 below). Check https://pytorch.org/get-started/locally/
#      and match it to `rocm-smi --showversion`.
#   2. The Unsloth install line. Paste the exact line from https://unsloth.ai/docs/basics/amd.
# When it works, write the versions that worked into docs/Decisions-Log.md.

set -euo pipefail
cd "$(dirname "$0")/.."

python3 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
pip install --upgrade pip

# --- PyTorch (ROCm build). Replace rocm7.0 with the index matching your ROCm. ---
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm7.0

python - <<'PY'
import torch
print("torch", torch.__version__, "| hip", torch.version.hip, "| gpu visible:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
PY

# --- Unsloth + training stack. Replace the first line with the one from the Unsloth AMD docs. ---
pip install unsloth
pip install transformers peft trl datasets accelerate

# bitsandbytes provides the 4-bit quantisation QLoRA relies on. On ROCm it needs the multi-backend build;
# the Unsloth AMD page says which wheel to use. If this plain install fails, do NOT skip it: without
# bitsandbytes you get LoRA on bf16 weights, which needs 3-4x the memory.
pip install bitsandbytes

# --- Tooling that works anywhere ---
pip install -r requirements.txt

python scripts/check_env.py
