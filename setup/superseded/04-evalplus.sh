#!/usr/bin/env bash
# Install EvalPlus (HumanEval+ / MBPP+) and pull the benchmark data so the first eval run isn't a download.
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck disable=SC1091
source .venv/bin/activate 2>/dev/null || true

pip install --upgrade evalplus

python - <<'PY'
from evalplus.data import get_human_eval_plus, get_mbpp_plus
he, mb = get_human_eval_plus(), get_mbpp_plus()
print(f"HumanEval+ problems: {len(he)}   MBPP+ problems: {len(mb)}")
print("Example prompt:\n", next(iter(he.values()))["prompt"][:400])
PY

echo
echo "CLI flags change between versions. Check the current ones with:"
echo "  evalplus.evaluate --help"
echo "Then run scripts/run_eval.sh <name> <model.gguf>"
