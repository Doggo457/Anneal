#!/usr/bin/env bash
# Evaluate a GGUF on HumanEval+ and MBPP+ via a local llama-server: thinking ON, reasoning budget 32k,
# full 262k context, MTP speculative decoding on by default.
#
# Usage:
#   scripts/run_eval.sh <name> <model.gguf> [seed]
#   LIMIT=20 scripts/run_eval.sh calib <model.gguf>          # first 20 problems: tokens/problem + projected hours
#   PORT=1234 scripts/run_eval.sh <name> external            # use an already-running server
#   NP=4 MTP=0 scripts/run_eval.sh <name> <model.gguf>       # parallel streams instead of MTP
#   MTP_GGUF=models/gguf/mtp-Qwen3.8-27B-Q4_0.gguf scripts/run_eval.sh ...   # MTP head shipped as a separate file
#   LORA=models/<run>/adapter.gguf scripts/run_eval.sh <name> <base.gguf>  # adapter loaded at runtime on the same base
#
# Results land in results/<name>/: samples, sanitized samples, EvalPlus logs, and *_stats.jsonl with
# tokens and seconds per problem. Name base and fine-tuned runs to pair up, with the seed:
#   qwen38-27b_base_s1   and   qwen38-27b_r16-traces3k_s1
#
# Settings and why (all measured on Ryzen AI Max+ 395, see docs/Hardware-Notes.md):
#   - MTP speculative decoding (--spec-type draft-mtp) roughly doubles generation on this chip
#     (Q8: 7.2 -> 15.3 tok/s; Q4_K_M: 12.3 -> ~24.5). Draft acceptance drops as temperature rises and
#     MTP was measured counterproductive with parallel requests, so it is enabled only when NP=1.
#   - Reasoning budget 32,768 (project decision). MAX_TOKENS is 40,960 so the answer always has room.
#   - Sampling temperature 0.6 / top-p 0.95 is Qwen's recommendation for thinking mode; greedy loops.
#     Sampling adds variance: one seed while iterating, three seeds for the headline before/after.
#   - Context 262,144 with a q8 KV cache (8GB). With NP>1 llama-server splits the context across slots
#     (262144/4 = 65k each), still far above what a benchmark problem needs (~41k worst case).
#
# EvalPlus CLI flags change between versions: `evalplus.sanitize --help`, `evalplus.evaluate --help`.
set -euo pipefail

NAME=${1:?name}
GGUF=${2:?path to .gguf, or 'external'}
SEED=${3:-1}
PORT=${PORT:-8080}
CTX=${CTX:-262144}
KV_TYPE=${KV_TYPE:-q8_0}
NP=${NP:-1}
MTP=${MTP:-1}
REASONING_BUDGET=${REASONING_BUDGET:-32768}
MAX_TOKENS=${MAX_TOKENS:-40960}
TEMP=${TEMP:-0.6}
TOP_P=${TOP_P:-0.95}
LIMIT=${LIMIT:-0}
LORA=${LORA:-}
MTP_GGUF=${MTP_GGUF:-}
OUT="results/$NAME"
mkdir -p "$OUT"

SPEC=()
if [ "$MTP" = "1" ] && [ "$NP" = "1" ]; then
  SPEC=(--spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-p-min 0.0)
  # Unsloth's GGUF repo ships the MTP head as its own file; llama-server takes it as the draft model.
  if [ -n "$MTP_GGUF" ]; then SPEC+=(-md "$MTP_GGUF"); fi
fi
EXTRA=()
if [ -n "$LORA" ]; then EXTRA+=(--lora "$LORA"); fi

if [ "$GGUF" != "external" ]; then
  llama-server -m "$GGUF" -c "$CTX" -np "$NP" -ngl 999 --host 127.0.0.1 --port "$PORT" -fa on \
    -ctk "$KV_TYPE" -ctv "$KV_TYPE" \
    --reasoning-format deepseek --reasoning-budget "$REASONING_BUDGET" --seed "$SEED" \
    "${SPEC[@]}" "${EXTRA[@]}" > "$OUT/server.log" 2>&1 &
  SERVER=$!
  trap 'kill $SERVER 2>/dev/null || true' EXIT
  echo "waiting for llama-server on :$PORT ..."
  for _ in $(seq 1 180); do
    curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break
    sleep 2
  done
fi

{
  echo "name: $NAME"; echo "model: $GGUF"; echo "seed: $SEED"; echo "ctx: $CTX"; echo "kv_cache: $KV_TYPE"
  echo "lora: ${LORA:-none}"; echo "parallel: $NP"; echo "mtp: ${SPEC[*]:-off}"; echo "thinking: on"; echo "reasoning_budget: $REASONING_BUDGET"
  echo "max_tokens: $MAX_TOKENS"; echo "sampling: temperature $TEMP top_p $TOP_P"; echo "limit: $LIMIT"
  echo "date: $(date -Iseconds)"
  command -v llama-server >/dev/null && llama-server --version 2>&1 | head -1
} > "$OUT/run_info.txt"

for DS in humaneval mbpp; do
  echo "=== $DS (seed $SEED) ==="
  python scripts/gen_samples.py --dataset "$DS" --url "http://127.0.0.1:$PORT" --workers "$NP" \
    --max-tokens "$MAX_TOKENS" --temperature "$TEMP" --top-p "$TOP_P" --seed "$SEED" --limit "$LIMIT" \
    --out "$OUT/${DS}_samples.jsonl" --resume 2>&1 | tee "$OUT/${DS}_gen.log"
  evalplus.sanitize --samples "$OUT/${DS}_samples.jsonl"
  if [ "$LIMIT" = "0" ]; then
    evalplus.evaluate --dataset "$DS" --samples "$OUT/${DS}_samples-sanitized.jsonl" 2>&1 | tee "$OUT/$DS.log"
  else
    # evalplus.evaluate asserts that every problem has a sample, so a LIMIT run stops at generation.
    echo "LIMIT=$LIMIT: skipped evalplus.evaluate (it needs all problems). Timing is in $OUT/${DS}_stats.jsonl"
  fi
done

echo
echo "pass@1 (base and plus) are in $OUT/*.log; tokens and seconds per problem in $OUT/*_stats.jsonl."
echo "Copy the scores, the seed, and the mean tokens/problem into experiments/LOG.md."
