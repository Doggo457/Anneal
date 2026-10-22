"""What a stop-thinking sentence does at the gate's thinking budget: answer chosen gate items again on a llama-server
with the loop's gate settings plus --reasoning-budget-message (loop.yaml gate.budget_message), with the gate pass's own
request seeds, and report for each answer whether thinking reached the budget, whether the sentence was added (the end
of the returned reasoning), whether the answer holds a finished code block, its finish reason and the gate's verdict.

The default sentence is the one Qwen's documentation uses to end Qwen3's thinking at a budget ("Thinking Budget" in
qwen.readthedocs.io's quickstart): two newlines, "Considering the limited time by the user, I have to give the solution
based on the thinking directly now.", a newline, then the end-of-thinking tag and two newlines. llama-server adds the
end-of-thinking tag itself, so the sentence here stops before it. It is passed with real newlines.

The loop must be paused (nothing may answer on its port 18090: the GPU has to be free). Items: --ids, or every row of
--rows (a gate.seed*.jsonl) with finish reason length. Writes results/budget-message/LABEL.jsonl (the gate's row
format, with this check's own fields) and LABEL.json (the summary).

  python tests/budget_message_check.py --rows results/loop/rounds/r000/gate.seed1.jsonl
  python tests/budget_message_check.py --ids write/go/cc/test/40 --no-message --label control
"""

import argparse
import json
import os
import sys
import threading
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import loop as L  # noqa: E402
from anneal import evalsets as E  # noqa: E402
from anneal import llama, sandbox  # noqa: E402
import evalplus_win as ew  # noqa: E402

PORT = 18095
LOOP_PORT = 18090
OUT_DIR = os.path.join(ROOT, "results", "budget-message")
QWEN_STOP = ("\n\nConsidering the limited time by the user, I have to give the solution based on the thinking "
             "directly now.\n")
TAIL = 200


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default=L.CONFIG)
    ap.add_argument("--rows", default=None, help="a gate.seed*.jsonl: its rows that ran out (finish reason length)")
    ap.add_argument("--ids", nargs="*", default=None, help="gate item ids (instead of --rows)")
    ap.add_argument("--message", default=QWEN_STOP, help="the sentence (default: Qwen's)")
    ap.add_argument("--no-message", action="store_true", help="no sentence: the gate's settings as they are")
    ap.add_argument("--budget", type=int, default=None, help="thinking budget (default: loop.yaml gate.budget)")
    ap.add_argument("--max-tokens", type=int, default=None, help="tokens an answer (default: loop.yaml)")
    ap.add_argument("--label", default=None)
    return ap.parse_args()


def make_check():
    """The gate's own check(item, content) -> (ok, verdict), as the loop judges a gate answer."""
    return L.Loop.gate_checker(None)


def code_block(content):
    """(fence lines, whether the last fence is closed) of a Markdown answer: an even count of 2 or more."""
    fences = [ln for ln in (content or "").splitlines() if ln.lstrip().startswith("```")]
    return len(fences), len(fences) >= 2 and len(fences) % 2 == 0


def sentence_seen(reasoning, message):
    """Whether the returned reasoning ends with the sentence (whitespace aside)."""
    return bool(message) and " ".join((reasoning or "").split()).endswith(" ".join(message.split()))


def main():
    args = parse_args()
    try:
        llama.http(f"http://127.0.0.1:{LOOP_PORT}/health", timeout=5)
        raise SystemExit(f"something answers on port {LOOP_PORT} (the loop's scoring server): pause the loop first")
    except SystemExit:
        raise
    except Exception:
        pass
    cfg = L.load_config(args.config)
    g = dict(cfg["gate"])
    if args.budget:
        g["budget"] = args.budget
    if args.max_tokens:
        g["max_tokens"] = args.max_tokens
    g["budget_message"] = None if args.no_message else args.message
    cfg["gate"] = g
    message = L.gate_budget_message(cfg)
    flags = L.Loop.gate_flags(None, cfg)
    binary = llama.server_binary(L.gate_build(cfg))
    if not os.path.exists(binary):
        raise SystemExit(f"{binary} is missing")
    gate = E.read_jsonl(E.GATE_PATH)
    index = {it["id"]: i for i, it in enumerate(gate)}
    if args.ids:
        ids = list(args.ids)
    elif args.rows:
        ids = [r["id"] for r in E.read_jsonl(args.rows) if r.get("finish_reason") == "length"]
    else:
        raise SystemExit("give --rows or --ids")
    missing = [i for i in ids if i not in index]
    if missing or not ids:
        raise SystemExit(f"not gate items: {missing}" if missing else "no items to answer")
    order = [index[i] for i in ids]
    seed = int(g["seed"])
    label = args.label or (("no-message" if message is None else "message") + time.strftime("-%Y%m%d-%H%M%S"))
    os.makedirs(OUT_DIR, exist_ok=True)
    out_rows = os.path.join(OUT_DIR, f"{label}.jsonl")
    if os.name == "nt":  # the answers' programs run in the code sandbox, as in a gate pass
        sandbox.ensure_private_folder_closed(print)
    srv = llama.Server(flags, os.path.join(OUT_DIR, f"{label}.server.log"), port=PORT, binary=binary)
    print("llama-server:", srv.describe(), flush=True)
    srv.start()
    print(f"up in {srv.load_s} s; {len(order)} items, seed {seed}, budget {g['budget']}, max_tokens "
          f"{g['max_tokens']}, message {'none' if message is None else repr(message)}", flush=True)
    lock, seen = threading.Lock(), {}

    def chat(url, msg, req_seed, max_tokens):
        r = ew.chat_request(url, msg, req_seed, max_tokens)
        with lock:
            seen[req_seed] = r
        return r
    try:
        rows = E.score(srv.url, gate, out_rows, seed, g["max_tokens"], make_check(), chat, ew.count_tokens,
                       g["budget"], concurrency=min(len(order), L.gate_slots(cfg)), order=order, log=print)
    finally:
        srv.stop()
    by_id = {r["id"]: r for r in rows}
    report = []
    for i in order:
        it, row = gate[i], by_id.get(gate[i]["id"])
        r = seen.get(100000 * seed + i) or {}
        fences, closed = code_block(r.get("content"))
        report.append({"id": it["id"], "finish_reason": row and row["finish_reason"],
                       "completion_tokens": row and row["completion_tokens"],
                       "reasoning_tokens": row and row["reasoning_tokens"], "budget_hit": row and row["budget_hit"],
                       "sentence_seen": sentence_seen(r.get("reasoning"), message), "fences": fences,
                       "code_block": closed, "pass": row and row["pass"], "verdict": row and row.get("verdict"),
                       "reasoning_tail": (r.get("reasoning") or "")[-TAIL:],
                       "answer_tail": (r.get("content") or "")[-TAIL:]})
    hits = [x for x in report if x["budget_hit"]]
    summary = {"label": label, "binary": binary, "flags": flags, "command": srv.describe(), "seed": seed,
               "budget": g["budget"], "max_tokens": g["max_tokens"], "message": message, "items": len(report),
               "budget_hits": len(hits), "budget_hits_with_code_block": sum(x["code_block"] for x in hits),
               "budget_hits_sentence_seen": sum(x["sentence_seen"] for x in hits),
               "ran_out": sum(x["finish_reason"] == "length" for x in report),
               "passed": sum(bool(x["pass"]) for x in report),
               "answers": report}
    with open(os.path.join(OUT_DIR, f"{label}.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=1)
    for x in report:
        print(f"  {x['id']}: finish {x['finish_reason']}, {x['completion_tokens']} tokens, thinking "
              f"{x['reasoning_tokens']}{' (budget)' if x['budget_hit'] else ''}, sentence "
              f"{'seen' if x['sentence_seen'] else 'not seen'}, code block {'yes' if x['code_block'] else 'no'} "
              f"({x['fences']} fences), pass {x['pass']}", flush=True)
    print(f"{label}: {summary['budget_hits']} of {len(report)} reached the budget; of those "
          f"{summary['budget_hits_with_code_block']} wrote a finished code block and the sentence was seen in "
          f"{summary['budget_hits_sentence_seen']}; {summary['ran_out']} ran out; {summary['passed']} passed -> "
          f"{os.path.join(OUT_DIR, label + '.json')}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
