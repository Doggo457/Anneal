"""Measure the gate server's decode speed on real gate answers: one llama-server on the loop's gate settings (loop.yaml
gate: flags, thinking budget, sampling), with a chosen llama.cpp build, slot count and MTP drafting.

The loop must be paused (nothing may answer on its port 18090: the GPU has to be free). The server starts on port
18095 with exactly the flags the loop's gate server gets (Loop.gate_flags) for the chosen settings, and one gate request
per slot is kept in flight, in a fixed shuffled order of gate.jsonl with the gate pass's request seeds (seed 1). Answers
average about 8,700 tokens, longer than the measurement, so speed comes from the server's own counts: /slots is read
every --sample-s seconds and each answer's n_decoded (tokens generated so far, thinking included) is followed by slot
and task. After --warmup minutes (the prompts are processed and every slot is decoding), the tokens decoded over the
next --minutes, divided by that time, is the result. An answer that finishes inside the window loses at most one
sample interval of its tokens; the count of such answers is reported. Draft acceptance comes from the answers that
finished (their timings: draft_n, draft_n_accepted). Writes results/gate-speed/<label>.json.

  python tests/gate_speed_bench.py --slots 6
  python tests/gate_speed_bench.py --slots 4 --mtp 1 [--build b11796-vulkan] [--lora <GGUF>] [--minutes 10]
"""

import argparse
import hashlib
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
from anneal import llama  # noqa: E402
import evalplus_win as ew  # noqa: E402

PORT = 18095
LOOP_PORT = 18090
OUT_DIR = os.path.join(ROOT, "results", "gate-speed")


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default=L.CONFIG)
    ap.add_argument("--build", default=llama.DEFAULT_BUILD, help="a folder under llama.cpp/")
    ap.add_argument("--slots", type=int, required=True)
    ap.add_argument("--mtp", type=int, default=0, help="MTP draft length (0: no speculative decoding)")
    ap.add_argument("--lora", default=None, help="a GGUF LoRA, as a round's gate server loads it")
    ap.add_argument("--warmup", type=float, default=2.0, help="minutes before the measured window")
    ap.add_argument("--minutes", type=float, default=10.0, help="the measured window")
    ap.add_argument("--sample-s", type=float, default=5.0)
    ap.add_argument("--label", default=None)
    return ap.parse_args()


def file_sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def find_decoded(obj):
    """n_decoded inside a /slots entry, wherever this build puts it (next_token as an object or a list)."""
    if isinstance(obj, dict):
        if isinstance(obj.get("n_decoded"), int):
            return obj["n_decoded"]
        vals = [find_decoded(v) for v in obj.values() if isinstance(v, (dict, list))]
    elif isinstance(obj, list):
        vals = [find_decoded(v) for v in obj]
    else:
        return None
    vals = [v for v in vals if v is not None]
    return max(vals) if vals else None


def read_slots(url):
    """[(slot id, task id, processing, n_decoded)] from /slots."""
    out = []
    for s in llama.http(url + "/slots", timeout=60):
        out.append((s.get("id"), s.get("id_task", s.get("task_id")), bool(s.get("is_processing")), find_decoded(s)))
    return out


def window_tokens(samples, t0, t1):
    """Tokens decoded between the sample at t0 and the last sample at or before t1: per (slot, task), its n_decoded
    at the end minus at t0 (0 for a task first seen after t0). Returns (tokens, tasks finished inside the window)."""
    start = next((s for s in samples if s[0] >= t0), None)
    inside = [s for s in samples if start is not None and start[0] < s[0] <= t1]
    if start is None or not inside:
        raise RuntimeError("no /slots samples in the window")
    first = {(sid, task): n or 0 for sid, task, busy, n in start[1] if busy}
    last = {}
    for _, rows in inside:
        for sid, task, busy, n in rows:
            k = (sid, task)
            if task is not None and task >= 0 and n is not None and (busy or k in last or k in first):
                last[k] = max(last.get(k, first.get(k, 0)), n)
    end_busy = {(sid, task) for sid, task, busy, _ in inside[-1][1] if busy}
    tokens = sum(n - first.get(k, 0) for k, n in last.items())
    finished = sum(1 for k in last if k not in end_busy)
    return tokens, finished, inside[-1][0] - start[0]


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
    cfg["gate"] = dict(cfg["gate"], slots=args.slots, build=args.build,
                       spec={"type": "draft-mtp", "draft_n_max": args.mtp} if args.mtp else None)
    flags = L.Loop.gate_flags(None, cfg, lora=args.lora)
    binary = llama.server_binary(L.gate_build(cfg))
    if not os.path.exists(binary):
        raise SystemExit(f"{binary} is missing")
    g = cfg["gate"]
    gate = E.read_jsonl(E.GATE_PATH)
    order = E.shuffled_order(gate, "gate-speed-bench")
    label = args.label or (f"{args.build}-np{args.slots}" + (f"-mtp{args.mtp}" if args.mtp else "")
                           + ("-lora" if args.lora else "") + time.strftime("-%Y%m%d-%H%M%S"))
    os.makedirs(OUT_DIR, exist_ok=True)
    srv = llama.Server(flags, os.path.join(OUT_DIR, f"{label}.server.log"), port=PORT, binary=binary)
    print("llama-server:", srv.describe(), flush=True)
    srv.start()
    print(f"up in {srv.load_s} s", flush=True)
    lock, stop = threading.Lock(), threading.Event()
    nxt, done, errors, samples = [0], [], [], []

    def worker():
        while not stop.is_set():
            with lock:
                if nxt[0] >= len(order):
                    return
                i = order[nxt[0]]
                nxt[0] += 1
            t = time.monotonic()
            try:
                r = ew.chat_request(srv.url, gate[i]["prompt"], 100000 * int(g["seed"]) + i, g["max_tokens"])
            except Exception as exc:  # the server is stopped under the requests at the end
                if not stop.is_set():
                    errors.append(f"{gate[i]['id']}: {type(exc).__name__}: {exc}")
                return
            with lock:
                done.append({"id": gate[i]["id"], "started": round(t, 1), "ended": round(time.monotonic(), 1),
                             "completion_tokens": r.get("completion_tokens"), "finish_reason": r.get("finish_reason"),
                             "timings": r.get("timings")})

    threads = [threading.Thread(target=worker, daemon=True) for _ in range(args.slots)]
    t_begin = time.monotonic()
    for th in threads:
        th.start()
    t0, t1 = t_begin + 60 * args.warmup, t_begin + 60 * (args.warmup + args.minutes)
    try:
        while True:
            now = time.monotonic()
            try:
                samples.append((now, read_slots(srv.url)))
            except Exception as exc:
                errors.append(f"/slots at {now - t_begin:.0f} s: {type(exc).__name__}: {exc}")
            if not srv.alive():
                raise RuntimeError(f"llama-server stopped (log {srv.log_path})")
            if now > t1:
                break
            time.sleep(max(0.0, args.sample_s - (time.monotonic() - now)))
    finally:
        stop.set()
        srv.stop()
        for th in threads:
            th.join(30)
    if not samples:
        raise SystemExit(f"/slots never answered: {errors[:3]}")
    if not any(n is not None for _, rows in samples for *_, n in rows):
        raise SystemExit("/slots gave no n_decoded on this build: speed can't be measured this way")
    tokens, finished, span = window_tokens(samples, t0, t1)
    busy = [sum(1 for r in rows if r[2]) for t, rows in samples if t0 <= t <= t1]
    drafted = sum((d["timings"] or {}).get("draft_n") or 0 for d in done)
    accepted = sum((d["timings"] or {}).get("draft_n_accepted") or 0 for d in done)
    res = {"label": label, "build": args.build, "binary": binary, "binary_sha256": file_sha(binary),
           "slots": args.slots, "mtp_draft_n_max": args.mtp or None, "lora": args.lora, "flags": flags,
           "command": srv.describe(), "warmup_min": args.warmup, "window_s": round(span, 1), "tokens": tokens,
           "tokens_per_s": round(tokens / span, 2), "answers_finished_in_window": finished,
           "busy_slots_min": min(busy) if busy else None, "samples": len(samples), "answers_done": len(done),
           "draft_n": drafted, "draft_n_accepted": accepted,
           "draft_acceptance": round(accepted / drafted, 4) if drafted else None, "errors": errors[:20],
           "done": done, "load_s": srv.load_s}
    out = os.path.join(OUT_DIR, f"{label}.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    print(f"{label}: {res['tokens_per_s']} tokens/s over {res['window_s']} s ({tokens} tokens; at least "
          f"{res['busy_slots_min']} of {args.slots} slots busy at every sample; {finished} answers finished inside "
          f"the window" + (f"; draft acceptance {res['draft_acceptance']} over {len(done)} finished answers"
                           if drafted else "") + f") -> {out}", flush=True)
    if errors:
        print(f"{len(errors)} errors, first: {errors[0]}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
