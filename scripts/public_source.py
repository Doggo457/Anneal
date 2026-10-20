"""Turn a public chat-format dataset (a "messages" list per row, single-turn or agent sessions with tool calls) into a
processed source folder for the improvement loop: <out>/train.jsonl, val.jsonl, data_card.md, measure.json.

For datasets with separate prompt / response / reasoning columns use prepare_dataset.py instead; this script is for
rows that already are conversations.

  python scripts/public_source.py --dataset nvidia/Nemotron-SFT-Math-v4 --files data/* --max-files 1 \
      --where subset=cot --where license=CC-BY-4.0 --require-think --tokenizer <snapshot> --max-tokens 8192 \
      --decontaminate --decontaminate-gate data/eval/decontam_extra.jsonl --out data/processed/loop/math_v4 [--measure-only]
  Agent sessions, one row per step: --turns each --drop-tool-name unknown_tool --min-target-chars 200

Rows. --turns last: one row per conversation, ending at its last assistant turn. --turns each: one row per assistant
turn (the conversation up to and including that turn), so every step of an agent session is a training target; the
trainer scores only the final assistant turn of a row (anneal/data.py encode_chat), so the earlier turns are context.
Assistant reasoning in a "reasoning_content", "reasoning" or "thinking" field is moved into the content as
"<think>\\nTRACE\\n</think>\\n\\nANSWER", the form train_longrun.py splits back out. ShareGPT rows (from/value,
human/gpt) are mapped to role/content.

Filters, in order, each counted in measure.json and the card:
  no_assistant / bad_format     conversation without an assistant turn, or messages that aren't role/content dicts,
                                or tool-call arguments that aren't a JSON object (kept as a JSON string; the
                                template gets them parsed, see anneal/data.py chat_messages)
  require_think                 (--require-think) the target turn has no reasoning
  short_target                  target text under --min-target-chars and no tool call
  dropped_tool_name             the target calls a tool named by --drop-tool-name (e.g. "unknown_tool": the name was
                                lost, so the call teaches nothing)
  duplicate                     the same target text was kept before
  too_long                      over --max-tokens through the model's chat template (rows over 6 chars per token
                                are counted without tokenising; later turns of a session only get longer)
  template_error                the chat template refused the row (the first messages are in measure.json)
  contaminated_evalplus / _gate EvalPlus 13-gram / Jaccard (prepare_dataset.Decontaminator) on the first user message
                                + target; the loop's held-out problems (anneal/evalsets.OverlapIndex) on all user text
                                + target
Validation is split by conversation, so no session has steps in both train and val. Conversations are read in
file order; --max-rows stops at that many kept rows.
"""

import argparse
import collections
import hashlib
import json
import os
import random
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

HUB = os.path.join(ROOT, "models", "hf", "hub")
REASONING_KEYS = ("reasoning_content", "reasoning", "thinking")
ROLE_MAP = {"human": "user", "user": "user", "gpt": "assistant", "assistant": "assistant", "system": "system",
            "tool": "tool", "function": "tool", "observation": "tool", "ipython": "tool"}


def fetch(repo, files, revision=None, max_files=0):
    from huggingface_hub import HfApi, hf_hub_download
    api = HfApi()
    sha = revision or api.dataset_info(repo).sha
    listed = api.list_repo_files(repo, repo_type="dataset", revision=sha)
    want = [f for f in listed if any(f == p or f.startswith(p.rstrip("*")) for p in files)
            and f.endswith((".jsonl", ".json", ".parquet"))]
    if any(f.endswith((".jsonl", ".parquet")) for f in want):  # a .json beside data files is metadata, not rows
        want = [f for f in want if not f.endswith(".json")]
    if not want:
        raise SystemExit(f"{repo}@{sha}: no files match {files}")
    want = sorted(want)[:max_files] if max_files else sorted(want)
    return [hf_hub_download(repo, f, repo_type="dataset", revision=sha, cache_dir=HUB) for f in want], sha


def iter_rows(paths, field, where_cols=()):
    """(messages, {where column: value}) per row."""
    cols = [field] + [c for c in where_cols if c != field]
    for p in paths:
        if p.endswith(".parquet"):
            import pyarrow.parquet as pq
            pf = pq.ParquetFile(p)
            missing = [c for c in cols if c not in pf.schema_arrow.names]
            if missing:
                raise SystemExit(f"{p}: columns {missing} not found in {pf.schema_arrow.names}")
            for batch in pf.iter_batches(batch_size=64, columns=cols):
                for r in batch.to_pylist():
                    yield r[field], {c: r.get(c) for c in where_cols}
        else:
            with open(p, encoding="utf-8") as fh:
                for line in fh:
                    if line.strip():
                        r = json.loads(line)
                        yield r.get(field), {c: r.get(c) for c in where_cols}


def parse_where(spec):
    for op in ("=", "~"):
        col, sep, val = spec.partition(op)
        if sep and col and "=" not in col and "~" not in col:
            return col.strip(), op, val.strip().lower()
    raise SystemExit(f"--where must be COL=VALUE or COL~TEXT, got {spec!r}")


def where_ok(value, op, want):
    v = "" if value is None else str(value).strip().lower()
    return v == want if op == "=" else want in v


def parse_arguments(args):
    """Tool-call arguments as a canonical JSON-object string, or None if they aren't a JSON object. Rows keep the
    string form (as the OpenAI wire format does): load_dataset's Arrow can't hold per-tool argument dicts, and
    anneal/data.py chat_messages() parses them for Qwen3.8's template, which only takes a mapping."""
    if args is None:
        args = {}
    if isinstance(args, str):
        try:
            args = json.loads(args) if args.strip() else {}
        except ValueError:
            return None
    return json.dumps(args, ensure_ascii=False) if isinstance(args, dict) else None


def normalise(messages):
    """-> list of clean message dicts, or None if the format isn't understood."""
    if isinstance(messages, str):
        try:
            messages = json.loads(messages)
        except ValueError:
            return None
    if not isinstance(messages, list):
        return None
    out = []
    for m in messages:
        if not isinstance(m, dict):
            return None
        role = ROLE_MAP.get(str(m.get("role", m.get("from", ""))).lower())
        if role is None:
            return None
        content = m.get("content", m.get("value"))
        content = "" if content is None else content
        if not isinstance(content, str):
            if isinstance(content, list):  # [{"type": "text", "text": ...}, ...]
                content = "".join(c.get("text", "") for c in content if isinstance(c, dict))
            else:
                return None
        msg = {"role": role, "content": content}
        if role == "assistant":
            trace = next((m[k] for k in REASONING_KEYS if isinstance(m.get(k), str) and m[k].strip()), None)
            if trace and "<think>" not in content:
                msg["content"] = f"<think>\n{trace.strip()}\n</think>\n\n{content.strip()}"
            calls = m.get("tool_calls")
            if calls:
                clean = []
                for c in calls:
                    fn = (c or {}).get("function") or {}
                    if not fn.get("name"):
                        return None
                    args = parse_arguments(fn.get("arguments"))
                    if args is None:
                        return None
                    clean.append({"type": "function", "id": c.get("id"),
                                  "function": {"name": fn["name"], "arguments": args}})
                msg["tool_calls"] = clean
        if role == "tool":
            for k in ("tool_call_id", "name"):
                if m.get(k):
                    msg[k] = m[k]
        out.append(msg)
    return out


def targets(messages, mode):
    """Indices of the assistant turns that end a row."""
    idx = [i for i, m in enumerate(messages) if m["role"] == "assistant"]
    return idx[-1:] if mode == "last" else idx


def target_text(m):
    return (m.get("content") or "") + "".join(json.dumps(c["function"], ensure_ascii=False) for c in m.get("tool_calls", []))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--revision", default=None)
    ap.add_argument("--files", required=True, help="comma-separated repo paths or prefixes (end with * for a prefix)")
    ap.add_argument("--from-local", default=None, help="comma-separated local files instead of downloading")
    ap.add_argument("--messages-field", default="messages")
    ap.add_argument("--max-files", type=int, default=0, help="use only the first N matching files (sorted)")
    ap.add_argument("--where", action="append", default=[],
                    help="keep rows where COL=VALUE (exact) or COL~TEXT (contains), case-insensitive; repeatable")
    ap.add_argument("--turns", choices=["last", "each"], default="last")
    ap.add_argument("--require-think", action="store_true")
    ap.add_argument("--min-target-chars", type=int, default=0)
    ap.add_argument("--drop-tool-name", action="append", default=[])
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--max-rows", type=int, default=0, help="stop after this many kept rows (0 = all)")
    ap.add_argument("--decontaminate", action="store_true", help="EvalPlus 13-gram / Jaccard check")
    ap.add_argument("--decontaminate-gate", default=None, help="data/eval/decontam_extra.jsonl")
    ap.add_argument("--val-fraction", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--measure-only", action="store_true", help="count what survives; write measure.json only")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    t0 = time.time()
    os.makedirs(args.out, exist_ok=True)

    if args.from_local:
        paths, sha = [p.strip() for p in args.from_local.split(",") if p.strip()], "local"
    else:
        paths, sha = fetch(args.dataset, [f.strip() for f in args.files.split(",") if f.strip()], args.revision,
                           args.max_files)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    tok = getattr(tok, "tokenizer", tok)

    from anneal.data import template_ids

    def n_tokens(msgs):
        thinking = "<think>" in (msgs[-1].get("content") or "")
        return len(template_ids(tok, msgs, add_generation_prompt=False, enable_thinking=thinking))

    deco = None
    if args.decontaminate:
        from prepare_dataset import Decontaminator
        deco = Decontaminator.from_evalplus()
        if deco is None:
            raise SystemExit("--decontaminate needs evalplus installed")
    gate = None
    if args.decontaminate_gate:
        from anneal.evalsets import OverlapIndex
        gate = OverlapIndex.from_jsonl(args.decontaminate_gate)

    counts = collections.Counter({k: 0 for k in (
        "conversations", "filtered_where", "bad_format", "no_assistant", "candidate_rows", "require_think", "short_target",
        "dropped_tool_name", "duplicate", "too_long", "template_error", "contaminated_evalplus", "contaminated_gate",
        "kept")})
    template_errors = {}  # first conversation index per distinct error message
    tool_names, lengths, kept_by_kind = collections.Counter(), [], collections.Counter()
    seen = set()
    groups = collections.OrderedDict()  # conversation index -> rows
    char_cap = 6 * args.max_tokens
    # Streamed in file order (a 565 MB file of agent sessions would take GBs as Python objects); only kept rows are
    # held. Train/val are shuffled at the end.
    wheres = [parse_where(w) for w in args.where]
    for ci, (raw, cols) in enumerate(iter_rows(paths, args.messages_field, [c for c, _, _ in wheres])):
        if args.max_rows and counts["kept"] >= args.max_rows:
            break
        counts["conversations"] += 1
        if wheres and not all(where_ok(cols.get(c), op, v) for c, op, v in wheres):
            counts["filtered_where"] += 1
            continue
        msgs = normalise(raw)
        if msgs is None:
            counts["bad_format"] += 1
            continue
        ends = targets(msgs, args.turns)
        if not ends:
            counts["no_assistant"] += 1
            continue
        for m in msgs:
            for c in m.get("tool_calls", []):
                tool_names[c["function"]["name"]] += 1
        user_text = "\n".join(m["content"] for m in msgs if m["role"] == "user")
        first_user = next((m["content"] for m in msgs if m["role"] == "user"), "")
        prefix_chars = 0
        cut = 0
        for end in ends:
            counts["candidate_rows"] += 1
            prefix_chars += sum(len(target_text(m)) for m in msgs[cut:end + 1])
            cut = end + 1
            t = msgs[end]
            text = target_text(t)
            has_call = bool(t.get("tool_calls"))
            if args.require_think and "<think>" not in (t.get("content") or ""):
                counts["require_think"] += 1
                continue
            if len((t.get("content") or "").strip()) < args.min_target_chars and not has_call:
                counts["short_target"] += 1
                continue
            if any(c["function"]["name"] in args.drop_tool_name for c in t.get("tool_calls", [])):
                counts["dropped_tool_name"] += 1
                continue
            key = hashlib.sha1(text.encode("utf-8")).hexdigest()
            if key in seen:
                counts["duplicate"] += 1
                continue
            if prefix_chars > char_cap:
                counts["too_long"] += 1
                continue
            row = msgs[:end + 1]
            try:
                n = n_tokens(row)
            except Exception as e:  # jinja2 TemplateError and the like: one odd row must not end the whole pool
                counts["template_error"] += 1
                template_errors.setdefault(f"{type(e).__name__}: {e}"[:300], ci)
                continue
            if n > args.max_tokens:
                counts["too_long"] += 1
                continue
            if deco is not None and deco.check(first_user, text):
                counts["contaminated_evalplus"] += 1
                continue
            if gate is not None and gate.hits(user_text + "\n" + text, jaccard_text=first_user):
                counts["contaminated_gate"] += 1
                continue
            seen.add(key)
            counts["kept"] += 1
            lengths.append(n)
            kept_by_kind["tool_call" if has_call else ("thinking" if "<think>" in (t.get("content") or "") else "text")] += 1
            groups.setdefault(ci, []).append({"messages": row})
            if args.max_rows and counts["kept"] >= args.max_rows:
                break

    s = sorted(lengths)
    q = (lambda x: s[min(len(s) - 1, int(x * len(s)))]) if s else (lambda x: None)
    measure = {"dataset": args.dataset, "revision": sha, "files": paths, "turns": args.turns, "counts": dict(counts),
               "kept_by_target_kind": dict(kept_by_kind), "tool_names_top": tool_names.most_common(20),
               "template_errors": [{"error": k, "first_conversation": v} for k, v in list(template_errors.items())[:10]],
               "kept_tokens": {"min": s[0] if s else None, "median": q(0.5), "p90": q(0.9), "max": s[-1] if s else None,
                               "total": sum(s)},
               "settings": {k: v for k, v in vars(args).items() if k not in ("tokenizer",)},
               "seconds": round(time.time() - t0)}
    with open(os.path.join(args.out, "measure.json"), "w", encoding="utf-8") as fh:
        json.dump(measure, fh, indent=1)
    print(json.dumps({k: v for k, v in measure.items() if k != "files"}, indent=1))
    if args.measure_only:
        return 0
    if not counts["kept"]:
        raise SystemExit("no rows survived the filters")

    keys = list(groups)
    rng = random.Random(args.seed)
    rng.shuffle(keys)
    n_val = max(1, int(counts["kept"] * args.val_fraction))
    train, val = [], []
    for k in keys:
        (val if len(val) < n_val else train).extend(groups[k])
    rng.shuffle(train)
    for name, data in (("train", train), ("val", val)):
        with open(os.path.join(args.out, f"{name}.jsonl"), "w", encoding="utf-8") as fh:
            for r in data:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(os.path.join(args.out, "data_card.md"), "w", encoding="utf-8") as fh:
        fh.write(f"# Data card\n\nSource: `{args.dataset}` @ `{sha}`, files `{args.files}`, rows: one per "
                 f"{'assistant turn' if args.turns == 'each' else 'conversation'}.\n\n| Stage | Rows |\n|---|---|\n")
        for k, v in counts.items():
            fh.write(f"| {k} | {v} |\n")
        fh.write(f"| train | {len(train)} |\n| val | {len(val)} |\n\nTarget kinds kept: {dict(kept_by_kind)}. "
                 f"Kept-row tokens: median {q(0.5)}, p90 {q(0.9)}, max {s[-1]}. Max {args.max_tokens} tokens through "
                 f"the chat template; EvalPlus decontamination {'on' if deco else 'off'}; held-out overlap check "
                 f"{'on' if gate else 'off'}; dropped tool names {args.drop_tool_name}; row filters {args.where or 'none'}.\n")
    print(f"wrote {len(train)} train / {len(val)} val rows -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
