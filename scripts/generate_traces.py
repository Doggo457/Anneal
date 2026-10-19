"""Generate thinking-mode training data from the base model itself (self-distillation).

Sends coding prompts to a local llama-server running the base model with thinking ON, keeps the
reasoning trace plus the answer, and keeps only answers whose code passes the prompt's tests.
Output is messages-format JSONL that prepare_dataset.py --from-jsonl consumes.

Why: thinking mode is ON for this project, so training rows need <think> traces. Traces written by the
model itself are in its own style (on-policy), and a test filter makes their quality measurable.

Server. Cap reasoning at what can still fit an 8,192-token training row (prompt + trace + answer);
longer traces are dropped, so generating past the cap only wastes time. Two ways to run it, and only a
measurement says which is faster on this chip (the script prints generated tok/s every 10 kept rows):
  # A: one stream with MTP speculative decoding (measured ~24.5 tok/s at Q4_K_M, Hardware-Notes.md)
  llama-server -m models/<Qwen3.8-27B-Q4_K_M>.gguf -c 32768 -ctk q8_0 -ctv q8_0 -ngl 999 -fa on \
      --reasoning-format deepseek --reasoning-budget 6144 --port 8080 -np 1 \
      --spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-p-min 0.0
  # B: 4 parallel slots, no MTP (MTP made 4 parallel requests slower, 0.78x). -c is split across
  # slots, so 65536 / 4 = 16k tokens each. Pass --concurrency 4 to this script.
  llama-server ... -c 65536 -np 4 --reasoning-format deepseek --reasoning-budget 6144 --port 8080

Prompt pools (all CC-BY-4.0):
  # OpenCodeInstruct: function-level tasks with 10 assert tests each (string-encoded list).
  python scripts/generate_traces.py --dataset nvidia/OpenCodeInstruct \
      --data-files data/train-00000-of-00050.parquet,data/train-00049-of-00050.parquet \
      --prompt-field input --tests-field unit_tests --min-test-score 1.0 --decontaminate \
      --run-tests --max-samples 10000 --resume --out data/raw/self_oci.jsonl
  # CodeContests: competitive programming with stdin/stdout tests (13,328 training problems).
  python scripts/generate_traces.py --dataset deepmind/code_contests --data-files "data/train-*.parquet" \
      --prompt-field description --tests-field public_tests,private_tests,generated_tests \
      --test-format stdio --decontaminate --run-tests --max-samples 6000 --resume --out data/raw/self_cc.jsonl
  # Round 2 (expert iteration): only the prompts round 1 failed, with the round-1 adapter loaded in
  # llama-server (--lora). Use a new --out; --retry-rejected reads round 1's .rejected sidecar.
  python scripts/generate_traces.py ...same pool... --retry-rejected data/raw/self_cc.jsonl.rejected \
      --out data/raw/self_cc_r2.jsonl --resume

Then: python scripts/prepare_dataset.py --from-jsonl data/raw/self_oci.jsonl --require-think \
          --tokenizer Qwen/Qwen3.8-27B --max-tokens 8192 --decontaminate --out data/processed/self_oci

Tests (--test-format auto picks per row):
  asserts: plain assert lines, or a string-encoded list of them (OpenCodeInstruct). A prompt whose tests
           contain no `assert`, or are pytest functions, is skipped: a plain run would execute nothing
           and pass every answer.
  stdio:   {"input": [...], "output": [...]} (CodeContests) or a JSON string {"inputs", "outputs"}
           (TACO/APPS style). Several --tests-field columns are merged in order, duplicates dropped,
           capped at --max-io-tests. Rows with "fn_name" (call-based tests) are skipped. Output is
           compared token by token after whitespace splitting, with numbers equal within 1e-6
           relative. The program must pass every case; it stops at the first failure. Problems that
           accept several correct outputs will be rejected even when right, which only lowers the yield.
           Stdio prompts get --prompt-suffix (a default asking for one Python 3 program reading stdin),
           and the suffix stays in the training row so prompt and trace match.

--decontaminate drops prompts that overlap HumanEval+/MBPP+ BEFORE generation (same rule as
prepare_dataset.py), so no generation time is spent on rows that would be thrown away.

--max-samples is the total kept in --out, counting rows already there with --resume. With
--concurrency N, requests already in flight when the target is reached still finish and are kept if
they pass, so the total can exceed it by up to N-1.

Every rejected prompt is written to <out>.rejected as "sha1<TAB>reason", so --resume never regenerates
it and --retry-rejected can pick out the ones that failed their tests.

Notes: --run-tests executes model-written code, so on this machine it runs only in the code sandbox
(anneal/sandbox.py: an AppContainer with no capabilities, so no network and no access to user files, inside
a job object with a 4 GiB memory limit and no subprocesses; results/sandbox_findings.md). --run-tests is
refused if the sandbox isn't set up. Each assert run gets --test-timeout seconds and each stdin/stdout case
--io-test-timeout seconds of wall clock, each in a fresh scratch folder.
"""
import argparse
import ast
import hashlib
import json
import math
import os
import sys
import time
import urllib.request
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # C:\AI\training
from anneal import sandbox  # noqa: E402

TEST_MEM_BYTES = 4 << 30  # job memory limit per test run

SYSTEM = None  # optional system prompt; keep None unless the eval uses one too
PY_TAGS = ("python", "py", "python3", "")
STDIO_SUFFIX = ("\n\nWrite a complete Python 3 program that reads the input from standard input and writes "
                "the answer to standard output. Put the whole program in a single ```python code block.")
REJECT_REASONS = ("length_or_empty", "budget_hit", "too_long_row", "tests_failed")


def post(url, path, body, timeout):
    req = urllib.request.Request(f"{url}{path}", data=json.dumps(body).encode("utf-8"),
                                 headers={"Content-Type": "application/json", "Authorization": "Bearer none"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def chat(url, model, prompt, temperature, top_p, max_tokens, timeout):
    body = {
        "model": model,
        "messages": ([{"role": "system", "content": SYSTEM}] if SYSTEM else []) + [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": True},
    }
    data = post(url, "/v1/chat/completions", body, timeout)
    msg = data["choices"][0]["message"]
    finish = data["choices"][0].get("finish_reason")
    content = (msg.get("content") or "").strip()
    reasoning = (msg.get("reasoning_content") or msg.get("reasoning") or "").strip()
    if not reasoning and "<think>" in content and "</think>" in content:
        # Server not started with --reasoning-format: split it ourselves.
        head, _, tail = content.partition("</think>")
        reasoning = head.replace("<think>", "").strip()
        content = tail.strip()
    generated = (data.get("usage") or {}).get("completion_tokens")
    return reasoning, content, finish, generated


def count_tokens(url, text):
    """Exact token count from llama-server's /tokenize. None if the endpoint is unavailable."""
    try:
        return len(post(url, "/tokenize", {"content": text, "add_special": False}, 60)["tokens"])
    except Exception:  # noqa: BLE001
        return None


# ---------------------------------------------------------------- tests

def _maybe_literal(s):
    s = s.strip()
    if s[:1] in "[{":
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            try:
                return ast.literal_eval(s)
            except (ValueError, SyntaxError):
                return s
    return s


def parse_assert_tests(values):
    """Assert lines from one or more fields, or None if nothing would execute."""
    parts = []
    for v in values:
        if v is None:
            continue
        v = _maybe_literal(v) if isinstance(v, str) else v
        if isinstance(v, (list, tuple)):
            parts += [str(t).strip() for t in v if str(t).strip()]
        else:
            parts.append(str(v).strip())
    tests = "\n".join(p for p in parts if p)
    if "assert" not in tests or "def test_" in tests:
        return None  # nothing would execute, or pytest-style functions that a plain run never calls
    return tests


def parse_io_tests(values, max_cases):
    """(stdin, expected stdout) pairs from one or more fields, or None if there are none or they are
    call-based. Accepts {"input": [...], "output": [...]} and {"inputs": [...], "outputs": [...]}."""
    cases, seen = [], set()
    for v in values:
        if v is None:
            continue
        v = _maybe_literal(v) if isinstance(v, str) else v
        if not isinstance(v, dict):
            continue
        if v.get("fn_name"):
            return None  # call-based tests (function signature), not stdin/stdout
        ins = v.get("input", v.get("inputs")) or []
        outs = v.get("output", v.get("outputs")) or []
        for i, o in zip(ins, outs):
            if not isinstance(i, str) or not isinstance(o, str):
                return None
            if (i, o) not in seen:
                seen.add((i, o))
                cases.append((i, o))
    return cases[:max_cases] if cases else None


def parse_tests(values, fmt, max_cases):
    """Returns ("asserts", str) or ("stdio", [(in, out), ...]) or None."""
    if fmt in ("stdio", "auto"):
        io = parse_io_tests(values, max_cases)
        if io:
            return "stdio", io
        if fmt == "stdio":
            return None
    a = parse_assert_tests(values)
    return ("asserts", a) if a else None


def extract_code(text):
    """First fenced code block tagged as Python (or untagged), else the first block, else the whole text."""
    if "```" not in text:
        return text
    parts = text.split("```")
    blocks = []
    for block in parts[1::2]:
        first_nl = block.find("\n")
        tag = block[:first_nl].strip().lower() if first_nl != -1 else ""
        body = block[first_nl + 1:] if first_nl != -1 and " " not in tag else block
        blocks.append((tag, body))
    for tag, body in blocks:
        if tag in PY_TAGS:
            return body
    return blocks[0][1] if blocks else text


def run_assert_tests(code, tests, timeout):
    r = sandbox.run(["candidate.py"], files={"candidate.py": code + "\n\n" + tests + "\n"}, timeout_s=timeout,
                    mem_bytes=TEST_MEM_BYTES)
    return r.returncode == 0 and not r.timed_out


def outputs_match(got, want, rel=1e-6):
    g, w = got.split(), want.split()
    if g == w:
        return True
    if len(g) != len(w):
        return False
    for a, b in zip(g, w):
        if a == b:
            continue
        try:
            fa, fb = float(a), float(b)
        except ValueError:
            return False
        if not (math.isfinite(fa) and math.isfinite(fb)) or abs(fa - fb) > rel * max(1.0, abs(fb)):
            return False
    return True


def run_io_tests(code, cases, timeout):
    for stdin, expected in cases:
        r = sandbox.run(["candidate.py"], stdin=stdin.encode("utf-8"), files={"candidate.py": code + "\n"},
                        timeout_s=timeout, mem_bytes=TEST_MEM_BYTES)
        if r.timed_out or r.returncode != 0 or not outputs_match(r.stdout.decode("utf-8", "replace"), expected):
            return False
    return True


# ---------------------------------------------------------------- prompts

def iter_prompts(args, fields):
    """Yields (prompt, [test values]). Applies the test-score filter while reading."""
    if args.prompts_jsonl:
        with open(args.prompts_jsonl, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    yield r["prompt"], [r.get("tests")]
        return
    from datasets import load_dataset

    files = [x.strip() for x in args.data_files.split(",") if x.strip()] if args.data_files else None
    ds = load_dataset(args.dataset, args.config, data_files=files, split="train" if files else args.split)
    print(f"  {len(ds)} rows, columns: {ds.column_names}")
    for need in [args.prompt_field] + fields:
        if need not in ds.column_names:
            raise SystemExit(f"Field {need} not found. Columns: {ds.column_names}")
    keep = [args.prompt_field] + fields + ([args.test_score_field] if args.min_test_score is not None else [])
    ds = ds.select_columns(list(dict.fromkeys(keep)))  # don't decode unused columns (CodeContests solutions)
    for ex in ds.shuffle(seed=args.seed):
        if args.min_test_score is not None:
            try:
                if float(ex.get(args.test_score_field)) < args.min_test_score:
                    continue
            except (TypeError, ValueError):
                continue
        prompt = ex[args.prompt_field]
        if isinstance(prompt, list):  # a messages column (e.g. Nemotron-Post-Training): use the first user turn
            prompt = next((m.get("content") for m in prompt if isinstance(m, dict) and m.get("role") == "user"), None)
        yield prompt, [ex.get(f) for f in fields]


def read_rejected(path, reasons=None):
    """Hashes in a .rejected sidecar. Old sidecars hold bare hashes (reason unknown, always included)."""
    out = set()
    if not os.path.exists(path):
        return out
    with open(path, encoding="utf-8") as f:
        for line in f:
            h, _, reason = line.strip().partition("\t")
            if h and (reasons is None or not reason or reason in reasons):
                out.add(h)
    return out


# ---------------------------------------------------------------- one attempt (runs in a worker thread)

def attempt(args, prompt, tests, tokenize_ok):
    """Generate once and apply every check. Returns (status, row or error text, generated tokens)."""
    try:
        reasoning, answer, finish, generated = chat(args.url, args.model, prompt, args.temperature, args.top_p,
                                                    args.max_tokens, timeout=args.request_timeout)
    except Exception as e:  # noqa: BLE001
        return "failed", f"{type(e).__name__}: {str(e)[:100]}", 0
    generated = generated or 0
    if finish == "length" or len(reasoning) < args.min_trace_chars or not answer:
        return "length_or_empty", None, generated
    # A trace that ran into the server's reasoning budget was cut off mid-thought by a forced
    # </think>. Training on it teaches unfinished thinking. Count it exactly and drop anything
    # within 3% of the budget.
    n_reason = count_tokens(args.url, reasoning) if tokenize_ok else len(reasoning) // 4
    if args.reasoning_budget_tokens and n_reason >= 0.97 * args.reasoning_budget_tokens:
        return "budget_hit", None, generated
    content = f"<think>\n{reasoning}\n</think>\n\n{answer}"
    n_row = count_tokens(args.url, prompt + "\n" + content) if tokenize_ok else (len(prompt) + len(content)) // 4
    if n_row + 16 > args.max_row_tokens:  # ~16 tokens of chat-template markup
        return "too_long_row", None, generated
    if args.run_tests:
        kind, spec = tests
        code = extract_code(answer)
        ok = (run_io_tests(code, spec, args.io_test_timeout) if kind == "stdio"
              else run_assert_tests(code, spec, args.test_timeout))
        if not ok:
            return "tests_failed", None, generated
    row = {"messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": content}]}
    return "kept", row, generated


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--url", default="http://127.0.0.1:8080")
    p.add_argument("--model", default="local")
    p.add_argument("--dataset", default="nvidia/OpenCodeInstruct")
    p.add_argument("--config", default=None)
    p.add_argument("--split", default="train")
    p.add_argument("--data-files", default=None,
                   help="comma-separated shard paths (globs allowed) inside the dataset repo")
    p.add_argument("--prompt-field", default="input")
    p.add_argument("--tests-field", default=None,
                   help="column(s) holding the tests for --run-tests; comma-separate several to merge them")
    p.add_argument("--test-format", choices=("auto", "asserts", "stdio"), default="auto")
    p.add_argument("--max-io-tests", type=int, default=20, help="stdin/stdout cases run per answer")
    p.add_argument("--prompt-suffix", default=None,
                   help="appended to every prompt (default: a stdin/stdout instruction for stdio tests)")
    p.add_argument("--min-test-score", type=float, default=None,
                   help="keep only prompts whose reference solution passed at least this fraction of its tests "
                        "(OpenCodeInstruct: 1.0)")
    p.add_argument("--test-score-field", default="average_test_score")
    p.add_argument("--prompts-jsonl", default=None, help='local JSONL with {"prompt": ..., "tests": ...} rows')
    p.add_argument("--retry-rejected", default=None,
                   help="only try prompts listed as tests_failed in this .rejected sidecar (expert iteration)")
    p.add_argument("--max-samples", type=int, default=300,
                   help="rows to keep in --out in total (with --resume, rows already there count)")
    p.add_argument("--concurrency", type=int, default=1,
                   help="requests in flight; match llama-server -np (parallel slots)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--temperature", type=float, default=0.6)   # matches the project's eval sampling
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--max-tokens", type=int, default=8192,
                   help="completion cap; a training row can't exceed the 8,192 seq_len anyway")
    p.add_argument("--max-row-tokens", type=int, default=8192,
                   help="drop rows whose prompt + trace + answer exceed this (exact count via /tokenize)")
    p.add_argument("--min-trace-chars", type=int, default=200, help="drop answers with a trivial or empty trace")
    p.add_argument("--reasoning-budget-tokens", type=int, default=6144,
                   help="the server's --reasoning-budget; traces that hit it are dropped as truncated (0 = off)")
    p.add_argument("--run-tests", action="store_true")
    p.add_argument("--test-timeout", type=int, default=20, help="seconds for one assert-test run")
    p.add_argument("--io-test-timeout", type=int, default=10, help="seconds per stdin/stdout case")
    p.add_argument("--request-timeout", type=int, default=3600)
    p.add_argument("--decontaminate", action="store_true",
                   help="skip prompts overlapping HumanEval+/MBPP+ before generating (needs evalplus)")
    p.add_argument("--decontaminate-extra", default=None,
                   help="file of extra evaluation questions to decontaminate against (JSONL or one per line)")
    p.add_argument("--out", default="data/raw/self_traces.jsonl")
    p.add_argument("--resume", action="store_true", help="skip prompts already present in --out")
    args = p.parse_args()

    fields = [f.strip() for f in args.tests_field.split(",") if f.strip()] if args.tests_field else []
    suffix = args.prompt_suffix if args.prompt_suffix is not None else (
        STDIO_SUFFIX if args.test_format == "stdio" else "")
    if args.run_tests and not (fields or args.prompts_jsonl):
        raise SystemExit("--run-tests needs --tests-field (or a prompts JSONL with a tests key)")
    if args.run_tests and not sandbox.available():
        raise SystemExit("--run-tests runs model-written code and needs the code sandbox, which isn't set up: "
                         "python -m anneal.sandbox --setup (see results/sandbox_findings.md)")

    deco = None
    if args.decontaminate:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from prepare_dataset import Decontaminator

        deco = Decontaminator.from_evalplus(extra=args.decontaminate_extra)
        if deco is None:
            raise SystemExit("--decontaminate needs evalplus: pip install evalplus")

    done, already = set(), 0
    rejected_path = args.out + ".rejected"
    tried = read_rejected(rejected_path) if args.resume else set()
    if args.resume and os.path.exists(args.out):
        with open(args.out, encoding="utf-8") as f:
            for line in f:
                try:
                    done.add(json.loads(line)["messages"][-2]["content"])
                except Exception:  # noqa: BLE001
                    pass
        already = len(done)
        print(f"  resuming: {already} rows already in {args.out}, {len(tried)} prompts rejected before")
    only = None
    if args.retry_rejected:
        only = read_rejected(args.retry_rejected, reasons={"tests_failed"})
        print(f"  retrying {len(only)} prompts that failed their tests in {args.retry_rejected}")
        if not only:
            raise SystemExit("Nothing to retry.")

    tokenize_ok = count_tokens(args.url, "probe") is not None
    if not tokenize_ok:
        print("  warning: /tokenize unavailable; falling back to ~4 chars per token for length checks")

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    skipped = {"no_tests": 0, "contaminated": 0}
    rejected = {r: 0 for r in REJECT_REASONS}
    stats = {"kept": 0, "failed": 0, "generated": 0, "attempts": 0}
    queued = set()

    def candidates():
        """Prompts worth a request: not done, not rejected before, has runnable tests, not contaminated."""
        for raw, test_values in iter_prompts(args, fields):
            raw = (raw or "").strip()
            if not raw:
                continue
            prompt = raw + suffix
            h = hashlib.sha1(prompt.encode("utf-8")).hexdigest()
            if prompt in done or h in tried or h in queued or (only is not None and h not in only):
                continue
            tests = None
            if args.run_tests:
                tests = parse_tests(test_values, args.test_format, args.max_io_tests)
                if tests is None:
                    skipped["no_tests"] += 1
                    continue
            if deco is not None and deco.check(raw):
                skipped["contaminated"] += 1
                continue
            queued.add(h)
            yield h, prompt, tests

    t0 = time.time()

    def progress(final=False):
        el = max(time.time() - t0, 1e-9)
        line = (f"kept {stats['kept']} (total {already + stats['kept']})  attempts {stats['attempts']}  "
                f"rejected {rejected}  skipped {skipped}  failed {stats['failed']}  "
                f"{stats['generated'] / el:.1f} generated tok/s  {3600 * stats['kept'] / el:.1f} kept/h")
        print(("\nDone: " if final else "  ") + line)

    src = candidates()
    pending = {}
    exhausted = False
    with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as pool, \
            open(args.out, "a", encoding="utf-8") as out, open(rejected_path, "a", encoding="utf-8") as rej:
        try:
            while True:
                while (not exhausted and already + stats["kept"] < args.max_samples
                       and len(pending) < max(1, args.concurrency)):
                    try:
                        h, prompt, tests = next(src)
                    except StopIteration:
                        exhausted = True
                        break
                    pending[pool.submit(attempt, args, prompt, tests, tokenize_ok)] = (h, prompt)
                if not pending:
                    break
                finished, _ = wait(pending, return_when=FIRST_COMPLETED)
                for fut in finished:
                    h, prompt = pending.pop(fut)
                    status, payload, generated = fut.result()
                    stats["generated"] += generated
                    if status == "failed":
                        stats["failed"] += 1  # not recorded as rejected: a server error says nothing
                        queued.discard(h)
                        print(f"  request failed: {payload}")
                        continue
                    stats["attempts"] += 1
                    if status != "kept":
                        rejected[status] += 1
                        rej.write(f"{h}\t{status}\n")
                        rej.flush()
                        continue
                    out.write(json.dumps(payload, ensure_ascii=False) + "\n")
                    out.flush()
                    done.add(prompt)
                    stats["kept"] += 1
                    if stats["kept"] % 10 == 0:
                        progress()
        except KeyboardInterrupt:
            out.flush()
            rej.flush()
            progress(final=True)
            print("Interrupted; everything kept so far is saved. Re-run with --resume to continue.")
            os._exit(130)  # don't wait for requests still running on the server

    progress(final=True)
    print(f"-> {args.out}")
    print("Next: python scripts/prepare_dataset.py --from-jsonl", args.out,
          "--require-think --tokenizer Qwen/Qwen3.8-27B --max-tokens 8192 --decontaminate")


if __name__ == "__main__":
    main()
