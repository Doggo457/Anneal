# EvalPlus on Windows (HumanEval+ / MBPP+) and the step D server checks

19 October 2026. Code: `scripts/evalplus_win.py` (generation, sanitize, evaluate, self-test), `anneal/llama.py` (server flags), `tests/template_budget_check.py`.

Runs:
- `results/evalplus-selftest`: CPU and sandbox only.
- `results/template-budget-check`: through the Training page.

Versions: evalplus 0.3.1, HumanEval+ v0.1.10 (164 tasks, hash `916d9bfe…`), MBPP+ v0.2.0 (378 tasks, hash `92743def…`). Both datasets come from EvalPlus's GitHub releases and are cached in `%LOCALAPPDATA%\evalplus\evalplus\Cache`.

## How the evaluator runs here

EvalPlus's own `evaluate()` is called unchanged, and so are problem loading, `get_groundtruth()` (the trusted canonical solutions, run natively in the parent and cached), `check_correctness()`, pass@1 and `_eval_results.json`. While it runs, two names in `evalplus.evaluate` are swapped:

- **`ProcessPoolExecutor` becomes `ThreadPoolExecutor`** (parallel = 8). The workers only wait on sandboxed processes.
- **`untrusted_check` becomes `sandboxed_untrusted_check`**, with the same signature.
  - It computes `time_limits` and `timeout = min(60, sum(time_limits)) + 1` (+1 without fast_check) exactly as EvalPlus does.
  - It pickles `(dataset, entry_point, code, inputs, expected, time_limits, atol, fast_check, nonce)` into the scratch folder and runs a child script through `anneal.sandbox.run` (AppContainer + 4 GiB job).
  - The child imports `evalplus.eval` and replaces two of its functions:
    - `time_limit` uses a `threading.Timer` that calls `_thread.interrupt_main()`. A SIGINT handler raises `TimeoutException` while the limit is armed; it is disarmed before the timer is cancelled, so a late interrupt is ignored.
    - `reliability_guard` calls the original with `maximum_memory_bytes=None`. The job object caps memory instead.
  - The child then calls `evalplus.eval.unsafe_execute` itself. The `details` list's `__setitem__` writes `<nonce> D i v` to the saved real stdout, `progress` writes `<nonce> P n`, and the child prints `<nonce> READY` just before `unsafe_execute` and `<nonce> STAT s` at the end.
  - The parent starts EvalPlus's join timeout (`timeout + 1`) when READY appears, with a separate 60 s allowance for start-up before it. It maps results as `untrusted_check` does: no STAT means timeout; PASS needs `len(details) == len(inputs)` and every detail True.
  - A check that never prints READY is retried once. After that it raises `HarnessError`, so a broken sandbox can't silently score as timeouts.
- **Subsets** (e.g. the 20-problem calibration): `evaluate()` insists on every task, so `score_subset()` calls the same `get_groundtruth` and `check_correctness` directly.

**Generation** (`generate()`):
- The user message is exactly EvalPlus 0.3.1's OpenAI backend text: instruction prefix, then `\n```python\n`, then `prompt.strip()`, then `\n```` `.
- One user message only, with `chat_template_kwargs {"enable_thinking": true}`.
- Sampling: temperature 0.6, top_p 0.95, plus top_k 20 and min_p 0.
- Seed = 100000·seed + task index (EvalPlus's task order). max_tokens 40,960.
- Saved per task: raw content as `solution`, the reasoning text, completion, reasoning and answer token counts, finish_reason, seconds and server timings. Runs are resumable by task_id.
- Then `evalplus.sanitize.sanitize(solution, entry_point)` writes `-sanitized.jsonl`, as `evalplus.sanitize` / codegen do.

## Self-test (`results/evalplus-selftest/summary.json`)

| | HumanEval+ base | HumanEval+ base+extra | MBPP+ base | MBPP+ base+extra |
|---|---|---|---|---|
| canonical solutions | 163/164 = 0.9939 | 0.9939 | **1.0000** | **1.0000** |
| `def f(*args, **kwargs): return None` | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| time | 26.5 s | | 66.7 s | |

Ground truth was computed natively once: 9.4 s for HumanEval+, 27.4 s for MBPP+. Across the self-test the sandbox ran 2,194 checks with 0 harness retries; the slowest start-up was 5.1 s with 8 in parallel.

**The one canonical failure, HumanEval/32 (`find_zero`), is in EvalPlus 0.3.1 itself, not this harness.** The special oracle in `evalplus/eval/__init__.py` `unsafe_execute` reads:

    if "find_zero" == entry_point:
        assert abs(_poly(*inp, out)) <= atol
        continue

- The `continue` skips `details[i] = True` and `progress.value += 1`.
- So `untrusted_check` receives `details[:0] == []`, finds `len(details) != len(inputs)`, and turns PASS into FAIL.
- Every HumanEval/32 solution therefore fails under evalplus 0.3.1, on Linux too, and the HumanEval+ ceiling is 163/164.
- The paired base vs adapter comparison is unaffected, since both always fail it.
- I kept EvalPlus's behaviour unpatched. Reports will note it and also give HumanEval+ excluding /32.

Hand-made cases (`score_subset`, base / plus status):

| case | expected | got |
|---|---|---|
| HumanEval/2, answer + 1e-9 (inside atol; EvalPlus enforces 1e-6 for floats) | pass / pass | pass / pass |
| HumanEval/2, answer + 1e-3 | fail / fail | fail / fail |
| HumanEval/32 find_zero, Newton + bisection root-finder | pass / pass | **fail / fail** (the upstream bug above) |
| HumanEval/32 find_zero, returns 0.0 | fail / fail | fail / fail |
| Mbpp/2 set-equality oracle, same elements in another order | pass / pass | pass / pass |
| Mbpp/2, wrong elements | fail / fail | fail / fail |
| HumanEval/0, `while True: pass` | fail / fail (EvalPlus's `time_limit` raises TimeoutException inside the test loop, which counts as a failure) | fail / fail |
| HumanEval/0, `sum(range(10**15))` (one C call the timer can't interrupt) | timeout / timeout | timeout / timeout |
| HumanEval/0, raises ValueError | fail / fail | fail / fail |
| HumanEval/0, calls `os.getcwd()` (set to None by `reliability_guard`) | fail / fail | fail / fail |
| HumanEval/0, correct but written differently | pass / pass | pass / pass |
| HumanEval/0, prints 100 kB and calls `sys.exit` at import | fail / fail | fail / fail |
| HumanEval/0, `io.open('C:\AI\portal\server.py')` (the sandbox denies it) | fail / fail | fail / fail |

Difference from EvalPlus on Linux: `_thread.interrupt_main` can't wake a blocking `time.sleep()` on Windows (only a real signal sets CPython's SIGINT event). A solution sleeping past its limit is therefore ended by the process timeout ("timeout") where Linux raises inside the sleep ("fail"). Both are non-passing, so pass@1 is the same.

## Step D: llama-server b11435 checks (`results/template-budget-check`)

- **Server**: the scoring server: `GGML_VK_DISABLE_COOPMAT=1`, `-c 262144 -np 4 -ctk q8_0 -ctv q8_0 -fa on -ngl 999 --ctx-checkpoints 0 --cache-ram 0 --reasoning-format deepseek`, base GGUF (Ollama `qwen3.8:27b` Q4_K_M blob `sha256-f5f1dd89…`).
- **Context**: n_ctx_slot = 65,536 per slot, so 40,960 tokens fit.
- **Template**: the `/apply-template` output for 5 prompts × enable_thinking true/false equals the HF snapshot tokenizer's `apply_chat_template(add_generation_prompt=True, enable_thinking=…)`.
  - The 5 prompts: HumanEval/0, HumanEval/32, Mbpp/2, an OpenCodeInstruct-style prompt, and Unicode with a tab.
  - **All 10 match character for character and token ID for token ID.** No `--chat-template-file` is needed.
- **Separation**: with `--reasoning-format deepseek`, `reasoning_content` and `content` come back separated, and `content` has no think tags.
- **Reasoning budget**: b11435 enforces it in the sampler.
  - Both the server flag `--reasoning-budget 64` and the per-request field `reasoning_budget_tokens: 64` stop the trace at exactly 64 tokens.
  - The forced end is `\n</think>\n\n`, and the model then writes its answer normally (242 tokens in all, finish `stop`).
- **Budget method chosen**: server-side. The scoring server gets `--reasoning-budget 32768` (same flag for base and adapter); the generation server gets `--reasoning-budget 6144`. No client-side two-phase completion is needed.
- **Speed**: single-request decode on the scoring server was about 12 tok/s.

## Choices made here

- **Sampler.** top_k 20 and min_p 0 are written out. llama-server's defaults are top_k 40 and min_p 0.05, while EvalPlus's OpenAI backend sends only temperature and top_p. top_k 20 is the model's `generation_config.json` value, which a vLLM/OpenAI server would apply.
- **No system message.** EvalPlus 0.3.1's `openai_request.make_request` also sends a system message ("You are a helpful assistant good at coding."). As in my training and generation rows, I send no system message. The Qwen3.8 template then inserts its own "Reasoning effort is set to xhigh…" system turn, the same one every training row gets.
- **Extra server flags.** `--cache-ram 0` and `-to 21600` are added to the scoring flags. Neither changes outputs: no RAM prompt cache is used, and a 6 h socket timeout covers 40,960-token requests.
