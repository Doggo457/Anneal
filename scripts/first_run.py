"""First real LoRA run (Anneal, 19 Oct 2026): one controller for every stage, launched from the Training page.

  1 calibrate  20 EvalPlus problems (10 HumanEval+, 10 MBPP+, chosen with seed 42), seed 1, scoring settings:
               setup A (-np 4) then setup B (-np 4 + MTP, --spec-type draft-mtp). B is used only if its total
               generated tok/s is at least 3% higher. Projects hours for a 542-problem pass.
  2 generate   base GGUF on the chosen setup with --reasoning-budget 6144, -c 65536 (16k per slot), Vulkan coopmat
               on (no LoRA): scripts/generate_traces.py on OpenCodeInstruct (tests run in the sandbox) until 520
               rows are kept, then prepare_dataset.py --from-jsonl. (Cut from 1,020 on 19 Oct 2026.)
  3 mix        mix_datasets.py: 500 self-generated rows + 1,500 OpenCodeReasoning rows (ocr_first 1,000 +
               ocr_first_b 500, a second draw de-duplicated against the first and the loop's held-out sets).
  4 train      train_longrun.py with experiments/2026-10-19_first-run.yaml (about 250 steps) in <run>/train.
  5 convert    the adapter to a GGUF LoRA (llama.cpp b11435 convert_lora_to_gguf.py, config-only --base).
  6 sanity     answer-token loss on the mix's val rows, base vs adapter (scripts/val_loss.py); 3 sample generations
               with --lora (thinks, answers, no loops).
  7 score      base seed 1, then adapter seed 1: all 164 + 378 problems on the identical scoring server; EvalPlus's
               evaluator with the sandboxed checker (scripts/evalplus_win.py).
  8 report     <run>/report.md: pass@1 base and base+extra per dataset for both, paired bootstrap 95% CI (10,000
               resamples over problems) and paired permutation p-value of the difference, reasoning tokens, finish
               reasons, generation keep rate and timings, training loss, val loss.

Every stage leaves <run>/stages/<stage>.done.json when it is complete and is skipped after that; work inside a
stage is resumable (generation by task or prompt, training from its checkpoint). STOP in the run folder (the
Training page's Pause) stops the current stage cleanly: llama-server is stopped (in-flight requests are redone on
resume), generate_traces.py is ended, training saves a checkpoint. A llama-server that won't start is retried
once; a stage that fails is retried once with a fresh server, then the run fails loudly.

Scoring server (identical for base and adapter): GGML_VK_DISABLE_COOPMAT=1, -c 262144 -np 4 -ctk q8_0 -ctv q8_0 -fa on
-ngl 999 --ctx-checkpoints 0 --cache-ram 0 --reasoning-format deepseek --reasoning-budget 32768, thinking on,
temperature 0.6, top_p 0.95, top_k 20, min_p 0, 40,960 tokens per answer; the adapter via --lora.

  first_run.py [--run-dir DIR] [--resume] [--only STAGE] [--stop-after STAGE]
"""

import argparse
import json
import math
import os
import random
import re
import statistics
import subprocess
import sys
import threading
import time
import zlib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import evalplus_win as ew  # noqa: E402
from anneal import llama, sandbox  # noqa: E402
from anneal.runtime import RESULTS_DIR, Run  # noqa: E402

PY = sys.executable
SNAP = llama.SNAPSHOT
PORT = 18090
STAGES = ["calibrate", "generate", "mix", "train", "convert", "sanity", "score", "report"]
SAMPLER_FLAGS = ["--temp", "0.6", "--top-p", "0.95", "--top-k", "20", "--min-p", "0"]  # server defaults too
BUDGET_SCORE = 32768
BUDGET_GEN = 6144
OCI_FILES = [os.path.join(ROOT, "models", "hf", "hub", "datasets--nvidia--OpenCodeInstruct", "snapshots",
                          "8f3ba5bafe4d6e8db46082cf7ae6741bc370604d", "data", f"train-000{n}-of-00050.parquet")
             for n in ("00", "49")]
RAW_SELF = os.path.join(ROOT, "data", "raw", "self_oci_r1.jsonl")
PROC_SELF = os.path.join(ROOT, "data", "processed", "self_oci_r1")
PROC_OCR = os.path.join(ROOT, "data", "processed", "ocr_first")
PROC_OCR_B = os.path.join(ROOT, "data", "processed", "ocr_first_b")
MIX = os.path.join(ROOT, "data", "processed", "mix_first_2k")
TRAIN_CONFIG = os.path.join(ROOT, "experiments", "2026-10-19_first-run.yaml")
CONVERT = os.path.join(ROOT, "llama.cpp", "b11435-src", "llama.cpp-b11435", "convert_lora_to_gguf.py")
GEN_TARGET = 520  # 500 train + 10 val, with room for rows the 8,192-token check drops
SANITY_PROMPTS = [
    ("function", "Write a Python function `merge_intervals(intervals)` that merges overlapping closed intervals given "
                 "as a list of [start, end] pairs and returns them sorted by start. Include a short docstring."),
    ("stdio", "Read an integer n and then n integers from standard input, and print the length of the longest "
              "strictly increasing contiguous run. Write a complete Python 3 program in a single ```python block."),
    ("explain", "Explain the difference between a Python list and a tuple, and when you would choose each. "
                "Keep it short and give one code example."),
]


class Paused(Exception):
    pass


class Controller:
    def __init__(self, args, run):
        self.args, self.run, self.dir = args, run, run.dir
        self.stop = threading.Event()
        self.server = None   # the llama.Server in use
        self.child = None    # the subprocess.Popen in use
        self.child_stop_dir = None  # a child that pauses itself on <dir>/STOP (training)
        self.summary = {}
        os.makedirs(os.path.join(self.dir, "stages"), exist_ok=True)
        threading.Thread(target=self._watch, daemon=True).start()

    # ----------------------------------------------------------------------------------------- plumbing

    def log(self, msg):
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)

    def _watch(self):
        while not self.stop.is_set():
            time.sleep(5)
            if self.run.stop_requested():
                self.log("STOP file found: stopping the current stage")
                self.stop.set()
                if self.child_stop_dir:
                    with open(os.path.join(self.child_stop_dir, "STOP"), "w", encoding="utf-8") as fh:
                        fh.write("pause requested through first_run.py\n")
                elif self.child is not None and self.child.poll() is None:
                    self.child.terminate()
                srv = self.server
                if srv is not None and srv.proc is not None and srv.proc.poll() is None:
                    srv.proc.terminate()  # in-flight requests fail and are redone on resume

    def stopping(self):
        return self.stop.is_set()

    def check_stop(self):
        if self.stop.is_set():
            raise Paused()

    def done_path(self, stage):
        return os.path.join(self.dir, "stages", f"{stage}.done.json")

    def done(self, stage):
        p = self.done_path(stage)
        if os.path.exists(p):
            with open(p, encoding="utf-8") as fh:
                return json.load(fh)
        return None

    def mark_done(self, stage, data):
        data = dict(data, finished=time.strftime("%Y-%m-%d %H:%M:%S"))
        with open(self.done_path(stage), "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=1, default=str)
        self.summary[stage] = data
        self.run.metric(event="stage_done", stage=stage)

    def status(self, stage, detail=""):
        self.run.write_status(state="running", step=STAGES.index(stage) + 1, max_steps=len(STAGES), stage=stage,
                              detail=detail)

    def serve(self, flags, name, coopmat=False):
        srv = llama.Server(flags, os.path.join(self.dir, "servers", f"{name}.log"), PORT, coopmat=coopmat)
        self.log(f"llama-server {name}: {srv.describe()}")
        srv.start()
        self.server = srv
        self.log(f"llama-server {name} up in {srv.load_s} s")
        return srv

    def unserve(self):
        if self.server is not None:
            self.server.stop()
            self.server = None

    def run_child(self, cmd, log_name, stop_dir=None, monitor=None, interval=30):
        """Run a subprocess with its output in <run>/logs/<log_name>; monitor() is called every `interval`
        seconds while it runs (it may return False to end the child). Returns the exit code."""
        os.makedirs(os.path.join(self.dir, "logs"), exist_ok=True)
        path = os.path.join(self.dir, "logs", log_name)
        self.log(f"$ {subprocess.list2cmdline(cmd)}   (log: {path})")
        env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
        with open(path, "a", encoding="utf-8", errors="replace") as fh:
            self.child = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=fh, stderr=subprocess.STDOUT,
                                          stdin=subprocess.DEVNULL, creationflags=0x08000000)
            self.child_stop_dir = stop_dir
            try:
                while True:
                    try:
                        rc = self.child.wait(interval)
                        break
                    except subprocess.TimeoutExpired:
                        if monitor is not None and monitor() is False:
                            self.child.terminate()
                            rc = self.child.wait(120)
                            break
            finally:
                self.child, self.child_stop_dir = None, None
        self.log(f"  exit code {rc}")
        return rc

    def scoring_flags(self, lora=None):
        cal = self.done("calibrate") or {}
        return llama.scoring_flags(lora=lora, extra=["--reasoning-budget", str(BUDGET_SCORE)] + SAMPLER_FLAGS
                                   + cal.get("chosen_extra_flags", []))

    # ----------------------------------------------------------------------------------------- stages

    def calibrate(self):
        he, mb = list(ew.load_problems("humaneval")), list(ew.load_problems("mbpp"))
        rng = random.Random(42)
        tasks = [("humaneval", t) for t in rng.sample(he, 10)] + [("mbpp", t) for t in rng.sample(mb, 10)]
        setups = {"A": [], "B": ["--spec-type", "draft-mtp"]}
        out = {"tasks": [t for _, t in tasks], "setups": {}}
        for name, extra in setups.items():
            res_path = os.path.join(self.dir, "calibrate", f"setup-{name}.json")
            if os.path.exists(res_path):
                out["setups"][name] = json.load(open(res_path, encoding="utf-8"))
                continue
            self.check_stop()
            self.status("calibrate", f"setup {name}")
            d = os.path.join(self.dir, "calibrate", name)
            raw = {ds: os.path.join(d, f"{ds}.raw.jsonl") for ds in ew.DATASETS}
            flags = llama.scoring_flags(extra=["--reasoning-budget", str(BUDGET_SCORE)] + SAMPLER_FLAGS + extra)
            try:
                self.serve(flags, f"calibrate-{name}")
            except RuntimeError as exc:
                if name == "B":
                    self.log(f"setup B could not start ({exc}); using A")
                    out["setups"][name] = {"error": str(exc)}
                    continue
                raise
            try:
                gen = ew.generate_tasks(tasks, self.server.url, raw, seed=1, concurrency=4, max_tokens=ew.MAX_TOKENS,
                                        reasoning_budget=BUDGET_SCORE, log=self.log, stop_check=self.stopping,
                                        on_row=lambda r, n, tot, tps: self.run.metric(stage="calibrate", setup=name,
                                                                                      done=n, of=tot, tok_s=round(tps, 2)))
                draft = self.draft_stats(self.server.log_path) if extra else None
            finally:
                self.unserve()
            rows = [r for ds in ew.DATASETS for r in ew.read_jsonl(raw[ds])]
            scores = {}
            for ds in ew.DATASETS:
                if os.path.exists(raw[ds]):
                    san = ew.sanitize_file(ds, raw[ds])
                    scores.update(ew.score_subset(ds, ew.read_jsonl(san)))
            res = self.token_stats(rows)
            res.update(session_rows=len(gen["rows"]), session_wall_s=round(gen["wall_s"], 1),
                       session_generated_tokens=gen["generated_tokens"],
                       total_generated_tok_s=round(gen["generated_tokens"] / gen["wall_s"], 2) if gen["wall_s"] else None,
                       pass_base=sum(v["base_pass"] for v in scores.values()),
                       pass_plus=sum(v["plus_pass"] for v in scores.values()), scored=len(scores),
                       per_task={t: dict(v, tokens=next((r["completion_tokens"] for r in rows if r["task_id"] == t),
                                                        None)) for t, v in scores.items()},
                       flags=flags, draft=draft)
            with open(res_path, "w", encoding="utf-8") as fh:
                json.dump(res, fh, indent=1)
            out["setups"][name] = res
            self.log(f"setup {name}: {res['total_generated_tok_s']} tok/s, completion tokens mean "
                     f"{res['completion_tokens']['mean']}, pass@1 {res['pass_base']}/{res['scored']} base, "
                     f"{res['pass_plus']}/{res['scored']} plus")
        a, b = out["setups"]["A"], out["setups"].get("B") or {}
        use_b = bool(b.get("total_generated_tok_s")) and b["total_generated_tok_s"] >= 1.03 * a["total_generated_tok_s"]
        chosen = "B" if use_b else "A"
        tps = (b if use_b else a)["total_generated_tok_s"]
        mean_tok = statistics.mean([x for s in (a, b) if s.get("completion_tokens")
                                    for x in [s["completion_tokens"]["mean"]]])
        out.update(chosen=chosen, chosen_extra_flags=setups[chosen], speedup_b_over_a=round(
            b["total_generated_tok_s"] / a["total_generated_tok_s"], 3) if b.get("total_generated_tok_s") else None,
            projected_hours_per_542=round(542 * mean_tok / tps / 3600, 1))
        self.log(f"calibration: chosen setup {chosen}; projected {out['projected_hours_per_542']} h per 542-problem pass")
        self.mark_done("calibrate", out)

    @staticmethod
    def token_stats(rows):
        def st(xs):
            xs = [x for x in xs if x is not None]
            return {"mean": round(statistics.mean(xs), 1), "median": statistics.median(xs), "max": max(xs),
                    "min": min(xs)} if xs else None
        fins = {}
        for r in rows:
            fins[r["finish_reason"]] = fins.get(r["finish_reason"], 0) + 1
        return {"rows": len(rows), "completion_tokens": st([r["completion_tokens"] for r in rows]),
                "reasoning_tokens": st([r["reasoning_tokens"] for r in rows]),
                "answer_tokens": st([r["answer_tokens"] for r in rows]), "finish_reasons": fins,
                "budget_hits": sum(bool(r.get("budget_hit")) for r in rows),
                "seconds": st([r["seconds"] for r in rows])}

    @staticmethod
    def draft_stats(log_path):
        try:
            text = open(log_path, encoding="utf-8", errors="replace").read()
        except OSError:
            return None
        rows = re.findall(r"draft acceptance = ([0-9.]+) \(\s*(\d+) accepted /\s*(\d+) generated\)", text)
        acc, gen = sum(int(a) for _, a, _ in rows), sum(int(g) for _, _, g in rows)
        return {"requests": len(rows), "accepted": acc, "drafted": gen, "acceptance": round(acc / gen, 3) if gen else None}

    def generate(self):
        cal = self.done("calibrate")
        extra = cal["chosen_extra_flags"]
        flags = ["-c", "65536", "-np", "4", "-ctk", "q8_0", "-ctv", "q8_0", "-fa", "on", "-ngl", "999",
                 "--ctx-checkpoints", "0", "--cache-ram", "0", "--reasoning-format", "deepseek",
                 "--reasoning-budget", str(BUDGET_GEN)] + SAMPLER_FLAGS + extra
        state = {"restarts": 0, "t0": time.time(), "kept0": self.count_lines(RAW_SELF)}
        info_path = os.path.join(self.dir, "generate", "progress.json")
        os.makedirs(os.path.dirname(info_path), exist_ok=True)
        while self.count_lines(RAW_SELF) < GEN_TARGET:
            self.check_stop()
            self.status("generate", f"{self.count_lines(RAW_SELF)}/{GEN_TARGET} rows kept")
            self.serve(flags, "generate", coopmat=True)
            cmd = [PY, "-u", os.path.join(ROOT, "scripts", "generate_traces.py"), "--url", self.server.url,
                   "--dataset", "parquet", "--data-files", ",".join(OCI_FILES), "--prompt-field", "input",
                   "--tests-field", "unit_tests", "--min-test-score", "1.0", "--decontaminate", "--run-tests",
                   "--concurrency", "4", "--max-samples", str(GEN_TARGET), "--resume", "--out", RAW_SELF]
            server_died = []

            def monitor():
                kept = self.count_lines(RAW_SELF)
                rej = self.rejected_counts()
                el = time.time() - state["t0"]
                prog = {"kept": kept, "rejected": rej, "attempts": kept + sum(rej.values()),
                        "kept_per_hour_this_session": round(3600 * (kept - state["kept0"]) / el, 2) if el else None}
                self.run.metric(stage="generate", **prog)
                self.status("generate", f"{kept}/{GEN_TARGET} rows kept, {sum(rej.values())} rejected")
                with open(info_path, "w", encoding="utf-8") as fh:
                    json.dump(prog, fh, indent=1)
                if not self.server.alive() and not self.stopping():
                    server_died.append(time.time())
                    return False
                return True

            try:
                rc = self.run_child(cmd, "generate_traces.log", monitor=monitor)
            finally:
                self.unserve()
            self.check_stop()
            if server_died:
                state["restarts"] += 1
                self.log(f"llama-server died during generation (restart {state['restarts']})")
                if state["restarts"] > 1:
                    raise RuntimeError("llama-server died twice during generation")
                continue
            if rc != 0:
                raise RuntimeError(f"generate_traces.py exited with {rc}")
            break  # the prompt pool ran out, or the target is reached
        kept = self.count_lines(RAW_SELF)
        self.log(f"generation done: {kept} rows kept, rejected {self.rejected_counts()}")
        self.check_stop()
        rc = self.run_child([PY, "-u", os.path.join(ROOT, "scripts", "prepare_dataset.py"), "--from-jsonl", RAW_SELF,
                             "--require-think", "--tokenizer", SNAP, "--max-tokens", "8192", "--decontaminate",
                             "--out", PROC_SELF], "prepare_self.log")
        if rc != 0:
            raise RuntimeError(f"prepare_dataset.py exited with {rc}")
        self.mark_done("generate", {"kept": kept, "rejected": self.rejected_counts(),
                                    "processed_train": self.count_lines(os.path.join(PROC_SELF, "train.jsonl")),
                                    "processed_val": self.count_lines(os.path.join(PROC_SELF, "val.jsonl")),
                                    "server_flags": flags, "server_restarts": state["restarts"]})

    @staticmethod
    def count_lines(path):
        if not os.path.exists(path):
            return 0
        with open(path, encoding="utf-8") as fh:
            return sum(1 for line in fh if line.strip())

    @staticmethod
    def rejected_counts():
        out = {}
        p = RAW_SELF + ".rejected"
        if os.path.exists(p):
            with open(p, encoding="utf-8") as fh:
                for line in fh:
                    reason = line.strip().partition("\t")[2] or "unknown"
                    out[reason] = out.get(reason, 0) + 1
        return out

    def mix(self):
        self.status("mix")
        rc = self.run_child([PY, "-u", os.path.join(ROOT, "scripts", "mix_datasets.py"),
                             "--source", f"self={PROC_SELF}:500", "--source", f"ocr={PROC_OCR}:1000",
                             "--source", f"ocr_b={PROC_OCR_B}:500",
                             "--tokenizer", SNAP, "--out", MIX], "mix.log")
        if rc != 0:
            raise RuntimeError(f"mix_datasets.py exited with {rc}")
        self.mark_done("mix", {"train_rows": self.count_lines(os.path.join(MIX, "train.jsonl")),
                               "val_rows": self.count_lines(os.path.join(MIX, "val.jsonl")),
                               "card": open(os.path.join(MIX, "data_card.md"), encoding="utf-8").read()})

    def train(self):
        tdir = os.path.join(self.dir, "train")
        os.makedirs(tdir, exist_ok=True)
        self.status("train")
        offset = [0]
        mpath = os.path.join(tdir, "metrics.jsonl")

        def relay():
            # copy the trainer's per-step rows into this run's metrics, so the Training page charts the loss
            if os.path.exists(mpath):
                with open(mpath, "rb") as fh:
                    fh.seek(offset[0])
                    data = fh.read()
                cut = data.rfind(b"\n") + 1
                offset[0] += cut
                for line in data[:cut].decode("utf-8", "replace").splitlines():
                    try:
                        m = json.loads(line)
                    except ValueError:
                        continue
                    if "loss" in m and "step" in m and "event" not in m:
                        self.run.metric(stage="train", **m)
            st = self.read_json(os.path.join(tdir, "status.json")) or {}
            self.status("train", f"step {st.get('step')}/{st.get('max_steps')} ({st.get('state')})")
            return True

        if os.path.exists(mpath):
            offset[0] = os.path.getsize(mpath)
        cmd = [PY, "-u", os.path.join(ROOT, "scripts", "train_longrun.py"), "--config", TRAIN_CONFIG,
               "--run-dir", tdir, "--resume"]
        rc = self.run_child(cmd, "train_longrun.log", stop_dir=tdir, monitor=relay, interval=20)
        relay()
        st = self.read_json(os.path.join(tdir, "status.json")) or {}
        if st.get("state") == "paused" or self.stopping():
            raise Paused()
        if rc != 0 or st.get("state") != "finished":
            raise RuntimeError(f"training ended with exit code {rc}, state {st.get('state')}: {st.get('error')}")
        s = self.read_json(os.path.join(tdir, "summary.json")) or {}
        self.mark_done("train", {k: s.get(k) for k in ("global_step", "total_steps", "trainable_params", "rows",
                                                        "rows_dropped_too_long", "tokens", "peak_reserved_gib",
                                                        "session_seconds", "adapter", "metrics", "data_sha256")})

    @staticmethod
    def read_json(path):
        try:
            with open(path, encoding="utf-8") as fh:
                return json.load(fh)
        except (OSError, ValueError):
            return None

    def adapter_gguf(self):
        return os.path.join(self.dir, "convert", "first-run-2k-r16-F16-LoRA.gguf")

    def convert(self):
        self.status("convert")
        out = self.adapter_gguf()
        os.makedirs(os.path.dirname(out), exist_ok=True)
        rc = self.run_child([PY, "-u", CONVERT, "--base", llama.LORA_BASE_CONFIG, "--outtype", "f16", "--outfile", out,
                             os.path.join(self.dir, "train", "adapter")], "convert_lora.log")
        if rc != 0 or not os.path.exists(out):
            raise RuntimeError(f"convert_lora_to_gguf.py exited with {rc}")
        log = open(os.path.join(self.dir, "logs", "convert_lora.log"), encoding="utf-8", errors="replace").read()
        n_a = len(re.findall(r"\.lora_a,", log))
        self.mark_done("convert", {"gguf": out, "bytes": os.path.getsize(out), "lora_a_tensors": n_a,
                                   "expected_lora_a_tensors": 256})

    def sanity(self):
        self.status("sanity", "val loss")
        vdir = os.path.join(self.dir, "sanity", "val-loss")
        if not os.path.exists(os.path.join(vdir, "summary.json")):
            rc = self.run_child([PY, "-u", os.path.join(ROOT, "scripts", "val_loss.py"), "--adapter",
                                 os.path.join(self.dir, "train", "adapter"), "--val-file",
                                 os.path.join(MIX, "val.jsonl"), "--run-dir", vdir], "val_loss.log")
            self.check_stop()
            if rc != 0:
                raise RuntimeError(f"val_loss.py exited with {rc}")
        val = self.read_json(os.path.join(vdir, "summary.json"))
        self.check_stop()
        self.status("sanity", "sample generations with --lora")
        self.serve(self.scoring_flags(lora=self.adapter_gguf()), "sanity-lora")
        samples = []
        try:
            for name, prompt in SANITY_PROMPTS:
                self.check_stop()
                r = ew.chat_request(self.server.url, prompt, seed=7, max_tokens=ew.MAX_TOKENS)
                samples.append(dict(name=name, **{k: r[k] for k in ("finish_reason", "completion_tokens", "seconds")},
                                    reasoning_chars=len(r["reasoning"]), content_chars=len(r["content"]),
                                    loop_suspect=loop_suspect(r["reasoning"]) or loop_suspect(r["content"]),
                                    reasoning_head=r["reasoning"][:400], content_head=r["content"][:600]))
                self.log(f"sample {name}: {samples[-1]['finish_reason']}, {r['completion_tokens']} tokens, reasoning "
                         f"{len(r['reasoning'])} chars, answer {len(r['content'])} chars, loop {samples[-1]['loop_suspect']}")
        finally:
            self.unserve()
        ok = all(s["finish_reason"] == "stop" and s["reasoning_chars"] > 0 and s["content_chars"] > 0
                 and not s["loop_suspect"] for s in samples)
        self.mark_done("sanity", {"val_loss": {k: v for k, v in (val or {}).items() if k != "per_row"},
                                  "samples": samples, "samples_ok": ok})
        if not ok:
            self.log("WARNING: a sample generation with the adapter did not think, answer and stop cleanly")

    def score(self):
        for model in ("base", "adapter"):
            res_path = os.path.join(self.dir, "score", f"{model}.json")
            if os.path.exists(res_path):
                continue
            d = os.path.join(self.dir, "score", model)
            raw = {ds: os.path.join(d, f"{ds}.raw.jsonl") for ds in ew.DATASETS}
            jobs = [(ds, t) for ds in ew.DATASETS for t in ew.load_problems(ds)]
            for attempt in range(2):
                self.check_stop()
                self.status("score", f"{model}: generating")
                self.serve(self.scoring_flags(lora=self.adapter_gguf() if model == "adapter" else None),
                           f"score-{model}")
                try:
                    ew.generate_tasks(jobs, self.server.url, raw, seed=1, concurrency=4, max_tokens=ew.MAX_TOKENS,
                                      reasoning_budget=BUDGET_SCORE, log=self.log, stop_check=self.stopping,
                                      on_row=lambda r, n, tot, tps: (self.run.metric(stage="score", model=model, done=n,
                                                                                     of=tot, tok_s=round(tps, 2)),
                                                                     self.status("score", f"{model}: {n}/{tot}")))
                    break
                except ew.Stopped:
                    raise Paused()
                except RuntimeError as exc:
                    if attempt == 1 or self.stopping():
                        raise
                    self.log(f"generation failed ({exc}); restarting llama-server once")
                finally:
                    self.unserve()
            self.check_stop()
            self.status("score", f"{model}: evaluating")
            res = {"datasets": {}}
            for ds in ew.DATASETS:
                san = ew.sanitize_file(ds, raw[ds])
                r = ew.evaluate(ds, san, parallel=8)
                res["datasets"][ds] = r
                self.log(f"{model} {ds}: pass@1 base {r['base']:.4f}, base+extra {r['plus']:.4f}")
            res["tokens"] = self.token_stats([r for ds in ew.DATASETS for r in ew.read_jsonl(raw[ds])])
            res["check_stats"] = dict(ew.CHECK_STATS)
            with open(res_path, "w", encoding="utf-8") as fh:
                json.dump(res, fh, indent=1)
        self.mark_done("score", {m: {ds: {k: v for k, v in self.read_json(
            os.path.join(self.dir, "score", f"{m}.json"))["datasets"][ds].items() if k != "per_task"}
            for ds in ew.DATASETS} for m in ("base", "adapter")})

    def report(self):
        self.status("report")
        text, stats = build_report(self)
        with open(os.path.join(self.dir, "report.md"), "w", encoding="utf-8") as fh:
            fh.write(text)
        self.mark_done("report", {"path": os.path.join(self.dir, "report.md"), "stats": stats})


def loop_suspect(text):
    """A line of 20+ characters repeated 8+ times, or text over 4,000 characters that zlib squeezes below 12%."""
    lines = [ln.strip() for ln in text.splitlines() if len(ln.strip()) >= 20]
    if lines and max(lines.count(x) for x in set(lines)) >= 8:
        return True
    return len(text) > 4000 and len(zlib.compress(text.encode("utf-8"))) / len(text.encode("utf-8")) < 0.12


# --------------------------------------------------------------------------------------------- report

def paired(a, b, n_boot=10000, seed=0):
    """a, b: per-problem 0/1 for adapter and base. Difference of means, bootstrap 95% CI over problems, two-sided
    paired permutation (sign-flip) p-value."""
    rng = random.Random(seed)
    n = len(a)
    d = [x - y for x, y in zip(a, b)]
    diff = sum(d) / n
    boots = []
    for _ in range(n_boot):
        s = 0
        for _ in range(n):
            s += d[rng.randrange(n)]
        boots.append(s / n)
    boots.sort()
    lo, hi = boots[int(0.025 * n_boot)], boots[int(0.975 * n_boot) - 1]
    nz = [x for x in d if x]
    extreme = 0
    for _ in range(n_boot):
        s = sum(x if rng.random() < 0.5 else -x for x in nz)
        extreme += abs(s) >= abs(sum(d)) - 1e-12
    return {"n": n, "adapter": sum(a) / n, "base": sum(b) / n, "diff": diff, "ci95": [lo, hi],
            "p_permutation": (extreme + 1) / (n_boot + 1), "adapter_only": sum(1 for x in d if x > 0),
            "base_only": sum(1 for x in d if x < 0)}


def build_report(c):
    load = lambda m: c.read_json(os.path.join(c.dir, "score", f"{m}.json"))  # noqa: E731
    base, ada = load("base"), load("adapter")
    stats, lines = {}, []
    cal, gen, mix, train, sanity = (c.done(s) or {} for s in ("calibrate", "generate", "mix", "train", "sanity"))
    lines += ["# First run: Qwen3.8-27B + first LoRA (first-run-2k-r16) vs base", "",
              f"Generated {time.strftime('%Y-%m-%d %H:%M')} by scripts/first_run.py from {c.dir}.", "",
              "## EvalPlus pass@1 (one sample per problem, seed 1, thinking on, identical scoring server)", "",
              "| benchmark | base | adapter | difference | bootstrap 95% CI | permutation p | adapter-only / base-only |",
              "|---|---|---|---|---|---|---|"]
    combined = {"base": ([], []), "plus": ([], [])}
    for ds, label in (("humaneval", "HumanEval+"), ("mbpp", "MBPP+")):
        pb, pa = base["datasets"][ds]["per_task"], ada["datasets"][ds]["per_task"]
        tasks = sorted(pb)
        for key, name in (("base_pass", f"{label} base tests"), ("plus_pass", f"{label} base+extra")):
            b = [int(pb[t][key]) for t in tasks]
            a = [int(pa[t][key]) for t in tasks]
            s = paired(a, b)
            stats[f"{ds}:{key}"] = s
            combined["base" if key == "base_pass" else "plus"][0].extend(a)
            combined["base" if key == "base_pass" else "plus"][1].extend(b)
            lines.append(f"| {name} | {s['base']:.4f} | {s['adapter']:.4f} | {s['diff']:+.4f} | "
                         f"[{s['ci95'][0]:+.4f}, {s['ci95'][1]:+.4f}] | {s['p_permutation']:.4f} | "
                         f"{s['adapter_only']} / {s['base_only']} |")
        if ds == "humaneval":
            ex = [t for t in tasks if t != "HumanEval/32"]
            s = paired([int(pa[t]["plus_pass"]) for t in ex], [int(pb[t]["plus_pass"]) for t in ex])
            stats["humaneval:plus_pass:excl32"] = s
            lines.append(f"| HumanEval+ base+extra excl. HumanEval/32 | {s['base']:.4f} | {s['adapter']:.4f} | "
                         f"{s['diff']:+.4f} | [{s['ci95'][0]:+.4f}, {s['ci95'][1]:+.4f}] | {s['p_permutation']:.4f} | "
                         f"{s['adapter_only']} / {s['base_only']} |")
    for key, name in (("base", "all 542, base tests"), ("plus", "all 542, base+extra")):
        s = paired(*combined[key])
        stats[f"all:{key}"] = s
        lines.append(f"| {name} | {s['base']:.4f} | {s['adapter']:.4f} | {s['diff']:+.4f} | "
                     f"[{s['ci95'][0]:+.4f}, {s['ci95'][1]:+.4f}] | {s['p_permutation']:.4f} | "
                     f"{s['adapter_only']} / {s['base_only']} |")
    lines += ["", "HumanEval/32 (find_zero) fails for every solution under evalplus 0.3.1 (results/"
              "evalplus_windows_findings.md), so HumanEval+ tops out at 163/164; the row without it is given too.",
              "CI: 10,000 bootstrap resamples over problems; p: 10,000 paired sign-flip permutations, two-sided.", "",
              "## Answers", "", "| | base | adapter |", "|---|---|---|"]
    tb, ta = base["tokens"], ada["tokens"]
    for k in ("completion_tokens", "reasoning_tokens", "answer_tokens"):
        lines.append(f"| {k.replace('_', ' ')} mean / median / max | {tb[k]['mean']} / {tb[k]['median']} / {tb[k]['max']}"
                     f" | {ta[k]['mean']} / {ta[k]['median']} / {ta[k]['max']} |")
    lines.append(f"| finish reasons | {tb['finish_reasons']} | {ta['finish_reasons']} |")
    lines.append(f"| reasoning budget (32,768) hit | {tb['budget_hits']} | {ta['budget_hits']} |")
    lines.append(f"| seconds per answer mean / median | {tb['seconds']['mean']} / {tb['seconds']['median']} | "
                 f"{ta['seconds']['mean']} / {ta['seconds']['median']} |")
    lines += ["", "## Calibration (20 problems)", "", f"Chosen setup: {cal.get('chosen')} "
              f"(B / A speed {cal.get('speedup_b_over_a')}); projected {cal.get('projected_hours_per_542')} h per "
              f"542-problem pass.", ""]
    for name, s in (cal.get("setups") or {}).items():
        if s.get("total_generated_tok_s"):
            lines.append(f"- {name}: {s['total_generated_tok_s']} tok/s, completion tokens mean "
                         f"{s['completion_tokens']['mean']}, pass@1 {s['pass_base']}/{s['scored']} base, "
                         f"{s['pass_plus']}/{s['scored']} base+extra, finish {s['finish_reasons']}")
    lines += ["", "## Training data", "",
              f"Self-generated rows kept: {gen.get('kept')} (rejected: {gen.get('rejected')}); processed "
              f"{gen.get('processed_train')} train / {gen.get('processed_val')} val.", "",
              mix.get("card", ""), "", "## Training", ""]
    tdir = os.path.join(c.dir, "train")
    losses = []
    if os.path.exists(os.path.join(tdir, "metrics.jsonl")):
        with open(os.path.join(tdir, "metrics.jsonl"), encoding="utf-8") as fh:
            for line in fh:
                m = json.loads(line)
                if "loss" in m and "step" in m and "event" not in m:
                    losses.append(m)
    if losses:
        def window(a, b):
            xs = [m["loss"] for m in losses if a <= m["step"] <= b]
            return round(statistics.mean(xs), 4) if xs else None
        n = losses[-1]["step"]
        lines.append(f"{n} optimizer steps; mean loss steps 1-10 {window(1, 10)}, middle {window(n // 2 - 5, n // 2 + 5)},"
                     f" last 10 {window(n - 9, n)}; final {losses[-1]['loss']:.4f}. Mean step "
                     f"{statistics.mean(m.get('seconds', 0) for m in losses):.0f} s.")
    lines.append(f"Trainer summary: {json.dumps({k: train.get(k) for k in ('trainable_params', 'rows', 'tokens', 'peak_reserved_gib', 'session_seconds')})}")
    vl = (sanity.get("val_loss") or {})
    lines += ["", "## Validation loss (mix val rows, base vs adapter, same 4-bit model)", ""]
    for key in [k for k in vl if k == "all" or k.startswith("source=")]:
        v = vl[key]
        lines.append(f"- {key}: trained-token loss {v['base']['trained_token_loss']:.4f} -> "
                     f"{v['adapter']['trained_token_loss']:.4f}; answer-token loss "
                     f"{v['base']['answer_token_loss']:.4f} -> {v['adapter']['answer_token_loss']:.4f} ({v['rows']} rows)")
    lines += ["", "## Sample generations with the adapter", ""]
    for s in sanity.get("samples") or []:
        lines.append(f"- {s['name']}: {s['finish_reason']}, {s['completion_tokens']} tokens, reasoning "
                     f"{s['reasoning_chars']} chars, answer {s['content_chars']} chars, loop suspected {s['loop_suspect']}")
    return "\n".join(lines) + "\n", stats


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run-dir", default=os.path.join(RESULTS_DIR, "first-run"))
    ap.add_argument("--resume", action="store_true", help="accepted for the Training page; stages resume anyway")
    ap.add_argument("--only", default=None, choices=STAGES)
    ap.add_argument("--stop-after", default=None, choices=STAGES)
    args = ap.parse_args()
    run = Run(args.run_dir)
    run.capture_output()
    run.write_status(state="starting", step=0, max_steps=len(STAGES))
    c = Controller(args, run)
    c.log(f"== first_run.py {vars(args)}")
    try:
        sandbox.require()
        for stage in STAGES:
            if args.only and stage != args.only:
                continue
            prev = c.done(stage)
            if prev is not None:
                c.summary[stage] = prev
                c.log(f"stage {stage}: done ({prev.get('finished')})")
                continue
            c.check_stop()
            c.log(f"== stage {stage}")
            c.status(stage)
            run.metric(event="stage_start", stage=stage)
            for attempt in range(2):
                try:
                    getattr(c, stage)()
                    break
                except (Paused, ew.Stopped):
                    raise Paused()
                except Exception as exc:
                    if attempt == 1 or c.stopping() or stage in ("train", "report"):
                        raise
                    c.log(f"stage {stage} failed ({type(exc).__name__}: {exc}); retrying once")
                    c.unserve()
            if args.stop_after == stage:
                break
        with open(os.path.join(run.dir, "summary.json"), "w", encoding="utf-8") as fh:
            json.dump(c.summary, fh, indent=1, default=str)
        run.write_status(state="finished", step=len(STAGES), max_steps=len(STAGES))
        c.log("first run complete" if not args.only and not args.stop_after else "requested stages complete")
        return 0
    except Paused:
        c.unserve()
        run.write_status(state="paused")
        run.metric(event="paused")
        c.log("paused; resume from the Training page")
        return 0
    except BaseException as exc:
        import traceback
        traceback.print_exc()
        c.unserve()
        run.write_status(state="failed", error=f"{type(exc).__name__}: {exc}")
        run.metric(event="failed", error=f"{type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
