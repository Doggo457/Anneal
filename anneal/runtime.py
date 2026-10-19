"""Run-time plumbing every training script shares, so the portal's Training page can follow and steer it.

A run lives in one folder (``--run-dir``, default C:\\AI\\training\\results\\<tag>):
  run.log          everything the script prints
  metrics.jsonl    one JSON line per optimizer step (loss, timing, memory) plus "event" lines
  status.json      the latest state: running / paused / finished / failed, step, max_steps
  STOP             created by the portal to pause: the script saves a checkpoint and exits cleanly
  summary.json     written by the script at the end
"""

import json
import math
import os
import sys
import threading
import time
import urllib.request

TRAINING_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(TRAINING_ROOT, "results")


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            try:
                s.write(data)
                s.flush()
            except (OSError, ValueError):
                pass

    def flush(self):
        for s in self.streams:
            try:
                s.flush()
            except (OSError, ValueError):
                pass

    def isatty(self):
        return False


def gib(n):
    return round(n / 2**30, 3)


class Run:
    """The run folder: log capture, metrics and status files, and the STOP file check."""

    def __init__(self, run_dir):
        self.dir = run_dir
        os.makedirs(run_dir, exist_ok=True)
        self.metrics_path = os.path.join(run_dir, "metrics.jsonl")
        self.status_path = os.path.join(run_dir, "status.json")
        self.stop_path = os.path.join(run_dir, "STOP")
        self.status = {"state": "starting", "pid": os.getpid(), "started": time.time(), "step": 0, "max_steps": None}

    def capture_output(self, name="run.log"):
        fh = open(os.path.join(self.dir, name), "a", encoding="utf-8", errors="replace", buffering=1)
        sys.stdout = Tee(sys.__stdout__, fh) if sys.__stdout__ else Tee(fh)
        sys.stderr = Tee(sys.__stderr__, fh) if sys.__stderr__ else Tee(fh)

    def write_status(self, **kw):
        self.status.update(kw, updated=time.time())
        tmp = self.status_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(self.status, fh, indent=1, default=str)
        os.replace(tmp, self.status_path)

    def metric(self, **kw):
        kw.setdefault("t", time.time())
        with open(self.metrics_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(kw, default=str) + "\n")

    def stop_requested(self):
        return os.path.exists(self.stop_path)


def trainer_callback(run, tokens_per_step=None, step_tokens=None):
    """A transformers TrainerCallback that streams metrics, honours STOP and keeps status.json current.

    tokens_per_step: a fixed token count per step for tokens_per_second. step_tokens: a function step -> (tokens,
    trained tokens) of that step's batch (e.g. CountingCollator.step_tokens), for exact per-step rates instead.
    """
    import torch
    from transformers import TrainerCallback

    class AnnealCallback(TrainerCallback):
        def __init__(self):
            self.t_step = None
            self.step_times = []
            self.paused = False

        def on_train_begin(self, args, state, control, **kw):
            run.write_status(state="running", max_steps=state.max_steps, step=state.global_step)
            run.metric(event="train_begin", max_steps=state.max_steps, step=state.global_step)

        def on_step_begin(self, args, state, control, **kw):
            torch.cuda.synchronize()
            self.t_step = time.time()

        def on_step_end(self, args, state, control, **kw):
            torch.cuda.synchronize()
            dt = time.time() - self.t_step
            self.step_times.append(dt)
            self.last = {"step": state.global_step, "seconds": round(dt, 3),
                         "peak_reserved_gib": gib(torch.cuda.max_memory_reserved()),
                         "peak_allocated_gib": gib(torch.cuda.max_memory_allocated())}
            counts = step_tokens(state.global_step) if step_tokens else None
            if counts:
                self.last.update(tokens=counts[0], trained_tokens=counts[1],
                                 tokens_per_second=round(counts[0] / dt, 1),
                                 trained_tokens_per_second=round(counts[1] / dt, 1))
            elif tokens_per_step:
                self.last["tokens_per_second"] = round(tokens_per_step / dt, 1)
            run.write_status(state="running", step=state.global_step, max_steps=state.max_steps)
            if run.stop_requested():
                # Pause: save a resumable checkpoint at this step, then end training cleanly.
                self.paused = True
                control.should_save = True
                control.should_training_stop = True
                run.metric(event="pause_requested", step=state.global_step)

        def on_log(self, args, state, control, logs=None, **kw):
            if not logs or "loss" not in logs:
                return
            m = dict(getattr(self, "last", {}), step=state.global_step)
            for k in ("loss", "grad_norm", "learning_rate", "epoch"):
                if k in logs:
                    try:
                        m[k] = float(logs[k])
                    except (TypeError, ValueError):
                        m[k] = logs[k]
            if isinstance(m.get("loss"), float) and (math.isnan(m["loss"]) or math.isinf(m["loss"])):
                m["nan"] = True
            run.metric(**m)

        def on_save(self, args, state, control, **kw):
            run.metric(event="checkpoint", step=state.global_step,
                       path=os.path.join(args.output_dir, f"checkpoint-{state.global_step}"))

        def on_train_end(self, args, state, control, **kw):
            run.write_status(state="paused" if self.paused else "finished", step=state.global_step)
            run.metric(event="paused" if self.paused else "train_end", step=state.global_step)

    return AnnealCallback()


class CountingCollator:
    """Wraps a data collator and records (tokens, trained tokens) of every batch in the order it is built. The
    trainer builds one batch per optimizer step (gradient accumulation 1), in the order it trains on them, so
    step N (1-based) trained on batch N even when the data loader fetches a batch ahead."""

    def __init__(self, collator, ignore_index=-100):
        self.collator, self.ignore_index, self.batches = collator, ignore_index, []

    def __call__(self, examples):
        out = self.collator(examples)
        labels = out["labels"]
        mask = out.get("attention_mask")
        tokens = int(mask.sum()) if mask is not None else int(labels.numel())
        # labels are shifted inside the model: position i predicts token i + 1, so a trained label at position 0
        # trains nothing; count the labels that are actually predicted.
        trained = int((labels[..., 1:] != self.ignore_index).sum())
        self.batches.append((tokens, trained))
        return out

    def step_tokens(self, step):
        return self.batches[step - 1] if 0 < step <= len(self.batches) else None


class WindowsMemorySampler(threading.Thread):
    """Every 2 s: system RAM, this process's RSS, and the portal's view of GPU memory (Windows counters), which
    matter on this unified-memory machine: the GPU's "shared" memory and offloaded activations come out of RAM."""

    PORTAL_METRICS = "http://127.0.0.1:8080/api/metrics"

    def __init__(self):
        super().__init__(daemon=True)
        import psutil
        self.psutil = psutil
        self.proc = psutil.Process()
        self.stop = threading.Event()
        self.peak = {"ram_used_gib": 0, "process_rss_gib": 0, "gpu_total_used_gib": 0, "gpu_training_gib": 0,
                     "gpu_training_dedicated_gib": 0, "gpu_training_shared_gib": 0}
        self.portal_ok = None

    def run(self):
        while not self.stop.wait(2):
            vm = self.psutil.virtual_memory()
            self._max("ram_used_gib", gib(vm.total - vm.available))
            try:
                self._max("process_rss_gib", gib(self.proc.memory_info().rss))
            except self.psutil.Error:
                pass
            try:
                with urllib.request.urlopen(self.PORTAL_METRICS, timeout=2) as r:
                    m = json.loads(r.read()).get("latest") or {}
                self.portal_ok = True
                if m.get("gpu_ok"):
                    self._max("gpu_total_used_gib", gib(m["gpu_ded_used"] + m["gpu_shr_used"]))
                    other = (m.get("gpu_procs") or {}).get("other") or {}
                    self._max("gpu_training_gib", gib(other.get("ded", 0) + other.get("shr", 0)))
                    self._max("gpu_training_dedicated_gib", gib(other.get("ded", 0)))
                    self._max("gpu_training_shared_gib", gib(other.get("shr", 0)))
            except Exception:
                if self.portal_ok is None:
                    self.portal_ok = False

    def _max(self, k, v):
        self.peak[k] = max(self.peak[k], v)


def install_guard_debug():
    """Print the source of a torch.compile guard that fails to build (the compile error hides it)."""
    import torch._dynamo.guards as dg
    orig = dg.GuardBuilder.get

    def get(self, guard, *a, **k):
        try:
            return orig(self, guard, *a, **k)
        except Exception as exc:
            print(f"[guard-debug] failed guard source: {getattr(guard, 'name', guard)!r}: {exc}")
            raise
    dg.GuardBuilder.get = get


def last_checkpoint(output_dir):
    """The newest checkpoint-N folder in output_dir, or None."""
    if not os.path.isdir(output_dir):
        return None
    cks = [d for d in os.listdir(output_dir) if d.startswith("checkpoint-") and d[11:].isdigit()]
    return os.path.join(output_dir, max(cks, key=lambda d: int(d[11:]))) if cks else None
