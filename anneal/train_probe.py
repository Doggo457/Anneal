"""Training speed checks (26 Oct 2026, speed-up item 3): an attention-input probe, an opt-in contiguous copy of the
attention inputs, the BLAS library in use, and a short torch.profiler pass with GPU time split by kind of work.

All of it is off unless a config key turns it on (scripts/train_longrun.py):
  sdpa_probe: [false]       record the shapes, strides and dtypes of the first scaled_dot_product_attention call, of
                            the first call with LONG_TOKENS+ query tokens (the model's own attention, not a library's
                            import-time self-test), and a count of long calls per memory layout
  sdpa_contiguous: [false]  make q, k and v contiguous before scaled_dot_product_attention. A contiguous copy holds the
                            same values, so results change only by kernel choice (rounding-level), never by data
  blas_backend: []          the BLAS library: "hipblaslt" or "hipblas" (CUDA names "cublaslt" / "cublas" too). Set
                            through TORCH_BLAS_PREFER_HIPBLASLT before torch is imported, because torch reads it once
                            and torch.backends.cuda.preferred_blas_library(x) changes nothing on Windows builds. The
                            run stops if torch then reports another library. Unset: Unsloth's RDNA default is kept
  profile_steps: [0]        torch.profiler over this many optimizer steps after profile_warmup [1] steps; writes
                            profile_table.txt and profile.json to the run folder

install_sdpa must run before Unsloth and transformers are imported, so that any module that copies the function by
name gets the wrapper too.
"""
import functools
import json
import os
import re
import sys
import time

# Kinds of GPU work for the profile split. The first pattern that matches a kernel or op name wins, so the more
# specific kinds come first. Names differ between CUDA and ROCm builds, so the full table is always written as well.
CATEGORIES = [
    ("cross_entropy", r"cross_?entropy|logsumexp|_ce_|ce_loss|chunked_ce"),
    ("bnb_dequant", r"dequant|bitsandbytes|bnb|nf4|kdequantize"),
    ("deltanet_fla", r"chunk_gated_delta|gated_delta|fused_recurrent|chunk_fwd|chunk_bwd|chunk_scaled_dot_kkt|"
                     r"solve_tril|wy_fast|l2norm|chunk_local_cumsum|\bfla\b|chunk_delta"),
    ("sdpa_attention", r"scaled_dot_product|flash|fmha|attention|aotriton|sdpa|efficient_attention|attn_fwd|attn_bwd|"
                       r"bwd_kernel_dk_dv|bwd_kernel_dq|bwd_preprocess"),
    ("short_conv", r"conv1d|short_conv|causal_conv|shift_add|convolution"),
    ("offload_copy", r"memcpy|copy_|_to_copy|empty_cache|htod|dtoh|pin_memory|memset"),
    ("matmul", r"gemm|cijk|rocblas|hipblaslt|cublas|addmm|\bmm\b|bmm|matmul|linear|_scaled_mm|tensile"),
]
_COMPILED = [(name, re.compile(pat, re.I)) for name, pat in CATEGORIES]


def categorize(name):
    for cat, pat in _COMPILED:
        if pat.search(name or ""):
            return cat
    return "other"


# Libraries run small attention self-tests while they are imported (Unsloth: 2 heads x 16 tokens), so the model's own
# calls are told apart by length: the first call with at least this many query tokens is recorded separately.
LONG_TOKENS = 256
MAX_LAYOUTS = 16


class SdpaReport:
    def __init__(self, contiguous, probe):
        self.contiguous, self.probe = contiguous, probe
        self.calls = 0
        self.copied_calls = 0
        self.first = None
        self.first_long = None
        self.layouts = {}  # layout -> calls, for calls with at least LONG_TOKENS query tokens

    def long_strided(self):
        """True when the first long call had a non-contiguous q, k or v; None when no long call was seen."""
        if not self.first_long:
            return None
        return any(not self.first_long[k]["contiguous"] for k in ("query", "key", "value"))

    def as_dict(self):
        return {"contiguous": self.contiguous, "probe": self.probe, "calls": self.calls,
                "calls_with_a_copy": self.copied_calls, "first_call": self.first,
                "long_tokens": LONG_TOKENS, "first_long_call": self.first_long, "long_strided": self.long_strided(),
                "long_layouts": self.layouts}


def _describe(t):
    return {"shape": list(t.shape), "stride": list(t.stride()), "contiguous": bool(t.is_contiguous()),
            "dtype": str(t.dtype).replace("torch.", ""), "device": str(t.device)}


def _describe_call(torch, query, key, value, args, kwargs):
    mask = kwargs.get("attn_mask", args[0] if args else None)
    return {"query": _describe(query), "key": _describe(key), "value": _describe(value),
            "attn_mask": None if mask is None else _describe(mask),
            "kwargs": {k: (v if isinstance(v, (bool, int, float, str, type(None))) else type(v).__name__)
                       for k, v in kwargs.items() if k != "attn_mask"},
            "grad_enabled": torch.is_grad_enabled()}


def _layout(query, key, value, kwargs):
    """A short label for a call's memory layout and shape family (not its length)."""
    lay = " ".join(f"{n}:{'contiguous' if t.is_contiguous() else 'strided'}"
                   for n, t in (("q", query), ("k", key), ("v", value)))
    heads = f"heads {query.shape[-3] if query.dim() >= 3 else 1}/{key.shape[-3] if key.dim() >= 3 else 1}"
    return f"{lay} {heads} dim {query.shape[-1]} {str(query.dtype).replace('torch.', '')} {','.join(sorted(kwargs))}"


def install_sdpa(contiguous=False, probe=False, log=print):
    """Wrap torch.nn.functional.scaled_dot_product_attention. Returns the SdpaReport the wrapper fills, or None when
    both switches are off (nothing is wrapped)."""
    if not (contiguous or probe):
        return None
    import torch
    import torch.nn.functional as F

    original = F.scaled_dot_product_attention
    if getattr(original, "_anneal_wrapped", False):
        raise RuntimeError("scaled_dot_product_attention is already wrapped")
    rep = SdpaReport(contiguous, probe)

    def wrapped(query, key, value, *args, **kwargs):
        rep.calls += 1
        if probe:
            if rep.first is None:
                rep.first = _describe_call(torch, query, key, value, args, kwargs)
                log(f"sdpa probe, first call: {json.dumps(rep.first)}")
            if query.dim() >= 2 and query.shape[-2] >= LONG_TOKENS:
                if rep.first_long is None:
                    rep.first_long = _describe_call(torch, query, key, value, args, kwargs)
                    log(f"sdpa probe, first call with {LONG_TOKENS}+ query tokens: {json.dumps(rep.first_long)}")
                lay = _layout(query, key, value, kwargs)
                if lay in rep.layouts or len(rep.layouts) < MAX_LAYOUTS:
                    rep.layouts[lay] = rep.layouts.get(lay, 0) + 1
        if contiguous and not (query.is_contiguous() and key.is_contiguous() and value.is_contiguous()):
            rep.copied_calls += 1
            query, key, value = query.contiguous(), key.contiguous(), value.contiguous()
        return original(query, key, value, *args, **kwargs)

    # Copy __doc__, __name__ and __wrapped__ from the original: Unsloth reads the docstring when it is imported
    # ("enable_gqa" in scaled_dot_product_attention.__doc__), and a None docstring stops it with a TypeError.
    functools.update_wrapper(wrapped, original)
    wrapped._anneal_wrapped = True
    F.scaled_dot_product_attention = wrapped
    torch.nn.functional.scaled_dot_product_attention = wrapped
    log(f"scaled_dot_product_attention wrapped: contiguous={contiguous}, probe={probe}")
    return rep


# blas_backend -> (TORCH_BLAS_PREFER_HIPBLASLT value, the library torch then reports)
BLAS_CHOICES = {"hipblaslt": ("1", "Cublaslt"), "cublaslt": ("1", "Cublaslt"),
                "hipblas": ("0", "Cublas"), "rocblas": ("0", "Cublas"), "cublas": ("0", "Cublas")}


def prefer_blas(backend, environ=None, modules=None):
    """Choose the BLAS library before torch is imported. torch reads TORCH_BLAS_PREFER_HIPBLASLT once, when it sets up
    its context (aten/src/ATen/Context.cpp); on Windows builds torch.backends.cuda.preferred_blas_library(x) only
    warns, and Unsloth sets the variable only when it is unset. -> the library torch should then report, or None."""
    if not backend:
        return None
    environ = os.environ if environ is None else environ
    modules = sys.modules if modules is None else modules
    key = str(backend).strip().lower()
    if key not in BLAS_CHOICES:
        raise ValueError(f"blas_backend {backend!r}: use one of {', '.join(sorted(BLAS_CHOICES))}")
    if "torch" in modules:
        raise RuntimeError("blas_backend must be applied before torch is imported (torch reads it only once)")
    value, expect = BLAS_CHOICES[key]
    environ["TORCH_BLAS_PREFER_HIPBLASLT"] = value
    return expect


def blas_report(requested=None, expect=None, log=print):
    """Report the BLAS library in use and the environment switch; stop if it is not the one prefer_blas chose."""
    import torch

    out = {"env_TORCH_BLAS_PREFER_HIPBLASLT": os.environ.get("TORCH_BLAS_PREFER_HIPBLASLT"),
           "hip": getattr(torch.version, "hip", None), "requested": requested}
    try:
        out["in_use"] = str(torch.backends.cuda.preferred_blas_library())
    except Exception as exc:  # noqa: BLE001  (an older build or a CPU-only test)
        out["error"] = f"{type(exc).__name__}: {exc}"
        if expect:
            raise
    log(f"BLAS library: {out}")
    if expect and out["in_use"].rsplit(".", 1)[-1] != expect:
        raise RuntimeError(f"blas_backend {requested!r} asked for {expect}, but torch uses {out['in_use']}")
    return out


def summarize_events(events):
    """events: (name, self_device_us, self_cpu_us, calls). Shares of device time by kind (CPU time when the build
    records no device time)."""
    use_device = any(d > 0 for _, d, _, _ in events)
    key = 1 if use_device else 2
    total = sum(e[key] for e in events) or 1
    cats = {}
    for e in events:
        c = categorize(e[0])
        cats[c] = cats.get(c, 0) + e[key]
    return {"basis": "self_device_time" if use_device else "self_cpu_time (no device activity recorded)",
            "total_ms": round(total / 1000, 1),
            "share": {c: round(v / total, 4) for c, v in sorted(cats.items(), key=lambda kv: -kv[1])}}


class Profile:
    """torch.profiler over optimizer steps: start() at train begin, step() after each optimizer step. Writes the top
    of the key_averages table and the split by kind when the active steps are done."""

    def __init__(self, run_dir, warmup=1, active=2, log=print):
        import torch

        self.run_dir, self.warmup, self.active, self.log = run_dir, int(warmup), int(active), log
        acts = [torch.profiler.ProfilerActivity.CPU]
        supported = torch.profiler.supported_activities()
        self.device = torch.profiler.ProfilerActivity.CUDA in supported
        if self.device:
            acts.append(torch.profiler.ProfilerActivity.CUDA)
        self.prof = torch.profiler.profile(
            activities=acts, record_shapes=False, profile_memory=False, with_stack=False,
            schedule=torch.profiler.schedule(wait=0, warmup=self.warmup, active=self.active, repeat=1),
            on_trace_ready=self._ready)
        self.result = None
        self.started = self.stopped = False
        self.t_active = None

    def start(self):
        self.prof.start()
        self.started = True
        self.log(f"profiler started: {self.warmup} warm-up + {self.active} active steps, device activity "
                 f"{'on' if self.device else 'not available in this build'}")

    def step(self):
        if self.started and not self.stopped:
            self.prof.step()

    def _ready(self, prof):
        ka = prof.key_averages()
        events = []
        for e in ka:
            dev = getattr(e, "self_device_time_total", None)
            if dev is None:
                dev = getattr(e, "self_cuda_time_total", 0)
            events.append((e.key, float(dev or 0), float(e.self_cpu_time_total or 0), int(e.count)))
        sort = "self_device_time_total" if any(d > 0 for _, d, _, _ in events) else "self_cpu_time_total"
        try:
            table = ka.table(sort_by=sort, row_limit=80, max_name_column_width=90)
        except Exception as exc:  # noqa: BLE001  (older builds name the column self_cuda_time_total)
            table = ka.table(sort_by="self_cuda_time_total" if sort.startswith("self_device") else sort,
                             row_limit=80, max_name_column_width=90) if sort.startswith("self_device") else str(exc)
        summary = summarize_events(events)
        summary.update(steps=self.active, warmup=self.warmup, device_activity=self.device,
                       top=[{"name": n[:120], "self_device_ms": round(d / 1000, 2), "self_cpu_ms": round(c / 1000, 2),
                             "calls": k, "kind": categorize(n)}
                            for n, d, c, k in sorted(events, key=lambda e: -(e[1] or e[2]))[:60]])
        with open(os.path.join(self.run_dir, "profile_table.txt"), "w", encoding="utf-8") as f:
            f.write(table)
        with open(os.path.join(self.run_dir, "profile.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=1)
        self.result = {k: summary[k] for k in ("basis", "total_ms", "share", "steps", "warmup", "device_activity")}
        self.log(f"profile written ({time.strftime('%H:%M:%S')}): {json.dumps(self.result)}")

    def stop(self):
        if self.started and not self.stopped:
            self.stopped = True
            self.prof.stop()
