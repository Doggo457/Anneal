"""GatedDeltaNet value-head pairing for Qwen3.8 under transformers 5.17 (qwen3_5 code): stock by default.

STATUS (18 Oct 2026, results\\parity_findings.md): transformers' stock pairing is CORRECT and is the default.
Qwen3.8-27B has 16 key heads and 48 value heads in its linear-attention (GatedDeltaNet) layers. transformers'
Qwen3_5GatedDeltaNet expands q and k with repeat_interleave, pairing value head j with key head j // 3, and
llama.cpp computes the same model on Ollama's GGUF (tests\\parity_llama_hf.py: loss 0.515 vs 0.519, 95% top-1
agreement, mean KL 0.02). The "tile" patch below (value head j -> key head j % 16) is WRONG: loss 3.04 on the
same text and repetitive greedy output. It is kept only as an explicit opt-in (--gdn-heads tile) so the runs
made with it before 18 Oct 2026 (smoke-27b, mem-27b-8k*, baseloss-*-fixed, which ran with the old "auto") can
be reproduced. "auto" used to mean tile for Qwen3.8; it now means stock for every model.

Why the tile patch looked right at first: on CodeAlpaca chats in Qwen3.8's chat template, the whole-row loss
was 6.8 / 7.6 / 4.0 / 5.1 with stock pairing and 1.26 / 1.50 / 1.67 / 1.21 tiled. A third of those rows is
the template's system prompt ("Reasoning effort is set to xhigh...") that the real model doesn't predict; on
the assistant tokens alone stock scores 0.81 and tile 1.80. Judge loss on assistant tokens (anneal\\data.py).

How the opt-in patch works: Unsloth runs its own copy of the GatedDeltaNet forward (unsloth_compiled_cache), so
it is made where both copies expand q and k: Tensor.repeat_interleave, for one exact call shape only, a 4-D
tensor expanded along dim 2 whose size there is the model's key-head count and whose last dim is the key head
size, by exactly value_heads // key_heads. Everything else goes to the original method untouched.
"""

import torch

STATE = {"mode": None, "requested": None, "patched": False, "pairing": None, "key_heads": None, "ratio": None,
         "head_dim": None, "calls": 0}
_ORIGINAL = torch.Tensor.repeat_interleave


def _install(key_heads, ratio, head_dim):
    def repeat_interleave(self, repeats, dim=None, *, output_size=None):
        if (dim == 2 and type(repeats) is int and repeats == ratio and self.dim() == 4
                and self.shape[2] == key_heads and self.shape[3] == head_dim):
            if not torch.compiler.is_compiling():
                STATE["calls"] += 1
            return self.repeat(1, 1, ratio, 1)
        if output_size is None:
            return _ORIGINAL(self, repeats, dim=dim)
        return _ORIGINAL(self, repeats, dim=dim, output_size=output_size)

    torch.Tensor.repeat_interleave = repeat_interleave
    STATE.update(key_heads=key_heads, ratio=ratio, head_dim=head_dim)


MODES = ("auto", "stock", "interleave", "tile")  # interleave = stock (its old name); tile = the wrong patch


def apply(model_name, mode="stock"):
    """Set up the head pairing for `model_name` before it is loaded. mode: stock (= auto, interleave) | tile.

    stock leaves transformers' pairing alone (no patch) for every model. tile installs the old, wrong patch and
    is only for reproducing runs made with it. Returns a short description for the run log.
    """
    from transformers import AutoConfig

    STATE["requested"] = mode
    cfg = AutoConfig.from_pretrained(model_name)
    text = getattr(cfg, "text_config", None) or cfg
    n_k = getattr(text, "linear_num_key_heads", None)
    n_v = getattr(text, "linear_num_value_heads", None)
    d = getattr(text, "linear_key_head_dim", None)
    if not n_k or not n_v or n_v <= n_k:
        STATE.update(mode="n/a", patched=False, pairing="n/a: no grouped value heads in this model")
        return "GatedDeltaNet heads: no grouped value heads in this model, nothing to do"
    if mode in ("auto", "stock", "interleave"):
        mode = "stock"
    if mode not in ("stock", "tile"):
        raise ValueError(f"unknown --gdn-heads mode {mode!r}; use stock or tile")
    STATE["mode"] = mode
    if mode == "tile":
        if torch.Tensor.repeat_interleave is _ORIGINAL:
            _install(n_k, n_v // n_k, d)
        STATE.update(patched=True, pairing=f"tile: value head j -> key head j % {n_k} (anneal/gdn_heads.py patch; "
                                           f"known WRONG, for reproducing old runs only)")
        return (f"GatedDeltaNet heads: TILE ({n_v} value heads paired with key head j % {n_k}); transformers' own "
                f"repeat_interleave pairing is replaced for this call shape only. This pairing is known to be wrong "
                f"(results/parity_findings.md); use it only to reproduce old runs")
    STATE.update(patched=False, pairing=f"stock: value head j -> key head j // {n_v // n_k} "
                                        f"(transformers' repeat_interleave, no patch; matches llama.cpp)")
    return (f"GatedDeltaNet heads: stock (transformers' own pairing, value head j -> key head j // {n_v // n_k}; "
            f"no patch)")


def report():
    return dict(STATE)
