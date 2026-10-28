"""Training loss computed here from the decoder's hidden states, for scripts/train_longrun.py (own_loss, kl_think_beta).

  loss = (cross entropy summed over the trained tokens + beta x KL(base || tuned) summed over the reasoning targets)
         / the Trainer's token count

The reasoning targets are the trained turn's think block, from its first token through the close tag, so the term
covers what the model writes while it thinks and the decision to stop. The base is the same model with the LoRA
adapter switched off (PeftModel.disable_adapter), so no second copy is loaded; its hidden states come from one extra
forward pass without gradients. At beta = 1 the reasoning targets are trained towards an equal mix of the teacher's
token and the base model's distribution (cross entropy + KL(p_base || p) is cross entropy against (onehot + p_base) / 2,
doubled, up to a constant). Why: round 2 learned to stop thinking early (research/thinking-length-fix.md).

Logits are never built for the whole sequence: the LM head runs on chunks of positions sized to a memory budget, each
chunk under activation checkpointing, so the backward pass rebuilds one chunk at a time.
"""

import contextlib

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

IGNORE_INDEX = -100


def think_targets(input_ids, labels, close_id):
    """Bool [B, T - 1] over next-token targets (aligned with labels[:, 1:]): True from a row's first trained token
    through its first trained close tag. A row without a trained close tag has none."""
    out = torch.zeros(labels.shape, dtype=torch.bool, device=labels.device)
    trained = labels != IGNORE_INDEX
    for b in range(labels.shape[0]):
        tr = trained[b].nonzero().flatten()
        if tr.numel() == 0:
            continue
        closes = ((input_ids[b] == close_id) & trained[b]).nonzero().flatten()
        if closes.numel() == 0:
            continue
        out[b, int(tr[0]):int(closes[0]) + 1] = True
    return out[:, 1:]


def chunk_rows(vocab, budget_bytes, bytes_per_logit):
    """Positions per chunk so one chunk's logits and their float32 work tensors fit the budget."""
    return max(16, int(budget_bytes // (vocab * bytes_per_logit)))


def _ce_part(head, h, y):
    return F.cross_entropy(head(h).float(), y, reduction="sum")


def _kl_part(head, h, hb):
    log_p = F.log_softmax(head(h).float(), dim=-1)
    with torch.no_grad():
        log_q = F.log_softmax(head(hb).float(), dim=-1)
    return F.kl_div(log_p, log_q, log_target=True, reduction="sum")  # sum q (log q - log p): KL(base || tuned)


def chunked_sum(fn, chunk, *xs):
    """sum(fn(*chunk of xs)) over chunks of the first dimension, each under non-reentrant checkpointing when grads are
    on."""
    total = None
    for s in range(0, xs[0].shape[0], chunk):
        xs_part = tuple(x[s:s + chunk] for x in xs)
        part = checkpoint(fn, *xs_part, use_reentrant=False) if torch.is_grad_enabled() else fn(*xs_part)
        total = part if total is None else total + part
    return total


def unwrap(model):
    return getattr(model, "module", model)


@contextlib.contextmanager
def no_checkpointing(module):
    """Layers run plainly inside (gradient_checkpointing off): the base pass keeps no activations anyway, and
    checkpointing would only copy them out (Unsloth's offloaded checkpoints) and drop them again."""
    layers = [m for m in module.modules() if getattr(m, "gradient_checkpointing", False) is True]
    for m in layers:
        m.gradient_checkpointing = False
    try:
        yield
    finally:
        for m in layers:
            m.gradient_checkpointing = True


def own_loss(model, inputs, beta, close_id, budget_bytes, denominator):
    """-> (loss, stats). model: the PEFT-wrapped causal LM. denominator(trained tokens of this micro-batch) gives
    the number the summed loss is divided by (the Trainer's normalisation; see train_longrun.py)."""
    peft = unwrap(model)
    inner = peft.get_base_model()
    decoder, head = inner.model, inner.lm_head
    ids, labels = inputs["input_ids"], inputs["labels"]
    mask = inputs.get("attention_mask")
    if mask is not None and not bool(mask.all()):
        raise RuntimeError("own_loss needs unpadded rows (batch_size 1): the attention mask has zeros")
    y = labels[:, 1:]
    trained = y != IGNORE_INDEX
    if not bool(trained.any()):
        raise RuntimeError("a training row with no trained tokens")
    vocab = head.weight.shape[0]
    stats = {"trained": int(trained.sum()), "think": 0, "kl_sum": 0.0}
    think = think_targets(ids, labels, close_id) if beta > 0 else None
    hb = None
    if think is not None and bool(think.any()):
        # base first: nothing of the training forward is held yet
        with torch.no_grad(), peft.disable_adapter(), no_checkpointing(decoder):
            hb = decoder(input_ids=ids, use_cache=False).last_hidden_state[:, :-1][think]
    h = decoder(input_ids=ids, use_cache=False).last_hidden_state[:, :-1]
    ce = chunked_sum(lambda a, b: _ce_part(head, a, b), chunk_rows(vocab, budget_bytes, 12), h[trained], y[trained])
    total = ce
    stats["ce_sum"] = float(ce.detach())
    if hb is not None:
        kl = chunked_sum(lambda a, b: _kl_part(head, a, b), chunk_rows(vocab, budget_bytes, 24), h[think], hb)
        total = total + beta * kl
        stats.update(think=int(think.sum()), kl_sum=float(kl.detach()))
    return total / denominator(stats["trained"]), stats
