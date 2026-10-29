"""Tests for tests/kl_base_check.py row_stats on CPU: the chunked numbers equal the whole-row formulas it replaced
(cross entropy per trained token, KL(base || adapter) per reasoning token, close-tag probability inside the reasoning),
for any chunk size, with untrained positions and a row that never reasons.

  python tests/test_kl_base_rowstats.py        (needs torch; or: python -m pytest tests/test_kl_base_rowstats.py)
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from kl_base_check import row_stats  # noqa: E402

V, H, CLOSE = 97, 16, 5


def whole_row(head, h_b, h_a, y, think, close_id):
    """The formulas kl_base_check used before 29 Oct 2026, on whole-row tensors."""
    trained = y != -100
    inside = think & (y != close_id)
    with torch.no_grad():
        n_tr = max(int(trained.sum()), 1)
        ce_b = float(F.cross_entropy(head(h_b[trained]).float(), y[trained], reduction="sum")) / n_tr
        ce_a = float(F.cross_entropy(head(h_a[trained]).float(), y[trained], reduction="sum")) / n_tr
        lp_b = F.log_softmax(head(h_b[think]).float(), -1)
        lp_a = F.log_softmax(head(h_a[think]).float(), -1)
        n_th = int(think.sum())
        kl = float(F.kl_div(lp_a, lp_b, log_target=True, reduction="sum")) / max(n_th, 1)
        pc_b = float(lp_b[:, close_id][inside[think]].exp().mean()) if bool(inside.any()) else None
        pc_a = float(lp_a[:, close_id][inside[think]].exp().mean()) if bool(inside.any()) else None
    return ce_b, ce_a, kl, pc_b, pc_a, n_th


def make_row(T, seed, reasoning=True):
    g = torch.Generator().manual_seed(seed)
    h_b = torch.randn(T, H, generator=g)
    h_a = h_b + 0.3 * torch.randn(T, H, generator=g)
    y = torch.randint(0, V, (T,), generator=g)
    y[: T // 5] = -100  # the prompt: not trained
    think = torch.zeros(T, dtype=torch.bool)
    if reasoning:
        think[T // 5: T // 2] = True
        y[T // 2 - 1] = CLOSE  # the close tag ends the reasoning
    return h_b, h_a, y, think


def close(a, b, tol=1e-4):
    if a is None or b is None:
        return a is None and b is None
    return abs(a - b) <= tol * max(1.0, abs(b))


def test_chunked_equals_whole_row():
    torch.manual_seed(0)
    head = torch.nn.Linear(H, V, bias=False)
    for T, seed in ((7, 1), (300, 2), (1031, 3)):
        h_b, h_a, y, think = make_row(T, seed)
        ref = whole_row(head, h_b, h_a, y, think, CLOSE)
        for chunk in (1, 64, 256, 4096):
            got = row_stats(head, h_b, h_a, y, think, CLOSE, chunk=chunk)
            assert got[5] == ref[5]
            for g, r in zip(got[:5], ref[:5]):
                assert close(g, r), (T, chunk, got, ref)


def test_row_without_reasoning():
    head = torch.nn.Linear(H, V, bias=False)
    h_b, h_a, y, think = make_row(200, 4, reasoning=False)
    got = row_stats(head, h_b, h_a, y, think, CLOSE, chunk=64)
    assert got[2] == 0.0 and got[3] is None and got[4] is None and got[5] == 0
    assert close(got[0], whole_row(head, h_b, h_a, y, think, CLOSE)[0])


def test_same_states_give_zero_kl():
    head = torch.nn.Linear(H, V, bias=False)
    h_b, _, y, think = make_row(500, 5)
    ce_b, ce_a, kl, pc_b, pc_a, _ = row_stats(head, h_b, h_b.clone(), y, think, CLOSE, chunk=128)
    assert abs(kl) < 1e-6 and ce_b == ce_a and pc_b == pc_a


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print("ok", t.__name__)
    print(f"{len(tests)}/{len(tests)} passed")
