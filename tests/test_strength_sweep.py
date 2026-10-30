"""Tests for scripts/strength_sweep.py without a GPU, a model or Windows: training, conversion and held-out losses by a
fake run_cmd, answers from a fake llama-server whose adapter is as strong as its learning rate (lr / 1e-4), the
choice rule and its tie-breaks, the hand-back to the loop, and every refusal. The tuning pass (30 Oct 2026): KL
weight candidates, the half-length run as a decay branch from the full run's checkpoint, and the close-call check.

  python tests/test_strength_sweep.py        (or: python -m pytest tests/test_strength_sweep.py)
"""

import contextlib
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import test_lora_scale_check as T  # noqa: E402  (the fake server, BFCL module and loop folder)

S, L, E = T.S, T.L, T.E
import strength_sweep as SW  # noqa: E402
import train_longrun as TL  # noqa: E402  (only its short_run_start; nothing trains)
import yaml  # noqa: E402

LRS = (1e-4, 5e-5, 2.5e-5)
CALLS = []
FAKE = {"pause_once": False, "drop_init_adapter": False}


STEPS = 20  # the fake full run's steps


def strength(lr, kl=None, half=False):
    """How strongly the fake adapter changes answers: its learning rate, less with a larger KL weight, less again
    at half length."""
    return float(lr) / 1e-4 / (1 + 0.1 * (kl or 0)) * (0.7 if half else 1.0)


# ============================================================================================ fakes

class LrServer(T.FakeServer):
    """The fake server with the adapter's strength read from the fake GGUF: scale 1 on an lr 5e-5 adapter answers as
    scale 0.5 on the lr 1e-4 one."""

    def __init__(self, flags, log_path, **kw):
        lora = flags[flags.index("--lora") + 1]
        assert re.search(r"-lr[0-9.e+-]+(_kl[0-9.e+-]+)?(_half)?-F16-LoRA\.gguf$", os.path.basename(lora)), lora
        self.k = json.loads(open(lora, "rb").read()[4:])["k"]
        super().__init__(flags, log_path, **kw)

    applied = property(lambda self: self._applied * self.k, lambda self, v: setattr(self, "_applied", v))


def args_of(cmd):
    a, rest = {}, list(cmd[3:])
    while rest:
        k = rest.pop(0)
        if k.startswith("--"):
            a[k] = True if k in ("--resume", "--no-base") else rest.pop(0)
        else:
            a["path"] = k
    return a


def fake_run_cmd(cmd, log_path):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    script, a = os.path.basename(cmd[2]), args_of(cmd)
    CALLS.append((script, a))
    if script == "train_longrun.py":
        assert a["--resume"] is True
        cfg = yaml.safe_load(open(a["--config"]))
        tdir = a["--run-dir"]
        branch = a.get("--branch-from")
        os.makedirs(os.path.join(tdir, "checkpoints", "checkpoint-10"), exist_ok=True)
        open(os.path.join(tdir, "checkpoints", "checkpoint-10", "optimizer.pt"), "wb").write(b"opt")
        if FAKE["pause_once"]:
            FAKE["pause_once"] = False
            json.dump({"state": "paused", "step": 5}, open(os.path.join(tdir, "status.json"), "w"))
            return 0
        if branch:  # as train_longrun.py: resumes the milestone, decays over branch_decay_steps
            assert "short_run_fractions" not in cfg and os.path.isdir(branch), branch
            start = json.load(open(os.path.join(branch, "trainer_state.json")))["global_step"]
            steps = start + int(cfg["branch_decay_steps"])
            ad = os.path.join(tdir, "branches", os.path.basename(branch), "adapter")
        else:
            steps, ad = STEPS, os.path.join(tdir, "adapter")
            for f in cfg.get("short_run_fractions") or []:
                ms = os.path.join(tdir, "milestones", f"checkpoint-{TL.short_run_start(steps, f, 0.2)[0]}")
                os.makedirs(ms, exist_ok=True)
                json.dump({"global_step": int(ms.rsplit("-", 1)[1])}, open(os.path.join(ms, "trainer_state.json"), "w"))
        k = strength(cfg["lr"], cfg.get("kl_think_beta"), bool(branch))
        os.makedirs(ad, exist_ok=True)
        json.dump({"r": 16, "lora_alpha": 32, "fake_k": k}, open(os.path.join(ad, "adapter_config.json"), "w"))
        open(os.path.join(ad, "adapter_model.safetensors"), "wb").write(repr(k).encode() * 100)
        open(os.path.join(tdir, "metrics.jsonl"), "w").write(json.dumps({"step": steps, "loss": 0.5}) + "\n")
        rows = sum(1 for _ in open(cfg["train_file"]))
        summ = {"global_step": steps, "total_steps": steps, "trainable_params": 1, "rows": rows,
                "rows_dropped_too_long": 0,
                "tokens": 1000, "peak_reserved_gib": 20.0, "session_seconds": 60, "adapter": ad,
                "metrics": os.path.join(tdir, "metrics.jsonl"), "data_sha256": "d",
                "init_adapter": None if FAKE["drop_init_adapter"] else cfg.get("init_adapter")}
        json.dump(summ, open(os.path.join(tdir, "summary.json"), "w"))
        json.dump({"state": "finished", "step": steps}, open(os.path.join(tdir, "status.json"), "w"))
        return 0
    if script == "convert_lora_to_gguf.py":
        assert a["--outfile"].endswith(".gguf.part") and a["--outtype"] == "f16"
        k = json.load(open(os.path.join(a["path"], "adapter_config.json")))["fake_k"]
        open(a["--outfile"], "wb").write(b"gguf" + json.dumps({"k": k}).encode())
        return 0
    if script == "val_loss.py":
        assert a["--no-base"] is True and a["--max-rows"] == "200"
        k = json.load(open(os.path.join(a["--adapter"], "adapter_config.json")))["fake_k"]
        per = [{"key": r["key"], "trained": 100, "adapter_loss": 1.0 + (T.h(r["key"]) % 50) / 100 - 0.1 * k}
               for r in map(json.loads, open(a["--val-file"]))]
        os.makedirs(a["--run-dir"], exist_ok=True)
        json.dump({"scored": ["adapter"], "per_row": per}, open(os.path.join(a["--run-dir"], "summary.json"), "w"))
        return 0
    raise AssertionError(f"unexpected command {cmd}")


def calls(script):
    return [a for s, a in CALLS if s == script]


# ============================================================================================ a fake round 3

def make_round3(init_adapter=None, hold=True, data=True, kl=None):
    r3 = SW.round_dir(3)
    os.makedirs(os.path.join(r3, "stages"))
    plan = {"recipe": {"id": "filter-kl"}, "init": "base" if not init_adapter else "best", "rows": 50,
            "init_adapter": init_adapter,
            "train": {"base_model": "m", "lr": 1e-4, "lora_r": 16, "lora_alpha": 32, "max_seq_len": 12288, "seed": 7}}
    if kl is not None:
        plan["train"]["kl_think_beta"] = kl
    json.dump(plan, open(os.path.join(r3, "stages", "plan.done.json"), "w"))
    if data:
        mix = os.path.join(r3, "data", "mix")
        E.write_jsonl(os.path.join(mix, "train.jsonl"), [{"text": f"row {i}"} for i in range(50)])
        E.write_jsonl(os.path.join(mix, "val.jsonl"), [{"text": f"val {i}"} for i in range(10)])
        json.dump({"train_rows": 50, "val_rows": 10}, open(os.path.join(r3, "stages", "data.done.json"), "w"))
    cfg = yaml.safe_load(open(L.CONFIG))
    cfg["min_free_gb"] = 0
    if hold:
        cfg["hold_before_train"] = [3]
    yaml.safe_dump(cfg, open(L.CONFIG, "w"))
    return plan


@contextlib.contextmanager
def sweep_world(**kw):
    old = [(o, k, getattr(o, k)) for o, k in ((SW, "LOOP_DIR"), (SW, "OUT_ROOT"), (SW, "run_cmd"),
                                              (SW, "CLOSE_MARGIN"))]
    with T.world() as gate:
        try:
            SW.LOOP_DIR = S.LOOP_DIR
            SW.OUT_ROOT = os.path.join(os.path.dirname(S.LOOP_DIR), "strength")
            SW.run_cmd = fake_run_cmd
            S.llama.Server = LrServer  # put back by T.world
            CALLS.clear()
            T.FakeServer.instances.clear()
            FAKE.update(pause_once=False, drop_init_adapter=False)
            plan = make_round3(**kw)
            yield gate, plan
        finally:
            for o, k, v in old:
                setattr(o, k, v)


def run(*argv, lrs=LRS, cands=None):
    which = ["--cands", cands] if cands is not None else ["--lrs", ",".join(f"{x:g}" for x in lrs)]
    sys.argv = ["strength_sweep.py", "--round", "3", *which, *argv]
    return SW.main()


def refused(argv, text, **kw):
    try:
        run(*argv, **kw)
    except SystemExit as exc:
        assert text in str(exc), exc
        return
    raise AssertionError(f"should have refused: {text}")


def want_gate(gate, ids, k):
    """{id: (pass, reasoning tokens)} by the fake server's rule at strength k, seed 1."""
    idx = {it["id"]: i for i, it in enumerate(gate)}
    out = {}
    for gid in ids:
        it, seed = gate[idx[gid]], 100000 + idx[gid]
        out[gid] = (T.h(it["prompt"], seed) % 100 < 50 + 30 * (1 - k),
                    int((2000 + T.h(it["prompt"]) % 3000) * (1.6 - k)))
    return out


# ============================================================================================ tests

def test_full_sweep_choose_and_adopt():
    with sweep_world() as (gate, plan):
        assert run("--n", "20", "--plan") == 0
        assert not CALLS and not os.path.exists(SW.OUT_ROOT)  # --plan starts nothing
        assert run("--n", "20") == 0
        assert len(calls("train_longrun.py")) == 3 and len(calls("convert_lora_to_gguf.py")) == 3
        assert len(calls("val_loss.py")) == 3 * len(L.HELDOUT)
        cfg = L.load_config(L.CONFIG)
        bitems = E.read_jsonl(L.Loop.guard_file(None, cfg, "bfcl"))
        sums = {}
        for lr in LRS:
            d, k = SW.cand_dir(3, lr), strength(lr)
            t = yaml.safe_load(open(os.path.join(d, "train.yaml")))
            assert t["lr"] == lr and {x: t[x] for x in plan["train"] if x != "lr"} == \
                {x: v for x, v in plan["train"].items() if x != "lr"}, t  # only the lr differs from the round's plan
            assert t["short_run_fractions"] == [0.5] and "kl_think_beta" not in t  # plus the half-length checkpoint
            assert os.path.isdir(os.path.join(d, "train", "milestones", "checkpoint-8"))
            assert t["train_file"].endswith("r003/data/mix/train.jsonl") and "init_adapter" not in t
            assert os.path.exists(os.path.join(d, "convert", f"loop-r003-lr{SW.lr_tag(lr)}-F16-LoRA.gguf"))
            rows = E.read_jsonl(os.path.join(d, "gate.seed1.jsonl"))
            sel = json.load(open(os.path.join(d, "selection.json")))["ids"]
            assert len(rows) == 20 and {r["id"] for r in rows} == set(sel)
            assert sel == [gate[i]["id"] for i in S.draw(gate, 20)]  # part 1's draw
            want = want_gate(gate, sel, k)
            assert all((r["pass"], r["reasoning_tokens"]) == want[r["id"]] for r in rows)  # answered at this strength
            brows = E.read_jsonl(os.path.join(d, "bfcl.seed1.jsonl"))
            bwant = {it["id"]: T.h(it["prompt"], 100000 + i) % 100 < 50 + 30 * (1 - k) for i, it in enumerate(bitems)}
            assert len(brows) == 30 and all(r["pass"] == bwant[r["id"]] for r in brows)
            s = json.load(open(os.path.join(d, "summary.json")))
            assert s["gate"]["all"]["cand"]["pass"] == sum(p for p, _ in want.values())
            assert s["gate"]["write"]["n"] + s["gate"]["fix"]["n"] == 20
            assert s["bfcl"]["all"]["cand"] == sum(bwant.values()) and s["bfcl"]["java"]["items"] == 15
            for kind in L.HELDOUT:
                ls = s["loss"][kind]
                assert ls["n"] == 30 and abs((ls["candidate"] - ls["best"]) + 0.1 * k) < 1e-9, ls
            assert [f for f in os.listdir(d) if f.startswith("probe-gate-")]
            assert json.load(open(os.path.join(d, "scale_after-gate.json")))["scales"] == {"0": 1.0}
            sums[lr] = s
        choice = json.load(open(SW.choice_path(3)))
        key = {lr: (s["gate"]["all"]["cand"]["pass"], s["bfcl"]["all"]["cand"],
                    -sum(v["rel_diff"] for v in s["loss"].values()) / len(s["loss"]), -lr) for lr, s in sums.items()}
        best = max(LRS, key=lambda x: key[x])
        assert choice["chosen"]["lr"] == best and [c["lr"] for c in choice["candidates"]] == list(LRS), choice
        assert choice["base"]["gate_pass"] == sums[LRS[0]]["gate"]["all"]["base"]["pass"]

        # a second run answers and trains nothing again
        CALLS.clear()
        T.FakeServer.instances.clear()
        assert run("--n", "20") == 0
        assert not CALLS and not T.FakeServer.instances

        # hand-back: the chosen adapter becomes round 3's train stage, checkpoints left out
        assert run("--n", "20", "--part", "adopt") == 0
        r3 = SW.round_dir(3)
        src, dst = os.path.join(SW.cand_dir(3, best), "train"), os.path.join(r3, "train")
        assert SW.tree_sha(os.path.join(src, "adapter")) == SW.tree_sha(os.path.join(dst, "adapter"))
        assert not os.path.exists(os.path.join(dst, "checkpoints")) and os.path.exists(os.path.join(src, "checkpoints"))
        done = json.load(open(os.path.join(r3, "stages", "train.done.json")))
        assert done["strength_sweep"]["lr"] == best and done["adapter"] == os.path.join(dst, "adapter")
        assert done["metrics"] == os.path.join(dst, "metrics.jsonl") and done["global_step"] == STEPS
        assert not os.path.exists(os.path.join(dst, "milestones"))
        assert set(SW.TRAIN_DONE_KEYS) <= set(done) and done["finished"]
        ytext = open(os.path.join(r3, "train.yaml")).read()
        assert ytext.startswith("# adopted from ") and yaml.safe_load(ytext)["lr"] == best
        assert SW.hold_check(L.load_config(L.CONFIG), 3) == "round 3 already has a train stage"
        refused(["--n", "20", "--part", "adopt"], "already has a train stage")
        refused(["--n", "20", "--part", "train"], "already has a train stage", lrs=(1.25e-5,))  # nothing new trains


def test_rank_tie_breaks():
    def s(lr, gate, bfcl, swe, term):
        return {"lr": lr, "gate": {"all": {"cand": {"pass": gate}}}, "bfcl": {"all": {"cand": bfcl}},
                "loss": {"swe": {"rel_diff": swe}, "terminal": {"rel_diff": term}}}

    def pick(*rows):
        return max(rows, key=SW.rank_key)["lr"]

    assert pick(s(1e-4, 30, 50, -0.2, -0.2), s(5e-5, 31, 10, 0.0, 0.0)) == 5e-5  # gate first
    assert pick(s(1e-4, 30, 50, -0.0, -0.0), s(5e-5, 30, 49, -0.3, -0.3)) == 1e-4  # then BFCL
    assert pick(s(1e-4, 30, 50, -0.10, -0.02), s(5e-5, 30, 50, -0.05, -0.05)) == 1e-4  # then mean loss change
    assert pick(s(1e-4, 30, 50, -0.05, -0.05), s(5e-5, 30, 50, -0.05, -0.05), s(2.5e-5, 30, 50, -0.05, -0.05)) \
        == 2.5e-5  # then the smaller change


def test_choose_needs_every_part():
    with sweep_world():
        assert run("--n", "10", "--part", "train", lrs=LRS[:2]) == 0
        assert not calls("val_loss.py") and not T.FakeServer.instances
        assert run("--n", "10", "--part", "loss", lrs=LRS[:2]) == 0
        assert run("--n", "10", "--part", "gate", lrs=LRS[:2]) == 0
        refused(["--n", "10", "--part", "choose"], "lr 0.0001; lr 5e-05", lrs=LRS[:2])  # no BFCL answers yet
        assert not os.path.exists(SW.choice_path(3))
        refused(["--n", "10", "--part", "adopt"], "run --part choose first", lrs=LRS[:2])
        assert run("--n", "10", "--part", "bfcl", lrs=LRS[:2]) == 0
        assert run("--n", "10", "--part", "choose", lrs=LRS[:2]) == 0
        assert os.path.exists(SW.choice_path(3))


def test_refusals_before_anything_runs():
    with sweep_world(hold=False):
        refused(["--part", "train"], "hold_before_train does not list round 3")
    with sweep_world(data=False):
        refused(["--part", "train"], "has no data yet")
    with sweep_world():
        open(os.path.join(SW.LOOP_DIR, "loop.pid"), "w").write(str(os.getpid()))
        refused(["--part", "train"], "loop.py is running")
    with sweep_world():
        S.llama_running = lambda: True  # put back by T.world
        refused(["--part", "train"], "llama-server.exe is running")
    with sweep_world():
        cfg = yaml.safe_load(open(L.CONFIG))
        cfg["min_free_gb"] = 10 ** 9
        yaml.safe_dump(cfg, open(L.CONFIG, "w"))
        refused(["--part", "train"], "GB free on the training drive")
        assert not CALLS
    with sweep_world():
        refused(["--part", "train"], "must be distinct", lrs=(1e-4, 1e-4))
        assert not CALLS


def test_pause_then_resume():
    with sweep_world():
        FAKE["pause_once"] = True
        refused(["--part", "train"], "training paused", lrs=LRS[:1])
        assert not os.path.exists(os.path.join(SW.cand_dir(3, LRS[0]), "convert"))
        assert run("--part", "train", lrs=LRS[:1]) == 0
        assert len(calls("train_longrun.py")) == 2 and len(calls("convert_lora_to_gguf.py")) == 1


def test_init_adapter_must_be_loaded():
    with sweep_world(init_adapter="C:/AI/results/loop/rounds/r001/train/adapter"):
        FAKE["drop_init_adapter"] = True
        refused(["--part", "train"], "did not report loading init_adapter", lrs=LRS[:1])
        t = yaml.safe_load(open(os.path.join(SW.cand_dir(3, LRS[0]), "train.yaml")))
        assert t["init_adapter"] == "C:/AI/results/loop/rounds/r001/train/adapter"
        assert not calls("convert_lora_to_gguf.py")


def test_probe_fails_when_adapter_does_nothing():
    with sweep_world():
        T.FakeServer.ignore_scale = True  # put back by T.world
        refused(["--n", "10", "--part", "gate"], "adapter probe failed", lrs=LRS[:1])
        assert not E.read_jsonl(os.path.join(SW.cand_dir(3, LRS[0]), "gate.seed1.jsonl"))


def test_bfcl_refuses_base_rows_from_other_settings():
    with sweep_world():
        path = os.path.join(SW.LOOP_DIR, "state.json")
        st = json.load(open(path))
        st["base_guards"]["bfcl"]["fingerprint"]["budget"] = 16384
        json.dump(st, open(path, "w"))
        refused(["--part", "bfcl"], "other settings", lrs=LRS[:1])
        assert not os.path.exists(os.path.join(SW.cand_dir(3, LRS[0]), "bfcl.seed1.jsonl"))


def test_short_run_start_matches_trainer():
    for total in range(2, 400):
        for f in (0.25, 0.5, 0.75):
            for df in (0.1, 0.2, 0.3):
                assert SW.short_run_start(total, f, df) == TL.short_run_start(total, f, df)
    assert SW.short_run_start(200, 0.5) == (80, 20)  # a 100-step run: stable to 80, decays over 20


def test_rank_tie_breaks_kl_and_length():
    def s(lr, kl=None, half=False, gate=30):
        return {"lr": lr, "kl": kl, "half": half, "gate": {"all": {"cand": {"pass": gate}}}, "bfcl": {"all": {"cand": 50}},
                "loss": {"swe": {"rel_diff": -0.1}, "terminal": {"rel_diff": -0.1}}}

    def pick(*rows):
        r = max(rows, key=SW.rank_key)
        return r["lr"], r["kl"], r["half"]

    assert pick(s(1e-4, kl=3.0), s(1e-4)) == (1e-4, None, False)  # the round's own KL weight on a tie
    assert pick(s(1e-4, half=True), s(1e-4)) == (1e-4, None, False)  # then the full length
    assert pick(s(1e-4, kl=3.0, half=True, gate=31), s(1e-4)) == (1e-4, 3.0, True)  # gate passes still come first
    assert pick(s(5e-5, kl=0.33), s(1e-4, kl=3.0)) == (5e-5, 0.33, False)  # then the smaller learning rate


def test_kl_and_half_length_candidates():
    with sweep_world(kl=1.0) as (gate, plan):
        refused(["--part", "train"], "/half candidate needs its full-length run", cands="1e-4/half")
        assert run("--n", "20", "--plan", cands="1e-4/half") == 0 and not CALLS  # --plan only reports
        refused(["--part", "train"], "equals round 3's own KL weight", cands="1e-4/kl1")
        refused(["--part", "train"], "cannot read candidate", cands="1e-4/kl")
        refused(["--part", "train"], "cannot read candidate", cands="1e-4/half/half")
        refused(["--part", "train"], "must be distinct", cands="1e-4/kl3,1e-4/kl3.0")
        sys.argv = ["strength_sweep.py", "--round", "3", "--lrs", "1e-4/kl3", "--part", "train"]
        try:
            SW.main()
            raise AssertionError("--lrs with /kl should have been refused")
        except SystemExit as exc:
            assert "--lrs takes learning rates only" in str(exc), exc
        assert not CALLS
        assert run("--n", "20", cands="1e-4,1e-4/kl3,1e-4/kl0.33,1e-4/half") == 0
        full, kl3, half = SW.Cand(1e-4), SW.Cand(1e-4, 3.0), SW.Cand(1e-4, None, True)
        t = yaml.safe_load(open(os.path.join(SW.cand_dir(3, kl3), "train.yaml")))
        assert t["kl_think_beta"] == 3.0 and t["short_run_fractions"] == [0.5]
        assert yaml.safe_load(open(os.path.join(SW.cand_dir(3, full), "train.yaml")))["kl_think_beta"] == 1.0
        assert os.path.basename(SW.cand_dir(3, kl3)) == "r003_lr1.00e-04_kl3"
        assert os.path.basename(SW.cand_dir(3, half)) == "r003_lr1.00e-04_half"
        # the half-length run: a decay branch from the full run's checkpoint where a 10-step run starts decaying
        b = json.load(open(os.path.join(SW.cand_dir(3, half), "branch.json")))
        ms = os.path.join(SW.cand_dir(3, full), "train", "milestones", "checkpoint-8")
        assert b == {"from": ms, "start": 8, "decay": 2, "steps": 10, "full_steps": STEPS, "fraction": 0.5}, b
        ht = yaml.safe_load(open(os.path.join(SW.cand_dir(3, half), "train.yaml")))
        assert ht["branch_decay_steps"] == 2 and "short_run_fractions" not in ht and ht["lr"] == 1e-4
        branch_calls = [a for a in calls("train_longrun.py") if a.get("--branch-from")]
        assert len(branch_calls) == 1 and branch_calls[0]["--branch-from"] == ms
        assert SW.adapter_dir(3, half).endswith(os.path.join("r003_lr1.00e-04_half", "train", "branches",
                                                             "checkpoint-8", "adapter"))
        assert calls("convert_lora_to_gguf.py")[-1]["path"] == SW.adapter_dir(3, half)
        assert all(a["--adapter"] == SW.adapter_dir(3, half) for a in calls("val_loss.py")[-len(L.HELDOUT):])
        sums = {c: json.load(open(os.path.join(SW.cand_dir(3, c), "summary.json")))
                for c in (full, kl3, SW.Cand(1e-4, 0.33), half)}
        assert sums[half]["half"] and sums[half]["train"]["global_step"] == 10 and sums[kl3]["kl"] == 3.0
        for c, sm in sums.items():  # each answered at its own strength
            sel = json.load(open(os.path.join(SW.cand_dir(3, c), "selection.json")))["ids"]
            want = want_gate(gate, sel, strength(c.lr, 1.0 if c.kl is None else c.kl, c.half))
            assert sm["gate"]["all"]["cand"]["pass"] == sum(p for p, _ in want.values()), c
        choice = json.load(open(SW.choice_path(3)))
        best = max(sums.values(), key=SW.rank_key)
        assert choice["chosen"]["tag"] == best["tag"] and len(choice["candidates"]) == 4 and "close_call" not in choice

        # adopting the half-length run puts the branch's adapter where the loop looks for it
        json.dump(dict(choice, chosen={"lr": 1e-4, "kl": None, "half": True, "tag": half.tag,
                                       "folder": SW.cand_dir(3, half)}), open(SW.choice_path(3), "w"))
        assert run("--part", "adopt", cands="1e-4/half") == 0
        dst = os.path.join(SW.round_dir(3), "train")
        assert SW.tree_sha(SW.adapter_dir(3, half)) == SW.tree_sha(os.path.join(dst, "adapter"))
        assert not os.path.exists(os.path.join(dst, "branches")) and not os.path.exists(os.path.join(dst, "checkpoints"))
        done = json.load(open(os.path.join(SW.round_dir(3), "stages", "train.done.json")))
        assert done["global_step"] == 10 and done["strength_sweep"]["half"] and done["strength_sweep"]["branch"] == b
        assert done["strength_sweep"]["adapter_from"] == SW.adapter_dir(3, half)


def test_close_call():
    with sweep_world():
        SW.CLOSE_MARGIN = -1  # never close: picked on the 20
        assert run("--n", "20", "--close-call", lrs=LRS[:2]) == 0
        c1 = json.load(open(SW.choice_path(3)))
        assert c1["close_call"]["triggered"] is False and "pair" not in c1["close_call"]
        assert not any(os.path.exists(os.path.join(SW.cand_dir(3, lr), "summary-n40.json")) for lr in LRS)
    with sweep_world() as (gate, plan):
        SW.CLOSE_MARGIN = 10 ** 6  # always close: the top two answer 20, the pick is made on all 20
        assert run("--n", "10", lrs=LRS) == 0  # no --close-call: picked on the 10
        first = json.load(open(SW.choice_path(3)))
        assert "close_call" not in first
        CALLS.clear()
        T.FakeServer.instances.clear()
        assert run("--n", "10", "--part", "choose", "--close-call", lrs=LRS) == 0
        assert not calls("train_longrun.py") and not calls("convert_lora_to_gguf.py")  # nothing trains again
        ch = json.load(open(SW.choice_path(3)))
        pair = [p["tag"] for p in ch["close_call"]["pair"]]
        assert ch["close_call"]["triggered"] and ch["close_call"]["n"] == 20 and len(pair) == 2
        ranked = sorted((json.load(open(os.path.join(SW.cand_dir(3, lr), "summary.json"))) for lr in LRS),
                        key=SW.rank_key, reverse=True)
        assert pair == [r["tag"] for r in ranked[:2]]  # the top two on the 10
        for lr in LRS:
            d = SW.cand_dir(3, lr)
            ext = os.path.join(d, "summary-n20.json")
            if SW.Cand(lr).tag in pair:
                sm = json.load(open(ext))
                ids = json.load(open(os.path.join(d, "selection-n20.json")))["ids"]
                assert set(json.load(open(os.path.join(d, "selection.json")))["ids"]) < set(ids)  # the same 10 + 10 more
                assert sm["gate"]["all"]["n"] == 20 and len(E.read_jsonl(os.path.join(d, "gate.seed1.jsonl"))) == 20
                want = want_gate(gate, ids, strength(lr))
                assert sm["gate"]["all"]["cand"]["pass"] == sum(p for p, _ in want.values())
            else:
                assert not os.path.exists(ext) and len(E.read_jsonl(os.path.join(d, "gate.seed1.jsonl"))) == 10
        exts = [json.load(open(os.path.join(SW.cand_dir(3, SW.Cand(float(p["lr"]))), "summary-n20.json")))
                for p in ch["close_call"]["pair"]]
        assert ch["chosen"]["tag"] == max(exts, key=SW.rank_key)["tag"]
        # the earlier pick is kept beside the new one
        kept = [f for f in os.listdir(SW.OUT_ROOT) if f.startswith("r003_choice.") and f != "r003_choice.json"]
        assert len(kept) == 1 and json.load(open(os.path.join(SW.OUT_ROOT, kept[0]))) == first


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
