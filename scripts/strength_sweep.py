"""A loop round trained at several learning rates outside the loop, each screened the same way, and the best one handed
back to the loop for its full keep test (30 Oct 2026, my ask: "i want major testing for the right strength").

Part 1 of the sweep scaled round 2's finished adapter (scripts/lora_scale_check.py, scales 0, 0.25, 0.5, 0.75 and 1).
This is part 2: the strength set where it is made, by the learning rate, on round N's own data and settings. The loop
plans round N and builds its data, then waits (loop.yaml hold_before_train lists N). Here, for each learning rate:

  train    scripts/train_longrun.py on the round's data (rounds/rNNN/data/mix) with the round's settings
           (stages/plan.done.json "train"), only lr changed; then the GGUF LoRA, as loop.py's convert stage makes it.
           Resumable (--resume); a STOP file in the training folder pauses it with a checkpoint.
  loss     the held-out bug-fix and terminal steps (loop.yaml swe_loss and term_loss) with scripts/val_loss.py
           --no-base, compared step by step with the base's losses on record.
  bfcl     BFCL's 150 Java and JavaScript tool calls on the gate server's settings and the guard's seed, paired per item
           with the base's rows (only rows scored under today's settings).
  gate     the same 60 gate problems as part 1 (lora_scale_check's draw: salt "lora-scale", 30 write + 30 fix), with
           the base's request seeds, on the gate server's exact settings, judged by the loop's gate checker; paired per
           problem with the base's answers.
  summary  each learning rate's numbers -> summary.json in its folder (numbers only).
  choose   the rule fixed before any result (30 Oct 2026): the most gate passes on the 60; on a tie, more BFCL passes;
           then the lower held-out loss (mean of the two changes against the base); then the lower learning rate (the
           smaller change). -> rNNN_choice.json.
  adopt    the chosen learning rate's training folder (checkpoints left out) becomes rounds/rNNN/train, with its
           train.yaml and stages/train.done.json written as loop.py's train stage writes them, plus a strength_sweep
           record. The loop then carries on from convert (a held round only waits while it has no train stage) and
           gives that adapter the full keep test with fresh answers on all 200 gate problems, so the screening answers
           that picked it never count towards keeping it.

  python scripts\\strength_sweep.py --round 3 --lrs 1e-4,5e-5,2.5e-5 [--n 60] [--part all|train|loss|bfcl|gate|
         summary|choose|adopt] [--plan]

all = train, loss, bfcl, gate and summary for each learning rate in the order given, then choose (adopt only when
asked). Every part runs only while loop.py is not running and holds round N untrained, with no llama-server.exe
running. Output: results/research/strength/rNNN_lr<lr> (kept, never deleted) and rNNN_choice.json.
"""

import argparse
import copy
import hashlib
import json
import os
import shutil
import statistics
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import lora_scale_check as S  # noqa: E402

L, E, ew, llama = S.L, S.E, S.ew, S.llama
PY = sys.executable
LOOP_DIR = os.path.join(ROOT, "results", "loop")
OUT_ROOT = os.path.join(ROOT, "results", "research", "strength")
PORT = 18094  # not the loop's, not lora_scale_check's
TRAIN_DONE_KEYS = ("global_step", "total_steps", "trainable_params", "rows", "rows_dropped_too_long", "tokens",
                   "peak_reserved_gib", "session_seconds", "adapter", "metrics", "data_sha256", "init_adapter")
log, read_json, write_json, resolve = S.log, S.read_json, S.write_json, S.resolve


def lr_tag(lr):
    return f"{float(lr):.2e}"


def round_dir(rnum):
    return os.path.join(LOOP_DIR, "rounds", f"r{rnum:03d}")


def cand_dir(rnum, lr):
    return os.path.join(OUT_ROOT, f"r{rnum:03d}_lr{lr_tag(lr)}")


def choice_path(rnum):
    return os.path.join(OUT_ROOT, f"r{rnum:03d}_choice.json")


def maybe_json(path):
    try:
        return read_json(path)
    except (OSError, ValueError):
        return None


def run_cmd(cmd, log_path):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    log(subprocess.list2cmdline(cmd))
    with open(log_path, "a", encoding="utf-8", errors="replace") as fh:
        return subprocess.run(cmd, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT,
                              env=dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")).returncode


# ============================================================================================ checks

def hold_check(cfg, rnum):
    """None when loop.py is not running and holds round rnum before training (data built, no train stage); else why
    not."""
    try:
        with open(os.path.join(LOOP_DIR, "loop.pid"), encoding="utf-8") as fh:
            pid = int(fh.read().strip() or 0)
    except (OSError, ValueError):
        pid = 0
    if pid and L.pid_alive(pid):
        return f"loop.py is running (pid {pid})"
    if rnum not in {int(r) for r in (cfg.get("hold_before_train") or [])}:
        return f"loop.yaml hold_before_train does not list round {rnum}"
    stages = os.path.join(round_dir(rnum), "stages")
    if not os.path.exists(os.path.join(stages, "data.done.json")):
        return f"round {rnum} has no data yet (the loop has not reached its hold)"
    if os.path.exists(os.path.join(stages, "train.done.json")):
        return f"round {rnum} already has a train stage"
    return None


def require_gpu_free(cfg, rnum):
    why = hold_check(cfg, rnum)
    if why:
        raise SystemExit(f"not started: {why}")
    if S.llama_running():
        raise SystemExit("not started: llama-server.exe is running")


def round_inputs(rnum):
    """The round's plan and data files, as loop.py's train stage reads them."""
    rdir = round_dir(rnum)
    plan = read_json(os.path.join(rdir, "stages", "plan.done.json"))
    mix = os.path.join(rdir, "data", "mix")
    files = {k: os.path.join(mix, f"{k}.jsonl") for k in ("train", "val")}
    missing = [p for p in files.values() if not os.path.exists(p)]
    if missing:
        raise SystemExit(f"round {rnum}'s data is missing: {missing}")
    return plan, files


def base_guard(cfg, state, name):
    """(items, seed, base rows) of a no-worse check, the base's rows only if scored under today's settings."""
    gc = (cfg.get("guards") or {}).get(name)
    if not isinstance(gc, dict):
        raise SystemExit(f"loop.yaml has no guards.{name}")
    fp = L.Loop.guard_fingerprint(S._LoopView(), cfg, name)
    have = (state.get("base_guards") or {}).get(name) or {}
    if have.get("fingerprint") != fp:
        diff = sorted(k for k in set(fp) | set(have.get("fingerprint") or {})
                      if fp.get(k) != (have.get("fingerprint") or {}).get(k))
        raise SystemExit(f"the base's {name} rows were scored under other settings ({diff}); nothing to pair with")
    seed = int(gc.get("seed", 1))
    items = E.read_jsonl(L.Loop.guard_file(None, cfg, name))
    base_rows = {r["id"]: r for r in E.read_jsonl(resolve(have["rows"])) if r["seed"] == seed}
    missing = [it["id"] for it in items if it["id"] not in base_rows]
    if not items or missing:
        raise SystemExit(f"{name}: {len(items)} items, {len(missing)} without a seed-{seed} answer from the base")
    return items, seed, base_rows


def adapter_probe(url):
    """Greedy probes with the adapter at 1, 0 and 1 again (lora_scale_check's prompts and tolerances); leaves it at 1.
    ok when the two runs at 1 agree and differ from the run at 0, so the adapter is loaded and doing something."""
    if set(S.lora_scales(url)) != {0}:
        raise SystemExit(f"expected one LoRA adapter on the server, /lora-adapters reads {S.lora_scales(url)}")
    runs = {}
    for name, s in (("one", 1.0), ("zero", 0.0), ("one2", 1.0)):
        S.set_scale(url, s)
        runs[name] = S.probe_once(url)
    res = {"repeat": S.probe_diff(runs["one"], runs["one2"]), "on_vs_off": S.probe_diff(runs["one"], runs["zero"])}
    res["need"] = max(S.DIFF_MIN, 3 * res["repeat"])
    res["ok"] = res["repeat"] <= S.REPEAT_TOL and res["on_vs_off"] > res["need"]
    res["scale_now"] = S.lora_scales(url)
    return {k: ("different text" if v == float("inf") else v) for k, v in res.items()}


# ============================================================================================ parts

def train_part(cfg, rnum, lr):
    """Trains (or resumes) this learning rate's adapter, then converts it; returns the GGUF path."""
    d = cand_dir(rnum, lr)
    tdir = os.path.join(d, "train")
    st = maybe_json(os.path.join(tdir, "status.json")) or {}
    if not (st.get("state") == "finished" and os.path.exists(os.path.join(tdir, "summary.json"))):
        require_gpu_free(cfg, rnum)
        free = shutil.disk_usage(ROOT).free / 2 ** 30
        if free < cfg.get("min_free_gb", 60):  # the loop's own rule before a round trains
            raise SystemExit(f"not started: only {free:.0f} GB free on the training drive "
                             f"(need {cfg.get('min_free_gb', 60)})")
        plan, files = round_inputs(rnum)
        t = copy.deepcopy(plan["train"])
        t.update(run_name=f"loop-r{rnum:03d}-lr{lr_tag(lr)}", train_file=files["train"].replace("\\", "/"),
                 val_file=files["val"].replace("\\", "/"), lr=float(lr))
        if plan.get("init_adapter"):
            t["init_adapter"] = plan["init_adapter"].replace("\\", "/")
        os.makedirs(d, exist_ok=True)
        ypath = os.path.join(d, "train.yaml")
        import yaml
        with open(ypath, "w", encoding="utf-8") as fh:
            fh.write(f"# loop round {rnum} at learning rate {float(lr):g} (scripts/strength_sweep.py): recipe "
                     f"{plan['recipe']['id']}, init {plan.get('init')}; every other setting is the round's own\n")
            yaml.safe_dump(t, fh, sort_keys=False)
        t0 = time.time()
        rc = run_cmd([PY, "-u", os.path.join(ROOT, "scripts", "train_longrun.py"), "--config", ypath, "--run-dir", tdir,
                      "--resume"], os.path.join(d, "logs", "train_longrun.log"))
        st = maybe_json(os.path.join(tdir, "status.json")) or {}
        if st.get("state") == "paused":
            raise SystemExit(f"lr {lr:g}: training paused (STOP in {tdir}); run again to resume")
        if rc != 0 or st.get("state") != "finished":
            raise SystemExit(f"lr {lr:g}: training ended with exit code {rc}, state {st.get('state')}: "
                             f"{st.get('error')}")
        s = maybe_json(os.path.join(tdir, "summary.json")) or {}
        if plan.get("init_adapter") and not s.get("init_adapter"):
            raise SystemExit(f"lr {lr:g}: train_longrun.py did not report loading init_adapter {plan['init_adapter']}")
        log(f"lr {lr:g}: trained {s.get('global_step')} steps in {(time.time() - t0) / 3600:.1f} h")
    gguf = os.path.join(d, "convert", f"loop-r{rnum:03d}-lr{lr_tag(lr)}-F16-LoRA.gguf")
    if not os.path.exists(gguf):
        os.makedirs(os.path.dirname(gguf), exist_ok=True)
        part = gguf + ".part"
        rc = run_cmd([PY, "-u", L.fr.CONVERT, "--base", llama.LORA_BASE_CONFIG, "--outtype", "f16", "--outfile", part,
                      os.path.join(tdir, "adapter")], os.path.join(d, "logs", "convert_lora.log"))
        if rc != 0 or not os.path.exists(part):
            raise SystemExit(f"lr {lr:g}: convert_lora_to_gguf.py exited with {rc}")
        os.replace(part, gguf)
        log(f"lr {lr:g}: GGUF {os.path.getsize(gguf)} bytes")
    return gguf


def loss_part(cfg, state, rnum, lr):
    d = cand_dir(rnum, lr)
    for kind, h in L.HELDOUT.items():
        sc = cfg.get(h["key"]) or {}
        if not sc:
            continue
        path = L.Loop.heldout_file(None, cfg, kind)
        ref = (state.get("base") or {}).get(h["key"])
        if not ref:
            raise SystemExit(f"no base {h['label']} losses in state.json")
        if ref.get("sha") and ref["sha"] != L.file_sha(path):
            raise SystemExit(f"{path} changed since the base was scored on it")
        vdir = os.path.join(d, f"loss_{kind}")
        if os.path.exists(os.path.join(vdir, "summary.json")):
            continue
        require_gpu_free(cfg, rnum)
        t0 = time.time()
        rc = run_cmd([PY, "-u", os.path.join(ROOT, "scripts", "val_loss.py"), "--adapter",
                      os.path.join(d, "train", "adapter"), "--val-file", path,
                      "--seq-len", str(int(sc.get("seq_len", 8192))), "--max-rows", str(int(sc.get("max_rows", 200))),
                      "--no-base", "--run-dir", vdir],
                     os.path.join(d, "logs", f"loss_{kind}.log"))
        if rc != 0 or not os.path.exists(os.path.join(vdir, "summary.json")):
            raise SystemExit(f"lr {lr:g}: val_loss.py ({h['label']}) exited with {rc}")
        log(f"lr {lr:g}: {h['label']} scored in {(time.time() - t0) / 60:.0f} min")


def serve(cfg, gguf, log_path, port):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    return llama.Server(L.Loop.gate_flags(None, cfg, lora=gguf), log_path, port=port,
                        binary=llama.server_binary(L.gate_build(cfg)))


def probe_or_stop(srv, d, tag):
    probe = adapter_probe(srv.url)
    write_json(os.path.join(d, f"probe-{tag}-{time.strftime('%Y%m%d-%H%M%S')}.json"), probe)
    log("adapter probe: " + json.dumps(probe))
    if not probe["ok"]:
        raise SystemExit("adapter probe failed: the adapter could not be shown to change the answers; nothing answered")


def check_scale_after(srv, d, tag):
    after = S.lora_scales(srv.url)
    write_json(os.path.join(d, f"scale_after-{tag}.json"), {"scales": {str(k): v for k, v in after.items()}})
    if after != {0: 1.0}:
        raise SystemExit(f"the adapter's scale read {after} after answering, not 1.0")


def bfcl_part(cfg, state, rnum, lr, gguf, port):
    name, d = "bfcl", cand_dir(rnum, lr)
    items, seed, _ = base_guard(cfg, state, name)
    path = os.path.join(d, f"{name}.seed{seed}.jsonl")
    if all(it["id"] in {r["id"] for r in E.read_jsonl(path) if r["seed"] == seed} for it in items):
        return
    require_gpu_free(cfg, rnum)
    S.gate_server_check(cfg, state)
    mod = L.GUARDS[name]["module"]
    mod.prepare()
    g = cfg["gate"]
    srv = serve(cfg, gguf, os.path.join(d, "logs", f"server-{name}.log"), port)
    log(f"llama-server ({name}, lr {lr:g}): {srv.describe()}")
    with srv:
        probe_or_stop(srv, d, name)
        first = [r for r in E.read_jsonl(path) if r["seed"] == seed][:L.GUARD_PROBE_ROWS]
        err = mod.probe(first) if len(first) == L.GUARD_PROBE_ROWS else None
        if err:
            raise SystemExit(err)

        def on_row(r, k, tot, tps):
            log(f"{name} {k}/{tot}: {r['id']} pass {r['pass']} verdict {r.get('verdict')} ({tps:.1f} tok/s)")
            if len(first) < L.GUARD_PROBE_ROWS:
                first.append(r)
                bad = mod.probe(first) if len(first) == L.GUARD_PROBE_ROWS else None
                if bad:
                    raise RuntimeError(bad)
        try:
            E.score(srv.url, items, path, seed, g["max_tokens"], None, None, ew.count_tokens, g["budget"],
                    concurrency=L.gate_concurrency(cfg), stop_check=lambda: os.path.exists(os.path.join(d, "STOP")),
                    log=log, request=lambda u, it, sd, mt: mod.request(u, it, sd, mt, ew.SAMPLING), judge=mod.judge,
                    on_row=on_row)
        except E.Stopped:
            log(f"{name}: stopped by STOP; answers so far kept")
        check_scale_after(srv, d, name)


def gate_part(cfg, state, rnum, lr, gguf, n, port):
    d = cand_dir(rnum, lr)
    gate = E.read_jsonl(E.GATE_PATH)
    base = state.get("base") or {}
    seed = int(base["gate_seed"])
    picked = S.draw(gate, n)
    base_rows = S.seed_rows(base["gate_rows"], seed)
    missing = [gate[i]["id"] for i in picked if gate[i]["id"] not in base_rows]
    if missing:
        raise SystemExit(f"{len(missing)} drawn problems have no seed-{seed} answer from the base")
    write_json(os.path.join(d, "selection.json"), {"salt": S.SALT, "n": n, "ids": [gate[i]["id"] for i in picked]})
    gpath = os.path.join(d, f"gate.seed{seed}.jsonl")
    if all(gate[i]["id"] in S.seed_rows(gpath, seed) for i in picked):
        return
    order = sorted(picked, key=lambda i: (-(base_rows[gate[i]["id"]].get("completion_tokens") or 0), i))
    require_gpu_free(cfg, rnum)
    S.gate_server_check(cfg, state)
    S.toolchain_check(gate, state)
    g = cfg["gate"]
    srv = serve(cfg, gguf, os.path.join(d, "logs", "server-gate.log"), port)
    log(f"llama-server (gate, lr {lr:g}): {srv.describe()}")
    with srv:
        probe_or_stop(srv, d, "gate")

        def on_row(r, k, tot, tps):
            log(f"gate {k}/{tot}: {r['id']} pass {r['pass']} reasoning {r['reasoning_tokens']} finish "
                f"{r['finish_reason']} ({tps:.1f} tok/s)")
        try:
            E.score(srv.url, gate, gpath, seed, g["max_tokens"], L.Loop.gate_checker(None),
                    lambda u, m, s, mt: ew.chat_request(u, m, s, mt), ew.count_tokens, g["budget"],
                    concurrency=L.gate_concurrency(cfg), stop_check=lambda: os.path.exists(os.path.join(d, "STOP")),
                    log=log, on_row=on_row, order=order)
        except E.Stopped:
            log("gate: stopped by STOP; answers so far kept")
        check_scale_after(srv, d, "gate")


# ============================================================================================ summary, choose, adopt

def summary_part(cfg, state, rnum, lr, n, quiet=False):
    d = cand_dir(rnum, lr)
    res = {"round": rnum, "lr": float(lr), "time": time.strftime("%Y-%m-%d %H:%M:%S")}
    tsum = maybe_json(os.path.join(d, "train", "summary.json")) or {}
    res["train"] = {k: tsum.get(k) for k in ("global_step", "rows", "tokens", "peak_reserved_gib", "session_seconds")}
    gate = E.read_jsonl(E.GATE_PATH)
    base = state.get("base") or {}
    seed = int(base["gate_seed"])
    gpath = os.path.join(d, f"gate.seed{seed}.jsonl")
    if os.path.exists(gpath):
        picked = S.draw(gate, n)
        brows, crows = S.seed_rows(base["gate_rows"], seed), S.seed_rows(gpath, seed)
        res["gate"] = {}
        for name in ("all", "write", "fix"):
            idx = [i for i in picked if name == "all" or gate[i]["kind"] == name]
            ids = [gate[i]["id"] for i in idx if gate[i]["id"] in crows]
            gsum = {"n": len(ids), "drawn": len(idx)}
            if ids:
                gsum.update(base=S.stats(brows, ids), cand=S.stats(crows, ids), vs_base=S.paired(brows, crows, ids))
            res["gate"][name] = gsum
    seed_b = int(((cfg.get("guards") or {}).get("bfcl") or {}).get("seed", 1))
    bpath = os.path.join(d, f"bfcl.seed{seed_b}.jsonl")
    if os.path.exists(bpath):
        items, _, brows = base_guard(cfg, state, "bfcl")
        crows = {r["id"]: r for r in E.read_jsonl(bpath) if r["seed"] == seed_b}
        mod = L.GUARDS["bfcl"]["module"]
        res["bfcl"] = {}
        for lang in ["all"] + sorted({mod.language(it) for it in items}):
            sel = [it for it in items if lang == "all" or mod.language(it) == lang]
            ids = [it["id"] for it in sel if it["id"] in crows]
            b = {"n": len(ids), "items": len(sel)}
            if ids:
                b.update(base=sum(1 for i in ids if brows[i]["pass"]), cand=sum(1 for i in ids if crows[i]["pass"]),
                         vs_base=S.paired(brows, crows, ids))
            res["bfcl"][lang] = b
    res["loss"] = {}
    for kind, h in L.HELDOUT.items():
        ps = os.path.join(d, f"loss_{kind}", "summary.json")
        ref = (state.get("base") or {}).get(h["key"])
        if not os.path.exists(ps) or not ref:
            continue
        part = read_json(ps)
        bsum = read_json(resolve(ref["path"]))
        cmp = E.compare_loss({r["key"]: r[ref["field"]] for r in bsum["per_row"] if r.get(ref["field"]) is not None},
                             {r["key"]: r["adapter_loss"] for r in part["per_row"]},
                             {r["key"]: r["trained"] for r in part["per_row"]})
        res["loss"][kind] = {k: cmp[k] for k in ("n", "best", "candidate", "rel_diff", "wins", "losses", "p_one_sided")}
    write_json(os.path.join(d, "summary.json"), res)
    if quiet:
        return res
    g, b = (res.get("gate") or {}).get("all") or {}, (res.get("bfcl") or {}).get("all") or {}
    log(f"lr {lr:g}: gate {g.get('n', 0)} of {g.get('drawn', n)}"
        + (f": pass base {g['base']['pass']} / lr {g['cand']['pass']}, reasoning mean {g['base']['reasoning_mean']} / "
           f"{g['cand']['reasoning_mean']}, {g['vs_base']['wins']} won {g['vs_base']['losses']} lost "
           f"(p {g['vs_base']['p_two_sided']})" if g.get("n") else "")
        + (f"; bfcl pass base {b['base']} / lr {b['cand']} of {b['n']}" if b.get("n") else "")
        + "".join(f"; {k} loss {v['best']:.4f} -> {v['candidate']:.4f} ({100 * v['rel_diff']:+.1f}%)"
                  for k, v in res["loss"].items()))
    return res


def rank_key(s):
    """The fixed rule: more gate passes, then more BFCL passes, then the lower held-out loss, then the lower lr."""
    loss = statistics.mean(v["rel_diff"] for v in s["loss"].values())
    return (s["gate"]["all"]["cand"]["pass"], s["bfcl"]["all"]["cand"], -loss, -s["lr"])


def choose_part(cfg, state, rnum, lrs, n):
    rows, bad = [], []
    kinds = [k for k, h in L.HELDOUT.items() if cfg.get(h["key"])]
    for lr in lrs:
        s = summary_part(cfg, state, rnum, lr, n, quiet=True)
        g, b = (s.get("gate") or {}).get("all") or {}, (s.get("bfcl") or {}).get("all") or {}
        if not g.get("n") or g["n"] != g["drawn"] or not b.get("n") or b["n"] != b["items"] or \
                sorted(s["loss"]) != sorted(kinds):
            bad.append(f"lr {lr:g}")
        rows.append(s)
    if bad:
        raise SystemExit(f"not finished: {', '.join(bad)} (every part must be complete before choosing)")
    best = max(rows, key=rank_key)
    out = {"round": rnum, "n": n, "time": time.strftime("%Y-%m-%d %H:%M:%S"),
           "rule": "most gate passes on the drawn problems; then most BFCL passes; then the lower mean relative "
                   "held-out loss change against the base; then the lower learning rate",
           "chosen": {"lr": best["lr"], "folder": cand_dir(rnum, best["lr"])},
           "base": {"gate_pass": rows[0]["gate"]["all"]["base"]["pass"], "bfcl_pass": rows[0]["bfcl"]["all"]["base"]},
           "candidates": [{"lr": s["lr"], "gate_pass": s["gate"]["all"]["cand"]["pass"],
                           "gate_vs_base": s["gate"]["all"]["vs_base"], "bfcl_pass": s["bfcl"]["all"]["cand"],
                           "bfcl_vs_base": s["bfcl"]["all"]["vs_base"],
                           "reasoning_mean": s["gate"]["all"]["cand"]["reasoning_mean"],
                           "loss_rel": {k: v["rel_diff"] for k, v in s["loss"].items()}} for s in rows]}
    write_json(choice_path(rnum), out)
    log(f"chosen: lr {best['lr']:g} (" + "; ".join(
        f"lr {c['lr']:g}: gate {c['gate_pass']}, bfcl {c['bfcl_pass']}" for c in out["candidates"])
        + f"; base gate {out['base']['gate_pass']}, bfcl {out['base']['bfcl_pass']})")
    return out


def tree_sha(folder):
    h = hashlib.sha256()
    for f in sorted(os.listdir(folder)):
        p = os.path.join(folder, f)
        if os.path.isfile(p):
            h.update(f.encode() + b"\0" + S.file_sha256(p).encode())
    return h.hexdigest()[:16]


def adopt_part(cfg, rnum):
    choice = maybe_json(choice_path(rnum))
    if not choice:
        raise SystemExit(f"no {choice_path(rnum)}: run --part choose first")
    why = hold_check(cfg, rnum)
    if why:
        raise SystemExit(f"not adopted: {why}")
    lr = float(choice["chosen"]["lr"])
    src = os.path.join(cand_dir(rnum, lr), "train")
    if (maybe_json(os.path.join(src, "status.json")) or {}).get("state") != "finished":
        raise SystemExit(f"{src} is not a finished training run")
    rdir = round_dir(rnum)
    dst = os.path.join(rdir, "train")
    if os.path.exists(dst) and os.listdir(dst):
        raise SystemExit(f"{dst} already has files; nothing copied")
    shutil.copytree(src, dst, dirs_exist_ok=True,
                    ignore=lambda folder, names: [x for x in names if x == "checkpoints" or x.startswith("checkpoint-")
                                                  or x == "STOP"])
    if tree_sha(os.path.join(src, "adapter")) != tree_sha(os.path.join(dst, "adapter")):
        raise SystemExit("the copied adapter does not match the chosen one")
    with open(os.path.join(cand_dir(rnum, lr), "train.yaml"), encoding="utf-8") as fh:
        ytext = fh.read()
    with open(os.path.join(rdir, "train.yaml"), "w", encoding="utf-8") as fh:
        fh.write(f"# adopted from {cand_dir(rnum, lr)} by scripts/strength_sweep.py ({choice_path(rnum)})\n" + ytext)
    s = read_json(os.path.join(dst, "summary.json"))
    done = {k: s.get(k) for k in TRAIN_DONE_KEYS}
    metrics = os.path.join(dst, "metrics.jsonl")
    done.update(adapter=os.path.join(dst, "adapter"), metrics=metrics if os.path.exists(metrics) else None,
                strength_sweep={"lr": lr, "trained_in": src, "choice": choice_path(rnum),
                                "lrs": [c["lr"] for c in choice["candidates"]]},
                finished=time.strftime("%Y-%m-%d %H:%M:%S"))
    write_json(os.path.join(rdir, "stages", "train.done.json"), done)
    log(f"adopted lr {lr:g}: {dst} (adapter {tree_sha(os.path.join(dst, 'adapter'))}); the loop carries on from "
        f"convert at its next start")
    return done


# ============================================================================================ main

def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--round", type=int, default=3)
    ap.add_argument("--lrs", required=True, help="learning rates, comma-separated, e.g. 1e-4,5e-5,2.5e-5")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--part", choices=["all", "train", "loss", "bfcl", "gate", "summary", "choose", "adopt"],
                    default="all")
    ap.add_argument("--port", type=int, default=PORT)
    ap.add_argument("--plan", action="store_true", help="check the inputs and print the plan; start nothing")
    args = ap.parse_args()
    lrs = [float(x) for x in args.lrs.split(",") if x.strip()]
    if not lrs or len(set(map(lr_tag, lrs))) != len(lrs) or any(not 0 < x < 1e-2 for x in lrs):
        raise SystemExit(f"--lrs must be distinct learning rates between 0 and 0.01, got {args.lrs}")
    cfg = L.load_config(L.CONFIG)
    state = read_json(os.path.join(LOOP_DIR, "state.json"))
    rnum = args.round
    if args.plan:
        gate = E.read_jsonl(E.GATE_PATH)
        picked = S.draw(gate, args.n)
        log(f"round {rnum}, learning rates {', '.join(f'{x:g}' for x in lrs)}; {len(picked)} gate problems, draw sha "
            f"{hashlib.sha256(json.dumps([gate[i]['id'] for i in picked]).encode()).hexdigest()[:16]}")
        log(f"hold: {hold_check(cfg, rnum) or 'loop not running and holding round %d before training' % rnum}; "
            f"llama-server running: {S.llama_running()}")
        try:
            plan, files = round_inputs(rnum)
            log(f"round data: {plan.get('rows')} rows, recipe {plan['recipe']['id']}, init {plan.get('init')}, "
                f"lr on record {plan['train'].get('lr')}")
        except (OSError, ValueError, KeyError, SystemExit) as exc:
            log(f"round data: not ready: {exc}")
        try:
            items, seed, _ = base_guard(cfg, state, "bfcl")
            log(f"bfcl: {len(items)} items, seed {seed}; base rows found under today's settings")
        except SystemExit as exc:
            log(f"bfcl: not ready: {exc}")
        for lr in lrs:
            log(f"lr {lr:g}: {cand_dir(rnum, lr)}")
        return 0
    if args.part == "adopt":
        adopt_part(cfg, rnum)
        return 0
    if args.part == "choose":
        choose_part(cfg, state, rnum, lrs, args.n)
        return 0
    for lr in lrs:
        if args.part in ("all", "train", "bfcl", "gate"):
            gguf = train_part(cfg, rnum, lr)
        if args.part in ("all", "loss"):
            loss_part(cfg, state, rnum, lr)
        if args.part in ("all", "bfcl"):
            bfcl_part(cfg, state, rnum, lr, gguf, args.port)
        if args.part in ("all", "gate"):
            gate_part(cfg, state, rnum, lr, gguf, args.n, args.port)
        summary_part(cfg, state, rnum, lr, args.n)
    if args.part == "all":
        choose_part(cfg, state, rnum, lrs, args.n)
    log("STRENGTH SWEEP DONE" if args.part == "all" else f"strength sweep part {args.part} done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
