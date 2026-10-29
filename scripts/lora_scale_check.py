"""A training round's adapter at a fraction of its strength (29 Oct 2026, my question: when round 2 does worse than the
base, does that point to the LoRA settings rather than to the LoRA itself?). A diagnostic only: never a gate score and
never part of the keep rule.

Round 2 thought less than the base on the gate and lost problems, while its loss on the held-out bug-fix and terminal
steps fell by about 14%. A LoRA's change to the model is scale x (alpha / r) x B A, so the same adapter at half strength
can tell two causes apart:
  - thinking length and passes come back towards the base while most of the loss gain stays: the change was too
    strong for this data, so the settings (learning rate, steps, rank, alpha) are part of the cause;
  - thinking stays short at half strength: the data itself teaches the shorter thinking.

Parts (each resumes where it stopped; output in results/research/lora_scale/rNNN_xS):
  loss     the held-out bug-fix and terminal steps (loop.yaml swe_loss and term_loss, same file, rows and length) on a
           copy of the adapter whose lora_alpha (and alpha_pattern) is x S (PEFT scales by lora_alpha / r), through
           scripts/val_loss.py --no-base; compared row by row with the base and the round at full strength.
  gate     --n gate problems drawn without looking at any answer: the same number of each kind (write, fix), within
           a kind the first by sha256 of "lora-scale:" + id. Answered on the gate server's exact settings (loop.py
           gate_flags, build, slots, cache, budget, sampling) with the round's GGUF LoRA at scale S (POST
           /lora-adapters), with the request seeds of the base's and the round's answers, and judged by the loop's gate
           checker. Before any answer: the gate file, the gate server settings and the toolchains must be the ones the
           round was scored with, and a greedy probe must show that scales 1, S and 0 give three different outputs and
           that each one repeats.
  summary  pass counts, reasoning tokens, budget hits and finish reasons of the base, the round and the round at S on
           the same problems (all, write, fix), and the loss comparisons -> summary.json (numbers only).

  python scripts\\lora_scale_check.py [--round 2] [--scale 0.5] [--n 60] [--part all|loss|gate|summary] [--plan]

--plan checks the inputs and prints the draw, and starts nothing. The loss and gate parts run only while the loop is
not running and would hold if the watchdog started it (pause_after_rejects_in_row reached), and with no
llama-server.exe running. A STOP file in the output folder stops the gate part cleanly (answers so far are kept).
"""

import argparse
import hashlib
import json
import math
import os
import shutil
import statistics
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import evalplus_win as ew  # noqa: E402
import loop as L  # noqa: E402
from anneal import evalsets as E  # noqa: E402
from anneal import llama  # noqa: E402

PY = sys.executable
LOOP_DIR = os.path.join(ROOT, "results", "loop")
OUT_ROOT = os.path.join(ROOT, "results", "research", "lora_scale")
PORT = 18093  # not the loop's
SALT = "lora-scale"
# Greedy probes: plain text and code openings, no gate problem. Each token's log-probabilities (the chosen token's and
# the top 5, sorted) are compared up to the first token where two outputs part. The same scale twice must agree within
# REPEAT_TOL, and scales 1, S and 0 must each differ by more than max(DIFF_MIN, 3 x the largest repeat difference).
PROBES = ["def merge_sorted(a, b):\n",
          "#include <vector>\n#include <iostream>\n\nint main() {\n",
          "The main reasons to write unit tests before fixing a bug are",
          "Question: Why is a hash map lookup constant time on average?\nAnswer:"]
PROBE_TOKENS = 32
REPEAT_TOL = 0.02
DIFF_MIN = 0.05


def log(msg):
    print(time.strftime("%H:%M:%S ") + msg, flush=True)


def read_json(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def write_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=1)
    os.replace(tmp, path)


def resolve(path):
    return path if os.path.isabs(path) else os.path.join(ROOT, path)


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ============================================================================================ the draw

def quotas(counts, n):
    """n split across groups in proportion to counts (largest remainder; ties to the bigger group, then the name)."""
    total = sum(counts.values())
    if n < 1 or total <= 0:
        raise ValueError(f"cannot split {n} across {counts}")
    exact = {k: n * c / total for k, c in counts.items()}
    q = {k: int(math.floor(v)) for k, v in exact.items()}
    for k in sorted(counts, key=lambda k: (-(exact[k] - q[k]), -counts[k], k))[:n - sum(q.values())]:
        q[k] += 1
    return q


def draw(gate, n, salt=SALT):
    """Indexes of n gate items: the same number of each kind (fix problems are fewer on the gate, and round 2's
    thinking fell most on them), within a kind the first by sha256(salt:id). It never looks at an answer, so no
    problem is picked by how a model did on it, and every result is reported by kind as well as overall."""
    if not 0 < n <= len(gate):
        raise ValueError(f"--n must be between 1 and {len(gate)}, got {n}")
    by_kind = {}
    for i, it in enumerate(gate):
        by_kind.setdefault(it["kind"], []).append(i)
    q = quotas({k: 1 for k in by_kind}, n)
    short = [k for k in by_kind if q[k] > len(by_kind[k])]
    if short:
        raise ValueError(f"--n {n} asks for more {short} problems than the gate has")
    key = lambda i: hashlib.sha256(f"{salt}:{gate[i]['id']}".encode()).hexdigest()  # noqa: E731
    picked = []
    for k in sorted(by_kind):
        picked += sorted(by_kind[k], key=key)[:q[k]]
    return sorted(picked)


def seed_rows(path, seed):
    return {r["id"]: r for r in E.read_jsonl(resolve(path)) if r["seed"] == seed}


# ============================================================================================ inputs and checks

def round_info(rnum):
    """The round's GGUF LoRA, adapter folder and gate answers; missing pieces listed under "missing"."""
    rdir = os.path.join(LOOP_DIR, "rounds", f"r{rnum:03d}")
    out = {"rdir": rdir, "adapter": os.path.join(rdir, "train", "adapter"), "missing": []}
    try:
        conv = read_json(os.path.join(rdir, "stages", "convert.done.json"))
        out["gguf"] = resolve(conv["gguf"])
        if not os.path.exists(out["gguf"]):
            out["missing"].append(f"GGUF {out['gguf']}")
    except (OSError, ValueError, KeyError):
        out["missing"].append("stages/convert.done.json")
    try:
        ev = read_json(os.path.join(rdir, "stages", "evaluate.done.json"))
        out["gate_rows"], out["gate_seed"] = resolve(ev["gate_rows"]), int(ev["gate_seed"])
    except (OSError, ValueError, KeyError):
        out["missing"].append("stages/evaluate.done.json (the round's gate answers are not finished)")
    if not os.path.exists(os.path.join(out["adapter"], "adapter_config.json")):
        out["missing"].append(f"adapter {out['adapter']}")
    return out


def loop_would_hold(cfg, state):
    """None when the loop is not running and, if the watchdog started it, would wait instead of training a round;
    otherwise why not."""
    try:
        with open(os.path.join(LOOP_DIR, "loop.pid"), encoding="utf-8") as fh:
            pid = int(fh.read().strip() or 0)
    except (OSError, ValueError):
        pid = 0
    if pid and L.pid_alive(pid):
        return f"loop.py is running (pid {pid})"
    pause = int(cfg.get("pause_after_rejects_in_row") or 0)
    if not pause or int(state.get("rejects_in_row") or 0) < pause:
        return (f"the loop would train a round if started (pause_after_rejects_in_row {pause}, rejects in a row "
                f"{state.get('rejects_in_row')})")
    return None


def llama_running():
    if os.name != "nt":
        return False
    out = subprocess.run(["tasklist", "/FI", "IMAGENAME eq llama-server.exe"], capture_output=True, text=True).stdout
    return "llama-server.exe" in out


def require_gpu_free(cfg, state):
    why = loop_would_hold(cfg, state)
    if why:
        raise SystemExit(f"not started: {why}")
    if llama_running():
        raise SystemExit("not started: llama-server.exe is running")


def gate_server_check(cfg, state):
    """The gate file and every gate server setting (slots and cache included) must be the ones on record."""
    sha = L.file_sha(E.GATE_PATH)
    if sha != state.get("gate_sha"):
        raise SystemExit(f"data/eval/gate.jsonl is {sha}, the rounds were scored on {state.get('gate_sha')}")
    want, known = L.gate_server(cfg), state.get("gate_server") or {}
    diff = [k for k in want if known.get(k) != want[k]]
    if diff:
        raise SystemExit(f"gate server settings differ from the record: {L.describe_gate_server(want, diff)} now, "
                         f"{L.describe_gate_server(known, diff)} on record")
    return want


def toolchain_check(gate, state):
    """loop.py's check before a gate pass, read-only: sandbox storage closed, toolchains as on record, smoke
    programs."""
    L.sandbox.require()
    L.sandbox.ensure_private_folder_closed(log)
    from anneal import polyglot as P
    from anneal import polyglot_smoke as S
    fp = P.toolchain_fingerprint()
    if fp is None or fp != state.get("toolchains"):
        raise SystemExit(f"toolchains are {fp}, the rounds were judged with {state.get('toolchains')}")
    langs = sorted({it["lang"] for it in gate if it.get("kind") in L.GATE_KINDS})
    ex, tc, bad = P.SandboxExecutor(), P.Toolchain.windows(), []
    for lang in langs:
        for prog in S.programs(lang):
            if not (prog["name"] in ("sum64", "wrong") or (prog["name"] == "isolation" and lang == langs[0])):
                continue
            v = P.judge(lang, prog["code"], prog["cases"], ex, tc, case_timeout_s=prog.get("timeout", 10.0),
                        all_cases=True)
            if v.stage != prog["expect"]:
                bad.append(f"{lang} {prog['name']}: {v.stage} (want {prog['expect']})")
    if bad:
        raise SystemExit("toolchain smoke check failed: " + "; ".join(bad))
    return fp


# ============================================================================================ scale probe

def lora_scales(url):
    return {int(a["id"]): float(a["scale"]) for a in llama.http(url + "/lora-adapters", timeout=60)}


def set_scale(url, scale):
    llama.http(url + "/lora-adapters", [{"id": 0, "scale": scale}], timeout=120)
    got = lora_scales(url)
    if got != {0: float(scale)}:
        raise SystemExit(f"/lora-adapters reads {got} after setting adapter 0 to {scale}")


def numbers(x):
    """Every float in a JSON value, in order."""
    if isinstance(x, float):
        return [x]
    if isinstance(x, dict):
        return [v for k in x for v in numbers(x[k])]
    if isinstance(x, list):
        return [v for e in x for v in numbers(e)]
    return []


def probe_once(url):
    out = []
    for p in PROBES:
        r = llama.http(url + "/completion", {"prompt": p, "n_predict": PROBE_TOKENS, "temperature": 0.0, "seed": 1,
                                             "n_probs": 5, "cache_prompt": False}, timeout=900)
        steps = [{"tok": t.get("id", t.get("token", t.get("content"))), "nums": sorted(numbers(t), reverse=True)}
                 for t in (r.get("completion_probabilities") or []) if isinstance(t, dict)]
        out.append({"text_sha": hashlib.sha256((r.get("content") or "").encode("utf-8")).hexdigest()[:16],
                    "steps": steps})
    return out


def probe_diff(a, b):
    """The largest log-probability difference between two probes, over each prompt's tokens up to and including the
    first one where the two outputs part (the distributions there still come from the same text). inf when the
    numbers don't line up, or, if no probabilities came back, when the texts differ."""
    d = 0.0
    for x, y in zip(a, b):
        if not x["steps"] or not y["steps"]:
            if x["text_sha"] != y["text_sha"] or x["steps"] or y["steps"]:
                return math.inf
            continue
        for s, t in zip(x["steps"], y["steps"]):
            if len(s["nums"]) != len(t["nums"]):
                return math.inf
            d = max([d] + [abs(u - v) for u, v in zip(s["nums"], t["nums"])])
            if s["tok"] != t["tok"]:
                break
    return d


def scale_probe(url, scale):
    """Greedy probes at scales 1, 0, S, 1, S (that order); leaves the server at S. ok when each scale repeats and the
    three scales all differ."""
    if set(lora_scales(url)) != {0}:
        raise SystemExit(f"expected one LoRA adapter on the server, /lora-adapters reads {lora_scales(url)}")
    runs = {}
    for name, s in (("full", 1.0), ("zero", 0.0), ("part", scale), ("full2", 1.0), ("part2", scale)):
        set_scale(url, s)
        runs[name] = probe_once(url)
    res = {"repeat_full": probe_diff(runs["full"], runs["full2"]),
           "repeat_part": probe_diff(runs["part"], runs["part2"]),
           "full_vs_zero": probe_diff(runs["full"], runs["zero"]),
           "part_vs_full": probe_diff(runs["part"], runs["full"]),
           "part_vs_zero": probe_diff(runs["part"], runs["zero"])}
    noise = max(res["repeat_full"], res["repeat_part"])
    need = max(DIFF_MIN, 3 * noise)
    res["tokens_per_probe"] = [len(p["steps"]) for p in runs["full"]]
    res["ok"] = noise <= REPEAT_TOL and all(res[k] > need for k in ("full_vs_zero", "part_vs_full", "part_vs_zero"))
    res["need"] = need
    res["scale_now"] = lora_scales(url)
    return {k: ("different text" if v == math.inf else v) for k, v in res.items()}


# ============================================================================================ parts

def scaled_adapter(src, dst, scale):
    """A copy of the adapter with lora_alpha and alpha_pattern x scale (DoRA refused: its change is not linear in
    alpha). Never deletes anything; a finished copy is reused when its config and weights match."""
    ac = read_json(os.path.join(src, "adapter_config.json"))
    if ac.get("use_dora"):
        raise SystemExit("a DoRA adapter: scaling alpha does not scale its change")

    def sc(v):
        x = v * scale
        return int(x) if float(x).is_integer() else x

    new = dict(ac, lora_alpha=sc(ac["lora_alpha"]),
               alpha_pattern={k: sc(v) for k, v in (ac.get("alpha_pattern") or {}).items()})
    weights = [f for f in os.listdir(src) if f.endswith((".safetensors", ".bin"))]
    if not weights:
        raise SystemExit(f"no adapter weights in {src}")
    cfg_path = os.path.join(dst, "adapter_config.json")
    if not (os.path.exists(cfg_path) and read_json(cfg_path) == new
            and all(os.path.exists(os.path.join(dst, f)) and file_sha256(os.path.join(dst, f))
                    == file_sha256(os.path.join(src, f)) for f in weights)):
        if os.path.exists(cfg_path):
            os.replace(cfg_path, cfg_path + ".old")  # marks the copy unfinished while it is redone
        shutil.copytree(src, dst, dirs_exist_ok=True, ignore=shutil.ignore_patterns("adapter_config.json"))
        for f in weights:
            if file_sha256(os.path.join(dst, f)) != file_sha256(os.path.join(src, f)):
                raise SystemExit(f"copy of {f} does not match")
        write_json(cfg_path, new)
    return {"path": dst, "r": ac.get("r"), "lora_alpha": [ac["lora_alpha"], new["lora_alpha"]],
            "alpha_pattern": bool(ac.get("alpha_pattern")), "use_rslora": bool(ac.get("use_rslora"))}


def loss_part(cfg, state, info, out, scale):
    todo = []
    for kind, h in L.HELDOUT.items():
        sc = cfg.get(h["key"]) or {}
        if not sc:
            continue
        done = read_json(os.path.join(info["rdir"], "stages", f"{h['stage']}.done.json"))
        path = L.Loop.heldout_file(None, cfg, kind)
        if L.file_sha(path) != done["sha"]:
            raise SystemExit(f"{path} changed since the round was scored on it")
        if os.path.exists(os.path.join(out, f"loss_{kind}", "summary.json")):
            log(f"loss {kind}: done before, skipped")
            continue
        todo.append((kind, sc, path))
    if not todo:
        return
    require_gpu_free(cfg, state)
    adapter = scaled_adapter(info["adapter"], os.path.join(out, "adapter"), scale)
    write_json(os.path.join(out, "adapter_scale.json"), adapter)
    log(f"loss: adapter copy with lora_alpha {adapter['lora_alpha'][0]} -> {adapter['lora_alpha'][1]} "
        f"(r {adapter['r']})")
    for kind, sc, path in todo:
        run_dir = os.path.join(out, f"loss_{kind}")
        require_gpu_free(cfg, state)
        cmd = [PY, "-u", os.path.join(ROOT, "scripts", "val_loss.py"), "--adapter", adapter["path"], "--val-file", path,
               "--seq-len", str(int(sc.get("seq_len", 8192))), "--max-rows", str(int(sc.get("max_rows", 200))),
               "--no-base", "--run-dir", run_dir]
        os.makedirs(os.path.join(out, "logs"), exist_ok=True)
        t0 = time.time()
        log(f"loss {kind}: {subprocess.list2cmdline(cmd)}")
        with open(os.path.join(out, "logs", f"loss_{kind}.log"), "a", encoding="utf-8", errors="replace") as fh:
            rc = subprocess.run(cmd, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT,
                                env=dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")).returncode
        if rc != 0 or not os.path.exists(os.path.join(run_dir, "summary.json")):
            raise SystemExit(f"loss {kind}: val_loss.py exited with {rc} (log {out}\\logs\\loss_{kind}.log)")
        log(f"loss {kind}: done in {(time.time() - t0) / 60:.0f} min")


def gate_part(cfg, state, info, base, out, scale, n, port):
    gate = E.read_jsonl(E.GATE_PATH)
    seed = info["gate_seed"]
    picked = draw(gate, n)
    full_rows, base_rows = seed_rows(info["gate_rows"], seed), seed_rows(base["gate_rows"], seed)
    missing = [gate[i]["id"] for i in picked if gate[i]["id"] not in full_rows or gate[i]["id"] not in base_rows]
    if missing:
        raise SystemExit(f"{len(missing)} drawn problems have no seed-{seed} answer from the base or the round")
    write_json(os.path.join(out, "selection.json"), {"salt": SALT, "n": n, "ids": [gate[i]["id"] for i in picked]})
    gpath = os.path.join(out, f"gate.seed{seed}.jsonl")
    have = seed_rows(gpath, seed)
    if all(gate[i]["id"] in have for i in picked):
        log("gate: every drawn problem answered before")
        return
    # longest first by the base's answer length, so the last answers don't leave slots idle; seeds stay tied to the
    # problem's place in the gate (evalsets.score)
    order = sorted(picked, key=lambda i: (-(base_rows[gate[i]["id"]].get("completion_tokens") or 0), i))
    require_gpu_free(cfg, state)
    gate_server_check(cfg, state)
    toolchain_check(gate, state)
    g = cfg["gate"]
    os.makedirs(os.path.join(out, "logs"), exist_ok=True)
    srv = llama.Server(L.Loop.gate_flags(None, cfg, lora=info["gguf"]), os.path.join(out, "logs", "server.log"),
                       port=port, binary=llama.server_binary(L.gate_build(cfg)))
    log(f"llama-server: {srv.describe()}")
    with srv:
        log(f"llama-server up in {srv.load_s} s")
        probe = scale_probe(srv.url, scale)
        write_json(os.path.join(out, f"probe-{time.strftime('%Y%m%d-%H%M%S')}.json"), probe)
        log("scale probe: " + json.dumps(probe))
        if not probe["ok"]:
            raise SystemExit("scale probe failed: the adapter's scale could not be shown to change the answers; "
                             "nothing answered")
        stop = os.path.join(out, "STOP")

        def on_row(r, k, tot, tps):
            log(f"gate {k}/{tot}: {r['id']} pass {r['pass']} reasoning {r['reasoning_tokens']} finish "
                f"{r['finish_reason']} ({tps:.1f} tok/s)")

        try:
            E.score(srv.url, gate, gpath, seed, g["max_tokens"], L.Loop.gate_checker(None),
                    lambda u, m, s, mt: ew.chat_request(u, m, s, mt), ew.count_tokens, g["budget"],
                    concurrency=L.gate_concurrency(cfg), stop_check=lambda: os.path.exists(stop), log=log,
                    on_row=on_row, order=order)
        except E.Stopped:
            log("gate: stopped by STOP; answers so far kept")
        after = lora_scales(srv.url)
        write_json(os.path.join(out, "scale_after.json"), {"scales": {str(k): v for k, v in after.items()}})
        if after != {0: float(scale)}:
            raise SystemExit(f"the adapter's scale read {after} after answering, not {scale}")


# ============================================================================================ summary

def tokens(row, key):
    return row.get(key) or 0


def paired(ref, cand, ids):
    wins = sum(1 for i in ids if cand[i]["pass"] and not ref[i]["pass"])
    losses = sum(1 for i in ids if ref[i]["pass"] and not cand[i]["pass"])
    p = min(1.0, 2 * min(E.sign_test_greater(wins, losses), E.sign_test_greater(losses, wins)))
    d = [tokens(cand[i], "reasoning_tokens") - tokens(ref[i], "reasoning_tokens") for i in ids]
    return {"wins": wins, "losses": losses, "p_two_sided": round(p, 4), "thinks_longer_on": sum(1 for x in d if x > 0),
            "reasoning_diff_median": statistics.median(d) if d else None}


def stats(rows, ids):
    rt = [tokens(rows[i], "reasoning_tokens") for i in ids]
    ct = [tokens(rows[i], "completion_tokens") for i in ids]
    fin = {}
    for i in ids:
        fin[str(rows[i]["finish_reason"])] = fin.get(str(rows[i]["finish_reason"]), 0) + 1
    return {"pass": sum(1 for i in ids if rows[i]["pass"]),
            "reasoning_mean": round(statistics.mean(rt)) if rt else None,
            "reasoning_median": statistics.median(rt) if rt else None,
            "completion_mean": round(statistics.mean(ct)) if ct else None,
            "budget_hits": sum(1 for i in ids if rows[i].get("budget_hit")), "finish": fin}


def gate_summary(gate, picked, base, full, part):
    out = {}
    groups = {"all": picked, "write": [i for i in picked if gate[i]["kind"] == "write"],
              "fix": [i for i in picked if gate[i]["kind"] == "fix"]}
    for name, idx in groups.items():
        ids = [gate[i]["id"] for i in idx if gate[i]["id"] in part]
        g = {"n": len(ids), "drawn": len(idx)}
        if ids:
            g.update(base=stats(base, ids), full=stats(full, ids), part=stats(part, ids),
                     part_vs_full=paired(full, part, ids), part_vs_base=paired(base, part, ids))
            gap = g["base"]["reasoning_mean"] - g["full"]["reasoning_mean"]
            g["thinking_gap_closed"] = (round((g["part"]["reasoning_mean"] - g["full"]["reasoning_mean"]) / gap, 3)
                                        if gap > 0 else None)
            pgap = g["base"]["pass"] - g["full"]["pass"]
            g["pass_gap_closed"] = round((g["part"]["pass"] - g["full"]["pass"]) / pgap, 3) if pgap > 0 else None
        out[name] = g
    ids = [it["id"] for it in gate]
    out["whole_gate"] = {"n": len(ids), "base_pass": sum(1 for i in ids if i in base and base[i]["pass"]),
                         "full_pass": sum(1 for i in ids if i in full and full[i]["pass"])}
    return out


def loss_summary(base_loss, full_loss, part_loss, weights):
    keys = sorted(set(base_loss) & set(full_loss) & set(part_loss))
    b, f, p = ({k: d[k] for k in keys} for d in (base_loss, full_loss, part_loss))
    full_vs_base, part_vs_base = E.compare_loss(b, f, weights), E.compare_loss(b, p, weights)
    part_vs_full = E.compare_loss(f, p, weights)
    gain = full_vs_base["best"] - full_vs_base["candidate"] if keys else None
    return {"rows": len(keys), "base": full_vs_base["best"], "full": full_vs_base["candidate"],
            "part": part_vs_base["candidate"], "full_rel_to_base": full_vs_base["rel_diff"],
            "part_rel_to_base": part_vs_base["rel_diff"],
            "gain_kept": (round((part_vs_base["best"] - part_vs_base["candidate"]) / gain, 3) if gain else None),
            "part_vs_base": {k: part_vs_base[k] for k in ("wins", "losses", "p_one_sided")},
            "part_vs_full": {k: part_vs_full[k] for k in ("wins", "losses", "p_one_sided")}}


def summary_part(cfg, state, info, base, out, scale, n):
    res = {"round": info["rdir"][-4:], "scale": scale, "time": time.strftime("%Y-%m-%d %H:%M:%S")}
    gate = E.read_jsonl(E.GATE_PATH)
    seed = info["gate_seed"]
    gpath = os.path.join(out, f"gate.seed{seed}.jsonl")
    if os.path.exists(gpath):
        res["gate"] = gate_summary(gate, draw(gate, n), seed_rows(base["gate_rows"], seed),
                                   seed_rows(info["gate_rows"], seed), seed_rows(gpath, seed))
    res["loss"] = {}
    for kind, h in L.HELDOUT.items():
        ps = os.path.join(out, f"loss_{kind}", "summary.json")
        ref = (state.get("base") or {}).get(h["key"])
        if not os.path.exists(ps) or not ref:
            log(f"loss {kind}: " + ("not scored at this scale yet" if ref else "no base reference in state.json"))
            continue
        done = read_json(os.path.join(info["rdir"], "stages", f"{h['stage']}.done.json"))
        full = read_json(resolve(done["summary"]))
        bsum = read_json(resolve(ref["path"]))
        part = read_json(ps)
        res["loss"][kind] = loss_summary(
            {r["key"]: r[ref["field"]] for r in bsum["per_row"] if r.get(ref["field"]) is not None},
            {r["key"]: r["adapter_loss"] for r in full["per_row"]},
            {r["key"]: r["adapter_loss"] for r in part["per_row"]},
            {r["key"]: r["trained"] for r in full["per_row"]})
    write_json(os.path.join(out, "summary.json"), res)
    for name in ("all", "write", "fix"):
        g = (res.get("gate") or {}).get(name)
        if g and g["n"]:
            log(f"gate {name} ({g['n']} of {g['drawn']}): pass base {g['base']['pass']} / full {g['full']['pass']} / "
                f"x{scale} {g['part']['pass']}; reasoning mean {g['base']['reasoning_mean']} / "
                f"{g['full']['reasoning_mean']} / {g['part']['reasoning_mean']}; budget hits "
                f"{g['base']['budget_hits']} / {g['full']['budget_hits']} / {g['part']['budget_hits']}; thinking gap "
                f"closed {g['thinking_gap_closed']}, pass gap closed {g['pass_gap_closed']}; x{scale} vs full "
                f"{g['part_vs_full']['wins']} won {g['part_vs_full']['losses']} lost "
                f"(p {g['part_vs_full']['p_two_sided']})")
    for kind, s in res["loss"].items():
        log(f"loss {kind} ({s['rows']} rows): base {s['base']:.4f} / full {s['full']:.4f} "
            f"({100 * s['full_rel_to_base']:+.1f}%) / x{scale} {s['part']:.4f} ({100 * s['part_rel_to_base']:+.1f}%); "
            f"gain kept {s['gain_kept']}")
    return res


# ============================================================================================ main

def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--round", type=int, default=2)
    ap.add_argument("--scale", type=float, default=0.5)
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--part", choices=["all", "loss", "gate", "summary"], default="all")
    ap.add_argument("--port", type=int, default=PORT)
    ap.add_argument("--plan", action="store_true", help="check the inputs and print the draw; start nothing")
    args = ap.parse_args()
    if not 0 < args.scale < 1:
        raise SystemExit("--scale must be between 0 and 1")
    cfg = L.load_config(L.CONFIG)
    state = read_json(os.path.join(LOOP_DIR, "state.json"))
    base = state.get("base") or {}
    info = round_info(args.round)
    out = os.path.join(OUT_ROOT, f"r{args.round:03d}_x{args.scale:.2f}")
    gate = E.read_jsonl(E.GATE_PATH)
    picked = draw(gate, args.n)
    kinds, langs = {}, {}
    for i in picked:
        kinds[gate[i]["kind"]] = kinds.get(gate[i]["kind"], 0) + 1
        langs[gate[i]["lang"]] = langs.get(gate[i]["lang"], 0) + 1
    log(f"round {args.round} at scale {args.scale}: {len(picked)} of {len(gate)} gate problems drawn, kinds {kinds}, "
        f"languages {langs}; output {out}")
    if args.plan:
        log(f"draw sha {hashlib.sha256(json.dumps([gate[i]['id'] for i in picked]).encode()).hexdigest()[:16]}")
        log("missing: " + ("; ".join(info["missing"]) or "nothing"))
        log(f"base gate rows: {base.get('gate_rows')} seed {base.get('gate_seed')}")
        log(f"loop: {loop_would_hold(cfg, state) or 'not running and would hold'}; llama-server running: "
            f"{llama_running()}")
        return 0
    if info["missing"]:
        raise SystemExit("not ready: " + "; ".join(info["missing"]))
    if int(base.get("gate_seed", -1)) != info["gate_seed"]:
        raise SystemExit(f"the base's gate seed {base.get('gate_seed')} is not the round's {info['gate_seed']}")
    os.makedirs(out, exist_ok=True)
    if args.part in ("all", "loss"):
        loss_part(cfg, state, info, out, args.scale)
    if args.part in ("all", "gate"):
        gate_part(cfg, state, info, base, out, args.scale, args.n, args.port)
    summary_part(cfg, state, info, base, out, args.scale, args.n)
    log("LORA SCALE CHECK DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
