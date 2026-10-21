"""Tests for scripts/glm_rollouts.py without the network: a small tar in the dataset's layout (run folders with
agent/trajectory.json and agent/episode-K/prompt.txt, response.txt, debug.json), its catalog and checksums, and a
local HTTP server that serves byte ranges and drops the first connection part-way, as a long download can.

  python tests/test_glm_rollouts.py        (or: python -m pytest tests/test_glm_rollouts.py)
"""
import hashlib
import io
import json
import os
import sys
import tarfile
import tempfile
import threading
import types
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import glm_rollouts as G  # noqa: E402

SHARD = "payload/s01/shards/trials-00001.tar"


def reply(analysis, keys, done=False):
    return json.dumps({"analysis": analysis, "plan": "Next step.", "commands": [{"keystrokes": keys, "duration": 1.0}],
                       "task_complete": done})


def run_files(prompts, replies, reasonings, system=None, break_context_at=None, debug_as_string=True):
    """The files of one run folder: debug.json logs the request as sent and the response as a JSON string."""
    files, sent = {}, ([{"role": "system", "content": system}] if system else [])
    steps = [{"step_id": 1, "source": "user", "message": prompts[0]}]
    for k, (p, r, t) in enumerate(zip(prompts, replies, reasonings)):
        sent = sent + [{"role": "user", "content": p}]
        req = [dict(m) for m in sent]
        if break_context_at == k:
            req[-2]["content"] = "a summary of the earlier steps"
        resp = {"id": f"r{k}", "choices": [{"index": 0, "message": {"role": "assistant", "content": r,
                                                                      "reasoning_content": t}}]}
        debug = {"model": "hosted_vllm/glm-5.3", "messages": req,
                 "original_response": json.dumps(resp) if debug_as_string else resp}
        files[f"agent/episode-{k}/prompt.txt"] = p.encode()
        files[f"agent/episode-{k}/response.txt"] = r.encode()
        files[f"agent/episode-{k}/debug.json"] = json.dumps(debug).encode()
        steps.append({"step_id": k + 2, "source": "agent", "model_name": "glm-5.3", "message": "Analysis: ...",
                      "reasoning_content": t, "tool_calls": [], "observation": {"results": []}})
        sent = sent + [{"role": "assistant", "content": r}]
    files["agent/trajectory.json"] = json.dumps({"schema_version": "ATIF-v1.7", "steps": steps}).encode()
    files["verifier/reward.txt"] = b"1"
    files["trial.log"] = b"x" * 3000
    return files


def make_dataset(d):
    """-> (raw dir, tar path, expected runs). Runs: two passing attempts of task t1 plus a third with more turns (cut
    by --per-task 2), one of t2 whose context was summarised after its first step, one failed run, one passing run in
    a shard nobody downloads."""
    a = run_files(["Task: make the build pass. Terminal:\n$ ", "New Terminal Output:\nok\n$ ",
                   "Previous response had parsing errors:\nno JSON\n\nPlease fix", "New Terminal Output:\ndone\n$ "],
                  [reply("Look first.", "ls -la\n"), "```json\n{\"analysis\": \"x\"}\n```",
                   reply("Build it.", "make -j4\n"), reply("Built.", "", done=True)],
                  ["I should list the files first.", "Now build.", "The JSON was wrong; resend.", "It passes."],
                  system="You solve terminal tasks in a Linux shell.")
    b = run_files(["Task: fix the cron job. Terminal:\n$ "], [reply("Edit crontab.", "crontab -l\n", done=True)],
                  ["Check the crontab."])
    c = run_files(["Task: count lines. Terminal:\n$ ", "New Terminal Output:\n$ ", "New Terminal Output:\n3\n$ "],
                  [reply("Use Python.", "python3 count.py data.txt\n"), reply("Again.", "wc -l data.txt\n"),
                   reply("Done.", "", done=True)],
                  ["A script is quickest.", "Check with wc.", "Matches."], break_context_at=1, debug_as_string=False)
    extra = run_files(["Task: slow way. Terminal:\n$ "] * 2, [reply("a", "ls\n"), reply("b", "", done=True)],
                      ["r1", "r2"])
    failed = run_files(["Task: broken. Terminal:\n$ "], [reply("a", "ls\n")], ["r"])
    runs = {"rts_a__r0_1": a, "rts_b__r0_2": b, "rts_c__r0_3": c, "rts_x__r0_4": extra, "rts_f__r0_5": failed}
    raw = os.path.join(d, "raw")
    tar_path = os.path.join(raw, *SHARD.split("/"))
    os.makedirs(os.path.dirname(tar_path))
    with tarfile.open(tar_path, "w") as tf:
        for pre, files in runs.items():
            for rel, data in files.items():
                ti = tarfile.TarInfo(f"{pre}/{rel}")
                ti.size = len(data)
                tf.addfile(ti, io.BytesIO(data))
    blob = open(tar_path, "rb").read()
    with open(os.path.join(raw, "checksums.sha256"), "w") as fh:
        fh.write(f"{hashlib.sha256(blob).hexdigest()}  {SHARD}\n{'0' * 64}  payload/s02/shards/trials-00002.tar\n")
    json.dump({"shards": [{"path": SHARD, "size": len(blob), "sha256": hashlib.sha256(blob).hexdigest()},
                          {"path": "payload/s02/shards/trials-00002.tar", "size": 10, "sha256": "0" * 64}]},
              open(os.path.join(raw, "shards.json"), "w"))
    import pyarrow as pa
    import pyarrow.parquet as pq

    def row(task, group, attempt, reward, pre, turns, out_tokens, summ=0, shard="s01/shards/trials-00001.tar",
            missing=None):
        return {"task_id": task, "task_group_id": group, "attempt": attempt, "reward": reward,
                "exception_type": None, "missing_artifacts": missing or [], "summarization_count": summ,
                "turn_count": turns, "output_tokens": out_tokens, "shard": shard, "member_prefix": pre,
                "execution_id": f"e-{pre}"}
    rows = [row("t1", "g1", 1, 1.0, "rts_a__r0_1", 4, 900), row("t1", "g1", 2, 1.0, "rts_x__r0_4", 9, 100),
            row("t1", "g1", 3, 1.0, "rts_b__r0_2", 1, 50), row("t2", "g2", 1, 1.0, "rts_c__r0_3", 3, 300),
            row("t3", "g3", 1, 0.0, "rts_f__r0_5", 1, 10), row("t4", "g4", 1, None, "rts_n__r0_6", 0, 0),
            row("t5", "g5", 1, 1.0, "rts_m__r0_7", 2, 20, missing=["agent/trajectory.json"]),
            row("t6", "g6", 2, 1.0, "rts_z__r0_8", 2, 20, shard="payload/s02/shards/trials-00002.tar")]
    pq.write_table(pa.Table.from_pylist(rows), os.path.join(raw, "logical_attempts.parquet"))
    return raw, tar_path, blob


def point_at(raw):
    G.RAW, G.SELECTION, G.PARTS = raw, os.path.join(raw, "selection.json"), os.path.join(raw, "parts")
    G.OUT, G.DONE = os.path.join(raw, "conversations.jsonl"), os.path.join(raw, "conversations.done.json")
    G.DOWNLOADS = os.path.join(raw, "downloads.json")
    G._SHARD_CACHE.clear()


def test_select_and_local_convert():
    with tempfile.TemporaryDirectory() as d:
        raw, tar_path, _ = make_dataset(d)
        point_at(raw)
        assert G.select(SimpleNamespace(per_task=2, revision="abc")) == 0
        sel = json.load(open(G.SELECTION))
        c = sel["counts"]
        assert (c["reward_1"], c["missing_agent_files"], c["tasks"], c["selected"], c["picked"], c["over_per_task"]) \
            == (6, 1, 3, 5, 4, 1), c
        assert sorted(sel["shards"][SHARD]) == ["rts_a__r0_1/", "rts_b__r0_2/", "rts_c__r0_3/", "rts_x__r0_4/"]
        x = sel["runs"]["rts_x__r0_4/"]  # t1's attempt with the most turns is kept but not picked
        assert (x["task_rank"], x["picked"]) == (3, False), x
        assert [sel["runs"][p]["task_rank"] for p in ("rts_b__r0_2/", "rts_a__r0_1/")] == [1, 2]
        assert G.convert(SimpleNamespace(local_tar=tar_path, parallel=1, max_shards=0)) == 0
        rows = [json.loads(x) for x in open(os.path.join(G.PARTS, G.part_name(SHARD) + ".jsonl"))]
        by = {r["task_id"] + ":" + str(r["attempt"]): r for r in rows}
        a = by["t1:1"]["messages"]
        assert [m["role"] for m in a] == ["system"] + ["user", "assistant"] * 4, [m["role"] for m in a]
        assert a[0]["content"].startswith("You solve terminal tasks")
        assert a[2]["reasoning_content"] == "I should list the files first." and "loss_mask" not in a[2]
        assert a[4]["loss_mask"] == 0  # the fenced reply Terminus-2 couldn't parse stays only as context
        assert a[6].get("loss_mask", 1) == 1 and a[8]["reasoning_content"] == "It passes."
        assert by["t1:1"]["uses_python"] is False and by["t1:1"]["task_group_id"] == "g1"
        assert (by["t1:1"]["picked"], by["t1:2"]["picked"], by["t1:2"]["task_rank"]) == (True, False, 3)
        cc = by["t2:1"]
        assert cc["steps"] == 1 and len(cc["messages"]) == 2, cc  # cut where the sent context stops matching
        assert cc["uses_python"] is True
        done = json.load(open(os.path.join(G.PARTS, G.part_name(SHARD) + ".done.json")))
        k = done["counts"]
        assert done["checked"] and done["from"] == "disk"
        assert (k["runs_written"], k["steps_written"], k["picked_runs_written"], k["picked_steps_written"],
                k["check_context_differs"], k["check_reasoning_match"], k["check_reply_match"]) == \
            (4, 8, 3, 6, 1, 8, 8), k
        assert not os.path.exists(G.DONE)  # the other shard isn't in yet


class RangeServer(BaseHTTPRequestHandler):
    blob, drops = b"", []

    def log_message(self, *a):
        pass

    def do_GET(self):
        start = 0
        rng = self.headers.get("Range")
        if rng:
            start = int(rng.split("=")[1].split("-")[0])
        body = self.blob[start:]
        self.send_response(206 if rng else 200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if self.drops:  # the first connection breaks part-way
            cut = self.drops.pop(0)
            self.wfile.write(body[:cut])
            self.wfile.flush()
            self.connection.shutdown(2)
            return
        self.wfile.write(body)


def serve(blob, drops):
    RangeServer.blob, RangeServer.drops = blob, list(drops)
    srv = ThreadingHTTPServer(("127.0.0.1", 0), RangeServer)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def test_stream_resumes_and_checks():
    with tempfile.TemporaryDirectory() as d:
        raw, tar_path, blob = make_dataset(d)
        point_at(raw)
        G.select(SimpleNamespace(per_task=2, revision="abc"))
        sel = json.load(open(G.SELECTION))
        sel["shards"] = {SHARD: sel["shards"][SHARD]}  # leave out the shard that only exists in the catalog
        json.dump(sel, open(G.SELECTION, "w"))
        os.remove(tar_path)  # not on disk, so it is streamed
        srv = serve(blob, [len(blob) // 3])
        old_url, old_sleep = G.shard_url, G.time.sleep
        G.shard_url = lambda path, rev: f"http://127.0.0.1:{srv.server_address[1]}/{path}"
        G.time.sleep = lambda s: None
        try:
            assert G.convert(SimpleNamespace(local_tar=None, parallel=2, max_shards=0)) == 0
            done = json.load(open(G.DONE))
            assert done["counts"]["runs_written"] == 4 and done["shards"] == 1, done
            part = json.load(open(os.path.join(G.PARTS, G.part_name(SHARD) + ".done.json")))
            assert part["from"] == "stream", part
            rows = [json.loads(x) for x in open(G.OUT)]
            assert sorted(r["task_id"] for r in rows) == ["t1", "t1", "t1", "t2"]
            # a second run finds every shard done and streams nothing
            assert G.convert(SimpleNamespace(local_tar=None, parallel=1, max_shards=0)) == 0
            # a shard whose bytes don't match the checksum writes nothing
            os.remove(os.path.join(G.PARTS, G.part_name(SHARD) + ".done.json"))
            os.remove(os.path.join(G.PARTS, G.part_name(SHARD) + ".jsonl"))
            RangeServer.blob = blob[:-1] + bytes([blob[-1] ^ 1])
            assert G.convert(SimpleNamespace(local_tar=None, parallel=1, max_shards=0)) == 1
            assert not os.path.exists(os.path.join(G.PARTS, G.part_name(SHARD) + ".jsonl"))
        finally:
            G.shard_url, G.time.sleep = old_url, old_sleep
            srv.shutdown()


def test_download_and_local_first():
    """download fetches every file of the listing into the raw folder, checks size and sha256, fetches a bad file
    again and skips what is already there; convert then reads shards from disk (never the network), and --local-only
    leaves a shard that isn't on disk for later."""
    with tempfile.TemporaryDirectory() as d:
        raw, tar_path, blob = make_dataset(d)
        point_at(raw)
        G.select(SimpleNamespace(per_task=2, revision="abc"))
        os.remove(tar_path)
        sha = hashlib.sha256(blob).hexdigest()
        other = b"second shard"
        task_file = b'{"task": 1}\n'
        hub = {SHARD: blob, "payload/s03/shards/trials-00003.tar": other, "tasks/a.jsonl": task_file}
        listing = [(SHARD, len(blob), sha), ("payload/s03/shards/trials-00003.tar", len(other),
                                             hashlib.sha256(other).hexdigest()), ("tasks/a.jsonl", len(task_file), None)]
        fetched, corrupt_once = [], {"payload/s03/shards/trials-00003.tar"}

        def fake_fetch(path, revision):
            fetched.append(path)
            dest = os.path.join(G.RAW, *path.split("/"))
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            data = hub[path]
            if path in corrupt_once:
                corrupt_once.discard(path)
                data = data[:-1] + b"X"
            with open(dest, "wb") as fh:
                fh.write(data)
            return dest

        saved = G.hub_files, G.hub_fetch, G.free_bytes, G.time.sleep, G.check_hub_version, G.shard_url
        G.hub_files, G.hub_fetch, G.time.sleep = (lambda rev: listing), fake_fetch, (lambda s: None)
        G.check_hub_version = lambda: None
        G.shard_url = lambda path, rev: (_ for _ in ()).throw(AssertionError("no network for a shard on disk"))
        try:
            args = SimpleNamespace(revision="abc", parallel=1, min_free_gb=100, floor_gb=50, retries=2)
            G.free_bytes = lambda p: 50e9  # 100 GB must stay free after everything: refuses, fetches nothing
            try:
                G.download(args)
                raise AssertionError("download started without enough free space")
            except SystemExit as e:
                assert "nothing downloaded" in str(e), e
            assert not fetched
            G.free_bytes = lambda p: 500e9
            assert G.download(args) == 0
            # the shard with selected runs and the small files first; s03 (no selected run) last, fetched twice
            assert fetched == [SHARD, "tasks/a.jsonl"] + ["payload/s03/shards/trials-00003.tar"] * 2, fetched
            state = json.load(open(G.DOWNLOADS))
            assert state["files"][SHARD]["checked"] and not state["files"]["tasks/a.jsonl"]["checked"], state
            fetched.clear()
            assert G.download(args) == 0 and not fetched  # all here already
            # convert: the shard is read from disk; t6's shard (s02) isn't there, so --local-only leaves it. A part
            # written for another selection (here: an older one without the unpicked run) is done again.
            os.makedirs(G.PARTS, exist_ok=True)
            G.write_json(os.path.join(G.PARTS, G.part_name(SHARD) + ".done.json"),
                         {"shard": SHARD, "counts": {"runs_written": 3},
                          "prefixes_sha": G.prefixes_sha(["rts_a__r0_1", "rts_b__r0_2", "rts_c__r0_3"])})
            assert G.convert(SimpleNamespace(local_tar=None, local_only=True, parallel=1, max_shards=0)) == 0
            part = json.load(open(os.path.join(G.PARTS, G.part_name(SHARD) + ".done.json")))
            assert part["from"] == "disk" and part["counts"]["runs_written"] == 4, part
            assert not os.path.exists(G.DONE)
            assert not os.path.exists(os.path.join(G.PARTS, G.part_name("payload/s02/shards/trials-00002.tar")
                                                   + ".done.json"))
        finally:
            G.hub_files, G.hub_fetch, G.free_bytes, G.time.sleep, G.check_hub_version, G.shard_url = saved


def test_reply_rules():
    assert G.parse_reply(reply("a", "ls\n")) is not None
    assert G.parse_reply("Sure! " + reply("a", "ls\n")) is None
    assert G.parse_reply(json.dumps({"analysis": "a", "plan": "p", "commands": [{"duration": 1}]})) is None
    assert G.PYTHON.search("cat > solve.py <<'EOF'") and G.PYTHON.search("/usr/bin/python3 -m venv v")
    assert G.PYTHON.search("pip install x") and not G.PYTHON.search("make pipeline && numpy-free build")


def test_inspect_runs():
    with tempfile.TemporaryDirectory() as d:
        raw, tar_path, _ = make_dataset(d)
        point_at(raw)
        out = io.StringIO()
        saved = sys.stdout
        sys.stdout = out
        try:
            assert G.inspect(SimpleNamespace(tar=tar_path, n=1)) == 0
        finally:
            sys.stdout = saved
        text = out.getvalue()
        assert "sha256 matches" in text and "5 reward-1 runs" in text and "episode-0/debug.json skeleton" in text, text
        assert "'runs_usable': 4" in text, text


def fake_transformers():
    """Tokenizer stand-in: 10 + content characters / 4 tokens; like Qwen3.8's template it renders reasoning only from
    reasoning_content, and here not at all, so only the cheap character check can call a row too long."""
    m = types.ModuleType("transformers")

    class Tok:
        def apply_chat_template(self, msgs, tokenize=True, add_generation_prompt=False, enable_thinking=True):
            assert not any("<think>" in (x.get("content") or "") for x in msgs), "reasoning left in the content"
            n = 10 + sum(len(x.get("content") or "") for x in msgs) // 4
            return list(range(n)) if tokenize else "t" * n

    m.AutoTokenizer = SimpleNamespace(from_pretrained=lambda name: Tok())
    return m


def test_public_source_on_runs():
    """Every step is a row; a reply Terminus-2 couldn't parse is context only; a long reasoning early in the run
    doesn't make the later steps count as too long (the template doesn't render it there)."""
    import public_source as ps
    files = run_files(["Task: tidy the logs. Terminal:\n$ ", "New Terminal Output:\n$ ",
                       "Previous response had parsing errors:\nno JSON", "New Terminal Output:\nok\n$ "],
                      [reply("Look.", "ls /var/log\n"), "not json", reply("Rotate.", "logrotate -f /etc/lr.conf\n"),
                       reply("Done.", "", done=True)],
                      ["y" * 40000, "Resend it as JSON.", "Rotate them now.", "All rotated."])
    msgs, info = G.build_run(files)
    assert info["reply_masked"] == 1 and info["steps"] == 4
    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(d, "conversations.jsonl")
        with open(src, "w") as fh:
            fh.write(json.dumps({"messages": msgs, "uses_python": False}) + "\n")
        out = os.path.join(d, "out")
        saved_argv, saved_tf = sys.argv, sys.modules.get("transformers")
        sys.modules["transformers"] = fake_transformers()
        sys.argv = ["public_source.py", "--dataset", "glm53", "--files", "*", "--from-local", src, "--turns", "each",
                    "--require-think", "--respect-loss-mask", "--where", "uses_python=false", "--tokenizer", "x",
                    "--max-tokens", "3000", "--val-fraction", "0", "--out", out]
        try:
            assert ps.main() == 0
        finally:
            sys.argv = saved_argv
            if saved_tf is None:
                sys.modules.pop("transformers", None)
            else:
                sys.modules["transformers"] = saved_tf
        c = json.load(open(os.path.join(out, "measure.json")))["counts"]
        # step 1 is too long for its own reasoning; steps 3 and 4 fit, the template drops step 1's reasoning there
        assert (c["candidate_rows"], c["loss_masked"], c["too_long"], c["kept"]) == (4, 1, 1, 2), c
        rows = [json.loads(x) for f in ("train.jsonl", "val.jsonl") for x in open(os.path.join(out, f))]
        assert sorted(len(r["messages"]) for r in rows) == [6, 8], [len(r["messages"]) for r in rows]


def main():
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS {name}")
        except Exception:
            import traceback
            failed += 1
            print(f"FAIL {name}")
            traceback.print_exc()
    print(f"{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
