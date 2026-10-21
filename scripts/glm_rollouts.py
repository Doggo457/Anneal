"""GLM-5.3 terminal runs -> conversations for public_source.py.

Source: open-athena/recursive-task-synthesis-glm-5.3-rollouts (CC-BY-4.0). These are Terminus-2 agent runs on
Linux terminal tasks (shell, builds, sysadmin, containers), each checked by the task's own tests (reward 1 = passed).
The teacher, GLM-5.3, scores 71.54 on Terminal-Bench 2.1 against Qwen3.8-27B's 58.43 (Vals AI, one harness for every
model; research/whole-job-multilang-data.md). Every agent step keeps the model's own reasoning (reasoning_content).

The whole dataset (110 shards, about 280 GB of tar files) is downloaded to data/raw/glm53 at one revision; every
file's size and sha256 are checked against the Hub's listing (and checksums.sha256) as it lands, and a file that fails
is fetched again. A shard that isn't on disk can still be streamed over HTTPS instead (Range resume, sha256 checked at
the end, nothing written for a shard that fails). Converting a shard from disk checks its sha256 again while reading.
Finished files and shards are recorded, so a stopped download or conversion carries on where it left off.

  python scripts/glm_rollouts.py select   --revision SHA [--per-task 2]  # catalog -> data/raw/glm53/selection.json
  python scripts/glm_rollouts.py download --revision SHA [--parallel 3] [--min-free-gb 100] [--floor-gb 50]  # all files
  python scripts/glm_rollouts.py inspect  --tar PATH [--n 2]             # structure and checks of a few runs
  python scripts/glm_rollouts.py convert  [--local-tar PATH] [--local-only] [--parallel 2] [--max-shards N]

Selection (from logical_attempts.parquet, the last infra retry of each attempt): every run with reward == 1 and no
missing agent files is converted. Per task they are ranked: no context summarisation, the fewest turns and the fewest
output tokens first; the first --per-task of each task get "picked": true (several passing runs of one task are close
copies, so the mix reads the picked ones; the rest stay on disk with their task_rank).

Conversation of one run, exactly as the model saw it: the prompt of each episode (agent/episode-K/prompt.txt) is a
user turn and its reply (response.txt, the raw JSON Terminus-2 parses) an assistant turn carrying that step's
reasoning (from debug.json's response, else the trajectory step). Where an episode's debug.json holds the request
messages, they are checked against the conversation built so far, and the run is cut before the first episode that
doesn't match (for example after Terminus-2 summarised the context), so every kept step has its real context. A reply
Terminus-2 could not parse cleanly (not a bare JSON object with analysis, plan and commands, or followed by a
"Previous response had ..." prompt) stays in the context but carries "loss_mask": 0, so public_source.py
--respect-loss-mask never trains on it.

Output: data/raw/glm53/conversations.jsonl (one run per line: messages, task_id, task_group_id, attempt, task_rank,
picked, uses_python, ...) and conversations.done.json once every selected shard is in. uses_python marks runs whose
commands run or write Python (21 Oct 2026: Python is the lowest priority).
"""

import argparse
import collections
import hashlib
import http.client
import io
import json
import os
import re
import shutil
import sys
import tarfile
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = "open-athena/recursive-task-synthesis-glm-5.3-rollouts"
RAW = os.path.join(ROOT, "data", "raw", "glm53")
SELECTION = os.path.join(RAW, "selection.json")
PARTS = os.path.join(RAW, "parts")
OUT = os.path.join(RAW, "conversations.jsonl")
DONE = os.path.join(RAW, "conversations.done.json")
CATALOG_COLS = ["task_id", "task_group_id", "attempt", "reward", "exception_type", "missing_artifacts",
                "summarization_count", "turn_count", "output_tokens", "shard", "member_prefix", "execution_id"]
EPISODE = re.compile(r"^agent/episode-(\d+)/(prompt\.txt|response\.txt|debug\.json)$")
NEEDED = re.compile(r"^agent/(trajectory\.json|episode-\d+/(prompt\.txt|response\.txt|debug\.json))$")
PYTHON = re.compile(r"(?:^|[^\w.-])(python[0-9.]*|pip[0-9.]*|pytest|ipython)(?=[^\w-]|$)|\w\.py\b")
FEEDBACK = "Previous response had"
DOWNLOADS = os.path.join(RAW, "downloads.json")
MIN_FREE_RAM = 3 << 30
LOCK = threading.Lock()


def log(msg):
    with LOCK:
        print(time.strftime("%H:%M:%S"), msg, flush=True)


def write_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=1, ensure_ascii=False)
    os.replace(tmp, path)


def ram_guard():
    """Wait while free RAM is under 3 GB (the standing rule for jobs beside training and scoring)."""
    try:
        import psutil
    except ImportError:
        return
    warned = False
    while psutil.virtual_memory().available < MIN_FREE_RAM:
        if not warned:
            log("free RAM under 3 GB: waiting")
            warned = True
        time.sleep(30)


def norm_prefix(p):
    p = (p or "").replace("\\", "/")
    while p.startswith("./"):
        p = p[2:]
    return p.rstrip("/") + "/"


def norm_name(n):
    n = n.replace("\\", "/")
    while n.startswith("./"):
        n = n[2:]
    return n


# ------------------------------------------------------------------------------------------------ catalog

def load_checksums(raw=None):
    """{tar path: (sha256, size or None)} from checksums.sha256 ("HEX  path" lines) and shards.json."""
    raw = raw or RAW
    out = {}
    p = os.path.join(raw, "checksums.sha256")
    if os.path.exists(p):
        for line in open(p, encoding="utf-8"):
            parts = line.strip().split(None, 1)
            if len(parts) == 2 and re.fullmatch(r"[0-9a-fA-F]{64}", parts[0]):
                out[norm_name(parts[1].lstrip("*"))] = (parts[0].lower(), None)
    p = os.path.join(raw, "shards.json")
    if os.path.exists(p):
        def walk(o, key=None):
            if isinstance(o, dict):
                path = next((v for k, v in o.items() if isinstance(v, str) and v.endswith(".tar")
                             and k in ("path", "file", "filename", "name", "shard", "tar", "relative_path")), None)
                if path is None and isinstance(key, str) and key.endswith(".tar"):
                    path = key
                sha = next((v for k, v in o.items() if k in ("sha256", "sha", "checksum") and isinstance(v, str)
                            and re.fullmatch(r"[0-9a-fA-F]{64}", v)), None)
                size = next((int(v) for k, v in o.items() if k in ("size", "bytes", "size_bytes", "num_bytes")
                             and isinstance(v, (int, float))), None)
                if path and (sha or size):
                    old = out.get(norm_name(path), (None, None))
                    out[norm_name(path)] = ((sha or old[0] or "").lower() or None, size or old[1])
                for k, v in o.items():
                    walk(v, k)
            elif isinstance(o, list):
                for v in o:
                    walk(v)
        walk(json.load(open(p, encoding="utf-8")))
    return out


_SHARD_CACHE = {}


def shard_path(shard, sums):
    """The dataset path of a catalog `shard` value (it may lack the payload/ folder or carry a URL prefix), or None
    if checksums.sha256 / shards.json don't name exactly one such tar."""
    s = norm_name(str(shard))
    if s in sums:
        return s
    if s not in _SHARD_CACHE:
        hits = [k for k in sums if k.endswith(".tar") and (k.endswith("/" + s) or s.endswith("/" + k))]
        _SHARD_CACHE[s] = hits[0] if len(hits) == 1 else None
    return _SHARD_CACHE[s]


def missing_agent_files(value):
    items = value if isinstance(value, (list, tuple)) else ([] if value is None else [value])
    return [str(x) for x in items if "agent" in str(x) or "trajectory" in str(x)]


def select(args):
    import pyarrow.parquet as pq
    rows = pq.read_table(os.path.join(RAW, "logical_attempts.parquet"), columns=CATALOG_COLS).to_pylist()
    sums = load_checksums()
    counts = collections.Counter(attempts=len(rows))
    by_task = collections.defaultdict(list)
    missing_names = collections.Counter()
    for r in rows:
        if r["reward"] is None or float(r["reward"]) != 1.0:
            continue
        counts["reward_1"] += 1
        miss = missing_agent_files(r["missing_artifacts"])
        if miss:
            counts["missing_agent_files"] += 1
            missing_names.update(miss)
            continue
        if not r["member_prefix"] or not r["shard"]:
            counts["no_location"] += 1
            continue
        r["shard_path"] = shard_path(r["shard"], sums)
        if r["shard_path"] is None:
            counts["shard_not_in_checksums"] += 1
            continue
        by_task[r["task_id"]].append(r)
    runs, shards = {}, collections.defaultdict(list)
    for task, rs in by_task.items():
        rs.sort(key=lambda r: ((r["summarization_count"] or 0) > 0, r["turn_count"] or 0, r["output_tokens"] or 0,
                               r["attempt"]))
        for rank, r in enumerate(rs, 1):
            sp = r["shard_path"]
            pre = norm_prefix(r["member_prefix"])
            runs[pre] = {k: r[k] for k in ("task_id", "task_group_id", "attempt", "execution_id",
                                           "summarization_count", "turn_count", "output_tokens")}
            runs[pre].update(shard=sp, task_rank=rank, picked=rank <= args.per_task)
            shards[sp].append(pre)
        counts["over_per_task"] += max(0, len(rs) - args.per_task)
    counts.update(tasks=len(by_task), selected=len(runs), picked=sum(1 for r in runs.values() if r["picked"]),
                  shards=len(shards),
                  picked_with_summarisation=sum(1 for r in runs.values()
                                                if r["picked"] and (r["summarization_count"] or 0) > 0))
    size = sum((sums[s][1] or 0) for s in shards)
    sel = {"repo": REPO, "revision": args.revision, "per_task": args.per_task, "counts": dict(counts),
           "missing_agent_files_top": missing_names.most_common(10), "stream_bytes": size,
           "shards": {s: sorted(p) for s, p in sorted(shards.items())}, "runs": runs}
    os.makedirs(RAW, exist_ok=True)
    write_json(SELECTION, sel)
    log(f"selected {len(runs)} runs ({counts['picked']} picked) of {len(by_task)} tasks in {len(shards)} shards "
        f"({size / 1e9:.1f} GB): {dict(counts)}")
    return 0


# ------------------------------------------------------------------------------------------------ one run

def _maybe_json(s):
    t = s.strip() if isinstance(s, str) else ""
    if t[:1] in "{[" and len(t) > 1:
        try:
            return json.loads(t)
        except ValueError:
            return None
    return None


def _walk(obj, depth=0):
    """Every dict and list inside obj, looking into strings that hold JSON too."""
    if depth > 8:
        return
    if isinstance(obj, str):
        inner = _maybe_json(obj)
        if inner is not None:
            yield from _walk(inner, depth + 1)
        return
    yield obj
    if isinstance(obj, dict):
        for v in obj.values():
            yield from _walk(v, depth + 1)
    elif isinstance(obj, list):
        for v in obj:
            yield from _walk(v, depth + 1)


def text_of(content):
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(c.get("text", "") if isinstance(c, dict) else str(c) for c in content)
    return str(content)


def debug_request(debug):
    """The longest list of chat messages (dicts with a role) in a debug.json: the request the model got."""
    best = None
    for o in _walk(debug):
        if isinstance(o, list) and o and all(isinstance(m, dict) and "role" in m for m in o):
            if any(m.get("role") == "user" for m in o) and (best is None or len(o) > len(best)):
                best = o
    return best


def debug_response(debug):
    """The model's reply message in a debug.json: choices[0].message (or .delta) of the response."""
    for o in _walk(debug):
        if isinstance(o, dict) and isinstance(o.get("choices"), list) and o["choices"]:
            c = o["choices"][0]
            if isinstance(c, dict):
                m = c.get("message") or c.get("delta")
                if isinstance(m, dict):
                    return m
    return None


def reasoning_of(m):
    if not isinstance(m, dict):
        return None
    for k in ("reasoning_content", "reasoning", "thinking"):
        v = m.get(k)
        if isinstance(v, str) and v.strip():
            return v
    return None


def parse_reply(text):
    """The Terminus-2 reply as a dict, or None unless the text is exactly one JSON object with analysis, plan and a
    list of commands (each with string keystrokes). Extra text or code fences around it draw a warning there."""
    try:
        obj = json.loads(text.strip())
    except ValueError:
        return None
    if not (isinstance(obj, dict) and "analysis" in obj and "plan" in obj and isinstance(obj.get("commands"), list)):
        return None
    if not all(isinstance(c, dict) and isinstance(c.get("keystrokes"), str) for c in obj["commands"]):
        return None
    return obj


def build_run(files):
    """files: {path inside the run folder: bytes} -> (messages, info). messages is [] when nothing usable is there."""
    info = collections.Counter()
    traj = {}
    if "agent/trajectory.json" in files:
        try:
            traj = json.loads(files["agent/trajectory.json"].decode("utf-8"))
        except ValueError:
            info["bad_trajectory"] += 1
    agent_steps = [s for s in (traj.get("steps") or []) if isinstance(s, dict) and s.get("source") == "agent"]
    eps = sorted({int(m.group(1)) for n in files for m in [EPISODE.match(n)] if m})
    answered = sum(1 for k in eps if f"agent/episode-{k}/response.txt" in files)
    # trajectory agent step K is episode K only when the counts agree
    steps_align = len(agent_steps) in (len(eps), answered)
    messages = []
    for k in range(len(eps)):
        if k != eps[k]:  # a gap in the episode numbers
            info["episode_gap"] += 1
            break
        pf, rf, df = (f"agent/episode-{k}/{x}" for x in ("prompt.txt", "response.txt", "debug.json"))
        if pf not in files or rf not in files:
            info["incomplete_episode"] += 1
            break
        prompt = files[pf].decode("utf-8", "replace")
        reply = files[rf].decode("utf-8", "replace")
        debug = None
        if df in files:
            try:
                debug = json.loads(files[df].decode("utf-8"))
            except ValueError:
                info["bad_debug"] += 1
        req = debug_request(debug) if debug is not None else None
        if req is not None:
            # the request = earlier turns + this prompt (a leading system message is kept as sent)
            sent = [{"role": m.get("role"), "content": text_of(m.get("content"))} for m in req]
            if k == 0 and len(sent) >= 1 and sent[0]["role"] == "system" and not messages:
                messages.append({"role": "system", "content": sent[0]["content"]})
            want = [{"role": m["role"], "content": m["content"]} for m in messages] + [{"role": "user", "content": prompt}]
            if sent == want:
                info["context_exact"] += 1
            elif [(m["role"], m["content"].strip()) for m in sent] == [(m["role"], m["content"].strip()) for m in want]:
                info["context_whitespace"] += 1
            else:
                info["context_differs"] += 1
                break
        else:
            info["context_unchecked"] += 1
        resp = debug_response(debug) if debug is not None else None
        r_debug = reasoning_of(resp)
        r_traj = reasoning_of(agent_steps[k]) if steps_align and k < len(agent_steps) else None
        if r_debug and r_traj:
            info["reasoning_match" if r_debug.strip() == r_traj.strip() else "reasoning_mismatch"] += 1
        if resp is not None and isinstance(resp.get("content"), str):
            info["reply_match" if resp["content"].strip() == reply.strip() else "reply_mismatch"] += 1
        reasoning = r_debug or r_traj
        nxt = files.get(f"agent/episode-{k + 1}/prompt.txt", b"").decode("utf-8", "replace")
        ok = parse_reply(reply) is not None and not nxt.lstrip().startswith(FEEDBACK)
        messages.append({"role": "user", "content": prompt})
        a = {"role": "assistant", "content": reply}
        if reasoning:
            a["reasoning_content"] = reasoning
        else:
            info["no_reasoning"] += 1
        if not ok:
            a["loss_mask"] = 0
            info["reply_masked"] += 1
        messages.append(a)
        info["steps"] += 1
    if not any(m["role"] == "assistant" for m in messages):
        return [], info
    keys = "".join(c["keystrokes"] for m in messages if m["role"] == "assistant"
                   for c in (parse_reply(m["content"]) or {}).get("commands", []))
    info["uses_python"] = int(bool(PYTHON.search(keys)))
    return messages, info


# ------------------------------------------------------------------------------------------------ streaming

class HashingReader(io.RawIOBase):
    """Bytes of a URL (resuming with a Range request after a dropped connection) or of a local file, hashed in order."""

    def __init__(self, url=None, path=None, size=None, retries=10, timeout=120):
        self.url, self.size, self.retries, self.timeout = url, size, retries, timeout
        self.fh = open(path, "rb") if path else None
        self.resp, self.pos, self.sha = None, 0, hashlib.sha256()

    def readable(self):
        return True

    def _open(self):
        headers = {"User-Agent": "anneal-glm-rollouts/1"}
        token = os.environ.get("HF_TOKEN")
        if token:
            headers["Authorization"] = f"Bearer {token}"
        if self.pos:
            headers["Range"] = f"bytes={self.pos}-"
        r = urllib.request.urlopen(urllib.request.Request(self.url, headers=headers), timeout=self.timeout)
        if self.pos and r.status != 206:
            r.close()
            raise OSError(f"server ignored the Range request (HTTP {r.status})")
        if self.size is None and not self.pos:
            n = r.headers.get("Content-Length")
            self.size = int(n) if n and n.isdigit() else None
        self.resp = r

    def readinto(self, b):
        if self.fh is not None:
            n = self.fh.readinto(b)
        else:
            if self.size is not None and self.pos >= self.size:
                return 0
            for attempt in range(self.retries + 1):
                try:
                    if self.resp is None:
                        self._open()
                    n = self.resp.readinto(b)
                    if not n and (self.size is None or self.pos < self.size):
                        if self.size is None:
                            break
                        raise OSError(f"connection closed at byte {self.pos} of {self.size}")
                    break
                except urllib.error.HTTPError as e:
                    if e.code < 500 and e.code != 429:
                        raise
                    err = e
                except (OSError, http.client.HTTPException) as e:
                    err = e
                if self.resp is not None:
                    try:
                        self.resp.close()
                    except Exception:
                        pass
                    self.resp = None
                if attempt == self.retries:
                    raise OSError(f"{self.url}: gave up at byte {self.pos}: {err}")
                log(f"stream error at byte {self.pos} ({err}); retrying")
                time.sleep(min(60, 2 ** attempt))
        if n:
            self.sha.update(memoryview(b)[:n])
            self.pos += n
        return n

    def close(self):
        for h in (self.fh, self.resp):
            if h is not None:
                try:
                    h.close()
                except Exception:
                    pass
        super().close()


def read_shard(reader, prefixes, keep_all=False):
    """Stream one tar. -> {prefix: {relative path: bytes}} for the wanted runs (only the files build_run reads,
    unless keep_all), plus {prefix: [(relative path, size)]} of every member, for inspect."""
    want = {norm_prefix(p) for p in prefixes}
    got, listing = collections.defaultdict(dict), collections.defaultdict(list)
    buf = io.BufferedReader(reader, 1 << 20)
    with tarfile.open(fileobj=buf, mode="r|") as tf:
        for m in tf:
            name = norm_name(m.name)
            top = name.split("/", 1)[0] + "/"
            pre = next((p for p in (top,) if p in want), None)
            if pre is None:
                pre = next((p for p in want if name.startswith(p)), None)
            if pre is None:
                continue
            rel = name[len(pre):]
            listing[pre].append((rel, m.size))
            if m.isfile() and (keep_all or NEEDED.match(rel)):
                ram_guard()
                f = tf.extractfile(m)
                got[pre][rel] = f.read() if f is not None else b""
    while buf.read(1 << 20):  # the rest of the stream (tar end blocks and padding), so the checksum covers it all
        pass
    return got, listing


def shard_url(path, revision):
    return f"https://huggingface.co/datasets/{REPO}/resolve/{revision}/{path}"


def part_name(path):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", path)


def local_copy(path):
    """The downloaded copy of a dataset file (its dataset path under data/raw/glm53), or None. huggingface_hub
    moves a file there only once it is complete."""
    p = os.path.join(RAW, *norm_name(path).split("/"))
    return p if os.path.isfile(p) else None


# ------------------------------------------------------------------------------------------------ download

def hub_files(revision):
    """[(path, size, sha256 or None)] of every file in the dataset at `revision`, from the Hub's own listing (LFS
    files carry their sha256)."""
    from huggingface_hub import HfApi
    out = []
    for f in HfApi().list_repo_tree(REPO, repo_type="dataset", revision=revision, recursive=True):
        size = getattr(f, "size", None)
        if size is None:  # a folder
            continue
        lfs = getattr(f, "lfs", None)
        sha = getattr(lfs, "sha256", None) if lfs is not None else None
        if sha is None and isinstance(lfs, dict):
            sha = lfs.get("sha256")
        out.append((norm_name(f.path), int(size), sha.lower() if sha else None))
    return out


def hub_fetch(path, revision):
    """Download one file into data/raw/glm53 (resuming a partial download) and return where it is."""
    from huggingface_hub import hf_hub_download
    return hf_hub_download(REPO, path, repo_type="dataset", revision=revision, local_dir=RAW)


def free_bytes(path):
    return shutil.disk_usage(path).free


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_hub_version():
    import huggingface_hub
    v = tuple(int(x) for x in re.findall(r"\d+", huggingface_hub.__version__)[:2])
    if v < (0, 23):  # older versions put local_dir files in the cache and link or copy them: twice the disk
        raise SystemExit(f"huggingface_hub {huggingface_hub.__version__} is too old for a direct local_dir download "
                         "(needs 0.23 or newer)")


def download(args):
    """Every file of the dataset at one revision into data/raw/glm53, checked as it lands. Skips files already
    recorded in downloads.json at the same size and sha256; refuses to start unless at least --min-free-gb stays
    free after the whole download, and never starts a file that would leave less than --floor-gb."""
    os.makedirs(RAW, exist_ok=True)
    check_hub_version()
    sums = load_checksums()
    files = hub_files(args.revision)
    state = json.load(open(DOWNLOADS, encoding="utf-8")) if os.path.exists(DOWNLOADS) else \
        {"repo": REPO, "revision": args.revision, "files": {}}
    if state.get("revision") != args.revision:
        raise SystemExit(f"{DOWNLOADS} is for revision {state.get('revision')}, not {args.revision}")
    todo, mismatched = [], []
    for path, size, sha in sorted(files):
        listed = sums.get(path, (None, None))
        if sha and listed[0] and sha != listed[0]:
            mismatched.append(path)
        want = sha or listed[0]
        rec = state["files"].get(path)
        local = local_copy(path)
        if rec and local and rec.get("bytes") == size and rec.get("sha256") == want and os.path.getsize(local) == size:
            continue
        todo.append((path, size, want))
    if mismatched:
        raise SystemExit(f"the Hub's sha256 and checksums.sha256 disagree for {mismatched[:5]} "
                         f"({len(mismatched)} files); stopping")
    # shards holding selected (test-passed) runs and the small catalog files first, the rest after
    selected = set(json.load(open(SELECTION, encoding="utf-8"))["shards"]) if os.path.exists(SELECTION) else set()
    todo.sort(key=lambda t: (t[0].startswith("payload/") and t[0] not in selected, t[0]))
    later = sum(size for path, size, _ in todo if path.startswith("payload/") and path not in selected)
    need = sum(size for _, size, _ in todo)
    free = free_bytes(RAW)
    log(f"{len(files)} files at {args.revision[:12]}: {len(files) - len(todo)} already here, {len(todo)} to fetch "
        f"({need / 1e9:.1f} GB, of which {later / 1e9:.1f} GB in shards with no selected run, fetched last); "
        f"{free / 1e9:.1f} GB free")
    if free - need < args.min_free_gb * 1e9:
        raise SystemExit(f"only {(free - need) / 1e9:.1f} GB would stay free (rule: at least {args.min_free_gb} GB); "
                         "nothing downloaded")
    done_bytes, t0, failed = [0], time.time(), []

    def fetch(item):
        path, size, want = item
        for attempt in range(args.retries + 1):
            if free_bytes(RAW) - size < args.floor_gb * 1e9:
                raise RuntimeError(f"under {args.floor_gb} GB would stay free; not starting {path}")
            try:
                got = hub_fetch(path, args.revision)
                if os.path.abspath(got) != os.path.abspath(os.path.join(RAW, *path.split("/"))):
                    raise RuntimeError(f"{path} landed at {got}, not in {RAW}")
                n = os.path.getsize(got)
                digest = file_sha256(got) if want else None
                if n != size or (want and digest != want):
                    os.remove(got)
                    raise ValueError(f"{path}: {n} bytes, sha256 {digest}; want {size} bytes, sha256 {want}")
                with LOCK:
                    state["files"][path] = {"bytes": n, "sha256": want, "checked": bool(want),
                                            "at": time.strftime("%Y-%m-%d %H:%M:%S")}
                    write_json(DOWNLOADS, state)
                    done_bytes[0] += n
                    rate = done_bytes[0] / max(1.0, time.time() - t0)
                    k = len(state["files"])
                log(f"{path}: {n / 1e9:.2f} GB, " + ("sha256 checked" if want else "no sha256 listed")
                    + f"; {k}/{len(files)} files, {done_bytes[0] / 1e9:.1f} of {need / 1e9:.1f} GB at "
                    f"{rate / 1e6:.1f} MB/s")
                return
            except Exception as e:  # network trouble or a bad file: fetch it again, a few times
                if attempt == args.retries:
                    raise
                log(f"{path}: {type(e).__name__}: {e}; trying again")
                time.sleep(min(600, 30 * 2 ** attempt))

    with ThreadPoolExecutor(max(1, args.parallel)) as ex:
        futs = {ex.submit(fetch, item): item[0] for item in todo}
        for f, path in futs.items():
            try:
                f.result()
            except Exception as e:
                failed.append(path)
                log(f"{path}: FAILED {type(e).__name__}: {e}")
    if failed:
        log(f"{len(failed)} files failed; run again to retry them")
        return 1
    log(f"all {len(files)} files here and checked ({sum(1 for r in state['files'].values() if r['checked'])} "
        f"with a sha256)")
    return 0


def prefixes_sha(prefixes):
    return hashlib.sha1("\n".join(sorted(norm_prefix(p) for p in prefixes)).encode("utf-8")).hexdigest()


def part_done(path, prefixes):
    """True if the shard's part was written for exactly these runs (a new selection redoes it)."""
    f = os.path.join(PARTS, part_name(path) + ".done.json")
    try:
        return json.load(open(f, encoding="utf-8")).get("prefixes_sha") == prefixes_sha(prefixes)
    except (OSError, ValueError):
        return False


def convert_shard(path, sel, sums, local_tar=None):
    prefixes = sel["shards"][path]
    sha, size = sums.get(path, (None, None))
    reader = HashingReader(path=local_tar) if local_tar else HashingReader(url=shard_url(path, sel["revision"]),
                                                                          size=size)
    t0 = time.time()
    try:
        got, _ = read_shard(reader, prefixes)
    finally:
        reader.close()
    digest = reader.sha.hexdigest()
    if sha and digest != sha:
        raise RuntimeError(f"{path}: sha256 {digest} != {sha} (nothing written for this shard)")
    counts = collections.Counter(runs_wanted=len(prefixes), runs_found=len(got))
    base = os.path.join(PARTS, part_name(path))
    with open(base + ".jsonl.tmp", "w", encoding="utf-8") as fh:
        for pre in prefixes:
            files = got.get(norm_prefix(pre))
            if not files:
                counts["run_not_in_shard"] += 1
                continue
            msgs, info = build_run(files)
            counts.update({f"check_{k}": v for k, v in info.items() if k != "uses_python"})
            if not msgs:
                counts["run_unusable"] += 1
                continue
            meta = sel["runs"][norm_prefix(pre)]
            row = {"messages": msgs, "task_id": meta["task_id"], "task_group_id": meta["task_group_id"],
                   "attempt": meta["attempt"], "execution_id": meta["execution_id"], "shard": path,
                   "task_rank": meta.get("task_rank", 1), "picked": bool(meta.get("picked", True)),
                   "uses_python": bool(info.get("uses_python")), "steps": info.get("steps", 0),
                   "summarization_count": meta["summarization_count"]}
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            counts["runs_written"] += 1
            counts["steps_written"] += info.get("steps", 0)
            counts["uses_python"] += int(bool(info.get("uses_python")))
            if row["picked"]:
                counts["picked_runs_written"] += 1
                counts["picked_steps_written"] += info.get("steps", 0)
    os.replace(base + ".jsonl.tmp", base + ".jsonl")
    done = {"shard": path, "sha256": digest, "bytes": reader.pos, "checked": bool(sha), "from": "disk" if local_tar
            else "stream", "prefixes_sha": prefixes_sha(prefixes), "seconds": round(time.time() - t0),
            "counts": dict(counts)}
    write_json(base + ".done.json", done)
    log(f"{path}: {counts['runs_written']}/{len(prefixes)} runs, {counts['steps_written']} steps, "
        f"{reader.pos / 1e9:.2f} GB in {done['seconds']} s" + ("" if sha else " (no checksum to compare)"))
    return done


def convert(args):
    sel = json.load(open(SELECTION, encoding="utf-8"))
    sums = load_checksums()
    os.makedirs(PARTS, exist_ok=True)
    if args.local_tar:
        name = norm_name(os.path.relpath(args.local_tar, RAW)) if os.path.abspath(args.local_tar).startswith(
            os.path.abspath(RAW)) else os.path.basename(args.local_tar)
        path = shard_path(name, sums)
        if path is None or path not in sel["shards"]:
            raise SystemExit(f"{path}: no selected runs in it")
        convert_shard(path, sel, sums, local_tar=args.local_tar)
        return 0
    todo = [p for p in sel["shards"] if not part_done(p, sel["shards"][p])]
    on_disk = {p: local_copy(p) for p in todo}
    if getattr(args, "local_only", False):
        waiting = [p for p in todo if not on_disk[p]]
        todo = [p for p in todo if on_disk[p]]
        if waiting:
            log(f"--local-only: {len(waiting)} shards aren't on disk yet; they wait for the next run")
    if args.max_shards:
        todo = todo[:args.max_shards]
    n_disk = sum(1 for p in todo if on_disk[p])
    log(f"{len(sel['shards']) - len(todo)} of {len(sel['shards'])} shards done or waiting; {len(todo)} to convert "
        f"({n_disk} from disk, {len(todo) - n_disk} streamed)")
    failed = []
    with ThreadPoolExecutor(max(1, args.parallel)) as ex:
        futs = {ex.submit(convert_shard, p, sel, sums, on_disk[p]): p for p in todo}
        for f, p in futs.items():
            try:
                f.result()
            except Exception as e:  # one bad shard must not stop the others; a rerun retries it
                failed.append(p)
                log(f"{p}: FAILED {type(e).__name__}: {e}")
    if failed:
        log(f"{len(failed)} shards failed; run again to retry them")
        return 1
    if any(not part_done(p, sel["shards"][p]) for p in sel["shards"]):
        return 0  # --max-shards: not all in yet
    totals, h = collections.Counter(), hashlib.sha256()
    with open(OUT + ".tmp", "wb") as out:
        for p in sorted(sel["shards"]):
            base = os.path.join(PARTS, part_name(p))
            totals.update(json.load(open(base + ".done.json", encoding="utf-8"))["counts"])
            with open(base + ".jsonl", "rb") as fh:
                for chunk in iter(lambda: fh.read(1 << 20), b""):
                    out.write(chunk)
                    h.update(chunk)
    os.replace(OUT + ".tmp", OUT)
    write_json(DONE, {"repo": REPO, "revision": sel["revision"], "per_task": sel["per_task"],
                      "shards": len(sel["shards"]), "sha256": h.hexdigest(), "counts": dict(totals),
                      "finished": time.strftime("%Y-%m-%d %H:%M:%S")})
    log(f"all shards in: {totals['runs_written']} runs, {totals['steps_written']} steps -> {OUT}")
    return 0


# ------------------------------------------------------------------------------------------------ inspect

def skeleton(obj, depth=0, max_depth=5, out=None, key="", limit=60):
    out = [] if out is None else out
    if len(out) >= limit:
        return out
    pad = "  " * depth
    if isinstance(obj, dict):
        out.append(f"{pad}{key}dict[{len(obj)}]")
        if depth < max_depth:
            for k, v in obj.items():
                skeleton(v, depth + 1, max_depth, out, f"{k}: ", limit)
    elif isinstance(obj, list):
        out.append(f"{pad}{key}list[{len(obj)}]")
        if obj and depth < max_depth:
            skeleton(obj[0], depth + 1, max_depth, out, "[0]: ", limit)
            if len(obj) > 1:
                skeleton(obj[-1], depth + 1, max_depth, out, "[-1]: ", limit)
    elif isinstance(obj, str):
        inner = _maybe_json(obj)
        if inner is not None and depth < max_depth:
            out.append(f"{pad}{key}str({len(obj)}) holding JSON:")
            skeleton(inner, depth + 1, max_depth, out, "", limit)
        else:
            out.append(f"{pad}{key}str({len(obj)}) {safe(obj[:70])!r}")
    else:
        out.append(f"{pad}{key}{type(obj).__name__} {obj!r}"[:120])
    return out


def safe(s):
    return s.replace("<", "[").replace(">", "]")


def inspect(args):
    import pyarrow.parquet as pq
    sums = load_checksums()
    name = norm_name(os.path.relpath(args.tar, RAW)) if os.path.abspath(args.tar).startswith(os.path.abspath(RAW)) \
        else os.path.basename(args.tar)
    path = shard_path(name, sums)
    if path is None:
        raise SystemExit(f"{name}: not in checksums.sha256 / shards.json")
    rows = pq.read_table(os.path.join(RAW, "logical_attempts.parquet"), columns=CATALOG_COLS).to_pylist()
    here = [r for r in rows if r["reward"] == 1.0 and r["shard"] and shard_path(r["shard"], sums) == path]
    print(f"{path}: {len(here)} reward-1 runs in the catalog")
    reader = HashingReader(path=args.tar)
    got, listing = read_shard(reader, [r["member_prefix"] for r in here], keep_all=True)
    reader.close()
    sha = sums.get(path, (None,))[0]
    print(f"sha256 {'matches' if reader.sha.hexdigest() == sha else 'DIFFERS' if sha else 'not listed'}; "
          f"{reader.pos} bytes")
    totals = collections.Counter()
    for i, r in enumerate(here):
        pre = norm_prefix(r["member_prefix"])
        files = got.get(pre, {})
        msgs, info = build_run(files)
        totals.update(info)
        totals["runs"] += 1
        totals["runs_usable"] += bool(msgs)
        if i >= args.n:
            continue
        print(f"\n=== run {pre} attempt {r['attempt']} turns {r['turn_count']} summarisations "
              f"{r['summarization_count']} output tokens {r['output_tokens']}")
        for rel, size in listing.get(pre, []):
            print(f"  {size:>9}  {rel}")
        print("checks:", dict(info))
        eps = sorted({int(m.group(1)) for n in files for m in [EPISODE.match(n)] if m})
        for k in sorted({eps[0], eps[-1]} if eps else set()):
            d = files.get(f"agent/episode-{k}/debug.json")
            if d is not None:
                print(f"-- episode-{k}/debug.json skeleton")
                try:
                    print("\n".join(skeleton(json.loads(d.decode("utf-8")))))
                except ValueError as e:
                    print("  not JSON:", e)
            for x in ("prompt.txt", "response.txt"):
                t = files.get(f"agent/episode-{k}/{x}", b"").decode("utf-8", "replace")
                print(f"-- episode-{k}/{x} ({len(t)} chars): {safe(t[:300])!r}")
        traj = files.get("agent/trajectory.json")
        if traj:
            t = json.loads(traj.decode("utf-8"))
            agent = [s for s in t.get("steps", []) if s.get("source") == "agent"]
            print(f"-- trajectory: {len(t.get('steps', []))} steps, {len(agent)} agent; first agent step skeleton:")
            if agent:
                print("\n".join(skeleton(agent[0], max_depth=4)))
        for j, m in enumerate(msgs[:6]):
            print(f"-- message {j} {m['role']} ({len(m['content'])} chars, reasoning "
                  f"{len(m.get('reasoning_content') or '')}, loss_mask {m.get('loss_mask', 1)}): "
                  f"{safe(m['content'][:160])!r}")
    print("\nall reward-1 runs in this shard:", dict(totals))
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("select")
    s.add_argument("--per-task", type=int, default=2)
    s.add_argument("--revision", required=True, help="full commit sha the catalog files were downloaded at")
    i = sub.add_parser("inspect")
    i.add_argument("--tar", required=True)
    i.add_argument("--n", type=int, default=2)
    c = sub.add_parser("convert")
    c.add_argument("--local-tar", default=None, help="convert the selected runs of one downloaded shard only")
    c.add_argument("--local-only", action="store_true", help="convert only shards already on disk (no streaming)")
    c.add_argument("--parallel", type=int, default=2)
    c.add_argument("--max-shards", type=int, default=0)
    d = sub.add_parser("download")
    d.add_argument("--revision", required=True, help="full commit sha (the one select used)")
    d.add_argument("--parallel", type=int, default=3)
    d.add_argument("--min-free-gb", type=float, default=100, help="free space that must remain after everything")
    d.add_argument("--floor-gb", type=float, default=50, help="never start a file that would leave less than this")
    d.add_argument("--retries", type=int, default=8, help="per file; waits 30 s doubling to 10 min between tries")
    args = ap.parse_args()
    return {"select": select, "inspect": inspect, "convert": convert, "download": download}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
