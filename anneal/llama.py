"""llama-server (llama.cpp b11435, Vulkan) for generation and scoring: one place for the binary, the base GGUF and
the flag sets, so every scored model runs on an identical server.

Scoring server (base and adapter alike; results/lora_testing_findings.md, results/mtp_findings.md):
  GGML_VK_DISABLE_COOPMAT=1 (llama-server dies on prompt batches over 8 tokens with a LoRA otherwise; set for the
  base too so only the adapter differs), -c 262144 -np 4 -ctk q8_0 -ctv q8_0 -fa on -ngl 999, --ctx-checkpoints 0
  and --cache-ram 0 (nothing rolls back; each checkpoint costs ~5 s here), --reasoning-format deepseek, and
  --lora <gguf> for an adapter. -np is the number of answers decoded side by side; the 262,144-token cache is split
  evenly between them. The first run and EvalPlus use 4; the loop's gate server takes its count from loop.yaml
  (gate.slots), and may name another llama.cpp build (gate.build: a folder under llama.cpp/) and MTP speculative
  decoding (gate.spec, spec_flags); EvalPlus always runs on b11435.
"""

import json
import os
import subprocess
import time
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_BUILD = "b11435-vulkan"  # the first run's and EvalPlus's build, and the loop's unless loop.yaml names another
SERVER = os.path.join(ROOT, "llama.cpp", DEFAULT_BUILD, "llama-server.exe")
OLLAMA_MANIFEST = os.path.expanduser(r"~\.ollama\models\manifests\registry.ollama.ai\library\qwen3.8\27b")
SNAPSHOT = os.path.join(ROOT, "models", "hf", "hub", "models--unsloth--Qwen3.8-27B-unsloth-bnb-4bit", "snapshots",
                        "8aa5f05d26b7205477066e1449e0af13f762a299")
LORA_BASE_CONFIG = os.path.join(ROOT, "models", "lora-base-config", "Qwen3.8-27B-unsloth-bnb-4bit-8aa5f05d")
MTP_GGUF = os.path.join(ROOT, "models", "unsloth-Qwen3.8-27B-GGUF-MTP", "mtp-Qwen3.8-27B-Q4_0.gguf")
CREATE_NO_WINDOW = 0x08000000


def ollama_gguf():
    """Ollama's qwen3.8:27b Q4_K_M model blob (the testing base)."""
    with open(OLLAMA_MANIFEST, encoding="utf-8") as fh:
        man = json.load(fh)
    digest = next(lay["digest"] for lay in man["layers"] if lay["mediaType"] == "application/vnd.ollama.image.model")
    models = OLLAMA_MANIFEST[:OLLAMA_MANIFEST.lower().index(os.sep + "manifests" + os.sep)]
    return os.path.join(models, "blobs", digest.replace(":", "-"))


SCORING_CTX = 262144  # the scoring server's whole KV cache, split evenly between its slots


def server_binary(build=DEFAULT_BUILD):
    """llama-server.exe of a llama.cpp build: its folder under llama.cpp/ (a plain folder name)."""
    if not isinstance(build, str) or not build or build in (".", "..") or os.path.basename(build) != build \
            or "/" in build or "\\" in build:
        raise ValueError(f"a llama.cpp build is a folder name under llama.cpp/, got {build!r}")
    return os.path.join(ROOT, "llama.cpp", build, "llama-server.exe")


def spec_flags(spec):
    """llama-server flags for speculative decoding: none for None, or {"type": "draft-mtp", "draft_n_max": N} for the
    MTP block built into the base GGUF (no separate draft file; results/mtp_findings.md). llama.cpp samples every
    position from the model being scored and keeps a draft token only if it matches, so answers come from the same
    distribution; the text for a given seed differs (at greedy, results/mtp_findings.md: only at near-ties)."""
    if not spec:
        return []
    if not isinstance(spec, dict) or spec.get("type") != "draft-mtp" or set(spec) - {"type", "draft_n_max"}:
        raise ValueError(f"gate.spec must be {{type: draft-mtp, draft_n_max: N}}, got {spec!r}")
    n = spec.get("draft_n_max", 1)
    if not isinstance(n, int) or isinstance(n, bool) or not 1 <= n <= 4:
        raise ValueError(f"gate.spec draft_n_max must be a whole number from 1 to 4, got {n!r}")
    return ["--spec-type", "draft-mtp", "--spec-draft-n-max", str(n)]


def scoring_flags(lora=None, extra=(), slots=4):
    flags = ["-c", str(SCORING_CTX), "-np", str(int(slots)), "-ctk", "q8_0", "-ctv", "q8_0", "-fa", "on", "-ngl", "999",
             "--ctx-checkpoints", "0", "--cache-ram", "0", "--reasoning-format", "deepseek"]
    if lora:
        flags += ["--lora", lora]
    return flags + list(extra)


def http(url, body=None, timeout=600):
    req = urllib.request.Request(url, data=None if body is None else json.dumps(body).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read() or b"{}")


class Server:
    """A llama-server process on 127.0.0.1:port with its own log file. Use as a context manager."""

    def __init__(self, flags, log_path, port=18090, coopmat=False, model=None, chat_template_file=None,
                 timeout_s=900, retries=1, binary=None):
        self.port, self.log_path, self.timeout_s, self.retries = port, log_path, timeout_s, retries
        self.url = f"http://127.0.0.1:{port}"
        self.cmd = [binary or SERVER, "-m", model or ollama_gguf(), "--host", "127.0.0.1", "--port", str(port),
                    "--no-webui", "-to", "21600"] + list(flags)
        if chat_template_file:
            self.cmd += ["--chat-template-file", chat_template_file]
        self.env = dict(os.environ)
        if not coopmat:
            self.env["GGML_VK_DISABLE_COOPMAT"] = "1"
        else:
            self.env.pop("GGML_VK_DISABLE_COOPMAT", None)
        self.proc = self.logf = None
        self.load_s = None

    def describe(self):
        return ("GGML_VK_DISABLE_COOPMAT=1 " if "GGML_VK_DISABLE_COOPMAT" in self.env else "") + \
            subprocess.list2cmdline(self.cmd)

    def _start_once(self):
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        self.logf = open(self.log_path, "a", encoding="utf-8", errors="replace")
        self.logf.write(f"\n==== {time.strftime('%Y-%m-%d %H:%M:%S')} {self.describe()}\n")
        self.logf.flush()
        self.proc = subprocess.Popen(self.cmd, stdout=self.logf, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                     env=self.env, creationflags=CREATE_NO_WINDOW)
        t0 = time.time()
        while time.time() - t0 < self.timeout_s:
            if self.proc.poll() is not None:
                break
            try:
                if http(self.url + "/health", timeout=5).get("status") == "ok":
                    self.load_s = round(time.time() - t0, 1)
                    return True
            except Exception:
                pass
            time.sleep(1)
        self.stop()
        return False

    def start(self):
        for attempt in range(1 + self.retries):
            if self._start_once():
                return self
            print(f"llama-server failed to start (attempt {attempt + 1}); log: {self.log_path}")
        raise RuntimeError(f"llama-server did not start: {self.describe()} (log {self.log_path})")

    def alive(self):
        return self.proc is not None and self.proc.poll() is None

    def stop(self):
        if self.proc is not None:
            if self.proc.poll() is None:
                self.proc.terminate()
                try:
                    self.proc.wait(60)
                except subprocess.TimeoutExpired:
                    self.proc.kill()
                    self.proc.wait(30)
            self.proc = None
        if self.logf is not None:
            self.logf.close()
            self.logf = None

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()
