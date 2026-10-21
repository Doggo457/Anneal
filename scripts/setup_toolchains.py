"""Install the compilers and runtimes the six-language gate uses, under sandbox\\toolchains (no admin rights).

  python scripts/setup_toolchains.py            download, verify, install what is missing, grant the containers
  python scripts/setup_toolchains.py --check    build and run the smoke programs (anneal/polyglot_smoke.py) in the
                                                sandbox, one language after another; writes results/toolchains/
  python scripts/setup_toolchains.py --only go  one toolchain (repeatable)

Every archive is checked against the publisher's own checksum before anything is unpacked: Node.js SHASUMS256.txt,
the Rust dist .sha256 file and the npm registry's sha512 (all three also pinned here), the go.dev release list, the
Adoptium API, and the GitHub release asset digest for WinLibs GCC. Versions: Node.js 24.22.0, Rust 1.99.1
(x86_64-pc-windows-gnu: rustc, its standard library and its own MinGW linker, no Visual Studio), TypeScript 6.0.3, the
newest Go 1.27.x, the newest Temurin JDK 25 and the newest WinLibs GCC (POSIX threads, SEH, UCRT) release. What was
installed, with every URL and checksum, goes to sandbox\\toolchains\\lock.json and results/toolchains/lock.json.

The toolchains run outside the sandbox only here, once, for "--version"; after that every compiler runs in the build
container and every program in the code sandbox (anneal/sandbox.py, anneal/polyglot.py).
"""

import argparse
import base64
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.request
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from anneal import polyglot  # noqa: E402

TC = os.path.join(ROOT, "sandbox", "toolchains")
DOWNLOADS = os.path.join(ROOT, "sandbox", "downloads")
RESULTS = os.path.join(ROOT, "results", "toolchains")
UA = {"User-Agent": "anneal-setup-toolchains"}

NODE = ("24.22.0", "f9683a19877e12a9b167305c8db71040d812876359de21f0d2e97c3f321f80c3")
RUST = ("1.99.1", "0210624b2eb2bdf4ec32c2a6caf5c3bd2ab84e5df66c7530023e37f02add1084")
RUST_COMPONENTS = ("rustc", "rust-std-x86_64-pc-windows-gnu", "rust-mingw")
TYPESCRIPT = ("6.0.3", "sha512-y2TvuxSZPDyQakkFRPZHKFm+KKVqIisdg9/CZwm9ftvKXLP8NRWj38/ODjNbr43SsoXqNuAisEf1GdCxqWcdBw==")
GO_MINOR = "1.27"
JDK_FEATURE = 25
WINLIBS_REPO = "brechtsanders/winlibs_mingw"
WINLIBS_ASSET = re.compile(r"^winlibs-x86_64-posix-seh-gcc-(\d+\.\d+\.\d+)-mingw-w64ucrt-([\d.]+)-r(\d+)\.zip$")


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def get(url, timeout=120):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def get_json(url):
    return json.loads(get(url).decode("utf-8"))


def download(url, name, sha256=None, sha512_b64=None):
    """url -> sandbox\\downloads\\name, checked against the given checksum (sha256 hex or sha512 base64). An existing
    file that matches is reused. Returns (path, sha256 hex)."""
    os.makedirs(DOWNLOADS, exist_ok=True)
    path = os.path.join(DOWNLOADS, name)

    def digests(p):
        h256, h512 = hashlib.sha256(), hashlib.sha512()
        with open(p, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h256.update(chunk)
                h512.update(chunk)
        return h256.hexdigest(), base64.b64encode(h512.digest()).decode("ascii")

    def ok(p):
        d256, d512 = digests(p)
        return (sha256 is None or d256 == sha256.lower()) and (sha512_b64 is None or d512 == sha512_b64), d256

    if os.path.exists(path):
        good, d = ok(path)
        if good:
            log(f"{name}: already downloaded, checksum matches")
            return path, d
        os.remove(path)
    for attempt in range(4):
        try:
            log(f"downloading {url}")
            tmp = path + ".part"
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=300) as r, open(tmp, "wb") as fh:
                shutil.copyfileobj(r, fh, 1 << 20)
            os.replace(tmp, path)
            break
        except OSError as exc:
            if attempt == 3:
                raise
            log(f"  download failed ({exc}); retrying")
            time.sleep(2 ** (attempt + 1))
    good, d = ok(path)
    if not good:
        os.remove(path)
        raise SystemExit(f"{name}: checksum mismatch (got sha256 {d}); the file was deleted")
    log(f"{name}: {os.path.getsize(path):,} bytes, checksum verified")
    return path, d


def unpack_zip(path, dest, top=None):
    """Unpack a zip into dest; with top, the zip's single top folder becomes dest."""
    stage = dest + ".stage"
    shutil.rmtree(stage, ignore_errors=True)
    with zipfile.ZipFile(path) as z:
        for m in z.infolist():
            name = m.filename.replace("\\", "/")
            if name.startswith("/") or ".." in name.split("/"):
                raise SystemExit(f"{path}: unsafe member {m.filename!r}")
        z.extractall(stage)
    src = os.path.join(stage, top) if top else stage
    if not os.path.isdir(src):
        raise SystemExit(f"{path}: expected top folder {top!r}, found {os.listdir(stage)}")
    shutil.rmtree(dest, ignore_errors=True)
    os.replace(src, dest)
    shutil.rmtree(stage, ignore_errors=True)


def single_top(path):
    with zipfile.ZipFile(path) as z:
        tops = {n.replace("\\", "/").split("/")[0] for n in z.namelist()}
    if len(tops) != 1:
        raise SystemExit(f"{path}: expected one top folder, found {sorted(tops)[:5]}")
    return tops.pop()


# ============================================================================================ each toolchain

def install_node():
    ver, sha = NODE
    name = f"node-v{ver}-win-x64.zip"
    sums = get(f"https://nodejs.org/dist/v{ver}/SHASUMS256.txt").decode("utf-8")
    published = next((ln.split()[0] for ln in sums.splitlines() if ln.strip().endswith(" " + name)), None)
    if published != sha:
        raise SystemExit(f"Node.js {ver}: SHASUMS256.txt gives {published}, pinned {sha}")
    url = f"https://nodejs.org/dist/v{ver}/{name}"
    path, d = download(url, name, sha256=sha)
    unpack_zip(path, os.path.join(TC, "node"), top=single_top(path))
    return {"version": ver, "url": url, "sha256": d, "checksum_source": "SHASUMS256.txt + pinned"}


def install_typescript():
    ver, integrity = TYPESCRIPT
    meta = get_json(f"https://registry.npmjs.org/typescript/{ver}")
    if meta["dist"]["integrity"] != integrity:
        raise SystemExit(f"TypeScript {ver}: registry integrity {meta['dist']['integrity']}, pinned {integrity}")
    url = meta["dist"]["tarball"]
    path, d = download(url, f"typescript-{ver}.tgz", sha512_b64=integrity.split("-", 1)[1])
    dest = os.path.join(TC, "typescript")
    stage = dest + ".stage"
    shutil.rmtree(stage, ignore_errors=True)
    with tarfile.open(path, "r:gz") as t:
        t.extractall(stage, filter="data")
    shutil.rmtree(dest, ignore_errors=True)
    os.replace(os.path.join(stage, "package"), dest)
    shutil.rmtree(stage, ignore_errors=True)
    return {"version": ver, "url": url, "sha256": d, "sha512": integrity, "checksum_source": "npm registry + pinned"}


def install_rust():
    ver, sha = RUST
    base = f"rust-{ver}-x86_64-pc-windows-gnu"
    url = f"https://static.rust-lang.org/dist/{base}.tar.xz"
    published = get(url + ".sha256").decode("utf-8").split()[0]
    if published != sha:
        raise SystemExit(f"Rust {ver}: {url}.sha256 gives {published}, pinned {sha}")
    path, d = download(url, base + ".tar.xz", sha256=sha)
    dest = os.path.join(TC, "rust")
    stage = dest + ".stage"
    shutil.rmtree(stage, ignore_errors=True)
    os.makedirs(stage)
    wanted = tuple(f"{base}/{c}/" for c in RUST_COMPONENTS)
    n = 0
    with tarfile.open(path, "r:xz") as t:
        for m in t:
            if m.name.startswith(wanted) and not m.name.endswith("/manifest.in") and (m.isfile() or m.isdir()):
                # merge the components into one prefix, as their install.sh would
                comp_rel = m.name.split("/", 2)[2] if m.name.count("/") >= 2 else ""
                if not comp_rel or comp_rel.startswith("/") or ".." in comp_rel.split("/"):
                    continue
                if m.isfile():
                    with t.extractfile(m) as src:
                        out = os.path.join(stage, *comp_rel.split("/"))
                        os.makedirs(os.path.dirname(out), exist_ok=True)
                        with open(out, "wb") as fh:
                            shutil.copyfileobj(src, fh, 1 << 20)
                    n += 1
                else:
                    os.makedirs(os.path.join(stage, *comp_rel.split("/")), exist_ok=True)
    if not os.path.isfile(os.path.join(stage, "bin", "rustc.exe")):
        raise SystemExit(f"{path}: no bin/rustc.exe after merging {RUST_COMPONENTS}")
    shutil.rmtree(dest, ignore_errors=True)
    os.replace(stage, dest)
    return {"version": ver, "url": url, "sha256": d, "components": list(RUST_COMPONENTS), "files": n,
            "checksum_source": ".sha256 + pinned"}


def install_go():
    rel = get_json("https://go.dev/dl/?mode=json&include=all")
    cands = [r for r in rel if r.get("stable") and re.fullmatch(rf"go{re.escape(GO_MINOR)}(\.\d+)?", r["version"])]
    if not cands:
        raise SystemExit(f"go.dev lists no stable Go {GO_MINOR}.x")
    key = lambda r: tuple(int(x) for x in re.findall(r"\d+", r["version"]))  # noqa: E731
    r = max(cands, key=key)
    f = next(f for f in r["files"] if f["os"] == "windows" and f["arch"] == "amd64" and f["kind"] == "archive")
    url = f"https://go.dev/dl/{f['filename']}"
    path, d = download(url, f["filename"], sha256=f["sha256"])
    unpack_zip(path, os.path.join(TC, "go"), top="go")
    return {"version": r["version"], "url": url, "sha256": d, "checksum_source": "go.dev/dl JSON"}


def install_jdk():
    q = (f"https://api.adoptium.net/v3/assets/latest/{JDK_FEATURE}/hotspot?architecture=x64&image_type=jdk"
         "&os=windows&vendor=eclipse")
    rel = get_json(q)
    pkg = [a for a in rel if a["binary"]["package"]["name"].endswith(".zip")]
    if not pkg:
        raise SystemExit(f"Adoptium lists no Windows x64 JDK {JDK_FEATURE} zip")
    a = pkg[0]
    p = a["binary"]["package"]
    path, d = download(p["link"], p["name"], sha256=p["checksum"])
    unpack_zip(path, os.path.join(TC, "jdk"), top=single_top(path))
    return {"version": a.get("release_name") or a["version"]["semver"], "url": p["link"], "sha256": d,
            "checksum_source": "Adoptium API"}


def install_gcc():
    rels = get_json(f"https://api.github.com/repos/{WINLIBS_REPO}/releases?per_page=30")
    best = None
    for r in rels:
        if r.get("prerelease") or r.get("draft"):
            continue
        for a in r.get("assets", []):
            m = WINLIBS_ASSET.match(a["name"])
            if m:
                k = (tuple(int(x) for x in m.group(1).split(".")), tuple(int(x) for x in m.group(2).split(".")),
                     int(m.group(3)))
                if best is None or k > best[0]:
                    best = (k, r, a)
    if best is None:
        raise SystemExit(f"no WinLibs x86_64 posix-seh UCRT zip among the last {len(rels)} releases")
    _, r, a = best
    sha = None
    side = next((x for x in r["assets"] if x["name"] == a["name"] + ".sha256"), None)
    if side:
        sha = get(side["browser_download_url"]).decode("utf-8").split()[0].lower()
        src = "release .sha256 asset"
    if (a.get("digest") or "").startswith("sha256:"):
        dg = a["digest"].split(":", 1)[1].lower()
        if sha and sha != dg:
            raise SystemExit(f"{a['name']}: .sha256 asset {sha} disagrees with GitHub's digest {dg}")
        sha, src = dg, ("release .sha256 asset + GitHub digest" if side else "GitHub asset digest")
    if not sha:
        raise SystemExit(f"{a['name']}: no published checksum (no .sha256 asset, no GitHub digest)")
    path, d = download(a["browser_download_url"], a["name"], sha256=sha)
    unpack_zip(path, os.path.join(TC, "gcc"), top="mingw64")
    return {"version": r["tag_name"], "url": a["browser_download_url"], "sha256": d, "checksum_source": src}


INSTALLERS = {"node": install_node, "typescript": install_typescript, "rust": install_rust, "go": install_go,
              "jdk": install_jdk, "gcc": install_gcc}
VERSION_CMDS = {"gcc": [["gcc", "bin", "g++.exe"], "--version"], "jdk": [["jdk", "bin", "javac.exe"], "-version"],
                "go": [["go", "bin", "go.exe"], "version"], "rust": [["rust", "bin", "rustc.exe"], "-vV"],
                "node": [["node", "node.exe"], "--version"]}


def version_line(name):
    if name == "typescript":
        return "TypeScript " + json.load(open(os.path.join(TC, "typescript", "package.json"), encoding="utf-8"))["version"]
    parts, arg = VERSION_CMDS[name]
    out = subprocess.run([os.path.join(TC, *parts), arg], capture_output=True, text=True, timeout=120)
    text = (out.stdout or out.stderr).strip()
    return " | ".join(text.splitlines()[:2]) if name == "rust" else text.splitlines()[0] if text else f"rc {out.returncode}"


def install(only):
    os.makedirs(TC, exist_ok=True)
    lock_path = os.path.join(TC, "lock.json")
    lock = json.load(open(lock_path, encoding="utf-8")) if os.path.exists(lock_path) else {}
    for name, fn in INSTALLERS.items():
        if only and name not in only:
            continue
        log(f"== {name}")
        info = fn()
        info["installed"] = time.strftime("%Y-%m-%d %H:%M:%S")
        info["version_output"] = version_line(name)
        log(f"{name}: {info['version_output']}")
        lock[name] = info
        with open(lock_path, "w", encoding="utf-8") as fh:
            json.dump(lock, fh, indent=1)
    shims = os.path.join(TC, "shims")
    os.makedirs(shims, exist_ok=True)
    for f in os.listdir(polyglot.SHIMS_SRC):
        shutil.copyfile(os.path.join(polyglot.SHIMS_SRC, f), os.path.join(shims, f))
    lock["shims"] = {f: hashlib.sha256(open(os.path.join(shims, f), "rb").read()).hexdigest() for f in sorted(os.listdir(shims))}
    from anneal import sandbox
    gocache = os.path.join(TC, "cache", "go-build")
    build_sid, run_sid = sandbox.setup_toolchains_access(
        runtime_dirs=[os.path.join(TC, "jdk"), os.path.join(TC, "node"), shims], build_writable=[gocache])
    lock["containers"] = {"build": build_sid, "run": run_sid}
    with open(lock_path, "w", encoding="utf-8") as fh:
        json.dump(lock, fh, indent=1)
    os.makedirs(RESULTS, exist_ok=True)
    shutil.copyfile(lock_path, os.path.join(RESULTS, "lock.json"))
    log(f"wrote {lock_path}")


def check(langs):
    """The smoke programs in the sandbox: every language must give the expected verdict for each program."""
    from anneal import polyglot_smoke as S
    ex, tc = polyglot.SandboxExecutor(), polyglot.Toolchain.windows()
    results, bad = [], 0
    for lang in langs:
        for prog in S.programs(lang):
            v = polyglot.judge(lang, prog["code"], prog["cases"], ex, tc, case_timeout_s=prog.get("timeout", 10.0),
                               all_cases=True)
            want = prog["expect"]
            good = v.stage == want
            bad += not good
            results.append({"lang": lang, "program": prog["name"], "expect": want, "got": v.stage,
                            "cases": v.cases, "build_s": v.build_s, "run_s": v.run_s, "ok": good,
                            "detail": v.detail[-600:] if not good else ""})
            log(f"{lang:10s} {prog['name']:12s} expect {want:13s} got {v.stage:13s} build {v.build_s:6.2f} s "
                f"run {v.run_s:6.2f} s {'OK' if good else 'MISMATCH ' + v.detail[-300:]}")
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, "smoke.json"), "w", encoding="utf-8") as fh:
        json.dump({"when": time.strftime("%Y-%m-%d %H:%M:%S"), "mismatches": bad, "results": results}, fh, indent=1)
    log(f"smoke: {len(results) - bad}/{len(results)} as expected; results/toolchains/smoke.json")
    return 1 if bad else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true", help="run the smoke programs instead of installing")
    ap.add_argument("--only", action="append", default=[], help=f"one of {list(INSTALLERS)} (install) or a "
                    f"language {list(polyglot.LANG_ORDER)} (check); repeatable")
    args = ap.parse_args()
    if args.check:
        return check(args.only or list(polyglot.LANG_ORDER))
    install(args.only)
    return 0


if __name__ == "__main__":
    sys.exit(main())
