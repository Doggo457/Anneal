# Code sandbox for model-written Python (AppContainer + Job Object)

19 October 2026. Code: `anneal/sandbox.py`. Test: `tests/sandbox_check.py` (`results/sandbox-check`, RESULT: PASS). No admin rights, no WSL, no new accounts, no firewall rules.

Everything model-written runs only through `anneal.sandbox.run()`: training-data unit tests (`generate_traces.py --run-tests`) and EvalPlus checks (`scripts/evalplus_win.py`).

## Design

**Interpreter**
- python.org's embeddable CPython 3.12.10 (amd64), the same version as the training venv's base Python, unpacked to `sandbox\python`.
- Hash check: SHA-256 `4acbed6d…3025a3c3`. It equals the digest in python.org's Sigstore bundle for the file. The bundle's certificate names `thomas@python.org` (the 3.12 release manager), issued through Google OIDC with one transparency-log entry.
  - The 3.12.10 release page no longer lists MD5 sums; it gives Sigstore bundles and SPDX files instead.
  - I compared the digest only. The Sigstore signature itself was not verified, since that would need the `sigstore` package.
- `sandbox\python\site` holds numpy 2.4.4, psutil 7.2.2 and evalplus 0.3.1 (`--no-deps`). These are the venv's exact versions, installed with the venv's `pip --target`. The tree is precompiled with the sandbox Python itself, because the container can't write `__pycache__`.
- `python312._pth` reads `python312.zip`, `.`, `site`, `import site`.
  - The `import site` line is there only so that `site\sitecustomize.py` runs (next section).
  - A `._pth` file puts Python in isolated mode, so `PYTHON*` variables are ignored. Every run therefore starts `python.exe -X utf8 -B <argv>`.

**AppContainer** `Anneal.CodeSandbox`
- SID `S-1-15-2-138348897-3284411920-451308414-994790979-2321626205-9386469-1996511325`.
- Zero capabilities.
- `icacls … /grant *SID:(OI)(CI)(RX)` on `sandbox\python` only. This is the only standing grant.
- Each run gets a new `sandbox\scratch\<uuid>` folder, granted Modify for the SID (ctypes `SetEntriesInAcl`/`SetNamedSecurityInfo`) and deleted afterwards.
- `setup()` also removes the container's access to the private folder Windows makes for every AppContainer (`%LOCALAPPDATA%\Packages\anneal.codesandbox`). Files therefore can't persist between runs.

**Process creation**
- `CreateProcessW` with `EXTENDED_STARTUPINFO_PRESENT | CREATE_SUSPENDED | CREATE_NO_WINDOW | CREATE_UNICODE_ENVIRONMENT`.
- Attributes: `SECURITY_CAPABILITIES`, plus `HANDLE_LIST` with only the three pipe ends.
- The working folder is the scratch folder.
- The environment is built from nothing: SystemRoot, PATH (sandbox Python + System32), TEMP/TMP, PYTHONIOENCODING, PYTHONDONTWRITEBYTECODE, PYTHONNOUSERSITE, LOCALAPPDATA.

**Job object**, assigned before `ResumeThread`
- Limits: `KILL_ON_JOB_CLOSE`, `ACTIVE_PROCESS` = 1, `JOB_MEMORY` (default 4 GiB, EvalPlus's default), `DIE_ON_UNHANDLED_EXCEPTION`, all eight UI restrictions, `BELOW_NORMAL` priority.
- Timeout: `TerminateJobObject` with exit code 124.
- stdout and stderr are drained by threads and capped at 16 MB each. The rest is read and discarded, so a print loop never blocks or fills anything.

API: `run(argv, stdin=b"", timeout_s, mem_bytes=4<<30, files={name: bytes})` returns `returncode, stdout, stderr, timed_out, wall_s` (plus `stdout_truncated`, `stderr_truncated`, `peak_job_memory`, `pid`).

## Things I had to work around

1. **CreateProcess refuses an AppContainer launch without `LOCALAPPDATA`** in the environment block: error 203, ERROR_ENVVAR_NOT_FOUND.
   - With it present, Windows rewrites `LOCALAPPDATA`, `TEMP` and `TMP` to the container's own `…\Packages\anneal.codesandbox\AC` folder, so TEMP can't point at the scratch folder.
   - The container has no access there, so `tempfile.gettempdir()` falls back to the current folder, which is the scratch folder. The test checks this.
2. **Python 3.12.4+ makes `os.mkdir(path, 0o700)` owner-only**: a protected ACL for SYSTEM, Administrators and OWNER RIGHTS (the CVE-2024-4030 fix).
   - An AppContainer must be named in the ACL itself, so OWNER RIGHTS isn't enough. A folder from `tempfile.mkdtemp()`/`TemporaryDirectory()` is unusable by the process that made it.
   - This would break EvalPlus's `create_tempdir()` on every check.
   - Fix: `sandbox\python\site\sitecustomize.py` maps mode 0o700 to the default, so the folder inherits the scratch ACL. This is in the sandbox interpreter only.
3. **The same 0o700 behaviour in pip** left `site\evalplus` with a protected owner-only ACL, so the container couldn't import it. `setup()` now runs `icacls sandbox\python /reset /T` before the grant.

## Acceptance (`results/sandbox-check/summary.json`)

| check | result |
|---|---|
| `print(1+1)` | `2`, rc 0 |
| read a file created under `C:\AI\training` | PermissionError |
| read a file created in the user profile; list the profile | denied. This PC's profile is `C:\Users\ai sever`; there is no `C:\Users\callum`. |
| read a private notes file in the user profile (`C:\Users\ai sever\…\MEMORY.md`) | denied |
| read `C:\AI\portal\server.py`; list `C:\AI\portal` | denied |
| write `C:\AI\training`, the interpreter tree, `C:\Users\Public`, user TEMP, the container's own AC folder | all PermissionError |
| `HKCU\Software\AnnealSandboxProbe`; `HKCU\Environment` | Access denied |
| TCP 127.0.0.1:8080 (portal) | timed out (loopback blocked) |
| TCP 1.1.1.1:443 | WinError 10013 (forbidden) |
| DNS `python.org` | getaddrinfo failed |
| UDP to 1.1.1.1:53 | nothing comes back (timed out) |
| `subprocess` cmd.exe / python; `os.system` | WinError 1816 "not enough quota" (job process limit); `os.system` returns -1 |
| `os.kill(<parent pid>)` | Access denied |
| `bytearray(8 GiB)` | MemoryError in 0.15 s; RAM available 23.4 GiB before and after; a 1 GiB allocation right after works |
| `bytearray(5 GiB)` / `bytearray(3 GiB)` | MemoryError / works (the 4 GiB job limit) |
| infinite loop, timeout 3 s | killed at 3.01 s, rc 124, no sandbox python.exe left |
| infinite print, timeout 3 s | killed at 3.02 s; 16,777,216 bytes kept (truncated); no process left |
| `os.environ` | only LOCALAPPDATA, PATH, PYTHONDONTWRITEBYTECODE, PYTHONIOENCODING, PYTHONNOUSERSITE, SYSTEMROOT, TEMP, TMP |
| scratch folder | write, read back, list, `tempfile.mkdtemp`, read a file passed in `files=`: all work |
| stdin | round trip works |
| 8 copies of the whole battery in parallel | 264 checks, 0 failures, no processes left |

Overhead per run:
- Trivial run (`-c pass`), sequential: **30.8 ms** mean, 30.6 median, 33.1 p95.
- With 8 threads: 5.8 ms per run.
- `import numpy, evalplus.eval` adds about 100 ms (130 ms per run).

## Known gaps (accepted)

- **Some reads are allowed.** The child can read whatever grants ALL APPLICATION PACKAGES: `C:\Windows`, `C:\Program Files` and parts of HKLM. The test reads `C:\Windows\System32\drivers\etc\hosts`. These are system files with no user data.
- **Name resolution.** `getaddrinfo('localhost')` resolves (locally); no DNS query leaves the machine.
- **Container registry key.** Windows gives each AppContainer a private registry key of its own (HKCU `…\AppContainer\Storage\anneal.codesandbox`). Writes elsewhere in HKCU fail.
- **Peak memory reporting.** `PeakJobMemoryUsed` reports the size of a refused allocation: 8.02 GiB and 5.02 GiB above, although neither was committed. RAM didn't move, so this is the counter, not memory use.
- **Timeouts are not CPU time.** A busy child keeps its core until the wall-clock timeout ends it.
