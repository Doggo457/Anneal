"""Memory guard for training on the EVO-X2 (26 Nov 2026).

The PC crashed twice while round 6 (PC r007) trained step 176 (bugcheck 0x9F on 25 Nov 09:16 UTC; 0x3B at 19:39 UTC
after Windows' free memory had fallen to 0.02 GiB at 18:58). On this unified-memory machine the GPU's shared memory is
locked Windows RAM, so a training process that grows there can starve Windows itself.

MemGuard, a thread, reads Windows' available physical memory every 0.5 s. After two readings in a row under the floor
(train config mem_guard_free_gib, default 1.0) it marks the run paused (status.json state paused: the loop shows Paused
and its watchdog leaves it alone rather than restarting into the same batch), writes STOP, saves the last minute of
readings and ends its own process at once, which gives all its memory back to Windows. Training resumes from the last
checkpoint like after any pause; nothing about what the model learns changes.

It never calls torch or HIP: torch's memory counters take the allocator's lock, which a HIP allocation stuck under
memory pressure would hold. Torch's view is logged by train_longrun.py at every micro-batch end (event micro_mem).

Trace (mem_trace.jsonl in the run folder): a reading every 5 s; every reading while available memory is under the watch
level (floor + 2 GiB); the processes using the most memory when a watch starts; the last 120 readings at a trip.
"""
import collections
import ctypes
import json
import operator
import os
import sys
import threading
import time

IS_WINDOWS = sys.platform == "win32"


def gib(n):
    return None if n is None else round(n / 2 ** 30, 3)


if IS_WINDOWS:
    from ctypes import wintypes

    class _MemoryStatusEx(ctypes.Structure):
        _fields_ = [("dwLength", wintypes.DWORD), ("dwMemoryLoad", wintypes.DWORD)] + [
            (n, ctypes.c_ulonglong) for n in ("ullTotalPhys", "ullAvailPhys", "ullTotalPageFile", "ullAvailPageFile",
                                              "ullTotalVirtual", "ullAvailVirtual", "ullAvailExtendedVirtual")]

    class _PerformanceInfo(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD)] + [
            (n, ctypes.c_size_t) for n in ("CommitTotal", "CommitLimit", "CommitPeak", "PhysicalTotal",
                                           "PhysicalAvailable", "SystemCache", "KernelTotal", "KernelPaged",
                                           "KernelNonpaged", "PageSize")] + [
            (n, wintypes.DWORD) for n in ("HandleCount", "ProcessCount", "ThreadCount")]

    _k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _k32.GetCurrentProcess.restype = wintypes.HANDLE
    _k32.GetCurrentThread.restype = wintypes.HANDLE
    _k32.TerminateProcess.argtypes = [wintypes.HANDLE, wintypes.UINT]
    _k32.SetThreadPriority.argtypes = [wintypes.HANDLE, ctypes.c_int]


def read_system():
    """Windows: available physical memory, commit charge and limit (GlobalMemoryStatusEx), kernel non-paged pool and
    system cache (GetPerformanceInfo). Elsewhere psutil's available memory (tests)."""
    if not IS_WINDOWS:
        import psutil
        return {"avail_gib": gib(psutil.virtual_memory().available)}
    ms = _MemoryStatusEx()
    ms.dwLength = ctypes.sizeof(ms)
    if not _k32.GlobalMemoryStatusEx(ctypes.byref(ms)):
        raise ctypes.WinError(ctypes.get_last_error())
    out = {"avail_gib": gib(ms.ullAvailPhys), "commit_gib": gib(ms.ullTotalPageFile - ms.ullAvailPageFile),
           "commit_limit_gib": gib(ms.ullTotalPageFile)}
    pi = _PerformanceInfo()
    pi.cb = ctypes.sizeof(pi)
    if _k32.K32GetPerformanceInfo(ctypes.byref(pi), pi.cb):
        out.update(nonpaged_gib=gib(pi.KernelNonpaged * pi.PageSize), cache_gib=gib(pi.SystemCache * pi.PageSize))
    return out


def end_process(code):
    """End this process now. On Windows with TerminateProcess, so no DLL unload code (the GPU runtime's included) can
    hold the exit up while Windows is short of memory."""
    try:
        sys.stdout.flush()
        sys.stderr.flush()
    except Exception:
        pass
    if IS_WINDOWS:
        _k32.TerminateProcess(_k32.GetCurrentProcess(), code)
    os._exit(code)


class MemGuard(threading.Thread):
    """job: anneal.runtime.Run. extra(): plain values added to every reading (no torch calls). read_sys and end are
    replaceable for tests."""

    def __init__(self, job, floor_gib=1.0, watch_gib=None, interval=0.5, trip_after=2, log_every=10, history=120,
                 extra=None, read_sys=read_system, end=end_process, exit_code=3):
        super().__init__(daemon=True, name="mem-guard")
        self.job, self.floor, self.interval, self.trip_after = job, float(floor_gib), float(interval), int(trip_after)
        self.watch = float(watch_gib) if watch_gib is not None else self.floor + 2.0
        self.log_every, self.extra, self.read_sys = int(log_every), extra, read_sys
        self.end, self.exit_code = end, exit_code
        self.history = collections.deque(maxlen=int(history))
        self.stop = threading.Event()
        self.trace_path = os.path.join(job.dir, "mem_trace.jsonl")
        self.tripped, self.min_avail, self.watching, self.errors = None, None, False, 0
        try:
            import psutil
            self.proc = psutil.Process()
        except Exception:
            self.proc = None

    def reading(self):
        r = {"t": round(time.time(), 2)}
        r.update(self.read_sys())
        if self.proc is not None:
            try:
                mi = self.proc.memory_info()
                r.update(proc_ws_gib=gib(mi.rss), proc_private_gib=gib(getattr(mi, "private", None)))
            except Exception:
                pass
        if self.extra is not None:
            try:
                r.update(self.extra())
            except Exception:
                pass
        return r

    def write(self, obj):
        try:
            with open(self.trace_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(obj, default=str) + "\n")
        except OSError:
            pass

    def top_processes(self, n=6):
        """The n processes with the most private memory (resident memory where private is not reported)."""
        try:
            import psutil
        except ImportError:
            return []
        rows = []
        for p in psutil.process_iter(["pid", "name"]):
            try:
                mi = p.memory_info()
                rows.append((getattr(mi, "private", mi.rss), mi.rss, p.info["pid"], p.info["name"]))
            except Exception:
                continue
        rows.sort(reverse=True)
        return [{"pid": pid, "name": name, "private_gib": gib(priv), "ws_gib": gib(ws)}
                for priv, ws, pid, name in rows[:n]]

    def run(self):
        if IS_WINDOWS:
            try:
                _k32.SetThreadPriority(_k32.GetCurrentThread(), 2)  # THREAD_PRIORITY_HIGHEST
            except Exception:
                pass
        self.write({"event": "start", "t": round(time.time(), 2), "floor_gib": self.floor, "watch_gib": self.watch,
                    "interval_s": self.interval, "trip_after": self.trip_after, "pid": os.getpid()})
        low, n = 0, 0
        while not self.stop.is_set():
            try:
                r = self.reading()
            except Exception as exc:  # a failed reading never stops training
                self.errors += 1
                if operator.le(self.errors, 5):
                    self.write({"event": "read_error", "t": round(time.time(), 2), "error": repr(exc)})
                self.stop.wait(self.interval)
                continue
            avail = r.get("avail_gib")
            self.history.append(r)
            if avail is not None:
                self.min_avail = avail if self.min_avail is None else min(self.min_avail, avail)
                if operator.lt(avail, self.watch) and not self.watching:
                    self.watching = True
                    self.write({"event": "watch", "t": r["t"], "avail_gib": avail, "top": self.top_processes()})
                elif operator.ge(avail, self.watch + 1.0) and self.watching:
                    self.watching = False
                    self.write({"event": "watch_end", "t": r["t"], "avail_gib": avail})
            if self.watching or n % self.log_every == 0:
                self.write(r)
            n += 1
            low = low + 1 if avail is not None and operator.lt(avail, self.floor) else 0
            if operator.ge(low, self.trip_after):
                self.trip(r)
                return
            self.stop.wait(self.interval)

    def trip(self, r):
        self.tripped = r
        msg = (f"mem_guard: Windows free memory {r['avail_gib']} GiB, under the {self.floor} GiB floor for "
               f"{self.trip_after} readings in a row; training ended itself to keep Windows running and is paused "
               f"(resume continues from the last checkpoint)")
        lock = None
        try:
            self.job.write_status(state="paused", error=msg, paused_by="mem_guard")
            lock = getattr(self.job, "_status_lock", None)
            if lock is not None and not lock.acquire(timeout=5):  # no later status write can undo paused
                lock = None
        except Exception:
            pass
        try:
            with open(self.job.stop_path, "w", encoding="utf-8") as fh:
                fh.write(msg + "\n")
        except OSError:
            pass
        try:
            self.write({"event": "trip", "t": r["t"], "floor_gib": self.floor, "reading": r,
                        "history": list(self.history)})
            self.job.metric(event="mem_guard_trip", floor_gib=self.floor, **{k: v for k, v in r.items() if k != "t"})
            print("\n" + msg, flush=True)
        except Exception:
            pass
        self.end(self.exit_code)
        if lock is not None:  # reached only when end() returns (tests)
            lock.release()

    def close(self):
        self.stop.set()
        self.write({"event": "stop", "t": round(time.time(), 2), "min_avail_gib": self.min_avail})


def start(job, floor_gib, extra=None, watch_gib=None):
    """Start a guard unless floor_gib is 0 or unset; returns it, or None."""
    if not floor_gib or operator.le(float(floor_gib), 0):
        print("mem guard: off")
        return None
    g = MemGuard(job, floor_gib=floor_gib, watch_gib=watch_gib, extra=extra)
    g.start()
    print(f"mem guard: training ends itself, paused, if Windows' free memory stays under {g.floor} GiB "
          f"(checked every {g.interval} s); trace {g.trace_path}")
    return g
