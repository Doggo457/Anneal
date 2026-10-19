"""Run model-written Python in a Windows sandbox: an AppContainer with no capabilities inside a Job Object.

No admin rights are needed for any of it. Everything model-written (generated solutions, unit tests run against
them, EvalPlus checks) goes through run() and nowhere else.

What the child gets:
  - AppContainer "Anneal.CodeSandbox" with zero capabilities: no network at all (no internetClient, no
    privateNetwork, and AppContainers can't reach loopback), and file or registry access only where an ACL names
    the container SID or ALL APPLICATION PACKAGES (Windows system folders, read-only). The only grants made here:
    RX on the sandbox interpreter tree (C:\\AI\\training\\sandbox\\python, setup()) and Modify on the run's own
    scratch folder (C:\\AI\\training\\sandbox\\scratch\\<uuid>), which is deleted afterwards.
  - Interpreter: python.org's embeddable CPython 3.12.10 (same version as the training venv) with numpy, psutil
    and evalplus (no deps) in its "site" folder. Its python312._pth puts it in isolated mode (no PYTHON* variables,
    no user site, the script's folder is not on sys.path), so it is always started with -X utf8 -B. The _pth also
    has "import site", only so that site\\sitecustomize.py runs: it maps os.mkdir(path, 0o700) to an inherited ACL,
    because Python 3.12.4+ makes 0o700 folders owner-only, which the container itself can't use (tempfile).
  - Job object, assigned while the process is still suspended: kill on job close, at most 1 active process (no
    subprocesses), a job memory limit (default 4 GiB, EvalPlus's own default), die on unhandled exception (no error
    dialogs), every UI restriction, below-normal priority. A wall-clock timeout terminates the job.
  - Only the three pipe handles are inherited (PROC_THREAD_ATTRIBUTE_HANDLE_LIST); the environment is built from
    scratch (SystemRoot, PATH, TEMP/TMP = scratch, PYTHONIOENCODING, PYTHONDONTWRITEBYTECODE, PYTHONNOUSERSITE),
    so no token or key in the parent's environment reaches the child.
  - stdout and stderr are drained by threads and capped (16 MB each by default); output past the cap is read and
    thrown away, so a print loop can fill neither memory nor disk.

Known gaps (results\\sandbox_findings.md): the child can read files that grant ALL APPLICATION PACKAGES (C:\\Windows,
C:\\Program Files: system files, no user data), and Windows gives every AppContainer a private storage folder and
registry key of its own (%LOCALAPPDATA%\\Packages\\anneal.codesandbox, setup() removes the container's own access to
the folder).

API:
  run(argv, stdin=b"", timeout_s=10, mem_bytes=4 << 30, files=None) -> Result(returncode, stdout, stderr,
      timed_out, wall_s, ...)
  argv are the interpreter's arguments, e.g. ["main.py"] with files={"main.py": b"..."}, or ["-c", "print(1)"].

  python -m anneal.sandbox --setup     creates the profile and grants the interpreter tree (once)
  python -m anneal.sandbox --check     a quick print(1+1)
"""

import ctypes
import os
import shutil
import subprocess
import threading
import time
import uuid
from collections import namedtuple
from ctypes import wintypes as wt

TRAINING_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SANDBOX_ROOT = os.path.join(TRAINING_ROOT, "sandbox")
PYTHON_DIR = os.path.join(SANDBOX_ROOT, "python")
PYTHON_EXE = os.path.join(PYTHON_DIR, "python.exe")
SCRATCH_ROOT = os.path.join(SANDBOX_ROOT, "scratch")
CONTAINER_NAME = "Anneal.CodeSandbox"
DEFAULT_MEM = 4 << 30
OUTPUT_CAP = 16 << 20

Result = namedtuple("Result", "returncode stdout stderr timed_out wall_s stdout_truncated stderr_truncated "
                              "peak_job_memory pid marker_s")


class SandboxUnavailable(RuntimeError):
    pass


# ---------------------------------------------------------------------------------------------- Win32 definitions

kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
advapi32 = ctypes.WinDLL("advapi32", use_last_error=True)
userenv = ctypes.WinDLL("userenv", use_last_error=True)

EXTENDED_STARTUPINFO_PRESENT = 0x00080000
CREATE_SUSPENDED = 0x00000004
CREATE_NO_WINDOW = 0x08000000
CREATE_UNICODE_ENVIRONMENT = 0x00000400
STARTF_USESTDHANDLES = 0x00000100
PROC_THREAD_ATTRIBUTE_HANDLE_LIST = 0x00020002
PROC_THREAD_ATTRIBUTE_SECURITY_CAPABILITIES = 0x00020009
JobObjectBasicUIRestrictions = 4
JobObjectExtendedLimitInformation = 9
JOB_OBJECT_LIMIT_ACTIVE_PROCESS = 0x00000008
JOB_OBJECT_LIMIT_PRIORITY_CLASS = 0x00000020
JOB_OBJECT_LIMIT_JOB_MEMORY = 0x00000200
JOB_OBJECT_LIMIT_DIE_ON_UNHANDLED_EXCEPTION = 0x00000400
JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
JOB_OBJECT_UILIMIT_ALL = 0x000000FF
BELOW_NORMAL_PRIORITY_CLASS = 0x00004000
WAIT_OBJECT_0 = 0
WAIT_TIMEOUT = 0x102
HRESULT_ALREADY_EXISTS = ctypes.c_long(0x800700B7).value
SE_FILE_OBJECT = 1
DACL_SECURITY_INFORMATION = 4
GRANT_ACCESS = 1
SUB_CONTAINERS_AND_OBJECTS_INHERIT = 3
TRUSTEE_IS_SID = 0
TRUSTEE_IS_WELL_KNOWN_GROUP = 5
FILE_MODIFY = 0x001301BF  # FILE_GENERIC_READ | FILE_GENERIC_WRITE | FILE_GENERIC_EXECUTE | DELETE
TIMEOUT_EXIT_CODE = 124  # the exit code a timed-out job is given (as timeout(1) uses)


class SID_AND_ATTRIBUTES(ctypes.Structure):
    _fields_ = [("Sid", ctypes.c_void_p), ("Attributes", wt.DWORD)]


class SECURITY_CAPABILITIES(ctypes.Structure):
    _fields_ = [("AppContainerSid", ctypes.c_void_p), ("Capabilities", ctypes.POINTER(SID_AND_ATTRIBUTES)),
                ("CapabilityCount", wt.DWORD), ("Reserved", wt.DWORD)]


class STARTUPINFOW(ctypes.Structure):
    _fields_ = [("cb", wt.DWORD), ("lpReserved", wt.LPWSTR), ("lpDesktop", wt.LPWSTR), ("lpTitle", wt.LPWSTR),
                ("dwX", wt.DWORD), ("dwY", wt.DWORD), ("dwXSize", wt.DWORD), ("dwYSize", wt.DWORD),
                ("dwXCountChars", wt.DWORD), ("dwYCountChars", wt.DWORD), ("dwFillAttribute", wt.DWORD),
                ("dwFlags", wt.DWORD), ("wShowWindow", wt.WORD), ("cbReserved2", wt.WORD),
                ("lpReserved2", ctypes.c_void_p), ("hStdInput", wt.HANDLE), ("hStdOutput", wt.HANDLE),
                ("hStdError", wt.HANDLE)]


class STARTUPINFOEXW(ctypes.Structure):
    _fields_ = [("StartupInfo", STARTUPINFOW), ("lpAttributeList", ctypes.c_void_p)]


class PROCESS_INFORMATION(ctypes.Structure):
    _fields_ = [("hProcess", wt.HANDLE), ("hThread", wt.HANDLE), ("dwProcessId", wt.DWORD),
                ("dwThreadId", wt.DWORD)]


class IO_COUNTERS(ctypes.Structure):
    _fields_ = [(n, ctypes.c_ulonglong) for n in ("ReadOperationCount", "WriteOperationCount",
                                                  "OtherOperationCount", "ReadTransferCount",
                                                  "WriteTransferCount", "OtherTransferCount")]


class JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
    _fields_ = [("PerProcessUserTimeLimit", ctypes.c_longlong), ("PerJobUserTimeLimit", ctypes.c_longlong),
                ("LimitFlags", wt.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wt.DWORD),
                ("Affinity", ctypes.c_size_t), ("PriorityClass", wt.DWORD), ("SchedulingClass", wt.DWORD)]


class JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
    _fields_ = [("BasicLimitInformation", JOBOBJECT_BASIC_LIMIT_INFORMATION), ("IoInfo", IO_COUNTERS),
                ("ProcessMemoryLimit", ctypes.c_size_t), ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryUsed", ctypes.c_size_t), ("PeakJobMemoryUsed", ctypes.c_size_t)]


class JOBOBJECT_BASIC_UI_RESTRICTIONS(ctypes.Structure):
    _fields_ = [("UIRestrictionsClass", wt.DWORD)]


class TRUSTEE_W(ctypes.Structure):
    _fields_ = [("pMultipleTrustee", ctypes.c_void_p), ("MultipleTrusteeOperation", ctypes.c_int),
                ("TrusteeForm", ctypes.c_int), ("TrusteeType", ctypes.c_int), ("ptstrName", ctypes.c_void_p)]


class EXPLICIT_ACCESS_W(ctypes.Structure):
    _fields_ = [("grfAccessPermissions", wt.DWORD), ("grfAccessMode", ctypes.c_int),
                ("grfInheritance", wt.DWORD), ("Trustee", TRUSTEE_W)]


def _proto(fn, res, *args):
    fn.restype, fn.argtypes = res, list(args)
    return fn


_CreateProcessW = _proto(kernel32.CreateProcessW, wt.BOOL, wt.LPCWSTR, wt.LPWSTR, ctypes.c_void_p,
                         ctypes.c_void_p, wt.BOOL, wt.DWORD, ctypes.c_void_p, wt.LPCWSTR,
                         ctypes.POINTER(STARTUPINFOEXW), ctypes.POINTER(PROCESS_INFORMATION))
_InitAttrList = _proto(kernel32.InitializeProcThreadAttributeList, wt.BOOL, ctypes.c_void_p, wt.DWORD, wt.DWORD,
                       ctypes.POINTER(ctypes.c_size_t))
_UpdateAttr = _proto(kernel32.UpdateProcThreadAttribute, wt.BOOL, ctypes.c_void_p, wt.DWORD, ctypes.c_size_t,
                     ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_void_p)
_DeleteAttrList = _proto(kernel32.DeleteProcThreadAttributeList, None, ctypes.c_void_p)
_CreateJobObjectW = _proto(kernel32.CreateJobObjectW, wt.HANDLE, ctypes.c_void_p, wt.LPCWSTR)
_SetInformationJobObject = _proto(kernel32.SetInformationJobObject, wt.BOOL, wt.HANDLE, ctypes.c_int,
                                  ctypes.c_void_p, wt.DWORD)
_QueryInformationJobObject = _proto(kernel32.QueryInformationJobObject, wt.BOOL, wt.HANDLE, ctypes.c_int,
                                    ctypes.c_void_p, wt.DWORD, ctypes.POINTER(wt.DWORD))
_AssignProcessToJobObject = _proto(kernel32.AssignProcessToJobObject, wt.BOOL, wt.HANDLE, wt.HANDLE)
_TerminateJobObject = _proto(kernel32.TerminateJobObject, wt.BOOL, wt.HANDLE, wt.UINT)
_ResumeThread = _proto(kernel32.ResumeThread, wt.DWORD, wt.HANDLE)
_WaitForSingleObject = _proto(kernel32.WaitForSingleObject, wt.DWORD, wt.HANDLE, wt.DWORD)
_GetExitCodeProcess = _proto(kernel32.GetExitCodeProcess, wt.BOOL, wt.HANDLE, ctypes.POINTER(wt.DWORD))
_TerminateProcess = _proto(kernel32.TerminateProcess, wt.BOOL, wt.HANDLE, wt.UINT)
_CloseHandle = _proto(kernel32.CloseHandle, wt.BOOL, wt.HANDLE)
_CreatePipe = _proto(kernel32.CreatePipe, wt.BOOL, ctypes.POINTER(wt.HANDLE), ctypes.POINTER(wt.HANDLE),
                     ctypes.c_void_p, wt.DWORD)
_SetHandleInformation = _proto(kernel32.SetHandleInformation, wt.BOOL, wt.HANDLE, wt.DWORD, wt.DWORD)
_LocalFree = _proto(kernel32.LocalFree, ctypes.c_void_p, ctypes.c_void_p)
_CreateAppContainerProfile = _proto(userenv.CreateAppContainerProfile, ctypes.c_long, wt.LPCWSTR, wt.LPCWSTR,
                                    wt.LPCWSTR, ctypes.c_void_p, wt.DWORD, ctypes.POINTER(ctypes.c_void_p))
_DeriveAppContainerSid = _proto(userenv.DeriveAppContainerSidFromAppContainerName, ctypes.c_long, wt.LPCWSTR,
                                ctypes.POINTER(ctypes.c_void_p))
_ConvertSidToStringSidW = _proto(advapi32.ConvertSidToStringSidW, wt.BOOL, ctypes.c_void_p,
                                 ctypes.POINTER(wt.LPWSTR))
_GetNamedSecurityInfoW = _proto(advapi32.GetNamedSecurityInfoW, wt.DWORD, wt.LPCWSTR, ctypes.c_int, wt.DWORD,
                                ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p,
                                ctypes.POINTER(ctypes.c_void_p))
_SetEntriesInAclW = _proto(advapi32.SetEntriesInAclW, wt.DWORD, wt.ULONG, ctypes.POINTER(EXPLICIT_ACCESS_W),
                           ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p))
_SetNamedSecurityInfoW = _proto(advapi32.SetNamedSecurityInfoW, wt.DWORD, wt.LPWSTR, ctypes.c_int, wt.DWORD,
                                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p)


def _check(ok, what):
    if not ok:
        raise ctypes.WinError(ctypes.get_last_error(), f"{what} failed")


# ---------------------------------------------------------------------------------------------- the container

_sid_lock = threading.Lock()
_sid = None  # (PSID as c_void_p, "S-1-15-2-...")


def container_sid():
    """The AppContainer SID (created on first use; derived from the name if the profile already exists)."""
    global _sid
    with _sid_lock:
        if _sid is None:
            psid = ctypes.c_void_p()
            hr = _CreateAppContainerProfile(CONTAINER_NAME, CONTAINER_NAME, "Anneal: runs model-written code",
                                            None, 0, ctypes.byref(psid))
            if hr == HRESULT_ALREADY_EXISTS:
                hr = _DeriveAppContainerSid(CONTAINER_NAME, ctypes.byref(psid))
            if hr != 0:
                raise SandboxUnavailable(f"AppContainer profile {CONTAINER_NAME}: HRESULT 0x{hr & 0xFFFFFFFF:08X}")
            s = wt.LPWSTR()
            _check(_ConvertSidToStringSidW(psid, ctypes.byref(s)), "ConvertSidToStringSidW")
            text = s.value
            _LocalFree(s)
            _sid = (psid, text)  # the PSID is kept for the life of the process
        return _sid


def grant(path, psid, access=FILE_MODIFY):
    """Add an inheritable allow ACE for the SID to path's DACL (what icacls /grant *SID:(OI)(CI)(M) does)."""
    old_dacl, sd, new_dacl = ctypes.c_void_p(), ctypes.c_void_p(), ctypes.c_void_p()
    err = _GetNamedSecurityInfoW(path, SE_FILE_OBJECT, DACL_SECURITY_INFORMATION, None, None,
                                 ctypes.byref(old_dacl), None, ctypes.byref(sd))
    if err:
        raise ctypes.WinError(err, f"GetNamedSecurityInfoW({path})")
    try:
        ea = EXPLICIT_ACCESS_W(access, GRANT_ACCESS, SUB_CONTAINERS_AND_OBJECTS_INHERIT,
                               TRUSTEE_W(None, 0, TRUSTEE_IS_SID, TRUSTEE_IS_WELL_KNOWN_GROUP, psid.value))
        err = _SetEntriesInAclW(1, ctypes.byref(ea), old_dacl, ctypes.byref(new_dacl))
        if err:
            raise ctypes.WinError(err, "SetEntriesInAclW")
        err = _SetNamedSecurityInfoW(path, SE_FILE_OBJECT, DACL_SECURITY_INFORMATION, None, None, new_dacl, None)
        if err:
            raise ctypes.WinError(err, f"SetNamedSecurityInfoW({path})")
    finally:
        if new_dacl:
            _LocalFree(new_dacl)
        _LocalFree(sd)


def profile_folder():
    return os.path.join(os.environ.get("LOCALAPPDATA", ""), "Packages", CONTAINER_NAME.lower())


def setup(verbose=True):
    """Create the profile, grant the container RX on the interpreter tree (and nothing else), and remove the
    container's own access to the private storage folder Windows created for the profile."""
    if not os.path.isfile(PYTHON_EXE):
        raise SandboxUnavailable(f"sandbox interpreter missing: {PYTHON_EXE}")
    psid, sid = container_sid()
    os.makedirs(SCRATCH_ROOT, exist_ok=True)
    # pip leaves folders with a protected owner-only ACL (Python's mkdir 0o700): make the whole tree inherit, then
    # grant the container RX at the top.
    subprocess.run(["icacls", PYTHON_DIR, "/reset", "/T", "/Q"], capture_output=True, text=True)
    out = subprocess.run(["icacls", PYTHON_DIR, "/grant", f"*{sid}:(OI)(CI)(RX)", "/Q"], capture_output=True,
                         text=True)
    if verbose:
        print(f"container SID {sid}; icacls interpreter tree: rc {out.returncode} {out.stdout.strip()}"
              f"{out.stderr.strip()}")
    folder = profile_folder()
    if os.path.isdir(folder):
        out = subprocess.run(["icacls", folder, "/remove:g", f"*{sid}", "/T", "/Q"], capture_output=True, text=True)
        if verbose:
            print(f"icacls remove container grant on {folder}: rc {out.returncode} {out.stdout.strip()}")
    return sid


def available():
    """True if the sandbox interpreter is installed and the container exists (setup() was run)."""
    try:
        container_sid()
    except (SandboxUnavailable, OSError):
        return False
    return os.path.isfile(PYTHON_EXE) and os.path.isdir(SCRATCH_ROOT)


def require():
    if not available():
        raise SandboxUnavailable("the code sandbox isn't set up: run  python -m anneal.sandbox --setup")


# ---------------------------------------------------------------------------------------------- one run

def _env_block(scratch):
    system_root = os.environ.get("SystemRoot", r"C:\Windows")
    env = {
        "SystemRoot": system_root,
        "PATH": PYTHON_DIR + ";" + os.path.join(system_root, "System32"),
        "TEMP": scratch,
        "TMP": scratch,
        "PYTHONIOENCODING": "utf-8",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONNOUSERSITE": "1",
        # CreateProcess refuses an AppContainer launch without LOCALAPPDATA (ERROR_ENVVAR_NOT_FOUND) and then
        # rewrites LOCALAPPDATA, TEMP and TMP to the container's own folder (...\Packages\anneal.codesandbox\AC),
        # where setup() removed the container's access, so tempfile falls back to the current folder (scratch).
        "LOCALAPPDATA": os.environ.get("LOCALAPPDATA", ""),
    }
    text = "".join(f"{k}={v}\0" for k, v in sorted(env.items(), key=lambda kv: kv[0].upper())) + "\0"
    return ctypes.create_unicode_buffer(text, len(text))


def _pipe():
    """(read end, write end), neither inheritable yet."""
    r, w = wt.HANDLE(), wt.HANDLE()
    _check(_CreatePipe(ctypes.byref(r), ctypes.byref(w), None, 0), "CreatePipe")
    return r.value, w.value


def _make_inheritable(h):
    _check(_SetHandleInformation(h, 1, 1), "SetHandleInformation")  # HANDLE_FLAG_INHERIT


def _reader(handle, cap, out, marker=None, on_marker=None):
    import msvcrt
    fd = msvcrt.open_osfhandle(handle, os.O_RDONLY)
    buf, n, truncated = [], 0, False
    tail = b""
    with os.fdopen(fd, "rb", buffering=0) as fh:
        while True:
            try:
                chunk = fh.read(65536)
            except OSError:
                break
            if not chunk:
                break
            if marker is not None:
                window = tail + chunk
                if marker in window:
                    on_marker()
                    marker = None
                else:
                    tail = window[-len(marker):]
            if n < cap:
                take = chunk[:cap - n]
                buf.append(take)
                n += len(take)
                truncated = truncated or len(take) < len(chunk)
            else:
                truncated = True
    out.append((b"".join(buf), truncated))


def _writer(handle, data):
    import msvcrt
    fd = msvcrt.open_osfhandle(handle, 0)
    try:
        with os.fdopen(fd, "wb", buffering=0) as fh:
            if data:
                fh.write(data)
    except OSError:
        pass  # the child exited without reading


def _new_job(mem_bytes):
    job = _CreateJobObjectW(None, None)
    _check(job, "CreateJobObjectW")
    info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
    info.BasicLimitInformation.LimitFlags = (JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE | JOB_OBJECT_LIMIT_ACTIVE_PROCESS
                                             | JOB_OBJECT_LIMIT_JOB_MEMORY
                                             | JOB_OBJECT_LIMIT_DIE_ON_UNHANDLED_EXCEPTION
                                             | JOB_OBJECT_LIMIT_PRIORITY_CLASS)
    info.BasicLimitInformation.ActiveProcessLimit = 1
    info.BasicLimitInformation.PriorityClass = BELOW_NORMAL_PRIORITY_CLASS
    info.JobMemoryLimit = mem_bytes
    _check(_SetInformationJobObject(job, JobObjectExtendedLimitInformation, ctypes.byref(info),
                                    ctypes.sizeof(info)), "SetInformationJobObject(limits)")
    ui = JOBOBJECT_BASIC_UI_RESTRICTIONS(JOB_OBJECT_UILIMIT_ALL)
    _check(_SetInformationJobObject(job, JobObjectBasicUIRestrictions, ctypes.byref(ui), ctypes.sizeof(ui)),
           "SetInformationJobObject(UI)")
    return job


def _peak_job_memory(job):
    info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
    if _QueryInformationJobObject(job, JobObjectExtendedLimitInformation, ctypes.byref(info), ctypes.sizeof(info),
                                  None):
        return int(info.PeakJobMemoryUsed)
    return None


def _remove_tree(path):
    try:
        shutil.rmtree(path)
    except OSError:
        # The child may have left a read-only file or its own restrictive ACL: reset (we own every file) and retry.
        subprocess.run(["icacls", path, "/reset", "/T", "/C", "/Q"], capture_output=True)
        subprocess.run(["attrib", "-R", "-H", "-S", os.path.join(path, "*"), "/S", "/D"], capture_output=True)
        shutil.rmtree(path, ignore_errors=True)


def run(argv, stdin=b"", timeout_s=10.0, mem_bytes=DEFAULT_MEM, files=None, output_cap=OUTPUT_CAP, marker=None,
        after_marker_s=None):
    """Run the sandbox interpreter with argv in a fresh scratch folder. files: {relative name: bytes or str}
    written there first. Returns Result; timed_out=True means the job was terminated at its deadline.

    The deadline is timeout_s after the start; with marker (bytes) and after_marker_s, it moves to after_marker_s
    after the moment the marker first appears on stdout (a start-up allowance, then the real time limit)."""
    psid, _ = container_sid()
    scratch = os.path.join(SCRATCH_ROOT, uuid.uuid4().hex)
    os.makedirs(scratch)
    handles, parent_ends = [], []
    job = attr = None
    attr_ready = False
    try:
        grant(scratch, psid)
        for name, data in (files or {}).items():
            path = os.path.normpath(os.path.join(scratch, name))
            if not path.startswith(scratch + os.sep):
                raise ValueError(f"file name outside the scratch folder: {name!r}")
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "wb") as fh:
                fh.write(data.encode("utf-8") if isinstance(data, str) else data)

        in_parent, in_child = _pipe()
        out_parent, out_child = _pipe()
        err_parent, err_child = _pipe()
        in_child, in_parent = in_parent, in_child  # stdin: the child reads the read end
        handles += [in_child, out_child, err_child]
        parent_ends += [in_parent, out_parent, err_parent]
        for h in (in_child, out_child, err_child):
            _make_inheritable(h)

        caps = SECURITY_CAPABILITIES(psid.value, None, 0, 0)
        inherit = (wt.HANDLE * 3)(in_child, out_child, err_child)
        size = ctypes.c_size_t()
        _InitAttrList(None, 2, 0, ctypes.byref(size))
        attr = ctypes.create_string_buffer(size.value)
        _check(_InitAttrList(ctypes.byref(attr), 2, 0, ctypes.byref(size)), "InitializeProcThreadAttributeList")
        attr_ready = True
        _check(_UpdateAttr(ctypes.byref(attr), 0, PROC_THREAD_ATTRIBUTE_SECURITY_CAPABILITIES, ctypes.byref(caps),
                           ctypes.sizeof(caps), None, None), "UpdateProcThreadAttribute(capabilities)")
        _check(_UpdateAttr(ctypes.byref(attr), 0, PROC_THREAD_ATTRIBUTE_HANDLE_LIST, ctypes.byref(inherit),
                           ctypes.sizeof(inherit), None, None),
               "UpdateProcThreadAttribute(handle list)")

        si = STARTUPINFOEXW()
        si.StartupInfo.cb = ctypes.sizeof(STARTUPINFOEXW)
        si.StartupInfo.dwFlags = STARTF_USESTDHANDLES
        si.StartupInfo.hStdInput, si.StartupInfo.hStdOutput, si.StartupInfo.hStdError = in_child, out_child, err_child
        si.lpAttributeList = ctypes.addressof(attr)
        cmd = subprocess.list2cmdline([PYTHON_EXE, "-X", "utf8", "-B"] + list(argv))
        cmdbuf = ctypes.create_unicode_buffer(cmd)
        envbuf = _env_block(scratch)
        pi = PROCESS_INFORMATION()
        job = _new_job(mem_bytes)
        t0 = time.perf_counter()
        _check(_CreateProcessW(PYTHON_EXE, cmdbuf, None, None, True,
                               EXTENDED_STARTUPINFO_PRESENT | CREATE_SUSPENDED | CREATE_NO_WINDOW
                               | CREATE_UNICODE_ENVIRONMENT, ctypes.byref(envbuf), scratch, ctypes.byref(si),
                               ctypes.byref(pi)), "CreateProcessW")
        for h in (in_child, out_child, err_child):
            _CloseHandle(h)
        handles = [pi.hProcess, pi.hThread]
        if not _AssignProcessToJobObject(job, pi.hProcess):
            err = ctypes.get_last_error()
            _TerminateProcess(pi.hProcess, 1)
            raise ctypes.WinError(err, "AssignProcessToJobObject")

        outs, errs = [], []
        deadline = [t0 + timeout_s]
        marker_at = []

        def on_marker():
            now = time.perf_counter()
            marker_at.append(now - t0)
            if after_marker_s is not None:
                deadline[0] = now + after_marker_s

        threads = [threading.Thread(target=_reader, args=(out_parent, output_cap, outs, marker, on_marker),
                                    daemon=True),
                   threading.Thread(target=_reader, args=(err_parent, output_cap, errs), daemon=True),
                   threading.Thread(target=_writer, args=(in_parent, stdin), daemon=True)]
        for t in threads:
            t.start()
        parent_ends = []  # the threads own (and close) them now
        _ResumeThread(pi.hThread)
        timed_out = False
        while True:
            left = deadline[0] - time.perf_counter()
            if left <= 0:
                timed_out = True
                break
            # short slices only while a marker can still move the deadline
            ms = int(min(left, 0.05 if marker is not None else left) * 1000) + 1
            if _WaitForSingleObject(pi.hProcess, ms) != WAIT_TIMEOUT:
                break
        if timed_out:
            _TerminateJobObject(job, TIMEOUT_EXIT_CODE)
            _WaitForSingleObject(pi.hProcess, 10_000)
        wall = time.perf_counter() - t0
        code = wt.DWORD()
        _GetExitCodeProcess(pi.hProcess, ctypes.byref(code))
        peak = _peak_job_memory(job)
        for t in threads:
            t.join(10)
        stdout, out_trunc = outs[0] if outs else (b"", False)
        stderr, err_trunc = errs[0] if errs else (b"", False)
        rc = code.value - (1 << 32) if code.value & 0x80000000 else code.value
        return Result(rc, stdout, stderr, timed_out, round(wall, 4), out_trunc, err_trunc, peak, pi.dwProcessId,
                      round(marker_at[0], 4) if marker_at else None)
    finally:
        for h in handles:
            if h:
                _CloseHandle(h)
        if job:
            _CloseHandle(job)  # kill on job close: nothing outlives the run
        if attr is not None and attr_ready:
            _DeleteAttrList(ctypes.byref(attr))
        for h in parent_ends:
            _CloseHandle(h)
        _remove_tree(scratch)


def run_python(code, stdin=b"", timeout_s=10.0, mem_bytes=DEFAULT_MEM, files=None, name="main.py"):
    """Run one source file (written to scratch as name)."""
    files = dict(files or {})
    files[name] = code
    return run([name], stdin=stdin, timeout_s=timeout_s, mem_bytes=mem_bytes, files=files)


if __name__ == "__main__":
    import sys
    if "--setup" in sys.argv:
        setup()
    r = run(["-c", "print(1+1)"], timeout_s=20)
    print(r)
