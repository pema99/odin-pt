# useful when profiling to get consistent measurement

import argparse
import ctypes
import os
import subprocess
import sys
import tempfile
from ctypes import wintypes

BASE_GRAPHICS_CLOCK_MHZ = 1980
MARKER_PATH = os.path.join(tempfile.gettempdir(), "odin_pt_gpu_clock_lock")

SEE_MASK_NOCLOSEPROCESS = 0x00000040
SW_HIDE = 0
INFINITE = 0xFFFFFFFF


class ShellExecuteInfo(ctypes.Structure):
    _fields_ = [
        ("cbSize", wintypes.DWORD),
        ("fMask", ctypes.c_ulong),
        ("hwnd", wintypes.HWND),
        ("lpVerb", wintypes.LPCWSTR),
        ("lpFile", wintypes.LPCWSTR),
        ("lpParameters", wintypes.LPCWSTR),
        ("lpDirectory", wintypes.LPCWSTR),
        ("nShow", ctypes.c_int),
        ("hInstApp", wintypes.HINSTANCE),
        ("lpIDList", ctypes.c_void_p),
        ("lpClass", wintypes.LPCWSTR),
        ("hkeyClass", wintypes.HKEY),
        ("dwHotKey", wintypes.DWORD),
        ("hIcon", wintypes.HANDLE),
        ("hProcess", wintypes.HANDLE),
    ]


def is_admin():
    if os.name != "nt":
        return os.geteuid() == 0
    try:
        return ctypes.windll.shell32.IsUserAnAdmin() != 0
    except Exception:
        return False


def elevate(arguments):
    report = tempfile.NamedTemporaryFile(prefix="odin_pt_clocks_", suffix=".txt", delete=False)
    report.close()

    parameters = [os.path.abspath(__file__), "--elevated", "--report", report.name]
    if arguments.clock != BASE_GRAPHICS_CLOCK_MHZ:
        parameters += ["--clock", str(arguments.clock)]

    info = ShellExecuteInfo()
    info.cbSize = ctypes.sizeof(info)
    info.fMask = SEE_MASK_NOCLOSEPROCESS
    info.lpVerb = "runas"
    info.lpFile = sys.executable
    info.lpParameters = subprocess.list2cmdline(parameters)
    info.nShow = SW_HIDE

    if not ctypes.windll.shell32.ShellExecuteExW(ctypes.byref(info)):
        os.remove(report.name)
        sys.exit("elevation was declined")

    ctypes.windll.kernel32.WaitForSingleObject(info.hProcess, INFINITE)
    exit_code = wintypes.DWORD()
    ctypes.windll.kernel32.GetExitCodeProcess(info.hProcess, ctypes.byref(exit_code))
    ctypes.windll.kernel32.CloseHandle(info.hProcess)

    with open(report.name, encoding="utf-8") as handle:
        output = handle.read()
    os.remove(report.name)
    sys.stdout.write(output)
    sys.exit(exit_code.value)


def run_smi(args):
    try:
        result = subprocess.run(["nvidia-smi"] + args, capture_output=True, text=True)
    except FileNotFoundError:
        fail("nvidia-smi not found on PATH")
    return result.returncode, (result.stdout or "").strip(), (result.stderr or "").strip()


def query(fields):
    code, out, err = run_smi(["--query-gpu=" + ",".join(fields), "--format=csv,noheader,nounits"])
    if code != 0:
        fail(err or out or "nvidia-smi query failed")
    return [value.strip() for value in out.splitlines()[0].split(",")]


def supported_graphics_clocks():
    code, out, _ = run_smi(["--query-supported-clocks=graphics", "--format=csv,noheader,nounits"])
    if code != 0:
        return []
    clocks = [int(line.strip()) for line in out.splitlines() if line.strip().isdigit()]
    return sorted(set(clocks))


def nearest_supported(target):
    clocks = supported_graphics_clocks()
    if not clocks:
        return target
    return min(clocks, key=lambda clock: abs(clock - target))


REPORT_PATH = None


def emit(text):
    if REPORT_PATH is None:
        print(text)
        return
    with open(REPORT_PATH, "a", encoding="utf-8") as handle:
        handle.write(text + "\n")


def fail(message):
    emit(message)
    sys.exit(1)


def report_state(action, locked, target, live):
    name, current, maximum = query(["name", "clocks.gr", "clocks.max.gr"])
    emit(name)
    emit("  action           %s" % action)
    emit("  locked           %s" % ("yes, %d MHz" % target if locked else "no"))
    if live:
        emit("  graphics clock   %s MHz (max %s MHz)" % (current, maximum))


def apply_change(args):
    code, out, err = run_smi(args)
    if code != 0:
        message = (err or out or "nvidia-smi failed").splitlines()[0]
        fail("nvidia-smi: %s" % message)


def lock(target):
    apply_change(["--lock-gpu-clocks=%d,%d" % (target, target)])
    with open(MARKER_PATH, "w") as marker:
        marker.write(str(target))
    report_state("locked", True, target, False)


def unlock():
    apply_change(["--reset-gpu-clocks"])
    if os.path.exists(MARKER_PATH):
        os.remove(MARKER_PATH)
    report_state("unlocked", False, 0, False)


def main():
    global REPORT_PATH

    parser = argparse.ArgumentParser(
        description="Toggle a fixed GPU clock for deterministic profiling.")
    parser.add_argument("--clock", type=int, default=BASE_GRAPHICS_CLOCK_MHZ,
                        help="graphics clock in MHz to lock to (default %d)" % BASE_GRAPHICS_CLOCK_MHZ)
    parser.add_argument("--status", action="store_true", help="print state without changing it")
    parser.add_argument("--elevated", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--report", help=argparse.SUPPRESS)
    arguments = parser.parse_args()
    REPORT_PATH = arguments.report

    if arguments.status:
        target = 0
        if os.path.exists(MARKER_PATH):
            with open(MARKER_PATH) as marker:
                target = int(marker.read().strip() or 0)
        report_state("none", target != 0, target, True)
        return

    if not arguments.elevated and not is_admin():
        elevate(arguments)

    if os.path.exists(MARKER_PATH):
        unlock()
    else:
        lock(nearest_supported(arguments.clock))


main()
