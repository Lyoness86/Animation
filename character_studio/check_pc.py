"""Check whether this PC can run Character Studio's local image model.

Uses only the Python standard library, so it works before setup.bat has
installed anything. Prints a plain-English report and saves it to
pc_report.txt next to this file.
"""
import ctypes
import os
import platform
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DISK_NEEDED_GB = 30
NO_WINDOW = 0x08000000 if os.name == "nt" else 0


def run(cmd):
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=30,
                             creationflags=NO_WINDOW)
        return out.stdout.strip() if out.returncode == 0 else ""
    except (OSError, subprocess.SubprocessError):
        return ""


def ram_gb():
    if os.name == "nt":
        class MemStatus(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
        stat = MemStatus()
        stat.dwLength = ctypes.sizeof(MemStatus)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
        return stat.ullTotalPhys / 1024 ** 3
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1024 ** 3
    except (ValueError, OSError):
        return 0.0


def nvidia_gpus():
    """[(name, vram_gb, driver)] from nvidia-smi, or [] if there is no NVIDIA GPU."""
    out = run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
               "--format=csv,noheader,nounits"])
    gpus = []
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 3:
            try:
                gpus.append((parts[0], float(parts[1]) / 1024, parts[2]))
            except ValueError:
                pass
    return gpus


def other_gpus():
    if os.name != "nt":
        return []
    out = run(["powershell", "-NoProfile", "-Command",
               "(Get-CimInstance Win32_VideoController).Name"])
    return [line.strip() for line in out.splitlines() if line.strip()]


def assess(gpus, ram, disk_free):
    """Return (verdict, explanation lines)."""
    lines = []
    if not gpus:
        return ("NOT PRACTICAL", [
            "No NVIDIA graphics card was found. The local model needs an NVIDIA",
            "card (CUDA). On the processor alone one picture would take",
            "10-30+ minutes, which is not usable."])
    name, vram, _ = max(gpus, key=lambda g: g[1])
    old = any(s in name.upper() for s in ("GTX", "RTX 20", "QUADRO P", "TITAN X"))
    if vram >= 16:
        verdict, speed = "EXCELLENT", "about 5-20 seconds per picture"
    elif vram >= 12:
        verdict, speed = "GOOD", "about 15-45 seconds per picture"
    elif vram >= 8:
        verdict, speed = "POSSIBLE BUT SLOW", "about 1-4 minutes per picture"
    else:
        verdict, speed = "NOT PRACTICAL", "too little graphics memory"
    lines.append(f"Graphics card: {name} with {vram:.0f} GB -> {speed}.")
    if old and verdict != "NOT PRACTICAL":
        lines.append("This is an older card generation; expect it to be slower than the estimate.")
    if ram < 16:
        verdict = "NOT PRACTICAL"
        lines.append(f"Only {ram:.0f} GB of RAM: at least 16 GB is needed, 32 GB recommended.")
    elif ram < 30:
        lines.append(f"{ram:.0f} GB of RAM works, but close other programs while generating"
                     " (32 GB is recommended).")
    else:
        lines.append(f"{ram:.0f} GB of RAM: fine.")
    if disk_free < DISK_NEEDED_GB:
        lines.append(f"Only {disk_free:.0f} GB free on this drive: about {DISK_NEEDED_GB} GB is needed."
                     " Free some space or move this folder to another drive.")
        if verdict != "NOT PRACTICAL":
            verdict += " (after freeing disk space)"
    else:
        lines.append(f"{disk_free:.0f} GB free disk space: fine.")
    return verdict, lines


def main():
    gpus = nvidia_gpus()
    ram = ram_gb()
    disk_free = shutil.disk_usage(HERE).free / 1024 ** 3
    verdict, lines = assess(gpus, ram, disk_free)

    report = ["Character Studio - PC check", "=" * 40,
              f"Windows / OS : {platform.platform()}",
              f"Processor    : {platform.processor() or 'unknown'} ({os.cpu_count()} threads)",
              f"RAM          : {ram:.1f} GB",
              f"Free disk    : {disk_free:.0f} GB (drive holding this folder)"]
    for name, vram, driver in gpus:
        report.append(f"NVIDIA GPU   : {name}, {vram:.1f} GB VRAM, driver {driver}")
    if not gpus:
        for name in other_gpus() or ["(none detected)"]:
            report.append(f"Graphics     : {name}")
    report += ["", f"VERDICT: {verdict}", *("  " + line for line in lines), "",
               "Please send this whole text to Claude."]
    text = "\n".join(report)
    print(text)
    try:
        with open(os.path.join(HERE, "pc_report.txt"), "w", encoding="utf-8") as f:
            f.write(text + "\n")
        print(f"\n(Also saved to {os.path.join(HERE, 'pc_report.txt')})")
    except OSError:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
