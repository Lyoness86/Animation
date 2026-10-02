"""Check whether this PC can run Character Studio's local image model.

Uses only the Python standard library, so it works before setup.bat has
installed anything. Prints a plain-English report and saves it to
pc_report.txt next to this file.
"""
import ctypes
import os
import platform
import re
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


# AMD cards that AMD's PyTorch for Windows supports: RX 7600 and up (RDNA3), RX 9000 (RDNA4).
AMD_SUPPORTED = re.compile(r"RX\s*(7[6-9]\d\d|9\d\d\d)", re.IGNORECASE)


def amd_gpus():
    """[(name, vram_gb)] for AMD Radeon cards. The registry has the real memory size
    (Windows' usual video-card query stops counting at 4 GB)."""
    if os.name != "nt":
        return []
    import winreg
    path = r"SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}"
    found = {}
    try:
        base = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, path)
    except OSError:
        return []
    for i in range(100):
        try:
            sub = winreg.EnumKey(base, i)
        except OSError:
            break
        if not sub.isdigit():
            continue
        try:
            with winreg.OpenKey(base, sub) as key:
                name = str(winreg.QueryValueEx(key, "DriverDesc")[0])
                try:
                    mem = winreg.QueryValueEx(key, "HardwareInformation.qwMemorySize")[0]
                except OSError:
                    mem = 0
        except OSError:
            continue
        if isinstance(mem, bytes):
            mem = int.from_bytes(mem, "little")
        if "AMD" in name.upper() or "RADEON" in name.upper():
            found[name] = max(found.get(name, 0), (mem or 0) / 1024 ** 3)
    return list(found.items())


def assess(gpus, ram, disk_free):
    """gpus: [(name, vram_gb, vendor)] with vendor "NVIDIA" or "AMD". Returns (verdict, lines)."""
    lines = []
    usable = [g for g in gpus if g[2] == "NVIDIA" or AMD_SUPPORTED.search(g[0])]
    if not usable:
        why = ("This AMD card is too old for AMD's AI software on Windows (it needs an"
               " RX 7600 or newer)." if any(g[2] == "AMD" for g in gpus)
               else "No suitable graphics card was found.")
        return ("NOT PRACTICAL", [
            why, "The local model needs an NVIDIA card or a recent AMD Radeon. On the",
            "processor alone one picture would take 10-30+ minutes, which is not usable."])
    name, vram, vendor = max(usable, key=lambda g: g[1])
    old = vendor == "NVIDIA" and any(s in name.upper()
                                     for s in ("GTX", "RTX 20", "QUADRO P", "TITAN X"))
    if vram >= 16:
        verdict, speed = "EXCELLENT", "about 5-20 seconds per picture"
    elif vram >= 12:
        verdict, speed = "GOOD", "about 15-45 seconds per picture"
    elif vram >= 8:
        verdict, speed = "POSSIBLE BUT SLOW", "about 1-4 minutes per picture"
    else:
        verdict, speed = "NOT PRACTICAL", "too little graphics memory"
    if vendor == "AMD" and verdict != "NOT PRACTICAL":
        if verdict == "EXCELLENT":
            verdict, speed = "GOOD", "about 20-60 seconds per picture"
        speed += " (AMD on Windows is newer and less tested than NVIDIA)"
    lines.append(f"Graphics card: {name} with {vram:.0f} GB -> {speed}.")
    if old and verdict != "NOT PRACTICAL":
        lines.append("This is an older card generation; expect it to be slower than the estimate.")
    if vendor == "AMD":
        lines.append("AMD needs Python 3.12 and an up-to-date AMD Adrenalin driver.")
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
    nvidia = nvidia_gpus()
    amd = amd_gpus()
    ram = ram_gb()
    disk_free = shutil.disk_usage(HERE).free / 1024 ** 3
    verdict, lines = assess([(n, v, "NVIDIA") for n, v, _ in nvidia] +
                            [(n, v, "AMD") for n, v in amd], ram, disk_free)

    report = ["Character Studio - PC check", "=" * 40,
              f"Windows / OS : {platform.platform()}",
              f"Processor    : {platform.processor() or 'unknown'} ({os.cpu_count()} threads)",
              f"RAM          : {ram:.1f} GB",
              f"Free disk    : {disk_free:.0f} GB (drive holding this folder)"]
    for name, vram, driver in nvidia:
        report.append(f"NVIDIA GPU   : {name}, {vram:.1f} GB VRAM, driver {driver}")
    for name, vram in amd:
        report.append(f"AMD GPU      : {name}, {vram:.1f} GB VRAM")
    if not nvidia and not amd:
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
