"""Install the PyTorch build that matches this PC's graphics card (used by setup.bat).

NVIDIA: PyTorch's newest builds (CUDA 12.8+) no longer run on GTX 10xx (Pascal)
cards, and the CUDA 12.6 builds don't know RTX 50xx cards, so pick per card.
AMD: AMD's own PyTorch for Windows (ROCm), RX 7600 and newer, Python 3.12 only.
"""
import subprocess
import sys

from check_pc import AMD_SUPPORTED, amd_gpus, nvidia_gpus, run

OLD = ("https://download.pytorch.org/whl/cu126", "torch<2.15")   # GTX 900/10xx .. RTX 40xx
NEW = ("https://download.pytorch.org/whl/cu128", "torch")        # RTX 50xx

# From AMD's "PyTorch via PIP installation" page for Windows (ROCm 7.2.1).
AMD_REPO = "https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/"
AMD_SDK = ["rocm_sdk_core-7.2.1-py3-none-win_amd64.whl",
           "rocm_sdk_devel-7.2.1-py3-none-win_amd64.whl",
           "rocm_sdk_libraries_custom-7.2.1-py3-none-win_amd64.whl",
           "rocm-7.2.1.tar.gz"]
# torch, torchvision and torchaudio go in one command so pip can't swap in the CPU-only torch
AMD_TORCH = ["torch-2.9.1%2Brocm7.2.1-cp312-cp312-win_amd64.whl",
             "torchaudio-2.9.1%2Brocm7.2.1-cp312-cp312-win_amd64.whl",
             "torchvision-0.24.1%2Brocm7.2.1-cp312-cp312-win_amd64.whl"]


def pick_nvidia():
    caps = run(["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"]).split()
    try:
        if caps and max(float(c) for c in caps) >= 10:
            return NEW
    except ValueError:
        pass
    if any("RTX 50" in name.upper() for name, _, _ in nvidia_gpus()):
        return NEW
    return OLD


def pip(*args):
    return subprocess.call([sys.executable, "-m", "pip", "install", "--no-cache-dir", *args])


def install_amd():
    if sys.version_info[:2] != (3, 12):
        print("AMD graphics cards need Python 3.12 exactly (this is Python "
              f"{sys.version_info[0]}.{sys.version_info[1]}).\n"
              "Install Python 3.12 from https://www.python.org/downloads/ , delete the .venv "
              "folder here and run setup.bat again.")
        return 1
    print("Installing AMD's PyTorch for Windows (ROCm)...")
    return pip(*[AMD_REPO + f for f in AMD_SDK]) or pip(*[AMD_REPO + f for f in AMD_TORCH])


def main():
    if nvidia_gpus():
        index, spec = pick_nvidia()
        print(f"NVIDIA card: installing {spec} from {index}")
        return pip(spec, "--index-url", index)
    if any(AMD_SUPPORTED.search(name) for name, _ in amd_gpus()):
        return install_amd()
    print("No supported graphics card found (needs NVIDIA, or AMD Radeon RX 7600 or newer).")
    return 1


if __name__ == "__main__":
    sys.exit(main())
