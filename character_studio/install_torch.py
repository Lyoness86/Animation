"""Install the PyTorch build that matches this PC's NVIDIA card (used by setup.bat).

PyTorch's newest builds (CUDA 12.8+) no longer run on GTX 10xx (Pascal) cards,
and the CUDA 12.6 builds don't know RTX 50xx cards, so pick per card.
"""
import subprocess
import sys

from check_pc import nvidia_gpus, run

OLD = ("https://download.pytorch.org/whl/cu126", "torch<2.15")   # GTX 900/10xx .. RTX 40xx
NEW = ("https://download.pytorch.org/whl/cu128", "torch")        # RTX 50xx


def pick():
    caps = run(["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"]).split()
    try:
        if caps and max(float(c) for c in caps) >= 10:
            return NEW
    except ValueError:
        pass
    if any("RTX 50" in name.upper() for name, _, _ in nvidia_gpus()):
        return NEW
    return OLD


if __name__ == "__main__":
    index, spec = pick()
    print(f"Installing {spec} from {index}")
    sys.exit(subprocess.call([sys.executable, "-m", "pip", "install", spec, "--index-url", index]))
