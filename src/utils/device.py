"""Device selection and hardware provenance for the report."""

from __future__ import annotations

import platform
import subprocess

import torch


def get_device(prefer: str = "auto") -> torch.device:
    """Pick the best available device. `prefer` may be auto/mps/cuda/cpu."""
    if prefer != "auto":
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def hardware_summary() -> dict:
    """Machine description recorded alongside every run, for the report's
    'training time and hardware used' requirement."""
    info = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "processor": platform.processor(),
    }
    if platform.system() == "Darwin":
        try:
            info["cpu_model"] = subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
            ).strip()
            mem_bytes = int(
                subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True).strip()
            )
            info["ram_gb"] = round(mem_bytes / 1024**3)
        except (subprocess.SubprocessError, OSError, ValueError):
            pass
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
    elif torch.backends.mps.is_available():
        info["gpu"] = "Apple Silicon GPU (Metal / MPS)"
    else:
        info["gpu"] = "none (CPU only)"
    return info
