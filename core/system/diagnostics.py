"""Hardware / software diagnostics (GPU, CUDA, VRAM, RAM, disk, library versions)."""

from __future__ import annotations

import importlib
import os
import platform
import shutil
import subprocess
from pathlib import Path

OPTIONAL_LIBS = ["numpy", "cv2", "scipy", "sklearn", "skimage", "torch", "torchvision", "anomalib",
                 "timm", "open_clip", "PySide6", "pyqtgraph", "pyvista", "pyvistaqt", "vtk", "open3d",
                 "matplotlib", "yaml", "jsonschema"]


def library_versions() -> dict[str, str | None]:
    out = {}
    for name in OPTIONAL_LIBS:
        try:
            m = importlib.import_module(name)
            out[name] = str(getattr(m, "__version__", getattr(m, "VTK_VERSION", "installed")))
        except Exception:
            out[name] = None
    return out


def gpu_info() -> dict:
    info: dict = {"torch_cuda_available": False}
    try:
        import torch
        info["torch"] = torch.__version__
        info["torch_cuda_build"] = torch.version.cuda
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            free, total = torch.cuda.mem_get_info(0)
            info.update(torch_cuda_available=True, device_name=p.name,
                        compute_capability=f"{p.major}.{p.minor}",
                        vram_total_gb=round(total / 2 ** 30, 2), vram_free_gb=round(free / 2 ** 30, 2),
                        cudnn=torch.backends.cudnn.version())
    except Exception as e:
        info["torch_error"] = str(e)
    smi = shutil.which("nvidia-smi")
    if smi:
        try:
            r = subprocess.run([smi, "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
                               capture_output=True, text=True, timeout=10)
            info["nvidia_smi"] = r.stdout.strip()
        except Exception:
            pass
    return info


def ram_gb() -> float | None:
    try:
        if os.name == "nt":
            import ctypes

            class MEMSTAT(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
            st = MEMSTAT()
            st.dwLength = ctypes.sizeof(MEMSTAT)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st))
            return round(st.ullTotalPhys / 2 ** 30, 1)
        return round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2 ** 30, 1)
    except Exception:
        return None


def disk_free_gb(path: Path) -> float:
    p = Path(path)
    while not p.exists() and p.parent != p:
        p = p.parent
    return round(shutil.disk_usage(p).free / 2 ** 30, 1)


def full_report(data_root: Path) -> dict:
    return {"platform": platform.platform(), "python": platform.python_version(),
            "cpu": platform.processor(), "ram_gb": ram_gb(), "gpu": gpu_info(),
            "data_root": str(data_root), "data_root_free_gb": disk_free_gb(data_root),
            "libraries": library_versions()}
