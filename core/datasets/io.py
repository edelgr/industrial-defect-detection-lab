"""Robust image / point-cloud readers (Unicode-safe on Windows, 8/16-bit, TIFF)."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

IMAGE_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
POINTCLOUD_EXT = {".pcd", ".ply", ".xyz", ".txt", ".npy", ".stl", ".obj"}


def read_image_raw(path: str | Path) -> np.ndarray:
    """Read an image exactly as stored (dtype and channel count preserved)."""
    path = Path(path)
    buf = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
    if img is None:
        if path.suffix.lower() in (".tif", ".tiff"):
            import tifffile  # anomalib dependency; handles float TIFFs OpenCV cannot
            return np.asarray(tifffile.imread(str(path)))
        raise IOError(f"Cannot decode image: {path}")
    if img.ndim == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    elif img.ndim == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    return img


def to_display_rgb8(img: np.ndarray) -> np.ndarray:
    """Convert any image (gray/RGB, uint8/uint16/float) to RGB uint8 for display/algorithms.

    For non-8-bit data a min–max stretch is applied; this is a *visualisation* normalisation and
    must not be interpreted as a calibrated physical quantity.
    """
    a = img
    if a.dtype != np.uint8:
        a = a.astype(np.float32)
        finite = np.isfinite(a)
        lo, hi = (float(a[finite].min()), float(a[finite].max())) if finite.any() else (0.0, 1.0)
        a = np.zeros_like(a) if hi <= lo else (a - lo) / (hi - lo)
        a = np.nan_to_num(a)
        a = (a * 255).clip(0, 255).astype(np.uint8)
    if a.ndim == 2:
        a = np.stack([a] * 3, axis=-1)
    elif a.shape[2] == 1:
        a = np.repeat(a, 3, axis=2)
    return np.ascontiguousarray(a[..., :3])


def read_rgb8(path: str | Path) -> np.ndarray:
    return to_display_rgb8(read_image_raw(path))


def read_mask(path: str | Path | list, shape: tuple[int, int] | None = None) -> np.ndarray:
    """Read one or several binary masks and return a single boolean array (union)."""
    paths_ = path if isinstance(path, list) else [path]
    out = None
    for p in paths_:
        m = read_image_raw(p)
        if m.ndim == 3:
            m = m.max(axis=2)
        m = m > 0
        out = m if out is None else (out | m)
    if out is None:
        raise IOError("no mask paths")
    if shape is not None and out.shape != shape:
        out = cv2.resize(out.astype(np.uint8), (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST) > 0
    return out


def write_image(path: str | Path, rgb: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR) if rgb.ndim == 3 else rgb
    ok, buf = cv2.imencode(path.suffix or ".png", arr)
    if not ok:
        raise IOError(f"Cannot encode {path}")
    buf.tofile(str(path))


def read_point_cloud(path: str | Path) -> tuple[np.ndarray, np.ndarray | None]:
    """Return (N×3 float64 points, optional N×3 float colours in [0,1]).

    Uses Open3D when available (PCD/PLY/STL/OBJ); plain-text XYZ files are parsed with NumPy.
    """
    path = Path(path)
    ext = path.suffix.lower()
    if ext == ".npy":
        a = np.load(path)
        return a[:, :3].astype(np.float64), None
    if ext in (".txt", ".xyz"):
        a = np.loadtxt(path, dtype=np.float64, ndmin=2)
        return a[:, :3], None
    try:
        import open3d as o3d
    except ImportError as e:  # pragma: no cover - depends on optional install
        raise IOError(f"Open3D is required to read {ext} files ({e})") from e
    if ext in (".stl", ".obj"):
        mesh = o3d.io.read_triangle_mesh(str(path))
        pts = np.asarray(mesh.vertices, dtype=np.float64)
        return pts, None
    pcd = o3d.io.read_point_cloud(str(path))
    pts = np.asarray(pcd.points, dtype=np.float64)
    cols = np.asarray(pcd.colors, dtype=np.float64) if pcd.has_colors() else None
    if pts.size == 0:
        raise IOError(f"Empty or unreadable point cloud: {path}")
    return pts, cols
