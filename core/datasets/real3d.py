"""Real3D-AD adapter (Liu et al., NeurIPS 2023 D&B) + point-label helpers.

Verified layout (README + official ``dataset_pc.py``)::

    <cls>/train/*.pcd            4 normal prototypes per class ("*template*" or "*prototype*")
    <cls>/test/*.pcd             e.g. 1_bulge.pcd, 2_sink.pcd, 3_good.pcd ("temp" files skipped)
    <cls>/gt/<stem>.txt          full cloud, space-delimited "x y z label" (label 0/1)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from core.datasets.base import BAD, GOOD, DatasetAdapter, DatasetIndex, Sample


def read_point_labels(path: Path, points: np.ndarray, tol: float | None = None) -> np.ndarray | None:
    """Per-point 0/1 labels for ``points`` from a GT text file.

    * 4 columns (x y z label), same length as the cloud -> labels used directly (Real3D-AD);
    * otherwise the file lists anomalous xyz points (MulSen-AD) or labelled points of a different
      sampling -> nearest-neighbour transfer with a KD-tree.
    """
    from scipy.spatial import cKDTree
    path = Path(path)
    txt = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    rows = [ln.replace(",", " ").split() for ln in txt if ln.strip() and not ln.lstrip().startswith("#")]
    if not rows:
        return np.zeros(len(points), np.float32)
    try:
        arr = np.asarray([[float(v) for v in r[:4]] for r in rows], dtype=np.float64)
    except ValueError:
        return None
    if arr.shape[1] >= 4:
        if len(arr) == len(points):
            return arr[:, 3].astype(np.float32)
        tree = cKDTree(arr[:, :3])
        _, j = tree.query(points)
        return arr[j, 3].astype(np.float32)
    # anomalous points only -> mark nearest cloud points
    lab = np.zeros(len(points), np.float32)
    tree = cKDTree(points)
    d, j = tree.query(arr[:, :3])
    if tol is not None:
        j = j[d <= tol]
    lab[j] = 1.0
    return lab


class Real3DADAdapter(DatasetAdapter):
    dataset_id = "D04"
    folder_name = "real3d_ad"
    modalities = ("pointcloud",)

    def list_categories(self, root: Path) -> list[str]:
        if not root.is_dir():
            return []
        return sorted(d.name for d in root.iterdir() if d.is_dir() and (d / "train").is_dir() and (d / "test").is_dir())

    def index(self, root: Path, category: str) -> DatasetIndex:
        cat = root / category
        samples: list[Sample] = []
        warnings: list[str] = []
        for f in sorted((cat / "train").glob("*.pcd")):
            samples.append(Sample(f"train/{f.stem}", self.dataset_id, category, "train", GOOD, None,
                                  {"pointcloud": f}, meta={"role": "normal prototype"}))
        for f in sorted((cat / "test").glob("*.pcd")):
            if "temp" in f.stem:
                continue
            good = "good" in f.stem
            typ = None if good else (f.stem.split("_", 1)[1] if "_" in f.stem else "defect")
            gt = cat / "gt" / f"{f.stem}.txt"
            if not good and not gt.exists():
                warnings.append(f"missing GT for {f.name}")
            samples.append(Sample(f"test/{f.stem}", self.dataset_id, category, "test", GOOD if good else BAD, typ,
                                  {"pointcloud": f}, masks={"pointcloud": gt} if gt.exists() else {},
                                  meta={"units": "as stored in PCD (not documented)"}))
        return DatasetIndex(self.dataset_id, category, root, samples, warnings)
