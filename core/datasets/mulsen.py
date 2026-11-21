"""MulSen-AD adapter (Li et al., CVPR 2025) — synchronised RGB, infrared and 3D (STL) samples.

Verified layout (official README tree, archive listing and the authors' ``dataset.py``)::

    <cls>/RGB/train/<i>.png                      <cls>/RGB/test/{good,<type>}/<i>.png
    <cls>/RGB/GT/<type>/<i>.png                  (mask only if anomalous *in RGB*)
    <cls>/RGB/GT/{good,<type>}/data.csv          (object,RGB,infrared,pointcloud -> 0/1 per modality)
    <cls>/Infrared/{train,test/<type>}/<i>.png   <cls>/Infrared/GT/<type>/<i>.png
    <cls>/Pointcloud/{train,test/<type>}/<i>.stl <cls>/Pointcloud/GT/<type>/<i>.txt  (xyz of anomalous points)

Notes: many ``.png`` files are BMP-encoded (decoded by content, not by extension); infrared images
are lock-in thermography *visualisations* (8-bit grey), not calibrated temperatures; samples are
paired across modalities by (split, type folder, integer file stem).
"""

from __future__ import annotations

import csv
from pathlib import Path

from core.datasets.base import BAD, GOOD, DatasetAdapter, DatasetIndex, Sample

MODS = {"rgb": ("RGB", ".png"), "infrared": ("Infrared", ".png"), "pointcloud": ("Pointcloud", ".stl")}


def _num(p: Path) -> int:
    try:
        return int(p.stem)
    except ValueError:
        return 10 ** 9


def _read_csv(f: Path) -> dict[int, dict[str, int]]:
    out: dict[int, dict[str, int]] = {}
    if not f.exists():
        return out
    with f.open(newline="", encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            try:
                k = int(row.get("object", "").strip())
            except ValueError:
                continue
            out[k] = {m: int(float(row.get(col, 0) or 0)) for m, col in
                      (("rgb", "RGB"), ("infrared", "infrared"), ("pointcloud", "pointcloud"))}
    return out


class MulSenADAdapter(DatasetAdapter):
    dataset_id = "D06"
    folder_name = "mulsen_ad"
    modalities = ("rgb", "infrared", "pointcloud")

    def list_categories(self, root: Path) -> list[str]:
        if not root.is_dir():
            return []
        return sorted(d.name for d in root.iterdir() if d.is_dir() and (d / "RGB").is_dir())

    def index(self, root: Path, category: str) -> DatasetIndex:
        cat = root / category
        samples: list[Sample] = []
        warnings: list[str] = []
        groups: list[tuple[str, str | None]] = [("train", None)]
        test_dir = cat / "RGB" / "test"
        if test_dir.is_dir():
            groups += [("test", d.name) for d in sorted(test_dir.iterdir()) if d.is_dir()]
        for split, typ in groups:
            files: dict[str, dict[int, Path]] = {}
            for m, (folder, ext) in MODS.items():
                d = cat / folder / split if typ is None else cat / folder / split / typ
                files[m] = {_num(p): p for p in d.glob(f"*{ext}")} if d.is_dir() else {}
            ids = sorted(set().union(*[set(v) for v in files.values()]))
            per_mod = _read_csv(cat / "RGB" / "GT" / (typ or "good") / "data.csv") if typ else {}
            for i in ids:
                mods = {m: files[m][i] for m in MODS if i in files[m]}
                missing = [m for m in MODS if i not in files[m]]
                if missing:
                    warnings.append(f"{split}/{typ or 'train'}/{i}: missing modality {missing}")
                label = GOOD if (typ in (None, "good")) else BAD
                masks: dict = {}
                if label == BAD:
                    for m, (folder, ext) in MODS.items():
                        gext = ".txt" if m == "pointcloud" else ".png"
                        g = cat / folder / "GT" / typ / f"{i}{gext}"
                        if g.exists():
                            masks[m] = g if m == "pointcloud" else [g]
                mod_labels = per_mod.get(i)
                samples.append(Sample(
                    sample_id=f"{split}/{typ or 'good'}/{i}", dataset_id=self.dataset_id, category=category,
                    split=split, label=label, defect_type=None if label == GOOD else typ, modalities=mods,
                    masks=masks,
                    meta={"modality_labels": mod_labels or ({} if label == GOOD else "unknown"),
                          "infrared_note": "lock-in thermography visualisation (8-bit); not calibrated temperature",
                          "pointcloud_units": "as stored in STL (sensor: Creaform MetraSCAN 750)",
                          "units": "mm (per paper; not verified from file headers)"}))
        return DatasetIndex(self.dataset_id, category, root, samples, warnings)
