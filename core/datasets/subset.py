"""Reproducible teaching-subset builder.

A subset keeps the ORIGINAL on-disk layout (so the same adapter can read it), original sample ids,
splits, annotations and modality pairing, and writes ``subset_manifest.json`` describing exactly
how every sample was selected. The official dataset is never modified (files are copied).
Results on subsets are local educational results, never official benchmark results.
"""

from __future__ import annotations

import json
import random
import shutil
import time
from pathlib import Path

import numpy as np

from core.datasets.base import BAD, GOOD, DatasetIndex, Sample
from core.datasets.io import read_mask


def _defect_area(s: Sample) -> float | None:
    m = s.masks.get("rgb")
    if not m:
        return None
    try:
        return float(read_mask(m).mean())
    except Exception:
        return None


def select(index: DatasetIndex, n_train: int = 40, n_test_good: int = 8, per_defect: int = 3,
           n_small_defects: int = 3, n_modality_specific: int = 2, seed: int = 0) -> list[tuple[Sample, str]]:
    """Return (sample, reason) pairs. Deterministic for a given seed."""
    rng = random.Random(seed)
    chosen: dict[str, tuple[Sample, str]] = {}

    def add(samples, k, reason):
        pool = [s for s in samples if s.sample_id not in chosen]
        for s in rng.sample(pool, min(k, len(pool))):
            chosen[s.sample_id] = (s, reason)

    add(index.train_normals(), n_train, "normal training sample")
    add([s for s in index.samples if s.split == "validation" and s.label == GOOD], n_test_good, "normal validation sample")
    add([s for s in index.samples if s.split.startswith("test") and s.label == GOOD], n_test_good, "normal test sample")
    by_type: dict[str, list[Sample]] = {}
    for s in index.samples:
        if s.label == BAD:
            by_type.setdefault(s.defect_type or "defect", []).append(s)
    for t, ss in sorted(by_type.items()):
        add(ss, per_defect, f"anomaly type '{t}'")
    # difficult visual cases: smallest annotated defects
    areas = [(a, s) for s in index.samples if s.label == BAD and s.sample_id not in chosen
             for a in [_defect_area(s)] if a]
    for a, s in sorted(areas, key=lambda t: t[0])[:n_small_defects]:
        chosen[s.sample_id] = (s, f"difficult: very small defect ({100 * a:.3f} % of pixels)")
    # modality-specific anomalies (MulSen-AD per-modality labels)
    for mod in ("infrared", "pointcloud", "rgb"):
        only = [s for s in index.samples if s.label == BAD and isinstance(s.meta.get("modality_labels"), dict)
                and s.meta["modality_labels"].get(mod) == 1
                and sum(s.meta["modality_labels"].values()) == 1]
        add(only, n_modality_specific, f"anomalous only in {mod}")
    return list(chosen.values())


def build(index: DatasetIndex, dest_root: Path, name: str, selection: list[tuple[Sample, str]],
          params: dict, source_url: str | None = None) -> Path:
    """Copy selected samples (all modalities + masks + side files) preserving relative paths."""
    out = dest_root / name
    src_root = index.root
    files: set[Path] = set()
    for s, _ in selection:
        files.update(s.modalities.values())
        for m in s.masks.values():
            files.update(m if isinstance(m, list) else [m])
    # per-type side files such as MulSen-AD data.csv
    for f in (src_root / index.category).rglob("*.csv"):
        files.add(f)
    for f in sorted(files):
        rel = f.relative_to(src_root)
        t = out / rel
        t.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, t)
    manifest = {
        "name": name, "created": time.strftime("%Y-%m-%d %H:%M:%S"), "dataset_id": index.dataset_id,
        "category": index.category, "source_root": str(src_root), "source_url": source_url,
        "selection_parameters": params, "n_samples": len(selection), "n_files": len(files),
        "note": "Teaching subset — results on it are local educational results, not official benchmark results.",
        "samples": [{**s.to_dict(src_root), "selection_reason": why} for s, why in selection],
    }
    (out / "subset_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return out
