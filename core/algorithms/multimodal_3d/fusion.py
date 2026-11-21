"""Group G — educational multimodal experiment: per-modality detectors + late score fusion.

IDD Lab EDUCATIONAL protocol (not the MulSen-TripleAD method of the MulSen-AD paper):

* RGB and infrared: an image method (default PatchCore) fitted separately per modality on
  ``train`` normal images;
* 3D: :class:`PointCloudDeviation` against merged normal reference clouds;
* each modality's raw score is standardised with mean/std measured on HELD-OUT normal training
  samples (never test data);
* late fusion: mean and max of the standardised scores.
Evaluation reports AUROC per modality against (a) the object-level label and (b) the
per-modality label from MulSen-AD's ``data.csv`` when present.
"""

from __future__ import annotations

import random

import numpy as np

from core.algorithms import registry as areg
from core.algorithms.base import AnomalyMethod
from core.algorithms.multimodal_3d.geometry import PointCloudDeviation
from core.datasets.base import BAD, GOOD, DatasetIndex
from core.datasets.io import read_point_cloud, read_rgb8
from core.evaluation import metrics as M

IMAGE_MODS = ("rgb", "infrared")


class MultimodalFusion:
    """Registry handle for the multimodal experiment (run from the 3D & Multimodal Lab)."""
    method_id = "multimodal_fusion"
    requires_fit = True
    requires_reference = False

    @classmethod
    def availability(cls):
        from core.algorithms.torch_common import have
        return have("open3d", "anomalib", "torch")

    @classmethod
    def param_specs(cls):
        return []


def _cloud(path, max_points=200_000, seed=0):
    pts, _ = read_point_cloud(path)
    pts = np.unique(pts, axis=0)
    if len(pts) > max_points:
        pts = pts[np.random.default_rng(seed).choice(len(pts), max_points, replace=False)]
    return pts


def run(index: DatasetIndex, image_method: str = "patchcore", n_fit: int = 40, n_ref_clouds: int = 3,
        max_test: int | None = None, modalities=("rgb", "infrared", "pointcloud"), seed: int = 42,
        image_params: dict | None = None, progress=None, cancel=None) -> dict:
    from core.inference.session import seed_everything
    seed_everything(seed)
    rng = random.Random(seed)
    train = [s for s in index.samples if s.split == "train" and s.label == GOOD and all(m in s.modalities for m in modalities)]
    rng.shuffle(train)
    n_hold = max(3, int(0.15 * len(train)))
    hold, fit = train[:n_hold], train[n_hold:n_hold + n_fit]
    test = [s for s in index.samples if s.split == "test" and s.label in (GOOD, BAD) and all(m in s.modalities for m in modalities)]
    if max_test and len(test) > max_test:
        test = rng.sample(test, max_test)
    steps = len(modalities) * 2 + 1
    k = 0

    def tick(m):
        nonlocal k
        k += 1
        if progress:
            progress(k / steps, m)

    detectors = {}
    for m in modalities:
        AnomalyMethod._check_cancel(cancel)
        if m in IMAGE_MODS:
            det = areg.create(image_method, **(image_params or {}))
            det.fit([read_rgb8(s.modalities[m]) for s in fit], cancel=cancel)
        else:
            det = PointCloudDeviation()
            det.fit([_cloud(s.modalities[m], seed=seed) for s in fit[:n_ref_clouds]])
        detectors[m] = det
        tick(f"fitted {m} detector")

    def score(det, m, s):
        if m in IMAGE_MODS:
            return det.predict(read_rgb8(s.modalities[m])).score
        return det.predict(_cloud(s.modalities[m], seed=seed)).score

    norm = {}
    for m in modalities:
        vals = [score(detectors[m], m, s) for s in hold]
        mu, sd = float(np.mean(vals)), float(np.std(vals) + 1e-8)
        norm[m] = (mu, sd)
        tick(f"standardised {m} on {len(hold)} held-out normals")

    rows = []
    for i, s in enumerate(test):
        AnomalyMethod._check_cancel(cancel)
        r = {"sample_id": s.sample_id, "label": 1 if s.label == BAD else 0, "defect_type": s.defect_type}
        ml = s.meta.get("modality_labels") if isinstance(s.meta.get("modality_labels"), dict) else {}
        for m in modalities:
            raw = score(detectors[m], m, s)
            r[f"{m}_raw"] = raw
            r[f"{m}_z"] = (raw - norm[m][0]) / norm[m][1]
            r[f"{m}_label"] = ml.get(m, 0 if s.label == GOOD else None)
        zs = [r[f"{m}_z"] for m in modalities]
        r["fused_mean"] = float(np.mean(zs))
        r["fused_max"] = float(np.max(zs))
        rows.append(r)
        if progress:
            progress((steps - 1 + (i + 1) / len(test)) / steps, f"testing {i + 1}/{len(test)}")

    y = [r["label"] for r in rows]
    summary = {"object_level": {}, "modality_level": {}}
    for key in [f"{m}_z" for m in modalities] + ["fused_mean", "fused_max"]:
        summary["object_level"][key] = M.image_auroc(y, [r[key] for r in rows])
    for m in modalities:
        pairs = [(r[f"{m}_label"], r[f"{m}_z"]) for r in rows if r[f"{m}_label"] is not None]
        if pairs:
            summary["modality_level"][m] = M.image_auroc([p[0] for p in pairs], [p[1] for p in pairs])
    return {"rows": rows, "auroc": summary, "n_fit": len(fit), "n_holdout": len(hold), "n_test": len(test),
            "image_method": image_method, "normalisation": norm,
            "protocol": (f"IDD Lab educational multimodal protocol: {image_method} per image modality fitted on "
                         f"{len(fit)} train/good samples; 3D deviation vs {min(n_ref_clouds, len(fit))} merged normal "
                         f"clouds; z-normalisation on {len(hold)} held-out train/good samples; late fusion (mean, max); "
                         f"{len(test)} test samples; seed {seed}. Not the MulSen-AD official benchmark method."),
            "result_category": "local_educational"}
