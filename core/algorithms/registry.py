"""Algorithm registry: maps catalog method ids to runnable IDD Lab adapters.

Heavy adapters are referenced by dotted path and imported lazily so that the GUI starts quickly
and keeps working when optional dependencies (torch, anomalib, open3d ...) are missing.
"""

from __future__ import annotations

import os

import importlib

from core.algorithms.base import AnomalyMethod

RUNTIMES: dict[str, str] = {
    # Group A — classical
    "otsu": "core.algorithms.classical.methods:OtsuThreshold",
    "adaptive_threshold": "core.algorithms.classical.methods:AdaptiveThreshold",
    "canny": "core.algorithms.classical.methods:CannyEdges",
    "gabor": "core.algorithms.classical.methods:GaborTexture",
    "morphology": "core.algorithms.classical.methods:MorphologyTopHat",
    "reference_difference": "core.algorithms.classical.methods:ReferenceDifference",
    "ssim": "core.algorithms.classical.methods:SSIMComparison",
    # Group D — feature distributions / memory banks
    "padim": "core.algorithms.memory_based.anomalib_models:PadimMethod",
    "patchcore": "core.algorithms.memory_based.anomalib_models:PatchcoreMethod",
    # Group E — teacher-student
    "efficientad": "core.algorithms.teacher_student.efficientad:EfficientADMethod",
    # Group F — foundation-model based
    "anomalydino": "core.algorithms.foundation.anomalydino:AnomalyDINOMethod",
    "winclip": "core.algorithms.foundation.winclip:WinCLIPMethod",
    # Group G — 3D
    "pc_deviation": "core.algorithms.multimodal_3d.geometry:PointCloudDeviation",
    "multimodal_fusion": "core.algorithms.multimodal_3d.fusion:MultimodalFusion",
}
NON_IMAGE = {"pc_deviation", "multimodal_fusion"}

_cache: dict[str, type[AnomalyMethod] | Exception] = {}


def runtime_class(method_id: str) -> type[AnomalyMethod] | None:
    """Return the adapter class or None (missing runtime or import failure)."""
    path = RUNTIMES.get(method_id)
    if path is None:
        return None
    if method_id not in _cache:
        mod, cls = path.split(":")
        try:
            _cache[method_id] = getattr(importlib.import_module(mod), cls)
        except Exception as e:  # import errors are surfaced via availability()
            _cache[method_id] = e
    v = _cache[method_id]
    return None if isinstance(v, Exception) else v


def availability(method_id: str) -> tuple[bool, str]:
    if method_id not in RUNTIMES:
        return False, "No runnable implementation in IDD Lab (educational explanation only)."
    cls = runtime_class(method_id)
    if cls is None:
        err = _cache.get(method_id)
        return False, f"Implementation could not be imported: {err}"
    ok, why = cls.availability()
    if ok and os.environ.get("IDD_OFFLINE_WEIGHTS") == "1":
        # offline / CI mode: never trigger a weight download — report missing pretrained weights instead
        from core.model_manager import WEIGHTS
        if method_id in WEIGHTS:
            if method_id == "efficientad":
                from core.algorithms.teacher_student.efficientad import assets_present
                cached = assets_present()
            else:
                from core.system.readiness import weights_cached
                cached = bool(weights_cached(method_id))
            if not cached:
                return False, "pretrained weights not cached and downloads are disabled (IDD_OFFLINE_WEIGHTS=1)"
    return ok, why


def create(method_id: str, **params) -> AnomalyMethod:
    cls = runtime_class(method_id)
    if cls is None:
        raise KeyError(availability(method_id)[1])
    ok, why = cls.availability()
    if not ok:
        raise RuntimeError(why)
    return cls(**params)


def image_methods() -> list[str]:
    return [m for m in RUNTIMES if m not in NON_IMAGE]
