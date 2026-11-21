"""Shared helpers for PyTorch-based adapters (device, preprocessing, model cache location)."""

from __future__ import annotations

import importlib.util
import os

import cv2
import numpy as np

from core import paths, settings

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def configure_model_cache() -> None:
    """Keep all downloaded weights inside the git-ignored ``checkpoints/`` folder."""
    os.environ.setdefault("TORCH_HOME", str(paths.CHECKPOINTS / "torch"))
    os.environ.setdefault("HF_HOME", str(paths.CHECKPOINTS / "huggingface"))


def have(*modules: str) -> tuple[bool, str]:
    missing = [m for m in modules if importlib.util.find_spec(m) is None]
    return (not missing, "ok" if not missing else f"Missing Python package(s): {', '.join(missing)}")


def device() -> str:
    import torch
    pref = settings.get().device
    if pref in ("auto", "cuda") and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def to_tensor(images: list[np.ndarray], size: tuple[int, int], mean=IMAGENET_MEAN, std=IMAGENET_STD,
              interpolation=cv2.INTER_AREA):
    """List of RGB uint8 images -> normalised float tensor (N,3,H,W) at ``size`` (h, w).

    A plain resize (no centre crop) keeps the anomaly map geometrically aligned with the full image
    and its ground-truth mask.
    """
    import torch
    h, w = size
    arr = np.stack([cv2.resize(im, (w, h), interpolation=interpolation) for im in images]).astype(np.float32) / 255.0
    arr = (arr - np.asarray(mean, np.float32)) / np.asarray(std, np.float32)
    return torch.from_numpy(arr.transpose(0, 3, 1, 2).copy())


def upsample_map(m: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    return cv2.resize(m.astype(np.float32), (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)


def anomalib_version() -> str:
    try:
        import anomalib
        return anomalib.__version__
    except Exception:
        return "?"
