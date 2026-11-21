"""Common interface for every runnable method in IDD Lab.

Output contract (checked by ``tests/test_algorithms_contract.py``):

* ``anomaly_map``  – H×W float32 array aligned with the input image, *raw* method-specific
  scores (higher = more anomalous). It is NOT a probability.
* ``score``        – scalar raw image-level score (higher = more anomalous) or ``None``.
* ``mask``         – optional H×W bool prediction obtained by thresholding ``anomaly_map``.
* ``visual``       – optional RGB uint8 image with a method-native visualisation (e.g. edges).
* ``score_semantics`` – human-readable description of what the numbers mean.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

ProgressFn = Callable[[float, str], None]   # (fraction 0..1, message)
CancelFn = Callable[[], bool]


class CancelledError(RuntimeError):
    pass


@dataclass
class ParamSpec:
    name: str
    kind: str                    # "int" | "float" | "choice" | "bool"
    default: Any
    label: str = ""
    minimum: float | None = None
    maximum: float | None = None
    step: float | None = None
    choices: list | None = None
    help: str = ""
    affects_fit: bool = False    # changing it requires re-fitting the model
    # Parameters added after fitted-model caches existed: leave them out of the cache key while
    # they hold their default value, so earlier caches (fitted with that behaviour) stay valid.
    key_omit_default: bool = False


@dataclass
class MethodResult:
    method_id: str
    anomaly_map: np.ndarray | None
    score: float | None
    mask: np.ndarray | None = None
    threshold: float | None = None
    visual: np.ndarray | None = None
    latency_ms: float = 0.0
    gpu_memory_mb: float | None = None
    device: str = "cpu"
    score_semantics: str = ""
    model_version: str = ""
    extras: dict = field(default_factory=dict)

    def with_threshold(self, thr: float) -> "MethodResult":
        """Re-threshold cached scores without repeating inference."""
        m = None if self.anomaly_map is None else self.anomaly_map >= thr
        return MethodResult(**{**self.__dict__, "mask": m, "threshold": thr})


class AnomalyMethod(ABC):
    method_id: str = ""            # must match research_catalog/algorithms/<id>.yaml
    display_name: str = ""
    requires_fit: bool = False      # learns from normal training images
    requires_reference: bool = False  # compares against one reference (normal) image
    supports_gpu: bool = False
    modality: str = "rgb"

    def __init__(self, **params):
        self.params = {p.name: p.default for p in self.param_specs()}
        self.params.update(params)
        self.fitted = False
        self.fit_info: dict = {}

    # ---------------------------------------------------------------- metadata
    @classmethod
    def param_specs(cls) -> list[ParamSpec]:
        return []

    @classmethod
    def availability(cls) -> tuple[bool, str]:
        """(available, reason).  Must be cheap and must not download anything."""
        return True, "ok"

    def model_version(self) -> str:
        return ""

    # ---------------------------------------------------------------- lifecycle
    def fit(self, images: list[np.ndarray], progress: ProgressFn | None = None,
            cancel: CancelFn | None = None) -> None:
        """Learn normality from *training* images only (never test images)."""
        self.fitted = True

    # ------------------------------------------------------- fitted-state cache
    cacheable: bool = False

    def save_state(self, path) -> None:
        """Persist the fitted state (only for ``cacheable`` methods)."""
        raise NotImplementedError

    def load_state(self, path) -> None:
        raise NotImplementedError

    @abstractmethod
    def _predict(self, image: np.ndarray, reference: np.ndarray | None) -> MethodResult:
        ...

    def predict(self, image: np.ndarray, reference: np.ndarray | None = None) -> MethodResult:
        if self.requires_fit and not self.fitted:
            raise RuntimeError(f"{self.display_name}: fit() must be called on normal training images first")
        if self.requires_reference and reference is None:
            raise ValueError(f"{self.display_name}: a normal reference image is required")
        gpu0 = _gpu_reset() if self.supports_gpu else False   # CPU methods: no GPU figure
        t0 = time.perf_counter()
        res = self._predict(image, reference)
        res.latency_ms = (time.perf_counter() - t0) * 1000.0
        res.gpu_memory_mb = _gpu_peak(gpu0)
        res.model_version = res.model_version or self.model_version()
        if res.anomaly_map is not None:
            res.anomaly_map = np.asarray(res.anomaly_map, dtype=np.float32)
            assert res.anomaly_map.shape == image.shape[:2], "anomaly map must align with the input"
        return res

    # --------------------------------------------------------------- helpers
    @staticmethod
    def _check_cancel(cancel: CancelFn | None) -> None:
        if cancel is not None and cancel():
            raise CancelledError("cancelled by user")


def _gpu_reset() -> bool:
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            return True
    except Exception:
        pass
    return False


def _gpu_peak(active: bool) -> float | None:
    if not active:
        return None
    import torch
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / 2 ** 20


def resolve_device(pref: str = "auto") -> str:
    try:
        import torch
        if pref in ("auto", "cuda") and torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"
