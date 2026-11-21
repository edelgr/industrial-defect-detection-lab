"""Dataset adapter interface and the in-memory sample model.

A *dataset adapter* knows the on-disk layout of one research dataset and turns it into a flat,
split-aware list of :class:`Sample` objects.  Adapters never modify the official files.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

GOOD = "good"
BAD = "bad"
UNKNOWN = "unknown"   # e.g. MVTec AD 2 private test images whose labels are withheld


@dataclass
class Sample:
    sample_id: str                      # unique within (dataset, category); derived from the original path
    dataset_id: str
    category: str
    split: str                          # train | validation | test | test_public | test_private ...
    label: str                          # good | bad | unknown
    defect_type: str | None             # original defect-type folder name, if any
    modalities: dict[str, Path]         # e.g. {"rgb": ..., "infrared": ..., "pointcloud": ...}
    masks: dict[str, Path | list[Path]] = field(default_factory=dict)   # GT per modality
    meta: dict = field(default_factory=dict)

    @property
    def primary_path(self) -> Path:
        for key in ("rgb", "image", "infrared", "pointcloud"):
            if key in self.modalities:
                return self.modalities[key]
        return next(iter(self.modalities.values()))

    @property
    def is_anomalous(self) -> bool | None:
        return {GOOD: False, BAD: True}.get(self.label)

    def to_dict(self, root: Path | None = None) -> dict:
        def rel(p: Path) -> str:
            try:
                return p.relative_to(root).as_posix() if root else p.as_posix()
            except ValueError:
                return p.as_posix()
        masks = {k: ([rel(x) for x in v] if isinstance(v, list) else rel(v)) for k, v in self.masks.items()}
        return {"sample_id": self.sample_id, "dataset_id": self.dataset_id, "category": self.category,
                "split": self.split, "label": self.label, "defect_type": self.defect_type,
                "modalities": {k: rel(v) for k, v in self.modalities.items()}, "masks": masks,
                "meta": self.meta}


@dataclass
class DatasetIndex:
    dataset_id: str
    category: str
    root: Path
    samples: list[Sample]
    warnings: list[str] = field(default_factory=list)

    def split(self, name: str) -> list[Sample]:
        return [s for s in self.samples if s.split == name]

    @property
    def splits(self) -> list[str]:
        seen: list[str] = []
        for s in self.samples:
            if s.split not in seen:
                seen.append(s.split)
        return seen

    def train_normals(self) -> list[Sample]:
        """Normal samples that may legitimately be used to fit a model (never test data)."""
        return [s for s in self.samples if s.split == "train" and s.label == GOOD]

    def labelled_test(self) -> list[Sample]:
        """Test samples whose ground-truth label is public."""
        return [s for s in self.samples if s.split.startswith("test") and s.label in (GOOD, BAD)]

    def summary(self) -> dict:
        out: dict[str, dict[str, int]] = {}
        for s in self.samples:
            out.setdefault(s.split, {}).setdefault(s.label, 0)
            out[s.split][s.label] += 1
        return out


class DatasetAdapter(ABC):
    """Reads one dataset family from a local directory."""

    dataset_id: str = ""          # catalog id, e.g. "D05"
    folder_name: str = ""         # default sub-folder under the data root, e.g. "mvtec_ad2"
    modalities: tuple[str, ...] = ("rgb",)
    synthetic: bool = False       # True only for clearly-labelled educational/dev data
    eval_pixel_max_side: int = 256   # resolution (longer side) for pixel-level metrics

    def default_root(self, data_root: Path) -> Path:
        return data_root / self.folder_name

    @abstractmethod
    def list_categories(self, root: Path) -> list[str]:
        ...

    @abstractmethod
    def index(self, root: Path, category: str) -> DatasetIndex:
        ...

    def is_installed(self, root: Path) -> bool:
        return root.is_dir() and bool(self.list_categories(root))
