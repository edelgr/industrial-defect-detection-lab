"""Adapters for the MVTec family of 2D benchmarks (MVTec AD, MVTec LOCO AD, MVTec AD 2).

All three share the ``<category>/<split>/<class-folder>/<image>`` convention but differ in where
the ground-truth masks live:

* MVTec AD       : ``<cat>/ground_truth/<defect>/<stem>_mask.png``
* MVTec LOCO AD  : ``<cat>/ground_truth/<defect>/<stem>/<k>.png`` (one or more masks per image)
* MVTec AD 2     : ``<cat>/test_public/ground_truth/bad/<stem>_mask.png``; private test splits
                   contain images whose labels are withheld by the benchmark organisers.
"""

from __future__ import annotations

from pathlib import Path

from core.datasets.base import BAD, GOOD, UNKNOWN, DatasetAdapter, DatasetIndex, Sample
from core.datasets.io import IMAGE_EXT

GT_DIR = "ground_truth"


def _images(folder: Path) -> list[Path]:
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXT)


class MVTecStyleAdapter(DatasetAdapter):
    modalities = ("rgb",)
    private_splits: tuple[str, ...] = ()

    def list_categories(self, root: Path) -> list[str]:
        if not root.is_dir():
            return []
        cats = []
        for d in sorted(root.iterdir()):
            if d.is_dir() and any((d / s).is_dir() for s in ("train", "test", "test_public")):
                cats.append(d.name)
        return cats

    # ------------------------------------------------------------------ masks
    def _find_masks(self, cat_dir: Path, split_dir: Path, defect: str, img: Path) -> list[Path]:
        candidates = [
            split_dir / GT_DIR / defect,      # MVTec AD 2 (inside the split)
            cat_dir / GT_DIR / defect,        # MVTec AD / LOCO
        ]
        for gdir in candidates:
            if not gdir.is_dir():
                continue
            for name in (f"{img.stem}_mask.png", f"{img.stem}.png"):
                f = gdir / name
                if f.is_file():
                    return [f]
            sub = gdir / img.stem            # LOCO: folder with one mask per annotated region
            if sub.is_dir():
                ms = _images(sub)
                if ms:
                    return ms
        return []

    # ------------------------------------------------------------------ index
    def index(self, root: Path, category: str) -> DatasetIndex:
        cat_dir = root / category
        samples: list[Sample] = []
        warnings: list[str] = []
        if not cat_dir.is_dir():
            raise FileNotFoundError(cat_dir)
        for split_dir in sorted(p for p in cat_dir.iterdir() if p.is_dir() and p.name != GT_DIR):
            split = split_dir.name
            if split.startswith("."):
                continue
            direct = _images(split_dir)
            if direct:   # private test splits: images without class sub-folders
                for img in direct:
                    samples.append(self._make(category, split, UNKNOWN, None, img, [], root))
            for cls_dir in sorted(p for p in split_dir.iterdir() if p.is_dir() and p.name != GT_DIR):
                defect = cls_dir.name
                label = GOOD if defect == "good" else BAD
                if split in self.private_splits:
                    label = UNKNOWN
                for img in _images(cls_dir):
                    masks = [] if label != BAD else self._find_masks(cat_dir, split_dir, defect, img)
                    if label == BAD and not masks:
                        warnings.append(f"No ground-truth mask for {img.relative_to(root).as_posix()}")
                    samples.append(self._make(category, split, label, None if label == GOOD else defect,
                                              img, masks, root))
        return DatasetIndex(self.dataset_id, category, root, samples, warnings)

    def _make(self, category, split, label, defect, img: Path, masks, root: Path) -> Sample:
        rel = img.relative_to(root / category).with_suffix("").as_posix()
        return Sample(sample_id=rel, dataset_id=self.dataset_id, category=category, split=split,
                      label=label, defect_type=defect, modalities={"rgb": img},
                      masks={"rgb": masks} if masks else {},
                      meta={"original_path": img.relative_to(root).as_posix()})


class MVTecADAdapter(MVTecStyleAdapter):
    dataset_id = "D01"
    folder_name = "mvtec_ad"


class MVTecLOCOAdapter(MVTecStyleAdapter):
    dataset_id = "D02"
    folder_name = "mvtec_loco_ad"


class MVTecAD2Adapter(MVTecStyleAdapter):
    """MVTec AD 2: only ``test_public`` carries public labels; private test labels are withheld.

    The official protocol evaluates the private test sets on the MVTec evaluation server; IDD Lab
    therefore never computes metrics on ``test_private*`` splits.
    """

    dataset_id = "D05"
    folder_name = "mvtec_ad2"
    private_splits = ("test_private", "test_private_mixed")
    eval_pixel_max_side = 1024    # large images with very small defects (median ≈ 0.25–0.4 % of the image)

    # File names are <id>_<condition>.png, e.g. 000_regular, 000_overexposed, 003_shift_2, 012_mixed.
    # In test_public the same scene id appears under several lighting conditions.
    def _make(self, category, split, label, defect, img: Path, masks, root: Path) -> Sample:
        s = super()._make(category, split, label, defect, img, masks, root)
        sid, _, cond = img.stem.partition("_")
        if sid.isdigit() and cond:
            s.meta["scene_id"] = sid
            s.meta["lighting_condition"] = cond
        return s
