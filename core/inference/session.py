"""Laboratory session: fit → calibrate → predict → evaluate, with leakage-safe data usage.

Data-usage rules (enforced, and recorded in each run's protocol string):
* ``fit``       uses only ``train`` split normal images (deterministic subset, seeded);
* ``calibrate`` uses only normal images that were NOT used for fitting: the ``validation``
                split if the dataset has one, otherwise a held-out part of ``train``;
* ``evaluate``  uses only test splits with public labels; private/withheld splits are refused.
Thresholds derived from test labels are never used for reported decision metrics.
"""

from __future__ import annotations

import random
from pathlib import Path
from dataclasses import dataclass, field

import numpy as np

from core.algorithms import registry as areg
from core.algorithms.base import AnomalyMethod, CancelFn, MethodResult, ProgressFn
from core.datasets import registry as dreg
from core.datasets.base import BAD, GOOD, DatasetIndex, Sample
from core.datasets.io import read_mask, read_rgb8
from core.evaluation import metrics as M
from core.experiments.store import ExperimentStore


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % 2 ** 32)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


@dataclass
class Calibration:
    image_threshold: float | None = None
    pixel_threshold: float | None = None
    source: str = ""
    n_images: int = 0
    normal_scores: list[float] = field(default_factory=list)
    # statistics of the (uniformly sub-sampled) normal pixel scores, for alternative label-free rules
    pixel_mean: float | None = None
    pixel_std: float | None = None
    sample_ids: list[str] = field(default_factory=list)
    stored: bool = False            # True when restored from a verified preset instead of recomputed

    @property
    def paper_pixel_threshold(self) -> float | None:
        """MVTec AD 2 paper baseline rule (Sec. IV-B2): mean + 3 std of validation normal pixel scores."""
        if self.pixel_mean is None or self.pixel_std is None:
            return None
        return self.pixel_mean + 3.0 * self.pixel_std


def fit_key(method: AnomalyMethod, method_id: str, fit_ids: list[str], seed: int, root) -> str:
    """16-hex cache key over the fit-relevant parameters, the ordered fit sample ids, the seed,
    the Anomalib version and the dataset root (see EXPERIMENT_PROTOCOLS.md, fitted-model cache)."""
    import hashlib
    import json as _json
    from core.algorithms.torch_common import anomalib_version
    fit_params = {p.name: method.params[p.name] for p in method.param_specs() if p.affects_fit
                  and not (p.key_omit_default and method.params[p.name] == p.default)}
    key = _json.dumps({"m": method_id, "p": fit_params, "ids": list(fit_ids), "seed": seed,
                       "anomalib": anomalib_version(), "root": str(root)}, sort_keys=True)
    return hashlib.sha1(key.encode()).hexdigest()[:16]


class LabSession:
    def __init__(self, method_id: str, params: dict | None = None, seed: int = 42):
        self.method_id = method_id
        self.method: AnomalyMethod = areg.create(method_id, **(params or {}))
        self.seed = seed
        self.index: DatasetIndex | None = None
        self.fit_ids: list[str] = []
        self.holdout: list[Sample] = []
        self.calibration = Calibration()
        self.reference: Sample | None = None
        self._ref_img: np.ndarray | None = None
        self.created_by = "lab"          # recorded in checkpoint manifests (lab | compare | script)
        self.checkpoint: Path | None = None   # checkpoint the fitted state came from / was saved to

    # ------------------------------------------------------------------ data
    def bind(self, index: DatasetIndex) -> None:
        self.index = index
        normals = index.train_normals()
        if self.method.requires_reference and normals:
            self.set_reference(normals[0])

    def set_reference(self, s: Sample) -> None:
        if s.split.startswith("test"):
            raise ValueError("The reference image must be a normal training/validation image, not a test image.")
        self.reference = s
        self._ref_img = read_rgb8(s.primary_path)

    def _split_train(self, max_images: int) -> tuple[list[Sample], list[Sample]]:
        normals = sorted(self.index.train_normals(), key=lambda s: s.sample_id)
        rng = random.Random(self.seed)
        rng.shuffle(normals)
        has_val = any(s.split == "validation" and s.label == GOOD for s in self.index.samples)
        n_hold = 0 if has_val else max(2, int(round(0.15 * len(normals))))
        n_hold = min(n_hold, max(0, len(normals) - 1))
        hold, rest = normals[:n_hold], normals[n_hold:]
        return rest[:max_images], hold

    # ------------------------------------------------------------------ fit
    def cache_path(self, fit_set: list[Sample]):
        """checkpoints/fitted/<method>_<dataset>_<category>_<hash>.pt — the hash covers the
        fit-relevant parameters, the exact fit sample ids, the seed and the Anomalib version."""
        from core.inference import model_registry as MR       # MR.FITTED can be redirected (tests)
        h = fit_key(self.method, self.method_id, [s.sample_id for s in fit_set], self.seed, self.index.root)
        return MR.FITTED / f"{self.method_id}_{self.index.dataset_id}_{self.index.category}_{h}.pt"

    def cached_fit(self, max_images: int):
        """Path of a compatible fitted-model cache for ``fit(max_images)``, or None."""
        if self.index is None or not (self.method.requires_fit and self.method.cacheable):
            return None
        fit_set, _ = self._split_train(max_images)
        p = self.cache_path(fit_set) if fit_set else None
        return p if p is not None and p.exists() else None

    def fit(self, max_images: int = 100, progress: ProgressFn | None = None, cancel: CancelFn | None = None,
            use_cache: bool = True) -> dict:
        assert self.index is not None, "bind() a dataset index first"
        fit_set, self.holdout = self._split_train(max_images)
        if not self.method.requires_fit:
            self.fit_ids = []
            return {"fitted": False, "reason": "method does not learn from data"}
        if not fit_set:
            raise RuntimeError("No normal training images available for fitting.")
        seed_everything(self.seed)          # PaDiM channel subset, coreset start, EfficientAD sampling
        cache = self.cache_path(fit_set) if self.method.cacheable else None
        if use_cache and cache is not None and cache.exists():
            if progress:
                progress(0.5, f"loading fitted model from {cache.name}")
            self.method.load_state(cache)
            self.fit_ids = [s.sample_id for s in fit_set]
            self.checkpoint = cache
            from core.inference import model_registry as MR
            if not MR.manifest_path(cache).exists():        # the key matched these exact ids → record them
                MR.write_manifest(self, cache, provenance="key-verified at load (fit date = file time)",
                                  max_images=max_images)
            return {"fitted": True, "n_fit": len(fit_set), "cache": cache.name, **self.method.fit_info}
        imgs = []
        for i, s in enumerate(fit_set):
            AnomalyMethod._check_cancel(cancel)
            imgs.append(read_rgb8(s.primary_path))
            if progress:
                progress(0.3 * (i + 1) / len(fit_set), f"loading training image {i + 1}/{len(fit_set)}")
        def sub(f, m):
            if progress:
                progress(0.3 + 0.7 * f, m)
        self.method.fit(imgs, progress=sub, cancel=cancel)
        self.fit_ids = [s.sample_id for s in fit_set]
        if cache is not None:
            import os
            cache.parent.mkdir(parents=True, exist_ok=True)
            partial = cache.with_name(cache.name + ".partial")     # not matched by the registry (*.pt)
            try:
                self.method.save_state(partial)
                os.replace(partial, cache)                          # atomic on the same volume
            finally:
                if partial.exists():
                    partial.unlink()
            self.checkpoint = cache
            from core.inference import model_registry as MR
            MR.write_manifest(self, cache, provenance="recorded at fit", max_images=max_images)
        return {"fitted": True, "n_fit": len(fit_set), **self.method.fit_info}

    def load_checkpoint(self, entry, progress: ProgressFn | None = None, cancel: CancelFn | None = None) -> dict:
        """Load a registry checkpoint exactly as fitted (its own fit ids and parameters), whatever
        the 'Fit images' spin box says. Refuses anything whose recomputed cache key differs."""
        from core.inference import model_registry as MR
        assert self.index is not None, "bind() a dataset index first"
        ok, reasons = MR.compatibility(entry, self.index)
        if not ok:
            raise RuntimeError("Incompatible checkpoint: " + "; ".join(reasons))
        man = entry.manifest
        self.method = areg.create(self.method_id, **man["params"])
        self.seed = int(man["seed"])
        by_id = {s.sample_id: s for s in self.index.samples}
        fit_set = [by_id[i] for i in man["fit_ids"]]
        if self.cache_path(fit_set) != entry.path:
            raise RuntimeError("Checkpoint key mismatch after applying its manifest (refusing to load).")
        if progress:
            progress(0.5, f"loading {entry.path.name}")
        seed_everything(self.seed)
        # identity and thresholds first: the GUI treats method.fitted (set by load_state) as "ready"
        self.fit_ids = list(man["fit_ids"])
        _, self.holdout = self._split_train(10 ** 9)
        self.checkpoint = entry.path
        if entry.verified and entry.verified.get("calibration"):
            self.apply_calibration(entry.verified["calibration"])
        self.method.load_state(entry.path)
        return {"fitted": True, "n_fit": len(fit_set), "cache": entry.path.name, "tier": entry.tier,
                **self.method.fit_info}

    def apply_calibration(self, c: dict) -> Calibration:
        """Restore label-free thresholds stored with a verified preset (computed on validation/good
        or held-out train/good by scripts/verify_workshop_models.py)."""
        self.calibration = Calibration(image_threshold=c.get("image_threshold"), pixel_threshold=c.get("pixel_threshold"),
                                       source=c.get("source", "") + " [stored with verified preset]",
                                       n_images=int(c.get("n_images") or 0), normal_scores=list(c.get("normal_scores") or []),
                                       pixel_mean=c.get("pixel_mean"), pixel_std=c.get("pixel_std"),
                                       sample_ids=list(c.get("sample_ids") or []), stored=True)
        return self.calibration

    # ------------------------------------------------------------- predict
    def predict_image(self, img: np.ndarray) -> MethodResult:
        return self.method.predict(img, self._ref_img if self.method.requires_reference else None)

    def predict(self, s: Sample) -> MethodResult:
        if self.reference is not None and s.sample_id == self.reference.sample_id:
            pass  # predicting the reference against itself is allowed (shows a ~zero map) but uninformative
        return self.predict_image(read_rgb8(s.primary_path))

    # ----------------------------------------------------------- calibrate
    def calibration_set(self) -> tuple[list[Sample], str]:
        val = [s for s in self.index.samples if s.split == "validation" and s.label == GOOD]
        if val:
            return val, "validation/good"
        if not self.holdout:
            _, self.holdout = self._split_train(10 ** 9)
        hold = [s for s in self.holdout if s.sample_id not in set(self.fit_ids)]
        return hold, "held-out train/good (not used for fitting)"

    def calibrate(self, image_percentile: float = 99.0, pixel_percentile: float = 99.5, max_images: int = 60,
                  progress: ProgressFn | None = None, cancel: CancelFn | None = None) -> Calibration:
        """Label-free thresholds from normal images that were not used for fitting."""
        cal, source = self.calibration_set()
        cal = cal[:max_images]
        if self.reference is not None:
            cal = [s for s in cal if s.sample_id != self.reference.sample_id]
        if not cal:
            raise RuntimeError("No unused normal images available for threshold calibration.")
        scores, pix = [], []
        rng = np.random.default_rng(self.seed)
        for i, s in enumerate(cal):
            AnomalyMethod._check_cancel(cancel)
            r = self.predict(s)
            if r.score is not None:
                scores.append(r.score)
            if r.anomaly_map is not None:
                flat = r.anomaly_map.ravel()
                pix.append(flat[rng.choice(flat.size, min(flat.size, 20000), replace=False)])
            if progress:
                progress((i + 1) / len(cal), f"calibrating on {source}: {i + 1}/{len(cal)}")
        allpix = np.concatenate(pix) if pix else None
        self.calibration = Calibration(
            image_threshold=M.threshold_from_normals(scores, image_percentile) if scores else None,
            pixel_threshold=float(np.percentile(allpix, pixel_percentile)) if pix else None,
            source=f"{source}, p{image_percentile:g} image / p{pixel_percentile:g} pixel, n={len(cal)}",
            n_images=len(cal), normal_scores=scores,
            pixel_mean=float(allpix.mean()) if pix else None, pixel_std=float(allpix.std()) if pix else None,
            sample_ids=[s.sample_id for s in cal])
        return self.calibration

    # ------------------------------------------------------------ evaluate
    def test_samples(self, split: str | None = None) -> list[Sample]:
        out = []
        for s in self.index.samples:
            if not s.split.startswith("test") or s.label not in (GOOD, BAD):
                continue
            if split and s.split != split:
                continue
            out.append(s)
        return out

    def evaluate(self, split: str | None = None, max_images: int | None = None, store: ExperimentStore | None = None,
                 progress: ProgressFn | None = None, cancel: CancelFn | None = None,
                 pixel_max_side: int | None = None, localization: bool = False,
                 extra_pixel_thresholds: dict | None = None) -> dict:
        """Evaluate on labelled public test samples.

        ``pixel_max_side``: anomaly maps and masks are down-sampled (area / nearest) so that their
        longer side is at most this many pixels before pixel-level metrics are computed (default:
        the dataset adapter's ``eval_pixel_max_side``).
        ``localization``: also compute AU-PRO (FPR ≤ 0.3 and ≤ 0.05) and pixel-level F1 at the
        label-free calibrated pixel threshold.
        """
        if split and split.startswith("test_private"):
            raise ValueError("Private test labels are withheld by the benchmark; local evaluation is not allowed.")
        tests = self.test_samples(split)
        if pixel_max_side is None:
            ad = dreg.get(self.index.dataset_id)
            pixel_max_side = getattr(ad, "eval_pixel_max_side", 256) if ad else 256
        if max_images and len(tests) > max_images:
            rng = random.Random(self.seed)
            goods = [s for s in tests if s.label == GOOD]
            bads = [s for s in tests if s.label == BAD]
            k = max(1, int(max_images * len(goods) / len(tests)))
            tests = rng.sample(goods, min(k, len(goods))) + rng.sample(bads, min(max_images - k, len(bads)))
        if not tests:
            raise RuntimeError("No labelled test samples available.")
        labels, scores, lat, masks, maps, rows, conds = [], [], [], [], [], [], []
        dice, iou = [], []
        gpu_peak = None
        for i, s in enumerate(tests):
            AnomalyMethod._check_cancel(cancel)
            r = self.predict(s)
            labels.append(1 if s.label == BAD else 0)
            scores.append(r.score)
            lat.append(r.latency_ms)
            if r.gpu_memory_mb is not None:
                gpu_peak = max(gpu_peak or 0.0, r.gpu_memory_mb)
            gt_path = s.masks.get("rgb")
            if r.anomaly_map is not None:
                gt = read_mask(gt_path, r.anomaly_map.shape) if gt_path else np.zeros(r.anomaly_map.shape, bool)
                small = _downsample(r.anomaly_map, gt, pixel_max_side)
                masks.append(small[1])
                maps.append(small[0])
                if self.calibration.pixel_threshold is not None and s.label == BAD:
                    d, j = M.dice_iou(r.anomaly_map >= self.calibration.pixel_threshold, gt)
                    if d is not None:
                        dice.append(d)
                        iou.append(j)
            conds.append(s.meta.get("lighting_condition"))
            rows.append({"sample_id": s.sample_id, "split": s.split, "label": s.label, "score": r.score,
                         "latency_ms": r.latency_ms, "lighting_condition": s.meta.get("lighting_condition"),
                         "scene_id": s.meta.get("scene_id"),
                         "map_max": None if r.anomaly_map is None else float(r.anomaly_map.max())})
            if progress:
                progress((i + 1) / len(tests), f"evaluating {i + 1}/{len(tests)}")
        res: dict = {
            "n_test": len(tests), "n_anomalous": int(sum(labels)),
            "image_auroc": M.image_auroc(labels, scores), "image_ap": M.image_ap(labels, scores),
            "pixel_auroc": M.pixel_auroc(masks, maps) if maps and any(m.any() for m in masks) else None,
            "latency": M.latency_stats(lat),
            "gpu_peak_mb": gpu_peak,
        }
        res["pixel_metric_max_side"] = pixel_max_side
        if localization and maps and any(m.any() for m in masks):
            res["au_pro_30"] = M.au_pro(masks, maps, 0.3)
            res["au_pro_05"] = M.au_pro(masks, maps, 0.05)
            if self.calibration.pixel_threshold is not None:
                pf = M.pixel_f1(masks, [a >= self.calibration.pixel_threshold for a in maps])
                if pf:
                    res["pixel_f1"] = {**pf, "threshold_source": self.calibration.source}
            for name, thr in (extra_pixel_thresholds or {}).items():
                pf = M.pixel_f1(masks, [a >= thr for a in maps])
                if pf:
                    res.setdefault("pixel_f1_alt", {})[name] = {**pf, "threshold": thr}
        if any(conds):
            by: dict = {}
            for c in sorted(set(c for c in conds if c)):
                ii = [i for i, x in enumerate(conds) if x == c]
                y = [labels[i] for i in ii]
                by[c] = {"n": len(ii), "n_anomalous": int(sum(y)),
                         "image_auroc": M.image_auroc(y, [scores[i] for i in ii])}
            res["by_condition"] = by
        if dice:
            res["mean_dice"] = float(np.mean(dice))
            res["mean_iou"] = float(np.mean(iou))
            res["segmentation_threshold_source"] = self.calibration.source
        if self.calibration.image_threshold is not None:
            res["decision"] = M.decision_metrics(labels, scores, self.calibration.image_threshold,
                                                 self.calibration.source).to_dict()
        res["protocol"] = (self.protocol_string(split, len(tests))
                           + f" Pixel metrics on maps/masks down-sampled to ≤{pixel_max_side} px (longer side).")
        a = dreg.get(self.index.dataset_id)
        res["result_category"] = "synthetic_demo" if (a and a.synthetic) else "local_educational"
        if store is not None:
            run = store.create_run(category=res["result_category"], method_id=self.method_id,
                                   model_version=self.method.model_version(), dataset_id=self.index.dataset_id,
                                   dataset_category=self.index.category, protocol=res["protocol"],
                                   params=self.method.params, seed=self.seed, metrics=res)
            store.add_predictions(run, rows)
            res["run_id"] = run
        res["samples"] = rows
        return res

    def protocol_string(self, split, n) -> str:
        fit = f"fit on {len(self.fit_ids)} train/good images" if self.fit_ids else "no fitting"
        ref = f"; reference={self.reference.sample_id}" if self.reference else ""
        cal = f"; thresholds from {self.calibration.source}" if self.calibration.source else "; no calibrated threshold"
        return (f"IDD Lab local protocol: {fit}{ref}{cal}; evaluated on {n} labelled samples of "
                f"{split or 'all public test splits'} (seed {self.seed}). Not an official benchmark result.")


def _downsample(amap: np.ndarray, gt: np.ndarray, max_side: int = 256):
    """Downsample map+mask for pixel-AUROC memory efficiency (documented in the protocol)."""
    import cv2
    h, w = amap.shape
    s = max_side / max(h, w)
    if s >= 1:
        return amap, gt
    size = (max(1, int(w * s)), max(1, int(h * s)))
    return (cv2.resize(amap, size, interpolation=cv2.INTER_AREA),
            cv2.resize(gt.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST) > 0)
