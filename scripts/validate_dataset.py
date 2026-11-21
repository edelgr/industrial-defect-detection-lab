"""Read-only integrity validation of an installed dataset (indexing, image/annotation pairing).

  python -m scripts.validate_dataset --dataset D05 --categories sheet_metal fruit_jelly
  python -m scripts.validate_dataset --dataset D05 --md docs/DATASET_VALIDATION_D05.md

Checks (nothing on disk is modified):
  * the adapter indexes every image file on disk exactly once, with unique sample ids;
  * every anomalous sample has exactly one ground-truth mask whose name matches the image
    (<stem>_mask.png), with identical height/width, binary values {0,255} and >= 1 defect pixel;
  * no normal or private sample carries a mask; no mask file is orphaned (unused);
  * every image decodes; per-split resolution / channel statistics;
  * private splits (MVTec AD 2 test_private*) are unlabelled and excluded from labelled_test();
  * no byte-identical file is shared between train, validation and test splits (leakage check);
  * per-split / per-lighting-condition counts.
Exit status 1 if any error is found.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from core import settings
from core.datasets import registry as dreg
from core.datasets.base import BAD, GOOD, UNKNOWN
from core.datasets.io import IMAGE_EXT, read_image_raw


def _sha1(p: Path) -> str:
    h = hashlib.sha1()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def validate(dataset_id: str, category: str) -> dict:
    adapter = dreg.get(dataset_id)
    idx = dreg.load_index(dataset_id, category, refresh=True)
    cat_dir = idx.root / category
    errors: list[str] = []
    info: dict = {"dataset": dataset_id, "category": category, "root": str(cat_dir),
                  "index_warnings": list(idx.warnings), "summary": idx.summary()}
    errors += [f"index warning: {w}" for w in idx.warnings]

    # 1. every image on disk indexed exactly once ---------------------------------------
    on_disk = sorted(p for p in cat_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXT
                     and "ground_truth" not in p.relative_to(cat_dir).parts)
    indexed = [s.primary_path.resolve() for s in idx.samples]
    if len(set(indexed)) != len(indexed):
        errors.append("an image file is indexed more than once")
    missing = set(p.resolve() for p in on_disk) - set(indexed)
    if missing:
        errors.append(f"{len(missing)} image files on disk are not indexed, e.g. {sorted(missing)[0]}")
    ids = [s.sample_id for s in idx.samples]
    if len(set(ids)) != len(ids):
        errors.append("duplicate sample ids")
    info["image_files_on_disk"] = len(on_disk)
    info["indexed_samples"] = len(idx.samples)

    # 2./3. masks ------------------------------------------------------------------------
    used_masks: set[Path] = set()
    defect_fraction = []
    shapes: dict[str, Counter] = defaultdict(Counter)
    bad_decode = 0
    for s in idx.samples:
        try:
            img = read_image_raw(s.primary_path)
        except Exception as e:  # noqa: BLE001
            errors.append(f"cannot decode {s.sample_id}: {e}")
            bad_decode += 1
            continue
        ch = 1 if img.ndim == 2 else img.shape[2]
        shapes[s.split][f"{img.shape[1]}x{img.shape[0]}x{ch} {img.dtype}"] += 1
        m = s.masks.get("rgb")
        if s.label == BAD:
            ms = m if isinstance(m, list) else ([m] if m else [])
            if len(ms) != 1:
                errors.append(f"{s.sample_id}: expected 1 mask, found {len(ms)}")
                continue
            mp = Path(ms[0])
            used_masks.add(mp.resolve())
            if mp.stem != f"{s.primary_path.stem}_mask":
                errors.append(f"{s.sample_id}: mask name {mp.name} does not match image {s.primary_path.name}")
            import cv2
            mk = cv2.imdecode(np.fromfile(str(mp), np.uint8), cv2.IMREAD_UNCHANGED)
            if mk is None:
                errors.append(f"{s.sample_id}: mask does not decode")
                continue
            if mk.ndim == 3:
                mk = mk[..., 0]
            if mk.shape != img.shape[:2]:
                errors.append(f"{s.sample_id}: mask shape {mk.shape} != image shape {img.shape[:2]}")
            vals = set(np.unique(mk).tolist())
            if not vals <= {0, 255}:
                errors.append(f"{s.sample_id}: non-binary mask values {sorted(vals)[:6]}")
            frac = float((mk > 0).mean())
            if frac == 0:
                errors.append(f"{s.sample_id}: anomalous sample with an empty mask")
            defect_fraction.append(frac)
        elif m:
            errors.append(f"{s.sample_id}: {s.label} sample carries a mask")
        if s.label == UNKNOWN and s.split not in getattr(adapter, "private_splits", ()):
            errors.append(f"{s.sample_id}: unlabelled sample outside a private split")
    gt_files = sorted(p.resolve() for p in cat_dir.rglob("*") if p.is_file()
                      and "ground_truth" in p.relative_to(cat_dir).parts)
    orphans = [p for p in gt_files if p not in used_masks]
    if orphans:
        errors.append(f"{len(orphans)} orphan mask files, e.g. {orphans[0].name}")
    info["mask_files"] = len(gt_files)
    info["image_shapes"] = {k: dict(v) for k, v in shapes.items()}
    if defect_fraction:
        a = np.asarray(defect_fraction)
        info["defect_area_percent"] = {"min": round(100 * a.min(), 4), "median": round(100 * float(np.median(a)), 4),
                                       "max": round(100 * a.max(), 3)}

    # 4. protocol -------------------------------------------------------------------------
    priv = set(getattr(adapter, "private_splits", ()))
    lt = idx.labelled_test()
    if any(s.split in priv for s in lt):
        errors.append("a private-split sample is in labelled_test()")
    if any(s.split != "train" for s in idx.train_normals()):
        errors.append("train_normals() contains non-train samples")
    info["labelled_test"] = {"n": len(lt), "good": sum(s.label == GOOD for s in lt),
                             "bad": sum(s.label == BAD for s in lt)}

    # 5. byte-level leakage between splits -------------------------------------------------
    by_hash: dict[str, set[str]] = defaultdict(set)
    for s in idx.samples:
        by_hash[_sha1(s.primary_path)].add("test" if s.split.startswith("test") else s.split)
    shared = [h for h, sp in by_hash.items() if len(sp) > 1]
    if shared:
        errors.append(f"{len(shared)} byte-identical images shared between splits")

    # 6. lighting conditions (MVTec AD 2) --------------------------------------------------
    cond = defaultdict(Counter)
    for s in idx.samples:
        if "lighting_condition" in s.meta:
            cond[s.split][f"{s.label}:{s.meta['lighting_condition']}"] += 1
    if cond:
        info["lighting_conditions"] = {k: dict(sorted(v.items())) for k, v in cond.items()}
        scenes = defaultdict(set)
        for s in idx.split("test_public"):
            scenes[(s.label, s.meta.get("scene_id"))].add(s.meta.get("lighting_condition"))
        info["test_public_scenes"] = {"good": sum(1 for (lab, _) in scenes if lab == GOOD),
                                      "bad": sum(1 for (lab, _) in scenes if lab == BAD)}
    info["errors"] = errors
    info["ok"] = not errors
    return info


def to_md(results: list[dict]) -> str:
    L = ["# Dataset integrity validation", "",
         f"Generated {time.strftime('%Y-%m-%d %H:%M')} by `python -m scripts.validate_dataset` "
         "(read-only; no file is modified). Checks: complete and unique indexing, image ↔ mask pairing by "
         "name, mask size equal to image size, binary non-empty masks, no masks on normal/private samples, "
         "no orphan masks, every image decodes, private splits unlabelled and excluded from evaluation, "
         "no byte-identical image shared between train / validation / test.", ""]
    for r in results:
        L += [f"## {r['dataset']} · {r['category']} — {'✅ OK' if r['ok'] else '❌ ' + str(len(r['errors'])) + ' error(s)'}", "",
              f"- Image files on disk: {r['image_files_on_disk']} · indexed samples: {r['indexed_samples']} · "
              f"mask files: {r['mask_files']}",
              "- Splits: " + "; ".join(f"`{k}` " + ", ".join(f"{lab} {n}" for lab, n in v.items())
                                       for k, v in r["summary"].items()),
              f"- Labelled test used for evaluation: {r['labelled_test']['n']} "
              f"({r['labelled_test']['good']} good / {r['labelled_test']['bad']} bad)",
              "- Image format per split: " + "; ".join(f"`{k}` " + ", ".join(f"{s} ×{n}" for s, n in v.items())
                                                      for k, v in r["image_shapes"].items())]
        if "defect_area_percent" in r:
            d = r["defect_area_percent"]
            L.append(f"- Defect area per anomalous image: min {d['min']} % · median {d['median']} % · max {d['max']} %")
        if "lighting_conditions" in r:
            L.append("- Lighting conditions: " + "; ".join(
                f"`{k}` " + ", ".join(f"{c} {n}" for c, n in v.items()) for k, v in r["lighting_conditions"].items()))
            t = r["test_public_scenes"]
            L.append(f"- `test_public` distinct scenes: {t['good']} good, {t['bad']} bad (each under several lighting conditions)")
        for e in r["errors"]:
            L.append(f"- ❌ {e}")
        L.append("")
    return "\n".join(L)


def main(argv=None):
    settings.load()
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="D05")
    ap.add_argument("--categories", nargs="+")
    ap.add_argument("--md", help="write a Markdown report to this path")
    ap.add_argument("--json", help="write the raw results to this path")
    a = ap.parse_args(argv)
    cats = a.categories or dreg.installed_categories(a.dataset)
    res = [validate(a.dataset, c) for c in cats]
    for r in res:
        print(f"{r['dataset']} {r['category']}: {'OK' if r['ok'] else 'ERRORS'} · {r['indexed_samples']} samples · "
              f"{r['mask_files']} masks · {r['summary']}")
        for e in r["errors"]:
            print("   ", e)
    if a.md:
        Path(a.md).write_text(to_md(res) + "\n", encoding="utf-8")
    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=2, default=str), encoding="utf-8")
    return 0 if all(r["ok"] for r in res) else 1


if __name__ == "__main__":
    sys.exit(main())
