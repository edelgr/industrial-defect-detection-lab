"""Verify the Workshop presets: reload each cached model, recalibrate, re-evaluate, compare with
the stored campaign results, and write ``configs/workshop_verified.json``.

  python -m scripts.verify_workshop_models            # all presets whose checkpoint exists
  python -m scripts.verify_workshop_models --only d05  # MVTec AD 2 presets only
  python -m scripts.verify_workshop_models --latency   # dedicated latency benchmark of verified presets

This script NEVER trains: a preset whose checkpoint is missing is reported and skipped.
A preset becomes *Workshop Verified* only if its reloaded model reproduces the stored reference
metrics (|Δ| ≤ 1e-6 for image AUROC / AP / pixel AUROC, and AU-PRO where stored).
The JSON contains configuration, thresholds, metrics and per-sample scores of the public test
split (sample ids only — no image data).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
import time

from core import paths, settings
from core.datasets import registry as dreg
from core.experiments.store import environment_snapshot
from core.inference import model_registry as MR
from core.inference.session import LabSession

D05_METHODS = [("patchcore", {}, "256"), ("padim", {}, "256"), ("efficientad", {}, "256"),
               ("anomalydino", {}, "448"), ("patchcore", {"keep_aspect": True}, "aspect"),
               ("padim", {"keep_aspect": True}, "aspect")]
D06_METHODS = [("patchcore", {}, "256"), ("padim", {}, "256"), ("efficientad", {}, "256"), ("anomalydino", {}, "448")]
PRESETS = (
    [dict(dataset="D05", category=c, method=m, params=p, tag=t, n_fit=1000, n_cal=60)
     for c in ("sheet_metal", "fruit_jelly") for m, p, t in D05_METHODS]
    + [dict(dataset="D06", category=c, method=m, params=p, tag=t, n_fit=60, n_cal=30)
       for c in ("capsule", "nut", "flat_pad") for m, p, t in D06_METHODS]
)
EXPERIMENTS = {("D05", "sheet_metal", "patchcore"): ["resolution", "illumination", "metrics"]}
TOL = 1e-6
NAMES = {"patchcore": "PatchCore", "padim": "PaDiM", "efficientad": "EfficientAD", "anomalydino": "AnomalyDINO"}


def _sha256(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def _reference(ds: str, cat: str, method: str, params: dict, n_test: int) -> tuple[dict | None, str]:
    """Stored campaign metrics for the same configuration."""
    if ds == "D05":
        f = paths.EXPERIMENTS / "results_D05.json"
        if f.exists():
            spec = method + (":keep_aspect=1" if params.get("keep_aspect") else "")
            for r in json.loads(f.read_text(encoding="utf-8")):
                if r["category"] == cat and r["method"] == spec:
                    return r, f"experiments/results_D05.json ({spec})"
        return None, "no stored reference"
    db = paths.EXPERIMENTS / "experiments.sqlite"
    if not db.exists():
        return None, "no stored reference"
    with sqlite3.connect(db) as c:
        rows = c.execute("SELECT run_id, metrics, params FROM runs WHERE dataset_id=? AND dataset_category=? AND method_id=? "
                         "ORDER BY created DESC", (ds, cat, method)).fetchall()
    for run_id, m, prm in rows:          # latest full-test-split run with the same parameters
        m, prm = json.loads(m), json.loads(prm)
        if m.get("n_test") == n_test and all(prm.get(k) == v for k, v in params.items()):
            return m, f"experiments.sqlite run {run_id}"
    return None, "no stored reference"


def verify(p: dict) -> dict:
    out = {"preset": p, "status": "skipped"}
    if p["category"] not in dreg.installed_categories(p["dataset"]):
        out["reason"] = "dataset category not installed"
        return out
    idx = dreg.load_index(p["dataset"], p["category"])
    s = LabSession(p["method"], dict(p["params"]), seed=settings.get().seed)
    s.created_by = "script"
    s.bind(idx)
    ck = s.cached_fit(p["n_fit"])
    if ck is None:
        out["reason"] = "no cached checkpoint for this configuration (this script never trains)"
        return out
    t0 = time.time()
    info = s.fit(max_images=p["n_fit"])                  # loads from cache (checked above)
    assert info.get("cache") == ck.name
    cal = s.calibrate(max_images=p["n_cal"])
    res = s.evaluate(localization=True, extra_pixel_thresholds={"paper_mean3std": cal.paper_pixel_threshold}
                     if cal.paper_pixel_threshold is not None else None)
    ref, ref_src = _reference(p["dataset"], p["category"], p["method"], p["params"], res["n_test"])
    keys = ["image_auroc", "image_ap", "pixel_auroc"] + (["au_pro_05", "au_pro_30"] if p["dataset"] == "D05" else [])
    diffs = {k: abs(res[k] - ref[k]) for k in keys if ref and res.get(k) is not None and ref.get(k) is not None}
    ok = bool(diffs) and max(diffs.values()) <= TOL
    man = json.loads(MR.manifest_path(ck).read_text(encoding="utf-8"))
    hw = man.get("input_hw")
    tag = f"{hw[1]}×{hw[0]}" if hw else p["tag"]
    label = (f"{NAMES[p['method']]} · {p['category']} · {tag}"
             + (" (aspect kept)" if p["params"].get("keep_aspect") else "") + f" · {man['n_fit']} train images")
    out.update({
        "status": "verified" if ok else "reference_mismatch" if diffs else "no_reference",
        "preset_id": f"{p['dataset'].lower()}-{p['category']}-{p['method']}-{p['tag']}",
        "label": label, "key": MR.parse_name(ck)["key"], "file": ck.name, "size_bytes": ck.stat().st_size,
        "sha256": _sha256(ck), "method_id": p["method"], "params": man["params"], "dataset_id": p["dataset"],
        "category": p["category"], "n_fit": man["n_fit"], "seed": man["seed"], "input_hw": hw,
        "fit_date": man["fit_date"], "verified_date": time.strftime("%Y-%m-%d %H:%M"),
        "experiments": EXPERIMENTS.get((p["dataset"], p["category"], p["method"]), []),
        "calibration": {"source": cal.source, "image_threshold": cal.image_threshold,
                        "pixel_threshold": cal.pixel_threshold, "pixel_mean": cal.pixel_mean,
                        "pixel_std": cal.pixel_std, "paper_pixel_threshold": cal.paper_pixel_threshold,
                        "n_images": cal.n_images, "normal_scores": cal.normal_scores, "sample_ids": cal.sample_ids},
        "metrics": {k: res.get(k) for k in ("n_test", "n_anomalous", "image_auroc", "image_ap", "pixel_auroc",
                                            "au_pro_30", "au_pro_05", "pixel_f1", "pixel_f1_alt", "decision",
                                            "latency", "gpu_peak_mb", "by_condition", "pixel_metric_max_side",
                                            "protocol", "result_category")},
        "reference_check": {"reference": ref_src, "compared": sorted(diffs),
                            "max_abs_diff": max(diffs.values()) if diffs else None, "tolerance": TOL},
        "samples": [{k: r[k] for k in ("sample_id", "label", "score", "map_max", "lighting_condition", "scene_id")}
                    for r in res["samples"]],
        "wall_s": round(time.time() - t0, 1),
    })
    return out


def latency_benchmark(n_images: int = 30, warmup: int = 3) -> None:
    """Per-image predict() latency with warm-up, on the first ``n_images`` public test images (sorted ids).
    The latency logged during evaluation includes the first (cold) calls and any concurrent load, so this
    separate measurement is what the GUI shows; the hardware power state is not controlled."""
    import numpy as np
    from core.datasets.io import read_rgb8
    doc = json.loads(MR.VERIFIED_FILE.read_text(encoding="utf-8"))
    env = environment_snapshot()
    for m in doc["models"]:
        e = MR.find_preset(m["preset_id"])
        if e is None:
            continue
        idx = dreg.load_index(m["dataset_id"], m["category"])
        s = LabSession(m["method_id"], seed=m["seed"])
        s.bind(idx)
        s.load_checkpoint(e)
        tests = sorted((x for x in idx.samples if x.split.startswith("test") and x.label in ("good", "bad")),
                       key=lambda x: x.sample_id)[: n_images + warmup]
        imgs = [read_rgb8(x.primary_path) for x in tests]
        res = [s.predict_image(i) for i in imgs]
        lat = np.array([r.latency_ms for r in res[warmup:]])
        gpu = [r.gpu_memory_mb for r in res[warmup:] if r.gpu_memory_mb is not None]
        m["latency_benchmark"] = {"median_ms": float(np.median(lat)), "p95_ms": float(np.percentile(lat, 95)),
                                  "n": int(lat.size), "warmup": warmup, "gpu_peak_mb": max(gpu) if gpu else None,
                                  "device": res[-1].device, "gpu": env.get("cuda_device"),
                                  "measured": time.strftime("%Y-%m-%d %H:%M"),
                                  "note": "predict() incl. resize + map up-sampling, excl. PNG decoding; power state not controlled"}
        print(f"{m['preset_id']:36s} median {m['latency_benchmark']['median_ms']:.1f} ms  p95 {m['latency_benchmark']['p95_ms']:.1f} ms", flush=True)
        del s
    MR.VERIFIED_FILE.write_text(json.dumps(doc, indent=1), encoding="utf-8")


def main(argv=None):
    settings.load()
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", help="prefix filter on dataset id, e.g. d05")
    ap.add_argument("--latency", action="store_true", help="only run the latency benchmark of verified presets")
    a = ap.parse_args(argv)
    if a.latency:
        latency_benchmark()
        return 0
    todo = [p for p in PRESETS if not a.only or p["dataset"].lower().startswith(a.only.lower())]
    old = json.loads(MR.VERIFIED_FILE.read_text(encoding="utf-8")) if MR.VERIFIED_FILE.exists() else {}
    keep = [m for m in old.get("models", []) if a.only and not m["dataset_id"].lower().startswith(a.only.lower())]
    results = []
    for p in todo:
        r = verify(p)
        results.append(r)
        print(f"{p['dataset']} {p['category']:12s} {p['method']:12s} {p['tag']:7s} {r['status']:18s} "
              f"{r.get('reference_check', {}).get('max_abs_diff', r.get('reason', ''))}", flush=True)
    not_verified = [{"preset": r["preset"], "status": r["status"], "reason": r.get("reason"),
                     "reference_check": r.get("reference_check")} for r in results if r["status"] != "verified"]
    models = keep + [{k: v for k, v in r.items() if k not in ("preset", "status")}
                     for r in results if r["status"] == "verified"]
    doc = {"description": "Workshop Verified presets — generated by scripts/verify_workshop_models.py. Local "
                          "educational results on public labelled test splits; not official benchmark results.",
           "generated": time.strftime("%Y-%m-%d %H:%M"), "environment": environment_snapshot(),
           "tolerance": TOL, "models": models,
           "not_verified": not_verified}
    MR.VERIFIED_FILE.write_text(json.dumps(doc, indent=1), encoding="utf-8")
    print(f"wrote {MR.VERIFIED_FILE} — {len(models)} verified, {len(doc['not_verified'])} not verified")
    return 0


if __name__ == "__main__":
    sys.exit(main())
