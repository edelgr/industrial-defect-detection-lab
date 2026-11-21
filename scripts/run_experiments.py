"""Reproducible local experiments → experiments/experiments.sqlite + docs/LOCAL_RESULTS.md.

  python -m scripts.run_experiments --dataset D06 --categories capsule nut flat_pad
  python -m scripts.run_experiments --dataset D06 --categories capsule --methods patchcore --fusion
  python -m scripts.run_experiments --dataset D05 --categories sheet_metal fruit_jelly \
      --methods patchcore padim efficientad anomalydino "patchcore:keep_aspect=1" \
      --n-fit 1000 --n-cal 60 --pixel-max-side 1024 --localization

A method may carry parameter overrides as ``<id>:<name>=<value>,...`` (an ablation variant).
D06 results go to docs/LOCAL_RESULTS.md; other datasets to docs/LOCAL_RESULTS_<dataset>.md
(unless ``--out`` is given).  Raw metrics are also written to experiments/results_<dataset>.json.

All results are LOCAL EDUCATIONAL results (or SYNTHETIC DEMONSTRATION on the synthetic set) and
are never presented as official benchmark numbers.
"""

from __future__ import annotations

import argparse
import json
import time

from core import paths, settings
from core.datasets import registry as dreg
from core.experiments.store import ExperimentStore, environment_snapshot
from core.inference.session import LabSession

DEFAULT_METHODS = ["canny", "ssim", "padim", "patchcore", "anomalydino", "winclip"]


def main(argv=None):
    settings.load()
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="D06")
    ap.add_argument("--categories", nargs="+")
    ap.add_argument("--methods", nargs="+", default=DEFAULT_METHODS)
    ap.add_argument("--n-fit", type=int, default=60)
    ap.add_argument("--max-test", type=int, default=None)
    ap.add_argument("--fusion", action="store_true", help="also run the multimodal fusion experiment (MulSen-AD)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-cal", type=int, default=30, help="max. normal images for threshold calibration")
    ap.add_argument("--pixel-max-side", type=int, default=None,
                    help="default: the adapter's eval_pixel_max_side (256; MVTec AD 2: 1024)")
    ap.add_argument("--localization", action="store_true", help="also compute AU-PRO and pixel F1")
    ap.add_argument("--out", help="Markdown output path")
    a = ap.parse_args(argv)
    store = ExperimentStore()
    cats = a.categories or dreg.installed_categories(a.dataset)
    rows = []
    for c in cats:
        idx = dreg.load_index(a.dataset, c)
        for spec in a.methods:
            mid, params = _parse_method(spec)
            t0 = time.time()
            if mid == "winclip":
                params.setdefault("class_name", c.replace("_", " "))
            s = LabSession(mid, params, seed=a.seed)
            s.bind(idx)
            fit = s.fit(max_images=a.n_fit)
            t_fit = time.time() - t0
            s.calibrate(max_images=a.n_cal)
            r = s.evaluate(max_images=a.max_test, store=store, pixel_max_side=a.pixel_max_side,
                           localization=a.localization)
            r["wall_s"] = round(time.time() - t0, 1)
            r["fit_s"] = round(t_fit, 1)
            r["fit"] = {k: v for k, v in fit.items() if isinstance(v, (int, float, str, bool, list))}
            r["calibration"] = {"source": s.calibration.source, "image_threshold": s.calibration.image_threshold,
                                "pixel_threshold": s.calibration.pixel_threshold}
            r["params"] = dict(s.method.params)
            rows.append((c, spec, s.method.model_version(), r))
            print(f"{c:12s} {mid:14s} imgAUROC={r['image_auroc']:.3f} pixAUROC={(r['pixel_auroc'] or float('nan')):.3f} "
                  f"lat={r['latency']['mean_ms']:.1f}ms wall={r['wall_s']}s", flush=True)
    fusion = []
    if a.fusion and a.dataset == "D06":
        from core.algorithms.multimodal_3d import fusion as F
        for c in cats:
            t0 = time.time()
            res = F.run(dreg.load_index("D06", c), image_method="patchcore", n_fit=40, seed=a.seed)
            res["wall_s"] = round(time.time() - t0, 1)
            store.create_run(category="local_educational", method_id="multimodal_fusion", dataset_id="D06",
                             dataset_category=c, protocol=res["protocol"], params={"image_method": "patchcore"},
                             seed=a.seed, metrics=res["auroc"])
            fusion.append((c, res))
            print(c, "fusion", json.dumps(res["auroc"]), f"wall={res['wall_s']}s", flush=True)
    out = json.dumps([{"category": c, "method": m, "model_version": v, **r} for c, m, v, r in rows],
                     indent=1, default=str)
    (paths.EXPERIMENTS / f"results_{a.dataset}.json").write_text(out, encoding="utf-8")
    if a.dataset == "D06" and not a.out:
        write_md(a, rows, fusion)
    else:
        write_md_generic(a, rows, a.out or (paths.DOCS / f"LOCAL_RESULTS_{a.dataset}.md"))


def _parse_method(spec: str) -> tuple[str, dict]:
    """'patchcore:keep_aspect=1,image_size=320' -> ('patchcore', {'keep_aspect': 1, 'image_size': 320})"""
    from core.algorithms import registry as areg
    mid, _, rest = spec.partition(":")
    kinds = {p.name: p.kind for p in areg.runtime_class(mid).param_specs()}
    cast = {"int": int, "float": float, "bool": lambda v: v.lower() in ("1", "true", "yes", "on"),
            "choice": str}
    params: dict = {}
    for kv in filter(None, rest.split(",")):
        k, _, v = kv.partition("=")
        if k not in kinds:
            raise SystemExit(f"{mid}: unknown parameter {k!r} (known: {', '.join(kinds)})")
        params[k] = cast.get(kinds[k], str)(v)      # same Python types as the GUI → same cache keys
    return mid, params


def _f(v):
    return "—" if v is None else f"{v:.3f}"


def write_md(a, rows, fusion):
    env = environment_snapshot()
    L = ["# Local Results (IDD Lab)", "",
         "> **Category: local educational results.** Computed with IDD Lab's local protocol (fit on "
         "train/good, thresholds from held-out normal images, evaluation on the public labelled test "
         "split). These are *not* official benchmark results and must not be compared with "
         "author-reported numbers without stating the protocol differences.", "",
         f"Generated {time.strftime('%Y-%m-%d %H:%M')} · dataset {a.dataset} · seed {a.seed} · "
         f"fit ≤ {a.n_fit} images · command: `python -m scripts.run_experiments {' '.join(__import__('sys').argv[1:])}`", "",
         "Environment: " + ", ".join(f"{k} {v}" for k, v in env.items()), "",
         "## Image-modality (RGB) results", "",
         "| Category | Method | Model / version | Image AUROC | Image AP | Pixel AUROC* | F1 @ calibrated thr. | Mean latency (ms) | GPU peak | n test |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for c, mid, ver, r in rows:
        d = r.get("decision") or {}
        gpu = "—" if r.get("gpu_peak_mb") is None else f"{r['gpu_peak_mb']:.0f} MB"
        L.append(f"| {c} | {mid} | {ver} | {_f(r['image_auroc'])} | {_f(r['image_ap'])} | {_f(r['pixel_auroc'])} | "
                 f"{_f(d.get('f1'))} | {r['latency']['mean_ms']:.1f} | {gpu} | "
                 f"{r['n_test']} ({r['n_anomalous']} anom.) |")
    L += ["", "\\* Pixel AUROC on RGB masks only; MulSen-AD provides RGB masks only for samples anomalous in RGB, "
          "so samples anomalous only in other modalities count as all-normal pixels. Maps are down-sampled to ≤256 px for this metric.", "",
          "Object-level labels in MulSen-AD mark a sample anomalous if *any* modality is anomalous. Observation "
          "(not investigated further): RGB-only PatchCore reaches ≥ 0.99 object-level AUROC even though, per "
          "`data.csv`, some anomalous samples show no anomaly in RGB, and its AUROC against the RGB per-modality "
          "label is lower (see fusion table). Possible explanations — e.g. systematic acquisition differences "
          "between normal and anomalous samples (a dataset 'shortcut') — are a good discussion topic, not a conclusion."]
    if fusion:
        L += ["", "## Multimodal fusion (educational protocol)", "",
              "| Category | RGB z | IR z | 3D z | Fused mean | Fused max | RGB (mod. label) | IR (mod. label) | 3D (mod. label) |",
              "|---|---|---|---|---|---|---|---|---|"]
        for c, res in fusion:
            o, m = res["auroc"]["object_level"], res["auroc"]["modality_level"]
            L.append(f"| {c} | {_f(o.get('rgb_z'))} | {_f(o.get('infrared_z'))} | {_f(o.get('pointcloud_z'))} | "
                     f"{_f(o.get('fused_mean'))} | {_f(o.get('fused_max'))} | {_f(m.get('rgb'))} | {_f(m.get('infrared'))} | {_f(m.get('pointcloud'))} |")
        L += ["", f"Protocol: {fusion[0][1]['protocol']}"]
    (paths.DOCS / "LOCAL_RESULTS.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("wrote docs/LOCAL_RESULTS.md")


def write_md_generic(a, rows, out_path):
    """Results report for single-modality 2D datasets (e.g. MVTec AD 2 test_public)."""
    from pathlib import Path
    env = environment_snapshot()
    try:
        from core.research.catalog import get_catalog
        name = get_catalog().datasets[a.dataset]["name"]
    except Exception:
        name = a.dataset
    L = [f"# Local Results — {name} ({a.dataset})", "",
         "> **Category: local educational results.** IDD Lab local protocol: fit only on `train/good`, "
         "label-free thresholds from `validation/good`, evaluation only on the publicly labelled test split. "
         "**These are not official benchmark results.** For MVTec AD 2, official scores are computed by "
         "MVTec's evaluation server on `test_private` / `test_private_mixed`, whose labels are withheld; IDD Lab "
         "never evaluates those splits. Do not compare these numbers with leaderboard or paper numbers "
         "without stating the protocol differences.", "",
         f"Generated {time.strftime('%Y-%m-%d %H:%M')} · seed {a.seed} · fit ≤ {a.n_fit} images · calibration ≤ "
         f"{a.n_cal} images · pixel metrics at ≤ {rows[0][3].get('pixel_metric_max_side') if rows else '?'} px · command: "
         f"`python -m scripts.run_experiments {' '.join(__import__('sys').argv[1:])}`", "",
         "Environment: " + ", ".join(f"{k} {v}" for k, v in env.items()), "",
         "## Detection and localisation", "",
         "| Category | Method | Model / input | n fit | Image AUROC | Image AP | Pixel AUROC | AU-PRO@0.3 | "
         "AU-PRO@0.05 | Pixel F1 (cal.) | Image F1 (cal.) | Latency mean / p95 (ms) | GPU peak |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for c, mid, ver, r in rows:
        d = r.get("decision") or {}
        gpu = "— (CPU)" if r.get("gpu_peak_mb") is None else f"{r['gpu_peak_mb']:.0f} MB"
        hw = r.get("fit", {}).get("input_hw")
        size = f" · input {hw[1]}×{hw[0]}" if hw else ""
        pf = (r.get("pixel_f1") or {}).get("f1")
        lat = r["latency"]
        L.append(f"| {c} | {mid} | {ver}{size} | {r.get('fit', {}).get('n_fit', '—')} | {_f(r['image_auroc'])} | "
                 f"{_f(r['image_ap'])} | {_f(r['pixel_auroc'])} | {_f(r.get('au_pro_30'))} | "
                 f"{_f(r.get('au_pro_05'))} | {_f(pf)} | {_f(d.get('f1'))} | "
                 f"{lat['mean_ms']:.1f} / {lat['p95_ms']:.1f} | {gpu} |")
    conds = sorted({k for _, _, _, r in rows for k in (r.get("by_condition") or {})})
    if conds:
        L += ["", "## Image AUROC per lighting condition (`test_public`)", "",
              "| Category | Method | " + " | ".join(conds) + " |", "|---|---|" + "---|" * len(conds)]
        for c, mid, _, r in rows:
            b = r.get("by_condition") or {}
            cells = [f"{_f(b[k]['image_auroc'])} ({b[k]['n_anomalous']}/{b[k]['n'] - b[k]['n_anomalous']})"
                     if k in b else "—" for k in conds]
            L.append(f"| {c} | {mid} | " + " | ".join(cells) + " |")
        L += ["", "Cell = image AUROC (n anomalous / n normal). Every condition re-photographs the *same* few "
              "scenes (sheet_metal: 15 defective + 4 normal; fruit_jelly: 15 + 5), so per-condition AUROC rests "
              "on 4–5 normal images and is very noisy. Training and validation images are all `regular`."]
    L += ["", "## Fitting and calibration", "",
          "| Category | Method | Fitted / loaded | Fit wall time (s) | Calibration source | Image thr. | Pixel thr. |",
          "|---|---|---|---|---|---|---|"]
    for c, mid, _, r in rows:
        f = r.get("fit", {})
        how = f"loaded cache `{f['cache']}`" if f.get("cache") else ("fitted → cached" if f.get("fitted") else "no fit")
        cal = r.get("calibration", {})
        L.append(f"| {c} | {mid} | {how} | {r.get('fit_s', '—')} | {cal.get('source', '—')} | "
                 f"{_f(cal.get('image_threshold'))} | {_f(cal.get('pixel_threshold'))} |")
    L += ["", "## Metric definitions", "",
          "- **Image AUROC / AP**: raw image scores against the public image labels (threshold-free).",
          "- **Pixel AUROC**: all test pixels pooled (uniformly sub-sampled to 4 M pixels if larger).",
          "- **AU-PRO@x**: area under the per-region-overlap curve up to false-positive rate x, normalised to "
          "[0, 1] (Bergmann et al., IJCV 2021, doi:10.1007/s11263-020-01400-4), at the down-sampled "
          "evaluation resolution.",
          "- **Pixel F1 (cal.)**: pixel-level F1 pooled over all test pixels at the pixel threshold fixed on "
          "`validation/good` (99.5th percentile of normal pixel scores); no test label is used to choose it. "
          "Analogous in spirit to, but not the same computation as, the official server-side SegF1.",
          "- **Image F1 (cal.)**: image-level F1 at the 99th percentile of validation/good image scores.",
          "- **Latency**: `predict()` wall time per image on the stated device (includes resizing and map "
          "up-sampling; excludes PNG decoding). **GPU peak**: `torch.cuda.max_memory_allocated` during inference."]
    Path(out_path).write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
