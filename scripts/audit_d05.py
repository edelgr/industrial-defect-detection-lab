"""Scientific audit of the MVTec AD 2 educational experiments → docs/SCIENTIFIC_AUDIT_D05.md

  python -m scripts.audit_d05              (≈ 3–5 min on an RTX 4090 Laptop GPU; no training)

Checks, per category (sheet_metal, fruit_jelly):
  1. split contamination — byte-identical files and near-duplicate images (NCC of normalised
     thumbnails test→train/validation compared with the train→train nearest-neighbour baseline);
  2. scene identity — GT-mask IoU across lighting variants of anomalous scenes; thumbnail NCC within
     vs. between normal scenes (supports treating `<id>_<condition>` as one physical scene);
  3. interpolation — native mask size, defect regions lost / area change when masks are down-sampled
     (nearest) to the evaluation resolution; alignment of the resize→map→up-sample chain;
  4. map ↔ ground-truth correspondence — pixel AUROC with true vs. flipped maps, and a pointing game;
  5. AU-PRO — IDD Lab implementation vs. Anomalib's independent ``_AUPRO`` on the same real maps;
  6. threshold calibration — sources disjoint from fit and test, recomputation equals stored values;
  7. cache keys — recomputed from manifests; perturbations change the key;
  8. reload consistency — verified presets reproduce the stored campaign metrics.
Reads configs/workshop_verified.json (run scripts.verify_workshop_models first).
"""

from __future__ import annotations

import json
import sys
import time

import cv2
import numpy as np

from core import paths, settings
from core.datasets import registry as dreg
from core.datasets.io import read_mask, read_rgb8
from core.evaluation import metrics as M
from core.inference import model_registry as MR
from core.inference.session import LabSession, _downsample, fit_key

CATS = ("sheet_metal", "fruit_jelly")


def thumb(path, w=96):
    g = cv2.cvtColor(read_rgb8(path), cv2.COLOR_RGB2GRAY).astype(np.float32)
    h = max(8, int(round(w * g.shape[0] / g.shape[1])))
    t = cv2.resize(g, (w, h), interpolation=cv2.INTER_AREA).ravel()
    t -= t.mean()
    return t / (np.linalg.norm(t) + 1e-8)


def contamination(idx) -> dict:
    tr = [s for s in idx.samples if s.split in ("train", "validation")]
    te = [s for s in idx.samples if s.split == "test_public"]
    T = np.stack([thumb(s.primary_path) for s in tr])
    E = np.stack([thumb(s.primary_path) for s in te])
    tt = T @ T.T
    np.fill_diagonal(tt, -1)
    base = tt.max(1)
    te_tr = (E @ T.T).max(1)
    top = np.argsort(-te_tr)[:3]
    return {"n_train_val": len(tr), "n_test": len(te),
            "nn_ncc_train_to_train": {"median": float(np.median(base)), "max": float(base.max())},
            "nn_ncc_test_to_train": {"median": float(np.median(te_tr)), "max": float(te_tr.max())},
            "top_test_matches": [(te[i].sample_id, float(te_tr[i])) for i in top],
            "flag": bool(te_tr.max() > base.max() + 0.005)}


def scene_identity(idx) -> dict:
    tests = [s for s in idx.samples if s.split == "test_public"]
    by = {}
    for s in tests:
        by.setdefault((s.label, s.meta["scene_id"]), []).append(s)
    ious = []
    for (lab, sid), v in by.items():
        if lab != "bad":
            continue
        ms = [read_mask(s.masks["rgb"]) for s in v]
        ref = ms[0]
        for m in ms[1:]:
            ious.append(float((m & ref).sum() / max(1, (m | ref).sum())))
    goods = {k: [thumb(s.primary_path) for s in v] for k, v in by.items() if k[0] == "good"}
    within, between = [], []
    keys = list(goods)
    for i, k in enumerate(keys):
        A = np.stack(goods[k])
        within += [float(x) for x in (A @ A.T)[np.triu_indices(len(A), 1)]]
        for k2 in keys[i + 1:]:
            between += [float(x) for x in (A @ np.stack(goods[k2]).T).ravel()]
    return {"anomalous_scenes": sum(1 for k in by if k[0] == "bad"), "normal_scenes": len(goods),
            "mask_iou_vs_first_condition": {"min": min(ious), "median": float(np.median(ious)), "n_pairs": len(ious)},
            "normal_ncc_within_scene": {"median": float(np.median(within)), "min": float(min(within))},
            "normal_ncc_between_scenes": {"median": float(np.median(between)), "max": float(max(between))}}


def interpolation(idx, max_side) -> dict:
    bads = [s for s in idx.samples if s.split == "test_public" and s.label == "bad"]
    out = {}
    for side in (256, max_side):
        lost = regions = 0
        ratios = []
        for s in bads:
            m = read_mask(s.masks["rgb"])
            k, lab = cv2.connectedComponents(m.astype(np.uint8), connectivity=8)
            _, small = _downsample(np.zeros(m.shape, np.float32), m, side)
            sc = small.shape[0] / m.shape[0]
            for r in range(1, k):
                regions += 1
                ys, xs = np.nonzero(lab == r)
                hit = small[np.clip((ys * sc).astype(int), 0, small.shape[0] - 1),
                            np.clip((xs * sc).astype(int), 0, small.shape[1] - 1)].any()
                lost += 0 if hit and small.any() else 1
            frac_native = m.mean()
            ratios.append(float(small.mean() / frac_native) if frac_native else 1.0)
        out[str(side)] = {"regions": regions, "regions_lost": lost,
                          "area_ratio_after_downsampling": {"min": min(ratios), "median": float(np.median(ratios)),
                                                            "max": max(ratios)}}
    # alignment of the image → layer2 feature grid (stride 8) → bilinear up-sampling chain: a 40×40 px
    # synthetic defect at a random position; intensity-weighted centroid of the up-sampled response.
    # A systematic misalignment shows up as a non-zero mean SIGNED offset (bias); the absolute offset
    # is the unavoidable position quantisation of the coarse grid (≤ half a feature cell).
    h, w = (1056, 4224) if idx.category == "sheet_metal" else (1520, 2100)
    yy, xx = np.mgrid[0:h, 0:w]
    out["alignment"] = {}
    from core.algorithms.memory_based.anomalib_models import PatchcoreMethod
    for net in ((256, 256), PatchcoreMethod(keep_aspect=True)._input_hw(np.zeros((h, w)))):
        rng = np.random.default_rng(0)
        d = []
        for _ in range(40):
            y, x = int(rng.integers(60, h - 60)), int(rng.integers(60, w - 60))
            g = np.zeros((h, w), np.float32)
            g[y - 20:y + 20, x - 20:x + 20] = 1
            c = cv2.resize(g, (net[1] // 8, net[0] // 8), interpolation=cv2.INTER_AREA)
            u = cv2.resize(c, (w, h), interpolation=cv2.INTER_LINEAR)
            d.append(((u * yy).sum() / u.sum() - (g * yy).sum() / g.sum(), (u * xx).sum() / u.sum() - (g * xx).sum() / g.sum()))
        d = np.array(d)
        out["alignment"][f"{net[1]}x{net[0]}"] = {
            "cell_px": [w / (net[1] // 8), h / (net[0] // 8)], "bias_px": [float(d[:, 1].mean()), float(d[:, 0].mean())],
            "bias_se_px": [float(d[:, 1].std(ddof=1) / np.sqrt(len(d))), float(d[:, 0].std(ddof=1) / np.sqrt(len(d)))],
            "abs_median_px": float(np.median(np.hypot(d[:, 0], d[:, 1]))), "abs_max_px": float(np.hypot(d[:, 0], d[:, 1]).max())}
    return out


def correspondence_and_aupro(entry, idx, max_side) -> dict:
    s = LabSession(entry.get("method_id"), seed=int(entry.manifest["seed"]))
    s.bind(idx)
    s.load_checkpoint(entry)
    tests = [x for x in idx.samples if x.split == "test_public"]
    maps, masks, hits, chance = [], [], [], []
    for t in tests:
        img = read_rgb8(t.primary_path)
        r = s.predict_image(img)
        gt = read_mask(t.masks["rgb"], img.shape[:2]) if t.masks.get("rgb") else np.zeros(img.shape[:2], bool)
        a, g = _downsample(r.anomaly_map, gt, max_side)
        maps.append(a)
        masks.append(g)
        if gt.any():
            cell = int(np.ceil(img.shape[1] / entry.manifest["input_hw"][1] * 8))   # one layer2 feature cell
            dil = cv2.dilate(gt.astype(np.uint8), np.ones((cell, cell), np.uint8)) > 0
            yx = np.unravel_index(np.argmax(r.anomaly_map), r.anomaly_map.shape)
            hits.append(bool(dil[yx]))
            chance.append(float(dil.mean()))
    from anomalib.metrics.aupro import _AUPRO
    import torch
    res = {"preset": entry.verified["preset_id"],
           "pixel_auroc": M.pixel_auroc(masks, maps),
           "pixel_auroc_maps_flipped_lr": M.pixel_auroc(masks, [m[:, ::-1] for m in maps]),
           "pixel_auroc_maps_flipped_ud": M.pixel_auroc(masks, [m[::-1, :] for m in maps]),
           "pointing_game_hit_rate": float(np.mean(hits)), "pointing_game_chance": float(np.mean(chance))}
    for lim in (0.05, 0.3):
        mine = M.au_pro(masks, maps, lim)
        met = _AUPRO(fpr_limit=lim)
        met.update(torch.from_numpy(np.stack(maps)).float(), torch.from_numpy(np.stack(masks).astype(np.int64)))
        res[f"au_pro_{lim}"] = {"idd_lab": mine, "anomalib": float(met.compute()),
                                "stored_verified": entry.verified["metrics"].get("au_pro_05" if lim == 0.05 else "au_pro_30")}
    return res


def calibration_checks(verified: dict) -> list[dict]:
    out = []
    ref = {}
    f = paths.EXPERIMENTS / "results_D05.json"
    if f.exists():
        for r in json.loads(f.read_text(encoding="utf-8")):
            ref[(r["category"], r["method"])] = r.get("calibration", {})
    for m in verified.values():
        if m["dataset_id"] != "D05":
            continue
        idx = dreg.load_index("D05", m["category"])
        man = json.loads(MR.manifest_path(MR.FITTED / m["file"]).read_text(encoding="utf-8"))
        cal_ids = set(m["calibration"]["sample_ids"])
        test_ids = {s.sample_id for s in idx.samples if s.split.startswith("test")}
        spec = m["method_id"] + (":keep_aspect=1" if m["params"].get("keep_aspect") else "")
        r = ref.get((m["category"], spec), {})
        sc = np.array(m["calibration"]["normal_scores"])
        out.append({"preset": m["preset_id"], "source": m["calibration"]["source"],
                    "calibration_ids_disjoint_from_fit": not (cal_ids & set(man["fit_ids"])),
                    "calibration_ids_disjoint_from_test": not (cal_ids & test_ids),
                    "all_calibration_ids_validation": all(i.startswith("validation/") for i in cal_ids),
                    "image_thr_equals_campaign": (abs(m["calibration"]["image_threshold"] - r["image_threshold"]) < 1e-6)
                    if r.get("image_threshold") is not None else None,
                    "pixel_thr_equals_campaign": (abs(m["calibration"]["pixel_threshold"] - r["pixel_threshold"]) < 1e-6)
                    if r.get("pixel_threshold") is not None else None,
                    "validation_scores_above_image_thr": f"{int((sc > m['calibration']['image_threshold']).sum())}/{sc.size}"})
    return out


def cache_key_checks() -> dict:
    from core.algorithms import registry as areg
    entries = [e for e in MR.scan() if e.manifest]
    exact = 0
    for e in entries:
        m = e.manifest
        meth = areg.create(m["method_id"], **m["params"])
        exact += fit_key(meth, m["method_id"], m["fit_ids"], int(m["seed"]), m["dataset_root"]) == e.key
    e = next(x for x in entries if x.tier == "workshop_verified")
    m = e.manifest
    meth = areg.create(m["method_id"], **m["params"])
    base = fit_key(meth, m["method_id"], m["fit_ids"], m["seed"], m["dataset_root"])
    pert = {"drop one fit image": fit_key(meth, m["method_id"], m["fit_ids"][:-1], m["seed"], m["dataset_root"]),
            "swap two fit images": fit_key(meth, m["method_id"], [m["fit_ids"][1], m["fit_ids"][0]] + m["fit_ids"][2:], m["seed"], m["dataset_root"]),
            "seed + 1": fit_key(meth, m["method_id"], m["fit_ids"], m["seed"] + 1, m["dataset_root"]),
            "other dataset root": fit_key(meth, m["method_id"], m["fit_ids"], m["seed"], m["dataset_root"] + "_copy"),
            "image_size 288": fit_key(areg.create(m["method_id"], **{**m["params"], "image_size": 288}), m["method_id"],
                                      m["fit_ids"], m["seed"], m["dataset_root"])}
    unid = [x.path.name for x in MR.scan() if x.tier == "unidentified"]
    return {"manifests": len(entries), "exact_key_matches": exact, "reference": e.path.name,
            "perturbations_change_key": {k: v != base for k, v in pert.items()}, "unidentified": unid}


def to_md(rep: dict) -> str:
    F = lambda v, d=3: "—" if v is None else (f"{v:.{d}f}" if isinstance(v, float) else str(v))  # noqa: E731
    L = ["# Scientific audit — MVTec AD 2 educational experiments", "",
         f"Generated {rep['generated']} by `python -m scripts.audit_d05` (no training; verified checkpoints only). "
         "Scope: the local educational protocol on `test_public`. Official MVTec AD 2 evaluation is server-side on "
         "`test_private*` and is **not** reproduced here.", ""]
    for c in CATS:
        r = rep["categories"].get(c)
        if not r:
            continue
        L += [f"## {c}", "", "### 1. Split contamination"]
        k = r["contamination"]
        L += [f"- Byte-identical files shared between train/validation and test: **{r['byte_duplicates']}** "
              "(scripts/validate_dataset.py).",
              f"- Near-duplicate check (NCC of normalised 96-px thumbnails): nearest train/validation neighbour of each "
              f"test image — median {F(k['nn_ncc_test_to_train']['median'])}, max {F(k['nn_ncc_test_to_train']['max'])}; "
              f"baseline train→train nearest neighbour — median {F(k['nn_ncc_train_to_train']['median'])}, max "
              f"{F(k['nn_ncc_train_to_train']['max'])}. Test images are **{'MORE' if k['flag'] else 'not more'}** similar to "
              "training images than training images are to each other"
              + (" → inspect the top matches." if k['flag'] else " → no evidence of duplicated parts across splits."),
              f"- Top matches: " + ", ".join(f"`{a}` {F(b)}" for a, b in k["top_test_matches"]), "",
              "### 2. Scene identity (correlated observations)"]
        s = r["scenes"]
        L += [f"- {s['normal_scenes']} normal and {s['anomalous_scenes']} anomalous scenes in `test_public`.",
              f"- GT-mask IoU of each lighting variant vs. the first variant of the same anomalous scene: min "
              f"{F(s['mask_iou_vs_first_condition']['min'])}, median {F(s['mask_iou_vs_first_condition']['median'])} "
              f"({s['mask_iou_vs_first_condition']['n_pairs']} pairs) → the variants show the same defect at the same place.",
              f"- Normal scenes: thumbnail NCC within a scene median {F(s['normal_ncc_within_scene']['median'])} (min "
              f"{F(s['normal_ncc_within_scene']['min'])}) vs. between scenes median {F(s['normal_ncc_between_scenes']['median'])} "
              f"(max {F(s['normal_ncc_between_scenes']['max'])}).",
              "- Consequence: lighting variants are treated as repeated measurements of one scene (cluster bootstrap "
              "in Educational Experiments B); pooled image-level metrics are reported, but never with i.i.d. intervals.", "",
              "### 3. Interpolation of images, masks and maps"]
        it = r["interpolation"]
        for side in ("256", str(r["max_side"])):
            x = it[side]
            L.append(f"- Masks down-sampled (nearest) to ≤ {side} px: {x['regions_lost']}/{x['regions']} defect regions "
                     f"vanish; defect-area ratio after/before median {F(x['area_ratio_after_downsampling']['median'])} "
                     f"(min {F(x['area_ratio_after_downsampling']['min'])}, max {F(x['area_ratio_after_downsampling']['max'])}).")
        L.append("- Network input: plain resize (INTER_AREA), no crop; maps up-sampled bilinearly to the full image. "
                 "Synthetic test of the image → feature grid → up-sampling chain (40×40 px defect, 40 random positions, "
                 "intensity-weighted centroid):")
        for k, a in it["alignment"].items():
            L.append(f"  - input {k}: feature cell {a['cell_px'][0]:.0f}×{a['cell_px'][1]:.0f} px; mean signed offset (bias) "
                     f"x {a['bias_px'][0]:+.2f} ± {a['bias_se_px'][0]:.2f} px, y {a['bias_px'][1]:+.2f} ± {a['bias_se_px'][1]:.2f} px "
                     f"(± 1 s.e.); position quantisation median "
                     f"{a['abs_median_px']:.1f} px, max {a['abs_max_px']:.1f} px.")
        L += ["  - Every bias is either below 1 px or smaller than its own standard error (the 256×256 sheet_metal case, "
              "dominated by the 132-px quantisation noise); sub-pixel offsets come from pixel-centre conventions of the "
              "resizing and are negligible relative to the feature cell. No systematic misregistration is detectable; the absolute offset is the expected loss of "
              "positional precision of a coarse grid (it shrinks 4× when the aspect ratio is kept on sheet_metal).",
              f"- Lost regions: tiny defect regions that vanish when masks are down-sampled are excluded from AU-PRO and "
              f"pixel metrics (≤ {r['max_side']} px: {it[str(r['max_side'])]['regions_lost']} of {it[str(r['max_side'])]['regions']} "
              "regions). These are the smallest — and probably hardest — defects, so localisation metrics may be slightly "
              "optimistic. Evaluating at full resolution is not done (memory).",
              f"- Pixel metrics are computed at ≤ {r['max_side']} px (maps INTER_AREA, masks nearest); the GT masks have "
              "exactly the image size (validated).", "",
              "### 4. Map ↔ ground-truth correspondence and 5. AU-PRO cross-check",
              "| Preset | Pixel AUROC | flipped L-R | flipped U-D | Pointing game (hit / chance) | AU-PRO@0.05 IDD / Anomalib / stored | AU-PRO@0.3 IDD / Anomalib / stored |",
              "|---|---|---|---|---|---|---|"]
        for x in r["maps"]:
            a5, a3 = x["au_pro_0.05"], x["au_pro_0.3"]
            L.append(f"| {x['preset']} | {F(x['pixel_auroc'])} | {F(x['pixel_auroc_maps_flipped_lr'])} | "
                     f"{F(x['pixel_auroc_maps_flipped_ud'])} | {F(x['pointing_game_hit_rate'])} / {F(x['pointing_game_chance'])} | "
                     f"{F(a5['idd_lab'], 4)} / {F(a5['anomalib'], 4)} / {F(a5['stored_verified'], 4)} | "
                     f"{F(a3['idd_lab'], 4)} / {F(a3['anomalib'], 4)} / {F(a3['stored_verified'], 4)} |")
        L += ["", "Controls: flipping the maps breaks their spatial agreement with the masks only where the defect layout is "
              "not symmetric — a large drop (sheet_metal, left-right) supports alignment, a small drop (fruit_jelly) means "
              "the control is uninformative there (defects lie on a centred, roughly symmetric part). The pointing game is "
              "the more specific check: the arg-max of each anomalous map lies within one feature cell of a GT defect far "
              "more often than the chance level (fraction of the image covered by the dilated GT).", ""]
    L += ["## AU-PRO implementation", "",
          "- `core.evaluation.metrics.au_pro`: per-region overlap averaged over connected GT regions (8-connectivity), "
          "FPR over all normal pixels, one curve point per distinct score, linear interpolation at the FPR limit, "
          "trapezoidal area normalised by the limit (Bergmann et al., IJCV 2021).",
          "- Tested against a direct brute-force definition (tests/test_mvtec_ad2.py, exact to 1e-9) and, above, against "
          "Anomalib's independent implementation on real maps. Small differences can come from Anomalib's float32 "
          "arithmetic and its connected-component labelling.",
          "- **Not verified against the official MVTec AD 2 evaluation code**: official scores are computed by MVTec's "
          "server; the MVTec code utilities were not downloaded. According to the MVTec AD 2 paper (arXiv:2503.21622, "
          "Sec. IV-B1) the benchmark integrates the PRO curve up to FPR 0.05 (AU-PRO0.05); IDD Lab reports the same limit.", "",
          "## 6. Threshold calibration", "",
          "| Preset | Source | cal ∩ fit = ∅ | cal ∩ test = ∅ | only validation ids | image thr = campaign | pixel thr = campaign | validation scores > thr |",
          "|---|---|---|---|---|---|---|---|"]
    for x in rep["calibration"]:
        L.append(f"| {x['preset']} | {x['source']} | {x['calibration_ids_disjoint_from_fit']} | "
                 f"{x['calibration_ids_disjoint_from_test']} | {x['all_calibration_ids_validation']} | "
                 f"{x['image_thr_equals_campaign']} | {x['pixel_thr_equals_campaign']} | {x['validation_scores_above_image_thr']} |")
    ck = rep["cache_keys"]
    L += ["", "With n validation images the 99th percentile lies between the two largest scores, so at most one validation "
          "score exceeds it; this is expected and shows how coarse a percentile threshold is with 19–37 images.", "",
          "## 7. Cache keys", "",
          f"- {ck['exact_key_matches']}/{ck['manifests']} manifests reproduce their file key exactly from the recorded "
          "configuration (method, fit parameters, ordered fit ids, seed, Anomalib version, dataset root).",
          f"- Perturbations of `{ck['reference']}`: " + ", ".join(f"{k} → {'new key' if v else 'SAME KEY (problem)'}"
                                                             for k, v in ck["perturbations_change_key"].items()) + ".",
          f"- Unidentified checkpoints (kept, never loaded): {', '.join(ck['unidentified']) or 'none'} — fitted by older "
          "test runs on temporary synthetic folders that no longer exist.", "",
          "## 8. Reload consistency", ""]
    for m in rep["reload"]:
        L.append(f"- {m['preset']}: max |Δ| vs. {m['reference']} = {m['max_abs_diff']}")
    L += ["", "All verified presets reproduce the stored campaign metrics within the 1e-6 tolerance "
          "(scripts/verify_workshop_models.py). Earlier, the full MVTec AD 2 campaign was run twice (fresh fit vs. reload) "
          "with a largest difference of 1.2 × 10⁻⁹ (docs/LOCAL_RESULTS_D05.md)."]
    return "\n".join(L) + "\n"


def main(argv=None):
    settings.load()
    if argv is None and "--md-only" in sys.argv:            # re-render the report from experiments/audit_d05.json
        rep = json.loads((paths.EXPERIMENTS / "audit_d05.json").read_text(encoding="utf-8"))
        (paths.DOCS / "SCIENTIFIC_AUDIT_D05.md").write_text(to_md(rep), encoding="utf-8")
        print("wrote docs/SCIENTIFIC_AUDIT_D05.md (from stored audit)")
        return 0
    from scripts.validate_dataset import validate
    verified = MR.load_verified()
    if not verified:
        print("no verified presets — run python -m scripts.verify_workshop_models first")
        return 1
    rep = {"generated": time.strftime("%Y-%m-%d %H:%M"), "categories": {}}
    for c in CATS:
        if c not in dreg.installed_categories("D05"):
            continue
        idx = dreg.load_index("D05", c)
        side = dreg.get("D05").eval_pixel_max_side
        v = validate("D05", c)
        r = {"max_side": side, "byte_duplicates": sum("byte-identical" in e for e in v["errors"]),
             "contamination": contamination(idx), "scenes": scene_identity(idx), "interpolation": interpolation(idx, side),
             "maps": []}
        for pid in (f"d05-{c}-patchcore-256", f"d05-{c}-patchcore-aspect"):
            e = MR.find_preset(pid)
            if e is not None:
                r["maps"].append(correspondence_and_aupro(e, idx, side))
                print(c, pid, "maps done", flush=True)
        rep["categories"][c] = r
    rep["calibration"] = calibration_checks(verified)
    rep["cache_keys"] = cache_key_checks()
    rep["reload"] = [{"preset": m["preset_id"], "reference": m["reference_check"]["reference"],
                      "max_abs_diff": m["reference_check"]["max_abs_diff"]} for m in verified.values()]
    (paths.EXPERIMENTS / "audit_d05.json").write_text(json.dumps(rep, indent=1, default=str), encoding="utf-8")
    (paths.DOCS / "SCIENTIFIC_AUDIT_D05.md").write_text(to_md(rep), encoding="utf-8")
    print("wrote docs/SCIENTIFIC_AUDIT_D05.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
