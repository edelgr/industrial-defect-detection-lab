# Scientific audit — MVTec AD 2 educational experiments

Generated 2026-11-21 09:17 by `python -m scripts.audit_d05` (no training; verified checkpoints only). Scope: the local educational protocol on `test_public`. Official MVTec AD 2 evaluation is server-side on `test_private*` and is **not** reproduced here.

## sheet_metal

### 1. Split contamination
- Byte-identical files shared between train/validation and test: **0** (scripts/validate_dataset.py).
- Near-duplicate check (NCC of normalised 96-px thumbnails): nearest train/validation neighbour of each test image — median 0.719, max 0.790; baseline train→train nearest neighbour — median 0.707, max 0.967. Test images are **not more** similar to training images than training images are to each other → no evidence of duplicated parts across splits.
- Top matches: `test_public/bad/009_shift_2` 0.790, `test_public/bad/009_shift_1` 0.789, `test_public/bad/012_shift_2` 0.787

### 2. Scene identity (correlated observations)
- 4 normal and 15 anomalous scenes in `test_public`.
- GT-mask IoU of each lighting variant vs. the first variant of the same anomalous scene: min 0.976, median 1.000 (75 pairs) → the variants show the same defect at the same place.
- Normal scenes: thumbnail NCC within a scene median 0.933 (min 0.887) vs. between scenes median 0.648 (max 0.836).
- Consequence: lighting variants are treated as repeated measurements of one scene (cluster bootstrap in Educational Experiments B); pooled image-level metrics are reported, but never with i.i.d. intervals.

### 3. Interpolation of images, masks and maps
- Masks down-sampled (nearest) to ≤ 256 px: 69/450 defect regions vanish; defect-area ratio after/before median 1.013 (min 0.524, max 1.805).
- Masks down-sampled (nearest) to ≤ 1024 px: 6/450 defect regions vanish; defect-area ratio after/before median 0.999 (min 0.916, max 1.032).
- Network input: plain resize (INTER_AREA), no crop; maps up-sampled bilinearly to the full image. Synthetic test of the image → feature grid → up-sampling chain (40×40 px defect, 40 random positions, intensity-weighted centroid):
  - input 256x256: feature cell 132×33 px; mean signed offset (bias) x -3.46 ± 4.29 px, y +0.29 ± 0.20 px (± 1 s.e.); position quantisation median 25.5 px, max 45.0 px.
  - input 1024x256: feature cell 33×33 px; mean signed offset (bias) x -0.57 ± 0.20 px, y +0.29 ± 0.20 px (± 1 s.e.); position quantisation median 1.8 px, max 3.1 px.
  - Every bias is either below 1 px or smaller than its own standard error (the 256×256 sheet_metal case, dominated by the 132-px quantisation noise); sub-pixel offsets come from pixel-centre conventions of the resizing and are negligible relative to the feature cell. No systematic misregistration is detectable; the absolute offset is the expected loss of positional precision of a coarse grid (it shrinks 4× when the aspect ratio is kept on sheet_metal).
- Lost regions: tiny defect regions that vanish when masks are down-sampled are excluded from AU-PRO and pixel metrics (≤ 1024 px: 6 of 450 regions). These are the smallest — and probably hardest — defects, so localisation metrics may be slightly optimistic. Evaluating at full resolution is not done (memory).
- Pixel metrics are computed at ≤ 1024 px (maps INTER_AREA, masks nearest); the GT masks have exactly the image size (validated).

### 4. Map ↔ ground-truth correspondence and 5. AU-PRO cross-check
| Preset | Pixel AUROC | flipped L-R | flipped U-D | Pointing game (hit / chance) | AU-PRO@0.05 IDD / Anomalib / stored | AU-PRO@0.3 IDD / Anomalib / stored |
|---|---|---|---|---|---|---|
| d05-sheet_metal-patchcore-256 | 0.844 | 0.480 | 0.728 | 0.222 / 0.054 | 0.0492 / 0.0492 / 0.0492 | 0.1778 / 0.1778 / 0.1778 |
| d05-sheet_metal-patchcore-aspect | 0.909 | 0.588 | 0.796 | 0.567 / 0.017 | 0.1923 / 0.1923 / 0.1923 | 0.4950 / 0.4950 / 0.4950 |

Controls: flipping the maps breaks their spatial agreement with the masks only where the defect layout is not symmetric — a large drop (sheet_metal, left-right) supports alignment, a small drop (fruit_jelly) means the control is uninformative there (defects lie on a centred, roughly symmetric part). The pointing game is the more specific check: the arg-max of each anomalous map lies within one feature cell of a GT defect far more often than the chance level (fraction of the image covered by the dilated GT).

## fruit_jelly

### 1. Split contamination
- Byte-identical files shared between train/validation and test: **0** (scripts/validate_dataset.py).
- Near-duplicate check (NCC of normalised 96-px thumbnails): nearest train/validation neighbour of each test image — median 0.925, max 0.953; baseline train→train nearest neighbour — median 0.913, max 0.994. Test images are **not more** similar to training images than training images are to each other → no evidence of duplicated parts across splits.
- Top matches: `test_public/good/002_underexposed` 0.953, `test_public/bad/009_underexposed` 0.944, `test_public/bad/009_shift_1` 0.944

### 2. Scene identity (correlated observations)
- 5 normal and 15 anomalous scenes in `test_public`.
- GT-mask IoU of each lighting variant vs. the first variant of the same anomalous scene: min 0.991, median 1.000 (45 pairs) → the variants show the same defect at the same place.
- Normal scenes: thumbnail NCC within a scene median 0.997 (min 0.983) vs. between scenes median 0.736 (max 0.924).
- Consequence: lighting variants are treated as repeated measurements of one scene (cluster bootstrap in Educational Experiments B); pooled image-level metrics are reported, but never with i.i.d. intervals.

### 3. Interpolation of images, masks and maps
- Masks down-sampled (nearest) to ≤ 256 px: 40/228 defect regions vanish; defect-area ratio after/before median 1.002 (min 0.915, max 1.101).
- Masks down-sampled (nearest) to ≤ 1024 px: 8/228 defect regions vanish; defect-area ratio after/before median 1.001 (min 0.988, max 1.023).
- Network input: plain resize (INTER_AREA), no crop; maps up-sampled bilinearly to the full image. Synthetic test of the image → feature grid → up-sampling chain (40×40 px defect, 40 random positions, intensity-weighted centroid):
  - input 256x256: feature cell 66×48 px; mean signed offset (bias) x -0.87 ± 1.13 px, y +0.60 ± 0.35 px (± 1 s.e.); position quantisation median 5.6 px, max 12.5 px.
  - input 352x256: feature cell 48×48 px; mean signed offset (bias) x +0.11 ± 0.34 px, y +0.60 ± 0.35 px (± 1 s.e.); position quantisation median 3.1 px, max 5.3 px.
  - Every bias is either below 1 px or smaller than its own standard error (the 256×256 sheet_metal case, dominated by the 132-px quantisation noise); sub-pixel offsets come from pixel-centre conventions of the resizing and are negligible relative to the feature cell. No systematic misregistration is detectable; the absolute offset is the expected loss of positional precision of a coarse grid (it shrinks 4× when the aspect ratio is kept on sheet_metal).
- Lost regions: tiny defect regions that vanish when masks are down-sampled are excluded from AU-PRO and pixel metrics (≤ 1024 px: 8 of 228 regions). These are the smallest — and probably hardest — defects, so localisation metrics may be slightly optimistic. Evaluating at full resolution is not done (memory).
- Pixel metrics are computed at ≤ 1024 px (maps INTER_AREA, masks nearest); the GT masks have exactly the image size (validated).

### 4. Map ↔ ground-truth correspondence and 5. AU-PRO cross-check
| Preset | Pixel AUROC | flipped L-R | flipped U-D | Pointing game (hit / chance) | AU-PRO@0.05 IDD / Anomalib / stored | AU-PRO@0.3 IDD / Anomalib / stored |
|---|---|---|---|---|---|---|
| d05-fruit_jelly-patchcore-256 | 0.947 | 0.933 | 0.793 | 0.667 / 0.033 | 0.2750 / 0.2750 / 0.2750 | 0.6145 / 0.6145 / 0.6145 |
| d05-fruit_jelly-patchcore-aspect | 0.943 | 0.926 | 0.797 | 0.683 / 0.027 | 0.2878 / 0.2878 / 0.2878 | 0.6285 / 0.6285 / 0.6285 |

Controls: flipping the maps breaks their spatial agreement with the masks only where the defect layout is not symmetric — a large drop (sheet_metal, left-right) supports alignment, a small drop (fruit_jelly) means the control is uninformative there (defects lie on a centred, roughly symmetric part). The pointing game is the more specific check: the arg-max of each anomalous map lies within one feature cell of a GT defect far more often than the chance level (fraction of the image covered by the dilated GT).

## AU-PRO implementation

- `core.evaluation.metrics.au_pro`: per-region overlap averaged over connected GT regions (8-connectivity), FPR over all normal pixels, one curve point per distinct score, linear interpolation at the FPR limit, trapezoidal area normalised by the limit (Bergmann et al., IJCV 2021).
- Tested against a direct brute-force definition (tests/test_mvtec_ad2.py, exact to 1e-9) and, above, against Anomalib's independent implementation on real maps. Small differences can come from Anomalib's float32 arithmetic and its connected-component labelling.
- **Not verified against the official MVTec AD 2 evaluation code**: official scores are computed by MVTec's server; the MVTec code utilities were not downloaded. According to the MVTec AD 2 paper (arXiv:2503.21622, Sec. IV-B1) the benchmark integrates the PRO curve up to FPR 0.05 (AU-PRO0.05); IDD Lab reports the same limit.

## 6. Threshold calibration

| Preset | Source | cal ∩ fit = ∅ | cal ∩ test = ∅ | only validation ids | image thr = campaign | pixel thr = campaign | validation scores > thr |
|---|---|---|---|---|---|---|---|
| d05-sheet_metal-patchcore-256 | validation/good, p99 image / p99.5 pixel, n=19 | True | True | True | True | True | 1/19 |
| d05-sheet_metal-padim-256 | validation/good, p99 image / p99.5 pixel, n=19 | True | True | True | True | True | 1/19 |
| d05-sheet_metal-efficientad-256 | validation/good, p99 image / p99.5 pixel, n=19 | True | True | True | True | True | 1/19 |
| d05-sheet_metal-anomalydino-448 | validation/good, p99 image / p99.5 pixel, n=19 | True | True | True | True | True | 1/19 |
| d05-sheet_metal-patchcore-aspect | validation/good, p99 image / p99.5 pixel, n=19 | True | True | True | True | True | 1/19 |
| d05-sheet_metal-padim-aspect | validation/good, p99 image / p99.5 pixel, n=19 | True | True | True | True | True | 1/19 |
| d05-fruit_jelly-patchcore-256 | validation/good, p99 image / p99.5 pixel, n=37 | True | True | True | True | True | 1/37 |
| d05-fruit_jelly-padim-256 | validation/good, p99 image / p99.5 pixel, n=37 | True | True | True | True | True | 1/37 |
| d05-fruit_jelly-efficientad-256 | validation/good, p99 image / p99.5 pixel, n=37 | True | True | True | True | True | 1/37 |
| d05-fruit_jelly-anomalydino-448 | validation/good, p99 image / p99.5 pixel, n=37 | True | True | True | True | True | 1/37 |
| d05-fruit_jelly-patchcore-aspect | validation/good, p99 image / p99.5 pixel, n=37 | True | True | True | True | True | 1/37 |
| d05-fruit_jelly-padim-aspect | validation/good, p99 image / p99.5 pixel, n=37 | True | True | True | True | True | 1/37 |

With n validation images the 99th percentile lies between the two largest scores, so at most one validation score exceeds it; this is expected and shows how coarse a percentile threshold is with 19–37 images.

## 7. Cache keys

- 27/27 manifests reproduce their file key exactly from the recorded configuration (method, fit parameters, ordered fit ids, seed, Anomalib version, dataset root).
- Perturbations of `anomalydino_D05_fruit_jelly_5e973935f3c2876d.pt`: drop one fit image → new key, swap two fit images → new key, seed + 1 → new key, other dataset root → new key, image_size 288 → new key.
- Unidentified checkpoints (kept, never loaded): padim_SYN_brushed_metal_43aa183962a9a9fa.pt, padim_SYN_brushed_metal_6a3bb384c7247a05.pt, padim_SYN_brushed_metal_6bd0e046b77d9373.pt — fitted by older test runs on temporary synthetic folders that no longer exist.

## 8. Reload consistency

- d05-sheet_metal-patchcore-256: max |Δ| vs. experiments/results_D05.json (patchcore) = 0.0
- d05-sheet_metal-padim-256: max |Δ| vs. experiments/results_D05.json (padim) = 0.0
- d05-sheet_metal-efficientad-256: max |Δ| vs. experiments/results_D05.json (efficientad) = 0.0
- d05-sheet_metal-anomalydino-448: max |Δ| vs. experiments/results_D05.json (anomalydino) = 0.0
- d05-sheet_metal-patchcore-aspect: max |Δ| vs. experiments/results_D05.json (patchcore:keep_aspect=1) = 0.0
- d05-sheet_metal-padim-aspect: max |Δ| vs. experiments/results_D05.json (padim:keep_aspect=1) = 7.329090667695937e-10
- d05-fruit_jelly-patchcore-256: max |Δ| vs. experiments/results_D05.json (patchcore) = 0.0
- d05-fruit_jelly-padim-256: max |Δ| vs. experiments/results_D05.json (padim) = 0.0
- d05-fruit_jelly-efficientad-256: max |Δ| vs. experiments/results_D05.json (efficientad) = 0.0
- d05-fruit_jelly-anomalydino-448: max |Δ| vs. experiments/results_D05.json (anomalydino) = 0.0
- d05-fruit_jelly-patchcore-aspect: max |Δ| vs. experiments/results_D05.json (patchcore:keep_aspect=1) = 0.0
- d05-fruit_jelly-padim-aspect: max |Δ| vs. experiments/results_D05.json (padim:keep_aspect=1) = 7.285084757668869e-10
- d06-capsule-patchcore-256: max |Δ| vs. experiments.sqlite run 102616af1a69 = 0.0
- d06-capsule-padim-256: max |Δ| vs. experiments.sqlite run bcface3b3fc8 = 1.7785495298738851e-10
- d06-capsule-efficientad-256: max |Δ| vs. experiments.sqlite run 0be4e9248bf8 = 0.0
- d06-capsule-anomalydino-448: max |Δ| vs. experiments.sqlite run e7318226f639 = 0.0
- d06-nut-patchcore-256: max |Δ| vs. experiments.sqlite run cef549f8281c = 0.0
- d06-nut-padim-256: max |Δ| vs. experiments.sqlite run f7e2590853bb = 2.8872948476532656e-10
- d06-nut-efficientad-256: max |Δ| vs. experiments.sqlite run cf6b1dfec06b = 0.0
- d06-nut-anomalydino-448: max |Δ| vs. experiments.sqlite run 9b02a9d0362a = 0.0
- d06-flat_pad-patchcore-256: max |Δ| vs. experiments.sqlite run 60b3e3751534 = 0.0
- d06-flat_pad-padim-256: max |Δ| vs. experiments.sqlite run d95a233043e2 = 1.846776065406175e-10
- d06-flat_pad-efficientad-256: max |Δ| vs. experiments.sqlite run a0c3e88fa2e9 = 0.0
- d06-flat_pad-anomalydino-448: max |Δ| vs. experiments.sqlite run db9090ce3331 = 0.0

All verified presets reproduce the stored campaign metrics within the 1e-6 tolerance (scripts/verify_workshop_models.py). Earlier, the full MVTec AD 2 campaign was run twice (fresh fit vs. reload) with a largest difference of 1.2 × 10⁻⁹ (docs/LOCAL_RESULTS_D05.md).

## Note

The checkpoints listed above as *unidentified* were later identified as synthetic test artefacts of the
test suite and removed from the local installation. They were never part of any reported result; every
other check above is unaffected. No checkpoint is distributed with this repository.

## Addendum (2026-11-21, Educational Experiments B redesign — no metric changed)

The illumination experiment (Educational Experiments → B) was redesigned (Basic / Advanced modes). Before
changing the interface, the data path behind it was re-checked; every check passed and **no stored metric,
threshold, checkpoint or dataset was changed**.

| Check (8 D05 presets without `keep_aspect`) | Result |
|---|---|
| Stored sample ↔ dataset index (`test_public`): sample id, label, `scene_id`, `lighting_condition` | 0 mismatches; sheet_metal 114 images = 19 scenes × 6 conditions, fruit_jelly 80 = 20 × 4 (no duplicated condition in a scene) |
| Checkpoint ↔ stored result | sha256 of every checkpoint file = stored `sha256`; registry key = stored key |
| Threshold used by the page | `calibration.image_threshold` of the stored result; equals the threshold of the loaded LIVE session (checked for PatchCore sheet_metal and PaDiM fruit_jelly; pixel threshold equal as well) |
| Decision rule | `score ≥ threshold → NOK`, identical to `metrics.decision_metrics`; recomputed TP/FP/TN/FN = stored `metrics.decision` for all 8 presets (also a regression test) |
| Per-condition and pooled AUROC | recomputed values = stored `metrics.by_condition` / `image_auroc` (exact) |
| LIVE vs stored scores | 12 images (PatchCore sheet_metal) and 8 images (PaDiM fruit_jelly), plus the scenes computed during the GUI validation: max \|Δ\| = 0 |

**Correlated observations and intervals.** Lighting variants of a scene are not independent; the page keeps
the scene-level cluster bootstrap (`core/evaluation/scenes.py`, 1000 resamples, seed 0) and shows the naive
image-level interval only for contrast. With 4 (sheet_metal) or 5 (fruit_jelly) normal scenes every preset is
in the *very limited* sufficiency level, and the cluster bootstrap has only 251–496 distinct resampled AUROC
values out of 1000. In **2 of 8 presets the cluster interval is not wider than the naive one** — EfficientAD
sheet_metal (width 0.205 vs 0.232) and PaDiM fruit_jelly (0.204 vs 0.263); in the other 6 it is wider. This is
a property of the percentile bootstrap with very few clusters (it underestimates the uncertainty), not an
implementation error; the page now states it explicitly for the selected model instead of only saying
"usually too narrow", and no interval was altered.

**Interface discrepancies corrected (presentation only).**
* The former banner read "LIVE — inference runs on this computer" for the whole tab although scores, decisions,
  AUROCs and intervals always came from the stored results; only the anomaly maps were computed live. The tab
  now shows two separate labels (STORED / LIVE), and each live map shows its live score next to the stored one.
* The former line plot connected categorical lighting conditions without saying so; the Advanced chart now
  states that lines only join the same scene, and threshold crossings are drawn as markers on the threshold,
  not as data.
