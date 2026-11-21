# Experiment Protocols

## Result categories (never mixed)
| Category | Definition | Requirement |
|---|---|---|
| `author_reported` | Numbers quoted from a publication | must cite `publication_id` (enforced by the store) |
| `reproduced` | Official protocol re-run completely (all data, official splits & metric code) | none produced in v0.1 |
| `local_educational` | Local experiment: subset, local thresholds, IDD Lab metrics | default for real data |
| `synthetic_demo` | Any result on SYNTHETIC data | automatic for the synthetic adapter |

## IDD Lab local protocol (`core/inference/session.py`)
1. **Fit**: up to *N* `train` images with label `good`, deterministic shuffle with the seed
   (default 42). If the dataset has no validation split, 15 % of train/good (≥ 2 images) is held
   out first and never used for fitting.
2. **Calibrate** (label-free): run the method on normal images *not used for fitting* — the
   `validation/good` split if present, else the held-out train images. Image threshold = 99th
   percentile of their scores; pixel threshold = 99.5th percentile of their map values.
3. **Evaluate**: only splits starting with `test` whose labels are public. Optional stratified
   subsampling (seeded). MVTec AD 2 `test_private*` → refused.
4. **Metrics**: image AUROC and AP (scikit-learn); pixel AUROC on maps down-sampled to the
   adapter's `eval_pixel_max_side` (256 px by default, **1024 px for MVTec AD 2**, whose defects
   cover a median 0.25–0.4 % of the image) (all test images, uniform sub-sampling to 4 M pixels if
   larger); optionally (`localization=True`, always in Methods Laboratory) AU-PRO up to FPR 0.3 and
   0.05 (Bergmann et al., IJCV 2021) and pooled pixel F1 at the calibrated pixel threshold; GPU peak
   memory during inference; image AUROC per MVTec AD 2 lighting condition; Dice/IoU on anomalous images at
   the calibrated pixel threshold; precision, recall, F1, false-acceptance rate (defective accepted)
   and false-rejection rate (good rejected) at the calibrated image threshold; latency mean/median/p95.
5. **Persist**: run metadata (method version, parameters, seed, environment snapshot, protocol
   string, category) and per-sample predictions in `experiments/experiments.sqlite`.

Thresholds are never chosen from test labels; decision metrics record `threshold_source` and
`threshold_from_test = false`.

## Fitted-model cache
Fitted states (PaDiM Gaussians, PatchCore and AnomalyDINO memory banks, EfficientAD
student/autoencoder/normalisation) are saved to `checkpoints/fitted/` under a key that hashes the
method id, fit-relevant parameters, the **exact list of fit sample ids**, the seed, the dataset
root and the Anomalib version. A cached model is reused only if all of these match, so loading
from cache is equivalent to re-fitting (verified by `tests/test_fit_cache.py`: identical maps
and scores). This lets the instructor pre-train before the workshop.

## Model Registry and Workshop Verified presets
- Every checkpoint in `checkpoints/fitted/` has a JSON manifest (method, all parameters, dataset,
  category, dataset root, **ordered fit sample ids**, seed, Anomalib version, network input size,
  fit date, provenance). Manifests of checkpoints created before the registry were *reconstructed*
  by searching the configuration that reproduces the file key exactly (27/27 real checkpoints
  identified; 3 test artefacts on deleted temporary folders remain *unidentified* and are never loaded).
- Loading recomputes the key from the manifest and refuses any mismatch, a different dataset root,
  category or Anomalib version, or fit ids that are not train/good images of the current index.
- Tiers: **Workshop Verified** (listed in `configs/workshop_verified.json` after reload + re-evaluation
  reproduced the stored campaign metrics within 1e-6; SHA-256 recorded), **Quick Experiment**
  (default fit parameters, ≤ 60 fit images), **Custom** (anything else).
- Verified presets store their validation-derived thresholds; loading restores them (marked
  *stored with verified preset*), so the workshop needs no calibration pass.

## Scene-aware statistics (MVTec AD 2 `test_public`)
Lighting variants of one scene are repeated measurements, not independent samples
(`core/evaluation/scenes.py`). Per-condition AUROC uses one image per scene; scene-level AUROC
uses one aggregated score per scene; 95 % percentile intervals come from a **cluster bootstrap**
that resamples scenes within each class (no interval below 3 scenes per class; flagged as very
limited below 10). The naive image-level bootstrap is shown only to illustrate the error.

## Alternative label-free threshold rule
Besides IDD Lab's p99 (image) / p99.5 (pixel) of validation normals, the experiments show the MVTec
AD 2 paper's baseline rule (Sec. IV-B2): pixel threshold = mean + 3 σ of the per-pixel scores of the
defect-free validation images (estimated from the same uniform 20 000-pixel sample per image), used
also for image decisions with the map maximum as image score. Neither rule uses test labels.
Moving the threshold on test scores (Educational Experiments C) is labelled *retrospective
exploration* and is never stored as a result.

## Preprocessing
Deep methods resize to a square input (PaDiM/PatchCore 256 px, AnomalyDINO 448 px, WinCLIP
240 px) **without centre-crop**, then up-sample the map to the original resolution. This keeps
maps aligned with ground truth but differs from some papers' crop protocols — one reason local
numbers differ from author-reported ones.

PaDiM and PatchCore also accept `keep_aspect` (default off): the shorter side is resized to the
input size and the longer side scaled with the image aspect ratio (multiple of 32, ≤ 4×) — e.g.
MVTec AD 2 `sheet_metal` 4224×1056 → 1024×256 instead of being squashed to 256×256. New
parameters such as this one are left out of the cache key while at their default
(`ParamSpec.key_omit_default`), so caches fitted before the parameter existed stay valid.

## MVTec AD 2 local protocol (`sheet_metal`, `fruit_jelly`)
- **Fit**: all `train/good` images (137 / 263); AnomalyDINO uses the first k = 4 of the same seeded
  order (few-shot); EfficientAD keeps the first 10 % of its fit images for its internal map
  quantiles (still train data).
- **Calibrate**: `validation/good` (19 / 37 images), never used for fitting.
- **Evaluate**: `test_public` only (114 = 24 good + 90 bad; 80 = 20 + 60). `test_private` and
  `test_private_mixed` are indexed as `unknown`, cannot be selected for evaluation and are
  never scored: official MVTec AD 2 results come only from MVTec's evaluation server.
- `test_public` re-photographs the same scenes under several lighting conditions (file suffix
  `_regular`, `_overexposed`, `_underexposed`, `_shift_k`), while train/validation are `regular`
  only; the adapter records `scene_id` and `lighting_condition` per sample. With only 4–5 normal
  scenes, per-condition AUROC is reported as a teaching aid, not as a robust estimate.
- Integrity is checked by `python -m scripts.validate_dataset --dataset D05`
  ([docs/DATASET_VALIDATION_D05.md](docs/DATASET_VALIDATION_D05.md)).
- Results: [docs/LOCAL_RESULTS_D05.md](docs/LOCAL_RESULTS_D05.md) — category *local educational*.

## Multimodal protocol (MulSen-AD)
See `core/algorithms/multimodal_3d/fusion.py`: per-modality detectors fitted on train/good,
z-normalisation on held-out train/good, mean/max late fusion, AUROC vs. object and per-modality
labels. It is not the MulSen-TripleAD method.

## Reproducing
```powershell
.\.venv\Scripts\python.exe -m scripts.run_experiments --dataset D06 --categories capsule nut flat_pad --fusion
```
writes `docs/LOCAL_RESULTS.md` (environment and command included).

```powershell
.\.venv\Scripts\python.exe -m scripts.run_experiments --dataset D05 --categories sheet_metal fruit_jelly `
    --methods patchcore padim efficientad anomalydino "patchcore:keep_aspect=1" "padim:keep_aspect=1" `
    --n-fit 1000 --n-cal 60 --pixel-max-side 1024 --localization
```
writes `docs/LOCAL_RESULTS_D05.md` and `experiments/results_D05.json` (≈ 25 min on an RTX 4090
Laptop GPU when nothing is cached, ≈ 5 min with all fitted models cached).
