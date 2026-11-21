# Local Results — MVTec AD 2 (D05)

> **Category: local educational results.** IDD Lab local protocol: fit only on `train/good`, label-free thresholds from `validation/good`, evaluation only on the publicly labelled test split. **These are not official benchmark results.** For MVTec AD 2, official scores are computed by MVTec's evaluation server on `test_private` / `test_private_mixed`, whose labels are withheld; IDD Lab never evaluates those splits. Do not compare these numbers with leaderboard or paper numbers without stating the protocol differences.

Generated 2026-11-21 06:58 · seed 42 · fit ≤ 1000 images · calibration ≤ 60 images · pixel metrics at ≤ 1024 px · command: `python -m scripts.run_experiments --dataset D05 --categories sheet_metal fruit_jelly --methods patchcore padim efficientad anomalydino patchcore:keep_aspect=1 padim:keep_aspect=1 --n-fit 1000 --n-cal 60 --pixel-max-side 1024 --localization`

Environment: python 3.12.6, platform Windows-11-10.0.26300-SP0, numpy 2.5.4, cv2 5.1.0, torch 2.14.1+cu130, torchvision 0.29.1+cu130, anomalib 2.7.1, sklearn 1.9.1, skimage 0.26.0, PySide6 6.12.0, open3d 0.20.0, cuda_device NVIDIA GeForce RTX 4090 Laptop GPU, cuda_runtime 13.0

## Detection and localisation

| Category | Method | Model / input | n fit | Image AUROC | Image AP | Pixel AUROC | AU-PRO@0.3 | AU-PRO@0.05 | Pixel F1 (cal.) | Image F1 (cal.) | Latency mean / p95 (ms) | GPU peak |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| sheet_metal | patchcore | Anomalib 2.7.1 PatchcoreModel (wide_resnet50_2, layer2+layer3) · input 256×256 | 137 | 0.741 | 0.917 | 0.844 | 0.178 | 0.049 | 0.100 | 0.417 | 42.0 / 63.0 | 278 MB |
| sheet_metal | padim | Anomalib 2.7.1 PadimModel (resnet18, layers 1-3) · input 256×256 | 137 | 0.630 | 0.881 | 0.806 | 0.194 | 0.045 | 0.075 | 0.492 | 30.6 / 37.5 | 194 MB |
| sheet_metal | efficientad | Anomalib 2.7.1 EfficientAdModel (small, 10000 steps, Imagenette2-320 penalty) | 137 | 0.677 | 0.883 | 0.802 | 0.165 | 0.050 | 0.113 | 0.762 | 43.0 / 50.6 | 107 MB |
| sheet_metal | anomalydino | Anomalib 2.7.1 AnomalyDINOModel (vit_small_patch14_dinov2, k=4) | 137 | 0.741 | 0.929 | 0.896 | 0.277 | 0.145 | 0.099 | 0.739 | 46.5 / 54.1 | 150 MB |
| sheet_metal | patchcore:keep_aspect=1 | Anomalib 2.7.1 PatchcoreModel (wide_resnet50_2, layer2+layer3) · input 1024×256 | 137 | 0.764 | 0.928 | 0.909 | 0.495 | 0.192 | 0.208 | 0.657 | 73.6 / 81.8 | 802 MB |
| sheet_metal | padim:keep_aspect=1 | Anomalib 2.7.1 PadimModel (resnet18, layers 1-3) · input 1024×256 | 137 | 0.562 | 0.846 | 0.890 | 0.325 | 0.056 | 0.073 | 0.649 | 43.5 / 70.1 | 718 MB |
| fruit_jelly | patchcore | Anomalib 2.7.1 PatchcoreModel (wide_resnet50_2, layer2+layer3) · input 256×256 | 263 | 0.777 | 0.928 | 0.947 | 0.614 | 0.275 | 0.324 | 0.571 | 34.5 / 43.2 | 431 MB |
| fruit_jelly | padim | Anomalib 2.7.1 PadimModel (resnet18, layers 1-3) · input 256×256 | 263 | 0.732 | 0.887 | 0.917 | 0.441 | 0.051 | 0.037 | 0.000 | 28.6 / 33.4 | 194 MB |
| fruit_jelly | efficientad | Anomalib 2.7.1 EfficientAdModel (small, 10000 steps, Imagenette2-320 penalty) | 263 | 0.784 | 0.929 | 0.839 | 0.178 | 0.156 | 0.460 | 0.421 | 36.2 / 44.4 | 107 MB |
| fruit_jelly | anomalydino | Anomalib 2.7.1 AnomalyDINOModel (vit_small_patch14_dinov2, k=4) | 263 | 0.767 | 0.929 | 0.917 | 0.684 | 0.312 | 0.301 | 0.033 | 37.9 / 43.4 | 150 MB |
| fruit_jelly | patchcore:keep_aspect=1 | Anomalib 2.7.1 PatchcoreModel (wide_resnet50_2, layer2+layer3) · input 352×256 | 263 | 0.775 | 0.924 | 0.943 | 0.628 | 0.288 | 0.314 | 0.235 | 38.6 / 45.0 | 553 MB |
| fruit_jelly | padim:keep_aspect=1 | Anomalib 2.7.1 PadimModel (resnet18, layers 1-3) · input 352×256 | 263 | 0.831 | 0.933 | 0.913 | 0.480 | 0.102 | 0.001 | 0.000 | 26.9 / 31.9 | 259 MB |

## Image AUROC per lighting condition (`test_public`)

| Category | Method | overexposed | regular | shift_1 | shift_2 | shift_3 | underexposed |
|---|---|---|---|---|---|---|---|
| sheet_metal | patchcore | 0.900 (15/4) | 0.850 (15/4) | 0.817 (15/4) | 0.800 (15/4) | 0.867 (15/4) | 0.817 (15/4) |
| sheet_metal | padim | 0.600 (15/4) | 0.617 (15/4) | 0.667 (15/4) | 0.650 (15/4) | 0.550 (15/4) | 0.817 (15/4) |
| sheet_metal | efficientad | 0.733 (15/4) | 0.800 (15/4) | 0.867 (15/4) | 0.800 (15/4) | 0.967 (15/4) | 0.717 (15/4) |
| sheet_metal | anomalydino | 0.750 (15/4) | 0.733 (15/4) | 0.783 (15/4) | 0.733 (15/4) | 0.800 (15/4) | 0.750 (15/4) |
| sheet_metal | patchcore:keep_aspect=1 | 0.783 (15/4) | 0.750 (15/4) | 0.817 (15/4) | 0.817 (15/4) | 0.800 (15/4) | 0.717 (15/4) |
| sheet_metal | padim:keep_aspect=1 | 0.767 (15/4) | 0.583 (15/4) | 0.500 (15/4) | 0.617 (15/4) | 0.417 (15/4) | 0.567 (15/4) |
| fruit_jelly | patchcore | 0.813 (15/5) | 0.747 (15/5) | 0.747 (15/5) | — | — | 0.800 (15/5) |
| fruit_jelly | padim | 0.400 (15/5) | 0.987 (15/5) | 0.867 (15/5) | — | — | 0.893 (15/5) |
| fruit_jelly | efficientad | 0.693 (15/5) | 0.840 (15/5) | 0.880 (15/5) | — | — | 0.920 (15/5) |
| fruit_jelly | anomalydino | 0.747 (15/5) | 0.773 (15/5) | 0.720 (15/5) | — | — | 0.800 (15/5) |
| fruit_jelly | patchcore:keep_aspect=1 | 0.853 (15/5) | 0.720 (15/5) | 0.760 (15/5) | — | — | 0.773 (15/5) |
| fruit_jelly | padim:keep_aspect=1 | 0.600 (15/5) | 0.893 (15/5) | 0.920 (15/5) | — | — | 0.947 (15/5) |

Cell = image AUROC (n anomalous / n normal). Every condition re-photographs the *same* few scenes (sheet_metal: 15 defective + 4 normal; fruit_jelly: 15 + 5), so per-condition AUROC rests on 4–5 normal images and is very noisy. Training and validation images are all `regular`.

## Fitting and calibration

| Category | Method | Fitted / loaded | Fit wall time (s) | Calibration source | Image thr. | Pixel thr. |
|---|---|---|---|---|---|---|
| sheet_metal | patchcore | loaded cache `patchcore_D05_sheet_metal_fe15a818a5e3fc75.pt` | 5.2 | validation/good, p99 image / p99.5 pixel, n=19 | 37.799 | 32.076 |
| sheet_metal | padim | loaded cache `padim_D05_sheet_metal_2dcb9f4332419e45.pt` | 0.6 | validation/good, p99 image / p99.5 pixel, n=19 | 42.249 | 21.479 |
| sheet_metal | efficientad | loaded cache `efficientad_D05_sheet_metal_5708efdb1c656fdc.pt` | 0.1 | validation/good, p99 image / p99.5 pixel, n=19 | 0.268 | 0.066 |
| sheet_metal | anomalydino | loaded cache `anomalydino_D05_sheet_metal_9670e08cacea1cd1.pt` | 0.7 | validation/good, p99 image / p99.5 pixel, n=19 | 0.165 | 0.114 |
| sheet_metal | patchcore:keep_aspect=1 | loaded cache `patchcore_D05_sheet_metal_f6ed0f988ae392f3.pt` | 1.4 | validation/good, p99 image / p99.5 pixel, n=19 | 46.197 | 39.043 |
| sheet_metal | padim:keep_aspect=1 | fitted → cached | 9.6 | validation/good, p99 image / p99.5 pixel, n=19 | 65.858 | 29.699 |
| fruit_jelly | patchcore | loaded cache `patchcore_D05_fruit_jelly_a00f2e153f73d3a1.pt` | 1.5 | validation/good, p99 image / p99.5 pixel, n=37 | 51.485 | 42.532 |
| fruit_jelly | padim | loaded cache `padim_D05_fruit_jelly_fbc34942d6588a83.pt` | 0.6 | validation/good, p99 image / p99.5 pixel, n=37 | 114.916 | 58.344 |
| fruit_jelly | efficientad | loaded cache `efficientad_D05_fruit_jelly_efa9f321a0658c12.pt` | 0.1 | validation/good, p99 image / p99.5 pixel, n=37 | 5.372 | 0.044 |
| fruit_jelly | anomalydino | loaded cache `anomalydino_D05_fruit_jelly_5e973935f3c2876d.pt` | 0.6 | validation/good, p99 image / p99.5 pixel, n=37 | 0.442 | 0.275 |
| fruit_jelly | patchcore:keep_aspect=1 | fitted → cached | 117.9 | validation/good, p99 image / p99.5 pixel, n=37 | 57.247 | 42.897 |
| fruit_jelly | padim:keep_aspect=1 | fitted → cached | 11.9 | validation/good, p99 image / p99.5 pixel, n=37 | 127.314 | 63.561 |

## Metric definitions

- **Image AUROC / AP**: raw image scores against the public image labels (threshold-free).
- **Pixel AUROC**: all test pixels pooled (uniformly sub-sampled to 4 M pixels if larger).
- **AU-PRO@x**: area under the per-region-overlap curve up to false-positive rate x, normalised to [0, 1] (Bergmann et al., IJCV 2021), at the down-sampled evaluation resolution.
- **Pixel F1 (cal.)**: pixel-level F1 pooled over all test pixels at the pixel threshold fixed on `validation/good` (99.5th percentile of normal pixel scores); no test label is used to choose it. Analogous in spirit to, but not the same computation as, the official server-side SegF1.
- **Image F1 (cal.)**: image-level F1 at the 99th percentile of validation/good image scores.
- **Latency**: `predict()` wall time per image on the stated device (includes resizing and map up-sampling; excludes PNG decoding). **GPU peak**: `torch.cuda.max_memory_allocated` during inference.

## Fitting cost from scratch and cache-reload verification

The table above was produced by a second, identical run: 9 of its 12 models were **reloaded from
`checkpoints/fitted/`** and 3 (the `keep_aspect` PaDiM variants and fruit_jelly `keep_aspect`
PatchCore) were re-fitted after a cache-key correction. All image AUROC, AP, pixel AUROC, AU-PRO
and pixel-F1 values were identical to the first run, where every model was fitted from scratch
(largest difference 1.2 × 10⁻⁹), so a reloaded model behaves exactly like a fresh fit.

Fit wall time measured in the first run (from scratch, includes reading the training PNGs; RTX 4090 Laptop GPU):

| Method | sheet_metal (137 images) | fruit_jelly (263 images) |
|---|---|---|
| PatchCore 256×256 | ≈ 27 s (smoke run) | 67 s |
| PatchCore keep_aspect | 253 s (1024×256; 561 k patches → 56 k coreset) | 118 s (352×256) |
| PaDiM 256×256 / keep_aspect | 7 s / 11 s | 12 s / 13 s |
| EfficientAD (10 000 steps) | 334 s | 334 s |
| AnomalyDINO (k = 4) | 5 s | 10 s |
