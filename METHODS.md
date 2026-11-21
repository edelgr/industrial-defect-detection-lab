# Methods

Full records (paper, authors' code, implementation used, licence, status, explanations EN/ES) are
in `research_catalog/algorithms/` and [RESEARCH_REFERENCES.md](RESEARCH_REFERENCES.md).

| Group | Method | Implementation in IDD Lab | Type | Status (v0.1) |
|---|---|---|---|---|
| A | Otsu thresholding | OpenCV `threshold(THRESH_OTSU)` | library | runnable |
| A | Adaptive thresholding | local (Gaussian) mean − C (as `cv2.adaptiveThreshold`) | library | runnable |
| A | Canny edges | OpenCV `Canny` + edge-density map | library | runnable |
| A | Gabor texture | OpenCV Gabor bank + robust z-score | IDD Lab educational | runnable |
| A | Morphology (top/black-hat) | OpenCV `morphologyEx` | library | runnable |
| A | Reference difference | ECC alignment + |difference| | IDD Lab educational | runnable |
| A | SSIM comparison | scikit-image `structural_similarity` | library | runnable |
| B | CNN classification / YOLO / U-Net | — | — | explanation only |
| C | Convolutional autoencoder / DRAEM | — | — | explanation only |
| D | **PaDiM** | Anomalib 2.7.1 `PadimModel` (ResNet-18) | Anomalib (no official code exists) | runnable, requires fitting |
| D | **PatchCore** | Anomalib 2.7.1 `PatchcoreModel` (WRN-50-2) | Anomalib (official code: amazon-science, Apache-2.0) | runnable, requires fitting |
| E | **EfficientAD** | Anomalib 2.7.1 `EfficientAdModel` + IDD Lab training loop (short schedule, Imagenette2-320 penalty) | Anomalib (no official code exists) | runnable, requires training (≈ 6 min / 10 000 steps on RTX 4090 Laptop; cached afterwards) |
| F | **AnomalyDINO** | Anomalib 2.7.1 `AnomalyDINOModel` (DINOv2 ViT-S/14) | Anomalib (official code: dammsi, Apache-2.0) | runnable (few-shot) |
| F | **WinCLIP** | Anomalib 2.7.1 `WinClipModel` (OpenCLIP ViT-B-16-plus-240) | Anomalib (no official code exists) | runnable (zero/few-shot) |
| F | Dinomaly, DINOv2/DINOv3 features | — | — | explanation only |
| G | Registration + deviation | Open3D FPFH+RANSAC, ICP, NN/point-to-plane | IDD Lab educational baseline | runnable, validated on MulSen-AD |
| G | Multimodal late fusion | per-modality PatchCore/PaDiM + 3D baseline, z-score fusion | IDD Lab educational | runnable, validated on MulSen-AD |
| G | M3DM, Reg3D-AD | — | — | explanation only |

## Output contract
Every runnable method returns a float32 anomaly map aligned with the input, a raw image score,
optional mask/threshold, latency, GPU peak memory and a sentence describing what its numbers
mean. Scores are **never** presented as probabilities or confidence percentages.

## Licensing notes
- Ultralytics YOLO is AGPL-3.0 (including trained models by default) — not bundled.
- `milesial/Pytorch-UNet` and `mala-lab/WinCLIP` are GPL-3.0 third-party code — not bundled.
- Real3D-AD code has no licence file — not bundled.
- DINOv3 uses the custom *DINOv3 License* with gated weights — not used; DINOv2 (Apache-2.0) is.

## Adding a method
Implement `AnomalyMethod` (`core/algorithms/base.py`), register it in
`core/algorithms/registry.py`, add a schema-valid YAML record with verified references, optional
steps in `education/<lang>/steps.yaml`, and contract tests.

## Model availability (Model Manager)
The **Model Manager** page lists, for every algorithm: implementation used, references, training
requirement, CPU support, measured GPU memory (from verified instructor runs), pretrained resource
(source, size, licence checked 2026-11-21), what is installed on this computer, and one of five
availability classes: *included* (no weights needed), *official pretrained (download)*, *fitted on this
computer*, *instructor-only (stored results)*, *unavailable*. Downloads happen only after an explicit,
confirmed action; missing checkpoints never trigger automatic retraining.
