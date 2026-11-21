# Local Results (IDD Lab)

> **Category: local educational results.** Computed with IDD Lab's local protocol (fit on train/good, thresholds from held-out normal images, evaluation on the public labelled test split). These are *not* official benchmark results and must not be compared with author-reported numbers without stating the protocol differences.

Generated 2026-11-21 00:40 · dataset D06 · seed 42 · fit ≤ 60 images · command: `python -m scripts.run_experiments --dataset D06 --categories capsule nut flat_pad --methods canny ssim padim patchcore efficientad anomalydino winclip --fusion`

Environment: python 3.12.6, platform Windows-11-10.0.26300-SP0, numpy 2.5.4, cv2 5.1.0, torch 2.14.1+cu130, torchvision 0.29.1+cu130, anomalib 2.7.1, sklearn 1.9.1, skimage 0.26.0, PySide6 6.12.0, open3d 0.20.0, cuda_device NVIDIA GeForce RTX 4090 Laptop GPU, cuda_runtime 13.0

## Image-modality (RGB) results

| Category | Method | Model / version | Image AUROC | Image AP | Pixel AUROC* | F1 @ calibrated thr. | Mean latency (ms) | GPU peak | n test |
|---|---|---|---|---|---|---|---|---|---|
| capsule | canny | OpenCV 5.1.0 | 0.343 | 0.794 | 0.788 | 0.323 | 11.2 | — (CPU) | 58 (48 anom.) |
| capsule | ssim | scikit-image 0.26.0 | 0.983 | 0.997 | 0.774 | 0.979 | 216.9 | — (CPU) | 58 (48 anom.) |
| capsule | padim | Anomalib 2.7.1 PadimModel (resnet18, layers 1-3) | 0.900 | 0.973 | 0.984 | 0.400 | 7.0 | 193 MB | 58 (48 anom.) |
| capsule | patchcore | Anomalib 2.7.1 PatchcoreModel (wide_resnet50_2, layer2+layer3) | 0.998 | 1.000 | 0.997 | 0.979 | 12.6 | 180 MB | 58 (48 anom.) |
| capsule | efficientad | Anomalib 2.7.1 EfficientAdModel (small, 10000 steps, Imagenette2-320 penalty) | 0.829 | 0.950 | 0.993 | 0.630 | 11.2 | 130 MB | 58 (48 anom.) |
| capsule | anomalydino | Anomalib 2.7.1 AnomalyDINOModel (vit_small_patch14_dinov2, k=4) | 0.902 | 0.976 | 0.989 | 0.545 | 21.7 | 150 MB | 58 (48 anom.) |
| capsule | winclip | Anomalib 2.7.1 WinClipModel (OpenCLIP ViT-B-16-plus-240, laion400m_e31, k=0) | 0.931 | 0.986 | 0.987 | 0.629 | 54.5 | 879 MB | 58 (48 anom.) |
| nut | canny | OpenCV 5.1.0 | 0.824 | 0.951 | 0.948 | 0.550 | 12.0 | — (CPU) | 39 (29 anom.) |
| nut | ssim | scikit-image 0.26.0 | 0.855 | 0.955 | 0.923 | 0.188 | 212.2 | — (CPU) | 39 (29 anom.) |
| nut | padim | Anomalib 2.7.1 PadimModel (resnet18, layers 1-3) | 0.993 | 0.998 | 0.986 | 0.967 | 7.1 | 193 MB | 39 (29 anom.) |
| nut | patchcore | Anomalib 2.7.1 PatchcoreModel (wide_resnet50_2, layer2+layer3) | 0.997 | 0.999 | 0.991 | 0.964 | 10.5 | 187 MB | 39 (29 anom.) |
| nut | efficientad | Anomalib 2.7.1 EfficientAdModel (small, 10000 steps, Imagenette2-320 penalty) | 1.000 | 1.000 | 0.987 | 0.964 | 9.1 | 130 MB | 39 (29 anom.) |
| nut | anomalydino | Anomalib 2.7.1 AnomalyDINOModel (vit_small_patch14_dinov2, k=4) | 0.586 | 0.858 | 0.988 | 0.571 | 21.4 | 150 MB | 39 (29 anom.) |
| nut | winclip | Anomalib 2.7.1 WinClipModel (OpenCLIP ViT-B-16-plus-240, laion400m_e31, k=0) | 0.924 | 0.979 | 0.958 | 0.188 | 57.0 | 879 MB | 39 (29 anom.) |
| flat_pad | canny | OpenCV 5.1.0 | 0.197 | 0.611 | 0.934 | 0.158 | 12.0 | — (CPU) | 40 (30 anom.) |
| flat_pad | ssim | scikit-image 0.26.0 | 0.950 | 0.985 | 0.775 | 0.000 | 261.5 | — (CPU) | 40 (30 anom.) |
| flat_pad | padim | Anomalib 2.7.1 PadimModel (resnet18, layers 1-3) | 0.467 | 0.820 | 0.981 | 0.857 | 6.5 | 193 MB | 40 (30 anom.) |
| flat_pad | patchcore | Anomalib 2.7.1 PatchcoreModel (wide_resnet50_2, layer2+layer3) | 0.990 | 0.997 | 0.989 | 0.857 | 9.8 | 187 MB | 40 (30 anom.) |
| flat_pad | efficientad | Anomalib 2.7.1 EfficientAdModel (small, 10000 steps, Imagenette2-320 penalty) | 0.840 | 0.949 | 0.989 | 0.896 | 9.3 | 130 MB | 40 (30 anom.) |
| flat_pad | anomalydino | Anomalib 2.7.1 AnomalyDINOModel (vit_small_patch14_dinov2, k=4) | 0.630 | 0.882 | 0.949 | 0.667 | 20.7 | 150 MB | 40 (30 anom.) |
| flat_pad | winclip | Anomalib 2.7.1 WinClipModel (OpenCLIP ViT-B-16-plus-240, laion400m_e31, k=0) | 0.693 | 0.895 | 0.980 | 0.378 | 60.1 | 879 MB | 40 (30 anom.) |

\* Pixel AUROC on RGB masks only; MulSen-AD provides RGB masks only for samples anomalous in RGB, so samples anomalous only in other modalities count as all-normal pixels. Maps are down-sampled to ≤256 px for this metric.

Object-level labels in MulSen-AD mark a sample anomalous if *any* modality is anomalous. Observation (not investigated further): RGB-only PatchCore reaches ≥ 0.99 object-level AUROC even though, per `data.csv`, some anomalous samples show no anomaly in RGB, and its AUROC against the RGB per-modality label is lower (see fusion table). Possible explanations — e.g. systematic acquisition differences between normal and anomalous samples (a dataset 'shortcut') — are a good discussion topic, not a conclusion.

## Multimodal fusion (educational protocol)

| Category | RGB z | IR z | 3D z | Fused mean | Fused max | RGB (mod. label) | IR (mod. label) | 3D (mod. label) |
|---|---|---|---|---|---|---|---|---|
| capsule | 0.992 | 0.990 | 0.779 | 0.927 | 0.879 | 0.897 | 0.909 | 0.954 |
| nut | 0.993 | 0.359 | 0.652 | 0.790 | 0.690 | 0.993 | 0.663 | 0.800 |
| flat_pad | 0.997 | 0.643 | 0.723 | 0.860 | 0.740 | 0.997 | 0.805 | 0.824 |

Protocol: IDD Lab educational multimodal protocol: patchcore per image modality fitted on 40 train/good samples; 3D deviation vs 3 merged normal clouds; z-normalisation on 9 held-out train/good samples; late fusion (mean, max); 58 test samples; seed 42. Not the MulSen-AD official benchmark method.
