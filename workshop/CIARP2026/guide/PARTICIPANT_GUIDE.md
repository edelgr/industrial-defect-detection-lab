# Workshop Participant Guide

**Industrial Defect Detection and Anomaly Analysis using Computer Vision, Multimodal Imaging, and AI**
CIARP 2026 — Mexico City · 3 hours · software: IDD Lab (Student Edition)

## Workshop objectives
By the end of the workshop you will be able to:
1. explain how industrial inspection benchmarks are built and why their protocols matter;
2. apply classical computer-vision methods and understand their limits;
3. explain how unsupervised anomaly detection (PaDiM, PatchCore, EfficientAD) and foundation-model
   methods (AnomalyDINO, WinCLIP) work;
4. reason about 3D and multimodal inspection;
5. read AUROC, AU-PRO and F1 correctly and choose thresholds without test labels;
6. discuss how anomaly scores become industrial OK/NOK decisions.

## Before the workshop: installation
| Path | What you need | Time |
|---|---|---|
| **Recommended** — core profile | any Windows 10/11 laptop, Python 3.12, ≥ 7 GB free disk (checked by the setup) | ≈ 10 min |
| Optional — full profile | ≥ 13 GB free disk (checked by the setup); an NVIDIA GPU (driver ≥ 580) makes deep methods fast | ≈ 20–30 min |

Follow [STUDENT_QUICK_START.md](../../../STUDENT_QUICK_START.md). Run the check:
`.\.venv\Scripts\python.exe -m scripts.readiness --profile student` → *0 required check(s) failing*.

**Recommended computer:** Windows 10/11 64-bit, 8 GB RAM (16 GB for the full profile), a display at least
1280 px wide. A GPU is optional: every activity has a path without one.

## Application overview
| Section | Use it for |
|---|---|
| Start Learning | the guided path (9 topics) and exercise sheets |
| Dataset Research Atlas | benchmarks, protocols, licences, papers (offline) |
| Dataset Explorer | browsing images and ground-truth masks |
| Methods Library | how each algorithm works, step by step, with papers and code |
| Methods Laboratory | running a method on an image, thresholds, evaluation |
| Compare Methods | several methods on the same image |
| Educational Experiments | resolution, illumination and metrics experiments (stored or live) |
| Model Manager | what can run on your computer and why |
| Guided Workshop | the activity sequence used in the room |

## Datasets in the workshop
- **Synthetic Educational Data** (on every computer): generated locally; *not a real industrial benchmark*.
- **MVTec AD 2** (instructor's computer; optional for you): real, high-resolution industrial images with
  lighting changes; CC BY-NC-SA 4.0; official download after registration; private test labels are
  evaluated only on MVTec's server.
- **MulSen-AD** (instructor's computer; optional): RGB + infrared + 3D; licence not stated by the
  authors — never redistribute it.

## Methods covered
Classical (Otsu, adaptive threshold, Canny, Gabor, morphology, reference comparison, SSIM) ·
unsupervised (PaDiM, PatchCore, EfficientAD) · foundation models (AnomalyDINO, WinCLIP) · 3D
registration and deviation · late multimodal fusion. Each method card links to its paper and code.

## Practical exercises
| # | Exercise | Duration | Without GPU / dataset? |
|---|---|---|---|
| 1 | [Classical Defect Detection](../../../docs/exercises/EXERCISE_1_CLASSICAL.md) | 25 min | yes (synthetic) |
| 2 | [Understanding PatchCore](../../../docs/exercises/EXERCISE_2_PATCHCORE.md) | 25 min | concept + stored results |
| 3 | [Comparing Anomaly Detection Algorithms](../../../docs/exercises/EXERCISE_3_COMPARING.md) | 30 min | stored results |
| 4 | [Resolution Matters](../../../docs/exercises/EXERCISE_4_RESOLUTION.md) | 15 min | yes (stored results) |
| 5 | [Illumination Robustness](../../../docs/exercises/EXERCISE_5_ILLUMINATION.md) | 15 min | yes (stored scores) |
| 6 | [Understanding Evaluation Metrics](../../../docs/exercises/EXERCISE_6_METRICS.md) | 20 min | yes |
| 7 | [Multimodal Anomaly Detection](../../../docs/exercises/EXERCISE_7_MULTIMODAL.md) | 30 min | concept (full profile for 3D) |
| 8 | [Industrial Decision Making](../../../docs/exercises/EXERCISE_8_DECISIONS.md) | 20 min | yes |

The workshop does not require training models on your laptop. Demonstrations that need the real
datasets or the instructor's fitted models run on the instructor's computer; on yours they appear as
**stored results**, clearly labelled.

## Scientific references
Open them from any Atlas or Methods Library card (*Research Sources & Downloads*, *Scientific References &
Source Code*), or see [RESEARCH_REFERENCES.md](../../../RESEARCH_REFERENCES.md) and [CITATIONS.bib](../../../CITATIONS.bib).
Key papers: MVTec AD (Bergmann et al., CVPR 2019 / IJCV 2021), MVTec AD 2 (Heckler-Kram et al., IJCV 2026),
PaDiM (Defard et al., 2021), PatchCore (Roth et al., CVPR 2022), EfficientAD (Batzner et al., WACV 2024),
AnomalyDINO (Damm et al., WACV 2025), WinCLIP (Jeong et al., CVPR 2023), MulSen-AD (Li et al., 2025).

## Troubleshooting
| Problem | Fix |
|---|---|
| Setup stops: *path too long* | move the folder to `C:\IDD_Lab` and run the setup again |
| *Python not found* | install Python 3.12 from python.org with *Add to PATH* |
| A method shows *unavailable* | open the Model Manager: it tells you what is missing |
| Experiments show *STORED RESULTS* | expected without the dataset and fitted models |
| Antivirus warning during tests | see TROUBLESHOOTING.md (test archives are built in memory) |
More: [TROUBLESHOOTING.md](../../../TROUBLESHOOTING.md).

## After the workshop
- Work through the remaining exercises and the Learning Center at your own pace.
- Request MVTec AD 2 from its official website and import it to run the experiments live.
- Install the full profile to fit PaDiM / PatchCore on your own data — remember that a research model is
  **not** a certified machinery safety system.
- Cite the original papers (BibTeX from the app) when you use these methods or datasets.

---

## Instructor preparation checklist (brief)
1. Instructor computer: `.\setup.ps1`, `.\prepare_models.ps1 -WithWinCLIP -WithEfficientAD`, datasets
   imported (MVTec AD 2 `sheet_metal`, `fruit_jelly`; MulSen-AD `capsule`, `nut`, `flat_pad`).
2. `python -m scripts.verify_workshop_models` (never trains) → `python -m scripts.readiness --profile instructor`
   (SHA-256 of the 24 Workshop Verified checkpoints) → 0 required failing.
3. `python -m scripts.qt_flow_check --lang en` and `--lang es` (real GUI flows).
4. Send participants the Student Edition and this guide **one week before**; ask them to run the readiness check.
5. On the day: `.\start.ps1 -Workshop` (offline demo profile; asks before any training).
Full instructor guide: [Instructor guide](INSTRUCTOR_GUIDE.md).
