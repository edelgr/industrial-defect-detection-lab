# Changelog

All notable changes to this project. The public history of this repository starts with the version below;
earlier development took place in a private working copy and is summarised here, not republished.

## [Unreleased] — first public preparation (2026-11-21)

### Application
- PySide6 desktop application with twelve pages: Dashboard, Start Learning, Dataset Atlas, Dataset
  Explorer, Methods Library, Inspection Lab (2D), 3D Lab, Compare, Educational Experiments, Model Manager,
  Guided Workshop and Settings; English and Spanish user interface.
- Research catalog of industrial anomaly-detection datasets and methods with verified references.
- Classical methods (thresholding, morphology, texture statistics) and deep anomaly-detection methods
  (PatchCore, PaDiM, EfficientAD, AnomalyDINO, CLIP-based zero-shot) through a common interface.
- Model Registry with Workshop Verified presets (SHA-256, load and score checks) and a stored-results mode
  that shows verified results without datasets or checkpoints, clearly labelled.
- Educational Experiments: resolution, illumination robustness and evaluation metrics.
- 3D inspection support for point-cloud / multimodal data (full profile).

### Student Edition
- Instructor / Student / Auto profiles from one codebase (`core/profile.py`).
- `setup_student.ps1` with *core* and *full* profiles, pinned locks in `requirements/`, CPU or CUDA
  PyTorch chosen from the detected NVIDIA driver, disk and path-length checks, readiness check.
- First-run assistant, Start Learning page (9 topics) and eight exercise sheets (`docs/exercises/`).
- Synthetic Educational Data generated on first run, labelled
  "Synthetic Educational Data — Not a Real Industrial Benchmark".
- `IDD_OFFLINE_WEIGHTS=1` reports methods without cached pretrained weights as unavailable; the Lab asks
  before a fit would download a pretrained backbone.

### Workshop
- CIARP 2026 workshop materials in [workshop/CIARP2026/](workshop/CIARP2026/README.md): participant and
  instructor guides, exercise index, scientific references; slides to be added from the original file.

### Tooling and documentation
- `scripts/readiness.py`, `scripts/check_docs.py`, `scripts/build_release.py`, Qt flow checks.
- CI workflow `.github/workflows/ci.yml` (static checks, docs, catalog, CPU tests, Qt smoke; no GPU, no
  downloads, never publishes).
- Licence audit, redistribution policy, publication readiness, distribution strategy, release checklist,
  IP and attribution review.
