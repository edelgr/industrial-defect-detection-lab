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
- Educational Experiments: resolution, illumination robustness and evaluation metrics. The illumination
  experiment has a **Basic** mode (one scene: image, anomaly map, score against the calibrated threshold,
  OK/NOK and TP/TN/FP/FN, plain-language explanation) and an **Advanced** mode (interactive chart of all
  scenes, scenes × lighting decisions, AUROC per lighting with cluster-bootstrap intervals, statistical
  limitations, references). Scores, decisions and statistics always come from stored verified results;
  live inference, when data and models are present, only adds anomaly maps, checked against the stored
  scores. Layout fits 1280×800 screens; images can be enlarged with synchronised zoom.
- 3D inspection support for point-cloud / multimodal data (full profile).

- Without a dataset (stored results), experiment B shows compact condition cards and experiment A shows its
  explanation instead of empty image panels; the Dataset Atlas gallery adapts its columns to the window
  width; the Model Manager table shows all five columns without horizontal scrolling; button and tab labels
  containing "&" are displayed correctly.
- **Visual Methods Library**: a method tree and one tabbed panel (Overview · How it works · Step by step ·
  Example · Strengths & limits · Technical sheet · References). Interactive pipeline diagrams (fitting and
  test lanes, clickable stages) and worked examples on synthetic images generated at run time
  (`core/visualization/method_examples.py`; no dataset, download or checkpoint). Classical methods use the
  real IDD Lab implementation (a test checks the replayed stages against it); feature, foundation, 3D and
  fusion methods are labelled *toy illustrations of the mechanism*. Curated EN/ES content for 13 methods
  (`education/<lang>/method_visuals.yaml`); the others fall back to a catalog-generated summary and diagram.
- **Reference comparison** in the Methods Laboratory (*Reference image difference*): reference, test, test
  aligned by ECC, difference map and detected differences with synchronised zoom/pan, previous/next
  reference, global score vs. local differences and a table of the references tried. The panels show the
  algorithm's own intermediate arrays (no numerical change). Changing the reference discards dependent
  results; `test_private` images get a prediction, never ground truth. The reference list is filled as soon
  as a category is selected and keeps the user's choice.
- UI strings may also live in feature files `education/i18n/ui_<lang>_<part>.yaml` (merged at start-up).
- **Supervised learning in the Methods Library** (U-Net segmentation, CNN image classification OK / NOK,
  YOLO object detection) at the same visual level as the other families: summary with the type of
  supervision and the labels needed; architecture diagrams drawn by the application (U-Net as a U with
  encoder, bottleneck, decoder and skip connections; CNN feature extraction → global pooling → classifier →
  OK / NOK; YOLO backbone → S×S head → box / class / confidence → threshold + NMS); worked examples with
  small models trained in about a second on labelled synthetic images (image labels, boxes, pixel masks),
  labelled *toy illustrations* and shown with their held-out figures; and a classification · detection ·
  segmentation comparison (label needed, output, localisation, annotation level, indicative labelling cost,
  when to use) in the Overview tab of the three methods. English and Spanish; no dataset, download or
  checkpoint. The diagram widget gains a grid layout and two node kinds (labelled data, learned prediction
  head). The comparison pictures have a fixed cell size so captions never overlap or crop the images.
  The Library header no longer repeats the year when the venue contains it.

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
- README screenshots regenerated in the Student Edition profile from a clean copy (no datasets, no
  pretrained weights) at one size, grouped as Explore and Learn, Run and Compare, Understand Experimental
  Results, and 3D and Model Management, including the illumination experiment in Basic and Advanced mode.
  The gallery now has 15 images: new Methods Library views (PatchCore pipeline diagram, worked morphology
  example) and the Methods Laboratory reference comparison; the Canny view was re-taken for the new
  Laboratory tabs; the other images are unchanged (renumbered). README sections on the visual library, the
  interactive diagrams, synthetic examples, real versus simplified examples and the reference comparison.
- Gallery extended to 18 images with three supervised-learning views — the U-Net architecture, a
  YOLO-style detection example and the classification · detection · segmentation comparison — taken in the
  Student profile from a clean copy (synthetic data only); the two earlier Methods Library views were
  re-taken (the method tree now marks the supervised methods ◆); the other images are unchanged (renumbered).
- Tests for the visual library (`tests/test_method_visuals.py`, `tests/test_supervised_visuals.py`) and the
  reference comparison; `scripts/qt_reference_check.py` (real Qt check of
  the reference comparison in English and Spanish at 1280×800; uses MVTec AD 2 if installed, otherwise the
  synthetic data).
- `scripts/readiness.py`, `scripts/check_docs.py`, `scripts/build_release.py`, Qt flow checks.
- CI workflow `.github/workflows/ci.yml` (static checks, docs, catalog, CPU tests, Qt smoke; no GPU, no
  downloads, never publishes).
- Licence audit, redistribution policy, publication readiness, distribution strategy, release checklist,
  IP and attribution review.
