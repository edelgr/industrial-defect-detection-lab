# Architecture

```
app/                 Qt GUI (PySide6) — no scientific logic
  main.py            entry point (python -m app.main)
  ui/                main window (sidebar navigation, lazy pages), themes
  pages/             dashboard, learning, atlas, explorer, library, lab, lab3d, compare, experiments, models,
                     workshop, settings
  widgets/first_run  first-run assistant (Student Edition)
  widgets/           references panel, image viewer, point-cloud viewer, sample browser,
                     parameter form, step-through widget
  workers/task.py    QThreadPool tasks with progress/cancel → GUI never blocks
core/                GUI-independent, unit-tested
  research/          catalog loader + JSON-schema validation, content loader (EN/ES fallback)
  datasets/          adapter interface + registry; MVTec AD/LOCO/AD 2, MulSen-AD, Real3D-AD,
                     SYNTHETIC; manager (safe extraction, downloads, disk guard); remote_zip
  algorithms/        AnomalyMethod interface + registry (lazy imports)
    classical/       OpenCV / scikit-image methods
    memory_based/    PaDiM, PatchCore (Anomalib torch models)
    foundation/      AnomalyDINO, WinCLIP (Anomalib torch models)
    multimodal_3d/   registration + deviation baseline, multimodal fusion
  inference/         LabSession: fit → calibrate → predict → evaluate (leakage rules);
                     model_registry: checkpoint manifests, tiers, compatibility, verified presets
  evaluation/        metrics (AUROC, AP, pixel AUROC, AU-PRO, pixel F1, Dice/IoU, P/R/F1, FAR/FRR, latency);
                     scenes: scene grouping, cluster bootstrap, decision stability
  experiments/       SQLite experiment store with provenance categories
  system/            diagnostics, offline readiness (instructor and student profiles)
  profile.py         Instructor / Student / Auto edition resolution
  model_manager.py   per-algorithm availability, weight licences, explicit preparation
research_catalog/    YAML records (datasets, algorithms, publications) + schemas + link audit
education/           EN/ES Markdown & YAML content, steps, workshop, i18n, conceptual diagrams
scripts/             CLI tools (prepare data/models, verify links, build references, experiments)
tests/               pytest + pytest-qt
data/ checkpoints/ cache/ experiments/   git-ignored local assets
```

## Key interfaces
- **`DatasetAdapter`** (`core/datasets/base.py`): `list_categories(root)`, `index(root, category)
  → DatasetIndex[Sample]`. A `Sample` keeps original ids/splits, per-modality paths and masks, and
  metadata (e.g. per-modality labels). Adding a dataset = one adapter + one catalog YAML.
- **`AnomalyMethod`** (`core/algorithms/base.py`): `param_specs()`, `availability()`, `fit(images)`
  (train normals only), `predict(image, reference) → MethodResult` with a documented contract
  (float32 map aligned with the input, raw score, semantics string, latency, GPU peak memory).
  Adding a method = one adapter registered in `core/algorithms/registry.py` + one catalog YAML.
- **Research catalog** (`core/research/catalog.py`): validated against JSON schemas
  (`research_catalog/schemas`), referential integrity checked (method ↔ publication ↔ dataset).
- **`LabSession`** (`core/inference/session.py`): enforces data-usage rules and writes runs to the
  `ExperimentStore` with protocol strings. `load_checkpoint(entry)` loads a registry checkpoint with
  its own parameters and fit ids after recomputing its key; `fit()` writes a manifest next to every
  checkpoint it saves.
- **Model Registry** (`core/inference/model_registry.py`): `scan()` lists `checkpoints/fitted/*.pt`
  with manifests (reconstructed by exact key search for pre-registry files), tiers
  (`workshop_verified` from `configs/workshop_verified.json`, `quick_experiment`, `custom`,
  `unidentified`), and `compatibility(entry, index)` with human-readable reasons.

## Design decisions
- Anomalib's **PyTorch model classes** are used directly (no Lightning `Engine`) to allow
  per-image interactive fitting/inference and full control over preprocessing (plain resize, no
  centre-crop, so maps align with ground truth).
- Heavy modules are imported lazily: the GUI starts in ~2 s and the Atlas/Library work without
  PyTorch.
- All model caches (`TORCH_HOME`, `HF_HOME`) point into the git-ignored `checkpoints/`.
- Text content lives in `education/` (Markdown/YAML) — reviewable without touching code.
