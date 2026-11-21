# Contributing

Thank you for improving IDD Lab. The project's defining rule: **scientific traceability and
honesty first**.

## Ground rules
- Never invent references, URLs, metrics or anomaly maps. Unknown → `null` / "unknown".
- Distinguish official code, third-party reproductions, Anomalib integrations and IDD Lab
  educational implementations (`implementation_type`).
- Do not commit datasets, model weights or images from datasets whose licence forbids it.
- Synthetic data must be labelled as such in code, UI and results.
- Thresholds must never be selected on test labels for reported metrics.

## Workflow
1. `.\setup.ps1`, then `.\test.ps1` must pass.
2. Catalog edits: run `python -m scripts.build_references` (regenerates `CITATIONS.bib` and
   `RESEARCH_REFERENCES.md`; a test checks they are in sync) and, when online,
   `python -m scripts.verify_research_links`.
3. Educational text goes to `education/en/…` and `education/es/…`, not into Python code.
4. New datasets/methods: see [DATASETS.md](DATASETS.md) and [METHODS.md](METHODS.md).
5. Keep GUI code in `app/` and scientific logic in `core/` (testable without Qt).

## Code style
Python ≥ 3.11, type hints, pathlib, no blocking work on the Qt GUI thread (use `app.workers.task.start`).

## Student Edition, CI and releases
- The core profile has no PyTorch: tests that need it use the `deep` fixture (skip cleanly);
  tests that read the real `checkpoints/fitted/` are marked `real_checkpoints` and must not write there.
- `python -m scripts.check_docs` must report 0 problems (links, commands, exercise sheets).
- CI (`.github/workflows/ci.yml`) runs without GPU and with `IDD_OFFLINE_WEIGHTS=1` — never add a test that
  downloads datasets or weights unconditionally.
- Releases are staged with `python -m scripts.build_release --out <dir outside the repo>`; never publish
  without the owner's authorisation (see docs/RELEASE_CHECKLIST.md).
