# Publication readiness

**Status: BLOCKED** — technically ready for the owner's review; publication is blocked by decisions that
only the project owner (and, where applicable, the owner's institution) can take.

Last assessed: 2026-11-21. A **private** GitHub repository exists
(https://github.com/edelgr/industrial-defect-detection-lab). It is not public and no release exists.

## Blockers (must be resolved before the first public release)

| # | Blocker | Who | How to resolve |
|---|---|---|---|
| B1 | **Project licence not applied** (`pyproject.toml` says "TO BE DECIDED"; no `LICENSE` file). Apache-2.0 is prepared as a proposal ([licensing/](licensing/)) | Owner | Confirm the right to license every affected component (B2), then apply Apache-2.0 per RELEASE_CHECKLIST |
| B2 | **Code ownership / institutional authorisation** — the project is a personal educational initiative; confirm that no institution claims rights over code or materials written for it | Owner (+ institution if applicable) | Confirm ownership; obtain any required approval |
| B3 | ~~Git history and author identity~~ — **resolved**: fresh public history, GitHub `noreply` identity, internal maintenance notes excluded | — | — |
| B4 | **Derived results from MulSen-AD** (`configs/workshop_verified.json` `d06-*` entries, `docs/LOCAL_RESULTS.md`) — dataset licence not stated | Owner | Keep (with attribution) after review, or remove those entries from the public release |
| B5 | ~~GitHub repository URL~~ — **resolved**: private repository created; URL used in README | — | — |
| B6 | **Workshop presentation** not yet added; its IP review (figures, dataset images, logos, templates, fonts) pending | Owner | Provide the original file; complete IP_AND_ATTRIBUTION_REVIEW §3 |

## Conditions (not blockers, but must stay true)

- No dataset, pretrained weight, fitted checkpoint, experiment database or cache in the release
  (`scripts/build_release.py` enforces it; see REDISTRIBUTION_POLICY).
- MVTec AD 2-derived results carry attribution and the *local educational, not official* label.
- Model-weight licences with caveats (ImageNet-1k terms, EfficientAD teacher weights *not stated*) are
  documented; weights are only downloaded by the user from the original source.

## Technical readiness (verified 2026-11-21)

| Check | Result |
|---|---|
| Automated tests (instructor installation) | 108 passed (0 failed) |
| Clean install, Student core profile (isolated copy, own venv) | passed (10/10 setup steps, readiness 0 required failing; 78 passed / 30 skipped) |
| Clean install, Student full profile, CPU PyTorch | passed (readiness 0 required failing; 80 passed / 28 skipped, no downloads) |
| Qt flows: instructor EN/ES, student EN/ES | 27/27, 27/27, 22/22, 22/22 |
| Release staging (`scripts/build_release.py`) | 0 forbidden files; user name redacted; 4 items flagged for review |
| Publication copy (this repository, fresh clone, offline, CI order) | check_docs 0 problems; 108 relative links / anchors valid; ruff 0.6.9 clean; pytest 80 passed / 28 skipped; student readiness exit 0; Qt smoke exit 0; Qt student flows EN/ES 21/21 on a real display; privacy scan clean |
| CI workflow | written (`.github/workflows/ci.yml`); runs on the private repository after the first push |

## Classification

- READY FOR REVIEW — the code, documentation and staging process can be reviewed now.
- READY FOR PRIVATE PUSH — yes, after the owner's authorisation.
- READY FOR PUBLIC RELEASE — **no** (B1, B2, B4, B6 open).
- **BLOCKED (public)** — until B1, B2, B4 and B6 are resolved and the owner authorises publication.
