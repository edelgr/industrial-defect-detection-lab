# Licence audit — IDD Lab (Student Edition release preparation)

Date: 2026-11-21 · Scope: everything that could be published in a public GitHub repository or a
Student Edition release package. This is a technical audit, **not legal advice**; items marked
*requires permission* or *not yet verified* must be resolved by the project owner (and, where
applicable, the owner's institution) before publication.

Classification used everywhere in this document:

| Class | Meaning |
|---|---|
| **Safe** | Safe to distribute (original IDD Lab work or permissively licensed and included as allowed) |
| **Conditional** | Distributable with attribution or conditions |
| **Permission** | Requires permission from a rights holder or a legal/institutional decision |
| **Not distributable** | Must never be included in a release |
| **Unverified** | Not yet verified — treat as *not distributable* until verified |

## 1. IDD Lab source code (ownership)

| Asset | Finding | Class |
|---|---|---|
| `app/`, `core/`, `scripts/`, `tests/`, PowerShell scripts | Written for this project (AI-assisted development commissioned by the project owner). No third-party source files are vendored; third-party code is used only as installed dependencies (`pip`). | **Permission** — ownership must be confirmed by the owner |
| Git history | The public repository starts from its own initial commit (GitHub web interface, noreply identity). The private development history is **not** published; its content was copied as a fresh commit authored with the owner's GitHub `noreply` address. No institutional e-mail appears in the public history. | Safe (subject to ownership confirmation, B2) |
| Project licence | `pyproject.toml` says *"TO BE DECIDED"*. No `LICENSE` file is applied. Apache-2.0 is **prepared, not applied**: [licensing/LICENSE-Apache-2.0.proposed.txt](licensing/LICENSE-Apache-2.0.proposed.txt), [../NOTICE.proposed](../NOTICE.proposed). | Blocker until decided |

**Recommendation (not applied):** Apache-2.0 for the original IDD Lab code. It is compatible with every
dependency used (all permissive, plus Qt for Python under LGPL-3.0 used as a dynamically linked
library), includes an explicit patent grant and is the licence of the main ML dependencies
(Anomalib, timm, OpenCV). It must **not** be assigned until (1) the owner confirms ownership,
(2) any required institutional authorisation is obtained, and (3) the owner chooses it explicitly.
Content (education texts, diagrams) could alternatively use CC BY 4.0 — also an owner decision.

## 2. Python dependencies (installed by the user from PyPI / download.pytorch.org — not redistributed)

Licences read from the installed package metadata (`importlib.metadata`) of the instructor environment
(148 packages in `requirements-lock.txt`) on 2026-11-21. No AGPL or GPL-only dependency is present.

| Component | Version | Licence (metadata) | Notes | Class |
|---|---|---|---|---|
| PySide6 / shiboken6 (Qt for Python) | 6.12.0 | LGPL-3.0-only OR GPL-2.0-only OR GPL-3.0-only | Used under LGPL-3.0 as a dynamically linked library installed by pip. A bundled executable would add LGPL obligations (licence text, relinkability, source offer for Qt) | Conditional |
| PyTorch | 2.14.1 | Apache-2.0 AND BSD-2/3-Clause AND MIT AND BSL-1.0 (+LLVM exception) | CUDA wheels bundle NVIDIA runtime libraries under NVIDIA's terms — never re-bundle them | Conditional |
| torchvision | 0.29.1 | BSD | | Safe (as dependency) |
| Anomalib | 2.7.1 | Apache-2.0 (Intel) | | Safe (as dependency) |
| timm | 1.0.31 | Apache-2.0 | | Safe (as dependency) |
| open-clip-torch | 2.24.0 | metadata `UNKNOWN` + classifier "Apache"; repository LICENSE text is MIT-style; GitHub reports NOASSERTION | Inconsistent metadata | Unverified (fine as a pip dependency; do not vendor) |
| OpenCV (headless) | 5.1.0.95 | Apache-2.0 | | Safe (as dependency) |
| Open3D | 0.20.0 | MIT | | Safe (as dependency) |
| PyVista / PyVistaQt / VTK | 0.49.2 / 0.14.0 / 9.7.2 | MIT / MIT / BSD | | Safe (as dependency) |
| NumPy, SciPy, scikit-learn, scikit-image, pandas | see lock | BSD-3-Clause (NumPy also 0BSD/MIT/Zlib/CC0 parts) | | Safe (as dependency) |
| Matplotlib | 3.11.2 | Matplotlib licence (PSF-based) | | Safe (as dependency) |
| pyqtgraph, PyYAML, jsonschema, pytest, pytest-qt, Pillow (MIT-CMU), narwhals, tiktoken | see lock | MIT / MIT-like | | Safe (as dependency) |
| requests, lightning, torchmetrics, kornia, huggingface_hub, ftfy, regex | see lock | Apache-2.0 (regex: + CNRI-Python) | | Safe (as dependency) |
| **Ultralytics YOLO** | — | AGPL-3.0 | **Not installed and not integrated**; only referenced in the Methods Library | n/a |

## 3. Algorithm implementations

| Method | Implementation used | Licence | Class |
|---|---|---|---|
| Otsu, adaptive threshold, Canny, morphology, Gabor, reference difference | OpenCV calls + IDD Lab code | Apache-2.0 (OpenCV) + IDD Lab | Safe (subject to §1) |
| SSIM | scikit-image | BSD-3-Clause | Safe |
| PaDiM, PatchCore, EfficientAD, AnomalyDINO, WinCLIP | Anomalib 2.7.1 PyTorch model classes (no code copied) | Apache-2.0 | Safe (as dependency) |
| 3D registration / deviation, multimodal fusion | Open3D + IDD Lab code | MIT + IDD Lab | Safe (subject to §1) |
| CNN, YOLO, U-Net, autoencoder, DRAEM, Dinomaly, M3DM, Reg3D-AD | explanation only (links to the authors' repositories) | GPL/AGPL/unlicensed repositories are linked, never copied | n/a |

## 4. Pretrained model weights (downloaded by the user from the original source; never redistributed)

Verified 2026-11-21 through the Hugging Face model-card API and the GitHub release API.

| Weights | Source | Licence found | Caveat | Class (for redistribution by IDD Lab) |
|---|---|---|---|---|
| `timm/resnet18.a1_in1k` (PaDiM) | huggingface.co/api/models/timm/resnet18.a1_in1k | `apache-2.0` | Trained on ImageNet-1k; ImageNet terms of access could not be retrieved (image-net.org refused the connection) | Unverified → do not redistribute; user downloads |
| `timm/wide_resnet50_2.racm_in1k` (PatchCore) | huggingface.co/api/models/timm/wide_resnet50_2.racm_in1k | `apache-2.0` | same ImageNet caveat | Unverified → do not redistribute |
| `timm/vit_small_patch14_dinov2.lvd142m` (AnomalyDINO) | huggingface.co/api/models/timm/vit_small_patch14_dinov2.lvd142m | `apache-2.0` | | Conditional (attribution) — still not bundled |
| `timm/vit_base_patch16_plus_clip_240.laion400m_e31` (WinCLIP) | huggingface.co/api/models/timm/vit_base_patch16_plus_clip_240.laion400m_e31 | `mit` | Trained on LAION-400M (web data; licence of the data not stated on the card) | Conditional — not bundled |
| EfficientAD teacher weights (Anomalib release `efficientad_pretrained_weights`) | api.github.com/repos/open-edge-platform/anomalib/releases/tags/efficientad_pretrained_weights | **not stated** in the release (repository: Apache-2.0) | | Unverified → do not redistribute |
| Imagenette2-320 (EfficientAD penalty images) | github.com/fastai/imagenette | repository Apache-2.0; image terms **not stated** (ImageNet subset) | | Unverified → do not redistribute |

A model being downloadable does not make it redistributable; IDD Lab only provides an explicit,
confirmed download action (Model Manager, `prepare_models.ps1`).

## 5. Datasets (never redistributed; user obtains them from the official source)

From the verified research catalog (`research_catalog/datasets/*.yaml`, verified 2026-11-20).

| Dataset | Licence | Class |
|---|---|---|
| MVTec AD, MVTec LOCO AD, **MVTec AD 2** | CC BY-NC-SA 4.0 (non-commercial, share-alike) | Not distributable by IDD Lab (official download only) |
| VisA | CC BY 4.0 | Conditional — still not bundled (size, policy) |
| Real3D-AD (data) | CC BY 4.0 (code: no licence file) | Conditional — not bundled |
| **MulSen-AD** | licence **not stated** by the authors | Not distributable |
| Real-IAD D³ | research-only, gated | Not distributable |
| 3D-ADAM | CC BY-NC-SA 4.0 | Not distributable by IDD Lab |
| Synthetic Educational Data | generated by IDD Lab code (fixed seed) | Safe — generated locally, not shipped |

## 6. Models trained/fitted with third-party data (the instructor's 27 checkpoints)

| Asset | Finding | Class |
|---|---|---|
| 27 fitted checkpoints in `checkpoints/fitted/` (24 Workshop Verified) | Fitted on MVTec AD 2 (CC BY-NC-SA 4.0 — whether a fitted model is "adapted material" under NC-SA is a legal question) and MulSen-AD (licence not stated), on top of backbones with the caveats of §4 | **Not distributable** without permission and legal review. Git-ignored; excluded by `scripts/build_release.py` |
| `configs/workshop_verified.json` | Derived evaluation results: metrics, thresholds, per-image scores and sample ids of public test splits; **no image data** | **Conditional** — attribution to MVTec AD 2 given; the MulSen-AD (`d06-*`) part needs the owner's decision |
| `docs/LOCAL_RESULTS.md`, `docs/LOCAL_RESULTS_D05.md`, `docs/SCIENTIFIC_AUDIT_D05.md` | Local educational results and audit text derived from the same datasets | Conditional (same as above) |

## 7. Illustrations, screenshots, educational content, references

| Asset | Finding | Class |
|---|---|---|
| `education/diagrams/*.svg` | Generated by `scripts/make_diagrams.py` (IDD Lab); conceptual, no paper figures | Safe (subject to §1) |
| `docs/screenshots/*.png` | IDD Lab UI with catalog content, SYNTHETIC data and stored metric plots derived from MVTec AD 2 (no dataset images) | Safe (subject to §1); metric plots conditional (attribution, CC BY-NC-SA) |
| Research paper figures | None copied into the repository | n/a |
| `education/` texts (EN/ES), exercises, guides | Original IDD Lab writing; cites sources | Safe (subject to §1) |
| `research_catalog/`, `CITATIONS.bib`, `RESEARCH_REFERENCES.md` | Bibliographic metadata (facts) and BibTeX from official sources; IDD Lab summaries | Safe |
| Local screenshots in `cache/` (qt_flow, qt_student) | Contain MVTec AD 2 / MulSen-AD images | Not distributable (git-ignored) |

## 8. Personal information found in tracked files

| Item | Finding | Handling |
|---|---|---|
| Internal maintenance notes (checkpoint maintenance log, local environment record, development status) | Contained local user names / absolute paths | **Not included** in the public repository |
| Other tracked files | Scanned for user name, home paths, e-mail addresses, institution names and secrets | None found (see [licensing/IP_AND_ATTRIBUTION_REVIEW.md](licensing/IP_AND_ATTRIBUTION_REVIEW.md) §4) |
| Git commit metadata | Owner's display name and GitHub `noreply` e-mail | Intended public identity |

## 9. Summary of blockers

See [PUBLICATION_READINESS.md](PUBLICATION_READINESS.md) and the separate code / presentation review in
[licensing/IP_AND_ATTRIBUTION_REVIEW.md](licensing/IP_AND_ATTRIBUTION_REVIEW.md).
