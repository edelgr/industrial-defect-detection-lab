# Datasets

IDD Lab catalogues eight research datasets (full verified records in `research_catalog/datasets/`,
all links in [RESEARCH_REFERENCES.md](RESEARCH_REFERENCES.md)). **No dataset is redistributed.**

| ID | Dataset | Year | Modalities | Licence | Access | Local adapter |
|---|---|---|---|---|---|---|
| D01 | MVTec AD | 2019 | RGB/grey | CC BY-NC-SA 4.0 | MVTec form; per-category tar.xz (104–705 MB) | ✓ |
| D02 | MVTec LOCO AD | 2022 | RGB | CC BY-NC-SA 4.0 | MVTec form; 0.6–1.6 GB per category | ✓ |
| D03 | VisA | 2022 | RGB | CC BY 4.0 | Direct S3 tar, 1.93 GB | — (import after `prepare_data.py`) |
| D04 | Real3D-AD | 2023 | point cloud | CC BY 4.0 (data); code unlicensed | Google Drive / Baidu | ✓ |
| D05 | MVTec AD 2 | 2025/26 | RGB | CC BY-NC-SA 4.0 | MVTec form; sheet_metal 1.53 GB; private test via server | ✓ |
| D06 | MulSen-AD | 2025 | RGB + infrared + 3D | **not stated** | Hugging Face (public), single 7.5–9 GB archive | ✓ |
| D07 | Real-IAD D³ | 2025 | RGB + pseudo-3D + 3D | CC BY-NC-SA 4.0 (card), research-only | Gated HF, ≈278 GB | — |
| D08 | 3D-ADAM | 2025/26 | RGB + 3D | CC BY-NC-SA 4.0 | HF (669 GB raw; 5.2 GB Anomalib subset) | — |

## Workshop selection
1. **MVTec AD 2 — `sheet_metal`** (primary 2D). 4224×1056 images, 137 train / 19 validation /
   114 test_public (24 good, 90 bad). Download it from the official downloads page after filling in
   MVTec's form, then import:
   `.\prepare_datasets.ps1 -Import D05 -Archive <file> -Categories sheet_metal`.
   Private test splits are indexed as *unlabelled* and never evaluated locally.

Verified MVTec AD 2 categories installed on the development machine (2026-11-21, imported from the
official per-category archives; SHA-256 in `data/mvtec_ad2/idd_install_log.json`; integrity report
[docs/DATASET_VALIDATION_D05.md](docs/DATASET_VALIDATION_D05.md), 0 errors):

| Category | Files | Image format | Train | Validation | test_public good / bad (masks) | test_private / _mixed |
|---|---|---|---|---|---|---|
| sheet_metal | 644 | 4224×1056, 1-channel 8-bit | 137 | 19 | 24 / 90 (90) | 142 / 142 |
| fruit_jelly | 950 | 2100×1520, RGB 8-bit | 263 | 37 | 20 / 60 (60) | 255 / 255 |

`test_public` contains few distinct scenes, each under several lighting conditions
(sheet_metal: 4 good + 15 bad scenes × 6 conditions; fruit_jelly: 5 + 15 × 4); defects cover a
median 0.25 % (sheet_metal) / 0.40 % (fruit_jelly) of the image.
2. **MulSen-AD — capsule, nut, flat_pad** (primary multimodal). Official route: Hugging Face
   `orgjy314159/MulSen_AD` (linked from the authors' repository). The authors provide only full
   archives; IDD Lab reads the ZIP's central directory via HTTP Range requests and fetches only
   the selected categories (generic technique, CRC-verified) — ≈ 655 MB for these three.
   The data licence is not stated: use locally, do not redistribute.

Verified MulSen-AD subset installed on the development machine (2026-11-21):

| Category | Train | Test good | Test anomalous | Extracted size |
|---|---|---|---|---|
| capsule | 64 | 10 | 48 | 817 MB |
| nut | 118 | 10 | 29 | 413 MB |
| flat_pad | 90 | 10 | 30 | 354 MB |

## SYNTHETIC demonstration data
`core/datasets/synthetic.py` renders brushed-metal textures with drawn scratches, dents and stains,
plus synthetic 3D height-field patches. It is **not** a research dataset; the GUI shows a banner
and all results on it are stored as *synthetic demonstration*.

## Teaching subsets
Teaching subsets must preserve original sample ids, splits, annotations and modality pairing,
and come with a manifest (`idd_install_log.json` records archive, checksum, selected categories
and date). Results on subsets are *local educational*, never official benchmark results.

## Adding a dataset
1. Verify sources (paper, website, licence, layout) — never add unverified datasets.
2. Add `research_catalog/datasets/Dxx_name.yaml` (schema-validated) and its publication.
3. Implement a `DatasetAdapter` in `core/datasets/` and register it in `core/datasets/registry.py`.
4. Add tests with a tiny synthetic fixture of the real layout.

## Student Edition: four data scenarios
IDD Lab never assumes a dataset is freely redistributable and never bundles one.

| Scenario | What works | How |
|---|---|---|
| **A — no dataset installed** | Atlas, Methods Library, Start Learning, Guided Workshop, classical methods and metrics on the **Synthetic Educational Data**, Educational Experiments with **stored results** | nothing to do: the synthetic set (6 defect kinds with exact masks, plus synthetic 3D surface patches; fixed seed) is generated on first run or by `.\prepare_datasets.ps1 -Synthetic`; every view shows *Synthetic Educational Data — Not a Real Industrial Benchmark* |
| **B — MVTec AD 2** | real images, masks, lighting conditions; live Educational Experiments when matching fitted models exist | request the dataset on the official MVTec website (Atlas → MVTec AD 2 → *Download / Request Dataset*; CC BY-NC-SA 4.0), download the category archive in your browser, then *Settings → Dataset Manager → Import downloaded archive…* or `.\prepare_datasets.ps1 -Import D05 -Archive <file> -Categories <name>`; private test splits are indexed as unlabelled and never evaluated |
| **C — MulSen-AD** | RGB + infrared + 3D viewer with per-modality labels, 3D baseline, fusion | official route on Hugging Face (linked from the authors' repository); `.\prepare_datasets.ps1 -MulSen capsule,nut,flat_pad` shows source, size and free space first, `-Yes` fetches only those categories (HTTP range requests, CRC-checked); RGB, infrared and point-cloud files keep their original names, so the adapter pairs them by sample id; licence **not stated** — do not redistribute |
| **D — another supported dataset** | the same tools through the adapter system — adapters exist for MVTec AD, MVTec LOCO AD and Real3D-AD (VisA, Real-IAD D³ and 3D-ADAM are described in the Atlas but have no adapter yet) | obtain it from its official source, then import the archive or *Use existing extracted folder…*; adapters are in `core/datasets/` (see "Adding a dataset" above) |
