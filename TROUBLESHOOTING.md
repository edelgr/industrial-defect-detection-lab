# Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `setup.ps1` cannot be executed | PowerShell execution policy | `Set-ExecutionPolicy -Scope Process Bypass` |
| Wrong Python picked up (OSGeo4W, Conda) | another interpreter first on PATH | `.\setup.ps1 -Python C:\Python312\python.exe` |
| Status bar shows "CPU mode" | CPU-only torch or old driver | `.\check_gpu.ps1`; reinstall with `-Cuda cu128` for drivers < 580 |
| `CUDA out of memory` | large input / many fit images | fewer fit images, smaller input size, close other GPU apps, or *Settings → device = cpu* |
| WinCLIP: `not enough values to unpack` | open-clip-torch ≥ 2.26.1 | `pip install "open-clip-torch>=2.23,<2.26.1"` (Anomalib 2.7.1 requirement) |
| Method fails offline at first use | weights not cached | run `.\prepare_models.ps1 -WithWinCLIP` while online |
| Import: "No known categories found" | wrong dataset selected or nested archive | select the matching dataset; extract nested archives first |
| MulSen-AD images fail in other tools | many `.png` files are BMP-encoded | IDD Lab decodes by content; other tools may need renaming |
| 3D deviation everywhere high | registration failed (low fitness) | enable coarse alignment, change voxel size, use more references |
| `OpenClipboard failed` warnings | another program holds the clipboard | retry *Copy BibTeX* |
| Link badge "blocks bots" | publisher refuses automated checks (HTTP 403) | not an invalid link; see `research_catalog/manual_verification.yaml` |
| Catalog validation problems shown on Dashboard | invalid YAML edit | `python -c "from core.research.catalog import load_catalog; print(load_catalog().problems)"` |

## Endpoint protection quarantines `evil.zip` during the tests
Older versions of `tests/test_datasets.py` wrote a small archive with a `../` entry to check that the
Dataset Manager rejects path traversal; corporate antivirus (e.g. Check Point Threat Emulation) flags
it. Since 2026-11-21 these hostile test archives are built **in memory** and never touch the disk.

## Native crash ("Illegal instruction" / access violation in `shibokensupport ... feature_imported`)
Calling scikit-learn metrics thousands of times inside the Qt process made scikit-learn's dataframe
probing (`narwhals`) trigger PySide6's import hook, which crashed natively. The bootstrap loops in
`core/evaluation/scenes.py` therefore use a rank-based AUROC (identical to `roc_auc_score`, tested).
If a similar crash appears elsewhere, run the failing computation outside the GUI thread's hot loop or
with NumPy/SciPy equivalents.

## A Workshop Verified model is listed as incompatible
The registry refuses a checkpoint when the dataset folder, category, Anomalib version, fit images or
recomputed cache key differ from the manifest (the reason is shown in the Lab). Re-run
`python -m scripts.verify_workshop_models` on the machine that holds the data; it never trains.

## Quarantined checkpoints (`checkpoints/quarantine/`)
Fitted models identified as test artefacts are moved there, never deleted automatically. Each folder
has a `QUARANTINE.json` (reason, SHA-256, size, original provenance, dependency checks). To restore
one, move its `.pt` and `.json` back to `checkpoints/fitted/`; to reclaim the space, delete the folder
once the maintainer has confirmed it is not needed.

## `TypeError: 'NoneType' object is not callable` from `torch/library.py` at exit
Seen as "Exception ignored in: <finalize object ...> ... torch\library.py, line 676, in _del_library".
Cause (diagnosed 2026-11-21): Python was finalising its modules while Qt worker threads (e.g. thumbnail
loading) were still running, so PyTorch's `torch.library` finalisers ran after their module globals had
been cleared. It is not a leak in IDD Lab — the process ends and the OS frees memory and GPU — but closing
the window during a fit could have interrupted a checkpoint write. Since then the app cancels and waits
for background tasks (≤ 10 s) when closing, checkpoints are written atomically, and the test suite waits
for the thread pool before ending; the messages no longer appear. If they reappear, check for a task that
ignores its `cancel()` callback.

## Setup stops with "installation folder path is too long" (or pip: "Windows Long Path support")
Qt for Python contains files ~140 characters below `site-packages`. Without Windows Long Path support the
full path must stay below 260 characters. Move the IDD_Lab folder to a short path such as `C:\IDD_Lab`
(no administrator rights needed) and run `setup_student.ps1` again, or ask an administrator to enable Long
Path support. Avoid deep folders (Desktop inside OneDrive, Downloads\…\…).

## A deep method shows "unavailable" in the Student Edition
Open the **Model Manager**: it states what is missing (PyTorch in the core profile → `setup_student.ps1
-Profile full`; pretrained weights → *Download pretrained weights*, online). With `IDD_OFFLINE_WEIGHTS=1`
methods whose weights are not cached are reported unavailable instead of trying to download.

## Educational Experiments say "STORED RESULTS"
Expected on a student computer: the results were computed and verified on the instructor's computer with
real MVTec AD 2 data. Live inference needs the dataset (official download) and matching fitted models.
