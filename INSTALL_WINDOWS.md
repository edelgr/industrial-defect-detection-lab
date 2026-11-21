# Installing IDD Lab on Windows

Two installation paths share the same source code:

| | Student Edition | Instructor Edition |
|---|---|---|
| Script | `setup_student.ps1` | `setup.ps1` |
| Python packages | pinned locks in `requirements/` | `pip install -e .[deep,3d,dev]`, frozen in `requirements-lock.txt` |
| PyTorch | none (core) or CPU / CUDA build chosen from the detected driver (full) | CUDA 13.0 build by default |
| Disk (environment) | core ≈ 0.8 GB · full ≈ 3 GB (CPU) / ≈ 6 GB (CUDA) | ≈ 6 GB |

## Requirements
- Windows 10 or 11, 64-bit. **No administrator rights needed.**
- **Python 3.11–3.13** (3.12 recommended) from python.org — tick *Add python.exe to PATH*. Avoid
  OSGeo4W/Conda base interpreters that may come first on PATH (`-Python C:\Python312\python.exe` selects one).
- **A short installation path**, e.g. `C:\IDD_Lab`. Qt for Python contains files about 140 characters below
  `site-packages`; without Windows *Long Path support* the total path must stay below 260 characters.
  `setup_student.ps1` checks this before installing and explains what to do.
- Optional NVIDIA GPU. The CUDA 13.0 PyTorch build needs driver **≥ 580**; with an older driver the setup
  installs the CPU build (update the driver and re-run with `-Torch cu130` to use the GPU).
- Free disk space: the profile size plus a 5 GB safety margin; datasets are separate and optional.
- Display ≥ 1280 px wide (every page fits a 1280-px window, English and Spanish).

## Student Edition
```powershell
cd C:\IDD_Lab
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass   # only for this PowerShell window
.\setup_student.ps1                       # core profile
.\setup_student.ps1 -Profile full         # deep learning + 3D, CPU/CUDA chosen automatically
.\setup_student.ps1 -Profile full -Torch cpu
.\start.ps1
```
The script performs ten steps: detect Python → verify the version → check disk space and path length →
detect GPU/driver → choose CPU or GPU PyTorch (full profile; a CUDA build is never installed just
because an NVIDIA GPU is present) → create `.venv` → install pinned dependencies → validate imports →
prepare folders and the Synthetic Educational Data → run the offline readiness check
(`python -m scripts.readiness --profile student`).

It downloads Python packages only. Pretrained weights and datasets are **not** downloaded; do that
later, if you want, from the Model Manager / `prepare_models.ps1` (full profile) and DATASETS.md.

### What works in each profile
| Capability | core | full (CPU) | full (CUDA) |
|---|---|---|---|
| Atlas, Methods Library, Start Learning, Guided Workshop, references, BibTeX | ✓ | ✓ | ✓ |
| Dataset Explorer, classical methods, synthetic demos, metrics | ✓ | ✓ | ✓ |
| Educational Experiments — stored verified results | ✓ | ✓ | ✓ |
| PaDiM / PatchCore / AnomalyDINO / WinCLIP (after downloading backbones and fitting) | — | ✓ (slower) | ✓ |
| EfficientAD training (10 000 steps) | — | impractical | ✓ (≈ 6 min on an RTX 4090 Laptop GPU) |
| 3D & multimodal (Open3D, PyVista) | — | ✓ | ✓ |
| Live Educational Experiments | — | needs MVTec AD 2 + fitted models | needs MVTec AD 2 + fitted models |

## Instructor Edition
```powershell
.\setup.ps1                     # or: .\setup.ps1 -Python C:\Python312\python.exe -Cuda cu130
.\check_gpu.ps1                 # CUDA, VRAM, recommended PyTorch build, library versions
.\prepare_models.ps1 -WithWinCLIP -WithEfficientAD
.\test.ps1
.\start.ps1 -Workshop           # offline demo profile for the workshop
```
Tested on Windows 11 Pro, Python 3.12.6, NVIDIA RTX 4090 Laptop GPU (16 GB), driver 595.79, 32 GB RAM.
Workshop preparation: [Instructor guide](workshop/CIARP2026/guide/INSTRUCTOR_GUIDE.md); readiness: `python -m scripts.readiness --profile instructor`.

## Datasets
```powershell
.\prepare_datasets.ps1 -Synthetic                         # Synthetic Educational Data (also done by the setup)
.\prepare_datasets.ps1                                    # status of every dataset adapter
.\prepare_datasets.ps1 -Import D05 -Archive C:\Downloads\sheet_metal.tar.gz -Categories sheet_metal
.\prepare_datasets.ps1 -MulSen capsule,nut,flat_pad       # dry run: shows source, size, free space
```
MVTec datasets require MVTec's form before downloading; download the archive in your browser, then
import it (GUI: *Settings → Dataset Manager → Import downloaded archive…*). See [DATASETS.md](DATASETS.md).

## Offline use
After the setup (and, optionally, model preparation and dataset import) IDD Lab runs without Internet.
`IDD_OFFLINE_WEIGHTS=1` makes methods whose pretrained weights are not cached report *unavailable*
instead of trying to download. Scientific links remain visible offline and open in the browser when a
connection is available.

## Executable packaging
A one-file executable is not provided: see [docs/DISTRIBUTION.md](docs/DISTRIBUTION.md) for the
investigation and the reasons.
