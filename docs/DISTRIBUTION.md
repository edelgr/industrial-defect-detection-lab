# Distribution strategy and packaging decision

## Editions
One codebase, two profiles (`core/profile.py`): **Instructor** (full verified installation, real data,
Workshop Verified checkpoints) and **Student** (works without datasets and pretrained/fitted models). The
profile only changes readiness checks, the first-run assistant and how missing assets are presented; it
never deletes data or models. Student installations: `setup_student.ps1` with the `core` or `full`
profile and pinned locks in `requirements/`.

## Source distribution (chosen)
A script-based source installation is the supported distribution:
- reproducible: pinned locks (`requirements/student-core-lock.txt` resolved in a clean venv;
  `requirements/student-full-lock.txt` derived from the instructor's tested environment);
- small download for the core profile (≈ 0.8 GB environment, no PyTorch);
- PyTorch CPU or CUDA chosen from the detected NVIDIA driver, never by GPU presence alone;
- release contents staged by `scripts/build_release.py` (policy in REDISTRIBUTION_POLICY.md).

## Windows executable (PyInstaller) — investigated, not built
| Aspect | Finding |
|---|---|
| Size | The full profile would bundle PyTorch (the CPU build alone is several hundred MB; CUDA builds add several GB of NVIDIA libraries), Open3D, VTK and Qt — a multi-GB executable folder that would be slow to download and to start. The measured core environment is 756 MB without PyTorch. |
| CUDA | Bundling CUDA libraries ties the package to one driver generation and redistributes NVIDIA binaries under NVIDIA's terms — explicitly avoided. |
| Licences | Bundling Qt for Python (LGPL-3.0) requires shipping the licence, keeping Qt relinkable and offering the corresponding source; feasible but an additional compliance task. |
| Antivirus | During development, endpoint protection quarantined test artefacts; unsigned PyInstaller executables are a frequent false-positive target. Code signing would be needed. |
| Long paths | Probably not solved by PyInstaller (Qt resources keep a similar folder depth) — not tested. |
| Core-only executable | Technically feasible (Qt + OpenCV + scikit-image, no PyTorch, estimated a few hundred MB). It is the best candidate for a **future optional installer**, after the licence decision and with code signing. |

Decision: provide the reliable script-based installation now; consider a signed, core-only installer as a
separate optional distribution later.

## Instructor preservation
Packaging and staging read the repository (`git archive HEAD`) and write only to a new directory outside
it. They never touch `data/`, `checkpoints/`, `experiments/` or verified manifests. The student setup
creates its own `.venv` in its own copy of the project.
