# IDD Lab — Student Quick Start

## 1. Install (≈ 10 minutes, no administrator rights)
1. Install **Python 3.12** from python.org (tick *Add python.exe to PATH*).
2. Put the `IDD_Lab` folder in a short path, e.g. `C:\IDD_Lab`.
3. Open PowerShell in that folder and run:
   ```powershell
   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
   .\setup_student.ps1
   ```
   This installs the **core** profile (no PyTorch). For neural networks and 3D, use
   `.\setup_student.ps1 -Profile full` (CPU or GPU chosen automatically). Details: [INSTALL_WINDOWS.md](INSTALL_WINDOWS.md).

## 2. Start
```powershell
.\start.ps1
```
The **first-run assistant** shows what your computer can do (GPU, installed libraries, datasets,
models) and lets you choose **Student Guided Mode**. Nothing is downloaded.

## 3. First steps
1. **Start Learning → 1 · Introduction** (10 min).
2. **Exercise 1 — Classical Defect Detection** ([docs/exercises/EXERCISE_1_CLASSICAL.md](docs/exercises/EXERCISE_1_CLASSICAL.md)):
   Otsu, Canny, morphology, Gabor and reference comparison on the Synthetic Educational Data.
3. **Educational Experiments → C · Understanding Evaluation Metrics**: why a good AUROC can produce poor
   decisions — with real, stored results from MVTec AD 2.

## 4. Know what you are looking at
- **SYNTHETIC EDUCATIONAL DATA — NOT A REAL INDUSTRIAL BENCHMARK** (banner): procedurally generated images;
  results are illustrative only.
- **STORED RESULTS** (blue banner): computed and verified on the instructor's computer with real data;
  nothing is computed on your computer.
- **LIVE** (green banner): inference running on your computer.
- *Local educational results* are not official benchmark scores.

## 5. Going further (optional)
- **Model Manager**: which algorithms can run here; download a pretrained backbone (source, size and
  licence shown first).
- **Real datasets**: get MVTec AD 2 or MulSen-AD from the official sources (Atlas → *Research Sources &
  Downloads*, [DATASETS.md](DATASETS.md)) and import them in *Settings → Dataset Manager*.
- All eight exercises: [docs/exercises/](docs/exercises/EXERCISE_1_CLASSICAL.md) ·
  workshop guide: [Participant guide](workshop/CIARP2026/guide/PARTICIPANT_GUIDE.md).

## 6. Problems?
[TROUBLESHOOTING.md](TROUBLESHOOTING.md) · offline check: `.\.venv\Scripts\python.exe -m scripts.readiness --profile student`
