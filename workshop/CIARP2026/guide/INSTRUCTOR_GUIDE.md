# Workshop Guide — CIARP 2026 (3 hours)

**Industrial Defect Detection and Anomaly Analysis using Computer Vision, Multimodal Imaging, and AI**

The full activity content (objectives, theory, instructions, expected observations, reflection
questions, references, extensions) is in the app (*Guided Workshop*) and in
`education/en/workshop.yaml` / `education/es/workshop.yaml`.

| # | Activity | Min | Data | Methods |
|---|---|---|---|---|
| 1 | Introduction and modern datasets | 15 | Atlas (no data needed) | — |
| 2 | Classical defect detection | 20 | MVTec AD 2 sheet_metal* | Otsu, adaptive, Canny, morphology, Gabor |
| 3 | Learning normality | 25 | MVTec AD 2 sheet_metal* | PaDiM vs. PatchCore |
| 4 | Low-latency anomaly detection | 25 | latency measurements | EfficientAD (explanation) + measured latencies |
| 5 | Foundation models | 25 | MVTec AD 2 sheet_metal* | AnomalyDINO vs. WinCLIP |
| 6 | 3D and multimodal inspection | 40 | MulSen-AD capsule/nut/flat_pad | 3D baseline, multimodal viewer, fusion |
| 7 | Algorithm comparison challenge | 30 | MVTec AD 2 sheet_metal* | Compare Methods |
| A | Why does image resolution matter? | 15 | MVTec AD 2 sheet_metal | PatchCore 256×256 vs. 1024×256 (verified) |
| B | Can a model handle changing illumination? | 15 | MVTec AD 2 sheet_metal, fruit_jelly | any verified model, scene-aware statistics |
| C | Why can a good AUROC produce poor decisions? | 15 | stored scores of verified models | PaDiM fruit_jelly example, thresholds |

\* Fallback if MVTec AD 2 is not installed: MulSen-AD RGB images (real data).

Activities A–C (MVTec AD 2 deep-dive, *Educational Experiments* page) can replace part of Activity 7
or be used as homework. They show real results of Workshop Verified models and never train.

## Instructor checklist (day before, while online)
1. `.\setup.ps1` on every workstation (or a prepared image of the folder).
2. `.\prepare_models.ps1 -WithWinCLIP` (≈ 1.2 GB, cached in `checkpoints/`).
3. Import MVTec AD 2 `sheet_metal`; fetch the MulSen-AD subset.
4. `.\prepare_models.ps1 -WithEfficientAD` (teacher weights 40 MB + Imagenette2-320 342 MB).
5. **Pre-train / pre-fit all models** (also fills the fitted-model cache in `checkpoints/fitted/`, so
   students' *Fit* loads instantly with identical results):
   `.\.venv\Scripts\python.exe -m scripts.run_experiments --dataset D05 --categories sheet_metal fruit_jelly --methods patchcore padim efficientad anomalydino --n-fit 1000 --n-cal 60 --localization`
   and `... --dataset D06 --categories capsule nut flat_pad --fusion`
   (EfficientAD ≈ 6 min per category on an RTX 4090 Laptop; everything else seconds to ≈ 1 min).
   The cache is valid for the same fit images, parameters, seed (Settings, default 42) and Anomalib version.
   **MVTec AD 2 models were fitted on all train/good images.** Load them from the Lab's *Fitted
   models (Model Registry)* box or let Compare use them (*Use Workshop Verified models*) — no need
   to change *Fit images*. If *Fit* would train a new model, the Lab asks first.
6. `.\.venv\Scripts\python.exe -m scripts.verify_workshop_models` — reloads every cached preset, re-evaluates it,
   checks it reproduces the stored metrics and writes `configs/workshop_verified.json` (never trains;
   ≈ 45 min for all 24 presets). Optional audit: `python -m scripts.audit_d05`.
7. Disconnect from the network and run *Settings → Diagnostics & readiness → Offline readiness check*
   (includes SHA-256 and cache-key checks of every Workshop Verified checkpoint).
8. On the day, launch with **`.\start.ps1 -Workshop`** (offline demo profile): Methods Laboratory and Compare
   Methods load Workshop Verified checkpoints (Model Registry box; Compare: *Use Workshop Verified models*)
   and ask before any training. Students never need to change *Fit images*.

## Timing tips (measured on the development laptop, RTX 4090 Laptop GPU)
- PatchCore fitting on 54 images ≈ 7 s; inference ≈ 10–22 ms per image. On CPU allow minutes.
- EfficientAD: 10 000 training steps ≈ 6 min (2 000 steps ≈ 1.5 min but clearly lower accuracy).
- WinCLIP first load ≈ 5–10 s (model on disk).
- 3D alignment of a MulSen-AD scan ≈ 0.3–2 s.

## Teaching points to emphasise
- Raw scores are not probabilities; compare methods on calibrated scales.
- Thresholds from test labels = optimistic bias.
- Local subset results ≠ official benchmark results (MVTec AD 2 private test server).
- Infrared images are visualisations, not temperatures.
