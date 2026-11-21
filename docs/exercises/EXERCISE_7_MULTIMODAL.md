# Exercise 7 — Multimodal Anomaly Detection

| | |
|---|---|
| **Duration** | 30 min |
| **Required hardware** | Full profile (Open3D, PyVista; PyTorch for the RGB/IR detectors); OpenGL-capable display |
| **Required dataset** | Synthetic 3D surface patches (generated) — real part: MulSen-AD `capsule`, `nut`, `flat_pad` (official distribution route, see DATASETS.md) |
| **Required checkpoint** | None for 3D registration; per-modality PatchCore is fitted on train/good during the experiment |

## Learning objectives
- Compare RGB, infrared and 3D information.
- Understand registration (FPFH + RANSAC, ICP) and geometric deviation.
- Understand late fusion and per-modality labels.

## Instructions
1. **3D & Multimodal Lab** → *Synthetic surface patches*: align a test scan to the normal reference and colour
   it by deviation. Find the bump/dent; read the registration fitness.
2. With MulSen-AD installed: open the synchronised RGB · infrared · 3D viewer for one object; read the
   per-modality labels (an object can be anomalous in one modality only).
3. Run the 3D baseline on a MulSen-AD scan; check the *registration failed* warning when fitness is low.
4. Read the stored fusion table in `docs/LOCAL_RESULTS.md` (local educational results, not the MulSen-AD
   official benchmark method).

## Expected qualitative observations
- Geometric defects can be invisible in RGB; some defects are only visible in infrared.
- A failed registration produces large false deviations everywhere.

## Reflection questions
- Which defects can only be seen in 3D? Which only in infrared?
- Why do per-modality labels matter when evaluating a single-modality detector?

## Scientific references
`li2025mulsenad`, `besl1992icp`, `rusu2009fpfh`, `chen1992pointplane`, `wang2023m3dm`.

## Alternative path
- Core profile (no Open3D): read the Methods Library cards *Registration + geometric deviation* and *M3DM*, and
  the 3D topic in **Start Learning**. MulSen-AD's licence is not stated by its authors: never redistribute it.
