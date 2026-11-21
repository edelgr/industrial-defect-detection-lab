# Exercise 4 — Resolution Matters

| | |
|---|---|
| **Duration** | 15 min |
| **Required hardware** | Any laptop for the stored results; live comparison: full profile with a CUDA GPU |
| **Required dataset** | None for the stored results; live comparison: MVTec AD 2 `sheet_metal` (official download) |
| **Required checkpoint** | None for the stored results; live comparison: the two Workshop Verified PatchCore checkpoints (instructor installation) or your own models fitted with *Input size 256* and with *Keep aspect ratio* |

## Learning objectives
- Understand the difference between square resizing and aspect-preserving preprocessing.
- See why localisation metrics react more than image AUROC to resolution.
- Relate resolution to latency and GPU memory.

## Instructions
1. **Educational Experiments → A · Why does image resolution matter?** Read the *Explanation* tab.
2. Read the stored table (real experiment on MVTec AD 2 `sheet_metal`, verified on the instructor's computer):
   PatchCore 256×256 vs 1024×256 — image AUROC 0.741 vs 0.764, AU-PRO@0.05 0.049 vs 0.192,
   GPU memory 278 MB vs 802 MB (local educational results, not official MVTec AD 2 scores).
3. If the banner says **LIVE**, choose a defective image and press *Run both PatchCore models*; compare
   *what the network sees*, the heat-maps (same colour scale), the predicted-mask error view and the zoom.
4. If the banner says **STORED RESULTS**, the live part needs the dataset and the checkpoints listed in it.

## Expected qualitative observations
- Squashing a 4224×1056 image to 256×256 shrinks it 16.5× horizontally; thin scratches fall below one feature cell.
- Aspect-preserving input improves localisation much more than image-level AUROC, at a higher cost.

## Reflection questions
- Why can image AUROC stay similar while AU-PRO changes a lot?
- Which resolution would you deploy for a 100 ms cycle?

## Scientific references
`hecklerkram2026mvtecad2` (the benchmark asks participants to disclose the image size), `roth2022patchcore`,
`vo2026beyondbenchmark` (complementary analysis of resolution, illumination and seed sensitivity on MVTec AD 2).

## Alternative path
- Without MVTec AD 2 and checkpoints the stored, verified results are shown — clearly labelled, never as live inference.
