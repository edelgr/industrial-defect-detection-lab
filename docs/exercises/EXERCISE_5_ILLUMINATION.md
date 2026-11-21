# Exercise 5 — Illumination Robustness

| | |
|---|---|
| **Duration** | 15 min |
| **Required hardware** | Any laptop for stored scores and the scene analysis; live anomaly maps: full profile + GPU |
| **Required dataset** | None for stored scores; images and live maps: MVTec AD 2 `sheet_metal` / `fruit_jelly` |
| **Required checkpoint** | None for stored scores; live maps: Workshop Verified checkpoints (instructor) or your own fitted models |

## Learning objectives
- Investigate why the same object can receive different anomaly scores under different lighting conditions.
- Understand why lighting variants of one scene are not independent observations.
- Read cluster-bootstrap confidence intervals — and their limits with very few scenes.

## Instructions
1. **Educational Experiments → B · Can a model handle changing illumination?** Choose *PatchCore · sheet_metal*.
2. Select scenes marked *⚠ decision flips*: compare stored scores and OK/NOK decisions across `regular`,
   `overexposed`, `underexposed` and `shift_k`.
3. Read the line plot (one line per scene) and the scene-aware analysis below it.
4. Compare the *cluster* (scene) interval with the *naive* (image) interval. With only 4–5 normal scenes
   neither is reliable; the report says so.
5. If live inference is available, press *Compute anomaly maps for this scene*.

## Expected qualitative observations
- Some scenes change decision when only the lighting changes (6 of 19 scenes for PatchCore 256×256 on
  `sheet_metal` in the stored results).
- Training and validation images use regular lighting only; other conditions are unseen during fitting.

## Reflection questions
- Why would 114 images give a misleadingly precise AUROC?
- How would you collect data to make a detector robust to lighting?

## Scientific references
`hecklerkram2026mvtecad2`, `vo2026beyondbenchmark`, `roth2022patchcore`, `defard2021padim`.

## Alternative path
- Without the dataset, image thumbnails show "image not installed on this computer"; stored scores, decisions
  and the statistics remain available.
