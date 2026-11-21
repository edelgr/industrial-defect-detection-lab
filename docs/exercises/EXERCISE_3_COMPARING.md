# Exercise 3 — Comparing Anomaly Detection Algorithms

| | |
|---|---|
| **Duration** | 30 min |
| **Required hardware** | Full profile; a CUDA GPU is strongly recommended for EfficientAD (training 10 000 steps takes ≈ 6 min on an RTX 4090 Laptop GPU and is impractical on a CPU) |
| **Required dataset** | MVTec AD 2 (official download) for real samples; Synthetic Educational Data otherwise |
| **Required checkpoint** | Fitted PaDiM / PatchCore / EfficientAD models for the chosen category (fit them yourself, or use the instructor's Workshop Verified models in the workshop room) |

## Learning objectives
- Compare PaDiM, PatchCore and EfficientAD on the same samples with a fair protocol.
- Use calibrated scales — raw scores of different methods are not comparable.
- Weigh accuracy against latency and memory.

## Instructions
1. **Model Manager**: check which methods are runnable here and which models are fitted.
2. **Compare Methods**: select PaDiM, PatchCore and EfficientAD; keep *Calibrate each method on normal images*
   enabled. In the workshop room keep *Use Workshop Verified models* on (no training).
3. Pick three samples: a small defect, a large defect and a normal image. Run the comparison; export CSV.
4. Note for each method: decision (OK/NOK) at its calibrated threshold, heat-map quality and latency.
5. Without models: open **Educational Experiments → C** and compare the *stored* image AUROC / AU-PRO of the
   verified models across MVTec AD 2 `sheet_metal` and `fruit_jelly`.

## Expected qualitative observations
- No method wins everywhere; failure modes differ (small defects, texture, lighting).
- Calibrated colour scales make heat-maps comparable; raw scales do not.
- Faster methods are not necessarily less accurate, and vice versa.

## Reflection questions
- Which result would you report in a paper, and which only in class?
- How would you choose a method for a 100 ms cycle time?

## Scientific references
`defard2021padim`, `roth2022patchcore`, `batzner2024efficientad`, `hecklerkram2026mvtecad2` (open from the Methods Library / Atlas).

## Alternative path
- No dataset / no models: step 5 (stored results, clearly labelled) and Exercise 6.
