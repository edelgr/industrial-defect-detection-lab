# Exercise 6 — Understanding Evaluation Metrics

| | |
|---|---|
| **Duration** | 20 min |
| **Required hardware** | Any laptop (CPU only) |
| **Required dataset** | None (uses stored per-image scores of verified models) |
| **Required checkpoint** | None |

## Learning objectives
- Understand AUROC, F1, AU-PRO and the importance of calibration thresholds.
- Distinguish ranking quality from decision quality.
- Recognise retrospective, test-informed threshold choices as invalid evaluation.

## Instructions
1. **Educational Experiments → C · Understanding Evaluation Metrics** → *Real example: PaDiM on fruit_jelly*.
2. Compare the grey (validation normals), blue (test normal) and red (test anomalous) histograms with the green
   validation threshold. Read the green column: TP 0, FN 60, image F1 0.000 although AUROC is 0.732.
3. Read the column for the MVTec AD 2 paper's baseline rule (mean + 3σ of validation pixel scores, map maximum).
4. Move the *Exploratory threshold* slider. Note the orange **RETROSPECTIVE EXPLORATION** banner.
5. Switch to another model and repeat.

## Expected qualitative observations
- A reasonable AUROC can coexist with F1 = 0 when the validation threshold lies above all test scores.
- Lower thresholds trade missed defects (false accepts) for false alarms (false rejects).

## Reflection questions
- Which metric would a quality manager ask for, and which a researcher?
- Why may numbers obtained with a test-informed threshold never be reported?

## Scientific references
`bergmann2021mvtecad_ijcv` (AU-PRO), `hecklerkram2026mvtecad2` (AU-PRO@0.05, SegF1, threshold rule), `defard2021padim`.

## Alternative path
- Works in every edition: the scores are stored results computed on the instructor's computer (labelled as such).
