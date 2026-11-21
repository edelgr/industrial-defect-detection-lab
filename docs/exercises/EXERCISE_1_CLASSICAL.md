# Exercise 1 — Classical Defect Detection

| | |
|---|---|
| **Duration** | 25 min |
| **Required hardware** | Any Windows 10/11 laptop (CPU only) |
| **Required dataset** | Synthetic Educational Data (generated automatically) — optional: MVTec AD 2 `sheet_metal` |
| **Required checkpoint** | None |
| **Edition** | Student core profile or higher |

## Learning objectives
- Understand image differences, filtering and thresholding.
- See why fixed rules break under lighting changes and texture.
- Read a ground-truth mask and compare it with a predicted mask.

## Instructions
1. Open **Dataset Explorer**, choose *Synthetic Educational Data → brushed_metal*. Look at one image of
   each defect type (scratch, dent, stain, spot, geometric, texture) with the ground-truth overlay.
   The banner reminds you: *Synthetic Educational Data — Not a Real Industrial Benchmark*.
2. Open **Methods Laboratory**, select a *scratch* image and run **Otsu**. Move the pixel threshold slider.
3. Run **Canny**, then **Morphology (black-hat)**, then **Gabor** on the same image. Compare the
   predicted masks with the ground truth.
4. Select a *texture* defect. Which method reacts to it? Which ignores it?
5. Run **Reference difference** and **SSIM** (they compare with a normal reference image). Change the
   reference in the drop-down: what happens when the reference is lit differently?
6. Press **Evaluate on test split** for one method: read image AUROC, precision/recall and the
   *synthetic_demo* result category.

## Expected qualitative observations
- Thresholding finds dark scratches and spots but also reacts to the illumination gradient.
- Edge detectors respond to every texture line; morphology suppresses small structures.
- Texture defects are hard for intensity-based rules; Gabor responds to orientation changes.
- Reference comparison works when alignment and lighting match, and fails when they do not.

## Reflection questions
- Why is a single global threshold fragile on a real production line?
- What would you need to control in the hardware (lighting, fixture) to make classical methods robust?
- Why are results on synthetic data *illustrative only*?

## Scientific references
- Otsu (1979), Canny (1986), Jain & Farrokhnia (1991, Gabor), Serra (1982, morphology), Wang et al. (2004, SSIM)
  — open them from the **Methods Library** cards (*Scientific References & Source Code*).

## Alternative path
- No synthetic data? Run `.\prepare_datasets.ps1 -Synthetic` or restart IDD Lab (it is generated on first run).
- With MVTec AD 2 `sheet_metal` installed (official download, see DATASETS.md), repeat steps 2–5 on a real image.
