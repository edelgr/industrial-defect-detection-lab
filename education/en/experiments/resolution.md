## Why does image resolution matter?

**Intuition.** A sheet_metal image is 4224 × 1056 pixels — four times wider than tall — and a typical
defect covers about 0.25 % of it (median of the 90 public masks), some only ≈ 0.01 %. Before PatchCore
sees anything, the image is resized to the network input:

| Variant | Network input | Shrink factor (x / y) | Geometry |
|---|---|---|---|
| Square (paper baseline size) | 256 × 256 | 16.5 × / 4.1 × | **squashed**: horizontal detail is compressed 4× more than vertical |
| Aspect kept | 1024 × 256 | 4.1 × / 4.1 × | undistorted |

PatchCore compares *patch features* of the test image with a memory bank of normal patches. At 256 × 256
each feature cell of `layer2` covers ≈ 132 × 33 original pixels — a thin scratch becomes a sub-cell
signal and is averaged away. Squashing also changes the *shape* of normal texture (brushing marks,
edges) relative to the ImageNet patterns the backbone learned, so even the notion of "normal patch" is
distorted. With the aspect ratio kept, each cell covers ≈ 33 × 33 pixels: 4× more patches, more detail,
but also more memory (bigger memory bank) and more latency.

**What to look at.**
1. *What the network sees*: compare the two resized inputs — find the defect in each.
2. *Heat-maps* use the same scale (map ÷ calibrated threshold), so colours are comparable.
3. *Predicted mask vs ground truth*: green = found defect pixels, red = false alarms, blue = missed pixels.
4. *Zoom*: the most informative view for small defects.
5. *Stored metrics*: AU-PRO@0.05 measures how much of **each** defect region is found while allowing
   only 5 % false-positive pixels — the localisation metric of the MVTec AD 2 benchmark.

**Interpretation (in the verified results below).** Image-level AUROC changes little, while localisation metrics (AU-PRO, pixel F1) change a lot. A plausible
explanation (a hypothesis to discuss, not something this experiment proves): the image score only needs
the *highest* patch to stand out, whereas localisation needs the *extent and position* of small defects,
which survive better at the higher resolution. Resolution is a cost/accuracy trade-off:
the MVTec AD 2 paper asks leaderboard participants to disclose the image size they used.

**Limitations.** One category, one backbone, one seed; `test_public` has only 15 defective and 4 normal
scenes, each under 6 lighting conditions, so differences of a few AUROC points are within noise.
Local educational results — not official MVTec AD 2 scores.

**Further reading.** Vo, Pham, Pham & Nguyen, *IEEE Access* 14 (2026), doi:10.1109/ACCESS.2026.3735579 —
an analysis of resolution, spatial structure, illumination and seed sensitivity of training-free
anomaly localization on MVTec AD 2 (Atlas → MVTec AD 2 → complementary references).
