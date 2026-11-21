## Can a model handle changing illumination?

**Intuition.** Unsupervised models learn "normal" from training images taken under one lighting set-up.
In MVTec AD 2 all train and validation images are `regular`; `test_public` re-photographs the *same
scenes* (the same physical parts) as `regular`, `overexposed`, `underexposed` and lighting shifts
(`shift_1…3`). A robust model should give a scene the same verdict whatever the lighting. A model that
learned the brightness of normal images, instead of their structure, raises false alarms on normal
parts under unseen light, or misses defects when contrast drops.

**What to look at.**
1. Pick a **scene**: the strip shows the same part under every condition, with its score and OK/NOK
   decision at the label-free validation threshold (✗ = error: false alarm or missed defect).
2. Press *Compute anomaly maps*: maps of all conditions use one colour scale, so you can see where the
   evidence moves when only the light changes.
3. The **line plot** has one line per scene. Flat lines mean stable predictions; lines crossing the green
   threshold mean the decision depends on lighting.
4. The **scene-aware analysis** below.

**Why scenes, not images.** sheet_metal has 114 test images but only 19 scenes (4 normal + 15 anomalous)
× 6 conditions; fruit_jelly 80 images = 20 scenes × 4. Variants of one scene are strongly correlated, so
they are **not** independent observations. IDD Lab therefore
* reports AUROC per condition (one image per scene, independent within the condition),
* reports a scene-level AUROC (one aggregated score per scene),
* computes confidence intervals by **cluster bootstrap**: resampling whole scenes (stratified by class)
  and keeping all their lighting variants together. The naive image-level interval is shown next to it:
  it pretends there are 114 (or 80) independent images and is usually narrower. With only 4–5 normal
  scenes, however, even the cluster bootstrap is unreliable (too few clusters) and can occasionally come
  out narrower than the naive one (e.g. PaDiM on fruit_jelly) — read both as rough indications only.

**Limitations.** With 4–5 normal scenes every interval is wide and coarse: removing or duplicating a
single normal scene can move AUROC by up to ≈ 0.2–0.25 (with 4 normal scenes, each one contributes a quarter of all normal–anomalous comparisons). Per-condition differences are exploratory and must not be
read as a ranking of lighting conditions. Below 3 scenes per class IDD Lab reports no interval at all.
Local educational results on `test_public` — the official robustness evaluation uses the private
`test_private_mixed` split on MVTec's server.

**Further reading.** Vo, Pham, Pham & Nguyen, *IEEE Access* 14 (2026), doi:10.1109/ACCESS.2026.3735579 —
an analysis of resolution, spatial structure, illumination and seed sensitivity of training-free
anomaly localization on MVTec AD 2 (Atlas → MVTec AD 2 → complementary references).
