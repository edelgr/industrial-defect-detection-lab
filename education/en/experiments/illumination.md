## Can a model handle changing illumination?

**Intuition.** Unsupervised models learn "normal" from training images taken under one lighting set-up.
In MVTec AD 2 all train and validation images are `regular`; `test_public` re-photographs the *same
scenes* (the same physical parts) as `regular`, `overexposed`, `underexposed` and lighting shifts
(`shift_1…3`). A robust model should give a scene the same verdict whatever the lighting. A model that
learned the brightness of normal images, instead of their structure, raises false alarms on normal
parts under unseen light, or misses defects when contrast drops.

**What to look at — Basic mode.**
1. Pick a **scene** (◀ ▶ to step through them). Each **card** is one lighting condition of the same part: the
   image (ground truth outlined), its anomaly map, a bar with the score against the threshold, the OK/NOK
   decision and its outcome — **TP** (defect detected), **TN** (good part accepted), **FP** (false alarm) or
   **FN** (missed defect). The threshold is the label-free validation threshold.
2. Press *Compute anomaly maps* (LIVE, where available): all maps use one colour scale, so you can see where
   the evidence moves when only the light changes. Double-click an image or press ⤢ to enlarge image and map
   side by side.
3. The **chart** shows this scene's score under every condition, with the OK and NOK zones; a ◆ marks where
   the decision changes. *What happened?* explains the result in plain words.

**What to look at — Advanced mode.**
1. The **chart of all scenes** has one line per scene: click a line to select that scene (the others are
   dimmed), click a point to select a condition. ✗ marks the selected scene's errors, ◆ its threshold crossings.
   Lines only connect the same scene — the conditions are categories, not a continuum.
2. *Scenes × lighting*: every decision as TP/TN/FP/FN, with errors, decision changes, score range and the
   smallest distance to the threshold per scene. *AUROC per lighting*: per-condition, pooled and scene-level
   AUROC with 95 % cluster-bootstrap intervals. *Statistics*: the scene-aware analysis and its limitations.
   *References*: the papers behind the dataset and the method.

**Stored or live?** Scores, decisions, AUROCs and intervals always come from the **stored** per-image scores of
the verified run. LIVE inference only adds the anomaly maps; each map shows its live score next to the stored
one.

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
