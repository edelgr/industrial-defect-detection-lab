### How industrial anomaly-detection benchmarks evolved

**2019 — MVTec AD: the unsupervised protocol.** Real factories rarely have enough examples of
every defect. MVTec AD fixed a protocol that matches this reality: *train on defect-free images
only*, then detect and localise unseen defects. Pixel-precise masks allowed localisation to be
measured, not only classification. Most methods you will meet today (PaDiM, PatchCore,
EfficientAD) were developed against it — and by 2022 many categories exceeded 99 % image AUROC.

**2022 — MVTec LOCO AD: logic, not only texture.** A breakfast box with *two* oranges instead of
one has no scratch anywhere. LOCO added *logical* anomalies (missing, extra or misplaced parts)
that patch-local methods cannot see, and a new localisation metric (sPRO).

**2022 — VisA: diversity.** Twelve product types with multiple instances, complex PCBs and
non-aligned objects tested whether methods generalise beyond MVTec's well-aligned scenes; its
few-shot splits fuelled zero/few-shot research.

**2023 — Real3D-AD: geometry alone.** High-resolution 360° point clouds with only four normal
prototypes per class: no colour, no texture — only shape, and registration becomes essential.

**2025/26 — MVTec AD 2: realism and honest evaluation.** Lighting changes between training and
test, transparent and overlapping objects, tiny defects on very large images — and **private test
labels** evaluated only on MVTec's server, so nobody can tune to the test set. State-of-the-art
methods drop substantially.

**2025 — MulSen-AD and Real-IAD D³: more than one sensor.** Some defects are *inside* the part and
only visible to infrared thermography; others are geometric and only visible to a 3D scanner.
MulSen-AD combines RGB, lock-in infrared and 3D laser scans; Real-IAD D³ combines RGB,
photometric-stereo pseudo-3D and structured-light 3D.

**2025/26 — 3D-ADAM: additive manufacturing.** 3D-printed parts have layer lines and process
artefacts that are *normal*, making the definition of "anomaly" itself harder.

> **Why it matters:** a method that is excellent on one benchmark can fail on another, because
> each benchmark tests a different assumption — alignment, locality, illumination, modality.
