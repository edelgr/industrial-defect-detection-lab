# Exercise 2 — Understanding PatchCore

| | |
|---|---|
| **Duration** | 25 min |
| **Required hardware** | Concept part: any laptop. Live part: full profile (PyTorch); a CUDA GPU makes fitting fast, a CPU works for the synthetic set |
| **Required dataset** | Synthetic Educational Data (live part) — optional: MVTec AD 2 |
| **Required checkpoint** | None for the concept part; the live part fits PatchCore on normal images (needs the Wide-ResNet-50-2 backbone, ≈ 270 MB, downloaded from Hugging Face after your confirmation) |

## Learning objectives
- Learn how normal feature representations identify visual anomalies.
- Understand the memory bank, coreset subsampling and nearest-neighbour distance.
- Distinguish raw anomaly scores from probabilities.

## Instructions
1. **Methods Library → PatchCore → How This Algorithm Works**: step through the conceptual diagrams
   (patch features → memory bank → coreset → nearest neighbour → anomaly map). The diagrams are conceptual,
   not model outputs.
2. (Full profile) **Model Manager → PatchCore**: check the backbone, its source and licence; download it only
   if you agree.
3. (Full profile) **Methods Laboratory → PatchCore → Synthetic Educational Data**: *Fit on normal training
   images* (train/good only), *Calibrate thresholds*, then *Run* on a defective sample.
4. Change *Coreset sampling ratio* (requires re-fitting) and compare memory-bank size, latency and the map.
5. Without PyTorch: open **Educational Experiments → A** and study the stored PatchCore results on real
   MVTec AD 2 `sheet_metal` (clearly labelled *stored results*).

## Expected qualitative observations
- Defective patches have no close normal neighbour; their distance lights up in the map.
- A smaller coreset is faster and uses less memory, with some loss of detail.
- Raw scores are distances: their scale depends on the backbone and the data.

## Reflection questions
- Why does PatchCore need only normal images?
- What happens if the training images contain an undetected defect?
- Why is calibration on separate normal images necessary before making decisions?

## Scientific references
- Roth et al., *Towards Total Recall in Industrial Anomaly Detection*, CVPR 2022 (catalog: `roth2022patchcore`).
- Akçay et al., *Anomalib* (implementation used; catalog: `akcay2022anomalib`).

## Alternative path
- Core profile (no PyTorch): steps 1 and 5 only. Nothing is presented as live inference.
