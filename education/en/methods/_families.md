## Families of industrial defect and anomaly detection

Select a method on the left. Filled dots (●) are runnable on this machine; empty dots (○) are
explained but not executed in IDD Lab v0.1.

| Group | Idea | Training data | Typical output |
|---|---|---|---|
| **A — Classical CV** | Hand-designed operators: thresholds, edges, texture filters, morphology, template comparison | None or one reference | Masks / response maps |
| **B — Supervised** | Learn from labelled defects (classification, detection, segmentation) | Many labelled defects | Classes, boxes, pixel masks |
| **C — Reconstruction** | Learn to redraw normal images; badly reconstructed regions are anomalous | Normal images (training from scratch) | Residual maps |
| **D — Feature distributions & memory banks** | Pretrained features + statistics (PaDiM) or nearest neighbours (PatchCore) | Normal images, no gradient training | Distance maps |
| **E — Teacher–Student** | A student imitates a teacher on normal data; disagreement = anomaly | Normal images (training) | Disagreement maps |
| **F — Foundation models** | Generic pretrained vision / vision-language features, zero- or few-shot | 0–k normal images | Similarity maps |
| **G — 3D & multimodal** | Geometry, registration, several sensors and fusion | Normal scans / samples | Per-point or per-modality scores |

**Supervised vs. anomaly detection.** Supervised models (B) answer "which of the defect classes I
was trained on is this?". Anomaly detectors (C–G) answer "is this different from normal?" — they
can find defect types never seen before, which matters because defects are rare and diverse.

**Classification vs. detection vs. segmentation.** Classification gives one label per image;
detection gives boxes around instances; segmentation gives a label per pixel. Anomaly maps are a
form of unsupervised segmentation.
