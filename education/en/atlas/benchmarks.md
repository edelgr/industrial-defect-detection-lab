## What is an industrial benchmark?

A benchmark is more than a folder of images. It fixes **five things** so that results from
different laboratories can be compared:

1. **Data** — which products, sensors and conditions are included.
2. **Splits** — which images may be used for training, validation (e.g. threshold selection) and
   testing. Test data must never influence training or thresholds.
3. **Annotations** — image labels, pixel masks, per-region masks, point labels, per-modality labels.
4. **Metrics** — e.g. image AUROC (detection), pixel AUROC / PRO / sPRO (localisation), SegF1
   (thresholded segmentation).
5. **Protocol** — how metrics are computed and reported (e.g. MVTec AD 2's private test server).

### Why protocols change published results
- Choosing a threshold on test labels inflates F1, precision and recall.
- Pixel AUROC is dominated by the many normal pixels and can look excellent even when small
  defects are missed — PRO and sPRO weigh each defect region equally.
- Resizing large images to 256 px can make tiny defects disappear before the method sees them.
- Different train/test splits (e.g. VisA 1-class vs. few-shot splits) are different experiments.

### Four kinds of results — never mix them
| Category | Meaning in IDD Lab |
|---|---|
| **Author-reported** | Numbers quoted from a paper, with the citation |
| **Reproduced benchmark** | The official protocol re-run in full |
| **Local educational** | A local experiment on a subset or a non-official protocol |
| **Synthetic demonstration** | Computed on synthetic data: illustrates behaviour only |

IDD Lab stores every run with its category and protocol string (Settings → Experiment log).
