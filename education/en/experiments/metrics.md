## Understanding Evaluation Metrics

### Ranking versus decision
An anomaly detector outputs a **score**, not a verdict. Two different questions can be asked:

* **Ranking** — *are defective parts scored higher than good ones?* Answered by threshold-free metrics
  (AUROC, AP, AU-PRO). No threshold is involved.
* **Decision** — *at the threshold we will actually use on the line, how many parts are accepted or
  rejected correctly?* Answered by TP/FP/TN/FN, precision, recall, F1, false-acceptance and
  false-rejection rates. These depend entirely on **one** chosen threshold.

A model can rank well and still decide badly, if the threshold is in the wrong place.

### The metrics, intuitively
* **Image AUROC** — pick one random defective and one random good image: AUROC is the probability that
  the defective one gets the higher score (0.5 = chance, 1.0 = perfect ranking). It ignores *how far*
  apart the scores are and where any threshold lies.
* **Pixel AUROC** — the same question for pixels: a random defect pixel versus a random normal pixel. On
  MVTec AD 2 defects are tiny (≈ 0.25–0.4 % of the image), so normal pixels dominate; a map can reach a
  high pixel AUROC while still missing whole small defects.
* **AU-PRO@0.05** — for every connected defect *region*, measure the fraction of its pixels above the
  threshold (per-region overlap, PRO) and average over regions, so a tiny scratch counts as much as a
  large dent. Sweep the threshold while the false-positive rate on normal pixels stays ≤ 5 %, take the
  area under that curve and normalise to 0–1. This is the main localisation metric of MVTec AD 2
  (the paper lowers the classic limit of 0.3 to 0.05 to tolerate fewer false-positive pixels).
* **F1** — harmonic mean of precision (of the parts rejected, how many were really defective?) and
  recall (of the defective parts, how many were rejected?). F1 is 0 if nothing is rejected.
* **False positives (FP)** — good parts rejected: scrap and re-inspection cost (false-rejection rate,
  FRR). **False negatives (FN)** — defective parts accepted: escapes to the customer
  (false-acceptance rate, FAR). Moving the threshold trades one for the other.

### Where the threshold comes from
A valid protocol fixes the threshold **without test labels**. IDD Lab uses the 99th percentile of the
scores of `validation/good` images (normal parts never used for fitting). The MVTec AD 2 paper's baseline
rule uses the mean + 3 standard deviations of the per-pixel scores of the defect-free validation
images, and the *maximum* of the anomaly map as the image score (Sec. IV-B2). Both are shown side by side.

If validation normals score lower than test normals (different lighting, fewer images, natural
variation), a percentile threshold can land **above every test image**: nothing is rejected, recall = 0,
F1 = 0 — even though the ranking (AUROC) may be reasonable. The real PaDiM · fruit_jelly example
reproduces exactly this situation: open it with the button above and compare the grey (validation),
blue (test normal) and red (test anomalous) histograms with the green threshold.

### Retrospective exploration is not evaluation
The slider lets you move the threshold over the *stored* test scores to see the precision/recall
trade-off. As soon as you move it, the threshold has been chosen **by looking at test labels** —
the orange banner marks those numbers as retrospective exploration. They are optimistic and must never
be reported as results (the MVTec AD 2 paper calls tuning on test data inappropriate for unsupervised
anomaly detection; its private test labels make it impossible on the official benchmark). The valid numbers are always those in the green column.
