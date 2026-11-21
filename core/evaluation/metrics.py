"""Evaluation metrics computed only from *real* labels.

Protocol rules enforced here and in ``core.evaluation.protocol``:
* metrics are computed only for samples whose ground truth is public;
* thresholds used for P/R/F1/FAR/FRR must come from a source other than the evaluated test
  labels (e.g. validation data, a fixed value, or a normal-only percentile) — otherwise the
  result is flagged ``threshold_from_test=True`` and must be reported as optimistic.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def _both_classes(y) -> bool:
    y = np.asarray(y)
    return y.size > 0 and 0 < y.sum() < y.size


def image_auroc(labels, scores) -> float | None:
    return float(roc_auc_score(labels, scores)) if _both_classes(labels) else None


def image_ap(labels, scores) -> float | None:
    return float(average_precision_score(labels, scores)) if _both_classes(labels) else None


def pixel_auroc(masks: list[np.ndarray], maps: list[np.ndarray], max_pixels: int = 4_000_000,
                seed: int = 0) -> float | None:
    """Pixel-level AUROC over all test images (sub-sampled uniformly if very large)."""
    y = np.concatenate([m.ravel().astype(np.uint8) for m in masks])
    s = np.concatenate([a.ravel().astype(np.float32) for a in maps])
    if y.size > max_pixels:
        idx = np.random.default_rng(seed).choice(y.size, max_pixels, replace=False)
        y, s = y[idx], s[idx]
    return float(roc_auc_score(y, s)) if _both_classes(y) else None


def au_pro(masks: list[np.ndarray], maps: list[np.ndarray], fpr_limit: float = 0.3) -> float | None:
    """Normalised area under the PRO curve up to ``fpr_limit`` (Bergmann et al., IJCV 2021).

    PRO = mean, over every connected ground-truth region of every test image, of the fraction of
    that region's pixels above the threshold; FPR is measured on all normal pixels. The curve is
    integrated from FPR 0 to ``fpr_limit`` (linear interpolation at the limit) and divided by
    ``fpr_limit`` so that 1.0 is perfect.  Uses the same sorted-cumulative-sum construction as the
    MVTec reference evaluation code; thresholds are every distinct score value.
    """
    import cv2
    y_parts, w_parts, s_parts = [], [], []
    n_regions = 0
    comps = []
    for m in masks:
        k, lab = cv2.connectedComponents(m.astype(np.uint8), connectivity=8)
        comps.append((k - 1, lab))
        n_regions += k - 1
    if n_regions == 0:
        return None
    for (k, lab), a in zip(comps, maps):
        lab = lab.ravel()
        sizes = np.bincount(lab, minlength=k + 1).astype(np.float64)
        w = np.zeros(lab.size, np.float64)
        pos = lab > 0
        w[pos] = 1.0 / (sizes[lab[pos]] * n_regions)       # PRO increment per anomalous pixel
        y_parts.append(~pos)
        w_parts.append(w)
        s_parts.append(a.ravel().astype(np.float32))
    normal = np.concatenate(y_parts)
    n_normal = int(normal.sum())
    if n_normal == 0:
        return None
    pro_inc = np.concatenate(w_parts)
    s = np.concatenate(s_parts)
    del y_parts, w_parts, s_parts
    order = np.argsort(-s, kind="stable")
    s = s[order]
    fpr = np.cumsum(normal[order], dtype=np.float64) / n_normal
    pro = np.cumsum(pro_inc[order])
    del order, normal, pro_inc
    last = np.r_[s[1:] != s[:-1], True]                      # keep one point per distinct threshold
    fpr = np.r_[0.0, fpr[last]]
    pro = np.r_[0.0, pro[last]]
    if fpr[-1] <= fpr_limit:
        x, yv = fpr, pro
    else:
        j = int(np.searchsorted(fpr, fpr_limit, side="right"))
        f0, f1, p0, p1 = fpr[j - 1], fpr[j], pro[j - 1], pro[j]
        p_lim = p0 + (p1 - p0) * (fpr_limit - f0) / (f1 - f0) if f1 > f0 else p0
        x, yv = np.r_[fpr[:j], fpr_limit], np.r_[pro[:j], p_lim]
    return float(np.trapezoid(yv, x) / fpr_limit)


def pixel_f1(masks: list[np.ndarray], preds: list[np.ndarray]) -> dict | None:
    """Pixel-level precision/recall/F1 aggregated over *all* pixels of all test images
    (binary predictions from a threshold that was fixed without test labels)."""
    tp = fp = fn = 0
    for g, p in zip(masks, preds):
        g, p = g.astype(bool), p.astype(bool)
        tp += int((g & p).sum())
        fp += int((~g & p).sum())
        fn += int((g & ~p).sum())
    if tp + fn == 0:
        return None
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn)
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"precision": prec, "recall": rec, "f1": f1}


def dice_iou(pred: np.ndarray, gt: np.ndarray) -> tuple[float | None, float | None]:
    pred, gt = pred.astype(bool), gt.astype(bool)
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    tot = pred.sum() + gt.sum()
    if tot == 0:
        return None, None   # undefined for an empty-vs-empty pair (normal image correctly empty)
    return float(2 * inter / tot), float(inter / union)


@dataclass
class DecisionMetrics:
    threshold: float
    tp: int
    fp: int
    tn: int
    fn: int
    precision: float | None
    recall: float | None
    f1: float | None
    false_acceptance_rate: float | None   # defective parts accepted as OK  (FN / positives)
    false_rejection_rate: float | None    # good parts rejected as NOK      (FP / negatives)
    threshold_source: str = "unspecified"
    threshold_from_test: bool = False

    def to_dict(self):
        return asdict(self)


def decision_metrics(labels, scores, threshold: float, threshold_source: str = "unspecified",
                     threshold_from_test: bool = False) -> DecisionMetrics:
    y = np.asarray(labels).astype(bool)
    p = np.asarray(scores) >= threshold
    tp, fp = int((p & y).sum()), int((p & ~y).sum())
    tn, fn = int((~p & ~y).sum()), int((~p & y).sum())
    prec = tp / (tp + fp) if tp + fp else None
    rec = tp / (tp + fn) if tp + fn else None
    f1 = 2 * prec * rec / (prec + rec) if prec and rec else (0.0 if prec == 0 or rec == 0 else None)
    far = fn / (tp + fn) if tp + fn else None
    frr = fp / (fp + tn) if fp + tn else None
    return DecisionMetrics(float(threshold), tp, fp, tn, fn, prec, rec, f1, far, frr,
                           threshold_source, threshold_from_test)


def threshold_from_normals(normal_scores, percentile: float = 99.0) -> float:
    """Label-free threshold: a high percentile of scores on held-out *normal* images
    (e.g. a validation split of good samples).  Never uses test labels."""
    return float(np.percentile(np.asarray(normal_scores), percentile))


def latency_stats(latencies_ms) -> dict:
    a = np.asarray(latencies_ms, dtype=float)
    if a.size == 0:
        return {}
    return {"mean_ms": float(a.mean()), "median_ms": float(np.median(a)),
            "p95_ms": float(np.percentile(a, 95)), "n": int(a.size)}
