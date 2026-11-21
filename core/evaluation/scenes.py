"""Scene-aware analysis of repeated observations (MVTec AD 2 ``test_public`` lighting variants).

In MVTec AD 2 ``test_public`` every *scene* (one physical part) is photographed under several
lighting conditions.  Those images are **not independent observations**: treating 114 images as
114 independent samples overstates the evidence.  This module therefore

* groups samples by scene (``(label, scene_id)``),
* computes per-condition AUROC (one image per scene per condition → independent within a condition),
* computes scene-level AUROC (one aggregated score per scene),
* gives percentile **cluster bootstrap** confidence intervals that resample *scenes* (stratified by
  class) and keep all lighting variants of a resampled scene together, and contrasts them with the
  naive image-level bootstrap that wrongly treats variants as independent,
* reports per-scene decision stability across lighting conditions.

With only a handful of normal scenes (4 in sheet_metal, 5 in fruit_jelly) the intervals are wide and
the bootstrap distribution is coarse; :func:`sufficiency` turns this into an explicit warning.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from scipy.stats import rankdata

MIN_SCENES_PER_CLASS_FOR_CI = 3       # below this no interval is reported at all
WARN_SCENES_PER_CLASS = 10            # below this intervals are flagged as very imprecise


def group(samples: list[dict]) -> dict[tuple[str, str], list[dict]]:
    """samples: dicts with sample_id, label ('good'|'bad'), score, lighting_condition, scene_id."""
    g: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for s in samples:
        if s.get("scene_id") is None:
            continue
        g[(s["label"], s["scene_id"])].append(s)
    for v in g.values():
        v.sort(key=lambda s: _cond_order(s.get("lighting_condition")))
    return dict(sorted(g.items()))


def _cond_order(c):
    order = ["regular", "overexposed", "underexposed", "shift_1", "shift_2", "shift_3"]
    return (order.index(c) if c in order else len(order), c or "")


def sufficiency(groups: dict) -> dict:
    n_good = sum(1 for (lab, _) in groups if lab == "good")
    n_bad = sum(1 for (lab, _) in groups if lab == "bad")
    m = min(n_good, n_bad)
    if m < MIN_SCENES_PER_CLASS_FOR_CI:
        level, msg = "insufficient", (f"Only {n_good} normal and {n_bad} anomalous scenes: confidence intervals are "
                                      "not reported. Treat every number as descriptive.")
    elif m < WARN_SCENES_PER_CLASS:
        level, msg = "very_limited", (f"{n_good} normal and {n_bad} anomalous scenes: intervals are shown but are wide and "
                                      "coarse (few distinct resamples); with ≤ 5 normal scenes each one carries ≥ 20 % of all "
                                      "normal–anomalous comparisons. Exploratory only.")
    else:
        level, msg = "adequate", f"{n_good} normal and {n_bad} anomalous scenes."
    return {"normal_scenes": n_good, "anomalous_scenes": n_bad, "level": level, "message": msg}


def auroc_ranks(labels, scores) -> float | None:
    """AUROC via the Mann–Whitney U statistic with average ranks for ties (identical to
    sklearn.metrics.roc_auc_score for binary labels; tested). Used inside the bootstrap loops:
    pure NumPy/SciPy, fast, and it avoids scikit-learn's dataframe-type probing, which crashed
    inside the Qt process (PySide6 import hook) when called thousands of times."""
    y = np.asarray(labels, bool)
    n1 = int(y.sum())
    n0 = y.size - n1
    if n1 == 0 or n0 == 0:
        return None
    r = rankdata(np.asarray(scores, float))
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def _auc(items):
    return auroc_ranks([s["label"] == "bad" for s in items], [s["score"] for s in items])


def cluster_bootstrap(groups: dict, stat, n_boot: int = 2000, seed: int = 0, alpha: float = 0.05):
    """Percentile CI of ``stat(list_of_scene_groups)`` resampling scenes within each class."""
    good = [v for (lab, _), v in groups.items() if lab == "good"]
    bad = [v for (lab, _), v in groups.items() if lab == "bad"]
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        gs = [good[i] for i in rng.integers(0, len(good), len(good))]
        bs = [bad[i] for i in rng.integers(0, len(bad), len(bad))]
        v = stat(gs + bs)
        if v is not None:
            vals.append(v)
    if not vals:
        return None
    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi), len(set(np.round(vals, 6)))


def naive_bootstrap(samples: list[dict], n_boot: int = 2000, seed: int = 0, alpha: float = 0.05):
    """Image-level stratified bootstrap that (wrongly) treats lighting variants as independent."""
    good = [s for s in samples if s["label"] == "good"]
    bad = [s for s in samples if s["label"] == "bad"]
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        it = [good[i] for i in rng.integers(0, len(good), len(good))] + [bad[i] for i in rng.integers(0, len(bad), len(bad))]
        v = _auc(it)
        if v is not None:
            vals.append(v)
    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def analyse(samples: list[dict], image_threshold: float | None, n_boot: int = 2000, seed: int = 0) -> dict:
    """Full scene-aware analysis for one model on one category."""
    groups = group(samples)
    suff = sufficiency(groups)
    ci_ok = suff["level"] != "insufficient"
    flat = [s for v in groups.values() for s in v]
    out: dict = {"sufficiency": suff, "n_images": len(flat), "n_scenes": len(groups)}

    def pooled(gs):
        return _auc([s for v in gs for s in v])
    out["pooled_auroc"] = _auc(flat)
    if ci_ok:
        out["pooled_ci_cluster"] = cluster_bootstrap(groups, pooled, n_boot, seed)
        out["pooled_ci_naive"] = naive_bootstrap(flat, n_boot, seed)

    def scene_mean(gs):
        return _auc([{"label": v[0]["label"], "score": float(np.mean([s["score"] for s in v]))} for v in gs])

    def scene_max(gs):
        return _auc([{"label": v[0]["label"], "score": float(np.max([s["score"] for s in v]))} for v in gs])
    out["scene_auroc_mean"] = scene_mean(list(groups.values()))
    out["scene_auroc_max"] = scene_max(list(groups.values()))
    if ci_ok:
        out["scene_auroc_mean_ci"] = cluster_bootstrap(groups, scene_mean, n_boot, seed)

    conds = sorted({s.get("lighting_condition") for s in flat}, key=_cond_order)
    per = {}
    for c in conds:
        def at_c(gs, c=c):
            return _auc([s for v in gs for s in v if s.get("lighting_condition") == c])
        items = [s for s in flat if s.get("lighting_condition") == c]
        per[c] = {"n_normal": sum(s["label"] == "good" for s in items), "n_anomalous": sum(s["label"] == "bad" for s in items),
                  "auroc": _auc(items), "ci": cluster_bootstrap(groups, at_c, n_boot, seed) if ci_ok else None}
    out["per_condition"] = per

    if image_threshold is not None:
        stab = {}
        for (lab, sid), v in groups.items():
            dec = [s["score"] >= image_threshold for s in v]
            truth = lab == "bad"
            sc = np.array([s["score"] for s in v], float)
            stab[f"{lab}:{sid}"] = {
                "label": lab, "scene_id": sid, "conditions": [s.get("lighting_condition") for s in v],
                "scores": sc.tolist(), "decisions": ["NOK" if d else "OK" for d in dec],
                "errors": sum(d != truth for d in dec), "consistent": len(set(dec)) == 1,
                "score_range": float(sc.max() - sc.min()),
                "score_cv": float(sc.std() / abs(sc.mean())) if sc.mean() else None,
                "margin_min": float(np.min(np.abs(sc - image_threshold))),
            }
        out["stability"] = stab
        out["consistent_scenes"] = sum(x["consistent"] for x in stab.values())
        out["scenes_with_errors"] = sum(x["errors"] > 0 for x in stab.values())
        out["decision_flip_scenes"] = sum(not x["consistent"] for x in stab.values())
    return out
