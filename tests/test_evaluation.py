"""Metrics, leakage-safe session protocol, experiment persistence."""

import numpy as np
import pytest

from core.evaluation import metrics as M
from core.experiments.store import ExperimentStore
from core.inference.session import LabSession


def test_auroc_ap_known_values():
    y = [0, 0, 1, 1]
    assert M.image_auroc(y, [0.1, 0.2, 0.8, 0.9]) == 1.0
    assert M.image_auroc(y, [0.9, 0.8, 0.2, 0.1]) == 0.0
    assert M.image_auroc([1, 1], [0.1, 0.2]) is None       # undefined with one class
    assert M.image_ap(y, [0.1, 0.2, 0.8, 0.9]) == 1.0


def test_pixel_auroc_and_dice():
    gt = np.zeros((10, 10), bool)
    gt[2:5, 2:5] = True
    amap = gt.astype(np.float32)
    assert M.pixel_auroc([gt], [amap]) == 1.0
    d, j = M.dice_iou(gt, gt)
    assert d == 1.0 and j == 1.0
    assert M.dice_iou(np.zeros_like(gt), np.zeros_like(gt)) == (None, None)


def test_decision_metrics_far_frr():
    dm = M.decision_metrics([0, 0, 1, 1], [0.1, 0.6, 0.7, 0.2], threshold=0.5, threshold_source="fixed")
    assert (dm.tp, dm.fp, dm.tn, dm.fn) == (1, 1, 1, 1)
    assert dm.false_acceptance_rate == 0.5 and dm.false_rejection_rate == 0.5
    assert dm.threshold_from_test is False


def test_session_never_leaks_test_or_fit_data(synthetic_index):
    s = LabSession("otsu")
    s.bind(synthetic_index)
    s.fit(max_images=100)
    cal, src = s.calibration_set()
    assert all(c.split == "train" and c.label == "good" for c in cal)
    assert not (set(c.sample_id for c in cal) & set(s.fit_ids))


def test_session_fit_uses_only_train_normals(synthetic_index):
    from core.algorithms import registry
    ok, why = registry.availability("padim")
    if not ok:
        pytest.skip(why)
    s = LabSession("padim")
    s.bind(synthetic_index)
    info = s.fit(max_images=6)
    train_ids = {x.sample_id for x in synthetic_index.train_normals()}
    assert set(s.fit_ids) <= train_ids and info["n_fit"] == len(s.fit_ids)
    cal = s.calibrate(max_images=4)
    assert cal.image_threshold is not None and "not used for fitting" in cal.source


def test_evaluate_reports_category_and_refuses_private(synthetic_index, tmp_path):
    s = LabSession("ssim")
    s.bind(synthetic_index)
    s.fit()
    s.calibrate(max_images=3)
    store = ExperimentStore(tmp_path / "e.sqlite")
    res = s.evaluate(store=store)
    assert res["result_category"] == "synthetic_demo"          # synthetic data is never "educational"
    assert 0.0 <= res["image_auroc"] <= 1.0
    assert "Not an official benchmark result" in res["protocol"]
    assert res["decision"]["threshold_from_test"] is False
    run = store.runs()[0]
    assert run["category"] == "synthetic_demo" and len(store.predictions(run["run_id"])) == res["n_test"]
    with pytest.raises(ValueError):
        s.evaluate(split="test_private")


def test_store_rejects_unknown_category_and_uncited_author_results(tmp_path):
    st = ExperimentStore(tmp_path / "e.sqlite")
    with pytest.raises(ValueError):
        st.create_run(category="benchmark", method_id="x")
    with pytest.raises(ValueError):
        st.create_run(category="author_reported", method_id="patchcore")
    rid = st.create_run(category="author_reported", method_id="patchcore", publication_id="roth2022patchcore")
    assert st.runs("author_reported")[0]["run_id"] == rid
    assert st.runs("local_educational") == []
