"""Reference image difference: real pipeline stages, reference/display synchronisation, stale-result
invalidation and withheld labels in the Methods Laboratory."""

import cv2
import numpy as np
import pytest

from core.algorithms import registry
from core.datasets.io import read_rgb8


def _legacy_reference_difference(image, reference, align, blur, thr):
    """Independent re-implementation of the algorithm as it was before the stage export."""
    from core.algorithms.classical.methods import align_to_reference
    g = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    r = cv2.cvtColor(reference, cv2.COLOR_RGB2GRAY)
    if r.shape != g.shape:
        r = cv2.resize(r, (g.shape[1], g.shape[0]), interpolation=cv2.INTER_AREA)
    if align != "none":
        g, _, _ = align_to_reference(g, r, align)
    k = int(blur) | 1
    if k > 1:
        g, r = cv2.GaussianBlur(g, (k, k), 0), cv2.GaussianBlur(r, (k, k), 0)
    amap = cv2.absdiff(g, r).astype(np.float32)
    return amap, float(np.percentile(amap, 99.9)), amap > thr


@pytest.mark.parametrize("align,blur", [("none", 1), ("none", 5), ("euclidean", 5), ("translation", 3)])
def test_stage_export_does_not_change_the_algorithm(synthetic_index, align, blur):
    test = next(s for s in synthetic_index.samples if s.label == "bad")
    ref = synthetic_index.train_normals()[1]
    img, rimg = read_rgb8(test.primary_path), read_rgb8(ref.primary_path)
    res = registry.create("reference_difference", align=align, blur=blur).predict(img, rimg)
    amap, score, mask = _legacy_reference_difference(img, rimg, align, blur, 30.0)
    assert np.array_equal(res.anomaly_map, amap) and res.score == score and np.array_equal(res.mask, mask)
    st = res.extras["stages"]
    assert np.array_equal(st["reference_gray"], cv2.cvtColor(rimg, cv2.COLOR_RGB2GRAY))
    assert np.array_equal(st["test_gray"], cv2.cvtColor(img, cv2.COLOR_RGB2GRAY))
    assert np.array_equal(res.anomaly_map, cv2.absdiff(st["test_pre"], st["reference_pre"]).astype(np.float32))
    if align == "none":
        assert np.array_equal(st["test_aligned"], st["test_gray"]) and st["warp"] is None
    assert st["blur_kernel"] == (blur | 1)


def test_reference_resized_to_test_size(rgb_image):
    ref = cv2.resize(rgb_image, (256, 192))
    res = registry.create("reference_difference", align="none").predict(rgb_image, ref)
    st = res.extras["stages"]
    assert st["reference_resized"] and st["reference_gray"].shape == rgb_image.shape[:2]


def test_display_mask_uses_the_algorithm_mask_at_its_threshold():
    from types import SimpleNamespace
    from app.widgets.reference_compare import display_mask
    amap = np.array([[29, 30, 31]], np.float32)
    r = SimpleNamespace(anomaly_map=amap, mask=amap > 30, threshold=30.0)
    m, op = display_mask(r, 30.0)
    assert op == ">" and m.tolist() == [[False, False, True]]
    m, op = display_mask(r, 29.5)
    assert op == "≥" and m.tolist() == [[False, True, True]]


def test_labels_withheld():
    from app.widgets.reference_compare import labels_withheld
    from core.datasets.base import Sample
    mk = lambda split, label: Sample("x", "D05", "c", split, label, None, {})
    assert labels_withheld(mk("test_private", "unknown")) and labels_withheld(mk("test_private_mixed", "unknown"))
    assert not labels_withheld(mk("test_public", "bad")) and not labels_withheld(None)


# ------------------------------------------------------------------ GUI
pytest.importorskip("pytestqt")


@pytest.fixture
def lab(qtbot):
    from app import i18n
    from core import settings
    settings.load()
    i18n.init("en")
    from app.ui.main_window import MainWindow
    w = MainWindow()
    qtbot.addWidget(w)
    w.navigate("lab", dataset_id="SYN", method_id="reference_difference")
    page = w.page("lab")
    if page.browser.index is None or page.browser.index.dataset_id != "SYN":
        pytest.skip("synthetic demo dataset not installed")
    yield page
    w.close()


def _pick_bad(lab):
    lab.browser.list.setCurrentRow(next(j for j, x in enumerate(lab.browser.samples) if x.label == "bad"))


def _run(lab, qtbot):
    lab._run()
    qtbot.waitUntil(lambda: lab.run_btn.isEnabled(), timeout=30000)


def test_reference_combo_and_display_in_sync_before_any_run(lab):
    idx = lab.browser.index
    assert lab.ref_combo.count() == min(200, len(idx.train_normals()))      # filled before the first run
    assert lab.center_tabs.currentWidget() is lab.refview
    lab.ref_combo.setCurrentIndex(4)
    sid = lab.ref_combo.currentData()
    ref = next(x for x in idx.samples if x.sample_id == sid)
    rv = lab.refview
    assert rv.ref_sample.sample_id == sid == lab.session.reference.sample_id
    assert np.array_equal(rv.ref_image, read_rgb8(ref.primary_path)) and np.array_equal(rv.p_ref._img, rv.ref_image)
    assert sid in rv.ref_info.text() and ref.split in rv.ref_info.text() and ref.category in rv.ref_info.text()


def test_run_keeps_the_user_reference(lab, qtbot):
    """Regression: the first Run used to reset the combo to the first normal image."""
    lab.ref_combo.setCurrentIndex(3)
    sid = lab.ref_combo.currentData()
    lab.session = None                    # e.g. after a category / method change
    _pick_bad(lab)
    _run(lab, qtbot)
    assert lab.result_ref == sid and lab.ref_combo.currentData() == sid and lab.session.reference.sample_id == sid


def test_panels_show_the_real_pipeline(lab, qtbot):
    _pick_bad(lab)
    _run(lab, qtbot)
    r, rv = lab.result, lab.refview
    st = r.extras["stages"]
    assert np.array_equal(st["reference_gray"], cv2.cvtColor(rv.ref_image, cv2.COLOR_RGB2GRAY))
    assert np.array_equal(st["test_gray"], cv2.cvtColor(rv.test_image, cv2.COLOR_RGB2GRAY))
    assert np.array_equal(rv.p_aligned._img[..., 0], st["test_aligned"])
    assert np.array_equal(r.anomaly_map, cv2.absdiff(st["test_pre"], st["reference_pre"]).astype(np.float32))
    assert rv.shown_op == ">" and np.array_equal(rv.shown_mask, r.mask)
    assert all(p._img is not None for p in rv.panels)
    assert len(rv.log) == 1 and rv.log[0]["ref"] == lab.result_ref


def test_changing_reference_discards_dependent_results(lab, qtbot):
    from core.inference.session import Calibration
    _pick_bad(lab)
    _run(lab, qtbot)
    old = lab.result_ref
    lab.session.calibration = Calibration(image_threshold=40.0, pixel_threshold=30.0, source="test")
    lab._cal_ref = old
    lab._eval_ref = old
    lab.metrics.setHtml("<p>old metrics</p>")
    lab._step_ref(+1)
    assert lab.ref_combo.currentData() != old
    assert lab.result is None and lab.result_ref is None
    assert lab.refview.result is None and lab.refview.stale
    assert lab.refview.p_diff._img is None and lab.refview.p_mask._img is None and lab.v_map.view._img is None
    assert lab.session.calibration.image_threshold is None
    assert "old metrics" not in lab.metrics.toPlainText()
    _run(lab, qtbot)                      # a new comparison only when the user asks
    assert lab.result_ref == lab.ref_combo.currentData() != old and len(lab.refview.log) == 2


def test_result_finishing_after_reference_change_is_discarded(lab, monkeypatch):
    import app.pages.lab as lab_mod
    pending = {}

    def fake_start(fn, on_done=None, on_error=None, on_progress=None, **kw):
        pending["out"] = fn(progress=lambda *a: None, cancel=lambda: False, **kw)
        pending["done"] = on_done
    monkeypatch.setattr(lab_mod, "start", fake_start)
    _pick_bad(lab)
    lab._run()
    lab._step_ref(+1)                    # user changes the reference while the job is running
    pending["done"](pending["out"])
    assert lab.result is None and lab.refview.result is None
    assert "discarded" in lab.progress_msg.text()


def test_withheld_label_gives_a_prediction_not_ground_truth(lab, qtbot):
    from dataclasses import replace
    from core.inference.session import Calibration
    _pick_bad(lab)
    lab.sample = replace(lab.sample, split="test_private", label="unknown", defect_type=None)
    lab.refview.set_test(lab.sample, lab.image)
    lab.session.calibration = Calibration(image_threshold=1.0, pixel_threshold=30.0, source="test")
    _run(lab, qtbot)
    txt = lab.decision.text()
    assert txt.startswith("Prediction: NOK") and "not ground truth" in txt and "(anomalous)" not in txt
    assert "test_private" in lab.refview.info.toPlainText()


def test_panel_hidden_for_other_methods(lab):
    lab.method.setCurrentIndex(lab.method.findData("otsu"))
    assert not lab.center_tabs.isTabVisible(1) and lab.ref_combo.isHidden()


def test_spanish_panel(qtbot):
    from app import i18n
    from core import settings
    settings.load()
    i18n.init("es")
    try:
        from app.widgets.reference_compare import ReferenceComparePanel, explanation_md
        p = ReferenceComparePanel()
        qtbot.addWidget(p)
        assert p.p_ref.caption.text() == "Referencia" and p.p_mask.caption.text() == "Detectadas"
        assert p.p_ref.caption.toolTip() == "Imagen de referencia"
        assert "Prueba alineada" in p.p_aligned.caption.toolTip() and "Comparar" in p.compare_btn.text()
        md = explanation_md()
        assert "Limitaciones" in md and "percentil 99,9" in md and "test_private" in md
    finally:
        i18n.init("en")
