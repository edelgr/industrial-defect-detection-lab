"""Educational Experiments UI: decision helpers, experiment B (Basic / Advanced) scene selection and
synchronisation between selector, condition cards, charts and tables, layout guards, Spanish UI.

Only the stored verified results (configs/workshop_verified.json) are needed; no checkpoint is loaded and
no inference runs (the fitted-model folder is isolated by conftest)."""

import numpy as np
import pytest

from core.evaluation import metrics as M
from core.evaluation import scenes
from core.inference import model_registry as MR

pytestmark = pytest.mark.skipif(not MR.load_verified(), reason="no stored verified results")


def _d05(preset_id="d05-sheet_metal-patchcore-256"):
    return next(v for v in MR.load_verified().values() if v["preset_id"] == preset_id)


# ------------------------------------------------------------------ pure helpers (no Qt widgets)
def test_outcomes_reproduce_stored_decision_counts():
    """The UI decision rule (score >= threshold → NOK) gives exactly the stored TP/FP/TN/FN."""
    from app.widgets.experiment_kit import is_nok, outcome
    for v in MR.load_verified().values():
        if v["dataset_id"] != "D05":
            continue
        thr = v["calibration"]["image_threshold"]
        codes = [outcome(s["label"] == "bad", is_nok(s["score"], thr)) for s in v["samples"]]
        d = v["metrics"]["decision"]
        assert [codes.count(k) for k in ("TP", "FP", "TN", "FN")] == [d["tp"], d["fp"], d["tn"], d["fn"]], v["preset_id"]
        y = np.array([s["label"] == "bad" for s in v["samples"]])
        m = M.decision_metrics(y, np.array([s["score"] for s in v["samples"]]), thr)
        assert (m.tp, m.fp, m.tn, m.fn) == (d["tp"], d["fp"], d["tn"], d["fn"])


def test_threshold_crossings():
    from app.widgets.experiment_kit import threshold_crossings
    assert threshold_crossings([1, 2, 3], 10) == []
    assert threshold_crossings([1, 3], 2) == [(0, 0.5)]
    assert threshold_crossings([1, 2, 1], 2) == [(0, 1.0), (1, 1.0)]       # score == threshold is NOK
    x = threshold_crossings([5, 1, 6], 3)
    assert [i for i, _ in x] == [0, 1] and 0 < x[0][1] < 1 and 1 < x[1][1] < 2
    # one crossing per decision change, for every stored scene
    v = _d05()
    thr = v["calibration"]["image_threshold"]
    for items in scenes.group(v["samples"]).values():
        dec = [s["score"] >= thr for s in items]
        changes = sum(a != b for a, b in zip(dec, dec[1:]))
        assert len(threshold_crossings([s["score"] for s in items], thr)) == changes


def test_card_orientation_and_condition_labels():
    from app.widgets.experiment_kit import ConditionCard, short_condition
    assert not ConditionCard.best_stacked(400, 80, 4.0)     # wide images in a short card: side by side
    assert ConditionCard.best_stacked(200, 300, 4.0)        # tall card: stacked
    assert short_condition("overexposed") == "over-\nexposed" and short_condition("shift_1") == "shift_1"
    assert short_condition("underexposed", two_lines=False) == "underexposed"


# ------------------------------------------------------------------ Qt: experiment B
pytest.importorskip("pytestqt")


@pytest.fixture
def ill(qtbot):
    from app import i18n
    from core import settings
    settings.load()
    i18n.init("en")
    from app.ui.main_window import MainWindow
    w = MainWindow()
    qtbot.addWidget(w)
    w.navigate("experiments", tab="illumination")
    tab = w.page("experiments").ill
    assert tab.ready
    tab.model.setCurrentIndex(tab.model.findData(_d05()["key"]))
    yield tab
    w.close()


def _stab(tab, key):
    return tab.analysis["stability"][f"{key[0]}:{key[1]}"]


def test_basic_scene_selection_syncs_cards_story_and_chart(ill):
    v = _d05()
    thr = v["calibration"]["image_threshold"]
    assert ill.level == "basic" and ill.stack.currentIndex() == 0 and ill.basic_btn.isChecked()
    groups = scenes.group(v["samples"])
    assert ill.scenes.count() == len(groups) == 19
    for row in (0, 5, ill.scenes.count() - 1):
        ill.scenes.setCurrentIndex(row)
        key = ill.scenes.itemData(row)
        items = groups[key]
        assert ill.cur_key == key and len(ill.cur) == 6
        cards = [c for c in ill.grid.cards if not c.isHidden()]
        assert [c.title.text() for c in cards] == [s["lighting_condition"] for s in items]
        expect = [("TP" if s["score"] >= thr else "FN") if key[0] == "bad" else ("FP" if s["score"] >= thr else "TN")
                  for s in items]
        assert [c.code for c in cards] == expect
        assert ill.scene_chart.last["mode"] == "scene" and ill.scene_chart.last["codes"] == expect
        assert f"Scene {key[1]}" in ill.story.toPlainText()
        assert ill.scenes.itemText(row).startswith(("ANOMALOUS", "NORMAL"))
    ill.select_condition(3)
    assert [c._selected for c in ill.grid.cards[:6]] == [False, False, False, True, False, False]
    ill.grid.conditionClicked.emit(1)                       # clicking a card selects the condition
    assert ill.cond_idx == 1 and ill.grid.cards[1]._selected
    ill.scene_chart.conditionClicked.emit(4)                # clicking a chart point too
    assert ill.cond_idx == 4
    ill.next_btn.click()
    assert ill.scenes.currentIndex() == 0                    # wraps around from the last scene


def test_advanced_chart_selection_highlights_scene_errors_and_crossings(ill):
    v = _d05()
    thr = v["calibration"]["image_threshold"]
    ill.set_level("advanced")
    assert ill.stack.currentIndex() == 1 and ill.adv_btn.isChecked()
    flip_key = next(k for k in ill.groups if not _stab(ill, k)["consistent"])
    ill.overview.sceneClicked.emit(flip_key)                 # click on a line of the chart
    assert ill.cur_key == flip_key and ill.scenes.currentData() == flip_key
    last = ill.overview.last
    assert last["mode"] == "overview" and last["highlight"] == flip_key and last["n_lines"] == 19
    items = ill.groups[flip_key]
    errors = [(i, s["score"]) for i, s in enumerate(items) if (s["score"] >= thr) != (flip_key[0] == "bad")]
    assert [(x, y) for x, y, _ in last["errors"]] == errors and len(errors) == _stab(ill, flip_key)["errors"]
    dec = [s["score"] >= thr for s in items]
    assert len(last["crossings"]) == sum(a != b for a, b in zip(dec, dec[1:])) > 0
    # the decision table follows the selection, and selecting a row selects the scene
    sel = ill.matrix.selectionModel().selectedRows()
    assert sel and ill.matrix.payload(sel[0].row()) == flip_key
    other = ill.matrix.payload(0)
    ill.matrix.selectRow(0)
    assert ill.cur_key == other and ill.overview.last["highlight"] == other
    # clicking a point of the selected scene selects that condition everywhere
    ill.overview.conditionClicked.emit(2)
    assert ill.cond_idx == 2 and ill.grid.cards[2]._selected


def test_advanced_filter_tables_report_and_references(ill):
    a = ill.analysis
    ill.set_level("advanced")
    ill.only_flips.setChecked(True)
    assert ill.scenes.count() == ill.matrix.rowCount() == a["decision_flip_scenes"] > 0
    assert ill.overview.last["n_lines"] >= a["decision_flip_scenes"]
    ill.only_flips.setChecked(False)
    assert ill.scenes.count() == ill.matrix.rowCount() == 19
    # AUROC per lighting: one row per condition + pooled + 2 scene-level rows; values = stored by_condition
    by = _d05()["metrics"]["by_condition"]
    assert ill.auroc_tbl.rowCount() == len(by) + 3
    for r in range(len(by)):
        c = ill.auroc_tbl.item(r, 0).text()
        assert ill.auroc_tbl.item(r, 2).text() == f"{by[c]['image_auroc']:.3f}"
    rep = ill.report.toPlainText()
    assert "Scene-aware analysis" in rep and "cluster" in rep and "stored" in rep
    refs = ill.refs.toPlainText()
    assert "MVTec AD 2 Dataset" in refs and "10.1109/ACCESS.2026.3735579" in refs and "Total Recall" in refs


def test_provenance_separates_stored_and_live(ill):
    txt = ill.prov.text()
    assert "STORED RESULTS" in txt
    assert ("LIVE —" in txt) == (not ill.reasons) and ill.maps_btn.isEnabled() == (not ill.reasons)


def test_fruit_jelly_has_four_conditions(ill):
    ill.model.setCurrentIndex(ill.model.findData(_d05("d05-fruit_jelly-padim-256")["key"]))
    assert ill.conds == ["regular", "overexposed", "underexposed", "shift_1"]
    assert sum(not c.isHidden() for c in ill.grid.cards) == 4 and ill.scenes.count() == 20


def test_condition_grid_reflows_without_widening_the_window(ill):
    g = ill.grid
    mins = []
    for width, cols in ((2000, 6), (900, 3), (480, 2)):
        assert g.columns_for(width) == cols
    for width in (1600, 700):
        g.resize(width, 300)
        g._reflow()
        mins.append(g.minimumSizeHint().width() if g.minimumWidth() == 0 else g.minimumWidth())
    assert mins[0] == mins[1] == 2 * g.MIN_CARD_W


def test_enlarge_dialog_synchronises_views(ill, qtbot):
    from app.widgets.experiment_kit import ImageDialog
    img = np.zeros((60, 240, 3), np.uint8)
    d = ImageDialog("t", [("a", img), ("b", img.copy())])
    qtbot.addWidget(d)
    d.views[0].scale(2.0, 2.0)
    d.views[0].viewChanged.emit()
    assert d.views[1].transform().m11() == pytest.approx(d.views[0].transform().m11())


def test_spanish_labels(qtbot):
    from app import i18n
    from core import settings
    settings.load()
    i18n.init("es")
    try:
        from app.ui.main_window import MainWindow
        w = MainWindow()
        qtbot.addWidget(w)
        w.navigate("experiments", tab="illumination")
        ill = w.page("experiments").ill
        assert ill.basic_btn.text() == "Básico" and ill.adv_btn.text() == "Avanzado"
        assert "RESULTADOS ALMACENADOS" in ill.prov.text()
        assert "Escena" in ill.story.toPlainText() and ill.basic_tabs.tabText(0) == "¿Qué ha pasado?"
        ill.set_level("advanced")
        assert ill.adv_tabs.tabText(0) == "Escenas × iluminación"
        assert ill.matrix.horizontalHeaderItem(0).text() == "Escena"
        w.close()
    finally:
        i18n.init("en")


def test_matrix_fits_main_columns_and_cell_click_selects_scene_and_condition(ill):
    ill.set_level("advanced")
    m = ill.matrix
    assert m.columnCount() == 2 + 6 + 2                   # scene, class, 6 conditions, errors, decision change
    row = next(r for r in range(m.rowCount()) if m.payload(r) != ill.cur_key)
    key = m.payload(row)
    m.cellClicked.emit(row, 2 + 4)                        # the shift_2 cell of that scene
    assert ill.cur_key == key and ill.cond_idx == 4 and ill.grid.cards[4]._selected
    assert ill.overview.last["highlight"] == key
    st = ill.analysis["stability"][f"{key[0]}:{key[1]}"]
    detail = ill.matrix_detail.text()
    # every per-scene metric is still available: errors, changes, range and margin, plus the clicked cell
    assert f"{st['score_range']:.2f}" in detail and f"{st['margin_min']:.2f}" in detail and "shift_2" in detail
    assert f"{st['score_range']:.2f}" in m.item(row, 0).toolTip()


def test_switching_models_back_and_forth_keeps_selection_valid(ill):
    fj = _d05("d05-fruit_jelly-padim-256")["key"]
    sm = _d05()["key"]
    ill.set_level("advanced")
    ill.model.setCurrentIndex(ill.model.findData(fj))
    ill.scenes.setCurrentIndex(ill.scenes.count() - 1)    # a scene id that sheet_metal does not have
    ill.select_condition(3)
    ill.model.setCurrentIndex(ill.model.findData(sm))
    assert ill.cur_key in ill.groups and len(ill.cur) == 6 and ill.cond_idx <= 5
    assert ill.thr == _d05()["calibration"]["image_threshold"]


def test_window_fits_1280x800_with_taskbar(qtbot):
    """1280×800 screen − 48 px taskbar − 31 px title bar ≈ 721 px of client area (real display only)."""
    from PySide6.QtGui import QGuiApplication
    if QGuiApplication.platformName() == "offscreen":
        pytest.skip("font metrics of the offscreen platform are not representative")
    from app import i18n
    from core import settings
    settings.load()
    for lang in ("en", "es"):
        i18n.init(lang)
        try:
            from app.ui.main_window import PAGES, MainWindow
            w = MainWindow()
            qtbot.addWidget(w)
            for key, *_ in PAGES:
                w.page(key)
            w.navigate("experiments", tab="illumination")
            w.page("experiments").ill.set_level("advanced")
            hint = w.minimumSizeHint()
            assert hint.height() <= 721 and hint.width() <= 1280, (lang, hint)
            w.close()
        finally:
            i18n.init("en")


def test_confidence_interval_warnings_are_kept(ill):
    """With 4–5 normal scenes the page must say the CIs are exploratory, and must flag the presets where the
    cluster interval is not wider than the naive one (EfficientAD sheet_metal, PaDiM fruit_jelly)."""
    for pid, narrower in (("d05-sheet_metal-patchcore-256", False), ("d05-sheet_metal-efficientad-256", True),
                          ("d05-fruit_jelly-padim-256", True)):
        ill.model.setCurrentIndex(ill.model.findData(_d05(pid)["key"]))
        rep = ill.report.toPlainText()
        assert "Exploratory only" in rep and "not independent" in rep, pid
        assert ("is not wider than the naive one" in rep) == narrower, pid
        assert ("wider than the naive one" in rep and not narrower) or narrower, pid


def test_stored_mode_shows_no_empty_image_panels(ill, monkeypatch):
    """Without installed images (Student Edition) the condition cards are compact and A shows its
    explanation instead of an empty heat-map grid."""
    monkeypatch.setattr(ill, "_thumb", lambda sid: None)
    ill._scene(ill.scenes.currentIndex())
    cards = [c for c in ill.grid.cards if not c.isHidden()]
    assert cards and all(not c.has_images and c.image.isHidden() and c.map.isHidden() and not c.no_images.isHidden()
                         for c in cards)
    assert not ill.has_images
    res = ill.page.res
    if res.ready:
        assert res.right_stack.currentIndex() == (0 if res.live_ok else 1)
        assert res.live.isHidden() == (not res.live_ok)
        if not res.live_ok:
            assert "Live comparison not available" in res.stored_view.toPlainText()


def test_button_and_tab_labels_show_ampersands(qtbot):
    from app.widgets.common import amp
    assert amp("Align & compute") == "Align && compute" and amp("no ampersand") == "no ampersand"
    from app import i18n
    from core import settings
    settings.load()
    i18n.init("en")
    from PySide6.QtWidgets import QPushButton
    from app.ui.main_window import MainWindow
    w = MainWindow()
    qtbot.addWidget(w)
    texts = [b.text() for b in w.page("dashboard").findChildren(QPushButton)]
    assert any("3D && Multimodal Lab" in x for x in texts)          # rendered as "3D & Multimodal Lab"
    atlas = w.page("atlas")
    from PySide6.QtWidgets import QTabWidget
    tabs = atlas.findChild(QTabWidget)
    assert any("&&" in tabs.tabText(i) for i in range(tabs.count()))
    w.close()


def test_atlas_gallery_columns_follow_the_width(qtbot):
    from app import i18n
    from core import settings
    settings.load()
    i18n.init("en")
    from app.ui.main_window import MainWindow
    w = MainWindow()
    qtbot.addWidget(w)
    w.resize(1600, 960)
    w.show()
    w.navigate("atlas")
    a = w.page("atlas")
    vp = a.gallery_scroll.viewport()
    for width, cols in ((1100, 3), (700, 2), (400, 1)):
        vp.resize(width, 500)
        qtbot.waitUntil(lambda c=cols: a._cols == c, timeout=3000)
    assert a.gallery_scroll.horizontalScrollBarPolicy() == a.gallery_scroll.horizontalScrollBarPolicy().ScrollBarAlwaysOff
    w.close()
