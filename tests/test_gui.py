"""GUI start-up, every page constructs, reference links render and open, offline operation."""

import socket

import pytest

pytest.importorskip("pytestqt")

from PySide6.QtWidgets import QPushButton  # noqa: E402


@pytest.fixture
def window(qtbot):
    from app import i18n
    from core import settings
    settings.load()
    i18n.init("en")
    from app.ui.main_window import MainWindow
    w = MainWindow()
    qtbot.addWidget(w)
    yield w
    w.close()


def test_main_window_starts(window):
    assert "IDD Lab" in window.windowTitle()
    assert window.nav.count() == 12      # 9 sections + Educational Experiments, Start Learning, Model Manager


def test_every_page_constructs(window):
    from app.ui.main_window import PAGES
    from PySide6.QtWidgets import QLabel
    for key, *_ in PAGES:
        page = window.page(key)
        assert page is not None
        assert not (isinstance(page, QLabel) and "failed to load" in page.text()), f"{key}: {page.text()}"


def test_spanish_interface(qtbot, monkeypatch):
    from app import i18n
    from core import settings
    s = settings.load()
    monkeypatch.setattr(s, "language", "es")
    i18n.init("es")
    from app.ui.main_window import PAGES, MainWindow
    w = MainWindow()
    qtbot.addWidget(w)
    atlas_row = next(i for i, p in enumerate(PAGES) if p[0] == "atlas")
    assert "Atlas de investigación" in w.nav.item(atlas_row).text()
    for key, *_ in PAGES:
        w.page(key)
    from core.research import content
    assert content.workshop()["activities"][0]["title"].startswith("Introducción")
    assert content.algorithm_steps("patchcore")["steps"][0]["title"] == "Imagen original"
    w.close()
    i18n.init("en")


def test_reference_panel_links(qtbot, window, monkeypatch):
    opened = []
    import app.widgets.references as refs
    monkeypatch.setattr(refs, "open_url", lambda u: opened.append(u) or True)
    rec = window.catalog.datasets["D05"]
    panel = refs.ReferencesPanel(window.catalog, rec, "dataset")
    qtbot.addWidget(panel)
    buttons = [b for b in panel.findChildren(QPushButton)
               if b.objectName() == "LinkButton" and b.text()[:1] in ("↗", "⊘")]
    enabled = [b for b in buttons if b.isEnabled()]
    disabled = [b for b in buttons if not b.isEnabled()]
    assert enabled and all(b.toolTip().startswith("http") for b in enabled)
    # MVTec AD 2 has no official code repository -> a disabled, explained action (no fake link)
    assert any("Repository" in b.text() for b in disabled)
    paper_btn = next(b for b in enabled if "Original Paper" in b.text())
    paper_btn.click()
    assert opened[-1] == rec["paper_url"]


def test_copy_bibtex(qtbot, window, monkeypatch):
    # The OS clipboard can be locked by other processes; verify what the button sends to it.
    import app.widgets.references as refs
    copied = []
    monkeypatch.setattr(refs, "copy_text", lambda t: copied.append(t))
    rec = window.catalog.algorithms["patchcore"]
    panel = refs.ReferencesPanel(window.catalog, rec, "algorithm")
    qtbot.addWidget(panel)
    btn = next(b for b in panel.findChildren(QPushButton) if b.text().startswith("Copy BibTeX Citation"))
    btn.click()
    assert "Roth_2022_CVPR" in copied[-1] and copied[-1].count("@") == len(rec["publication_ids"])


def test_offline_atlas_and_library(qtbot, monkeypatch):
    """With all network connections blocked, the Atlas, Library and classical methods still work."""
    def no_net(*a, **k):
        raise OSError("network disabled for test")
    monkeypatch.setattr(socket.socket, "connect", no_net)
    from app import i18n
    i18n.init("en")
    from app.ui.main_window import MainWindow
    w = MainWindow()
    qtbot.addWidget(w)
    atlas = w.page("atlas")
    atlas.open_detail("D06")
    lib = w.page("library")
    lib.show("patchcore")
    import numpy as np
    from core.algorithms import registry
    r = registry.create("canny").predict(np.zeros((32, 32, 3), np.uint8))
    assert r.anomaly_map.shape == (32, 32)
    w.close()


def test_compare_methods_runs(qtbot, window):
    from PySide6.QtCore import Qt
    cmp = window.page("compare")
    if cmp.browser.index is None:
        pytest.skip("no dataset installed")
    cmp.browser.select_dataset("SYN")
    for i in range(cmp.mlist.count()):
        it = cmp.mlist.item(i)
        it.setCheckState(Qt.Checked if it.data(Qt.UserRole) in ("canny", "otsu") else Qt.Unchecked)
    cmp.browser.list.setCurrentRow(0)
    cmp.calibrate.setChecked(True)
    cmp._run()
    qtbot.waitUntil(lambda: len(cmp.rows) == 2, timeout=60000)
    assert {r["method_id"] for r in cmp.rows} == {"canny", "otsu"}
    assert all(r["calibrated_image_threshold"] is not None for r in cmp.rows)


def test_lab_runs_classical_method_in_background(qtbot, window):
    lab = window.page("lab")
    if lab.browser.index is None:
        pytest.skip("no dataset installed")
    lab.method.setCurrentIndex(lab.method.findData("canny"))
    lab.browser.list.setCurrentRow(0)
    assert lab.image is not None
    lab._run()
    qtbot.waitUntil(lambda: lab.result is not None, timeout=20000)
    assert lab.result.anomaly_map is not None


# ------------------------------------------------------------ MVTec AD 2 (only if installed)
def _select_d05(browser, qtbot, cat):
    from core.datasets import registry as dreg
    if cat not in dreg.installed_categories("D05"):
        pytest.skip(f"MVTec AD 2 {cat} not installed")
    browser.select_dataset("D05")
    browser.cat.setCurrentIndex(browser.cat.findData(cat))
    qtbot.waitUntil(lambda: browser.index is not None and browser.index.category == cat, timeout=30000)


def _row_of(browser, pred):
    for i in range(browser.list.count()):
        if pred(browser.list.item(i).data(0x0100)):      # Qt.UserRole
            return i
    raise AssertionError("no matching sample")


@pytest.mark.parametrize("cat", ["sheet_metal", "fruit_jelly"])
def test_explorer_shows_mvtec_ad2(qtbot, window, cat):
    exp = window.page("explorer")
    _select_d05(exp.browser, qtbot, cat)
    assert {"train", "validation", "test_public", "test_private", "test_private_mixed"} <= set(exp.browser.index.splits)
    exp.browser.list.setCurrentRow(_row_of(exp.browser, lambda s: s.label == "bad"))
    assert exp.rgb is not None and exp.mask is not None
    assert exp.mask.shape == exp.rgb.shape[:2] and exp.mask.any()
    assert "lighting_condition" in exp.meta.toHtml()
    exp.browser.list.setCurrentRow(_row_of(exp.browser, lambda s: s.split == "test_private"))
    assert exp.mask is None and "withheld" in exp.meta.toHtml()


@pytest.mark.real_checkpoints
def test_lab_and_compare_reload_cached_mvtec_ad2_models(qtbot, window):
    from PySide6.QtCore import Qt
    from tests.conftest import cuda_available
    if not cuda_available():
        pytest.skip("CUDA not available")
    lab = window.page("lab")
    _select_d05(lab.browser, qtbot, "sheet_metal")
    lab.method.setCurrentIndex(lab.method.findData("patchcore"))
    lab.n_fit.setValue(1000)
    if "cached" not in lab.avail.text():
        pytest.skip("no cached D05 PatchCore model (run scripts.run_experiments --dataset D05)")
    lab._fit()
    qtbot.waitUntil(lambda: lab.session.method.fitted, timeout=120000)
    assert lab.session.method.fit_info.get("loaded_from_cache")
    lab.browser.list.setCurrentRow(_row_of(lab.browser, lambda s: s.label == "bad"))
    lab._run()
    qtbot.waitUntil(lambda: lab.result is not None, timeout=60000)
    assert lab.result.anomaly_map.shape == lab.image.shape[:2]

    cmp = window.page("compare")
    _select_d05(cmp.browser, qtbot, "sheet_metal")
    for i in range(cmp.mlist.count()):
        it = cmp.mlist.item(i)
        it.setCheckState(Qt.Checked if it.data(Qt.UserRole) in ("patchcore", "padim") else Qt.Unchecked)
    cmp.n_fit.setValue(1000)
    cmp.calibrate.setChecked(True)
    cmp.browser.list.setCurrentRow(_row_of(cmp.browser, lambda s: s.label == "bad"))
    cmp._run()
    qtbot.waitUntil(lambda: len(cmp.rows) == 2, timeout=180000)
    assert {r["method_id"] for r in cmp.rows} == {"patchcore", "padim"}
    assert all(r["n_fit_images"] == 137 for r in cmp.rows)
    assert all(s.method.fit_info.get("loaded_from_cache") for k, s in cmp.sessions.items() if k[1] == "D05")
