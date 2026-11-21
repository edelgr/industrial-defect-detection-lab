"""IDD-LAB-WORKSHOP-READY-1: Model Registry, verified presets, scene-aware statistics, educational
experiments, workshop activities, offline profile and the complementary IEEE reference."""

import json

import numpy as np
import pytest

from core.evaluation import scenes
from core.inference import model_registry as MR
from core.inference.session import LabSession
from tests.conftest import cuda_available


# ------------------------------------------------------------------ scene-aware statistics
def _samples(n_good=4, n_bad=15, conds=("regular", "overexposed", "shift_1"), seed=0, corr=1.0):
    rng = np.random.default_rng(seed)
    out = []
    for lab, n, mu in (("good", n_good, 0.0), ("bad", n_bad, 1.0)):
        for i in range(n):
            base = rng.normal(mu, 1.0)
            for c in conds:
                out.append({"sample_id": f"{lab}/{i:03d}_{c}", "label": lab, "scene_id": f"{i:03d}",
                            "lighting_condition": c, "score": corr * base + (1 - corr) * rng.normal(mu, 1.0)})
    return out


def test_rank_auroc_equals_sklearn():
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(1)
    for _ in range(20):
        y = rng.integers(0, 2, 40)
        s = np.round(rng.normal(size=40) + y, 1)          # with ties
        if 0 < y.sum() < y.size:
            assert scenes.auroc_ranks(y, s) == pytest.approx(roc_auc_score(y, s), abs=1e-12)
    assert scenes.auroc_ranks([1, 1], [0.1, 0.2]) is None


def test_grouping_and_sufficiency():
    g = scenes.group(_samples())
    assert len(g) == 19 and all(len(v) == 3 for v in g.values())
    assert scenes.sufficiency(g)["level"] == "very_limited"
    assert scenes.sufficiency(scenes.group(_samples(n_good=2)))["level"] == "insufficient"
    assert scenes.sufficiency(scenes.group(_samples(n_good=12, n_bad=12)))["level"] == "adequate"


def test_cluster_bootstrap_wider_than_naive_for_correlated_variants():
    s = _samples(n_good=8, n_bad=8, conds=("a", "b", "c", "d", "e", "f"), corr=1.0, seed=3)
    a = scenes.analyse(s, image_threshold=0.5, n_boot=600)
    lo_c, hi_c, _ = a["pooled_ci_cluster"]
    lo_n, hi_n = a["pooled_ci_naive"]
    assert (hi_c - lo_c) > (hi_n - lo_n)          # i.i.d. resampling understates uncertainty
    assert all(x["consistent"] for x in a["stability"].values())   # identical variants never flip


def test_insufficient_data_reports_no_interval():
    a = scenes.analyse(_samples(n_good=2), image_threshold=0.5, n_boot=100)
    assert "pooled_ci_cluster" not in a and all(p["ci"] is None for p in a["per_condition"].values())


# ------------------------------------------------------------------ registry on synthetic data
@pytest.fixture
def tmp_fitted(tmp_path, monkeypatch):
    monkeypatch.setattr(MR, "FITTED", tmp_path / "fitted")
    monkeypatch.setattr(MR, "VERIFIED_FILE", tmp_path / "workshop_verified.json")
    return tmp_path / "fitted"


def test_manifest_registry_and_exact_reload(synthetic_index, tmp_fitted, deep):
    s = LabSession("padim")
    s.bind(synthetic_index)
    s.fit(max_images=5)
    man = json.loads(MR.manifest_path(s.checkpoint).read_text(encoding="utf-8"))
    assert man["fit_ids"] == s.fit_ids and man["n_fit"] == 5 and man["provenance"] == "recorded at fit"
    (e,) = MR.for_index(synthetic_index, "padim")
    assert e.tier == "quick_experiment"
    ok, why = MR.compatibility(e, synthetic_index)
    assert ok, why
    img = np.full((64, 64, 3), 120, np.uint8)
    r1 = s.predict_image(img)
    s2 = LabSession("padim")
    s2.bind(synthetic_index)
    info = s2.load_checkpoint(e)                  # no 'Fit images' value involved
    assert info["cache"] == s.checkpoint.name and s2.fit_ids == s.fit_ids
    assert np.allclose(s2.predict_image(img).anomaly_map, r1.anomaly_map, atol=1e-5)


def test_incompatible_and_tampered_checkpoints(synthetic_index, tmp_fitted, deep):
    s = LabSession("padim")
    s.bind(synthetic_index)
    s.fit(max_images=4)
    (e,) = MR.for_index(synthetic_index, "padim")
    from core.datasets.base import DatasetIndex
    other = DatasetIndex("SYN", "other_category", synthetic_index.root, synthetic_index.samples)
    ok, why = MR.compatibility(e, other)
    assert not ok and any("other_category" in w for w in why)
    e.manifest["fit_ids"] = e.manifest["fit_ids"][:-1]          # manifest no longer matches the file key
    ok, why = MR.compatibility(e, synthetic_index)
    assert not ok and any("key differs" in w for w in why)
    s3 = LabSession("padim")
    s3.bind(synthetic_index)
    with pytest.raises(RuntimeError, match="Incompatible"):
        s3.load_checkpoint(e)


def test_reconstructs_manifest_of_pre_registry_checkpoint(synthetic_index, tmp_fitted, monkeypatch, deep):
    from core.datasets import registry as dreg
    s = LabSession("padim")
    s.bind(synthetic_index)
    s.fit(max_images=3)
    MR.manifest_path(s.checkpoint).unlink()
    monkeypatch.setattr(dreg, "installed_categories", lambda did: [synthetic_index.category])
    monkeypatch.setattr(dreg, "load_index", lambda did, cat, refresh=False: synthetic_index)
    man = MR.reconstruct_manifest(s.checkpoint)
    assert man and man["fit_ids"] == s.fit_ids and man["provenance"].startswith("reconstructed")


# ------------------------------------------------------------------ verified presets (real data)
def _verified_or_skip(preset_id):
    e = MR.find_preset(preset_id)
    if e is None:
        pytest.skip(f"preset {preset_id} not verified / installed")
    return e


@pytest.mark.real_checkpoints
def test_verified_file_contract():
    v = MR.load_verified()
    if not v:
        pytest.skip("configs/workshop_verified.json not generated")
    for m in v.values():
        assert m["reference_check"]["max_abs_diff"] is not None and m["reference_check"]["max_abs_diff"] <= 1e-6
        assert m["metrics"]["result_category"] == "local_educational"
        assert not any(s["sample_id"].startswith("test_private") for s in m["samples"])
        assert all(i.startswith(("validation/", "train/")) for i in m["calibration"]["sample_ids"])
        assert len(m["sha256"]) == 64


@pytest.mark.real_checkpoints
def test_readiness_verified_presets():
    from core import profile
    from core.system.readiness import verified_checks
    if not MR.load_verified() or profile.resolve("auto") != "instructor":
        pytest.skip("instructor installation only (Workshop Verified checkpoints not present)")
    (c,) = verified_checks(deep=False)
    assert c.ok, c.detail


@pytest.mark.real_checkpoints
def test_padim_fruit_jelly_good_auroc_zero_f1_is_real():
    e = MR.find_stored("d05-fruit_jelly-padim-256")        # stored results: also available in the Student Edition
    if e is None:
        pytest.skip("no stored results")
    m = e.verified["metrics"]
    assert m["image_auroc"] > 0.7 and m["decision"]["f1"] == 0.0
    assert e.verified["calibration"]["image_threshold"] > max(s["score"] for s in e.verified["samples"])


# ------------------------------------------------------------------ GUI (real data, CUDA)
pytest.importorskip("pytestqt")


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


def _need_gpu_presets():
    if not cuda_available():
        pytest.skip("CUDA not available")
    _verified_or_skip("d05-sheet_metal-patchcore-256")


@pytest.mark.real_checkpoints
def test_lab_loads_verified_checkpoint_without_touching_fit_images(qtbot, window, monkeypatch):
    _need_gpu_presets()
    from tests.test_gui import _select_d05
    lab = window.page("lab")
    _select_d05(lab.browser, qtbot, "sheet_metal")
    lab.method.setCurrentIndex(lab.method.findData("patchcore"))
    assert lab.ckpt.count() >= 2 and lab.ckpt.itemText(0).startswith("[Workshop Verified]")
    assert lab.n_fit.value() == 60
    asked = []
    monkeypatch.setattr(lab, "_ask_train", lambda text, can_load: asked.append(text) or "cancel")
    lab._fit()                                       # 60 images: no cache → must ask, not train
    assert asked and not lab.session.method.fitted
    lab.ckpt.setCurrentIndex(0)
    lab._load_ckpt()
    qtbot.waitUntil(lambda: lab.session.method.fitted and lab.load_btn.isEnabled(), timeout=120000)
    assert lab.n_fit.value() == 60 and len(lab.session.fit_ids) == 137
    assert lab.session.calibration.stored and lab.session.calibration.image_threshold is not None
    assert "Workshop Verified" in lab.avail.text()


@pytest.mark.real_checkpoints
def test_lab_warns_about_incompatible_checkpoint(qtbot, window):
    _need_gpu_presets()
    from tests.test_gui import _select_d05
    lab = window.page("lab")
    _select_d05(lab.browser, qtbot, "sheet_metal")
    lab.method.setCurrentIndex(lab.method.findData("patchcore"))
    e = lab._entries[0]
    e.manifest = {**e.manifest, "dataset_root": e.manifest["dataset_root"] + "_moved"}
    lab._on_ckpt()
    assert not lab.load_btn.isEnabled() and "Incompatible" in lab.ckpt_info.text()


@pytest.mark.real_checkpoints
def test_compare_uses_verified_models_with_default_fit_images(qtbot, window, monkeypatch):
    _need_gpu_presets()
    from PySide6.QtCore import Qt
    from tests.test_gui import _row_of, _select_d05
    cmp = window.page("compare")
    _select_d05(cmp.browser, qtbot, "fruit_jelly")
    for i in range(cmp.mlist.count()):
        it = cmp.mlist.item(i)
        it.setCheckState(Qt.Checked if it.data(Qt.UserRole) in ("patchcore", "padim") else Qt.Unchecked)
    assert cmp.n_fit.value() == 40 and cmp.use_verified.isChecked()
    monkeypatch.setattr(cmp, "_confirm_training", lambda mids: pytest.fail(f"would train {mids}"))
    cmp.browser.list.setCurrentRow(_row_of(cmp.browser, lambda s: s.label == "bad"))
    cmp._run()
    qtbot.waitUntil(lambda: len(cmp.rows) == 2, timeout=180000)
    assert all(r["n_fit_images"] == 263 for r in cmp.rows)


@pytest.mark.real_checkpoints
def test_educational_experiments_page(qtbot, window):
    _need_gpu_presets()
    window.navigate("experiments", tab="resolution")
    xp = window.page("experiments")
    assert xp.res.ready and xp.ill.ready and xp.met.ready
    xp.res._run()
    qtbot.waitUntil(lambda: xp.res.live.rowCount() > 0 and xp.res.run.isEnabled(), timeout=180000)
    assert xp.res.views["heat2"].view.scene().items()
    xp.ill.scenes.setCurrentIndex(1)
    xp.ill._maps()
    qtbot.waitUntil(lambda: xp.ill.maps_btn.isEnabled() and "Same colour" in xp.ill.msg.text(), timeout=180000)
    assert "cluster" in xp.ill.report.toPlainText().lower()
    window.navigate("experiments", tab="metrics", preset_id="d05-fruit_jelly-padim-256")
    met = xp.met
    assert met.v["preset_id"] == "d05-fruit_jelly-padim-256" and "Valid protocol" in met.banner.text()
    assert met.tbl.item(7, 1).text() == "0.000"                   # F1 at the valid threshold
    met.slider.setValue(max(0, met.slider0 - 300))
    assert "RETROSPECTIVE" in met.banner.text()
    met.reset.click()
    assert "Valid protocol" in met.banner.text()


@pytest.mark.real_checkpoints
def test_workshop_activities_show_real_results(qtbot, window):
    if not MR.load_verified():
        pytest.skip("no verified presets")
    ws = window.page("workshop")
    ids = [a["id"] for a in ws.acts]
    assert ids[-3:] == ["A", "B", "C"]
    for i in range(len(ids) - 3, len(ids)):
        ws.list.setCurrentRow(i)
        text = ws.text.toPlainText()
        assert "Real results" in text and "Limitations" in text and "Results unavailable" not in text


def test_complementary_ieee_reference_verified_and_cited(window):
    cat = window.catalog
    (c,) = cat.datasets["D05"]["complementary_references"]
    assert c["url"] == "https://ieeexplore.ieee.org/document/11701006"
    assert c["verification_status"] == "verified" and c["doi"] == "10.1109/ACCESS.2026.3735579"
    p = cat.publications[c["publication_id"]]
    assert p["venue"].startswith("IEEE Access 14") and p["authors"][0] == "Vinh Phong Vo"
    assert "Vo_2026_BeyondBenchmarkScores" in (cat.publications[c["publication_id"]]["bibtex"])
    assert "418" in c["note"]                     # automated full-text access may be blocked
    from core import paths
    assert "10.1109/ACCESS.2026.3735579" in (paths.PROJECT_ROOT / "CITATIONS.bib").read_text(encoding="utf-8")


def test_workshop_profile_env_is_session_only(tmp_path, monkeypatch):
    from core import settings
    f = tmp_path / "s.json"
    f.write_text(json.dumps({"workshop_mode": False}), encoding="utf-8")
    monkeypatch.setenv("IDD_WORKSHOP", "1")
    s = settings.load(f)
    assert s.workshop_mode
    settings.save(s, f)
    assert json.loads(f.read_text(encoding="utf-8"))["workshop_mode"] is False
    monkeypatch.delenv("IDD_WORKSHOP")
    settings.load()


@pytest.mark.real_checkpoints
def test_educational_experiments_translated_to_spanish(qtbot):
    if not MR.load_verified():
        pytest.skip("no verified presets")
    from app import i18n
    from core import settings
    settings.load()
    i18n.init("es")
    try:
        from app.ui.main_window import MainWindow
        w = MainWindow()
        qtbot.addWidget(w)
        w.navigate("experiments", tab="metrics", preset_id="d05-fruit_jelly-padim-256")
        met = w.page("experiments").met
        assert met.tbl.item(0, 0).text() == "Umbral" and met.tbl.item(5, 0).text() == "Precisión"
        assert met.fig.axes[0].get_xlabel() == "puntuación de anomalía de imagen"
        assert "imágenes de entrenamiento" in met.model.currentText()
        assert "Protocolo válido" in met.banner.text()
        ill = w.page("experiments").ill
        assert "Análisis por escenas" in ill.report.toPlainText()
        assert ill.scenes.itemText(0).startswith(("ANÓMALA", "NORMAL"))
        from app.pages.experiments import results_markdown
        assert "AUROC de imagen" in results_markdown("resolution")
        w.close()
    finally:
        i18n.init("en")
