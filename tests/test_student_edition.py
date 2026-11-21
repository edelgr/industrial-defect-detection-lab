"""Student Edition (IDD-LAB-STUDENT-RELEASE-1): profiles, readiness, Model Manager, stored-results mode
without checkpoints or datasets, Learning Center, first-run assistant, synthetic data, offline weights,
release staging and documentation checks."""

import json

import numpy as np
import pytest

from core.inference import model_registry as MR


@pytest.fixture
def no_checkpoints(tmp_path, monkeypatch):
    """Simulate a fresh student computer: empty model folder, no real datasets installed."""
    from core.datasets import registry as dreg
    monkeypatch.setattr(MR, "FITTED", tmp_path / "fitted_empty")
    real = dreg.installed_categories
    monkeypatch.setattr(dreg, "installed_categories", lambda did: [] if did != "SYN" else real(did))
    return tmp_path


# ------------------------------------------------------------------ profile and readiness
def test_profile_resolution(no_checkpoints):
    from core import profile
    assert profile.resolve("student") == "student" and profile.resolve("instructor") == "instructor"
    assert profile.resolve("auto") == "student"          # no verified checkpoint on this "computer"


def test_student_readiness_does_not_require_instructor_assets(no_checkpoints):
    from core.system.readiness import run_checks
    checks = run_checks(profile="student")
    names = " ".join(c.name for c in checks)
    assert "Workshop Verified presets" not in names       # instructor-only check is not part of the student report
    assert all(c.ok for c in checks if c.required), [c.name for c in checks if c.required and not c.ok]
    assert any(not c.required for c in checks)             # optional paths reported as information


def test_readiness_cli_student_exit_code(no_checkpoints):
    from scripts.readiness import main
    assert main(["--profile", "student"]) == 0


# ------------------------------------------------------------------ Model Manager
def test_model_manager_classes(no_checkpoints):
    from core import model_manager as mm
    st = {s.method_id: s for s in mm.statuses()}
    assert st["otsu"].classes == ["included"] and st["otsu"].runnable
    assert st["yolo"].classes == ["unavailable"] and not st["yolo"].runnable
    assert "instructor_only" in st["patchcore"].classes and not st["patchcore"].fitted   # stored results only
    assert "student_trained" not in st["patchcore"].classes
    assert "not stated" in mm.WEIGHTS["efficientad"]["licence"]                           # verified fact, not assumed
    assert "Not distributable" in mm.FITTED_REDISTRIBUTION


def test_offline_weights_mode_never_downloads(monkeypatch):
    from core.algorithms import registry
    ok, _ = registry.availability("padim")
    if not ok:
        pytest.skip("anomalib not installed")
    import core.system.readiness as R
    monkeypatch.setenv("IDD_OFFLINE_WEIGHTS", "1")
    monkeypatch.setattr(R, "weights_cached", lambda mid: False)
    ok, why = registry.availability("padim")
    assert not ok and "downloads are disabled" in why
    assert registry.availability("canny")[0]


# ------------------------------------------------------------------ synthetic data
def test_synthetic_kinds_label_and_reproducibility(tmp_path):
    from core.datasets.io import read_rgb8
    from core.datasets.synthetic import DEFECT_KINDS, LABEL, SyntheticAdapter, generate
    a, b = tmp_path / "a", tmp_path / "b"
    generate(a, n_train=3, n_test_good=2, n_test_per_defect=2, size=96, seed=3, kinds=("scratch", "dent", "stain"))
    generate(b, n_train=3, n_test_good=2, n_test_per_defect=2, size=96, seed=3)
    for p in a.rglob("*.png"):                            # adding kinds never changes the earlier images
        assert np.array_equal(read_rgb8(p), read_rgb8(b / p.relative_to(a)))
    idx = SyntheticAdapter().index(b, "brushed_metal")
    assert {s.defect_type for s in idx.samples if s.defect_type} == set(DEFECT_KINDS)
    assert all(s.masks.get("rgb") for s in idx.samples if s.label == "bad")
    man = json.loads((b / "manifest.json").read_text(encoding="utf-8"))
    assert man["label"] == LABEL == "Synthetic Educational Data — Not a Real Industrial Benchmark"


# ------------------------------------------------------------------ learning content
def test_learning_center_references_resolve():
    from app.pages.learning import EXERCISE_FILES, exercise_path, learning
    from app.ui.main_window import PAGES
    from core.research.catalog import load_catalog
    cat = load_catalog()
    topics = learning()["topics"]
    assert len(topics) == 9
    pages = {p[0] for p in PAGES}
    for t in topics:
        assert all(p in cat.publications for p in t["papers"]), t["id"]
        assert all(a in cat.algorithms for a in t["algorithms"]), t["id"]
        assert all(d in cat.datasets for d in t["datasets"]), t["id"]
        assert all(a["page"] in pages for a in t["demo"]), t["id"]
        assert exercise_path(t["exercise"]["id"]).exists()
    assert len(EXERCISE_FILES) == 8 and all(exercise_path(k).exists() for k in EXERCISE_FILES)


def test_exercise_sheets_have_required_sections():
    from app.pages.learning import EXERCISE_FILES, exercise_path
    for ex in EXERCISE_FILES:
        txt = exercise_path(ex).read_text(encoding="utf-8")
        for field in ("Duration", "Required hardware", "Required dataset", "Required checkpoint", "Learning objectives",
                      "Instructions", "Expected qualitative observations", "Reflection questions",
                      "Scientific references", "Alternative path"):
            assert field in txt, (ex, field)


def test_docs_check_passes():
    from scripts.check_docs import check
    assert check() == []


def test_docs_scope_without_git_equals_tracked_documentation():
    """A ZIP download has no .git: the fallback must check exactly the tracked documentation."""
    import subprocess

    from core import paths
    from scripts import check_docs as cd
    if not (paths.PROJECT_ROOT / ".git").exists():
        pytest.skip("not a Git checkout")
    tracked = set(subprocess.run(["git", "ls-files", "--", "*.md"], cwd=paths.PROJECT_ROOT, capture_output=True,
                                 text=True, check=True).stdout.split())
    nogit = {p.relative_to(paths.PROJECT_ROOT).as_posix() for p in cd.md_files(use_git=False)}
    untracked_docs = {p for p in nogit - tracked if cd._in_doc_dirs(p)}     # new docs not committed yet
    assert nogit - untracked_docs == tracked


def test_docs_scope_ignores_personal_drafts_but_not_project_docs(tmp_path):
    import shutil
    import subprocess

    from app.pages.learning import EXERCISE_FILES
    from scripts import check_docs as cd
    (tmp_path / "docs" / "exercises").mkdir(parents=True)
    for name in EXERCISE_FILES.values():
        (tmp_path / "docs" / "exercises" / name).write_text("x", encoding="utf-8")
    (tmp_path / "README.md").write_text("[a](missing_from_readme.md)", encoding="utf-8")
    (tmp_path / "docs" / "new_doc.md").write_text("[b](missing_from_docs.md)", encoding="utf-8")
    (tmp_path / "MY_DRAFT.md").write_text("[c](missing_from_draft.md)", encoding="utf-8")
    def found():
        return sorted(p.split(" -> ")[1] for p in cd.check(tmp_path) if "broken link" in p)
    # no Git (ZIP download): project docs are checked, a personal draft at the root is not
    names = {p.name for p in cd.md_files(tmp_path, use_git=False) if p.parent.name != "exercises"}
    assert names == {"README.md", "new_doc.md"}
    assert found() == ["missing_from_docs.md", "missing_from_readme.md"]
    if shutil.which("git") is None:
        return
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "add", "README.md"], cwd=tmp_path, check=True)
    (tmp_path / "notes").mkdir()
    (tmp_path / "notes" / "tracked.md").write_text("[d](missing_from_tracked.md)", encoding="utf-8")
    subprocess.run(["git", "add", "notes/tracked.md"], cwd=tmp_path, check=True)
    # Git checkout: tracked files anywhere + untracked files in docs/ are checked; the untracked draft is not
    assert found() == ["missing_from_docs.md", "missing_from_readme.md", "missing_from_tracked.md"]
    assert cd.unchecked_md(tmp_path) == ["MY_DRAFT.md"]


# ------------------------------------------------------------------ release staging
def test_release_staging_policy(tmp_path):
    from scripts.build_release import _excluded, _redact, _redactions
    for bad in ("data/mvtec_ad2/x.png", "checkpoints/fitted/a.pt", "experiments/experiments.sqlite", "configs/user_settings.json",
                "cache/qt_flow/en/01.png", "x/model.safetensors", "checkpoints/quarantine/QUARANTINE.json"):
        assert _excluded(bad), bad
    for ok in ("data/README.md", "checkpoints/README.md", "app/main.py", "configs/workshop_verified.json"):
        assert not _excluded(ok), ok
    from pathlib import Path
    home = str(Path.home())
    data, n = _redact("docs/x.md", f"path {home}\\project".encode(), _redactions())
    assert n >= 1 and home.encode() not in data


# ------------------------------------------------------------------ GUI: student computer without data/checkpoints
pytest.importorskip("pytestqt")


@pytest.fixture
def window(qtbot, no_checkpoints):
    from app import i18n
    from core import settings
    settings.load()
    i18n.init("en")
    from app.ui.main_window import MainWindow
    w = MainWindow()
    qtbot.addWidget(w)
    yield w
    w.close()


def test_experiments_show_stored_results_without_checkpoints(qtbot, window):
    if not MR.load_verified():
        pytest.skip("no stored results")
    window.navigate("experiments", tab="resolution")
    xp = window.page("experiments")
    assert xp.res.ready and not xp.res.live_ok and not xp.res.run.isEnabled()
    assert xp.res.stored.item(0, 1).text() == "0.741"                       # stored, verified image AUROC
    texts = " ".join(w.text() for w in xp.res.findChildren(type(xp.res.msg)))
    assert "STORED RESULTS" in texts and "LIVE" not in texts.split("STORED")[0]
    assert xp.ill.ready and not xp.ill.maps_btn.isEnabled()
    assert "not installed" in xp.ill.maptiles[0].caption.text() or xp.ill.reasons
    window.navigate("experiments", tab="metrics", preset_id="d05-fruit_jelly-padim-256")
    assert xp.met.tbl.item(7, 1).text() == "0.000"


def test_learning_and_model_manager_pages(qtbot, window):
    window.navigate("learning", topic="T3")
    lp = window.page("learning")
    assert lp.list.currentRow() == 2 and lp.refs.count() > 5
    window.navigate("models")
    mp = window.page("models")
    assert mp.table.rowCount() == len(window.catalog.algorithms)
    row = next(i for i, s in enumerate(mp.rows) if s.method_id == "otsu")
    mp.table.selectRow(row)
    mp._show(row)
    assert not mp.prepare_btn.isEnabled()                                    # nothing to download for Otsu


def test_first_run_assistant(qtbot, window, monkeypatch):
    from core import settings
    from app.widgets import first_run
    s = settings.get()
    monkeypatch.setattr(s, "first_run_done", False)
    monkeypatch.setattr(s, "profile", "student")
    assert first_run.should_show()
    saved = []
    monkeypatch.setattr(settings, "save", lambda obj=None, path=None: saved.append(obj))
    d = first_run.FirstRunDialog(window)
    txt = d.text.toPlainText()
    assert "Synthetic Educational Data" in txt and "Recommended first exercise" in txt
    d._finish(True)
    assert saved and s.first_run_done and window.stack.currentWidget() is window.page("learning")


def test_lab_asks_before_downloading_weights(qtbot, window, monkeypatch):
    from core.algorithms import registry
    if not registry.availability("patchcore")[0]:
        pytest.skip("anomalib not installed")
    import core.system.readiness as R
    monkeypatch.setattr(R, "weights_cached", lambda mid: False)
    lab = window.page("lab")
    lab.browser.select_dataset("SYN")
    lab.method.setCurrentIndex(lab.method.findData("patchcore"))
    asked = []
    monkeypatch.setattr(lab, "_confirm_download", lambda w: asked.append(w["resource"]) or False)
    lab._fit()
    assert asked == ["timm/wide_resnet50_2.racm_in1k"] and not lab.session.method.fitted
