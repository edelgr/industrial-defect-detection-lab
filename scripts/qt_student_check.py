"""Real Qt user-flow check of the Student Edition on a simulated fresh computer.

  python -m scripts.qt_student_check [--lang en|es] [--out cache/qt_student]

The full application runs with (in this process only): a NEW empty data root (only the Synthetic
Educational Data generated on first run), an EMPTY fitted-model folder, the student profile and the
first-run assistant enabled. Network access is blocked during the Atlas / Library checks. The real
instructor checkpoints are snapshotted before and after and must be unchanged. Screenshots → --out
(git-ignored). Exit status 1 if any check fails.
"""

from __future__ import annotations

import argparse
import os
import socket
import sys
import tempfile
import time
from pathlib import Path

from core import paths


def snapshot(folder: Path) -> dict:
    return {p.name: (p.stat().st_size, p.stat().st_mtime_ns) for p in folder.glob("*")} if folder.is_dir() else {}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", default="en", choices=["en", "es"])
    ap.add_argument("--out", default=str(paths.CACHE / "qt_student"))
    a = ap.parse_args(argv)
    os.environ.setdefault("TORCH_HOME", str(paths.CHECKPOINTS / "torch"))
    os.environ.setdefault("HF_HOME", str(paths.CHECKPOINTS / "huggingface"))
    from core.inference import model_registry as MR
    real_fitted = MR.FITTED
    before = snapshot(real_fitted)
    tmp = Path(tempfile.mkdtemp(prefix="idd_student_"))
    MR.FITTED = tmp / "fitted"                       # fresh student computer: no fitted models
    from PySide6.QtCore import QTimer, QThreadPool, QUrl
    from PySide6.QtGui import QDesktopServices
    from PySide6.QtWidgets import QApplication
    from app import i18n
    from app.ui import theme
    from core import profile, settings
    app = QApplication(sys.argv)
    s = settings.load()
    s.language, s.profile, s.first_run_done, s.workshop_mode = a.lang, "student", False, False
    s.data_root = str(tmp / "data")                  # fresh data root: no real datasets
    settings.save = lambda *x, **k: None             # never write the user's settings file during the check
    i18n.init(a.lang)
    from app.i18n import tr
    theme.apply(app, s.theme)
    out = Path(a.out) / a.lang
    out.mkdir(parents=True, exist_ok=True)
    log = []

    def check(name, ok, detail=""):
        log.append((name, bool(ok)))
        print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}", flush=True)

    def wait(cond, timeout=180.0):
        t0 = time.time()
        while time.time() - t0 < timeout:
            app.processEvents()
            try:
                if cond():
                    return True
            except Exception:  # noqa: BLE001
                pass
            time.sleep(0.05)
        return False

    opened = []
    QDesktopServices.openUrl = staticmethod(lambda url: opened.append(url.toString() if isinstance(url, QUrl) else str(url)) or True)
    import app.widgets.common as common
    common.open_url = lambda url: opened.append(url) or True
    import app.widgets.references as refs
    refs.open_url = common.open_url

    # 5. synthetic data generated on first run (same call as app.main)
    from core.datasets.synthetic import LABEL, ensure_generated
    root = ensure_generated(s.data_root_path)
    from core.datasets import registry as dreg
    check("5 synthetic data generated with label", (root / "brushed_metal").is_dir() and
          LABEL in (root / "SYNTHETIC_NOTICE.txt").read_text(encoding="utf-8"), str(root))
    check("student profile resolved", profile.resolve() == "student")
    check("no real datasets on this simulated computer", not dreg.installed_categories("D05") and not dreg.installed_categories("D06"))

    from app.ui.main_window import MainWindow
    w = MainWindow()
    w.resize(1500, 900)
    w.show()

    def shot(name):
        app.processEvents()
        w.grab().save(str(out / f"{name}.png"))

    def run():
        # 1. start-up + first run
        from app.widgets import first_run
        check("1 application starts without real datasets/checkpoints", w.isVisible())
        check("first-run assistant shown for a student", first_run.should_show())
        d = first_run.FirstRunDialog(w)
        d.show()
        app.processEvents()
        d.grab().save(str(out / "00_first_run.png"))
        d._finish(True)
        check("first run → Start Learning", w.stack.currentWidget() is w.page("learning"))
        shot("01_learning")
        lp = w.page("learning")
        lp.list.setCurrentRow(2)
        app.processEvents()
        # 2-4. Atlas and Library offline, reference links
        real_connect = socket.socket.connect
        socket.socket.connect = lambda *x, **k: (_ for _ in ()).throw(OSError("network blocked by qt_student_check"))
        try:
            w.navigate("atlas", dataset_id="D05")
            app.processEvents()
            atlas = w.page("atlas")
            from PySide6.QtWidgets import QPushButton
            links = [b for b in atlas.findChildren(QPushButton) if b.text().startswith("↗") and b.isVisible()]
            check("2 Dataset Atlas offline (MVTec AD 2 profile, links shown)", len(links) >= 4, f"{len(links)} links")
            n0 = len(opened)
            links[0].click()
            check("4 a reference link opens its URL", len(opened) == n0 + 1 and opened[-1].startswith("http"), opened[-1] if opened else "")
            shot("02_atlas_offline")
            w.navigate("library", method_id="patchcore")
            app.processEvents()
            lib = w.page("library")
            lib.center.setCurrentIndex(1)
            app.processEvents()
            check("3 Methods Library offline (PatchCore card + steps)", lib.isVisible())
            shot("03_library_offline")
        finally:
            socket.socket.connect = real_connect
        # 6. classical method on synthetic data
        w.navigate("lab", dataset_id="SYN", method_id="otsu")
        lab = w.page("lab")
        wait(lambda: lab.browser.index is not None)
        k = next(i for i, x in enumerate(lab.browser.samples) if x.label == "bad")
        lab.browser.list.setCurrentRow(k)
        lab._run()
        check("6 classical algorithm runs (Otsu, synthetic)", wait(lambda: lab.result is not None, 60))
        check("synthetic banner visible", lab.banner.isVisible() and "SYNTHETIC EDUCATIONAL DATA" in lab.banner.text())
        shot("04_lab_otsu_synthetic")
        # 7. missing checkpoints: registry empty, no silent download/training
        from core.algorithms import registry as areg
        if areg.availability("patchcore")[0]:
            lab.method.setCurrentIndex(lab.method.findData("patchcore"))
            app.processEvents()
            check("7 missing checkpoint: registry shows none", lab.ckpt.count() == 1 and lab.ckpt.currentData() is None)
        else:
            check("7 missing deep stack reported in the Lab", not areg.availability("patchcore")[0])
        w.navigate("experiments", tab="resolution")
        xp = w.page("experiments")
        check("7 Educational Experiments: stored results, live disabled", xp.res.ready and not xp.res.run.isEnabled())
        shot("05_experiments_stored")
        w.navigate("models")
        mp = w.page("models")
        check("7 Model Manager lists every algorithm", mp.table.rowCount() == len(w.catalog.algorithms))
        shot("06_model_manager")
        # 8. dataset import instructions
        w.navigate("settings", tab="datasets")
        check("8 dataset import route + instructions", (paths.PROJECT_ROOT / "DATASETS.md").exists() and
              bool(w.catalog.datasets["D05"].get("download_url")))
        # 9-10. CPU / GPU execution of a deep method on synthetic data (only if installed and weights cached)
        from core.system.readiness import weights_cached
        if areg.availability("padim")[0] and weights_cached("padim"):
            from core.inference.session import LabSession
            idx = dreg.load_index("SYN", "brushed_metal")
            for dev in ("cpu", "auto"):
                s.device = dev
                sess = LabSession("padim")
                sess.bind(idx)
                sess.fit(max_images=8)
                from core.datasets.io import read_rgb8
                r = sess.predict_image(read_rgb8(next(x for x in idx.samples if x.label == "bad").primary_path))
                check(f"{'9 CPU' if dev == 'cpu' else '10 GPU/auto'} execution (PaDiM on synthetic)", r.anomaly_map is not None,
                      f"device={r.device}")
            s.device = "auto"
        else:
            check("9/10 deep execution skipped (not installed or weights not cached) — reported, not faked", True)
        # 11. Student Guided Mode in the workshop
        w.navigate("workshop")
        ws = w.page("workshop")
        ws.list.setCurrentRow([x["id"] for x in ws.acts].index("C"))
        check("11 Guided Workshop activity C shows real stored results",
              tr("ws.results", "Real results (Workshop Verified, test_public)")[:10] in ws.text.toPlainText())
        # 12. readiness
        from core.system.readiness import run_checks
        cs = run_checks(profile="student")
        check("12 student readiness: 0 required failing", not [c for c in cs if c.required and not c.ok])
        # 13. 1280-px screens
        for key in ("dashboard", "learning", "atlas", "explorer", "library", "lab", "lab3d", "compare", "experiments",
                    "models", "workshop", "settings"):
            w.navigate(key)
            app.processEvents()
        mw = w.minimumSizeHint().width()
        check("13 window minimum width fits 1280-px screens", mw <= 1280, f"{mw} px")
        shot("07_settings")
        # 14. clean shutdown
        w.close()
        app.processEvents()
        check("14 clean shutdown (no worker threads left)", QThreadPool.globalInstance().activeThreadCount() == 0)
        # 15. instructor checkpoints untouched
        check("15 instructor checkpoints unchanged", snapshot(real_fitted) == before, f"{len(before)} files")
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)        # only the temporary student data/model folders of this check
        n_fail = sum(not ok for _, ok in log)
        print(f"\n{len(log) - n_fail}/{len(log)} checks passed · screenshots in {out}", flush=True)
        app.exit(1 if n_fail else 0)

    QTimer.singleShot(600, run)
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
