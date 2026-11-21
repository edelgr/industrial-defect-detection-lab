"""Capture README screenshots of the running GUI into docs/screenshots/.

Only views WITHOUT third-party dataset imagery are captured for the repository (Atlas, Library,
Workshop, Settings) plus Explorer/Laboratory on the SYNTHETIC demo set — dataset licences (e.g.
CC BY-NC-SA, unstated MulSen-AD licence) do not allow committing their images.

  python -m scripts.screenshots
"""

from __future__ import annotations

import os
import sys

from core import paths


def main():
    os.environ.setdefault("TORCH_HOME", str(paths.CHECKPOINTS / "torch"))
    os.environ.setdefault("HF_HOME", str(paths.CHECKPOINTS / "huggingface"))
    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QApplication
    from app import i18n
    from app.ui import theme
    from core import settings
    app = QApplication(sys.argv)
    s = settings.load()
    i18n.init("en")
    theme.apply(app, s.theme)
    from app.ui.main_window import MainWindow
    w = MainWindow()
    w.resize(1600, 960)
    w.show()
    out = paths.DOCS / "screenshots"
    out.mkdir(parents=True, exist_ok=True)
    shots = []

    def shot(name):
        app.processEvents()
        w.grab().save(str(out / f"{name}.png"))
        shots.append(name)

    def run():
        w.navigate("dashboard"); shot("01_dashboard")
        w.navigate("atlas"); shot("02_atlas_gallery")
        a = w.page("atlas"); a.tabs.setCurrentIndex(1); shot("03_atlas_timeline")
        a.tabs.setCurrentIndex(2); shot("04_atlas_compare")
        w.navigate("atlas", dataset_id="D05"); shot("05_atlas_mvtec_ad2")
        w.navigate("library", method_id="patchcore"); shot("06_library_patchcore")
        lib = w.page("library"); lib.center.setCurrentIndex(1); lib.steps.go(3); shot("07_how_patchcore_works")
        w.navigate("explorer", dataset_id="SYN")
        def first_bad(b):
            return next(i for i, s in enumerate(b.samples) if s.label == "bad" and s.defect_type == "scratch")
        ex = w.page("explorer"); ex.browser.list.setCurrentRow(first_bad(ex.browser)); shot("08_explorer_synthetic")
        w.navigate("lab", dataset_id="SYN", method_id="morphology")
        lab = w.page("lab"); lab.browser.list.setCurrentRow(first_bad(lab.browser)); lab._run()
        QTimer.singleShot(2500, finish)

    def finish():
        shot("09_lab_synthetic_morphology")
        w.navigate("workshop"); ws = w.page("workshop"); ws.list.setCurrentRow(5); shot("10_workshop")
        print("saved:", ", ".join(shots))
        app.quit()

    QTimer.singleShot(1500, run)
    app.exec()


if __name__ == "__main__":
    main()
