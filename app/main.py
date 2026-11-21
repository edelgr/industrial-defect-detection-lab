"""IDD Lab entry point:  python -m app.main"""

from __future__ import annotations

import os
import sys

from core import paths


def _configure_env() -> None:
    # Keep downloaded model weights inside the (git-ignored) project checkpoints folder.
    os.environ.setdefault("TORCH_HOME", str(paths.CHECKPOINTS / "torch"))
    os.environ.setdefault("HF_HOME", str(paths.CHECKPOINTS / "huggingface"))
    os.environ.setdefault("QT_API", "pyside6")


def main(argv: list[str] | None = None) -> int:
    _configure_env()
    paths.ensure_runtime_dirs()
    from PySide6.QtCore import QCoreApplication, Qt
    from PySide6.QtWidgets import QApplication

    from app import i18n
    from app.ui import theme
    from core import settings

    QCoreApplication.setOrganizationName("IDD Lab")
    QCoreApplication.setApplicationName("IDD Lab")
    QApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    app = QApplication(argv if argv is not None else sys.argv)
    s = settings.load()
    i18n.init(s.language)
    theme.apply(app, s.theme)

    try:                                          # small, seeded, generated once (Student Edition fallback data)
        from core.datasets.synthetic import ensure_generated
        ensure_generated(s.data_root_path)
    except Exception as e:  # noqa: BLE001 — never block start-up
        print("synthetic data not generated:", e)

    from app.ui.main_window import MainWindow
    w = MainWindow()
    w.show()
    from app.widgets import first_run
    if first_run.should_show() and not os.environ.get("IDD_SMOKE_TEST"):
        w._first_run = first_run.FirstRunDialog(w)
        w._first_run.show()                       # non-modal; nothing is downloaded
    if os.environ.get("IDD_SMOKE_TEST"):          # used by tests / start-up check
        from PySide6.QtCore import QTimer
        QTimer.singleShot(int(os.environ.get("IDD_SMOKE_TEST_MS", "1500")), app.quit)
    code = app.exec()
    from app.workers import task as _task
    _task.shutdown(10000)                  # no worker threads alive during interpreter finalisation
    return code


if __name__ == "__main__":
    sys.exit(main())
