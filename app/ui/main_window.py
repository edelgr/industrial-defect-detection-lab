"""Main application window: sidebar navigation + lazily constructed pages."""

from __future__ import annotations

import importlib

from PySide6.QtCore import QSettings, Qt
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (QHBoxLayout, QLabel, QListWidget, QListWidgetItem, QMainWindow,
                               QMessageBox, QStackedWidget, QVBoxLayout, QWidget)

from app.i18n import tr
from core import __version__
from core.research.catalog import get_catalog

PAGES = [
    ("dashboard", "nav.dashboard", "Dashboard", "app.pages.dashboard:DashboardPage", "◉"),
    ("learning", "nav.learning", "Start Learning", "app.pages.learning:LearningPage", "★"),
    ("atlas", "nav.atlas", "Dataset Research Atlas", "app.pages.atlas:AtlasPage", "▦"),
    ("explorer", "nav.explorer", "Dataset Explorer", "app.pages.explorer:ExplorerPage", "▣"),
    ("library", "nav.library", "Methods Library", "app.pages.library:LibraryPage", "☰"),
    ("lab", "nav.lab", "Methods Laboratory", "app.pages.lab:LabPage", "⚗"),
    ("lab3d", "nav.lab3d", "3D & Multimodal Lab", "app.pages.lab3d:Lab3DPage", "◈"),
    ("compare", "nav.compare", "Compare Methods", "app.pages.compare:ComparePage", "⇆"),
    ("experiments", "nav.experiments", "Educational Experiments", "app.pages.experiments:ExperimentsPage", "✦"),
    ("models", "nav.models", "Model Manager", "app.pages.models:ModelsPage", "⬢"),
    ("workshop", "nav.workshop", "Guided Workshop", "app.pages.workshop:WorkshopPage", "✎"),
    ("settings", "nav.settings", "Settings", "app.pages.settings_page:SettingsPage", "⚙"),
]


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"Industrial Defect Detection Lab — IDD Lab {__version__}")
        self.catalog = get_catalog()
        self._pages: dict[str, QWidget] = {}

        root = QWidget()
        lay = QHBoxLayout(root)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)

        side = QWidget()
        side.setObjectName("Sidebar")
        side.setFixedWidth(232)
        sl = QVBoxLayout(side)
        sl.setContentsMargins(6, 14, 6, 10)
        t = QLabel("IDD Lab")
        t.setObjectName("AppTitle")
        st = QLabel(tr("app.subtitle", "Interactive Learning and Benchmarking for Industrial Defect and Anomaly Detection"))
        st.setObjectName("AppSubtitle")
        st.setWordWrap(True)
        for w in (t, st):
            w.setContentsMargins(10, 0, 8, 0)
            sl.addWidget(w)
        sl.addSpacing(14)
        self.nav = QListWidget()
        self.nav.setWordWrap(True)                 # long translated section names wrap instead of scrolling
        self.nav.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        for i, (key, tk, default, _, icon) in enumerate(PAGES):
            it = QListWidgetItem(f"{icon}   {tr(tk, default)}")
            it.setData(Qt.UserRole, key)
            it.setToolTip(f"Ctrl+{(i + 1) % 10}")
            self.nav.addItem(it)
        sl.addWidget(self.nav, 1)
        foot = QLabel("CIARP 2026 Workshop")
        foot.setObjectName("AppSubtitle")
        foot.setContentsMargins(10, 0, 0, 0)
        sl.addWidget(foot)

        self.stack = QStackedWidget()
        content = QWidget()
        cl = QVBoxLayout(content)
        cl.setContentsMargins(16, 12, 16, 8)
        cl.addWidget(self.stack)
        lay.addWidget(side)
        lay.addWidget(content, 1)
        self.setCentralWidget(root)
        self.nav.currentRowChanged.connect(self._on_nav)

        for i in range(min(10, len(PAGES))):          # Ctrl+1 … Ctrl+9, Ctrl+0 for the 10th page
            a = QAction(self)
            a.setShortcut(QKeySequence(f"Ctrl+{(i + 1) % 10}"))
            a.triggered.connect(lambda _=False, r=i: self.nav.setCurrentRow(r))
            self.addAction(a)

        self._build_status()
        self._restore()
        if self.catalog.problems:
            self.statusBar().showMessage(f"⚠ {len(self.catalog.problems)} research-catalog validation problems "
                                         f"(see Settings → Diagnostics)", 15000)

    # ---------------------------------------------------------------- status
    def _build_status(self):
        sb = self.statusBar()
        self.gpu_chip = QLabel(tr("status.gpu_checking", "GPU: checking…"))
        self.gpu_chip.setObjectName("StatusChip")
        sb.addPermanentWidget(self.gpu_chip)
        from PySide6.QtCore import QTimer
        QTimer.singleShot(400, self._probe_gpu)

    def _probe_gpu(self):
        from app.workers.task import start

        def probe(progress, cancel):
            from core.system.diagnostics import gpu_info
            return gpu_info()

        def done(info):
            if info.get("torch_cuda_available"):
                self.gpu_chip.setText(f"CUDA ✓ {info['device_name']} · {info['vram_total_gb']} GB")
            else:
                self.gpu_chip.setText(tr("status.cpu_only", "CPU mode (CUDA not available)"))
        start(probe, on_done=done, on_error=lambda m, tb: self.gpu_chip.setText("GPU: " + m[:60]))

    # ---------------------------------------------------------------- pages
    def page(self, key: str) -> QWidget:
        if key not in self._pages:
            spec = next(p for p in PAGES if p[0] == key)
            mod, cls = spec[3].split(":")
            try:
                w = getattr(importlib.import_module(mod), cls)(self)
            except Exception as e:
                import traceback
                traceback.print_exc()
                w = QLabel(f"Page failed to load: {type(e).__name__}: {e}")
                w.setWordWrap(True)
            self._pages[key] = w
            self.stack.addWidget(w)
        return self._pages[key]

    def _on_nav(self, row: int):
        key = PAGES[row][0]
        self.stack.setCurrentWidget(self.page(key))
        QSettings().setValue("ui/last_page_key", key)

    def navigate(self, key: str, **kwargs):
        """Switch page and optionally pass a preset (used by Guided Workshop / Atlas links)."""
        row = next(i for i, p in enumerate(PAGES) if p[0] == key)
        self.nav.setCurrentRow(row)
        w = self.page(key)
        if kwargs and hasattr(w, "apply_preset"):
            w.apply_preset(**kwargs)

    # ---------------------------------------------------------------- prefs
    def _restore(self):
        s = QSettings()
        geo = s.value("ui/geometry")
        if geo is not None:
            self.restoreGeometry(geo)
        else:
            self.resize(1500, 920)
        key = str(s.value("ui/last_page_key", "dashboard"))
        rows = [i for i, p in enumerate(PAGES) if p[0] == key]
        self.nav.setCurrentRow(rows[0] if rows else 0)

    def closeEvent(self, e):
        QSettings().setValue("ui/geometry", self.saveGeometry())
        for w in self._pages.values():
            if hasattr(w, "shutdown"):
                try:
                    w.shutdown()
                except Exception:
                    pass
        from app.workers import task as _task
        _task.shutdown(10000)              # cancel + wait for background work before the window goes away
        super().closeEvent(e)

    def show_error(self, title: str, message: str, hint: str = ""):
        QMessageBox.warning(self, title, message + (f"\n\n{tr('err.suggestion', 'Suggestion')}: {hint}" if hint else ""))
