"""Model Manager — what each algorithm needs, what is installed, where it comes from and under which
licence. Downloads happen only after an explicit, confirmed action; nothing is trained here."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QHBoxLayout, QHeaderView, QLabel, QMessageBox, QProgressBar, QPushButton,
                               QSplitter, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget)

from app.i18n import tr
from app.widgets.common import PageHeader, markdown_browser, muted
from app.widgets.references import ReferencesPanel
from app.workers.task import start
from core import model_manager as mm

CLASS_LABEL = {"included": "Included", "downloadable": "Official pretrained (download)",
               "student_trained": "Fitted on this computer", "instructor_only": "Instructor-only (stored results)",
               "unavailable": "Unavailable"}


class ModelsPage(QWidget):
    def __init__(self, main):
        super().__init__()
        self.main = main
        self.task = None
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(PageHeader(tr("nav.models", "Model Manager"),
                                 tr("mm.subtitle", "What each algorithm needs, what is installed on this computer, "
                                                   "its source and its licence. Nothing is downloaded or trained "
                                                   "without your explicit action.")))
        sp = QSplitter(Qt.Vertical)
        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels([tr("mm.h_alg", "Algorithm"), tr("mm.h_avail", "Availability"),
                                              tr("mm.h_status", "Status on this computer"), tr("mm.h_cpu", "CPU"),
                                              tr("mm.h_gpu", "GPU memory (measured)")])
        h = self.table.horizontalHeader()
        h.setSectionResizeMode(QHeaderView.ResizeToContents)
        for col in (1, 2, 3, 4):                    # long texts share the width and wrap: all five columns
            h.setSectionResizeMode(col, QHeaderView.Stretch)   # stay visible without horizontal scrolling
        self.table.setWordWrap(True)
        self.table.verticalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        sp.addWidget(self.table)
        bottom = QWidget()
        bl = QHBoxLayout(bottom)
        left = QWidget()
        ll = QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 0, 0)
        self.details = markdown_browser("")
        ll.addWidget(self.details, 1)
        row = QHBoxLayout()
        self.prepare_btn = QPushButton(tr("mm.prepare", "Download pretrained weights (online)…"))
        self.prepare_btn.clicked.connect(self._prepare)
        row.addWidget(self.prepare_btn)
        self.refresh_btn = QPushButton(tr("mm.refresh", "Refresh"))
        self.refresh_btn.clicked.connect(self.refresh)
        row.addWidget(self.refresh_btn)
        row.addStretch(1)
        ll.addLayout(row)
        self.progress = QProgressBar()
        self.progress.setRange(0, 1000)
        self.msg = muted("")
        ll.addWidget(self.progress)
        ll.addWidget(self.msg)
        bl.addWidget(left, 3)
        self.refs_holder = QVBoxLayout()
        holder = QWidget()
        holder.setLayout(self.refs_holder)
        from PySide6.QtWidgets import QFrame, QScrollArea
        sa = QScrollArea()                         # long link labels scroll instead of widening the page
        sa.setWidgetResizable(True)
        sa.setFrameShape(QFrame.NoFrame)
        sa.setWidget(holder)
        sa.setMinimumWidth(260)
        bl.addWidget(sa, 2)
        sp.addWidget(bottom)
        sp.setSizes([420, 420])
        lay.addWidget(sp, 1)
        self.table.currentCellChanged.connect(lambda r, *_: self._show(r))
        self.refresh()

    def refresh(self):
        self.rows = mm.statuses()
        self.table.setRowCount(len(self.rows))
        for i, s in enumerate(self.rows):
            gpu = "—" if s.gpu_inference_mb is None else f"{s.gpu_inference_mb:.0f} MB " + tr("mm.peak", "(inference peak, instructor GPU)")
            vals = [s.name, ", ".join(tr(f"mm.c_{c}", CLASS_LABEL[c]) for c in s.classes), self._status(s), s.cpu or "—", gpu]
            for j, v in enumerate(vals):
                self.table.setItem(i, j, QTableWidgetItem(v))
        if self.rows:
            self.table.selectRow(max(0, self.table.currentRow()))
            self._show(max(0, self.table.currentRow()))

    @staticmethod
    def _status(s) -> str:
        key, text = s.summary_key()
        return tr(key, text) + (" — " + s.reason if not s.runnable else "")

    def _show(self, row: int):
        if row < 0 or row >= len(self.rows):
            return
        s = self.rows[row]
        rec = self.main.catalog.algorithms.get(s.method_id, {})
        w = s.weights or {}
        fitted = ", ".join(f"{k.replace('_', ' ')}: {v}" for k, v in s.fitted.items()) or "none"
        L = [f"### {s.name}",
             f"- **{tr('mm.implementation', 'Implementation used')}:** {s.implementation}",
             f"- **{tr('mm.training', 'Training requirement')}:** {rec.get('training_data_requirements', '—')}",
             f"- **{tr('mm.modalities', 'Modalities')}:** {', '.join(rec.get('modalities') or []) or '—'}",
             f"- **{tr('mm.status', 'Status')}:** {self._status(s)}"]
        if w:
            L += [f"- **{tr('mm.weights', 'Pretrained resource')}:** [{w['resource']}]({w['url']}) · {w['size']}",
                  f"- **{tr('mm.weights_lic', 'Weights licence')}:** {w['licence']}",
                  f"- **{tr('mm.cached', 'Cached on this computer')}:** {'yes' if s.weights_cached else 'no'}"]
        if s.needs_fit or s.instructor_presets:
            L += [f"- **{tr('mm.fitted', 'Fitted models on this computer')}:** {fitted}",
                  f"- **{tr('mm.instructor', 'Instructor Workshop Verified presets')}:** {s.instructor_presets} "
                  + tr("mm.instructor_note", "(their stored, verified results are shown in Educational Experiments; "
                                             "the checkpoints themselves are not distributed)")]
        L += [f"- **{tr('mm.redistribution', 'Licence / redistribution')}:** {s.redistribution}"]
        if not s.runnable:
            L += ["", tr("mm.unavailable_note", "This method is described in the Methods Library (concept, steps, "
                                                "papers) but cannot be executed in IDD Lab on this computer: ") + s.reason]
        self.details.setMarkdown("\n".join(L))
        self.prepare_btn.setEnabled(bool(w) and s.runnable and not s.weights_cached)
        self.prepare_btn.setToolTip(tr("mm.prepare_tip", "Downloads only the pretrained backbone from its original "
                                                          "source into checkpoints/. Does not fit or train on any data."))
        while self.refs_holder.count():
            it = self.refs_holder.takeAt(0)
            if it.widget():
                it.widget().deleteLater()
        if rec:
            self.refs_holder.addWidget(ReferencesPanel(self.main.catalog, rec, "algorithm"))

    def _confirm(self, s) -> bool:
        w = s.weights
        return QMessageBox.question(self, tr("mm.confirm_title", "Download pretrained weights?"),
                                    tr("mm.confirm", "{name}: download {res} ({size}) from {url}.\n\nLicence: {lic}\n\n"
                                                     "Requires an Internet connection. Nothing is trained.").format(
                                        name=s.name, res=w["resource"], size=w["size"], url=w["url"], lic=w["licence"])
                                    ) == QMessageBox.Yes

    def _prepare(self):
        s = self.rows[self.table.currentRow()]
        if not s.weights or not self._confirm(s):
            return
        self.prepare_btn.setEnabled(False)
        self.task = start(mm.prepare, method_id=s.method_id,
                          on_done=lambda m: (self.msg.setText("✓ " + m), self.refresh()),
                          on_error=lambda m, tb: (self.msg.setText("✗ " + m), self.prepare_btn.setEnabled(True)),
                          on_progress=lambda f, m: (self.progress.setValue(int(1000 * f)), self.msg.setText(m)))
