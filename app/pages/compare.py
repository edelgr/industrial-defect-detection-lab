"""Compare Methods — several compatible methods on the same input, side by side."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QCheckBox, QFileDialog, QGridLayout, QGroupBox, QHBoxLayout,
                               QHeaderView, QListWidget, QListWidgetItem, QProgressBar, QPushButton,
                               QScrollArea, QSpinBox, QSplitter, QTableWidget, QTableWidgetItem,
                               QVBoxLayout, QWidget, QLabel)

from app.i18n import tr
from app.widgets.common import PageHeader, colorize, mask_overlay, muted, overlay, synthetic_banner
from app.widgets.image_viewer import TitledImage
from app.widgets.sample_browser import SampleBrowser
from app.workers.task import start
from core import settings
from core.algorithms import registry as areg
from core.datasets import registry as dreg
from core.datasets.io import read_mask, read_rgb8
from core.evaluation import metrics as M
from core.inference.session import LabSession

DEFAULT_SET = ["canny", "otsu", "ssim", "padim", "patchcore", "anomalydino"]


class ComparePage(QWidget):
    def __init__(self, main):
        super().__init__()
        self.main = main
        self.sessions: dict[tuple, LabSession] = {}
        self.rows: list[dict] = []
        self.task = None
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(PageHeader(tr("nav.compare", "Compare Methods"),
                                 tr("cmp.subtitle", "Run several methods on the same image. Raw scores of different methods are NOT on a common scale.")))
        self.banner = synthetic_banner()
        self.banner.hide()
        lay.addWidget(self.banner)
        hs = QSplitter(Qt.Horizontal)
        left = QWidget()
        ll = QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 0, 0)
        self.browser = SampleBrowser(main.catalog, modality_filter="rgb")
        self.browser.indexChanged.connect(self._on_index)
        ll.addWidget(self.browser, 3)
        g = QGroupBox(tr("cmp.methods", "Methods"))
        gl = QVBoxLayout(g)
        self.mlist = QListWidget()
        for mid in areg.image_methods():
            ok, why = areg.availability(mid)
            rec = main.catalog.algorithms.get(mid)
            it = QListWidgetItem(rec["name"] if rec else mid)
            it.setData(Qt.UserRole, mid)
            it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
            it.setCheckState(Qt.Checked if (ok and mid in DEFAULT_SET) else Qt.Unchecked)
            if not ok:
                it.setFlags(it.flags() & ~Qt.ItemIsEnabled)
                it.setToolTip(why)
            self.mlist.addItem(it)
        gl.addWidget(self.mlist)
        row = QHBoxLayout()
        row.addWidget(QLabel(tr("cmp.n_fit", "Fit images")))
        self.n_fit = QSpinBox()
        self.n_fit.setRange(1, 1000)
        self.n_fit.setValue(40)
        row.addWidget(self.n_fit)
        gl.addLayout(row)
        self.calibrate = QCheckBox(tr("cmp.calibrate", "Calibrate each method on normal images (enables a comparable scale)"))
        self.calibrate.setChecked(True)
        gl.addWidget(self.calibrate)
        self.use_verified = QCheckBox(tr("cmp.use_verified", "Use Workshop Verified models (no training)"))
        self.use_verified.setToolTip(tr("cmp.use_verified_tip", "When a Workshop Verified checkpoint exists for a method and this "
                                                                "category, it is loaded with its own fit images and stored "
                                                                "validation thresholds instead of fitting a new model."))
        self.use_verified.setChecked(True)
        gl.addWidget(self.use_verified)
        self.run = QPushButton("▶  " + tr("cmp.run", "Run comparison on selected sample"))
        self.run.setObjectName("Primary")
        gl.addWidget(self.run)
        self.progress = QProgressBar()
        self.progress.setRange(0, 1000)
        self.msg = muted("")
        gl.addWidget(self.progress)
        gl.addWidget(self.msg)
        ex = QHBoxLayout()
        self.png = QPushButton(tr("cmp.export_png", "Export PNG"))
        self.csv = QPushButton(tr("cmp.export_csv", "Export CSV/JSON"))
        ex.addWidget(self.png)
        ex.addWidget(self.csv)
        gl.addLayout(ex)
        ll.addWidget(g, 2)
        left.setMinimumWidth(320)
        hs.addWidget(left)
        hs.setStretchFactor(1, 1)

        right = QSplitter(Qt.Vertical)
        sc = QScrollArea()
        sc.setWidgetResizable(True)
        self.grid_host = QWidget()
        self.grid = QGridLayout(self.grid_host)
        sc.setWidget(self.grid_host)
        right.addWidget(sc)
        self.table = QTableWidget(0, 9)
        self.table.setHorizontalHeaderLabels(["Method", "Model / version", "Raw score", "Calibrated image thr.",
                                              "Score / thr.", "Per-image pixel AUROC", "Latency (ms)", "GPU peak (MB)",
                                              "Training requirement"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self.table.verticalHeader().setVisible(False)
        right.addWidget(self.table)
        right.setSizes([620, 260])
        hs.addWidget(right)
        hs.setSizes([340, 1100])
        lay.addWidget(hs, 1)
        self.run.clicked.connect(self._run)
        self.png.clicked.connect(self._export_png)
        self.csv.clicked.connect(self._export_csv)
        if self.browser.index:
            self._on_index(self.browser.index)

    def _on_index(self, idx):
        a = dreg.get(idx.dataset_id)
        self.banner.setVisible(bool(a and a.synthetic))

    def _selected(self) -> list[str]:
        return [self.mlist.item(i).data(Qt.UserRole) for i in range(self.mlist.count())
                if self.mlist.item(i).checkState() == Qt.Checked]

    def _run(self):
        s, idx = self.browser.current_sample(), self.browser.index
        if s is None or idx is None:
            self.msg.setText(tr("lab.select_sample", "Select a sample first."))
            return
        methods = self._selected()
        n_fit, do_cal, seed = self.n_fit.value(), self.calibrate.isChecked(), settings.get().seed
        sessions = self.sessions
        from core.inference import model_registry as MR
        verified = {}
        if self.use_verified.isChecked():
            for e in MR.verified_presets(idx.dataset_id, idx.category):
                if not e.verified["params"].get("keep_aspect"):
                    verified.setdefault(e.get("method_id"), e)
        would_train = []
        for mid in methods:
            cls = areg.runtime_class(mid)
            if cls and cls.requires_fit and mid not in verified:
                probe = LabSession(mid, seed=seed)
                probe.bind(idx)
                if not cls.cacheable or probe.cached_fit(n_fit) is None:
                    would_train.append(mid)
        if would_train and (settings.get().workshop_mode or MR.for_index(idx)) and not self._confirm_training(would_train):
            return

        def job(progress, cancel):
            img = read_rgb8(s.primary_path)
            gt = read_mask(s.masks["rgb"], img.shape[:2]) if s.masks.get("rgb") else None
            out = []
            for k, mid in enumerate(methods):
                base = k / len(methods)
                def sub(f, m, base=base, mid=mid):
                    progress(base + f / len(methods), f"{mid}: {m}")
                ve = verified.get(mid)
                key = ("verified", ve.key) if ve else (mid, idx.dataset_id, idx.category, n_fit)
                sess = sessions.get(key)
                if sess is None and ve is not None:
                    sess = LabSession(mid, seed=seed)
                    sess.created_by = "compare"
                    sess.bind(idx)
                    sess.load_checkpoint(ve, progress=sub, cancel=cancel)
                    sessions[key] = sess
                if sess is None:
                    sess = LabSession(mid, seed=seed)
                    sess.created_by = "compare"
                    sess.bind(idx)
                    if sess.method.requires_fit:
                        sess.fit(max_images=n_fit, progress=sub, cancel=cancel)
                    sessions[key] = sess
                if do_cal and sess.calibration.image_threshold is None:
                    sess.calibrate(max_images=20, progress=sub, cancel=cancel)
                r = sess.predict_image(img)
                pix_auc = None
                if gt is not None and r.anomaly_map is not None and gt.any() and not gt.all():
                    pix_auc = M.pixel_auroc([gt], [r.anomaly_map])
                out.append({"method_id": mid, "result": r, "cal": sess.calibration, "pixel_auroc": pix_auc,
                            "fit": len(sess.fit_ids)})
            return img, gt, out

        self.run.setEnabled(False)
        self.task = start(job, on_done=self._show, on_error=self._err,
                          on_progress=lambda f, m: (self.progress.setValue(int(1000 * f)), self.msg.setText(m)))

    def _confirm_training(self, mids: list[str]) -> bool:
        """Ask before training; patched in tests."""
        from PySide6.QtWidgets import QMessageBox
        msg = (tr("cmp.train_msg", "These methods have no matching fitted model and would be TRAINED now") + ": "
               + ", ".join(mids) + ".\n\n" + tr("cmp.train_hint", "Tip: enable 'Use Workshop Verified models' or set "
                                                                  "'Fit images' to the value used for the cached model."))
        return QMessageBox.question(self, tr("lab.train_q", "Train a new model?"), msg) == QMessageBox.Yes

    def _err(self, m, tb):
        self.run.setEnabled(True)
        print(tb)
        self.main.show_error(tr("lab.error", "Method error"), m)

    def _show(self, payload):
        self.run.setEnabled(True)
        self.progress.setValue(1000)
        self.msg.setText("✓")
        img, gt, out = payload
        while self.grid.count():
            w = self.grid.takeAt(0).widget()
            if w:
                w.deleteLater()
        cols = 3
        first = TitledImage(tr("cmp.input", "Input") + (" + ground truth" if gt is not None else ""))
        first.setMinimumSize(280, 240)
        first.set(mask_overlay(img, gt) if gt is not None else img)
        self.grid.addWidget(first, 0, 0)
        self.rows = []
        self.table.setRowCount(len(out))
        for i, o in enumerate(out, start=1):
            r, cal = o["result"], o["cal"]
            rec = self.main.catalog.algorithms.get(o["method_id"], {})
            if r.anomaly_map is not None:
                if cal.pixel_threshold:
                    cm = colorize(r.anomaly_map / cal.pixel_threshold, 0.0, 2.0)  # 1.0 == calibrated threshold
                    note = "scale: map / calibrated pixel threshold, [0, 2]"
                else:
                    cm = colorize(r.anomaly_map)
                    note = "scale: per-image min–max (not comparable)"
                vis = overlay(img, cm, 0.55)
            else:
                vis, note = r.visual, ""
            t = TitledImage(f"{rec.get('name', o['method_id'])} — score {r.score:.4g} · {r.latency_ms:.0f} ms\n{note}")
            t.setMinimumSize(280, 240)
            t.set(vis)
            self.grid.addWidget(t, i // cols, i % cols)
            ratio = (r.score / cal.image_threshold) if (cal.image_threshold and r.score is not None) else None
            row = {"method_id": o["method_id"], "method": rec.get("name", o["method_id"]), "model_version": r.model_version,
                   "raw_score": r.score, "calibrated_image_threshold": cal.image_threshold,
                   "score_over_threshold": ratio, "pixel_auroc_this_image": o["pixel_auroc"],
                   "latency_ms": r.latency_ms, "gpu_peak_mb": r.gpu_memory_mb,
                   "training_requirement": rec.get("training_data_requirements", ""), "n_fit_images": o["fit"],
                   "calibration_source": cal.source, "score_semantics": r.score_semantics}
            self.rows.append(row)
            vals = [row["method"], r.model_version, f"{r.score:.5g}", "—" if cal.image_threshold is None else f"{cal.image_threshold:.5g}",
                    "—" if ratio is None else f"{ratio:.3f}", "—" if o["pixel_auroc"] is None else f"{o['pixel_auroc']:.4f}",
                    f"{r.latency_ms:.1f}", "—" if r.gpu_memory_mb is None else f"{r.gpu_memory_mb:.0f}",
                    rec.get("training_data_requirements", "")]
            for j, v in enumerate(vals):
                it = QTableWidgetItem(str(v))
                it.setToolTip(str(v))
                self.table.setItem(i - 1, j, it)
        self.table.resizeColumnsToContents()
        self._sample = self.browser.current_sample()

    def _export_png(self):
        f, _ = QFileDialog.getSaveFileName(self, "PNG", str(Path.home() / "idd_comparison.png"), "PNG (*.png)")
        if f:
            self.grid_host.grab().save(f)

    def _export_csv(self):
        if not self.rows:
            return
        f, _ = QFileDialog.getSaveFileName(self, "CSV", str(Path.home() / "idd_comparison.csv"), "CSV (*.csv)")
        if not f:
            return
        with open(f, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(self.rows[0]))
            w.writeheader()
            w.writerows(self.rows)
        s = getattr(self, "_sample", None)
        meta = {"sample": s.to_dict() if s else None, "results": self.rows,
                "note": "Local educational comparison; raw scores are method-specific and not probabilities."}
        Path(f).with_suffix(".json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
        self.msg.setText(f"Saved {f} and .json")
