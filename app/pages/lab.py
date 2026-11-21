"""Methods Laboratory — run a real method on a real sample, inspect maps, scores and metrics."""

from __future__ import annotations

import json

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QComboBox, QGridLayout, QGroupBox, QHBoxLayout, QLabel, QProgressBar,
                               QPushButton, QScrollArea, QSlider, QSpinBox, QSplitter, QTabWidget,
                               QTextBrowser, QVBoxLayout, QWidget)

from app.i18n import tr
from app.widgets.common import (Badge, PageHeader, colorize, mask_overlay, markdown_browser, muted,
                                overlay, synthetic_banner)
from app.widgets.image_viewer import TitledImage
from app.widgets.param_form import ParamForm
from app.widgets.reference_compare import PANEL_METHODS, ReferenceComparePanel, explanation_md, labels_withheld
from app.widgets.references import ReferencesPanel
from app.widgets.sample_browser import SampleBrowser
from app.workers.task import start
from core import settings
from core.algorithms import registry as areg
from core.algorithms.base import MethodResult
from core.datasets import registry as dreg
from core.datasets.base import Sample
from core.datasets.io import read_mask, read_rgb8
from core.experiments.store import ExperimentStore
from core.inference.session import Calibration, LabSession

FMT = lambda v, d=4: "—" if v is None else (f"{v:.{d}f}" if isinstance(v, float) else str(v))


class LabPage(QWidget):
    def __init__(self, main):
        super().__init__()
        self.main = main
        self.session: LabSession | None = None
        self.session_key = None
        self.result: MethodResult | None = None
        self.sample: Sample | None = None
        self.image: np.ndarray | None = None
        self.gt: np.ndarray | None = None
        self.task = None
        self.store = ExperimentStore()
        # reference each displayed result / calibration / evaluation was computed with (stale-result guard)
        self.result_ref: str | None = None
        self._cal_ref: str | None = None
        self._eval_ref: str | None = None
        self._thr_src = ""

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(PageHeader(tr("nav.lab", "Methods Laboratory"),
                                 tr("lab.subtitle", "Select dataset → sample → method → parameters, then run the real implementation.")))
        self.banner = synthetic_banner()
        self.banner.hide()
        lay.addWidget(self.banner)
        vs = QSplitter(Qt.Vertical)
        hs = QSplitter(Qt.Horizontal)

        # ------------------------------------------------------------ left
        left = QScrollArea()
        left.setWidgetResizable(True)
        lw = QWidget()
        ll = QVBoxLayout(lw)
        self.browser = SampleBrowser(main.catalog, modality_filter="rgb")
        self.browser.setMinimumHeight(330)
        self.browser.sampleSelected.connect(self._on_sample)
        self.browser.indexChanged.connect(self._on_index)
        ll.addWidget(self.browser, 3)
        g = QGroupBox(tr("lab.method", "Method"))
        gl = QVBoxLayout(g)
        self.method = QComboBox()
        for mid in areg.image_methods():
            rec = main.catalog.algorithms.get(mid)
            ok, why = areg.availability(mid)
            self.method.addItem(("" if ok else "✗ ") + (rec["name"] if rec else mid), mid)
            if not ok:
                self.method.setItemData(self.method.count() - 1, why, Qt.ToolTipRole)
        gl.addWidget(self.method)
        self.avail = muted("")
        gl.addWidget(self.avail)
        # ---- Model Registry: pick a fitted checkpoint by its exact configuration
        self.reg_box = QGroupBox(tr("lab.registry", "Fitted models (Model Registry)"))
        rg = QVBoxLayout(self.reg_box)
        self.ckpt = QComboBox()
        self.ckpt.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.ckpt.setMinimumContentsLength(24)      # long checkpoint labels must not widen the window
        self.ckpt.setToolTip(tr("lab.registry_tip", "Checkpoints for this method and dataset category. Loading uses the "
                                                     "checkpoint's own parameters and fit images — no training."))
        rg.addWidget(self.ckpt)
        self.ckpt_info = QLabel("")
        self.ckpt_info.setWordWrap(True)
        self.ckpt_info.setTextFormat(Qt.RichText)
        rg.addWidget(self.ckpt_info)
        self.load_btn = QPushButton(tr("lab.load_ckpt", "Load selected model (no training)"))
        rg.addWidget(self.load_btn)
        gl.addWidget(self.reg_box)
        self.params = ParamForm()
        gl.addWidget(self.params)
        self.ref_row = QHBoxLayout()
        self.ref_combo = QComboBox()
        self.ref_label = QLabel(tr("lab.reference", "Reference (normal)"))
        self.ref_row.addWidget(self.ref_label)
        self.ref_row.addWidget(self.ref_combo, 1)
        gl.addLayout(self.ref_row)
        fit_row = QHBoxLayout()
        self.n_fit = QSpinBox()
        self.n_fit.setRange(1, 2000)
        self.n_fit.setValue(60)
        self.n_fit.setToolTip(tr("lab.n_fit_tip", "Max. number of train/good images used to fit the model"))
        self.fit_btn = QPushButton(tr("lab.fit", "Fit on normal training images"))
        fit_row.addWidget(self.fit_btn, 1)
        fit_row.addWidget(self.n_fit)
        self.n_fit.valueChanged.connect(lambda _v: self._update_buttons())
        gl.addLayout(fit_row)
        self.cal_btn = QPushButton(tr("lab.calibrate", "Calibrate thresholds (normal images only)"))
        gl.addWidget(self.cal_btn)
        self.run_btn = QPushButton("▶  " + tr("lab.run", "Run on selected sample"))
        self.run_btn.setObjectName("Primary")
        self.run_btn.setShortcut("F5")
        gl.addWidget(self.run_btn)
        ev = QHBoxLayout()
        self.eval_btn = QPushButton(tr("lab.evaluate", "Evaluate on test split"))
        self.n_eval = QSpinBox()
        self.n_eval.setRange(4, 5000)
        self.n_eval.setValue(80)
        self.n_eval.setToolTip(tr("lab.n_eval_tip", "Max. number of labelled test samples (stratified, seeded)"))
        ev.addWidget(self.eval_btn, 1)
        ev.addWidget(self.n_eval)
        gl.addLayout(ev)
        self.progress = QProgressBar()
        self.progress.setRange(0, 1000)
        self.progress_msg = muted("")
        self.cancel_btn = QPushButton(tr("lab.cancel", "Cancel"))
        self.cancel_btn.setEnabled(False)
        gl.addWidget(self.progress)
        gl.addWidget(self.progress_msg)
        gl.addWidget(self.cancel_btn)
        ll.addWidget(g)
        left.setWidget(lw)
        left.setMinimumWidth(290)
        left.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        hs.addWidget(left)
        hs.setStretchFactor(1, 1)

        # ------------------------------------------------------------ center
        center = QWidget()
        grid = QGridLayout(center)
        grid.setContentsMargins(0, 0, 0, 0)
        self.v_orig = TitledImage(tr("lab.v_orig", "Original sample"))
        self.v_gt = TitledImage(tr("lab.v_gt", "Ground truth"))
        self.v_map = TitledImage(tr("lab.v_map", "Anomaly map (raw scores)"))
        self.v_over = TitledImage(tr("lab.v_over", "Heat-map overlay"))
        self.v_mask = TitledImage(tr("lab.v_mask", "Predicted mask @ threshold"))
        self.v_native = TitledImage(tr("lab.v_native", "Method-native output"))
        for v in (self.v_orig, self.v_gt, self.v_map, self.v_over, self.v_mask, self.v_native):
            v.caption.setWordWrap(True)        # long captions must not widen the page (1280-px laptops)
            v.setMinimumSize(90, 90)
        for i, v in enumerate((self.v_orig, self.v_gt, self.v_native, self.v_map, self.v_over, self.v_mask)):
            grid.addWidget(v, i // 3, i % 3)
        # synchronise zoom/pan across panels
        views = [v.view for v in (self.v_orig, self.v_gt, self.v_map, self.v_over, self.v_mask, self.v_native)]
        for a in views:
            a.viewChanged.connect(lambda src=a: [b.setTransform(src.transform()) for b in views if b is not src])
        self.center_tabs = QTabWidget()
        self.center_tabs.addTab(center, tr("lab.tab_results", "Results"))
        self.refview = ReferenceComparePanel()
        self.refview.compareRequested.connect(self._run)
        self.refview.stepRequested.connect(self._step_ref)
        self.center_tabs.addTab(self.refview, tr("ref.tab", "Reference comparison"))
        hs.addWidget(self.center_tabs)

        # ------------------------------------------------------------ right
        right = QScrollArea()
        right.setWidgetResizable(True)
        right.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        right.setMinimumWidth(300)
        rw = QWidget()
        rl = QVBoxLayout(rw)
        rl.setSizeConstraint(rl.SizeConstraint.SetMinimumSize)
        sc = QGroupBox(tr("lab.scores", "Scores and decision"))
        sl = QVBoxLayout(sc)
        self.score_lbl = QLabel("—")
        self.score_lbl.setObjectName("PageTitle")
        sl.addWidget(muted(tr("lab.raw_score", "Image-level anomaly score (raw, uncalibrated — not a probability)")))
        sl.addWidget(self.score_lbl)
        self.decision = Badge("—")
        self.decision.setWordWrap(True)
        sl.addWidget(self.decision)
        self.img_thr_lbl = muted("")
        sl.addWidget(self.img_thr_lbl)
        pl = QLabel(tr("lab.pixel_thr", "Pixel threshold (re-thresholds cached map, no re-inference)"))
        pl.setWordWrap(True)
        sl.addWidget(pl)
        self.thr = QSlider(Qt.Horizontal)
        self.thr.setRange(0, 1000)
        self.thr_lbl = muted("")
        sl.addWidget(self.thr)
        sl.addWidget(self.thr_lbl)
        self.fixed_scale = QComboBox()
        self.fixed_scale.addItem(tr("lab.scale_image", "Colour scale: this image min–max"), "image")
        self.fixed_scale.addItem(tr("lab.scale_cal", "Colour scale: calibrated (comparable)"), "cal")
        sl.addWidget(self.fixed_scale)
        self.perf = QLabel("")
        self.perf.setWordWrap(True)
        sl.addWidget(self.perf)
        rl.addWidget(sc)
        evg = QGroupBox(tr("lab.metrics", "Evaluation (local, educational)"))
        el = QVBoxLayout(evg)
        self.metrics = QTextBrowser()
        self.metrics.setMinimumHeight(260)
        el.addWidget(self.metrics)
        rl.addWidget(evg)
        rl.addStretch(1)
        right.setWidget(rw)
        hs.addWidget(right)
        hs.setSizes([340, 860, 330])
        vs.addWidget(hs)

        self.bottom = QWidget()
        self.bl = QHBoxLayout(self.bottom)
        self.bl.setSizeConstraint(self.bl.SizeConstraint.SetMinimumSize)
        self.bl.setContentsMargins(0, 4, 0, 0)
        bscroll = QScrollArea()
        bscroll.setWidgetResizable(True)
        bscroll.setWidget(self.bottom)
        bscroll.setMinimumHeight(120)
        vs.addWidget(bscroll)
        vs.setStretchFactor(0, 4)
        vs.setStretchFactor(1, 1)
        vs.setSizes([780, 240])
        lay.addWidget(vs, 1)

        self.method.currentIndexChanged.connect(self._on_method)
        self.params.changed.connect(self._on_param)
        self.ckpt.currentIndexChanged.connect(self._on_ckpt)
        self.load_btn.clicked.connect(self._load_ckpt)
        self.fit_btn.clicked.connect(self._fit)
        self.cal_btn.clicked.connect(self._calibrate)
        self.run_btn.clicked.connect(self._run)
        self.eval_btn.clicked.connect(self._evaluate)
        self.cancel_btn.clicked.connect(lambda: self.task and self.task.cancel())
        self.thr.valueChanged.connect(self._rethreshold)
        self.fixed_scale.currentIndexChanged.connect(self._show_result)
        self.ref_combo.currentIndexChanged.connect(self._on_ref)
        self._on_method()
        if self.browser.index:
            self._on_index(self.browser.index)

    # ----------------------------------------------------------- helpers
    @property
    def method_id(self) -> str:
        return self.method.currentData()

    def _busy(self, busy: bool):
        for b in (self.fit_btn, self.cal_btn, self.run_btn, self.eval_btn, self.method, self.load_btn):
            b.setEnabled(not busy)
        self.cancel_btn.setEnabled(busy)
        if not busy:
            self._update_buttons()

    def _progress(self, f: float, m: str):
        self.progress.setValue(int(f * 1000))
        self.progress_msg.setText(m)

    def _error(self, msg: str, tb: str):
        self._busy(False)
        self.progress_msg.setText("✗ " + msg)
        print(tb)
        hint = ""
        if "CUDA out of memory" in msg:
            hint = "Reduce the number of fitting images or the input size, or switch the device to CPU in Settings."
        elif "fit()" in msg:
            hint = "Press 'Fit on normal training images' first."
        elif "download" in msg.lower() or "connection" in msg.lower() or "urlopen" in msg.lower():
            hint = "Pretrained weights are not cached yet. Run prepare_models.ps1 while online."
        self.main.show_error(tr("lab.error", "Method error"), msg, hint)

    def _session(self) -> LabSession | None:
        idx = self.browser.index
        if idx is None:
            return None
        vals = self.params.values()
        fit_params = {p.name: vals[p.name] for p in self.params.specs if p.affects_fit}
        key = (self.method_id, idx.dataset_id, idx.category, json.dumps(fit_params, sort_keys=True))
        if self.session is None or self.session_key != key:
            self.session = LabSession(self.method_id, vals, seed=settings.get().seed)
            self.session.bind(idx)
            self.session_key = key
            self._cal_ref = None                 # a new session starts uncalibrated
            ref = self._selected_ref()      # keep the user's choice (bind() defaults to the first normal)
            if self.session.method.requires_reference and ref is not None:
                self.session.set_reference(ref)
        else:
            self.session.method.params.update(vals)
        return self.session

    def _update_buttons(self):
        ok, why = areg.availability(self.method_id)
        cls = areg.runtime_class(self.method_id)
        s = self.session
        needs_fit = bool(cls and cls.requires_fit)
        fitted = bool(s and s.method.fitted)
        has_data = self.browser.index is not None
        self.fit_btn.setVisible(needs_fit)
        self.n_fit.setVisible(needs_fit)
        self.fit_btn.setEnabled(ok and has_data)
        self.run_btn.setEnabled(ok and has_data and (fitted or not needs_fit))
        self.cal_btn.setEnabled(ok and has_data and (fitted or not needs_fit))
        self.eval_btn.setEnabled(ok and has_data and (fitted or not needs_fit))
        self.avail.setText(("✓ " + tr("lab.ready", "ready")) if ok else "✗ " + why)
        if fitted and s.checkpoint is not None:
            self.avail.setText("✓ " + tr("lab.loaded", "Fitted model loaded") + f": {s.checkpoint.name}"
                               + (" · Workshop Verified" if s.calibration.stored else ""))
        if ok and needs_fit and not fitted:
            self.avail.setText(tr("lab.needs_fit", "Requires fitting on normal training images (train split only)."))
            if any(e.tier == "workshop_verified" for e in getattr(self, "_entries", [])):
                self.avail.setText(tr("lab.verified_avail", "A Workshop Verified model is available below — load it "
                                                            "instead of fitting (no training)."))
            sess = self._session() if has_data else None
            cached = sess.cached_fit(self.n_fit.value()) if sess else None
            if cached:
                self.avail.setText(tr("lab.cached_fit", "A fitted model for these settings is cached — "
                                                        "'Fit' reloads it in seconds") + f" ({cached.name})")
        is_ref = bool(cls and cls.requires_reference)
        for w in (self.ref_combo, self.ref_label):
            w.setVisible(is_ref)
        self.center_tabs.setTabVisible(1, self.method_id in PANEL_METHODS)

    # ----------------------------------------------------------- events
    def _on_index(self, idx):
        a = dreg.get(idx.dataset_id)
        self.banner.setVisible(bool(a and a.synthetic))
        self.session = None
        self.result = None
        self._fill_refs()
        self._refresh_ckpts()
        self._update_buttons()

    # ----------------------------------------------------------- model registry
    def _refresh_ckpts(self):
        from core.inference import model_registry as MR
        idx = self.browser.index
        cls = areg.runtime_class(self.method_id)
        self._entries = MR.for_index(idx, self.method_id) if idx is not None and cls and cls.cacheable else []
        self.reg_box.setVisible(bool(cls and cls.requires_fit and cls.cacheable))
        self.ckpt.blockSignals(True)
        self.ckpt.clear()
        for e in self._entries:
            self.ckpt.addItem(f"[{MR.TIER_LABEL[e.tier]}] {e.label()}", e.key)
        if not self._entries:
            self.ckpt.addItem(tr("lab.no_ckpt", "— no fitted model for this method and category —"), None)
        self.ckpt.blockSignals(False)
        self._on_ckpt()

    def _entry(self):
        k = self.ckpt.currentData()
        return next((e for e in getattr(self, "_entries", []) if e.key == k), None)

    def _on_ckpt(self):
        from core.inference import model_registry as MR
        e = self._entry()
        if e is None:
            self.ckpt_info.setText("")
            self.load_btn.setEnabled(False)
            return
        ok, why = MR.compatibility(e, self.browser.index)
        rows = "".join(f"<tr><td><b>{k}</b></td><td>{v}</td></tr>" for k, v in e.describe())
        warn = "" if ok else ("<p style='color:#e8590c'><b>⚠ " + tr("lab.incompatible", "Incompatible with the selected "
                                                                    "experiment") + ":</b> " + "; ".join(why) + "</p>")
        self.ckpt_info.setText(f"<table cellpadding=1>{rows}</table>{warn}")
        self.load_btn.setEnabled(ok)

    def _load_ckpt(self):
        e = self._entry()
        if e is None:
            return
        self.params.blockSignals(True)
        self.params.set_specs(self.params.specs, e.manifest["params"])
        self.params.blockSignals(False)
        self.session = None
        s = self._session()
        self._busy(True)

        def done(info):
            self._busy(False)
            self.progress_msg.setText(f"✓ {tr('lab.loaded', 'Fitted model loaded')} ({info.get('tier')}): {info.get('cache')}"
                                      + (" · " + tr("lab.cal_restored", "verified thresholds restored")
                                         if s.calibration.stored else ""))
            self._update_buttons()
        self.task = start(s.load_checkpoint, entry=e, on_done=done, on_error=self._error, on_progress=self._progress)

    def _weights_ok(self, method_id: str) -> bool:
        """Pretrained backbone missing → ask before the (online) download; never download silently."""
        from core import model_manager as mm
        from core.system.readiness import weights_cached
        w = mm.WEIGHTS.get(method_id)
        if w is None or method_id == "efficientad" or weights_cached(method_id):
            return True
        return self._confirm_download(w)

    def _confirm_download(self, w: dict) -> bool:
        """Patched in tests."""
        from PySide6.QtWidgets import QMessageBox
        return QMessageBox.question(self, tr("mm.confirm_title", "Download pretrained weights?"),
                                    tr("lab.confirm_download", "Fitting needs the pretrained backbone {res} ({size}), "
                                                               "which is not on this computer. Download it now from {url}?"
                                                               "\n\nLicence: {lic}").format(res=w["resource"], size=w["size"],
                                                                                            url=w["url"], lic=w["licence"])
                                    ) == QMessageBox.Yes

    def _ask_train(self, text: str, can_load: bool) -> str:
        """'load' | 'train' | 'cancel' — patched in tests."""
        from PySide6.QtWidgets import QMessageBox
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Question)
        box.setWindowTitle(tr("lab.train_q", "Train a new model?"))
        box.setText(text)
        load = box.addButton(tr("lab.use_ckpt", "Load the selected checkpoint"), QMessageBox.AcceptRole) if can_load else None
        train = box.addButton(tr("lab.train_anyway", "Train now"), QMessageBox.DestructiveRole)
        box.addButton(QMessageBox.Cancel)
        box.exec()
        c = box.clickedButton()
        return "load" if load is not None and c is load else "train" if c is train else "cancel"

    # ----------------------------------------------------------- reference (normal) image
    def _fill_refs(self):
        """List the train/good images of the current category. Filled as soon as a category is selected
        (not only at the first run), so what the combo shows is what the next comparison uses."""
        prev = self.ref_combo.currentData()
        self.ref_combo.blockSignals(True)
        self.ref_combo.clear()
        if self.browser.index:
            for s in self.browser.index.train_normals()[:200]:
                self.ref_combo.addItem(s.sample_id, s.sample_id)
        self.ref_combo.setCurrentIndex(max(0, self.ref_combo.findData(prev)) if prev is not None else 0)
        self.ref_combo.blockSignals(False)
        self._on_ref()

    def _selected_ref(self) -> Sample | None:
        idx, sid = self.browser.index, self.ref_combo.currentData()
        if idx is None or sid is None:
            return None
        return next((x for x in idx.train_normals() if x.sample_id == sid), None)

    def current_ref_id(self) -> str | None:
        s = self.session
        if s is not None and s.method.requires_reference and s.reference is not None:
            return s.reference.sample_id
        return None

    def _step_ref(self, d: int):
        n = self.ref_combo.count()
        if n:
            self.ref_combo.setCurrentIndex((self.ref_combo.currentIndex() + d) % n)

    def _on_ref(self):
        cls = areg.runtime_class(self.method_id)
        ref = self._selected_ref()
        if not (cls and cls.requires_reference) or ref is None:
            return
        s = self._session()
        if s is None:
            return
        if s.reference is None or s.reference.sample_id != ref.sample_id:
            s.set_reference(ref)
        self._invalidate_ref_results(ref.sample_id)
        rec = self.main.catalog.datasets.get(ref.dataset_id) or {}
        name = rec.get("name") or ("Synthetic demo" if ref.dataset_id == "SYN" else ref.dataset_id)
        self.refview.set_reference(ref, s._ref_img, f"{name} ({ref.dataset_id})" if name != ref.dataset_id else name)

    def _invalidate_ref_results(self, ref_id: str):
        """Everything computed against another reference is discarded (never shown as current)."""
        msgs = []
        if self.result is not None and self.result_ref is not None and self.result_ref != ref_id:
            self.result, self.result_ref = None, None
            for v in (self.v_map, self.v_over, self.v_mask, self.v_native):
                v.set(None)
            self.score_lbl.setText("—")
            self.decision.setText("—")
            self.thr_lbl.setText("")
            self.perf.setText("")
            msgs.append(tr("ref.inv_result", "result"))
        s = self.session
        if s is not None and self._cal_ref is not None and self._cal_ref != ref_id and not s.calibration.stored:
            s.calibration = Calibration()           # thresholds came from differences to the old reference
            self._cal_ref = None
            self.img_thr_lbl.setText("")
            msgs.append(tr("ref.inv_cal", "calibrated thresholds"))
        if self._eval_ref is not None and self._eval_ref != ref_id:
            self.metrics.setHtml("<p style='color:#e8590c'>⚠ " + tr("ref.inv_eval_html", "This evaluation used another "
                                 "reference and was discarded. Press 'Evaluate' again.") + "</p>")
            self._eval_ref = None
            msgs.append(tr("ref.inv_eval", "evaluation"))
        if self.refview.result is not None and self.refview.result_ref_id != ref_id:
            self.refview.invalidate()
        if msgs:
            self.progress_msg.setText("⚠ " + tr("ref.invalidated", "Reference changed — discarded") + ": "
                                      + ", ".join(msgs) + ". " + tr("ref.press_run", "Press Run / Compare."))

    def _on_method(self):
        cls = areg.runtime_class(self.method_id)
        self.params.set_specs(cls.param_specs() if cls else [])
        self.session = None
        self.result = None
        rec = self.main.catalog.algorithms.get(self.method_id)
        if hasattr(self, "ckpt"):
            self._refresh_ckpts()
        while self.bl.count():
            w = self.bl.takeAt(0).widget()
            if w:
                w.deleteLater()
        if rec:
            es = settings.get().language == "es"
            md = (f"**{rec['name']}** — {rec['intuitive_explanation_es' if es else 'intuitive_explanation_en']}\n\n"
                  f"**{tr('lib.q4', 'Training data required')}:** {rec['training_data_requirements']}\n\n"
                  f"**{tr('lib.q8', 'Limitations')}:** " + "; ".join(rec["limitations"][:3]))
            if self.method_id in PANEL_METHODS:
                md += "\n\n" + explanation_md()
            self.bl.addWidget(markdown_browser(md), 3)
            self.bl.addWidget(ReferencesPanel(self.main.catalog, rec, "algorithm"), 2)
        self._update_buttons()
        if hasattr(self, "refview"):
            self.result_ref = None
            self.refview.set_test(self.sample, self.image)
            self._on_ref()
            if self.method_id in PANEL_METHODS:
                self.center_tabs.setCurrentIndex(1)

    def _on_param(self, name: str):
        spec = next((p for p in self.params.specs if p.name == name), None)
        if spec and spec.affects_fit and self.session and self.session.method.fitted:
            self.progress_msg.setText(tr("lab.refit", "Parameter changed — the model must be re-fitted."))
            self.session = None
        self._update_buttons()

    def _on_sample(self, s: Sample):
        self.sample = s
        self.image = read_rgb8(s.primary_path)
        self.gt = read_mask(s.masks["rgb"], self.image.shape[:2]) if s.masks.get("rgb") else None
        self.result = None
        self.v_orig.set(self.image, f"{tr('lab.v_orig', 'Original sample')} — {s.split}/{s.defect_type or s.label}")
        if self.gt is not None:
            self.v_gt.set(mask_overlay(self.image, self.gt), tr("lab.v_gt", "Ground truth"))
        else:
            self.v_gt.set(None, tr("lab.no_gt", "Ground truth: none (normal image or labels withheld)"))
        for v in (self.v_map, self.v_over, self.v_mask, self.v_native):
            v.set(None)
        self.score_lbl.setText("—")
        self.decision.setText("—")
        self.result_ref = None
        self.refview.set_test(s, self.image)

    # ----------------------------------------------------------- actions
    def _fit(self):
        s = self._session()
        if not s:
            return
        # Never start an unexpected training run: if no cache matches these exact settings, say so
        # when other checkpoints exist for this method/category or the workshop profile is active.
        if s.method.requires_fit and s.cached_fit(self.n_fit.value()) is None:
            entries = getattr(self, "_entries", [])
            if entries or settings.get().workshop_mode:
                best = next((e for e in entries if e.tier == "workshop_verified"), entries[0] if entries else None)
                msg = (tr("lab.train_msg", "No fitted model matches these exact settings (parameters and number of fit "
                                           "images), so 'Fit' would TRAIN a new model now.")
                       + (f"\n\n{tr('lab.available', 'Available')}: {best.label()} [{best.tier}]" if best else "")
                       + ("\n\n" + tr("lab.workshop_on", "Workshop offline profile is active.") if settings.get().workshop_mode else ""))
                choice = self._ask_train(msg, best is not None)
                if choice == "cancel":
                    return
                if choice == "load":
                    self.ckpt.setCurrentIndex(self.ckpt.findData(best.key))
                    self._load_ckpt()
                    return
        if s.method.requires_fit and s.cached_fit(self.n_fit.value()) is None and not self._weights_ok(s.method_id):
            return
        self._busy(True)

        def done(info):
            self._busy(False)
            self.progress_msg.setText(f"✓ fitted: {info}")
            self._refresh_ckpts()
            if s.checkpoint is not None:
                self.ckpt.setCurrentIndex(max(0, self.ckpt.findData(s.checkpoint.stem.rsplit("_", 1)[-1])))
        self.task = start(s.fit, max_images=self.n_fit.value(), on_done=done, on_error=self._error,
                          on_progress=self._progress)

    def _calibrate(self):
        s = self._session()
        if not s:
            return
        self._busy(True)

        ref_id = self.current_ref_id()

        def done(cal):
            self._busy(False)
            if ref_id != self.current_ref_id():
                s.calibration = Calibration()
                self.progress_msg.setText("⚠ " + tr("ref.discarded_cal", "Calibration discarded: the reference changed "
                                                                       "during the run."))
                return
            self._cal_ref = ref_id
            self.progress_msg.setText(f"✓ thresholds from {cal.source}")
            self._show_result()
        self.task = start(s.calibrate, on_done=done, on_error=self._error, on_progress=self._progress)

    def _run(self):
        s = self._session()
        if not s or self.image is None:
            self.progress_msg.setText(tr("lab.select_sample", "Select a sample first."))
            return
        self._busy(True)
        img = self.image
        ref_id = self.current_ref_id()
        sample_id = self.sample.sample_id if self.sample is not None else None

        def job(progress, cancel):
            progress(0.1, "running inference…")
            return s.predict_image(img)

        def done(res):
            self._busy(False)
            self.progress.setValue(1000)
            cur = self.sample.sample_id if self.sample is not None else None
            if s is not self.session or ref_id != self.current_ref_id() or sample_id != cur:
                # the reference, sample or method changed while this ran: never show it as current
                self.progress_msg.setText("⚠ " + tr("ref.discarded", "Result discarded: the reference or sample "
                                                                   "changed during the run. Press Run again."))
                return
            self.result = res
            self.result_ref = ref_id
            self.progress_msg.setText("✓ " + tr("lab.done", "done"))
            self._init_threshold()
            self._show_result(log=True)
        self.task = start(job, on_done=done, on_error=self._error, on_progress=self._progress)

    def _evaluate(self):
        s = self._session()
        if not s:
            return
        self._busy(True)

        ref_id = self.current_ref_id()

        def done(res):
            self._busy(False)
            if ref_id != self.current_ref_id():
                self.progress_msg.setText("⚠ " + tr("ref.discarded_eval", "Evaluation discarded: the reference changed "
                                                                        "during the run."))
                return
            self._eval_ref = ref_id
            self._show_metrics(res)
        self.task = start(s.evaluate, max_images=self.n_eval.value(), store=self.store, localization=True,
                          on_done=done, on_error=self._error, on_progress=self._progress)

    # ----------------------------------------------------------- display
    def _init_threshold(self):
        r = self.result
        if r is None or r.anomaly_map is None:
            return
        cal = self.session.calibration if self.session else None
        lo, hi = float(r.anomaly_map.min()), float(r.anomaly_map.max())
        if cal and cal.pixel_threshold is not None:
            hi = max(hi, cal.pixel_threshold * 1.5)
        self._range = (lo, hi if hi > lo else lo + 1)
        t = cal.pixel_threshold if cal and cal.pixel_threshold is not None else r.threshold
        t = t if t is not None else (lo + hi) / 2
        self._thr_src = (tr("ref.src_cal", "calibrated") if cal and cal.pixel_threshold is not None
                         else tr("ref.src_param", "method parameter") if r.threshold is not None
                         else tr("ref.src_mid", "mid-range"))
        self.thr.blockSignals(True)
        self.thr.setValue(int(round(1000 * (t - self._range[0]) / (self._range[1] - self._range[0]))))
        self.thr.blockSignals(False)
        self._thr_exact = t        # the slider is quantised: until it is moved, use the exact threshold

    def _thr_value(self) -> float:
        if getattr(self, "_thr_exact", None) is not None:
            return self._thr_exact
        lo, hi = getattr(self, "_range", (0.0, 1.0))
        return lo + (hi - lo) * self.thr.value() / 1000

    def _rethreshold(self):
        self._thr_exact = None
        self._thr_src = tr("ref.src_manual", "slider")
        if self.result is not None:
            self._show_result()

    def _show_result(self, log: bool = False):
        r = self.result
        if r is None or self.image is None:
            return
        thr = self._thr_value()
        cal = self.session.calibration if self.session else None
        if r.anomaly_map is not None:
            if self.fixed_scale.currentData() == "cal" and cal and cal.pixel_threshold is not None:
                cm = colorize(r.anomaly_map, 0.0, cal.pixel_threshold * 2)
            else:
                cm = colorize(r.anomaly_map)
            mask = r.anomaly_map >= thr
            self.v_map.set(cm, f"{tr('lab.v_map', 'Anomaly map (raw scores)')}: [{r.anomaly_map.min():.3g}, {r.anomaly_map.max():.3g}]")
            self.v_over.set(overlay(self.image, cm, 0.5))
            self.v_mask.set(mask_overlay(self.image, mask, color=(255, 200, 0)),
                            f"{tr('lab.v_mask', 'Predicted mask @ threshold')} {thr:.4g}")
            self.thr_lbl.setText(f"{thr:.4g}  ({100 * mask.mean():.2f} % px)"
                                 + (f" · calibrated: {cal.pixel_threshold:.4g}" if cal and cal.pixel_threshold is not None else ""))
        self.v_native.set(r.visual, tr("lab.v_native", "Method-native output") if r.visual is not None
                          else tr("lab.no_native", "Method-native output: none"))
        self.score_lbl.setText(FMT(r.score))
        pred = None
        if cal and cal.image_threshold is not None and r.score is not None:
            nok = r.score >= cal.image_threshold
            pred = "NOK" if nok else "OK"
            if labels_withheld(self.sample):
                # test_private: labels withheld → a prediction only, never presented as ground truth
                self.decision.setText(f"{tr('lab.prediction', 'Prediction')}: {pred} — "
                                      + tr("lab.not_gt", "not ground truth (labels withheld)"))
                self.decision.setObjectName("BadgeWarn")
            else:
                self.decision.setText("NOK (anomalous)" if nok else "OK (normal)")
                self.decision.setObjectName("BadgeBad" if nok else "BadgeOk")
            self.img_thr_lbl.setText(f"{tr('lab.img_thr', 'Image threshold')}: {cal.image_threshold:.4g} — {cal.source}")
        else:
            self.decision.setText(tr("lab.no_decision", "No OK/NOK decision: calibrate a threshold on normal images first"))
            self.decision.setObjectName("Badge")
            self.img_thr_lbl.setText("")
        self.decision.style().unpolish(self.decision)
        self.decision.style().polish(self.decision)
        gpu = f"{r.gpu_memory_mb:.0f} MB peak" if r.gpu_memory_mb is not None else "n/a"
        self.perf.setText(f"<b>Latency:</b> {r.latency_ms:.1f} ms · <b>Device:</b> {r.device} · <b>GPU mem:</b> {gpu}"
                          f"<br><b>Model:</b> {r.model_version}<br><i>{r.score_semantics}</i>")
        if self.method_id in PANEL_METHODS and r.extras.get("stages") and self.result_ref is not None:
            if log or self.refview.result is not r:
                self.refview.set_result(r, self.result_ref, thr, self._thr_src, self.session.method.params,
                                        decision=pred, log=log)
            else:
                self.refview.update_threshold(thr, self._thr_src)

    def _show_metrics(self, res: dict):
        rows = [("Result category", res["result_category"]), ("Test samples", f"{res['n_test']} ({res['n_anomalous']} anomalous)"),
                ("Image AUROC", FMT(res["image_auroc"])), ("Image AP", FMT(res["image_ap"])),
                ("Pixel AUROC", FMT(res["pixel_auroc"]) + (f" (maps ≤{res.get('pixel_metric_max_side', 256)} px)"
                                                          if res["pixel_auroc"] else "")),
                ("AU-PRO @ FPR 0.3 / 0.05", f"{FMT(res.get('au_pro_30'))} / {FMT(res.get('au_pro_05'))}"),
                ("Pixel F1 (calibrated thr.)", FMT((res.get("pixel_f1") or {}).get("f1"))),
                ("Mean Dice (anomalous)", FMT(res.get("mean_dice"))), ("Mean IoU (anomalous)", FMT(res.get("mean_iou")))]
        d = res.get("decision")
        if d:
            rows += [("Threshold", f"{d['threshold']:.4g} ({d['threshold_source']})"),
                     ("Precision / Recall / F1", f"{FMT(d['precision'], 3)} / {FMT(d['recall'], 3)} / {FMT(d['f1'], 3)}"),
                     ("False acceptance (defect → OK)", FMT(d["false_acceptance_rate"], 3)),
                     ("False rejection (good → NOK)", FMT(d["false_rejection_rate"], 3)),
                     ("TP / FP / TN / FN", f"{d['tp']} / {d['fp']} / {d['tn']} / {d['fn']}")]
        else:
            rows.append(("Decision metrics", "not computed — calibrate a threshold on normal images first"))
        lat = res.get("latency", {})
        if lat:
            rows.append(("Latency mean / p95", f"{lat['mean_ms']:.1f} / {lat['p95_ms']:.1f} ms"))
        rows.append(("GPU peak (inference)", "— (CPU)" if res.get("gpu_peak_mb") is None else f"{res['gpu_peak_mb']:.0f} MB"))
        for c, b in (res.get("by_condition") or {}).items():
            rows.append((f"Image AUROC · {c}", f"{FMT(b['image_auroc'])} ({b['n_anomalous']} anom. / {b['n'] - b['n_anomalous']} normal)"))
        rows.append(("Run id", res.get("run_id", "—")))
        html = "<table cellpadding=3>" + "".join(f"<tr><td><b>{k}</b></td><td>{v}</td></tr>" for k, v in rows) + "</table>"
        html += f"<p style='color:gray'><i>{res['protocol']}</i></p>"
        self.metrics.setHtml(html)

    def apply_preset(self, method_id: str | None = None, dataset_id: str | None = None, **_):
        if dataset_id:
            self.browser.reload_datasets(select=dataset_id)
        if method_id:
            i = self.method.findData(method_id)
            if i >= 0:
                self.method.setCurrentIndex(i)

    def showEvent(self, e):
        super().showEvent(e)
        if self.browser.ds.count() == 0:
            self.browser.reload_datasets()
