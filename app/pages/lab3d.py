"""3D & Multimodal Lab — point-cloud inspection, synchronised RGB/IR/3D viewer, fusion experiment."""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QComboBox, QGroupBox, QHBoxLayout, QLabel, QProgressBar, QPushButton,
                               QSpinBox, QSplitter, QTabWidget, QTextBrowser, QVBoxLayout, QWidget)

from app.i18n import tr
from app.widgets.common import PageHeader, markdown_browser, mask_overlay, muted, synthetic_banner
from app.widgets.image_viewer import TitledImage
from app.widgets.param_form import ParamForm
from app.widgets.pointcloud_view import PointCloudView
from app.widgets.sample_browser import SampleBrowser
from app.workers.task import start
from core import settings
from core.algorithms.multimodal_3d.geometry import PointCloudDeviation
from core.datasets import registry as dreg
from core.datasets.base import Sample
from core.datasets.io import read_mask, read_point_cloud, read_rgb8
from core.datasets.real3d import read_point_labels
from core.evaluation import metrics as M
from core.research import content


def _md_inline(md: str) -> str:
    """Tiny Markdown → HTML for inline emphasis inside an HTML panel."""
    import re
    h = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", md)
    h = re.sub(r"\*(.+?)\*", r"<i>\1</i>", h)
    return h.replace("\n", "<br>")


def _load_cloud(path, max_points=250_000):
    pts, cols = read_point_cloud(path)
    if len(pts) > max_points:
        idx = np.random.default_rng(0).choice(len(pts), max_points, replace=False)
        pts, cols = pts[idx], (cols[idx] if cols is not None else None)
    return pts, cols


class PointCloudTab(QWidget):
    def __init__(self, main):
        super().__init__()
        self.main = main
        self.method: PointCloudDeviation | None = None
        self.method_key = None
        lay = QHBoxLayout(self)
        left = QWidget()
        ll = QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 0, 0)
        self.browser = SampleBrowser(main.catalog, modality_filter="pointcloud")
        self.browser.sampleSelected.connect(self._show)
        ll.addWidget(self.browser, 3)
        g = QGroupBox(tr("pc.method", "Registration + deviation (educational baseline)"))
        gl = QVBoxLayout(g)
        self.form = ParamForm()
        self.form.set_specs(PointCloudDeviation.param_specs())
        gl.addWidget(self.form)
        r = QHBoxLayout()
        r.addWidget(QLabel(tr("pc.n_ref", "Normal reference clouds")))
        self.n_ref = QSpinBox()
        self.n_ref.setRange(1, 10)
        self.n_ref.setValue(1)
        r.addWidget(self.n_ref)
        gl.addLayout(r)
        self.run = QPushButton("▶  " + tr("pc.run", "Align & compute deviation"))
        self.run.setObjectName("Primary")
        gl.addWidget(self.run)
        self.prog = QProgressBar()
        self.prog.setRange(0, 1000)
        gl.addWidget(self.prog)
        self.info = QTextBrowser()
        self.info.setMaximumHeight(210)
        gl.addWidget(self.info)
        ll.addWidget(g, 2)
        self.view = PointCloudView()
        sp = QSplitter(Qt.Horizontal)
        sp.addWidget(left)
        sp.addWidget(self.view)
        sp.setSizes([360, 1000])
        lay.addWidget(sp)
        self.run.clicked.connect(self._run)
        self.sample: Sample | None = None

    def _show(self, s: Sample):
        self.sample = s
        try:
            pts, cols = _load_cloud(s.modalities["pointcloud"])
        except Exception as e:
            self.info.setPlainText(str(e))
            return
        scal = {}
        if s.masks.get("pointcloud"):
            lab = read_point_labels(s.masks["pointcloud"], pts)
            if lab is not None:
                scal["ground-truth label"] = lab
        warn = "" if s.meta.get("units") else "⚠ units/registration not documented"
        self.view.set_cloud(pts, cols, scal, warn)
        self.info.setHtml(f"<b>{s.sample_id}</b> · {s.label} · {s.defect_type or '-'}<br>"
                          f"Units: {s.meta.get('units', 'unknown')}")

    def _run(self):
        s, idx = self.sample, self.browser.index
        if s is None or idx is None:
            return
        params = self.form.values()
        n_ref = self.n_ref.value()
        key = (idx.dataset_id, idx.category, n_ref, params["voxel"], params["coarse"], params["icp"])

        def job(progress, cancel):
            if self.method is None or self.method_key != key:
                refs = idx.train_normals()[:n_ref]
                if not refs:
                    raise RuntimeError("No normal training point clouds available as reference.")
                m = PointCloudDeviation(**params)
                progress(0.1, "loading normal reference clouds…")
                m.fit([_load_cloud(r.modalities["pointcloud"])[0] for r in refs], progress=lambda f, t: progress(0.1 + 0.4 * f, t))
                self.method, self.method_key = m, key
            self.method.params.update(params)
            progress(0.55, "aligning test cloud and measuring deviation…")
            pts, _ = _load_cloud(s.modalities["pointcloud"])
            res = self.method.predict(pts)
            lab = None
            if s.masks.get("pointcloud"):
                full = read_point_labels(s.masks["pointcloud"], pts)
                if full is not None:
                    lab = self.method.labels_on(res, pts, full)
            return res, lab

        def done(out):
            res, lab = out
            self.prog.setValue(1000)
            scal = {"deviation": res.deviation}
            if lab is not None:
                scal = {"ground-truth label": lab, "deviation": res.deviation}
            self.view.set_cloud(res.points, None, scal, "(test cloud aligned to reference frame)")
            self.view.color_by.setCurrentIndex(self.view.color_by.findData("deviation"))
            auc = M.image_auroc(lab.astype(int), res.deviation) if lab is not None and 0 < lab.sum() < len(lab) else None
            st = res.steps
            warn = "" if st.get("registration_ok", True) else (
                "<p style='color:#e5534b'><b>⚠ Registration failed (fitness &lt; 0.3): the deviation map is not "
                "meaningful.</b> Try coarse alignment on, another voxel size or another reference.</p>")
            self.info.setHtml(warn +
                f"<b>Score (p{self.method.params['percentile']:g} deviation):</b> {res.score:.4g} (input units)<br>"
                f"<b>Coarse fitness:</b> {st.get('coarse_fitness', float('nan')):.3f} · <b>ICP ({st.get('icp_used', '-')}) fitness:</b> "
                f"{res.fitness:.3f} · <b>RMSE:</b> {res.inlier_rmse:.4g} {st.get('note', '')}<br>"
                f"<b>Voxel:</b> {st['voxel']:.4g} · test pts {st['n_test']:,} · "
                f"ref pts {st['n_ref']:,}<br><b>Latency:</b> {res.latency_ms:.0f} ms<br>"
                + (f"<b>Point-level AUROC (this sample):</b> {auc:.4f}<br>" if auc is not None else "")
                + f"<i>{res.score_semantics}. Educational baseline — not a published 3D AD method.</i>")
        self.task = start(job, on_done=done, on_progress=lambda f, m: self.prog.setValue(int(f * 1000)),
                          on_error=lambda m, tb: (print(tb), self.main.show_error("3D", m)))


class MultimodalTab(QWidget):
    """Synchronised RGB / infrared / 3D views of one MulSen-AD sample."""

    def __init__(self, main):
        super().__init__()
        self.main = main
        lay = QHBoxLayout(self)
        sp = QSplitter(Qt.Horizontal)
        self.browser = SampleBrowser(main.catalog, modality_filter="infrared")
        self.browser.sampleSelected.connect(self._show)
        sp.addWidget(self.browser)
        right = QSplitter(Qt.Vertical)
        top = QWidget()
        tl = QHBoxLayout(top)
        tl.setContentsMargins(0, 0, 0, 0)
        self.rgb = TitledImage("RGB appearance")
        self.ir = TitledImage("Infrared (lock-in thermography, visualisation)")
        tl.addWidget(self.rgb)
        tl.addWidget(self.ir)
        right.addWidget(top)
        bot = QWidget()
        bl = QHBoxLayout(bot)
        bl.setContentsMargins(0, 0, 0, 0)
        self.pc = PointCloudView()
        bl.addWidget(self.pc, 3)
        self.info = QTextBrowser()
        self.info.setOpenExternalLinks(True)
        bl.addWidget(self.info, 2)
        right.addWidget(bot)
        sp.addWidget(right)
        sp.setSizes([330, 1100])
        lay.addWidget(sp)

    def _show(self, s: Sample):
        notes = []
        for key, view in (("rgb", self.rgb), ("infrared", self.ir)):
            p = s.modalities.get(key)
            if not p:
                view.set(None, f"{key}: missing")
                continue
            img = read_rgb8(p)
            m = s.masks.get(key)
            if m:
                img = mask_overlay(img, read_mask(m, img.shape[:2]))
            view.set(img, f"{'RGB appearance' if key == 'rgb' else 'Infrared (normalised display, not °C)'}"
                          f" — {img.shape[1]}×{img.shape[0]}" + (" · GT mask" if m else ""))
        if s.modalities.get("pointcloud"):
            try:
                pts, cols = _load_cloud(s.modalities["pointcloud"])
                scal = {}
                if s.masks.get("pointcloud"):
                    lab = read_point_labels(s.masks["pointcloud"], pts)
                    if lab is not None:
                        scal["ground-truth label"] = lab
                self.pc.set_cloud(pts, cols, scal, f"({s.meta.get('pointcloud_units', '')})")
            except Exception as e:
                notes.append(f"3D: {e}")
        ml = s.meta.get("modality_labels")
        mltxt = ""
        if isinstance(ml, dict) and ml:
            mltxt = "<br>".join(f"&nbsp;&nbsp;{k}: {'<b>anomalous</b>' if v else 'normal'}" for k, v in ml.items())
        self.info.setHtml(
            f"<b>{s.category} · {s.sample_id}</b><br>Object label: <b>{s.label}</b> ({s.defect_type or '-'})<br>"
            f"<b>Per-modality labels (data.csv):</b><br>{mltxt or '&nbsp;&nbsp;n/a'}<br><br>"
            f"<i>Pairing: same split, defect-type folder and file index across RGB/Infrared/Pointcloud.</i><br>"
            f"{s.meta.get('infrared_note', '')}<br>" + "<br>".join(notes)
            + "<hr>" + _md_inline(content.markdown("atlas", "modalities_short") or ""))


class FusionTab(QWidget):
    def __init__(self, main):
        super().__init__()
        self.main = main
        lay = QHBoxLayout(self)
        left = QWidget()
        ll = QVBoxLayout(left)
        self.browser = SampleBrowser(main.catalog, modality_filter="infrared")
        ll.addWidget(self.browser, 2)
        g = QGroupBox(tr("fus.setup", "Experiment"))
        gl = QVBoxLayout(g)
        self.method = QComboBox()
        for mid in ("patchcore", "padim"):
            self.method.addItem(mid, mid)
        self.n_fit = QSpinBox()
        self.n_fit.setRange(4, 200)
        self.n_fit.setValue(40)
        self.max_test = QSpinBox()
        self.max_test.setRange(5, 500)
        self.max_test.setValue(60)
        for lab, w in (("Image detector (RGB & IR)", self.method), ("Fit samples", self.n_fit), ("Max test samples", self.max_test)):
            gl.addWidget(QLabel(lab))
            gl.addWidget(w)
        self.run = QPushButton("▶  " + tr("fus.run", "Run RGB / IR / 3D + fusion"))
        self.run.setObjectName("Primary")
        gl.addWidget(self.run)
        self.prog = QProgressBar()
        self.prog.setRange(0, 1000)
        self.msg = muted("")
        gl.addWidget(self.prog)
        gl.addWidget(self.msg)
        ll.addWidget(g)
        lay.addWidget(left, 1)
        self.out = markdown_browser(content.markdown("methods", "_fusion") or "")
        lay.addWidget(self.out, 2)
        self.run.clicked.connect(self._run)

    def _run(self):
        idx = self.browser.index
        if idx is None or idx.dataset_id != "D06":
            self.msg.setText("Select an installed MulSen-AD category (dataset D06).")
            return
        from core.algorithms.multimodal_3d import fusion
        self.run.setEnabled(False)

        def done(res):
            self.run.setEnabled(True)
            a = res["auroc"]
            f = lambda v: "—" if v is None else f"{v:.3f}"
            md = [f"### Multimodal experiment — {idx.category}", f"*{res['protocol']}*", "",
                  "| Scores | AUROC vs object label | AUROC vs modality label |", "|---|---|---|"]
            for m in ("rgb", "infrared", "pointcloud"):
                md.append(f"| {m} (z) | {f(a['object_level'].get(m + '_z'))} | {f(a['modality_level'].get(m))} |")
            md += [f"| fused mean | {f(a['object_level']['fused_mean'])} | |",
                   f"| fused max | {f(a['object_level']['fused_max'])} | |", "",
                   "**Result category:** local educational experiment (not an official benchmark result).", "",
                   "| sample | label | RGB z | IR z | 3D z | fused max |", "|---|---|---|---|---|---|"]
            for r in res["rows"][:80]:
                md.append(f"| {r['sample_id']} | {r['label']} | {r['rgb_z']:.2f} | {r['infrared_z']:.2f} | "
                          f"{r['pointcloud_z']:.2f} | {r['fused_max']:.2f} |")
            self.out.setMarkdown("\n".join(md))
            from core.experiments.store import ExperimentStore
            st = ExperimentStore()
            run = st.create_run(category="local_educational", method_id="multimodal_fusion",
                                dataset_id="D06", dataset_category=idx.category, protocol=res["protocol"],
                                params={"image_method": res["image_method"]}, seed=settings.get().seed,
                                metrics=res["auroc"])
            self.msg.setText(f"✓ saved run {run}")
        start(fusion.run, index=idx, image_method=self.method.currentData(), n_fit=self.n_fit.value(),
              max_test=self.max_test.value(), seed=settings.get().seed, on_done=done,
              on_progress=lambda f, m: (self.prog.setValue(int(1000 * f)), self.msg.setText(m)),
              on_error=lambda m, tb: (print(tb), self.run.setEnabled(True), self.main.show_error("Fusion", m)))


class Lab3DPage(QWidget):
    def __init__(self, main):
        super().__init__()
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(PageHeader(tr("nav.lab3d", "3D & Multimodal Lab"),
                                 tr("l3d.subtitle", "Real point clouds, registration, geometric deviation and RGB / infrared / 3D comparison.")))
        self.tabs = QTabWidget()
        self.pc = PointCloudTab(main)
        self.mm = MultimodalTab(main)
        self.fu = FusionTab(main)
        self.tabs.addTab(self.pc, tr("l3d.tab_pc", "Point-cloud inspection"))
        self.tabs.addTab(self.mm, tr("l3d.tab_mm", "Multimodal viewer (RGB · IR · 3D)"))
        self.tabs.addTab(self.fu, tr("l3d.tab_fu", "Multimodal fusion experiment"))
        self.tabs.addTab(markdown_browser(content.markdown("methods", "_3d_steps") or ""), tr("l3d.tab_how", "How 3D inspection works"))
        lay.addWidget(self.tabs, 1)

    def shutdown(self):
        self.pc.view.close_plotter()
        self.mm.pc.close_plotter()

    def apply_preset(self, tab: str | None = None, **_):
        if tab:
            self.tabs.setCurrentIndex({"pc": 0, "mm": 1, "fusion": 2}.get(tab, 0))
