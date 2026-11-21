"""Dataset Explorer — browse normal / defective samples with ground-truth overlays."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QCheckBox, QFileDialog, QHBoxLayout, QLabel, QPushButton, QSlider,
                               QSplitter, QStackedWidget, QTextBrowser, QVBoxLayout, QWidget)

from app.i18n import tr
from app.widgets.common import PageHeader, markdown_browser, mask_overlay, muted, synthetic_banner
from app.widgets.image_viewer import ImageView
from app.widgets.pointcloud_view import PointCloudView
from app.widgets.references import ReferencesPanel
from app.widgets.sample_browser import SampleBrowser
from core import settings
from core.datasets import registry as dreg
from core.datasets.base import Sample
from core.datasets.io import read_image_raw, read_mask, read_point_cloud, to_display_rgb8, write_image


class ExplorerPage(QWidget):
    def __init__(self, main):
        super().__init__()
        self.main = main
        self.sample: Sample | None = None
        self.raw: np.ndarray | None = None
        self.rgb: np.ndarray | None = None
        self.mask: np.ndarray | None = None
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(PageHeader(tr("nav.explorer", "Dataset Explorer"),
                                 tr("exp.subtitle", "Compare normal and anomalous samples; inspect ground truth at pixel level.")))
        self.banner = synthetic_banner()
        self.banner.hide()
        lay.addWidget(self.banner)

        vsplit = QSplitter(Qt.Vertical)
        hsplit = QSplitter(Qt.Horizontal)
        self.browser = SampleBrowser(main.catalog)
        self.browser.sampleSelected.connect(self.show_sample)
        self.browser.indexChanged.connect(self._on_index)
        self.browser.setMinimumWidth(300)
        hsplit.addWidget(self.browser)

        center = QWidget()
        cl = QVBoxLayout(center)
        cl.setContentsMargins(0, 0, 0, 0)
        bar = QHBoxLayout()
        self.gray = QCheckBox(tr("exp.grayscale", "Grayscale"))
        self.show_gt = QCheckBox(tr("exp.show_gt", "Ground-truth overlay"))
        self.show_gt.setChecked(True)
        self.alpha = QSlider(Qt.Horizontal)
        self.alpha.setRange(0, 100)
        self.alpha.setValue(45)
        self.alpha.setMaximumWidth(140)
        fit = QPushButton(tr("exp.fit", "Fit"))
        export = QPushButton(tr("exp.export", "Export view…"))
        for w in (self.gray, self.show_gt, QLabel(tr("exp.opacity", "Opacity")), self.alpha, fit, export):
            bar.addWidget(w)
        bar.addStretch(1)
        self.pixel = muted("")
        bar.addWidget(self.pixel)
        cl.addLayout(bar)
        self.views = QStackedWidget()
        self.view = ImageView()
        self.pc = PointCloudView()
        self.views.addWidget(self.view)
        self.views.addWidget(self.pc)
        cl.addWidget(self.views, 1)
        hsplit.addWidget(center)

        self.meta = QTextBrowser()
        self.meta.setOpenExternalLinks(True)
        hsplit.addWidget(self.meta)
        hsplit.setSizes([330, 820, 300])
        vsplit.addWidget(hsplit)

        self.bottom = QWidget()
        self.bl = QHBoxLayout(self.bottom)
        self.bl.setSizeConstraint(self.bl.SizeConstraint.SetMinimumSize)
        self.bl.setContentsMargins(0, 4, 0, 0)
        from PySide6.QtWidgets import QScrollArea
        bscroll = QScrollArea()
        bscroll.setWidgetResizable(True)
        bscroll.setWidget(self.bottom)
        bscroll.setMinimumHeight(120)
        vsplit.addWidget(bscroll)
        vsplit.setStretchFactor(0, 4)
        vsplit.setStretchFactor(1, 1)
        vsplit.setSizes([760, 240])
        lay.addWidget(vsplit, 1)

        self.gray.toggled.connect(self._render)
        self.show_gt.toggled.connect(self._render)
        self.alpha.valueChanged.connect(self._render)
        fit.clicked.connect(self.view.fit)
        export.clicked.connect(self._export)
        self.view.pixelHovered.connect(self._on_pixel)
        if self.browser.index:
            self._on_index(self.browser.index)

    # ---------------------------------------------------------------- dataset
    def _on_index(self, idx):
        a = dreg.get(idx.dataset_id)
        self.banner.setVisible(bool(a and a.synthetic))
        while self.bl.count():
            w = self.bl.takeAt(0).widget()
            if w:
                w.deleteLater()
        rec = self.main.catalog.datasets.get(idx.dataset_id)
        if rec:
            desc = rec["description_es"] if settings.get().language == "es" else rec["description_en"]
            md = (f"**{rec['name']}** — {desc}\n\n**{tr('atlas.protocol', 'Benchmark protocol')}:** "
                  f"{rec.get('benchmark_protocol') or '-'}\n\n**{tr('exp.summary', 'Local index')}:** "
                  + ", ".join(f"{sp}: {d}" for sp, d in idx.summary().items()))
            self.bl.addWidget(markdown_browser(md), 3)
            self.bl.addWidget(ReferencesPanel(self.main.catalog, rec, "dataset"), 2)
        else:
            self.bl.addWidget(markdown_browser(
                "**SYNTHETIC demonstration dataset** generated by `core/datasets/synthetic.py` "
                "(procedural brushed-metal textures with drawn scratches, dents and stains; synthetic 3D "
                "height-field patches). It is not a research benchmark and has no publication. Use it "
                "only to learn the interface and the algorithms' behaviour.\n\n"
                + ", ".join(f"{sp}: {d}" for sp, d in idx.summary().items())))

    # ---------------------------------------------------------------- sample
    def show_sample(self, s: Sample):
        self.sample = s
        self.mask = None
        try:
            if "rgb" in s.modalities or "infrared" in s.modalities:
                path = s.modalities.get("rgb") or s.modalities["infrared"]
                self.raw = read_image_raw(path)
                self.rgb = to_display_rgb8(self.raw)
                mpaths = s.masks.get("rgb") or s.masks.get("infrared")
                if mpaths:
                    self.mask = read_mask(mpaths, self.rgb.shape[:2])
                self.views.setCurrentWidget(self.view)
                self._render()
                self.view.fit()
            elif "pointcloud" in s.modalities:
                pts, cols = read_point_cloud(s.modalities["pointcloud"])
                scal = {}
                gt = s.masks.get("pointcloud")
                if gt:
                    from core.datasets.real3d import read_point_labels
                    lab = read_point_labels(gt, pts)
                    if lab is not None:
                        scal[tr("pc.gt", "ground-truth label")] = lab
                warn = "" if s.meta.get("units") else tr("pc.units_warn", "⚠ units / registration not documented")
                self.pc.set_cloud(pts, cols, scal, warn)
                self.views.setCurrentWidget(self.pc)
        except Exception as e:
            self.meta.setPlainText(f"{tr('exp.load_error', 'Could not load sample')}: {e}")
            return
        self._meta(s)

    def _render(self):
        if self.rgb is None:
            return
        img = self.rgb
        if self.gray.isChecked():
            g = img.mean(axis=2).astype(np.uint8)
            img = np.stack([g] * 3, -1)
        if self.mask is not None and self.show_gt.isChecked():
            img = mask_overlay(img, self.mask, alpha=self.alpha.value() / 100)
        self.view.set_image(img, keep_view=True)

    def _meta(self, s: Sample):
        rows = [("Sample ID", s.sample_id), ("Split", s.split), ("Label", s.label),
                ("Defect type", s.defect_type or "—"), ("Category", s.category)]
        if self.raw is not None and self.views.currentWidget() is self.view:
            rows += [("Resolution", f"{self.raw.shape[1]} × {self.raw.shape[0]}"),
                     ("Stored dtype", f"{self.raw.dtype} · {1 if self.raw.ndim == 2 else self.raw.shape[2]} channel(s)")]
            if self.mask is not None:
                rows.append(("Defect area", f"{100 * self.mask.mean():.3f} % of pixels"))
        rows += [(f"File ({k})", Path(v).name) for k, v in s.modalities.items()]
        if s.label == "unknown":
            rows.append(("Note", "Label withheld by the benchmark (private test split). "
                                 "Do not compute metrics on these images."))
        for k, v in s.meta.items():
            rows.append((k, str(v)))
        self.meta.setHtml("<table cellpadding=3>" + "".join(f"<tr><td><b>{k}</b></td><td>{v}</td></tr>" for k, v in rows) + "</table>")

    def _on_pixel(self, x: int, y: int):
        if self.raw is None:
            return
        v = self.raw[y, x]
        gt = "" if self.mask is None else (" · GT: defect" if self.mask[y, x] else " · GT: normal")
        self.pixel.setText(f"x={x} y={y} value={v.tolist() if hasattr(v, 'tolist') else v}{gt}")

    def _export(self):
        if self.rgb is None or self.sample is None:
            return
        f, _ = QFileDialog.getSaveFileName(self, tr("exp.export", "Export view…"),
                                           str(Path.home() / f"{self.sample.sample_id.replace('/', '_')}.png"), "PNG (*.png)")
        if not f:
            return
        img = self.rgb if self.mask is None or not self.show_gt.isChecked() else mask_overlay(self.rgb, self.mask, alpha=self.alpha.value() / 100)
        write_image(f, img)
        rec = self.main.catalog.datasets.get(self.sample.dataset_id)
        attribution = (f"Source: {rec['name']} ({rec.get('official_website') or rec.get('paper_url')}); licence: "
                       f"{rec.get('license')}; sample {self.sample.sample_id}. Check the licence before sharing.") \
            if rec else "SYNTHETIC demonstration image generated by IDD Lab."
        Path(f).with_suffix(".attribution.txt").write_text(attribution + "\n", encoding="utf-8")
        self.main.statusBar().showMessage(f"Exported {f} (+ attribution file)", 6000)

    def apply_preset(self, dataset_id: str | None = None, **_):
        if dataset_id:
            self.browser.reload_datasets(select=dataset_id)

    def showEvent(self, e):
        super().showEvent(e)
        if self.browser.ds.count() == 0:
            self.browser.reload_datasets()

    def shutdown(self):
        self.pc.close_plotter()
