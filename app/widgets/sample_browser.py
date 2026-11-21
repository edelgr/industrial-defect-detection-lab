"""Dataset / category / split / label selectors + lazily loaded thumbnail gallery."""

from __future__ import annotations

import cv2
from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (QComboBox, QFormLayout, QLabel, QListWidget, QListWidgetItem,
                               QVBoxLayout, QWidget)

from app.i18n import tr
from app.widgets.common import muted, np_to_pixmap
from app.workers.task import start
from core.datasets import registry as dreg
from core.datasets.base import DatasetIndex, Sample
from core.datasets.io import read_rgb8

THUMB = 96


def installed_datasets(catalog) -> list[tuple[str, str]]:
    """(dataset_id, label) for every adapter with local data, synthetic last."""
    out = []
    for did, a in dreg.adapters().items():
        if dreg.installation_status(did) != "installed":
            continue
        rec = catalog.datasets.get(did)
        label = rec["acronym"] if rec else ("SYNTHETIC demo data" if a.synthetic else did)
        out.append((did, label))
    out.sort(key=lambda t: (dreg.get(t[0]).synthetic, t[0]))
    return out


class SampleBrowser(QWidget):
    sampleSelected = Signal(object)        # Sample
    indexChanged = Signal(object)          # DatasetIndex
    thumbsReady = Signal(object)           # list[(row, thumbnail)]

    def __init__(self, catalog, modality_filter: str | None = None, parent=None):
        super().__init__(parent)
        self.catalog = catalog
        self.modality_filter = modality_filter
        self.index: DatasetIndex | None = None
        self.samples: list[Sample] = []
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        form = QFormLayout()
        self.ds = QComboBox()
        self.cat = QComboBox()
        self.split = QComboBox()
        self.label = QComboBox()
        for k, v in (("all", tr("exp.all", "All")), ("good", tr("exp.normal", "Normal")),
                     ("bad", tr("exp.defective", "Defective")), ("unknown", tr("exp.unlabelled", "Unlabelled (withheld)"))):
            self.label.addItem(v, k)
        form.addRow(tr("exp.dataset", "Dataset"), self.ds)
        form.addRow(tr("exp.category", "Category"), self.cat)
        form.addRow(tr("exp.split", "Split"), self.split)
        form.addRow(tr("exp.label", "Label"), self.label)
        lay.addLayout(form)
        self.count = muted("")
        lay.addWidget(self.count)
        self.list = QListWidget()
        self.list.setViewMode(QListWidget.IconMode)
        self.list.setIconSize(QSize(THUMB, THUMB))
        self.list.setResizeMode(QListWidget.Adjust)
        self.list.setUniformItemSizes(True)
        self.list.setMovement(QListWidget.Static)
        self.list.setSpacing(4)
        lay.addWidget(self.list, 1)
        self.empty = QLabel()
        self.empty.setWordWrap(True)
        lay.addWidget(self.empty)
        self.ds.currentIndexChanged.connect(self._on_ds)
        self.cat.currentIndexChanged.connect(self._on_cat)
        self.split.currentIndexChanged.connect(self._fill)
        self.label.currentIndexChanged.connect(self._fill)
        self.list.currentItemChanged.connect(self._on_item)
        self._thumb_task = None
        self._thumb_gen = 0
        self.thumbsReady.connect(self._apply_thumbs)
        self.reload_datasets()

    def reload_datasets(self, select: str | None = None):
        self.ds.blockSignals(True)
        self.ds.clear()
        for did, lab in installed_datasets(self.catalog):
            a = dreg.get(did)
            if self.modality_filter and self.modality_filter not in a.modalities:
                continue
            self.ds.addItem(lab, did)
        self.ds.blockSignals(False)
        if self.ds.count() == 0:
            self.empty.setText(tr("exp.none_installed", "No dataset installed. Import one in Settings → Dataset "
                                  "Manager, or generate the SYNTHETIC demo set there."))
        else:
            self.empty.setText("")
        i = self.ds.findData(select) if select else 0
        self.ds.setCurrentIndex(max(i, 0))
        self._on_ds()

    def _on_ds(self):
        did = self.ds.currentData()
        self.cat.blockSignals(True)
        self.cat.clear()
        if did:
            for c in dreg.installed_categories(did):
                self.cat.addItem(c, c)
        self.cat.blockSignals(False)
        self._on_cat()

    def _on_cat(self):
        did, c = self.ds.currentData(), self.cat.currentData()
        self.index = None
        if did and c:
            try:
                self.index = dreg.load_index(did, c)
            except Exception as e:
                self.empty.setText(f"{tr('exp.index_error', 'Could not index dataset')}: {e}")
        self.split.blockSignals(True)
        self.split.clear()
        self.split.addItem(tr("exp.all", "All"), None)
        if self.index:
            for s in self.index.splits:
                self.split.addItem(s, s)
        self.split.blockSignals(False)
        if self.index:
            self.indexChanged.emit(self.index)
        self._fill()

    def _fill(self):
        if self._thumb_task:
            self._thumb_task.cancel()
        self.list.clear()
        if not self.index:
            self.samples = []
            self.count.setText("")
            return
        sp, lab = self.split.currentData(), self.label.currentData()
        self.samples = [s for s in self.index.samples
                        if (sp is None or s.split == sp) and (lab == "all" or s.label == lab)
                        and (self.modality_filter is None or self.modality_filter in s.modalities)]
        self.count.setText(f"{len(self.samples)} {tr('exp.samples', 'samples')}"
                           + (f" · ⚠ {len(self.index.warnings)} {tr('exp.warnings', 'index warnings')}" if self.index.warnings else ""))
        for s in self.samples:
            it = QListWidgetItem(f"{s.defect_type or s.label}\n{s.sample_id.split('/')[-1]}")
            it.setData(Qt.UserRole, s)
            it.setToolTip(f"{s.split} · {s.label} · {s.defect_type or '-'}\n{s.sample_id}")
            it.setSizeHint(QSize(THUMB + 16, THUMB + 36))
            self.list.addItem(it)
        items = [(i, s) for i, s in enumerate(self.samples) if "rgb" in s.modalities or "infrared" in s.modalities]
        self._thumb_gen += 1
        if items:
            self._thumb_task = start(self._load_thumbs, items=items, gen=self._thumb_gen)

    def _load_thumbs(self, items, gen, progress, cancel):
        # Runs in a worker thread; batches are handed to the GUI thread through a queued signal.
        out = []
        for k, (i, s) in enumerate(items):
            if cancel():
                return None
            try:
                img = read_rgb8(s.modalities.get("rgb") or s.modalities["infrared"])
                h, w = img.shape[:2]
                sc = THUMB / max(h, w)
                out.append((i, cv2.resize(img, (max(1, int(w * sc)), max(1, int(h * sc))), interpolation=cv2.INTER_AREA)))
            except Exception:
                continue
            if len(out) >= 24 or k == len(items) - 1:
                batch, out = out, []
                self._deliver((gen, batch))
        return True

    def _deliver(self, batch):
        try:
            self.thumbsReady.emit(batch)      # emitted from the worker -> queued to the GUI thread
        except RuntimeError:                  # widget destroyed while thumbnails were loading
            pass

    def _apply_thumbs(self, payload):
        gen, batch = payload
        if gen != self._thumb_gen:        # stale batch from a previous filter
            return
        for i, thumb in batch:
            if i < self.list.count():
                self.list.item(i).setIcon(QIcon(np_to_pixmap(thumb)))

    def _on_item(self, cur, _prev):
        if cur is not None:
            self.sampleSelected.emit(cur.data(Qt.UserRole))

    def select_dataset(self, dataset_id: str):
        i = self.ds.findData(dataset_id)
        if i >= 0:
            self.ds.setCurrentIndex(i)

    def current_sample(self) -> Sample | None:
        it = self.list.currentItem()
        return it.data(Qt.UserRole) if it else None
