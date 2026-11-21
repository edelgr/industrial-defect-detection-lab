"""Interactive 3D point-cloud viewer (PyVistaQt / VTK) with scalar colouring."""

from __future__ import annotations

import numpy as np
from PySide6.QtWidgets import (QCheckBox, QComboBox, QHBoxLayout, QLabel, QPushButton, QSpinBox,
                               QVBoxLayout, QWidget)

from app.i18n import tr
from app.widgets.common import muted


def pyvista_available() -> tuple[bool, str]:
    try:
        import pyvista  # noqa: F401
        import pyvistaqt  # noqa: F401
        return True, "ok"
    except Exception as e:
        return False, f"PyVista/PyVistaQt not available: {e}"


class PointCloudView(QWidget):
    """Shows N×3 points coloured by height, RGB, an anomaly score or ground-truth labels."""

    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        ok, why = pyvista_available()
        self.plotter = None
        self.points: np.ndarray | None = None
        self.scalars: dict[str, np.ndarray] = {}
        self.rgb: np.ndarray | None = None
        bar = QHBoxLayout()
        self.color_by = QComboBox()
        self.color_by.addItem(tr("pc.height", "Height (z)"), "z")
        self.point_size = QSpinBox()
        self.point_size.setRange(1, 15)
        self.point_size.setValue(3)
        self.eye_dome = QCheckBox(tr("pc.edl", "Eye-dome shading"))
        reset = QPushButton(tr("pc.reset", "Reset camera"))
        bar.addWidget(QLabel(tr("pc.color_by", "Colour by")))
        bar.addWidget(self.color_by)
        bar.addWidget(QLabel(tr("pc.size", "Point size")))
        bar.addWidget(self.point_size)
        bar.addWidget(self.eye_dome)
        bar.addWidget(reset)
        bar.addStretch(1)
        self.info = muted("")
        lay.addLayout(bar)
        if not ok:
            lay.addWidget(muted(why))
            return
        from pyvistaqt import QtInteractor
        self.plotter = QtInteractor(self)
        self.plotter.set_background("#11161c", top="#1f2a36")
        lay.addWidget(self.plotter.interactor, 1)
        lay.addWidget(self.info)
        self.color_by.currentIndexChanged.connect(self._render)
        self.point_size.valueChanged.connect(self._render)
        self.eye_dome.toggled.connect(self._render)
        reset.clicked.connect(lambda: self.plotter.reset_camera())

    def set_cloud(self, points: np.ndarray, rgb: np.ndarray | None = None,
                  scalars: dict[str, np.ndarray] | None = None, info: str = "") -> None:
        self.points = np.asarray(points, dtype=np.float32)
        self.rgb = rgb
        self.scalars = {"z": self.points[:, 2]}
        self.scalars.update(scalars or {})
        cur = self.color_by.currentData()
        self.color_by.blockSignals(True)
        self.color_by.clear()
        self.color_by.addItem(tr("pc.height", "Height (z)"), "z")
        if rgb is not None:
            self.color_by.addItem(tr("pc.rgb", "Stored colour"), "__rgb__")
        for k in (scalars or {}):
            self.color_by.addItem(k, k)
        i = self.color_by.findData(cur)
        self.color_by.setCurrentIndex(i if i >= 0 else (self.color_by.count() - 1 if scalars else 0))
        self.color_by.blockSignals(False)
        ext = self.points.max(0) - self.points.min(0)
        self.info.setText(f"{len(self.points):,} points · extent {ext[0]:.3g} × {ext[1]:.3g} × {ext[2]:.3g} "
                          f"(units as stored in the file) {info}")
        self._render(reset=True)

    def _render(self, *_, reset: bool = False) -> None:
        if self.plotter is None or self.points is None:
            return
        import pyvista as pv
        self.plotter.clear()
        cloud = pv.PolyData(self.points)
        key = self.color_by.currentData()
        kw = dict(point_size=self.point_size.value(), render_points_as_spheres=True)
        if key == "__rgb__" and self.rgb is not None:
            cloud["rgb"] = (np.clip(self.rgb, 0, 1) * 255).astype(np.uint8)
            self.plotter.add_mesh(cloud, scalars="rgb", rgb=True, **kw)
        else:
            s = self.scalars.get(key, self.points[:, 2])
            cloud["value"] = s
            cmap = "coolwarm" if key and ("label" in key.lower() or "truth" in key.lower()) else "inferno" if key != "z" else "viridis"
            self.plotter.add_mesh(cloud, scalars="value", cmap=cmap,
                                  scalar_bar_args={"title": str(key), "fmt": "%.4g", "n_labels": 3,
                                                   "title_font_size": 14, "label_font_size": 12}, **kw)
        if self.eye_dome.isChecked():
            self.plotter.enable_eye_dome_lighting()
        else:
            self.plotter.disable_eye_dome_lighting()
        if reset:
            self.plotter.reset_camera()
        self.plotter.render()

    def close_plotter(self):
        if self.plotter is not None:
            self.plotter.close()
