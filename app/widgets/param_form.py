"""Auto-generated parameter editor for a method's ParamSpec list."""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout, QSpinBox, QWidget

from core.algorithms.base import ParamSpec


class ParamForm(QWidget):
    changed = Signal(str)          # name of the parameter that changed

    def __init__(self, parent=None):
        super().__init__(parent)
        self.form = QFormLayout(self)
        self.form.setContentsMargins(0, 0, 0, 0)
        # long labels go above their field instead of widening the page (1280-px laptop screens)
        self.form.setRowWrapPolicy(QFormLayout.WrapLongRows)
        self.widgets: dict[str, QWidget] = {}
        self.specs: list[ParamSpec] = []

    def set_specs(self, specs: list[ParamSpec], values: dict | None = None) -> None:
        while self.form.rowCount():
            self.form.removeRow(0)
        self.widgets.clear()
        self.specs = specs
        values = values or {}
        for p in specs:
            v = values.get(p.name, p.default)
            if p.kind == "int":
                w = QSpinBox()
                w.setRange(int(p.minimum if p.minimum is not None else -10 ** 6), int(p.maximum if p.maximum is not None else 10 ** 6))
                w.setSingleStep(int(p.step or 1))
                w.setValue(int(v))
                w.valueChanged.connect(lambda _=0, n=p.name: self.changed.emit(n))
            elif p.kind == "float":
                w = QDoubleSpinBox()
                w.setDecimals(3)
                w.setRange(float(p.minimum if p.minimum is not None else -1e9), float(p.maximum if p.maximum is not None else 1e9))
                w.setSingleStep(float(p.step or 0.1))
                w.setValue(float(v))
                w.valueChanged.connect(lambda _=0, n=p.name: self.changed.emit(n))
            elif p.kind == "bool":
                w = QCheckBox()
                w.setChecked(bool(v))
                w.toggled.connect(lambda _=0, n=p.name: self.changed.emit(n))
            else:
                w = QComboBox()
                for c in p.choices or []:
                    w.addItem(str(c), c)
                w.setCurrentIndex(max(0, w.findData(v)))
                w.currentIndexChanged.connect(lambda _=0, n=p.name: self.changed.emit(n))
            tip = p.help + ("  [changing it requires re-fitting]" if p.affects_fit else "")
            w.setToolTip(tip.strip())
            self.widgets[p.name] = w
            self.form.addRow((p.label or p.name) + (" *" if p.affects_fit else ""), w)

    def values(self) -> dict:
        out = {}
        for p in self.specs:
            w = self.widgets[p.name]
            if isinstance(w, QSpinBox):
                out[p.name] = w.value()
            elif isinstance(w, QDoubleSpinBox):
                out[p.name] = w.value()
            elif isinstance(w, QCheckBox):
                out[p.name] = w.isChecked()
            else:
                out[p.name] = w.currentData()
        return out
