"""'How This Algorithm Works' — step-through explanation widget.

Steps come from education/<lang>/steps.yaml.  Each step may reference a diagram in
education/diagrams/ (SVG/PNG).  Diagrams are explicitly labelled CONCEPTUAL: they are not real
internal activations of a model.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtSvgWidgets import QSvgWidget
from PySide6.QtWidgets import (QCheckBox, QHBoxLayout, QLabel, QPushButton, QStackedWidget,
                               QVBoxLayout, QWidget)

from app.i18n import tr
from app.widgets.common import Badge, markdown_browser, muted, open_url
from core import paths
from core.research import content


class StepsWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.steps: list[dict] = []
        self.i = 0
        self.record: dict | None = None
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        top = QHBoxLayout()
        self.title = QLabel()
        self.title.setObjectName("CardTitle")
        self.counter = Badge("")
        self.kind = Badge(tr("steps.conceptual", "conceptual diagram"), "warn")
        top.addWidget(self.title, 1)
        top.addWidget(self.kind)
        top.addWidget(self.counter)
        lay.addLayout(top)
        self.diagram_host = QStackedWidget()
        self.svg = QSvgWidget()
        self.img = QLabel()
        self.img.setAlignment(Qt.AlignCenter)
        self.none = muted(tr("steps.no_diagram", "(no diagram for this step)"))
        self.none.setAlignment(Qt.AlignCenter)
        for w in (self.svg, self.img, self.none):
            self.diagram_host.addWidget(w)
        self.diagram_host.setMinimumHeight(230)
        lay.addWidget(self.diagram_host, 3)
        self.text = markdown_browser()
        lay.addWidget(self.text, 2)
        ctrl = QHBoxLayout()
        self.prev = QPushButton("◀ " + tr("steps.prev", "Previous step"))
        self.next = QPushButton(tr("steps.next", "Next step") + " ▶")
        self.next.setObjectName("Primary")
        self.reset = QPushButton("⟲ " + tr("steps.reset", "Reset"))
        self.math = QCheckBox(tr("steps.math", "Show mathematical explanation"))
        self.impl = QCheckBox(tr("steps.impl", "Show implementation details"))
        self.paper = QPushButton("↗ " + tr("steps.paper", "Open original paper"))
        self.code = QPushButton("↗ " + tr("steps.code", "Open original code repository"))
        # three short rows so the controls fit on 1280-px (150 % scaled) laptop screens
        for w in (self.prev, self.next, self.reset):
            ctrl.addWidget(w)
        ctrl.addStretch(1)
        lay.addLayout(ctrl)
        opts = QVBoxLayout()                      # stacked: Spanish labels are long
        opts.addWidget(self.math)
        opts.addWidget(self.impl)
        links = QHBoxLayout()
        links.addWidget(self.paper)
        links.addWidget(self.code)
        links.addStretch(1)
        lay.addLayout(links)
        lay.addLayout(opts)
        self.prev.clicked.connect(lambda: self.go(self.i - 1))
        self.next.clicked.connect(lambda: self.go(self.i + 1))
        self.reset.clicked.connect(lambda: self.go(0))
        self._urls: dict = {}
        self.paper.clicked.connect(lambda: self._urls.get("paper") and open_url(self._urls["paper"]))
        self.code.clicked.connect(lambda: self._urls.get("code") and open_url(self._urls["code"]))
        self.math.toggled.connect(lambda _: self.go(self.i))
        self.impl.toggled.connect(lambda _: self.go(self.i))

    def load(self, record: dict) -> bool:
        self.record = record
        d = content.algorithm_steps(record["id"])
        self.steps = (d or {}).get("steps", [])
        paper = record.get("paper_url") or (f"https://doi.org/{record['doi']}" if record.get("doi") else None)
        code = record.get("official_repository_url") or record.get("implementation_repository_url")
        self._urls = {"paper": paper, "code": code}
        for b, u in ((self.paper, paper), (self.code, code)):
            b.setEnabled(bool(u))
            b.setToolTip(u or tr("refs.not_available", "Not available"))
        self.go(0)
        return bool(self.steps)

    def go(self, i: int) -> None:
        if not self.steps:
            self.title.setText(tr("steps.none", "No step-by-step explanation for this method yet."))
            self.counter.setText("")
            self.text.setMarkdown("")
            self.diagram_host.setCurrentWidget(self.none)
            for b in (self.prev, self.next, self.reset):
                b.setEnabled(False)
            return
        self.i = max(0, min(i, len(self.steps) - 1))
        s = self.steps[self.i]
        self.title.setText(f"{self.i + 1}. {s.get('title', '')}")
        self.counter.setText(f"{self.i + 1} / {len(self.steps)}")
        self.kind.setText(tr("steps.conceptual", "conceptual diagram") if s.get("diagram_kind", "conceptual") == "conceptual"
                          else tr("steps.real", "real data"))
        md = s.get("text", "")
        if self.math.isChecked() and s.get("math"):
            md += "\n\n**" + tr("steps.math_h", "Mathematics") + "**\n\n" + s["math"]
        if self.impl.isChecked() and s.get("implementation"):
            md += "\n\n**" + tr("steps.impl_h", "Implementation") + "**\n\n" + s["implementation"]
        self.text.setMarkdown(md)
        diag = s.get("diagram")
        f = paths.EDUCATION / "diagrams" / diag if diag else None
        if f and f.exists():
            if f.suffix.lower() == ".svg":
                self.svg.load(str(f))
                self.diagram_host.setCurrentWidget(self.svg)
            else:
                self.img.setPixmap(QPixmap(str(f)).scaledToHeight(240, Qt.SmoothTransformation))
                self.diagram_host.setCurrentWidget(self.img)
        else:
            self.diagram_host.setCurrentWidget(self.none)
        self.prev.setEnabled(self.i > 0)
        self.next.setEnabled(self.i < len(self.steps) - 1)
        self.reset.setEnabled(True)
