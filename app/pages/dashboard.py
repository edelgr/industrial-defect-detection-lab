"""Dashboard: overview, readiness status and entry points."""

from __future__ import annotations

from PySide6.QtWidgets import QGridLayout, QHBoxLayout, QPushButton, QVBoxLayout, QWidget

from app.i18n import tr
from app.widgets.common import Card, PageHeader, markdown_browser, muted, title_label
from core.algorithms import registry as areg
from core.datasets import registry as dreg
from core.research import content


class DashboardPage(QWidget):
    def __init__(self, main):
        super().__init__()
        self.main = main
        cat = main.catalog
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(PageHeader("Industrial Defect Detection Lab",
                                 tr("app.subtitle", "Interactive Learning and Benchmarking for Industrial Defect and Anomaly Detection")))
        grid = QGridLayout()
        grid.setSpacing(12)

        installed = [d for d in cat.datasets if dreg.get(d) and dreg.installation_status(d) == "installed"]
        runnable = [a for a in cat.algorithms if areg.availability(a)[0]]
        stats = [
            (tr("dash.datasets", "Research datasets catalogued"), str(len(cat.datasets)),
             f"{len(installed)} {tr('dash.installed_locally', 'installed locally')}", "atlas"),
            (tr("dash.methods", "Methods documented"), str(len(cat.algorithms)),
             f"{len(runnable)} {tr('dash.runnable', 'runnable on this machine')}", "library"),
            (tr("dash.publications", "Publications referenced"), str(len(cat.publications)),
             tr("dash.bibtex", "BibTeX available for every entry"), "atlas"),
            (tr("dash.workshop", "Guided workshop"), "3 h", tr("dash.activities", "7 activities, CIARP 2026"), "workshop"),
        ]
        for i, (t, big, sub, target) in enumerate(stats):
            c = Card()
            c.lay.addWidget(muted(t))
            c.lay.addWidget(title_label(big, "PageTitle"))
            c.lay.addWidget(muted(sub))
            b = QPushButton(tr("dash.open", "Open"))
            b.clicked.connect(lambda _=False, k=target: main.navigate(k))
            c.lay.addWidget(b)
            grid.addWidget(c, 0, i)
        lay.addLayout(grid)

        if cat.problems:
            warn = Card()
            warn.lay.addWidget(title_label(f"⚠ {len(cat.problems)} catalog validation problems"))
            warn.lay.addWidget(muted("\n".join(cat.problems[:6])))
            lay.addWidget(warn)

        row = QHBoxLayout()
        row.addWidget(markdown_browser(content.markdown("general", "welcome") or ""), 3)
        quick = Card()
        quick.lay.addWidget(title_label(tr("dash.quick", "Quick start")))
        for label, key in ((tr("nav.atlas", "Dataset Research Atlas"), "atlas"),
                           (tr("nav.explorer", "Dataset Explorer"), "explorer"),
                           (tr("nav.library", "Methods Library"), "library"),
                           (tr("nav.lab", "Methods Laboratory"), "lab"),
                           (tr("nav.compare", "Compare Methods"), "compare"),
                           (tr("nav.lab3d", "3D & Multimodal Lab"), "lab3d"),
                           (tr("nav.workshop", "Guided Workshop"), "workshop"),
                           (tr("dash.readiness", "Offline readiness check"), "settings")):
            b = QPushButton("→  " + label)
            b.clicked.connect(lambda _=False, k=key: main.navigate(k))
            quick.lay.addWidget(b)
        quick.lay.addStretch(1)
        row.addWidget(quick, 1)
        lay.addLayout(row, 1)
