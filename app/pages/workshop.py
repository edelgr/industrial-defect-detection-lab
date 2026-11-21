"""Guided Workshop Mode — the 3-hour CIARP 2026 session, activity by activity."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QHBoxLayout, QListWidget, QListWidgetItem, QPushButton, QSplitter,
                               QVBoxLayout, QWidget)

from app.i18n import tr
from app.widgets.common import PageHeader, markdown_browser, muted
from core.research import content


def activity_markdown(a: dict, catalog) -> str:
    def bl(xs):
        return "\n".join(f"- {x}" for x in xs or [])
    refs = []
    for pid in a.get("references", []):
        p = catalog.publications.get(pid)
        if p:
            url = p.get("paper_url") or (f"https://doi.org/{p['doi']}" if p.get("doi") else "")
            refs.append(f"- [{p['title']}]({url}) — {', '.join(p['authors'][:3])}{' et al.' if len(p['authors']) > 3 else ''}, {p['venue']} {p['year']}")
    return "\n\n".join([
        f"## {a['title']}", f"**{tr('ws.duration', 'Duration')}:** {a['duration_min']} min",
        f"### {tr('ws.objectives', 'Learning objectives')}", bl(a.get("objectives")),
        f"### {tr('ws.theory', 'Brief theory')}", a.get("theory", ""),
        f"### {tr('ws.setup', 'Setup')}",
        f"- **Dataset:** {a.get('dataset', '')}\n- **Algorithm(s):** {a.get('algorithm', '')}\n- **Starting parameters:** {a.get('parameters', '')}",
        f"### {tr('ws.instructions', 'Instructions')}", "\n".join(f"{i + 1}. {s}" for i, s in enumerate(a.get("instructions", []))),
        f"### {tr('ws.observations', 'Expected qualitative observations')}", bl(a.get("observations")),
        *([f"### {tr('ws.results', 'Real results (Workshop Verified, test_public)')}", _results(a["results_from"]),
           f"### {tr('ws.interpretation', 'Interpretation')}", tr('ws.interp_note',
               "Local educational results on the public labelled test split, re-evaluated from the verified "
               "checkpoints (configs/workshop_verified.json). They are not official MVTec AD 2 scores.")]
          if a.get("results_from") else []),
        *([f"### {tr('ws.limitations', 'Limitations')}", bl(a.get("limitations"))] if a.get("limitations") else []),
        f"### {tr('ws.reflection', 'Reflection questions')}", bl(a.get("reflection")),
        f"### {tr('ws.extension', 'Optional extension')}", a.get("extension", ""),
        f"### {tr('ws.references', 'Scientific references')}", "\n".join(refs) or "—",
    ])


def _results(kind: str) -> str:
    try:
        from app.pages.experiments import results_markdown
        return results_markdown(kind)
    except Exception as e:  # noqa: BLE001 — never break the workshop page
        return f"_Results unavailable: {e}_"


class WorkshopPage(QWidget):
    def __init__(self, main):
        super().__init__()
        self.main = main
        data = content.workshop()
        self.acts = data.get("activities", [])
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(PageHeader(data.get("title", tr("nav.workshop", "Guided Workshop")), data.get("subtitle", "")))
        sp = QSplitter(Qt.Horizontal)
        left = QWidget()
        ll = QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 0, 0)
        self.list = QListWidget()
        total = 0
        for a in self.acts:
            total += a["duration_min"]
            it = QListWidgetItem(f"{a['id']}  {a['title']}\n      {a['duration_min']} min")
            self.list.addItem(it)
        ll.addWidget(self.list, 1)
        ll.addWidget(muted(f"{tr('ws.total', 'Total')}: {total} min"))
        rb = QPushButton(tr("ws.readiness", "Run offline readiness check"))
        rb.clicked.connect(lambda: (main.navigate("settings", tab="diagnostics"), main.page("settings").diag._ready()))
        ll.addWidget(rb)
        sp.addWidget(left)
        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)
        self.text = markdown_browser(data.get("intro", ""))
        rl.addWidget(self.text, 1)
        self.actions = QHBoxLayout()
        rl.addLayout(self.actions)
        sp.addWidget(right)
        sp.setSizes([330, 1100])
        lay.addWidget(sp, 1)
        self.list.currentRowChanged.connect(self._show)

    def _show(self, row: int):
        if row < 0:
            return
        a = self.acts[row]
        self.text.setMarkdown(activity_markdown(a, self.main.catalog))
        while self.actions.count():
            w = self.actions.takeAt(0).widget()
            if w:
                w.deleteLater()
        for act in a.get("actions", []):
            b = QPushButton("→ " + act["label"])
            b.setObjectName("Primary")
            b.clicked.connect(lambda _=False, x=act: self.main.navigate(x["page"], **x.get("preset", {})))
            self.actions.addWidget(b)
        self.actions.addStretch(1)
