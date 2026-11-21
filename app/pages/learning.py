"""Start Learning — the Student Learning Center: nine guided topics with explanations, demos inside
IDD Lab, exercises, reflection questions, and direct links to papers, datasets and code."""

from __future__ import annotations

from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (QFrame, QHBoxLayout, QLabel, QListWidget, QListWidgetItem, QPushButton,
                               QScrollArea, QSplitter, QVBoxLayout, QWidget)

from app.i18n import tr
from app.widgets.common import PageHeader, markdown_browser, muted
from app.widgets.references import LinkAction, PublicationRow
from core import paths
from core.research import content


def learning() -> dict:
    return content.yaml_content("learning.yaml") or {}


def topic_markdown(t: dict) -> str:
    def bl(xs):
        return "\n".join(f"- {x}" for x in xs or [])
    return "\n\n".join([
        f"## {t['title']}", f"*≈ {t.get('minutes', '?')} min*",
        f"### {tr('learn.explanation', 'In short')}", t.get("explanation", ""),
        f"### {tr('learn.example', 'Intuitive example')}", t.get("example", ""),
        f"### {tr('learn.background', 'Scientific background')}", t.get("background", ""),
        f"### {tr('learn.exercise', 'Practical exercise')}",
        (t.get("exercise") or {}).get("title", "—") + "  \n" + tr("learn.exercise_hint", "Use the button below to open the exercise sheet."),
        f"### {tr('learn.reflection', 'Questions for reflection')}", bl(t.get("reflection")),
    ])


EXERCISE_FILES = {"EX1": "EXERCISE_1_CLASSICAL.md", "EX2": "EXERCISE_2_PATCHCORE.md", "EX3": "EXERCISE_3_COMPARING.md",
                  "EX4": "EXERCISE_4_RESOLUTION.md", "EX5": "EXERCISE_5_ILLUMINATION.md", "EX6": "EXERCISE_6_METRICS.md",
                  "EX7": "EXERCISE_7_MULTIMODAL.md", "EX8": "EXERCISE_8_DECISIONS.md"}


def exercise_path(ex_id: str):
    f = EXERCISE_FILES.get(ex_id)
    return paths.DOCS / "exercises" / f if f else None


class LearningPage(QWidget):
    def __init__(self, main):
        super().__init__()
        self.main = main
        data = learning()
        self.topics = data.get("topics", [])
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(PageHeader(data.get("title", tr("nav.learning", "Start Learning")), data.get("subtitle", "")))
        sp = QSplitter(Qt.Horizontal)
        self.list = QListWidget()
        self.list.setWordWrap(True)
        self.list.setMinimumWidth(220)
        for t in self.topics:
            it = QListWidgetItem(f"{t['title']}\n      ≈ {t.get('minutes', '?')} min")
            self.list.addItem(it)
        sp.addWidget(self.list)
        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)
        self.text = markdown_browser(data.get("intro", ""))
        rl.addWidget(self.text, 3)
        self.actions = QHBoxLayout()
        self.actions_box = QWidget()
        self.actions_box.setLayout(self.actions)
        rl.addWidget(self.actions_box)
        self.refs_inner = QWidget()
        self.refs = QVBoxLayout(self.refs_inner)
        self.refs.setSizeConstraint(QVBoxLayout.SizeConstraint.SetMinimumSize)   # rows must not overlap
        sa = QScrollArea()
        sa.setWidgetResizable(True)
        sa.setFrameShape(QFrame.NoFrame)
        sa.setWidget(self.refs_inner)
        sa.setMinimumHeight(180)
        rl.addWidget(sa, 2)
        sp.addWidget(right)
        sp.setSizes([300, 1000])
        lay.addWidget(sp, 1)
        self.list.currentRowChanged.connect(self._show)

    def _clear(self, layout):
        while layout.count():
            item = layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.hide()
                w.setParent(None)
                w.deleteLater()

    def _show(self, row: int):
        if row < 0:
            return
        t = self.topics[row]
        self.text.setMarkdown(topic_markdown(t))
        self._clear(self.actions)
        # wrap the action buttons over several rows (1280-px screens)
        col = QVBoxLayout()
        for a in t.get("demo", []):
            b = QPushButton("→ " + a["label"])
            b.setObjectName("Primary")
            b.clicked.connect(lambda _=False, x=a: self.main.navigate(x["page"], **x.get("preset", {})))
            col.addWidget(b)
        ex = (t.get("exercise") or {}).get("id")
        p = exercise_path(ex) if ex else None
        if p is not None:
            b = QPushButton("📄 " + tr("learn.open_exercise", "Open exercise sheet") + f" ({ex})")
            b.setEnabled(p.exists())
            b.setToolTip(str(p))
            b.clicked.connect(lambda _=False, f=p: QDesktopServices.openUrl(QUrl.fromLocalFile(str(f))))
            col.addWidget(b)
        holder = QWidget()
        holder.setLayout(col)
        self.actions.addWidget(holder)
        self._fill_refs(t)

    def _fill_refs(self, t: dict):
        self._clear(self.refs)
        cat = self.main.catalog
        pubs = [cat.publications[p] for p in t.get("papers", []) if p in cat.publications]
        if pubs:
            self.refs.addWidget(QLabel("<b>" + tr("learn.papers", "Recommended papers") + "</b>"))
            for p in pubs:
                self.refs.addWidget(PublicationRow(cat, p))
        algs = [cat.algorithms[a] for a in t.get("algorithms", []) if a in cat.algorithms]
        if algs:
            self.refs.addWidget(QLabel("<b>" + tr("learn.code", "Code repositories") + "</b>"))
            for a in algs:
                self.refs.addWidget(LinkAction(cat, f"{a['name']} — " + tr("learn.authors_code", "authors' repository"),
                                               a.get("official_repository_url"),
                                               tr("refs.no_official_code", "No official authors' code verified")))
                impl = a.get("implementation_repository_url")
                if impl:
                    self.refs.addWidget(LinkAction(cat, f"{a['name']} — " + tr("learn.impl_used", "implementation used in IDD Lab")
                                                   + f" ({a.get('implementation_type')})", impl, ""))
        dss = [cat.datasets[d] for d in t.get("datasets", []) if d in cat.datasets]
        if dss:
            self.refs.addWidget(QLabel("<b>" + tr("learn.datasets", "Datasets (official access pages)") + "</b>"))
            for d in dss:
                self.refs.addWidget(LinkAction(cat, f"{d['name']} — " + tr("refs.website", "Open Official Dataset Website"),
                                               d.get("official_website") or d.get("download_url"),
                                               tr("refs.no_website", "No official website")))
        self.refs.addWidget(muted(tr("learn.refs_note", "All links come from the verified research catalog "
                                                        "(research_catalog/); BibTeX can be copied offline.")))
        self.refs.addStretch(1)

    def apply_preset(self, topic: str | None = None, **_):
        if topic:
            for i, t in enumerate(self.topics):
                if t["id"] == topic:
                    self.list.setCurrentRow(i)
