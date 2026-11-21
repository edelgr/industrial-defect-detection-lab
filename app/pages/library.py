"""Methods Library — scientific explanation of each method family and algorithm."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QHBoxLayout, QLabel, QLayout, QLineEdit, QPushButton, QScrollArea, QSplitter,
                               QTabWidget, QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget)

from app.i18n import tr
from app.widgets.common import Badge, Card, PageHeader, clear_layout, markdown_browser, title_label
from app.widgets.references import ReferencesPanel
from app.widgets.steps import StepsWidget
from core import settings
from core.algorithms import registry as areg
from core.research import content

GROUPS = {
    "A": "A — Classical computer vision",
    "B": "B — Supervised learning",
    "C": "C — Reconstruction",
    "D": "D — Feature distributions & memory banks",
    "E": "E — Teacher–Student",
    "F": "F — Foundation models",
    "G": "G — 3D & multimodal",
}
STATUS_LABEL = {
    "explanation_only": ("Educational explanation only", ""),
    "runnable_classical": ("Runnable classical implementation", "ok"),
    "runnable_pretrained": ("Runnable pretrained model", "ok"),
    "requires_fitting": ("Runnable — requires fitting on normal images", "ok"),
    "missing_checkpoint": ("Missing compatible checkpoint", "warn"),
    "unavailable_dependencies": ("Unavailable due to dependencies", "bad"),
    "validated_local_dataset": ("Validated on local dataset", "ok"),
}


def method_markdown(r: dict) -> str:
    es = settings.get().language == "es"
    intuitive = r["intuitive_explanation_es"] if es else r["intuitive_explanation_en"]
    def bl(xs):
        return "\n".join(f"- {x}" for x in xs) if xs else "_—_"
    steps = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(r["main_processing_steps"]))
    parts = [
        f"## {r['name']}",
        f"*{r['family']}* · {r.get('venue') or ''} {r.get('year') or ''}",
        f"### {tr('lib.q1', 'What problem does this method solve?')}", r.get("industrial_use") or r["scientific_principle"].split(".")[0] + ".",
        f"### {tr('lib.q2', 'Underlying idea (intuitive)')}", intuitive,
        f"### {tr('lib.q3', 'How does it work?')}", r["scientific_principle"], "", steps,
        f"### {tr('lib.q4', 'Training data required')}", r["training_data_requirements"],
        f"### {tr('lib.q5', 'Inputs and outputs')}", f"**Inputs:** {r['inputs']}\n\n**Outputs:** {r['outputs']}",
        f"### {tr('lib.q6', 'How is an anomaly identified?')}", r.get("mathematical_formulation") or "_see paper_",
        f"### {tr('lib.q7', 'Strengths')}", bl(r["advantages"]),
        f"### {tr('lib.q8', 'Limitations')}", bl(r["limitations"]),
        f"### {tr('lib.failures', 'Typical failure cases')}", bl(r["failure_cases"]),
        f"### {tr('lib.q9', 'When to use it in industry?')}", r.get("industrial_use") or "",
        f"### {tr('lib.compute', 'Computational requirements')}", r["computational_requirements"],
        f"### {tr('lib.exercises', 'Exercises')}", bl(r["educational_exercises"]),
    ]
    extra = content.markdown("methods", r["id"])
    if extra:
        parts += ["---", extra]
    return "\n\n".join(parts)


class LibraryPage(QWidget):
    def __init__(self, main):
        super().__init__()
        self.main = main
        cat = main.catalog
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(PageHeader(tr("nav.library", "Methods Library"),
                                 tr("lib.subtitle", "Main families of industrial defect and anomaly detection — idea, mechanism, limits and sources.")))
        split = QSplitter(Qt.Horizontal)
        left = QWidget()
        ll = QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 0, 0)
        self.search = QLineEdit()
        self.search.setPlaceholderText(tr("lib.search", "Search methods…"))
        ll.addWidget(self.search)
        self.tree = QTreeWidget()
        self.tree.setHeaderHidden(True)
        ll.addWidget(self.tree, 1)
        for g, recs in cat.algorithms_by_group().items():
            gi = QTreeWidgetItem([GROUPS.get(g, g)])
            gi.setData(0, Qt.UserRole, None)
            f = gi.font(0)
            f.setBold(True)
            gi.setFont(0, f)
            self.tree.addTopLevelItem(gi)
            for r in recs:
                ok, _ = areg.availability(r["id"])
                it = QTreeWidgetItem([("● " if ok else "○ ") + r["name"]])
                it.setData(0, Qt.UserRole, r["id"])
                it.setToolTip(0, tr("lib.runnable", "runnable here") if ok else tr("lib.not_runnable", "explanation only / not runnable here"))
                gi.addChild(it)
            gi.setExpanded(True)
        self.tree.currentItemChanged.connect(lambda cur, _p: cur and cur.data(0, Qt.UserRole) and self.show(cur.data(0, Qt.UserRole)))
        self.search.textChanged.connect(self._filter)
        group_md = content.markdown("methods", "_families")
        self.center = QTabWidget()
        self.overview = markdown_browser(group_md or "")
        self.steps = StepsWidget()
        self.center.addTab(self.overview, tr("lib.tab_overview", "Concept and explanation"))
        self.center.addTab(self.steps, tr("lib.tab_steps", "How This Algorithm Works"))
        self.right = QScrollArea()
        self.right.setWidgetResizable(True)
        self.right.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.rhost = QWidget()
        self.rlay = QVBoxLayout(self.rhost)
        self.rlay.setSizeConstraint(QLayout.SetMinimumSize)   # scroll instead of squeezing
        self.right.setWidget(self.rhost)
        left.setMinimumWidth(240)          # page must fit on 1280-px (150 % scaled) laptop screens
        self.right.setMinimumWidth(330)
        self.right.setMaximumWidth(700)
        split.addWidget(left)
        split.addWidget(self.center)
        split.addWidget(self.right)
        split.setStretchFactor(1, 1)
        split.setSizes([290, 900, 460])
        lay.addWidget(split, 1)

    def _filter(self, q: str):
        q = q.lower()
        for i in range(self.tree.topLevelItemCount()):
            g = self.tree.topLevelItem(i)
            any_vis = False
            for j in range(g.childCount()):
                c = g.child(j)
                rec = self.main.catalog.algorithms[c.data(0, Qt.UserRole)]
                vis = not q or q in (rec["name"] + rec["family"] + rec["intuitive_explanation_en"]).lower()
                c.setHidden(not vis)
                any_vis |= vis
            g.setHidden(not any_vis)

    def show(self, method_id: str):
        r = self.main.catalog.algorithms[method_id]
        self.overview.setMarkdown(method_markdown(r))
        has_steps = self.steps.load(r)
        self.center.setTabText(1, tr("lib.tab_steps", "How This Algorithm Works") + ("" if has_steps else " (—)"))
        clear_layout(self.rlay)
        props = Card()
        props.lay.addWidget(title_label(r["name"]))
        lab, kind = STATUS_LABEL.get(r["implementation_status"], (r["implementation_status"], ""))
        row = QHBoxLayout()
        row.addWidget(Badge(lab, kind))
        row.addStretch(1)
        props.lay.addLayout(row)
        ok, why = areg.availability(method_id)
        props.lay.addWidget(QLabel(f"<b>{tr('lib.local_runtime', 'Runtime on this machine')}:</b> "
                                   f"{'✓ ' + tr('lib.available', 'available') if ok else '✗ ' + why}"))
        for k, label in (("learning_paradigm", "Learning paradigm"), ("supported_modalities", "Modalities"),
                         ("supported_tasks", "Tasks"), ("implementation_type", "Implementation type"),
                         ("implementation_version", "Implementation version"), ("source_license", "Code licence"),
                         ("recommended_datasets", "Recommended datasets")):
            v = r.get(k)
            if isinstance(v, list):
                v = ", ".join(v)
            if v:
                q = QLabel(f"<b>{label}:</b> {v}")
                q.setWordWrap(True)
                props.lay.addWidget(q)
        if ok:
            b = QPushButton(tr("lib.try_lab", "Try it in the Methods Laboratory →"))
            b.setObjectName("Primary")
            b.clicked.connect(lambda: self.main.navigate("lab", method_id=method_id))
            props.lay.addWidget(b)
        self.rlay.addWidget(ReferencesPanel(self.main.catalog, r, "algorithm"))
        self.rlay.addWidget(props)
        self.rlay.addStretch(1)

    def apply_preset(self, method_id: str | None = None, **_):
        if not method_id:
            return
        for i in range(self.tree.topLevelItemCount()):
            g = self.tree.topLevelItem(i)
            for j in range(g.childCount()):
                if g.child(j).data(0, Qt.UserRole) == method_id:
                    self.tree.setCurrentItem(g.child(j))
                    return
