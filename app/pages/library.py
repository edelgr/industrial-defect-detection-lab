"""Methods Library — scientific explanation of each method family and algorithm.

Layout: method tree on the left, one wide tabbed panel on the right with three levels of
explanation (Overview → How it works (pipeline diagram) → Example (worked visual example)), plus the
existing step-by-step walk-through, strengths & limitations, the full technical sheet and references.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import (QHBoxLayout, QLabel, QLayout, QLineEdit, QPushButton, QScrollArea, QSplitter,
                               QTabWidget, QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget)

from app import i18n
from app.i18n import tr
from app.widgets.common import Badge, Card, PageHeader, amp, markdown_browser, muted, title_label
from app.widgets.method_visuals import (MethodDiagramPanel, MethodExamplePanel, MethodSummaryCard, ProsConsPanel,
                                        drop_layout_items, wrap_label)
from app.widgets.references import ReferencesPanel
from app.widgets.steps import StepsWidget
from core import settings
from core.algorithms import registry as areg
from core.research import content
from core.visualization import method_examples

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


def visual_record(r: dict, lang: str) -> dict:
    """Didactic content for a method: the curated YAML entry, or a fallback built from the catalog."""
    v = content.method_visual(r["id"], lang)
    if v:
        return {**v, "curated": True}
    intuitive = r.get(f"intuitive_explanation_{lang}") or r["intuitive_explanation_en"]
    return {
        "curated": False,
        "summary": {"problem": r.get("industrial_use") or r["scientific_principle"].split(".")[0] + ".",
                    "idea": intuitive, "when": r.get("industrial_use") or "",
                    "limits": "\n".join(f"- {x}" for x in r.get("limitations", [])[:3]),
                    "io": f"{r['inputs']} → {r['outputs']}", "training": r["training_data_requirements"]},
        "diagram": {"lanes": [{"nodes": [{"id": f"s{i}", "kind": "step", "title": s, "text": ""}
                                          for i, s in enumerate(r.get("main_processing_steps", []))]}], "links": []},
        "strengths": r.get("advantages", []), "limitations": r.get("limitations", []),
        "failures": r.get("failure_cases", []), "industry": r.get("industrial_use") or "",
    }


def _scroll(w: QWidget) -> QScrollArea:
    sa = QScrollArea()
    sa.setWidgetResizable(True)
    sa.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
    sa.setFrameShape(QScrollArea.NoFrame)
    sa.setWidget(w)
    return sa


class LibraryPage(QWidget):
    TAB_OVERVIEW, TAB_HOW, TAB_STEPS, TAB_EXAMPLE, TAB_PROS, TAB_SHEET, TAB_REFS = range(7)

    def __init__(self, main):
        super().__init__()
        self.main = main
        self.method_id: str | None = None
        self._example_for: str | None = None
        self._visual: dict = {}
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
        legend = muted(tr("mv.tree_legend", "● runnable here · ○ explanation only · ◆ visual explanation"))
        ll.addWidget(legend)
        for g, recs in cat.algorithms_by_group().items():
            gi = QTreeWidgetItem([tr(f"lib.group.{g}", GROUPS.get(g, g))])
            gi.setData(0, Qt.UserRole, None)
            f = gi.font(0)
            f.setBold(True)
            gi.setFont(0, f)
            self.tree.addTopLevelItem(gi)
            for r in recs:
                ok, _ = areg.availability(r["id"])
                visual = " ◆" if content.method_visual(r["id"], "en") else ""
                it = QTreeWidgetItem([("● " if ok else "○ ") + r["name"] + visual])
                it.setData(0, Qt.UserRole, r["id"])
                it.setToolTip(0, r["name"] + " — " + (tr("lib.runnable", "runnable here") if ok else tr("lib.not_runnable", "explanation only / not runnable here")))
                gi.addChild(it)
            gi.setExpanded(True)
        self.tree.currentItemChanged.connect(lambda cur, _p: cur and cur.data(0, Qt.UserRole) and self.show(cur.data(0, Qt.UserRole)))
        self.search.textChanged.connect(self._filter)

        # ---------------------------------------------------------------- tabs
        self.center = QTabWidget()
        self.center.setUsesScrollButtons(True)
        # Overview
        self.ov_host = QWidget()
        ov = QVBoxLayout(self.ov_host)
        ov.setContentsMargins(4, 4, 8, 4)
        ov.setSpacing(8)
        self.ov_title = title_label("", "PageTitle")
        self.ov_family = muted("")
        self.ov_badges = QHBoxLayout()
        ov.addWidget(self.ov_title)
        ov.addWidget(self.ov_family)
        ov.addLayout(self.ov_badges)
        self.summary = MethodSummaryCard()
        ov.addWidget(self.summary)
        nav = QHBoxLayout()
        self.go_how = QPushButton(tr("mv.go_how", "See the pipeline diagram →"))
        self.go_example = QPushButton(tr("mv.go_example", "See a worked visual example →"))
        self.go_lab = QPushButton(tr("lib.try_lab", "Try it in the Methods Laboratory →"))
        self.go_lab.setObjectName("Primary")
        self.go_how.clicked.connect(lambda: self.center.setCurrentIndex(self.TAB_HOW))
        self.go_example.clicked.connect(lambda: self.center.setCurrentIndex(self.TAB_EXAMPLE))
        self.go_lab.clicked.connect(lambda: self.method_id and self.main.navigate("lab", method_id=self.method_id))
        for b in (self.go_how, self.go_example, self.go_lab):
            nav.addWidget(b)
        nav.addStretch(1)
        ov.addLayout(nav)
        self.families = markdown_browser(content.markdown("methods", "_families") or "")
        self.families.setMinimumHeight(420)
        ov.addWidget(self.families)
        ov.addStretch(1)
        # How it works (diagram)
        self.how = MethodDiagramPanel()
        how_host = QWidget()
        hl = QVBoxLayout(how_host)
        hl.setContentsMargins(4, 4, 8, 4)
        hl.addWidget(self.how)
        # Step by step (existing conceptual walk-through)
        self.steps = StepsWidget()
        # Example
        self.example = MethodExamplePanel()
        ex_host = QWidget()
        el = QVBoxLayout(ex_host)
        el.setContentsMargins(4, 4, 8, 4)
        el.addWidget(self.example)
        # Strengths & limitations
        self.pros = ProsConsPanel()
        pros_host = QWidget()
        pl = QVBoxLayout(pros_host)
        pl.setContentsMargins(4, 4, 8, 4)
        pl.addWidget(self.pros)
        # Technical sheet (the complete catalog text, unchanged)
        self.overview = markdown_browser("")
        # References + implementation facts
        self.rhost = QWidget()
        self.rlay = QVBoxLayout(self.rhost)
        self.rlay.setContentsMargins(4, 4, 8, 4)
        self.rlay.setSizeConstraint(QLayout.SetMinimumSize)
        for w, key, default in ((_scroll(self.ov_host), "mv.tab_overview", "Overview"),
                                (_scroll(how_host), "mv.tab_how", "How it works"),
                                (self.steps, "mv.tab_steps", "Step by step"),
                                (_scroll(ex_host), "mv.tab_example", "Example"),
                                (_scroll(pros_host), "mv.tab_pros", "Strengths & limits"),
                                (self.overview, "mv.tab_sheet", "Technical sheet"),
                                (_scroll(self.rhost), "mv.tab_refs", "References")):
            self.center.addTab(w, amp(tr(key, default)))
        self.center.currentChanged.connect(self._tab_changed)
        left.setMinimumWidth(230)          # page must fit on 1280-px (150 % scaled) laptop screens
        split.addWidget(left)
        split.addWidget(self.center)
        split.setStretchFactor(1, 1)
        split.setSizes([280, 1000])
        lay.addWidget(split, 1)
        self._show_landing()

    # ------------------------------------------------------------------ helpers
    def _show_landing(self):
        self.ov_title.setText(tr("mv.landing_title", "Choose a method on the left"))
        self.ov_family.setText(tr("mv.landing_text", "Methods marked ◆ have a pipeline diagram and a worked visual "
                                                     "example. Below: an overview of the method families."))
        self.summary.setVisible(False)
        for b in (self.go_how, self.go_example, self.go_lab):
            b.setVisible(False)
        self.families.setVisible(True)
        for i in range(1, self.center.count()):
            self.center.setTabEnabled(i, False)

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

    def _tab_changed(self, i: int):
        if i == self.TAB_EXAMPLE:
            self._load_example()

    def _load_example(self):
        mid = self.method_id
        if not mid or self._example_for == mid:
            return
        self._example_for = mid
        v = self._visual
        if not method_examples.available(mid):
            self.example.clear(tr("mv.ex.none", "No worked visual example for this method yet. See the pipeline "
                                                "diagram and the step-by-step explanation."))
            return
        lang = i18n.current()
        QGuiApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            ex = method_examples.build(mid, lambda k, d: tr("mv." + k, d), lang)
        except Exception as e:  # an example must never break the library page
            self.example.clear(f"{tr('mv.ex.failed', 'The example could not be generated')}: {e}")
            return
        finally:
            QGuiApplication.restoreOverrideCursor()
        self.example.set_example(ex, v.get("example", {}))

    # ------------------------------------------------------------------ main
    def show(self, method_id: str):
        r = self.main.catalog.algorithms[method_id]
        self.method_id = method_id
        lang = i18n.current()
        v = self._visual = visual_record(r, lang)
        for i in range(self.center.count()):
            self.center.setTabEnabled(i, True)
        # Overview
        self.ov_title.setText(r["name"])
        self.ov_family.setText(f"{r['family']} · {r.get('venue') or ''} {r.get('year') or ''}".strip(" ·"))
        drop_layout_items(self.ov_badges)
        lab, kind = STATUS_LABEL.get(r["implementation_status"], (r["implementation_status"], ""))
        self.ov_badges.addWidget(Badge(lab, kind))
        ok, why = areg.availability(method_id)
        self.ov_badges.addWidget(Badge(("✓ " + tr("lib.available", "available")) if ok else ("✗ " + tr("mv.not_here", "not runnable here")),
                                       "ok" if ok else ""))
        if not v["curated"]:
            self.ov_badges.addWidget(Badge(tr("mv.from_catalog", "summary generated from the catalog"), "warn"))
        self.ov_badges.addStretch(1)
        self.summary.set_summary(v.get("summary", {}))
        self.summary.setVisible(True)
        self.families.setVisible(False)
        self.go_how.setVisible(True)
        has_example = method_examples.available(method_id)
        self.go_example.setVisible(has_example)
        self.go_lab.setVisible(ok)
        # How it works
        cap = "" if v["curated"] else tr("mv.generic_diagram", "Generic pipeline generated from the catalog's "
                                                                "processing steps (no curated diagram yet).")
        self.how.set_diagram(v.get("diagram"), cap)
        # Step by step
        has_steps = self.steps.load(r)
        self.center.setTabText(self.TAB_STEPS, tr("mv.tab_steps", "Step by step") + ("" if has_steps else " (—)"))
        self.center.setTabText(self.TAB_EXAMPLE, tr("mv.tab_example", "Example") + ("" if has_example else " (—)"))
        # Example (built lazily when its tab is opened)
        self._example_for = None
        if self.center.currentIndex() == self.TAB_EXAMPLE:
            self._load_example()
        # Strengths & limitations
        self.pros.set_lists(v.get("strengths", []), v.get("limitations", []), v.get("failures", []), v.get("industry", ""))
        # Technical sheet
        self.overview.setMarkdown(method_markdown(r))
        # References + implementation facts
        drop_layout_items(self.rlay)
        props = Card()
        props.lay.addWidget(title_label(r["name"]))
        props.lay.addWidget(QLabel(f"<b>{tr('lib.local_runtime', 'Runtime on this machine')}:</b> "
                                   f"{'✓ ' + tr('lib.available', 'available') if ok else '✗ ' + why}"))
        for k, label in (("learning_paradigm", "Learning paradigm"), ("supported_modalities", "Modalities"),
                         ("supported_tasks", "Tasks"), ("implementation_type", "Implementation type"),
                         ("implementation_version", "Implementation version"), ("source_license", "Code licence"),
                         ("recommended_datasets", "Recommended datasets")):
            val = r.get(k)
            if isinstance(val, list):
                val = ", ".join(val)
            if val:
                props.lay.addWidget(wrap_label(f"**{tr('mv.prop.' + k, label)}:** {val}", rich=True))
        self.rlay.addWidget(ReferencesPanel(self.main.catalog, r, "algorithm"))
        self.rlay.addWidget(props)
        self.rlay.addStretch(1)

    def apply_preset(self, method_id: str | None = None, tab: str | None = None, **_):
        if not method_id:
            return
        for i in range(self.tree.topLevelItemCount()):
            g = self.tree.topLevelItem(i)
            for j in range(g.childCount()):
                if g.child(j).data(0, Qt.UserRole) == method_id:
                    self.tree.setCurrentItem(g.child(j))
                    if self.method_id != method_id:     # already current: show() was not re-triggered
                        self.show(method_id)
                    if tab:
                        idx = {"overview": self.TAB_OVERVIEW, "how": self.TAB_HOW, "steps": self.TAB_STEPS,
                               "example": self.TAB_EXAMPLE, "pros": self.TAB_PROS, "sheet": self.TAB_SHEET,
                               "refs": self.TAB_REFS}.get(tab)
                        if idx is not None:
                            self.center.setCurrentIndex(idx)
                    return
