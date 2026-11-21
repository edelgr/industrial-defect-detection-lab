"""Dataset Research Atlas — interactive scientific catalog of industrial AD benchmarks.

Works fully offline and without any dataset installed.
"""

from __future__ import annotations

from PySide6.QtCore import QRectF, Qt, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import (QComboBox, QGridLayout, QHBoxLayout, QHeaderView, QLabel,
                               QLineEdit, QPushButton, QScrollArea, QSplitter, QStackedWidget,
                               QTableWidget, QTableWidgetItem, QTabWidget, QVBoxLayout, QWidget)

from app.i18n import tr
from app.ui import theme
from app.widgets.common import (Badge, Card, PageHeader, clear_layout, markdown_browser, muted,
                                np_to_pixmap, title_label)
from app.widgets.references import ReferencesPanel
from core import settings
from core.datasets import registry as dreg
from core.research import content

MODALITY_GROUP = {"rgb": "2D", "grayscale": "2D", "point_cloud": "3D", "depth": "3D", "infrared": "IR",
                  "pseudo_3d": "3D", "mesh": "3D", "xyz": "3D"}


def _lang_desc(rec: dict) -> str:
    return rec.get("description_es") if settings.get().language == "es" else rec.get("description_en")


def _install_status(rec: dict) -> str:
    return dreg.installation_status(rec["id"]) if dreg.get(rec["id"]) else "not_installed"


class DatasetCard(Card):
    clicked = Signal(str)

    def __init__(self, rec: dict, parent=None):
        super().__init__(parent)
        self.rid = rec["id"]
        self.setCursor(Qt.PointingHandCursor)
        self.setMinimumWidth(300)
        top = QHBoxLayout()
        top.addWidget(title_label(f"{rec['acronym']}"))
        top.addStretch(1)
        top.addWidget(Badge(str(rec.get("publication_year") or rec.get("year_first_release") or "?")))
        self.lay.addLayout(top)
        self.lay.addWidget(muted(rec["name"]))
        d = QLabel(_lang_desc(rec))
        d.setWordWrap(True)
        d.setMaximumHeight(70)
        self.lay.addWidget(d)
        row = QHBoxLayout()
        for m in rec["modalities"][:4]:
            row.addWidget(Badge(m))
        row.addStretch(1)
        self.lay.addLayout(row)
        row2 = QHBoxLayout()
        st = _install_status(rec)
        row2.addWidget(Badge(tr("atlas.installed", "installed") if st == "installed"
                             else tr("atlas.not_installed", "not installed"), "ok" if st == "installed" else ""))
        acc = (rec.get("access_method") or "").lower()
        if any(k in acc for k in ("registration", "form", "request", "agreement", "application", "gated")):
            row2.addWidget(Badge(tr("atlas.restricted", "access conditions"), "warn"))
        row2.addStretch(1)
        self.lay.addLayout(row2)

    def mousePressEvent(self, e):
        self.clicked.emit(self.rid)


class TimelineWidget(QWidget):
    clicked = Signal(str)

    def __init__(self, records: list[dict], parent=None):
        super().__init__(parent)
        self.records = sorted(records, key=lambda r: (r.get("year_first_release") or 0, r["id"]))
        self.setMinimumHeight(320)
        self._hits: list[tuple[QRectF, str]] = []
        self.setMouseTracking(True)

    def paintEvent(self, e):
        c = theme.colors(settings.get().theme)
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        years = [r.get("year_first_release") or r.get("publication_year") for r in self.records]
        y0, y1 = min(years) - 0.5, max(years) + 0.8
        axis_y = h * 0.55
        X = lambda yr: 50 + (yr - y0) / (y1 - y0) * (w - 100)
        p.setPen(QPen(QColor(c["border"]), 2))
        p.drawLine(40, axis_y, w - 40, axis_y)
        f = QFont(self.font())
        p.setFont(f)
        for yr in range(int(y0 + 0.5), int(y1) + 1):
            x = X(yr)
            p.setPen(QPen(QColor(c["muted"]), 1))
            p.drawLine(x, axis_y - 5, x, axis_y + 5)
            p.drawText(QRectF(x - 30, axis_y + 8, 60, 18), Qt.AlignCenter, str(yr))
        colors = {"2D": c["accent"], "3D": c["accent2"], "MM": "#b18cf0"}
        self._hits.clear()
        stack: dict[int, int] = {}
        for r in self.records:
            yr = r.get("year_first_release") or r.get("publication_year")
            groups = {MODALITY_GROUP.get(m, "2D") for m in r["modalities"]}
            kind = "MM" if len(groups) > 1 else ("3D" if "3D" in groups else "2D")
            k = stack.get(yr, 0)
            stack[yr] = k + 1
            up = (k % 2 == 0)
            lvl = k // 2
            x = X(yr) + (k - 1.5) * 4
            ty = axis_y - 40 - lvl * 54 if up else axis_y + 34 + lvl * 54
            col = QColor(colors[kind])
            p.setPen(QPen(col, 1.5))
            p.drawLine(x, axis_y, x, ty + (22 if up else 0))
            p.setBrush(col)
            p.drawEllipse(QRectF(x - 5, axis_y - 5, 10, 10))
            rect = QRectF(x - 70, ty - (8 if up else 0), 140, 30)
            p.setBrush(QColor(c["panel"]))
            p.drawRoundedRect(rect, 6, 6)
            p.setPen(QColor(c["text"]))
            p.drawText(rect, Qt.AlignCenter, f"{r['acronym']}")
            self._hits.append((rect, r["id"]))
        # legend
        lx = 50
        for name, key in (("2D RGB / grey", "2D"), ("3D geometry", "3D"), ("Multimodal", "MM")):
            p.setBrush(QColor(colors[key]))
            p.setPen(Qt.NoPen)
            p.drawEllipse(QRectF(lx, 14, 10, 10))
            p.setPen(QColor(c["muted"]))
            p.drawText(QRectF(lx + 14, 8, 140, 22), Qt.AlignVCenter, name)
            lx += 150
        p.end()

    def mousePressEvent(self, e):
        for rect, rid in self._hits:
            if rect.contains(e.position()):
                self.clicked.emit(rid)
                return

    def mouseMoveEvent(self, e):
        self.setCursor(Qt.PointingHandCursor if any(r.contains(e.position()) for r, _ in self._hits)
                       else Qt.ArrowCursor)


class DatasetDetail(QWidget):
    back = Signal()

    def __init__(self, main, parent=None):
        super().__init__(parent)
        self.main = main
        self.lay = QVBoxLayout(self)
        self.lay.setContentsMargins(0, 0, 0, 0)

    def show_record(self, rec: dict):
        clear_layout(self.lay)
        cat = self.main.catalog
        top = QHBoxLayout()
        b = QPushButton("← " + tr("atlas.back", "Back to Atlas"))
        b.clicked.connect(self.back.emit)
        top.addWidget(b)
        top.addSpacing(10)
        top.addWidget(title_label(f"{rec['name']} ({rec['acronym']})", "PageTitle"), 1)
        if dreg.get(rec["id"]):
            ex = QPushButton(tr("atlas.open_explorer", "Open in Dataset Explorer"))
            ex.setEnabled(_install_status(rec) == "installed")
            ex.setToolTip("" if ex.isEnabled() else tr("atlas.install_first", "Install/import the dataset first (Settings → Dataset Manager)"))
            ex.clicked.connect(lambda: self.main.navigate("explorer", dataset_id=rec["id"]))
            top.addWidget(ex)
        imp = QPushButton(tr("atlas.manage", "Dataset Manager…"))
        imp.clicked.connect(lambda: self.main.navigate("settings", tab="datasets", dataset_id=rec["id"]))
        top.addWidget(imp)
        self.lay.addLayout(top)

        split = QSplitter(Qt.Horizontal)
        left = QWidget()
        ll = QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 0, 0)
        md = self._markdown(rec)
        long_md = content.markdown("datasets", rec["id"])
        if long_md:
            md += "\n\n---\n\n" + long_md
        ll.addWidget(markdown_browser(md), 1)
        gal = self._gallery(rec)
        if gal is not None:
            ll.addWidget(gal)
        split.addWidget(left)

        right = QScrollArea()
        right.setWidgetResizable(True)
        right.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        right.setMinimumWidth(460)
        rw = QWidget()
        rl = QVBoxLayout(rw)
        rl.setSizeConstraint(rl.SizeConstraint.SetMinimumSize)
        rl.addWidget(ReferencesPanel(cat, rec, "dataset"))
        facts = Card()
        facts.lay.addWidget(title_label(tr("atlas.key_facts", "Key facts")))
        for k, label in (("publication_venue", "Venue"), ("year_first_release", "First release"),
                         ("license", "Licence"), ("access_method", "Access"), ("sample_count", "Samples"),
                         ("image_resolution", "Resolution"), ("size_notes", "Size")):
            v = rec.get(k)
            if v not in (None, ""):
                lab = QLabel(f"<b>{label}:</b> {v}")
                lab.setWordWrap(True)
                facts.lay.addWidget(lab)
        st = _install_status(rec)
        facts.lay.addWidget(QLabel(f"<b>{tr('atlas.local_status', 'Local status')}:</b> {st}"
                                   + (f" — {', '.join(dreg.installed_categories(rec['id']))}" if st == "installed" else "")))
        rl.addWidget(facts)
        rl.addStretch(1)
        right.setWidget(rw)
        split.addWidget(right)
        split.setSizes([800, 480])
        self.lay.addWidget(split, 1)

    def _markdown(self, r: dict) -> str:
        def bullets(xs):
            return "\n".join(f"- {x}" for x in xs) if xs else "_unknown_"
        authors = ", ".join(r.get("authors") or []) or "_unknown_"
        s = [f"### {r.get('publication_title') or r['name']}",
             f"*{authors}* — {r.get('publication_venue') or ''} {r.get('publication_year') or ''}",
             "", f"**{tr('atlas.intro', 'Introduction')}.** {_lang_desc(r)}", "",
             f"**{tr('atlas.purpose', 'Scientific purpose')}.** {r.get('scientific_purpose')}", "",
             f"**{tr('atlas.contribution', 'Scientific contribution')}.** {r.get('scientific_contribution') or '_unknown_'}", "",
             f"#### {tr('atlas.modalities', 'Modalities')}", bullets(r["modalities"]),
             f"#### {tr('atlas.anomaly_types', 'Anomaly types')}", bullets(r["anomaly_types"]),
             f"#### {tr('atlas.categories', 'Categories')}", ", ".join(r.get("categories") or []) or "_unknown_",
             f"#### {tr('atlas.annotations', 'Annotations & formats')}",
             bullets((r.get("annotation_types") or []) + [f"format: {x}" for x in (r.get("data_formats") or [])]),
             f"#### {tr('atlas.splits', 'Splits & ground truth')}",
             f"- **Train:** {r.get('train_split') or 'unknown'}",
             f"- **Validation:** {r.get('validation_split') or 'unknown'}",
             f"- **Test:** {r.get('test_split') or 'unknown'}",
             f"- **Ground truth:** {r.get('ground_truth_availability') or 'unknown'}",
             f"#### {tr('atlas.protocol', 'Benchmark protocol')}", r.get("benchmark_protocol") or "_unknown_",
             f"#### {tr('atlas.challenges', 'Principal challenges')}", bullets(r.get("principal_challenges")),
             f"#### {tr('atlas.limitations', 'Limitations')}", bullets(r.get("limitations")),
             f"#### {tr('atlas.exercises', 'Recommended exercises')}", bullets(r.get("recommended_educational_exercises"))]
        return "\n".join(s)

    def _gallery(self, rec: dict) -> QWidget | None:
        """Thumbnails of locally installed samples (never shipped with the app)."""
        cats = dreg.installed_categories(rec["id"])
        if not cats:
            return None
        from core.datasets.io import read_rgb8
        import cv2
        try:
            idx = dreg.load_index(rec["id"], cats[0])
        except Exception:
            return None
        goods = [s for s in idx.samples if s.label == "good" and "rgb" in s.modalities][:3]
        bads = [s for s in idx.samples if s.label == "bad" and "rgb" in s.modalities][:5]
        w = QWidget()
        lay = QHBoxLayout(w)
        lay.setContentsMargins(0, 4, 0, 0)
        lay.addWidget(muted(f"{tr('atlas.local_samples', 'Local samples')} ({cats[0]}):"))
        for s in goods + bads:
            try:
                img = read_rgb8(s.modalities["rgb"])
                img = cv2.resize(img, (110, int(110 * img.shape[0] / img.shape[1])))
            except Exception:
                continue
            lab = QLabel()
            lab.setPixmap(np_to_pixmap(img))
            lab.setToolTip(f"{s.split}/{s.defect_type or 'good'} — {s.sample_id}")
            lay.addWidget(lab)
        lay.addStretch(1)
        return w


class AtlasPage(QWidget):
    def __init__(self, main):
        super().__init__()
        self.main = main
        self.records = sorted(main.catalog.datasets.values(), key=lambda r: (r.get("year_first_release") or 0, r["id"]))
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(PageHeader(tr("nav.atlas", "Dataset Research Atlas"),
                                   tr("atlas.subtitle", "How industrial anomaly-detection benchmarks evolved — and why each one exists.")))
        self.stack = QStackedWidget()
        outer.addWidget(self.stack, 1)

        self.tabs = QTabWidget()
        self.tabs.addTab(self._gallery_tab(), tr("atlas.tab_gallery", "Gallery"))
        tl = TimelineWidget(self.records)
        tl.clicked.connect(self.open_detail)
        tlw = QWidget()
        tll = QVBoxLayout(tlw)
        tll.addWidget(tl, 2)
        tll.addWidget(markdown_browser(content.markdown("atlas", "evolution") or ""), 3)
        self.tabs.addTab(tlw, tr("atlas.tab_timeline", "Timeline"))
        self.tabs.addTab(self._table_tab(), tr("atlas.tab_compare", "Compare datasets"))
        self.tabs.addTab(markdown_browser(content.markdown("atlas", "modalities") or ""),
                         tr("atlas.tab_modalities", "Modalities & anomaly types"))
        self.tabs.addTab(markdown_browser(content.markdown("atlas", "benchmarks") or ""),
                         tr("atlas.tab_benchmarks", "What is a benchmark?"))
        self.stack.addWidget(self.tabs)
        self.detail = DatasetDetail(main)
        self.detail.back.connect(lambda: self.stack.setCurrentIndex(0))
        self.stack.addWidget(self.detail)

    # ------------------------------------------------------------------ gallery
    def _gallery_tab(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        bar = QHBoxLayout()
        self.search = QLineEdit()
        self.search.setPlaceholderText(tr("atlas.search", "Search datasets, categories, anomaly types…"))
        self.f_year = QComboBox()
        self.f_year.addItem(tr("atlas.all_years", "All years"), None)
        for y in sorted({r.get("year_first_release") for r in self.records if r.get("year_first_release")}):
            self.f_year.addItem(str(y), y)
        self.f_mod = QComboBox()
        self.f_mod.addItem(tr("atlas.all_modalities", "All modalities"), None)
        for m in sorted({m for r in self.records for m in r["modalities"]}):
            self.f_mod.addItem(m, m)
        self.f_anom = QComboBox()
        self.f_anom.addItem(tr("atlas.all_anomalies", "All anomaly types"), None)
        for a in sorted({a for r in self.records for a in r["anomaly_types"]}):
            self.f_anom.addItem(a, a)
        for x in (self.search, self.f_year, self.f_mod, self.f_anom):
            bar.addWidget(x, 2 if x is self.search else 1)
        lay.addLayout(bar)
        sc = QScrollArea()
        sc.setWidgetResizable(True)
        self.grid_host = QWidget()
        self.grid = QGridLayout(self.grid_host)
        self.grid.setSpacing(12)
        sc.setWidget(self.grid_host)
        lay.addWidget(sc, 1)
        self.search.textChanged.connect(self._refresh_gallery)
        for c in (self.f_year, self.f_mod, self.f_anom):
            c.currentIndexChanged.connect(self._refresh_gallery)
        self._refresh_gallery()
        return w

    def _match(self, r: dict) -> bool:
        q = self.search.text().strip().lower()
        if q:
            hay = " ".join(str(v) for v in (r["name"], r["acronym"], r.get("description_en"),
                                              r.get("categories"), r["anomaly_types"], r["modalities"])).lower()
            if q not in hay:
                return False
        y, m, a = self.f_year.currentData(), self.f_mod.currentData(), self.f_anom.currentData()
        return ((y is None or r.get("year_first_release") == y) and (m is None or m in r["modalities"])
                and (a is None or a in r["anomaly_types"]))

    def _refresh_gallery(self):
        clear_layout(self.grid)
        recs = [r for r in self.records if self._match(r)]
        for i, r in enumerate(recs):
            c = DatasetCard(r)
            c.clicked.connect(self.open_detail)
            self.grid.addWidget(c, i // 3, i % 3)
        if not recs:
            self.grid.addWidget(muted(tr("atlas.no_match", "No dataset matches the filters.")), 0, 0)
        self.grid.setRowStretch(len(recs) // 3 + 1, 1)

    # ------------------------------------------------------------------ table
    def _table_tab(self) -> QWidget:
        cols = [("acronym", "Dataset"), ("year_first_release", "Year"), ("categories", "# Categories"),
                ("modalities", "Modalities"), ("anomaly_types", "Defect types"),
                ("annotation_types", "Annotations"), ("principal_challenges", "Main research challenge"),
                ("access_method", "Access restrictions"), ("_install", "Installation")]
        t = QTableWidget(len(self.records), len(cols))
        t.setHorizontalHeaderLabels([tr(f"atlas.col.{k}", lab) for k, lab in cols])
        t.setAlternatingRowColors(True)
        t.setWordWrap(True)
        t.verticalHeader().setVisible(False)
        for i, r in enumerate(self.records):
            for j, (k, _) in enumerate(cols):
                if k == "_install":
                    v = _install_status(r)
                elif k == "categories":
                    v = len(r.get("categories") or []) or "?"
                elif k == "principal_challenges":
                    v = (r.get(k) or ["?"])[0]
                else:
                    v = r.get(k)
                    v = ", ".join(map(str, v)) if isinstance(v, list) else ("?" if v is None else v)
                it = QTableWidgetItem(str(v))
                it.setFlags(it.flags() & ~Qt.ItemIsEditable)
                it.setToolTip(str(v))
                t.setItem(i, j, it)
        hh = t.horizontalHeader()
        hh.setSectionResizeMode(QHeaderView.Interactive)
        for j, wdt in enumerate((110, 55, 90, 170, 200, 170, 280, 230, 100)):
            t.setColumnWidth(j, wdt)
        t.resizeRowsToContents()
        t.cellDoubleClicked.connect(lambda row, _c: self.open_detail(self.records[row]["id"]))
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.addWidget(muted(tr("atlas.table_hint", "Double-click a row to open the dataset's scientific profile.")))
        lay.addWidget(t, 3)
        lay.addWidget(markdown_browser(content.markdown("atlas", "comparison") or ""), 2)
        return w

    # ------------------------------------------------------------------ detail
    def open_detail(self, rid: str):
        rec = self.main.catalog.datasets.get(rid)
        if rec:
            self.detail.show_record(rec)
            self.stack.setCurrentIndex(1)

    def apply_preset(self, dataset_id: str | None = None, **_):
        if dataset_id:
            self.open_detail(dataset_id)
