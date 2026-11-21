"""Reusable didactic widgets for the Methods Library.

* :class:`MethodSummaryCard`   — the intuitive summary (problem · idea · when to use · limitations).
* :class:`MethodDiagramWidget` — a pipeline drawn with QPainter (boxes, arrows, training/test lanes,
  cross-lane links), wraps to the available width; click a stage to read its description.
* :class:`MethodStepGallery`   — numbered thumbnails of a worked example (input → intermediates → output).
* :class:`MethodExamplePanel`  — provenance banner + step gallery + large outputs + interpretation.
* :class:`ProsConsPanel`       — strengths / limitations / typical failures side by side.

Content comes from ``education/<lang>/method_visuals.yaml`` (see ``core.research.content.method_visual``)
and example images from ``core.visualization.method_examples``.  Nothing here runs an algorithm.
"""

from __future__ import annotations

import html

import numpy as np
from PySide6.QtCore import QPointF, QRectF, QSize, Qt, Signal
from PySide6.QtGui import QColor, QFont, QFontMetrics, QPainter, QPainterPath, QPen, QPolygonF
from PySide6.QtWidgets import (QDialog, QFrame, QGridLayout, QHBoxLayout, QLabel, QSizePolicy, QTextBrowser,
                               QVBoxLayout, QWidget)

from app.i18n import tr
from app.widgets.common import Badge, Card, np_to_pixmap, title_label

KINDS = ("data", "train", "pre", "feature", "model", "score", "decision", "step")
KIND_COLORS = {"data": "#4f8fd6", "train": "#4e9f55", "pre": "#8a98a8", "feature": "#d08a2a",
               "model": "#2bb3a3", "score": "#9b6cd0", "decision": "#d9534f", "step": "#6f8fa8"}
KIND_LABELS = {"data": "data", "train": "normal data / fitting", "pre": "pre-processing", "feature": "features",
               "model": "model of normality", "score": "scoring", "decision": "decision",
               "step": "processing step"}


def kind_label(kind: str) -> str:
    return tr(f"mv.kind.{kind}", KIND_LABELS.get(kind, kind))


def _palette() -> dict:
    from app.ui import theme
    from core import settings
    return theme.colors(settings.get().theme)


class _SafeDict(dict):
    def __missing__(self, key):
        return "{" + key + "}"


def fill(text: str, facts: dict | None) -> str:
    """Interpolate example numbers (``{score:.2f}``) into a caption; never fails."""
    if not text or not facts or "{" not in text:
        return text or ""
    try:
        return text.format_map(_SafeDict({k: v for k, v in facts.items() if not k.startswith("_")}))
    except (ValueError, KeyError, IndexError, TypeError):
        return text


def drop(w: QWidget | None) -> None:
    """Remove a widget immediately (deleteLater alone leaves it painted until the event loop runs)."""
    if w is not None:
        w.hide()
        w.setParent(None)
        w.deleteLater()


def drop_layout_items(layout) -> None:
    while layout.count():
        it = layout.takeAt(0)
        drop(it.widget())
        if it.layout() is not None:
            drop_layout_items(it.layout())


def wrap_label(text: str, obj: str = "", rich: bool = False) -> QLabel:
    lab = QLabel(text)
    lab.setWordWrap(True)
    lab.setTextFormat(Qt.MarkdownText if rich else Qt.PlainText)
    if obj:
        lab.setObjectName(obj)
    lab.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)
    return lab


# ================================================================================ summary card
class MethodSummaryCard(QWidget):
    """Four short answers + 'input → output' and 'training data' chips."""

    FIELDS = (("problem", "◎", "What problem does it solve?"), ("idea", "💡", "Basic idea"),
              ("when", "✓", "When to use it"), ("limits", "⚠", "Main limitations"))

    def __init__(self, parent=None):
        super().__init__(parent)
        self.lay = QVBoxLayout(self)
        self.lay.setContentsMargins(0, 0, 0, 0)
        self.lay.setSpacing(8)
        self.chips = QHBoxLayout()
        self.chips.setSpacing(8)
        self.lay.addLayout(self.chips)
        self.grid = QGridLayout()
        self.grid.setSpacing(8)
        self.lay.addLayout(self.grid)
        self.cards: dict[str, QLabel] = {}
        for i, (key, icon, title) in enumerate(self.FIELDS):
            c = Card()
            c.lay.addWidget(title_label(f"{icon}  {tr('mv.sum.' + key, title)}"))
            body = wrap_label("", rich=True)
            c.lay.addWidget(body)
            c.lay.addStretch(1)
            self.cards[key] = body
            self.grid.addWidget(c, i // 2, i % 2)
        self.grid.setColumnStretch(0, 1)
        self.grid.setColumnStretch(1, 1)

    def set_summary(self, summary: dict) -> None:
        drop_layout_items(self.chips)
        for key, label in (("io", tr("mv.sum.io", "Input → output")), ("training", tr("mv.sum.training", "Training data"))):
            if summary.get(key):
                chip = wrap_label(f"**{label}:** {summary[key]}", "Muted", rich=True)
                self.chips.addWidget(chip, 1)
        for key, lab in self.cards.items():
            lab.setText(summary.get(key) or "—")


# ================================================================================ diagram
class MethodDiagramWidget(QWidget):
    """Pipeline diagram drawn by the application (no static image): lanes of stages joined by arrows.

    ``diagram`` = {"lanes": [{"label": str?, "nodes": [{"id", "kind", "title", "text"}]}],
                   "links": [{"from": id, "to": id, "label": str?}]}
    Stages inside a lane are chained left → right (wrapping to new rows when narrow); ``links``
    connect stages of different lanes (e.g. the memory bank built in training feeds test scoring).
    """

    nodeSelected = Signal(str)
    MIN_W, MAX_W, GAP, ROW_GAP, LANE_GAP, PAD = 112, 172, 30, 30, 22, 10

    def __init__(self, parent=None):
        super().__init__(parent)
        self.diagram: dict = {"lanes": [], "links": []}
        self.selected: str | None = None
        self.hover: str | None = None
        self._geo: tuple[int, dict, list, int] | None = None
        self.setMouseTracking(True)
        sp = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        sp.setHeightForWidth(True)
        self.setSizePolicy(sp)
        self.setMinimumWidth(2 * self.MIN_W)

    # ------------------------------------------------------------------ data
    def set_diagram(self, diagram: dict | None) -> None:
        self.diagram = diagram or {"lanes": [], "links": []}
        nodes = self.nodes()
        self.selected = nodes[0]["id"] if nodes else None
        self._geo = None
        self.updateGeometry()
        self.update()

    def nodes(self) -> list[dict]:
        return [n for lane in self.diagram.get("lanes", []) for n in lane.get("nodes", [])]

    def number(self, node_id: str) -> int:
        return next((i + 1 for i, n in enumerate(self.nodes()) if n["id"] == node_id), 0)

    def select(self, node_id: str) -> None:
        if node_id != self.selected:
            self.selected = node_id
            self.update()
        self.nodeSelected.emit(node_id)

    # ---------------------------------------------------------------- layout
    @staticmethod
    def _title_font(bold: QFont, title: str, width: float) -> QFont:
        """Bold font, shrunk (down to 75 %) when one word of the title is wider than the box."""
        f = QFont(bold)
        size = bold.pointSizeF()
        words = title.split() or [""]
        while size > bold.pointSizeF() * 0.75 and max(QFontMetrics(f).horizontalAdvance(w) for w in words) > width:
            size -= 0.5
            f.setPointSizeF(size)
        return f

    def _fonts(self):
        f = QFont(self.font())
        f.setBold(True)
        small = QFont(self.font())
        small.setPointSizeF(max(7.0, self.font().pointSizeF() * 0.82))
        return f, small

    def _layout(self, width: int):
        if self._geo and self._geo[0] == width:
            return self._geo
        bold, small = self._fonts()
        fm, fms = QFontMetrics(bold), QFontMetrics(small)
        avail = max(width - 2 * self.PAD, self.MIN_W)
        rects: dict[str, QRectF] = {}
        lanes_geo = []
        y = self.PAD
        for lane in self.diagram.get("lanes", []):
            nodes = lane.get("nodes", [])
            if not nodes:
                continue
            label_y = None
            if lane.get("label"):
                label_y = y
                y += fms.height() + 6
            per_row = max(1, min(len(nodes), int((avail + self.GAP) // (self.MIN_W + self.GAP))))
            bw = min(self.MAX_W, (avail - (per_row - 1) * self.GAP) / per_row)
            rows = [nodes[i:i + per_row] for i in range(0, len(nodes), per_row)]
            for row in rows:
                h = 0
                for n in row:
                    tf = QFontMetrics(self._title_font(bold, n.get("title", ""), bw - 37))
                    tr_ = tf.boundingRect(0, 0, int(bw) - 37, 2000, Qt.TextWordWrap, n.get("title", ""))
                    h = max(h, tr_.height() + fms.height() + 20)
                h = max(h, 54)
                for k, n in enumerate(row):
                    rects[n["id"]] = QRectF(self.PAD + k * (bw + self.GAP), y, bw, h)
                y += h + self.ROW_GAP
            y += self.LANE_GAP - self.ROW_GAP
            lanes_geo.append((lane, label_y, nodes))
        legend_y = y
        total = int(legend_y + fms.height() + 8 + self.PAD) if rects else 60
        self._geo = (width, rects, lanes_geo, total)
        return self._geo

    def hasHeightForWidth(self):
        return True

    def heightForWidth(self, w):
        return self._layout(w)[3]

    def sizeHint(self):
        w = max(self.width(), 640)
        return QSize(w, self.heightForWidth(w))

    def resizeEvent(self, e):
        self._geo = None
        self.updateGeometry()
        super().resizeEvent(e)

    # ------------------------------------------------------------------ mouse
    def _hit(self, pos) -> str | None:
        _, rects, _, _ = self._layout(self.width())
        return next((k for k, r in rects.items() if r.contains(QPointF(pos))), None)

    def mouseMoveEvent(self, e):
        h = self._hit(e.position())
        if h != self.hover:
            self.hover = h
            self.setCursor(Qt.PointingHandCursor if h else Qt.ArrowCursor)
            n = next((n for n in self.nodes() if n["id"] == h), None)
            self.setToolTip(f"<b>{html.escape(n.get('title', ''))}</b><br>{html.escape(n.get('text', ''))}" if n else "")
            self.update()

    def leaveEvent(self, e):
        self.hover = None
        self.update()

    def mousePressEvent(self, e):
        h = self._hit(e.position())
        if h:
            self.select(h)

    # ------------------------------------------------------------------ paint
    @staticmethod
    def _arrow_head(p: QPainter, tip: QPointF, direction: QPointF, color: QColor, size=7.0):
        d = direction / max((direction.x() ** 2 + direction.y() ** 2) ** 0.5, 1e-6)
        n = QPointF(-d.y(), d.x())
        poly = QPolygonF([tip, tip - d * size + n * size * 0.55, tip - d * size - n * size * 0.55])
        p.setBrush(color)
        p.setPen(Qt.NoPen)
        p.drawPolygon(poly)

    def paintEvent(self, _e):
        pal = _palette()
        _, rects, lanes_geo, total = self._layout(self.width())
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        bold, small = self._fonts()
        text_c, muted_c = QColor(pal["text"]), QColor(pal["muted"])
        line_c = QColor(pal["muted"])
        # lane labels + in-lane arrows
        for lane, label_y, nodes in lanes_geo:
            if label_y is not None:
                p.setFont(small)
                p.setPen(muted_c)
                p.drawText(QRectF(self.PAD, label_y, self.width() - 2 * self.PAD, 20), Qt.AlignLeft | Qt.AlignTop,
                           lane["label"].upper())
            pen = QPen(line_c, 1.6)
            for a, b in zip(nodes, nodes[1:]):
                ra, rb = rects[a["id"]], rects[b["id"]]
                p.setPen(pen)
                p.setBrush(Qt.NoBrush)
                if abs(ra.top() - rb.top()) < 1:                      # same row
                    s, t = QPointF(ra.right() + 2, ra.center().y()), QPointF(rb.left() - 2, rb.center().y())
                    p.drawLine(s, t)
                    self._arrow_head(p, t, t - s, line_c)
                else:                                                 # wrapped: down, left, down
                    mid = (ra.bottom() + rb.top()) / 2
                    path = QPainterPath(QPointF(ra.center().x(), ra.bottom() + 2))
                    path.lineTo(ra.center().x(), mid)
                    path.lineTo(rb.center().x(), mid)
                    path.lineTo(rb.center().x(), rb.top() - 2)
                    p.drawPath(path)
                    self._arrow_head(p, QPointF(rb.center().x(), rb.top() - 2), QPointF(0, 1), line_c)
        # cross-lane links (dashed)
        accent = QColor(pal["accent"])
        for link in self.diagram.get("links", []):
            ra, rb = rects.get(link.get("from")), rects.get(link.get("to"))
            if ra is None or rb is None:
                continue
            down = rb.top() >= ra.bottom()
            s = QPointF(ra.center().x() + 10, ra.bottom() + 2 if down else ra.top() - 2)
            t = QPointF(rb.center().x() + 10, rb.top() - 2 if down else rb.bottom() + 2)
            dy = (t.y() - s.y()) * 0.5
            path = QPainterPath(s)
            path.cubicTo(QPointF(s.x(), s.y() + dy), QPointF(t.x(), t.y() - dy), t)
            pen = QPen(accent, 1.8, Qt.DashLine)
            p.setPen(pen)
            p.setBrush(Qt.NoBrush)
            p.drawPath(path)
            self._arrow_head(p, t, QPointF(0, 1 if down else -1), accent, 8)
            if link.get("label"):
                p.setFont(small)
                m = path.pointAtPercent(0.5)
                fm = QFontMetrics(small)
                tw = fm.horizontalAdvance(link["label"]) + 10
                box = QRectF(m.x() + 6, m.y() - fm.height() / 2 - 1, tw, fm.height() + 2)
                p.setPen(Qt.NoPen)
                p.setBrush(QColor(pal["panel"]))
                p.drawRoundedRect(box, 4, 4)
                p.setPen(accent)
                p.drawText(box, Qt.AlignCenter, link["label"])
        # boxes
        fm_small = QFontMetrics(small)
        for k, n in enumerate(self.nodes()):
            r = rects.get(n["id"])
            if r is None:
                continue
            col = QColor(KIND_COLORS.get(n.get("kind", "pre"), "#8a98a8"))
            fillc = QColor(col)
            fillc.setAlphaF(0.30 if n["id"] in (self.selected, self.hover) else 0.15)
            p.setBrush(QColor(pal["panel"]))
            p.setPen(Qt.NoPen)
            p.drawRoundedRect(r, 8, 8)
            p.setBrush(fillc)
            p.setPen(QPen(col, 2.6 if n["id"] == self.selected else 1.4))
            p.drawRoundedRect(r, 8, 8)
            # number badge
            badge = QRectF(r.left() + 7, r.top() + 7, 20, 20)
            p.setBrush(col)
            p.setPen(Qt.NoPen)
            p.drawEllipse(badge)
            p.setPen(QColor("#ffffff"))
            p.setFont(small)
            p.drawText(badge, Qt.AlignCenter, str(k + 1))
            # title + kind
            p.setFont(self._title_font(bold, n.get("title", ""), r.width() - 37))
            p.setPen(text_c)
            p.drawText(QRectF(r.left() + 31, r.top() + 7, r.width() - 37, r.height() - fm_small.height() - 12),
                       Qt.TextWordWrap | Qt.AlignLeft | Qt.AlignTop, n.get("title", ""))
            p.setFont(small)
            p.setPen(muted_c)
            p.drawText(QRectF(r.left() + 8, r.bottom() - fm_small.height() - 6, r.width() - 14, fm_small.height() + 2),
                       Qt.AlignLeft | Qt.AlignVCenter, fm_small.elidedText(kind_label(n.get("kind", "pre")), Qt.ElideRight, int(r.width() - 14)))
        # legend (only kinds used)
        used = [k for k in KINDS if any(n.get("kind") == k for n in self.nodes())]
        x, y = float(self.PAD), float(total - self.PAD - fm_small.height() - 4)
        p.setFont(small)
        for kind in used:
            lab = kind_label(kind)
            w = fm_small.horizontalAdvance(lab) + 26
            if x + w > self.width() - self.PAD:
                break
            col = QColor(KIND_COLORS[kind])
            p.setBrush(col)
            p.setPen(Qt.NoPen)
            p.drawRoundedRect(QRectF(x, y + 3, 11, 11), 3, 3)
            p.setPen(muted_c)
            p.drawText(QRectF(x + 15, y, w, fm_small.height() + 4), Qt.AlignLeft | Qt.AlignVCenter, lab)
            x += w + 6
        p.end()


class MethodDiagramPanel(QWidget):
    """Diagram + the numbered description of every stage (selected stage highlighted)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.caption = wrap_label("", "Muted")
        lay.addWidget(self.caption)
        self.diagram = MethodDiagramWidget()
        lay.addWidget(self.diagram)
        self.details = QTextBrowser()
        self.details.setOpenLinks(False)
        self.details.setMinimumHeight(150)
        lay.addWidget(self.details, 1)
        self.diagram.nodeSelected.connect(self._render)

    def set_diagram(self, diagram: dict | None, caption: str = "") -> None:
        self.caption.setText(caption)
        self.caption.setVisible(bool(caption))
        self.diagram.set_diagram(diagram)
        self._render(self.diagram.selected)

    def _render(self, selected: str | None) -> None:
        pal = _palette()
        rows = []
        for i, n in enumerate(self.diagram.nodes()):
            col = KIND_COLORS.get(n.get("kind", "pre"), "#8a98a8")
            sel = n["id"] == selected
            bg = f"background-color:{pal['sel']};" if sel else ""
            rows.append(f"<a name='{html.escape(n['id'])}'></a><div style='{bg} padding:4px; margin-bottom:4px;'>"
                        f"<span style='color:{col}; font-weight:bold;'>{i + 1}. {html.escape(n.get('title', ''))}</span>"
                        f" <span style='color:{pal['muted']};'>({html.escape(kind_label(n.get('kind', 'pre')))})</span><br>"
                        f"{html.escape(n.get('text', ''))}</div>")
        hint = html.escape(tr("mv.diagram_hint", "Click a stage in the diagram to highlight its description."))
        self.details.setHtml(f"<p style='color:{pal['muted']};'>{hint}</p>" + "".join(rows))
        if selected:
            self.details.scrollToAnchor(selected)


# ================================================================================ step gallery
class _Tile(QFrame):
    clicked = Signal(int)

    def __init__(self, index: int, image: np.ndarray, caption: str, role: str, size: int):
        super().__init__()
        self.index = index
        self.setObjectName("Card")
        self.setCursor(Qt.PointingHandCursor)
        col = {"input": KIND_COLORS["data"], "train": KIND_COLORS["train"], "output": KIND_COLORS["decision"]}.get(role)
        if col:
            self.setStyleSheet(f"#Card {{ border: 2px solid {col}; }}")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(6, 6, 6, 6)
        lay.setSpacing(4)
        head = QLabel(f"<b>{index + 1}</b>")
        lay.addWidget(head)
        pic = QLabel()
        pm = np_to_pixmap(image)
        pic.setPixmap(pm.scaled(size, size, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        pic.setAlignment(Qt.AlignCenter)
        pic.setMinimumHeight(min(size, pm.height() * size // max(pm.width(), 1)))
        lay.addWidget(pic)
        cap = wrap_label(caption)
        cap.setFixedWidth(size)
        lay.addWidget(cap)
        lay.addStretch(1)
        self.setFixedWidth(size + 14)
        self.setToolTip(tr("mv.enlarge", "Click to enlarge"))

    def mousePressEvent(self, e):
        self.clicked.emit(self.index)


class MethodStepGallery(QWidget):
    """Numbered thumbnails joined by arrows; reflows to the available width; click to enlarge."""

    TILE = 140
    ARROW = 18

    def __init__(self, parent=None):
        super().__init__(parent)
        self.grid = QGridLayout(self)
        self.grid.setContentsMargins(0, 0, 0, 0)
        self.grid.setHorizontalSpacing(2)
        self.grid.setVerticalSpacing(8)
        self.items: list[tuple[np.ndarray, str, str]] = []
        self.tiles: list[_Tile] = []
        self._cols = 0
        # take the width we are given (and reflow to it) instead of imposing the current grid's minimum
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.setMinimumWidth(self.TILE + 14)

    def set_stages(self, items: list[tuple[np.ndarray, str, str]]) -> None:
        """items = [(rgb image, caption, role)]"""
        self.items = items
        for t in self.tiles:
            drop(t)
        self.tiles = [_Tile(i, im, cap, role, self.TILE) for i, (im, cap, role) in enumerate(items)]
        for t in self.tiles:
            t.clicked.connect(self.enlarge)
        self._cols = 0
        self._reflow()

    def _columns(self) -> int:
        w = max(self.width(), self.TILE + 14)
        return max(1, int((w + self.ARROW) // (self.TILE + 14 + self.ARROW)))

    def _reflow(self) -> None:
        cols = self._columns()
        if cols == self._cols:
            return
        self._cols = cols
        while self.grid.count():
            it = self.grid.takeAt(0)
            w = it.widget()
            if w is not None and not isinstance(w, _Tile):
                drop(w)
        for i, t in enumerate(self.tiles):
            r, c = divmod(i, cols)
            self.grid.addWidget(t, r, 2 * c, Qt.AlignTop)
            if c < cols - 1 and i < len(self.tiles) - 1:
                a = QLabel("→")
                a.setObjectName("Muted")
                a.setAlignment(Qt.AlignCenter)
                a.setFixedWidth(self.ARROW)
                self.grid.addWidget(a, r, 2 * c + 1, Qt.AlignVCenter)
        self.grid.setColumnStretch(2 * cols, 1)

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._reflow()

    def enlarge(self, index: int) -> None:
        im, cap, _ = self.items[index]
        d = QDialog(self)
        d.setWindowTitle(f"{index + 1}. {cap[:60]}")
        lay = QVBoxLayout(d)
        pic = QLabel()
        pm = np_to_pixmap(im)
        side = 460
        pic.setPixmap(pm.scaled(side, side, Qt.KeepAspectRatio, Qt.FastTransformation if pm.width() <= 200 else Qt.SmoothTransformation))
        pic.setAlignment(Qt.AlignCenter)
        lay.addWidget(pic)
        lay.addWidget(wrap_label(cap))
        d.resize(500, 560)
        d.show()
        self._dialog = d


# ================================================================================ example panel
class MethodExamplePanel(QWidget):
    """Worked example: provenance, step-by-step gallery, final outputs and interpretation."""

    OUT = 210

    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(10)
        top = QHBoxLayout()
        self.badge = Badge("")
        top.addWidget(self.badge)
        top.addStretch(1)
        lay.addLayout(top)
        self.prov = wrap_label("", "Muted")
        lay.addWidget(self.prov)
        self.steps_title = title_label(tr("mv.ex.steps", "Step by step"))
        lay.addWidget(self.steps_title)
        self.gallery = MethodStepGallery()
        lay.addWidget(self.gallery)
        self.out_title = title_label(tr("mv.ex.result", "Result"))
        lay.addWidget(self.out_title)
        self.out_row = QHBoxLayout()
        self.out_row.setSpacing(10)
        lay.addLayout(self.out_row)
        self.legend = wrap_label("", "Muted")
        lay.addWidget(self.legend)
        self.interp_card = Card()
        self.interp_card.lay.addWidget(title_label(tr("mv.ex.interpretation", "How to read this result")))
        self.interp = wrap_label("", rich=True)
        self.interp_card.lay.addWidget(self.interp)
        lay.addWidget(self.interp_card)
        self.notice = wrap_label("", "Muted")
        lay.addWidget(self.notice)
        self.empty = wrap_label("", "Muted")
        lay.addWidget(self.empty)
        lay.addStretch(1)

    def clear(self, message: str) -> None:
        for w in (self.badge, self.prov, self.steps_title, self.gallery, self.out_title, self.legend,
                  self.interp_card, self.notice):
            w.setVisible(False)
        self._clear_outputs()
        self.empty.setText(message)
        self.empty.setVisible(True)

    def _clear_outputs(self):
        drop_layout_items(self.out_row)

    def set_example(self, example, texts: dict) -> None:
        """``example``: core.visualization.method_examples.Example; ``texts``: YAML 'example' block."""
        caps = texts.get("captions", {})
        facts = example.facts
        cap = lambda key: fill(caps.get(key, key.replace("_", " ")), facts)
        for w in (self.badge, self.prov, self.steps_title, self.gallery, self.out_title, self.legend,
                  self.interp_card, self.notice):
            w.setVisible(True)
        self.empty.setVisible(False)
        if example.provenance == "implementation":
            self.badge.setText(tr("mv.ex.real_badge", "Real IDD Lab implementation · synthetic image"))
            self.badge.setObjectName("BadgeOk")
            self.prov.setText(tr("mv.ex.real_text",
                                 "The final map and mask are computed by the same code as the Methods Laboratory, "
                                 "with default parameters. The intermediate panels replay the same operations."))
        else:
            self.badge.setText(tr("mv.ex.toy_badge", "Toy illustration of the mechanism — not the real model"))
            self.badge.setObjectName("BadgeWarn")
            self.prov.setText(tr("mv.ex.toy_text",
                                 "Simple hand-crafted patch features replace the pretrained network so the example runs "
                                 "instantly and without downloads. The mechanism (what is compared with what, and how "
                                 "the score is formed) is the method's; the numbers are not the real model's."))
        if texts.get("note"):                     # method-specific provenance (e.g. 3D re-enactment)
            self.prov.setText(texts["note"])
        self.badge.style().unpolish(self.badge)
        self.badge.style().polish(self.badge)
        self.gallery.set_stages([(s.image, cap(s.key), s.role) for s in example.stages])
        self._clear_outputs()
        for s in example.outputs:
            box = QVBoxLayout()
            pic = QLabel()
            pm = np_to_pixmap(s.image)
            side = self.OUT if len(example.outputs) > 1 else 2 * self.OUT
            pic.setPixmap(pm.scaled(side, side, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            box.addWidget(pic)
            lab = wrap_label(cap(s.key))
            lab.setFixedWidth(min(side, pm.width() * side // max(pm.height(), 1)))
            box.addWidget(lab)
            box.addStretch(1)
            holder = QWidget()
            holder.setLayout(box)
            self.out_row.addWidget(holder)
        self.out_row.addStretch(1)
        self.legend.setText(tr("mv.ex.legend", "Green outline: where the synthetic defect really is (known because "
                                               "the image was generated). Red: what the method flags."))
        self.legend.setVisible(example.method_id != "multimodal_fusion")
        self.interp.setText(fill(texts.get("interpretation", ""), facts))
        self.notice.setText(tr("mv.ex.synthetic", "Synthetic educational image generated by IDD Lab (fixed seed) — "
                                                  "not real industrial data; freely redistributable."))


# ================================================================================ pros & cons
class ProsConsPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        row = QHBoxLayout()
        row.setSpacing(8)
        self.pros, self.cons = Card(), Card()
        self.pros.lay.addWidget(title_label("✓  " + tr("mv.strengths", "Strengths")))
        self.cons.lay.addWidget(title_label("✗  " + tr("mv.limitations", "Limitations")))
        self.pros_l, self.cons_l = wrap_label("", rich=True), wrap_label("", rich=True)
        self.pros.lay.addWidget(self.pros_l)
        self.cons.lay.addWidget(self.cons_l)
        for c in (self.pros, self.cons):
            c.lay.addStretch(1)
            row.addWidget(c, 1)
        lay.addLayout(row)
        self.fail = Card()
        self.fail.lay.addWidget(title_label("⚠  " + tr("mv.failures", "Typical failure cases")))
        self.fail_l = wrap_label("", rich=True)
        self.fail.lay.addWidget(self.fail_l)
        lay.addWidget(self.fail)
        self.use = Card()
        self.use.lay.addWidget(title_label("🏭  " + tr("mv.industry", "When to use it in industry")))
        self.use_l = wrap_label("", rich=True)
        self.use.lay.addWidget(self.use_l)
        lay.addWidget(self.use)
        lay.addStretch(1)

    def set_lists(self, strengths: list[str], limitations: list[str], failures: list[str], use: str) -> None:
        bl = lambda xs: "\n".join(f"- {x}" for x in xs) if xs else "—"
        self.pros_l.setText(bl(strengths))
        self.cons_l.setText(bl(limitations))
        self.fail_l.setText(bl(failures))
        self.fail.setVisible(bool(failures))
        self.use_l.setText(use or "—")
        self.use.setVisible(bool(use))
