"""Reusable Qt building blocks for the Educational Experiments.

* decision helpers   : :func:`is_nok`, :func:`outcome`, :func:`threshold_crossings` (pure, unit-tested)
* evidence           : :class:`ProvenanceBar` — what is STORED and what is computed LIVE
* images             : :class:`ZoomImage` (enlarge on double-click / ⤢), :class:`ImageDialog`
                       (synchronised zoom & pan), :class:`ThresholdGauge`, :class:`ConditionCard`,
                       :class:`ConditionGrid` (re-flows to the available width)
* charts             : :class:`MplCanvas`, :class:`SceneScoreChart` (one scene, or all scenes with
                       interactive selection, dimming, error marks and threshold crossings)
* tables / text      : :class:`ResultTable` (coloured, selectable rows), :class:`ExplanationBox`

The decision rule is the one of :func:`core.evaluation.metrics.decision_metrics`:
``score >= threshold`` → NOK (rejected as anomalous).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import ClassVar

import numpy as np
from PySide6.QtCore import QEvent, QPoint, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QGuiApplication, QPainter, QPen
from PySide6.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QTextBrowser,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from app.i18n import tr
from app.widgets.image_viewer import ImageView, TitledImage
from core import settings

# ------------------------------------------------------------------------- decision helpers
OUTCOME_COLORS = {"TP": "#37b24d", "TN": "#37b24d", "FP": "#f08c00", "FN": "#f03e3e"}
CLASS_COLORS = {"bad": "#ff6b6b", "good": "#4dabf7"}
THRESHOLD_COLOR = "#51cf66"


def is_nok(score: float, threshold: float) -> bool:
    """Same rule as metrics.decision_metrics: a score at or above the threshold is rejected (NOK)."""
    return bool(score >= threshold)


def outcome(anomalous: bool, nok: bool) -> str:
    """Confusion-matrix cell of one decision: TP, TN, FP (false alarm) or FN (missed defect)."""
    if anomalous:
        return "TP" if nok else "FN"
    return "FP" if nok else "TN"


def outcome_name(code: str) -> str:
    return {"TP": tr("kit.o_tp", "TP · defect detected"), "TN": tr("kit.o_tn", "TN · good part accepted"),
            "FP": tr("kit.o_fp", "FP · false alarm"), "FN": tr("kit.o_fn", "FN · missed defect")}[code]


def threshold_crossings(scores, threshold: float) -> list[tuple[int, float]]:
    """Where consecutive conditions get different decisions: (index i of the left condition,
    fractional x in [i, i+1] at which the straight segment between the two scores meets the
    threshold). The x position is only a drawing aid — conditions are categories, not a continuum."""
    s = [float(v) for v in scores]
    out = []
    for i in range(len(s) - 1):
        if is_nok(s[i], threshold) != is_nok(s[i + 1], threshold):
            d = s[i + 1] - s[i]
            f = 0.5 if d == 0 else min(1.0, max(0.0, (threshold - s[i]) / d))
            out.append((i, i + f))
    return out


def short_condition(c: str, two_lines: bool = True) -> str:
    """'overexposed' → 'over-' + newline + 'exposed', so six condition names fit under a narrow chart."""
    if two_lines and len(c) > 9 and c.endswith("exposed"):
        return c[:-7] + "-\nexposed"
    return c


def _theme() -> dict:
    from app.ui import theme
    return theme.colors(settings.get().theme)


def _mix(hex_color: str, alpha: float) -> QColor:
    c = QColor(hex_color)
    c.setAlphaF(alpha)
    return c


# ------------------------------------------------------------------------- evidence
class ProvenanceBar(QWidget):
    """Two explicit chips: what comes from STORED verified results and what is computed LIVE here."""

    STYLES: ClassVar[dict[str, str]] = {"stored": "background:#3b4a6b;color:white;padding:4px 8px;border-radius:4px",
              "live": "background:#1f6f3f;color:white;padding:4px 8px;border-radius:4px",
              "off": "background:#4a4f57;color:white;padding:4px 8px;border-radius:4px"}

    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)
        self.stored = QLabel()
        self.live = QLabel()
        for w in (self.stored, self.live):
            w.setWordWrap(True)
            w.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
            lay.addWidget(w, 1)

    def set_state(self, stored_text: str, live: bool, live_text: str):
        self.stored.setText("◆ " + stored_text)
        self.stored.setStyleSheet(self.STYLES["stored"])
        self.live.setText(("● " if live else "○ ") + live_text)
        self.live.setStyleSheet(self.STYLES["live" if live else "off"])

    def text(self) -> str:
        return self.stored.text() + "  " + self.live.text()


# ------------------------------------------------------------------------- images
class ImageDialog(QDialog):
    """Large view of one or more images of the same size with synchronised zoom and pan."""

    def __init__(self, title: str, images: list[tuple[str, np.ndarray]], parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        lay = QVBoxLayout(self)
        hint = QLabel(tr("kit.zoom_hint", "Mouse wheel: zoom · drag: pan · double-click: fit. "
                                          "All views move together."))
        hint.setObjectName("Muted")
        lay.addWidget(hint)
        grid = QGridLayout()
        lay.addLayout(grid, 1)
        imgs = [(t, i) for t, i in images if i is not None]
        wide = bool(imgs) and imgs[0][1].shape[1] >= 2 * imgs[0][1].shape[0]
        self.views: list[ImageView] = []
        for k, (t, img) in enumerate(imgs):
            cap = QLabel(t)
            cap.setWordWrap(True)
            v = ImageView()
            v.set_image(img)
            r, c = (2 * k, 0) if wide else (0, k)
            grid.addWidget(cap, r if wide else 0, c)
            grid.addWidget(v, r + 1 if wide else 1, c)
            grid.setRowStretch(r + 1 if wide else 1, 1)
            self.views.append(v)
        self._syncing = False
        for v in self.views:
            v.viewChanged.connect(lambda v=v: self._sync(v))
            v.horizontalScrollBar().valueChanged.connect(lambda _=0, v=v: self._sync(v))
            v.verticalScrollBar().valueChanged.connect(lambda _=0, v=v: self._sync(v))
        scr = QGuiApplication.primaryScreen().availableGeometry()
        self.resize(int(scr.width() * 0.85), int(scr.height() * 0.85))

    def _sync(self, src: ImageView):
        if self._syncing:
            return
        self._syncing = True
        for v in self.views:
            if v is not src:
                v.setTransform(src.transform())
                v._fitted = src._fitted
                v.horizontalScrollBar().setValue(src.horizontalScrollBar().value())
                v.verticalScrollBar().setValue(src.verticalScrollBar().value())
        self._syncing = False


class ZoomImage(TitledImage):
    """TitledImage with a wrapping caption, a ⤢ button and double-click to enlarge, and a
    ``clicked`` signal (a click without dragging) for selection."""

    clicked = Signal()

    def __init__(self, title: str = "", min_w: int = 150, min_h: int = 110, parent=None, header: bool = True):
        super().__init__(title, parent)
        self.caption.setWordWrap(True)
        # ImageView's own 160×120 minimum would make a grid of cards taller than a 1280×800 screen
        self.view.setMinimumSize(max(60, min_w - 10), max(30, min_h - (34 if header else 0)))
        lay = self.layout()
        lay.removeWidget(self.caption)
        head = QHBoxLayout()
        head.setContentsMargins(0, 0, 0, 0)
        head.addWidget(self.caption, 1)
        self.zoom_btn = QToolButton()
        self.zoom_btn.setText("⤢")
        self.zoom_btn.setAutoRaise(True)
        self.zoom_btn.setToolTip(tr("kit.enlarge", "Enlarge (or double-click the image)"))
        self.zoom_btn.clicked.connect(self.enlarge)
        head.addWidget(self.zoom_btn, 0, Qt.AlignTop)
        lay.insertLayout(0, head)
        if not header:                 # compact tile (condition cards): caption kept for tool-tips/tests only
            self.caption.hide()
            self.zoom_btn.hide()
        self.setMinimumSize(min_w, min_h)
        self.provider: Callable[[], list[tuple[str, np.ndarray]]] | None = None
        self._img: np.ndarray | None = None
        self._press: QPoint | None = None
        self.view.viewport().installEventFilter(self)
        self.zoom_btn.setEnabled(False)

    def set(self, img, caption=None):
        self._img = img
        self.zoom_btn.setEnabled(img is not None)
        super().set(img, caption)
        self.view.setToolTip(self.caption.text() + "\n" + tr("kit.enlarge", "Enlarge (or double-click the image)"))

    def images(self) -> list[tuple[str, np.ndarray]]:
        if self.provider is not None:
            return [x for x in self.provider() if x[1] is not None]
        return [(self.caption.text(), self._img)] if self._img is not None else []

    def enlarge(self):
        imgs = self.images()
        if imgs:
            self.dialog = ImageDialog(self.caption.text(), imgs, self.window())
            self.dialog.show()

    def eventFilter(self, obj, e):
        t = e.type()
        if t == QEvent.MouseButtonDblClick:
            self.enlarge()
            return True
        if t == QEvent.MouseButtonPress:
            self._press = e.position().toPoint()
        elif t == QEvent.MouseButtonRelease and self._press is not None:
            if (e.position().toPoint() - self._press).manhattanLength() < 5:
                self.clicked.emit()
            self._press = None
        return False


class ThresholdGauge(QWidget):
    """One horizontal bar: OK zone (left of the threshold), NOK zone (right), the threshold tick and
    the score as a dot coloured by its outcome. ``lo``/``hi`` are shared by all gauges of a scene so
    the positions are comparable."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(16)
        self.setMaximumHeight(18)
        self.score = self.thr = None
        self.lo, self.hi, self.color = 0.0, 1.0, "#999"

    def set_values(self, score: float, thr: float, lo: float, hi: float, color: str):
        self.score, self.thr, self.lo, self.hi, self.color = score, thr, lo, max(hi, lo + 1e-9), color
        self.setToolTip(tr("kit.gauge_tip", "Left of the tick: accepted (OK). Right of the tick: rejected (NOK)."))
        self.update()

    def _x(self, v, w):
        return 6 + (w - 12) * (min(max(v, self.lo), self.hi) - self.lo) / (self.hi - self.lo)

    def paintEvent(self, _):
        if self.score is None:
            return
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        xt = self._x(self.thr, w)
        p.fillRect(QRectF(6, h / 2 - 3, xt - 6, 6), _mix("#37b24d", 0.35))
        p.fillRect(QRectF(xt, h / 2 - 3, w - 6 - xt, 6), _mix("#f03e3e", 0.35))
        p.setPen(QPen(QColor(THRESHOLD_COLOR), 2))
        p.drawLine(int(xt), 1, int(xt), h - 1)
        p.setPen(QPen(QColor("#000000"), 1))
        p.setBrush(QColor(self.color))
        xs = self._x(self.score, w)
        p.drawEllipse(QRectF(xs - 5, h / 2 - 5, 10, 10))
        p.end()


class ConditionCard(QFrame):
    """One lighting condition of one scene: image, anomaly map, score vs threshold and the outcome."""

    clicked = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("Card")
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.lay = QVBoxLayout(self)
        self.lay.setContentsMargins(6, 4, 6, 4)
        self.lay.setSpacing(2)
        head = QHBoxLayout()
        self.title = QLabel()
        self.title.setStyleSheet("font-weight:bold")
        self.badge = QLabel()
        self.zoom_btn = QToolButton()
        self.zoom_btn.setText("⤢")
        self.zoom_btn.setAutoRaise(True)
        self.zoom_btn.setToolTip(tr("kit.enlarge_pair", "Enlarge image and anomaly map side by side (or double-click)"))
        self.live = QLabel(tr("kit.live_tag", "● LIVE"))
        self.live.setStyleSheet("color:#51cf66;font-size:8pt")
        self.live.hide()
        head.addWidget(self.title, 1)
        head.addWidget(self.live)
        head.addWidget(self.zoom_btn)
        head.addWidget(self.badge)
        self.lay.addLayout(head)
        self.pics = QGridLayout()
        self.pics.setSpacing(3)
        self.image = ZoomImage("", 70, 30, header=False)
        self.map = ZoomImage("", 70, 30, header=False)
        self.zoom_btn.clicked.connect(self.image.enlarge)
        for z in (self.image, self.map):
            z.caption.setObjectName("Muted")
            z.clicked.connect(self.clicked)
        self.lay.addLayout(self.pics, 1)
        self.no_images = QLabel(tr("kit.no_images", "image not installed — stored score"))
        self.no_images.setObjectName("Muted")
        self.no_images.hide()
        self.lay.addWidget(self.no_images)
        self.has_images = True
        row = QHBoxLayout()
        row.setSpacing(6)
        self.gauge = ThresholdGauge()
        self.gauge.setMinimumWidth(50)
        row.addWidget(self.gauge, 1)
        self.score = QLabel()
        self.score.setTextFormat(Qt.RichText)
        row.addWidget(self.score)
        self.lay.addLayout(row)
        self.code = None
        self._selected = False
        self.aspect = 2.0
        self.stacked = None
        self.set_orientation(False)

    def set_aspect(self, aspect: float):
        """Image width / height; the card then picks the arrangement that shows the images larger."""
        self.aspect = aspect or 2.0
        self._auto()

    @staticmethod
    def best_stacked(w: float, h: float, aspect: float) -> bool:
        """True if two images of this aspect appear larger stacked than side by side in a w×h area."""
        side = min(w / 2, h * aspect)          # displayed width of each image
        stack = min(w, h / 2 * aspect)
        return stack > side

    def _auto(self):
        r = self.pics.geometry()
        if r.width() > 0 and r.height() > 0:
            self.set_orientation(self.best_stacked(r.width(), r.height(), self.aspect))

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._auto()

    def set_has_images(self, on: bool):
        """Without the dataset (Student Edition, stored results) the card is compact: no empty image boxes."""
        self.has_images = on
        for w in (self.image, self.map, self.zoom_btn):
            w.setVisible(on)
        self.no_images.setVisible(not on)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding if on else QSizePolicy.Maximum)

    def set_orientation(self, stacked: bool):
        """Image and map side by side or stacked."""
        if stacked == self.stacked:
            return
        self.stacked = stacked
        for z in (self.image, self.map):
            self.pics.removeWidget(z)
        self.pics.addWidget(self.image, 0, 0)
        if stacked:
            self.pics.addWidget(self.map, 1, 0)
        else:
            self.pics.addWidget(self.map, 0, 1)

    def set_result(self, condition: str, score: float, thr: float, anomalous: bool, lo: float, hi: float):
        nok = is_nok(score, thr)
        self.code = outcome(anomalous, nok)
        col = OUTCOME_COLORS[self.code]
        self.title.setText(condition)
        self.badge.setText(f" {self.code} ")
        self.badge.setToolTip(outcome_name(self.code))
        self.badge.setStyleSheet(f"background:{col};color:black;border-radius:6px;font-weight:bold;padding:0 4px")
        self.gauge.set_values(score, thr, lo, hi, col)
        ok = self.code in ("TP", "TN")
        self.score.setText(f"{score:.2f} {'≥' if nok else '&lt;'} {thr:.2f} → <b>{'NOK' if nok else 'OK'}</b> {'✓' if ok else '✗'}")
        tip = tr("kit.card_tip", "{c}: image score {s:.3f}, threshold {t:.3f} → {d} — {o}").format(
            c=condition, s=score, t=thr, d="NOK" if nok else "OK", o=outcome_name(self.code))
        self.setToolTip(tip)
        self.score.setToolTip(tip)
        self.set_live("")
        self._style()

    def set_live(self, text: str, ok: bool = True):
        """Marks a card whose anomaly map was computed LIVE; ``text`` (live vs stored score) is its tool-tip."""
        self.live.setText(tr("kit.live_tag", "● LIVE") + ("" if ok else " ⚠"))
        self.live.setToolTip(text)
        self.live.setVisible(bool(text))

    def set_selected(self, on: bool):
        self._selected = on
        self._style()

    def _style(self):
        c = _theme()
        col = OUTCOME_COLORS.get(self.code or "", c["border"])
        w = 3 if self._selected else 1
        bg = c["sel"] if self._selected else c["panel"]
        self.setStyleSheet(f"#Card {{ border: {w}px solid {col}; border-radius: 6px; background: {bg}; }}")

    def mouseReleaseEvent(self, e):
        self.clicked.emit()
        super().mouseReleaseEvent(e)


class ConditionGrid(QWidget):
    """Cards in as many columns as fit (a divisor-friendly count: 6 → 6/3/2, 4 → 4/2)."""

    conditionClicked = Signal(int)
    MIN_CARD_W = 215

    def __init__(self, parent=None):
        super().__init__(parent)
        self.grid = QGridLayout(self)
        self.grid.setContentsMargins(0, 0, 0, 0)
        self.grid.setSpacing(6)
        self.cards: list[ConditionCard] = []
        self._cols = 0
        # the minimum width must not grow with the number of columns (that would widen the window)
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Expanding)
        self.setMinimumWidth(2 * self.MIN_CARD_W)

    def set_count(self, n: int) -> list[ConditionCard]:
        while len(self.cards) < n:
            c = ConditionCard()
            c.clicked.connect(lambda c=c: self.conditionClicked.emit(self.cards.index(c)))
            self.cards.append(c)
        for i, c in enumerate(self.cards):
            c.setVisible(i < n)
        self._n = n
        self._cols = 0
        self._reflow()
        return self.cards[:n]

    def columns_for(self, width: int) -> int:
        n = getattr(self, "_n", 0)
        if n == 0:
            return 1
        fit = max(1, width // self.MIN_CARD_W)
        options = sorted({n, (n + 1) // 2, (n + 2) // 3, 1}, reverse=True)
        return next((c for c in options if c <= fit), 1)

    def _reflow(self):
        cols = self.columns_for(self.width() or 900)
        if cols == self._cols:
            return
        self._cols = cols
        for c in self.cards:
            self.grid.removeWidget(c)
        for i, c in enumerate(self.cards[: self._n]):
            self.grid.addWidget(c, i // cols, i % cols)
        rows = (self._n + cols - 1) // cols
        for r in range(self.grid.rowCount()):            # rows left over from a previous layout get no space
            self.grid.setRowStretch(r, 1 if r < rows else 0)
        for k in range(self.grid.columnCount()):
            self.grid.setColumnStretch(k, 1 if k < cols else 0)

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._reflow()

    def select(self, i: int):
        for k, c in enumerate(self.cards):
            c.set_selected(k == i)


# ------------------------------------------------------------------------- charts
def style_ax(ax, dark: bool):
    fg = "#d8dde3" if dark else "#222"
    ax.set_facecolor("#1e2228" if dark else "white")
    for s in ax.spines.values():
        s.set_color(fg)
    ax.tick_params(colors=fg, labelsize=8)
    ax.xaxis.label.set_color(fg)
    ax.yaxis.label.set_color(fg)
    ax.title.set_color(fg)


def mpl_canvas(width=6.0, height=3.2, min_height: int | None = None):
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
    from matplotlib.figure import Figure
    dark = settings.get().theme == "dark"
    fig = Figure(figsize=(width, height), tight_layout=True, facecolor="#1e2228" if dark else "white")
    c = FigureCanvasQTAgg(fig)
    c.setMinimumHeight(int(height * 80) if min_height is None else min_height)
    c.dark = dark
    return c, fig


class MplCanvas(QWidget):
    """Matplotlib figure in a widget that may shrink (charts must not force page scrolling)."""

    def __init__(self, width=6.0, height=3.0, min_height=170, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.canvas, self.fig = mpl_canvas(width, height, min_height)
        self.dark = self.canvas.dark
        self.fg = "#d8dde3" if self.dark else "#222"
        lay.addWidget(self.canvas, 1)

    def draw(self):
        self.canvas.draw_idle()


class SceneScoreChart(MplCanvas):
    """Image anomaly score per lighting condition.

    ``plot_scene`` — Basic: one scene, OK/NOK zones, every point labelled with its outcome.
    ``plot_overview`` — Advanced: every scene as one line; the selected scene is highlighted, the
    others dimmed; its errors are marked (✗) and its threshold crossings shown (◆). Clicking a line
    selects that scene; clicking a point of the selected scene selects that condition."""

    sceneClicked = Signal(object)
    conditionClicked = Signal(int)

    def __init__(self, parent=None, min_height=170):
        super().__init__(6.5, 3.0, min_height, parent)
        self.canvas.mpl_connect("pick_event", self._pick)
        self.canvas.mpl_connect("motion_notify_event", self._hover)
        self._lines: dict = {}
        self.selected = None
        self.last = {}

    # ---- helpers
    def _zones(self, ax, thr, lo, hi):
        ax.axhspan(thr, hi, color="#f03e3e", alpha=0.07, lw=0)
        ax.axhspan(lo, thr, color="#37b24d", alpha=0.07, lw=0)
        ax.axhline(thr, color=THRESHOLD_COLOR, ls="--", lw=1.6)
        ax.set_ylim(lo, hi)

    @staticmethod
    def _limits(values, thr):
        v = list(values) + [thr]
        lo, hi = min(v), max(v)
        pad = 0.12 * (hi - lo or abs(thr) or 1)
        return lo - pad, hi + 3.0 * pad                     # room above for point labels

    def _xaxis(self, ax, conds, sel_cond):
        crowded = self.canvas.width() < 110 * len(conds)     # long condition names would overlap
        ax.set_xticks(range(len(conds)), [short_condition(c, crowded) for c in conds])
        ax.set_xlim(-0.5, len(conds) - 0.5)
        for lab in ax.get_xticklabels():
            lab.set_fontsize(8)
        if sel_cond is not None and 0 <= sel_cond < len(conds):
            ax.axvspan(sel_cond - 0.45, sel_cond + 0.45, color="#ffd43b", alpha=0.13, lw=0)
            ax.get_xticklabels()[sel_cond].set_color("#ffd43b")

    def _ylabel(self, ax):
        """Full axis label when the chart is tall enough, a short one otherwise (never clipped)."""
        full = tr("xp.ax_score", "image anomaly score")
        short = tr("kit.ax_score_short", "anomaly score")
        ax.set_ylabel(full if self.canvas.height() >= 9 * len(full) else short, fontsize=8)

    # ---- Basic
    def plot_scene(self, items: list[dict], thr: float, sel_cond: int | None = None):
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        style_ax(ax, self.dark)
        self._lines = {}
        conds = [s["lighting_condition"] for s in items]
        y = [s["score"] for s in items]
        anomalous = items[0]["label"] == "bad"
        lo, hi = self._limits(y, thr)
        self._zones(ax, thr, lo, hi)
        ax.plot(range(len(y)), y, "-", color="#868e96", lw=1, zorder=2)
        codes = [outcome(anomalous, is_nok(v, thr)) for v in y]
        pts = ax.scatter(range(len(y)), y, s=[90 if c in ("FP", "FN") else 60 for c in codes],
                         c=[OUTCOME_COLORS[c] for c in codes], edgecolors=self.fg, linewidths=0.8, zorder=4,
                         picker=6, marker="o")
        pts.set_gid("points")
        for i, (v, c) in enumerate(zip(y, codes)):
            ax.annotate(f"{v:.2f}\n{c}", (i, v), textcoords="offset points", xytext=(0, 9), ha="center",
                        fontsize=7.5, color=OUTCOME_COLORS[c], fontweight="bold")
        for _, xc in threshold_crossings(y, thr):
            ax.plot([xc], [thr], "D", color="#ffd43b", ms=7, mec=self.fg, zorder=5)
        ax.text(1.0, thr, "  " + tr("kit.thr_label", "threshold {t:.2f}").format(t=thr), transform=ax.get_yaxis_transform(),
                va="center", ha="left", fontsize=7.5, color=THRESHOLD_COLOR, clip_on=False)
        ax.text(0.01, 0.97, "▲ NOK — " + tr("kit.reject", "rejected"), transform=ax.transAxes, va="top", fontsize=7.5, color="#f03e3e")
        ax.text(0.01, 0.03, "▼ OK — " + tr("kit.accept", "accepted"), transform=ax.transAxes, va="bottom", fontsize=7.5, color="#37b24d")
        self._xaxis(ax, conds, sel_cond)
        self._ylabel(ax)
        self.last = {"mode": "scene", "codes": codes, "conds": conds}
        self.draw()

    # ---- Advanced
    def plot_overview(self, groups: dict, conds: list[str], thr: float, selected=None, sel_cond: int | None = None,
                      dim: bool = True, only=None):
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        style_ax(ax, self.dark)
        self.selected = selected
        self._lines = {}
        allv = [s["score"] for v in groups.values() for s in v]
        lo, hi = self._limits(allv, thr)
        self._zones(ax, thr, lo, hi)
        errors, crossings = [], []
        for key, items in groups.items():
            if only is not None and key not in only and key != selected:
                continue
            x = [conds.index(s["lighting_condition"]) for s in items]
            y = [s["score"] for s in items]
            col = CLASS_COLORS[key[0]]
            sel = key == selected
            lw, al, z = (2.8, 1.0, 6) if sel else ((0.9, 0.2, 2) if (dim and selected is not None) else (1.1, 0.6, 3))
            (ln,) = ax.plot(x, y, "-o", color=col, lw=lw, alpha=al, ms=4 if sel else 2.5, zorder=z, picker=5)
            ln.set_gid(f"{key[0]}:{key[1]}")
            self._lines[ln] = key
            if sel:
                for xi, yi in zip(x, y):
                    c = outcome(key[0] == "bad", is_nok(yi, thr))
                    if c in ("FP", "FN"):
                        errors.append((xi, yi, c))
                crossings = [(xc, thr) for _, xc in threshold_crossings(y, thr)]
        for xi, yi, c in errors:
            ax.plot([xi], [yi], "X", color=OUTCOME_COLORS[c], ms=11, mec=self.fg, mew=0.8, zorder=8)
            ax.annotate(c, (xi, yi), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=7.5,
                        color=OUTCOME_COLORS[c], fontweight="bold", zorder=9)
        for xc, yc in crossings:
            ax.plot([xc], [yc], "D", color="#ffd43b", ms=8, mec=self.fg, zorder=9)
        self._xaxis(ax, conds, sel_cond)
        self._ylabel(ax)
        from matplotlib.lines import Line2D
        handles = [Line2D([], [], color=CLASS_COLORS["bad"], lw=1.5, label=tr("kit.lg_anom", "anomalous scene")),
                   Line2D([], [], color=CLASS_COLORS["good"], lw=1.5, label=tr("kit.lg_norm", "normal scene")),
                   Line2D([], [], color=THRESHOLD_COLOR, ls="--", lw=1.5, label=tr("kit.lg_thr", "threshold")),
                   Line2D([], [], marker="X", ls="", color=OUTCOME_COLORS["FN"], ms=8, label=tr("kit.lg_err", "error (FP/FN)")),
                   Line2D([], [], marker="D", ls="", color="#ffd43b", ms=6, label=tr("kit.lg_cross", "decision changes"))]
        leg = ax.legend(handles=handles, fontsize=7, loc="lower center", bbox_to_anchor=(0.5, 1.0),
                        ncol=5 if self.canvas.width() > 720 else 3,
                        framealpha=0.2, labelcolor=self.fg, borderaxespad=0.3, handlelength=1.5, columnspacing=1.0)
        leg.get_frame().set_edgecolor(self.fg)
        self.last = {"mode": "overview", "errors": errors, "crossings": crossings,
                     "highlight": selected, "n_lines": len(self._lines)}
        self.draw()

    # ---- interaction
    def _pick(self, ev):
        a = ev.artist
        if a.get_gid() == "points":
            self.conditionClicked.emit(int(ev.ind[0]))
            return
        key = self._lines.get(a)
        if key is None:
            return
        if key == self.selected and len(ev.ind):
            x = a.get_xdata()[int(ev.ind[0])]
            self.conditionClicked.emit(int(x))
        else:
            self.sceneClicked.emit(key)

    def _hover(self, ev):
        if ev.inaxes is None or not self._lines:
            self.canvas.setToolTip("")
            return
        for ln, key in self._lines.items():
            if ln.contains(ev)[0]:
                kind = tr("xp.anomalous", "anomalous") if key[0] == "bad" else tr("xp.normal", "normal")
                self.canvas.setToolTip(f"{tr('xp.scene', 'scene')} {key[1]} ({kind}) — " + tr("kit.click_select", "click to select"))
                return
        self.canvas.setToolTip("")


# ------------------------------------------------------------------------- tables and text
class ResultTable(QTableWidget):
    """Read-only table with optional per-cell colours/tool-tips and row payloads (``rowSelected``)."""

    rowSelected = Signal(object)

    def __init__(self, header: list[str] | None = None, parent=None):
        super().__init__(0, len(header or []), parent)
        if header:
            self.setHorizontalHeaderLabels(header)
        self.verticalHeader().setVisible(False)
        self.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.setSelectionMode(QAbstractItemView.SingleSelection)
        self.setWordWrap(False)
        self.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.horizontalHeader().setStretchLastSection(True)
        self.itemSelectionChanged.connect(self._sel)
        self._payload: list = []

    def set_rows(self, rows: list[list], header: list[str] | None = None, payload: list | None = None):
        self.blockSignals(True)
        if header is not None:
            self.setColumnCount(len(header))
            self.setHorizontalHeaderLabels(header)
        self.setRowCount(len(rows))
        for i, r in enumerate(rows):
            for j, cell in enumerate(r):
                d = cell if isinstance(cell, dict) else {"text": cell}
                it = QTableWidgetItem(str(d.get("text", "")))
                if d.get("bg"):
                    it.setBackground(_mix(d["bg"], d.get("alpha", 0.35)))
                if d.get("fg"):
                    it.setForeground(QColor(d["fg"]))
                if d.get("tip"):
                    it.setToolTip(d["tip"])
                if d.get("bold"):
                    f = it.font()
                    f.setBold(True)
                    it.setFont(f)
                if d.get("center"):
                    it.setTextAlignment(Qt.AlignCenter)
                self.setItem(i, j, it)
        self._payload = list(payload or [])
        self.blockSignals(False)

    def payload(self, row: int):
        return self._payload[row] if 0 <= row < len(self._payload) else None

    def select_payload(self, data):
        if data in self._payload:
            self.blockSignals(True)
            r = self._payload.index(data)
            self.selectRow(r)
            self.scrollToItem(self.item(r, 0))
            self.blockSignals(False)
        else:
            self.clearSelection()

    def _sel(self):
        rows = self.selectionModel().selectedRows()
        if rows:
            p = self.payload(rows[0].row())
            if p is not None:
                self.rowSelected.emit(p)


class ExplanationBox(QTextBrowser):
    """Markdown explanation panel (links open in the system browser)."""

    def __init__(self, md: str = "", min_height: int = 0, parent=None):
        super().__init__(parent)
        self.setOpenExternalLinks(True)
        if min_height:
            self.setMinimumHeight(min_height)
        self.setMarkdown(md)

    def set_markdown(self, md: str):
        v = self.verticalScrollBar().value()
        self.setMarkdown(md)
        self.verticalScrollBar().setValue(min(v, self.verticalScrollBar().maximum()))
