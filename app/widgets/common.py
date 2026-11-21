"""Small shared widgets and helpers."""

from __future__ import annotations

import cv2
import numpy as np
from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QDesktopServices, QGuiApplication, QImage, QPixmap
from PySide6.QtWidgets import (QFrame, QHBoxLayout, QLabel, QLayout, QSizePolicy, QTextBrowser,
                               QVBoxLayout, QWidget)


def open_url(url: str) -> bool:
    return QDesktopServices.openUrl(QUrl(url))


def copy_text(text: str) -> None:
    QGuiApplication.clipboard().setText(text)


def np_to_qimage(a: np.ndarray) -> QImage:
    a = np.ascontiguousarray(a)
    if a.ndim == 2:
        if a.dtype != np.uint8:
            a = cv2.normalize(a, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        h, w = a.shape
        return QImage(a.data, w, h, w, QImage.Format_Grayscale8).copy()
    h, w, ch = a.shape
    if ch == 4:
        return QImage(a.data, w, h, 4 * w, QImage.Format_RGBA8888).copy()
    return QImage(a.data, w, h, 3 * w, QImage.Format_RGB888).copy()


def np_to_pixmap(a: np.ndarray) -> QPixmap:
    return QPixmap.fromImage(np_to_qimage(a))


def colorize(amap: np.ndarray, vmin: float | None = None, vmax: float | None = None,
             cmap: int = cv2.COLORMAP_INFERNO) -> np.ndarray:
    """Map a float array to RGB using a fixed or per-image scale."""
    a = amap.astype(np.float32)
    lo = float(np.nanmin(a)) if vmin is None else vmin
    hi = float(np.nanmax(a)) if vmax is None else vmax
    n = np.zeros_like(a) if hi <= lo else np.clip((a - lo) / (hi - lo), 0, 1)
    rgb = cv2.applyColorMap((n * 255).astype(np.uint8), cmap)
    return cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)


def overlay(img: np.ndarray, color_map: np.ndarray, alpha: float) -> np.ndarray:
    return cv2.addWeighted(img, 1 - alpha, color_map, alpha, 0)


def mask_overlay(img: np.ndarray, mask: np.ndarray, color=(255, 60, 60), alpha=0.45, contour=True) -> np.ndarray:
    out = img.copy()
    m = mask.astype(bool)
    if m.shape != img.shape[:2]:
        m = cv2.resize(m.astype(np.uint8), (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
    col = np.zeros_like(img)
    col[:] = color
    out[m] = (out[m] * (1 - alpha) + col[m] * alpha).astype(np.uint8)
    if contour:
        cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(out, cs, -1, color, max(1, img.shape[0] // 400))
    return out


class Badge(QLabel):
    def __init__(self, text: str, kind: str = "", parent=None):
        super().__init__(text, parent)
        self.setObjectName({"ok": "BadgeOk", "warn": "BadgeWarn", "bad": "BadgeBad"}.get(kind, "Badge"))
        self.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)


class Card(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("Card")
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)
        self.lay = QVBoxLayout(self)
        self.lay.setContentsMargins(14, 12, 14, 12)
        self.lay.setSpacing(6)


def title_label(text: str, obj="CardTitle") -> QLabel:
    lab = QLabel(text)
    lab.setObjectName(obj)
    lab.setWordWrap(True)
    return lab


def amp(text: str) -> str:
    """Text for buttons and tabs: Qt reads a single '&' as a keyboard-shortcut marker (it would show
    'Align _compute'); '&&' displays a literal ampersand."""
    return text.replace("&", "&&")


def muted(text: str) -> QLabel:
    lab = QLabel(text)
    lab.setObjectName("Muted")
    lab.setWordWrap(True)
    return lab


def markdown_browser(md: str = "", min_height: int = 0) -> QTextBrowser:
    tb = QTextBrowser()
    tb.setOpenExternalLinks(True)      # links open in the system browser
    tb.setMarkdown(md)
    if min_height:
        tb.setMinimumHeight(min_height)
    return tb


def clear_layout(layout: QLayout) -> None:
    while layout.count():
        item = layout.takeAt(0)
        w = item.widget()
        if w is not None:
            w.deleteLater()
        elif item.layout() is not None:
            clear_layout(item.layout())


class PageHeader(QWidget):
    def __init__(self, title: str, subtitle: str = "", parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 0, 4, 6)
        lay.setSpacing(0)
        self.t = title_label(title, "PageTitle")
        self.s = title_label(subtitle, "PageSubtitle")
        lay.addWidget(self.t)
        if subtitle:
            lay.addWidget(self.s)


def hbox(*widgets, spacing=6, stretch_end=False) -> QHBoxLayout:
    lay = QHBoxLayout()
    lay.setSpacing(spacing)
    for w in widgets:
        if isinstance(w, QLayout):
            lay.addLayout(w)
        else:
            lay.addWidget(w)
    if stretch_end:
        lay.addStretch(1)
    return lay


def synthetic_banner(text: str = "SYNTHETIC EDUCATIONAL DATA — NOT A REAL INDUSTRIAL BENCHMARK. Procedurally "
                     "generated (fixed seed); results are illustrative only.") -> QLabel:
    lab = QLabel(text)
    lab.setObjectName("SyntheticBanner")
    lab.setWordWrap(True)
    lab.setAlignment(Qt.AlignCenter)
    return lab
