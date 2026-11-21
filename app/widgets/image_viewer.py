"""Zoom/pan image viewer with optional overlay layer and pixel inspection."""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QPainter, QPixmap, QWheelEvent
from PySide6.QtWidgets import (QGraphicsPixmapItem, QGraphicsScene, QGraphicsView, QLabel,
                               QVBoxLayout, QWidget)

from app.widgets.common import np_to_pixmap


class ImageView(QGraphicsView):
    pixelHovered = Signal(int, int)        # x, y in image coordinates
    viewChanged = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setScene(QGraphicsScene(self))
        self.setRenderHints(QPainter.SmoothPixmapTransform | QPainter.Antialiasing)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setMouseTracking(True)
        self.setMinimumSize(160, 120)
        self._base = QGraphicsPixmapItem()
        self._over = QGraphicsPixmapItem()
        self._base.setTransformationMode(Qt.SmoothTransformation)
        self.scene().addItem(self._base)
        self.scene().addItem(self._over)
        self._img: np.ndarray | None = None
        self._fitted = True

    def set_image(self, img: np.ndarray | None, keep_view: bool = False) -> None:
        self._img = img
        if img is None:
            self._base.setPixmap(QPixmap())
            self._over.setPixmap(QPixmap())
            return
        self._base.setPixmap(np_to_pixmap(img))
        self.scene().setSceneRect(QRectF(0, 0, img.shape[1], img.shape[0]))
        if not keep_view or self._fitted:
            self.fit()

    def set_overlay(self, rgba: np.ndarray | None, opacity: float = 1.0) -> None:
        if rgba is None:
            self._over.setPixmap(QPixmap())
            return
        self._over.setPixmap(np_to_pixmap(rgba))
        self._over.setOpacity(opacity)

    def set_overlay_opacity(self, a: float) -> None:
        self._over.setOpacity(a)

    def fit(self) -> None:
        if self._img is not None:
            self.fitInView(self.scene().sceneRect(), Qt.KeepAspectRatio)
            self._fitted = True

    def wheelEvent(self, e: QWheelEvent) -> None:
        f = 1.25 if e.angleDelta().y() > 0 else 0.8
        self.scale(f, f)
        self._fitted = False
        self.viewChanged.emit()

    def resizeEvent(self, e) -> None:
        super().resizeEvent(e)
        if self._fitted:
            self.fit()

    def mouseMoveEvent(self, e) -> None:
        super().mouseMoveEvent(e)
        if self._img is None:
            return
        p: QPointF = self.mapToScene(e.position().toPoint())
        x, y = int(p.x()), int(p.y())
        if 0 <= x < self._img.shape[1] and 0 <= y < self._img.shape[0]:
            self.pixelHovered.emit(x, y)

    def mouseDoubleClickEvent(self, e) -> None:
        self.fit()


class TitledImage(QWidget):
    """An ImageView with a caption, used in result grids."""

    def __init__(self, title: str = "", parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(2)
        self.caption = QLabel(title)
        self.caption.setObjectName("Muted")
        self.view = ImageView()
        lay.addWidget(self.caption)
        lay.addWidget(self.view, 1)

    def set(self, img: np.ndarray | None, caption: str | None = None) -> None:
        if caption is not None:
            self.caption.setText(caption)
        self.view.set_image(img)
