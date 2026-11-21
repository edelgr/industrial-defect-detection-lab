"""Light / dark themes with a 'scientific instrument' look (Qt style sheets)."""

from __future__ import annotations

from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QApplication

PALETTES = {
    "dark": dict(bg="#11161c", panel="#171e26", panel2="#1d2630", border="#2a3542", text="#d8e0e8",
                 muted="#8a98a8", accent="#2bb3a3", accent2="#f0a742", danger="#e5534b", ok="#57ab5a",
                 link="#58b6f0", sel="#244a5a"),
    "light": dict(bg="#eef1f4", panel="#ffffff", panel2="#f5f7f9", border="#cfd6de", text="#1b232c",
                  muted="#5b6875", accent="#107d72", accent2="#b56d00", danger="#c4302b", ok="#2e7d32",
                  link="#0b63a8", sel="#cfe7ec"),
}


def colors(name: str) -> dict:
    return PALETTES.get(name, PALETTES["dark"])


def stylesheet(name: str) -> str:
    c = colors(name)
    return f"""
    QWidget {{ background: {c['bg']}; color: {c['text']}; font-family: 'Segoe UI', 'Inter', sans-serif; font-size: 10pt; }}
    QToolTip {{ background: {c['panel2']}; color: {c['text']}; border: 1px solid {c['border']}; padding: 4px; }}
    #Sidebar {{ background: {c['panel']}; border-right: 1px solid {c['border']}; }}
    #Sidebar QListWidget {{ background: transparent; border: none; outline: 0; }}
    #Sidebar QListWidget::item {{ padding: 9px 12px; border-radius: 6px; margin: 1px 6px; }}
    #Sidebar QListWidget::item:selected {{ background: {c['sel']}; color: {c['text']}; border-left: 3px solid {c['accent']}; }}
    #Sidebar QListWidget::item:hover {{ background: {c['panel2']}; }}
    #AppTitle {{ font-size: 15pt; font-weight: 600; color: {c['accent']}; background: transparent; }}
    #AppSubtitle {{ color: {c['muted']}; font-size: 8.5pt; background: transparent; }}
    #PageTitle {{ font-size: 16pt; font-weight: 600; background: transparent; }}
    #PageSubtitle {{ color: {c['muted']}; background: transparent; }}
    #StatusChip {{ border: 1px solid {c['border']}; border-radius: 10px; padding: 2px 10px; background: {c['panel']}; color: {c['muted']}; }}
    #Card, QFrame#Card {{ background: {c['panel']}; border: 1px solid {c['border']}; border-radius: 8px; }}
    #Card:hover {{ border: 1px solid {c['accent']}; }}
    #CardTitle {{ font-size: 11.5pt; font-weight: 600; background: transparent; }}
    #Muted {{ color: {c['muted']}; background: transparent; }}
    #Badge {{ border-radius: 8px; padding: 1px 7px; font-size: 8.5pt; background: {c['panel2']}; border: 1px solid {c['border']}; }}
    #BadgeOk {{ border-radius: 8px; padding: 1px 7px; font-size: 8.5pt; color: {c['ok']}; border: 1px solid {c['ok']}; background: transparent; }}
    #BadgeWarn {{ border-radius: 8px; padding: 1px 7px; font-size: 8.5pt; color: {c['accent2']}; border: 1px solid {c['accent2']}; background: transparent; }}
    #BadgeBad {{ border-radius: 8px; padding: 1px 7px; font-size: 8.5pt; color: {c['danger']}; border: 1px solid {c['danger']}; background: transparent; }}
    #SyntheticBanner {{ background: {c['accent2']}; color: #111; font-weight: 600; padding: 4px 8px; border-radius: 4px; }}
    #RefPanel {{ background: {c['panel2']}; border: 1px solid {c['accent']}; border-radius: 8px; }}
    #RefTitle {{ font-weight: 600; color: {c['accent']}; background: transparent; font-size: 11pt; }}
    QPushButton {{ background: {c['panel2']}; border: 1px solid {c['border']}; border-radius: 5px; padding: 5px 12px; }}
    QPushButton:hover {{ border-color: {c['accent']}; }}
    QPushButton:pressed {{ background: {c['sel']}; }}
    QPushButton:disabled {{ color: {c['muted']}; border-color: {c['border']}; }}
    QPushButton#Primary {{ background: {c['accent']}; color: #fff; border: none; font-weight: 600; }}
    QPushButton#Primary:disabled {{ background: {c['border']}; color: {c['muted']}; }}
    QPushButton#LinkButton {{ text-align: left; color: {c['link']}; background: transparent; border: none; padding: 3px 4px; }}
    QPushButton#LinkButton:hover {{ text-decoration: underline; }}
    QPushButton#LinkButton:disabled {{ color: {c['muted']}; text-decoration: none; }}
    QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QPlainTextEdit {{ background: {c['panel']}; border: 1px solid {c['border']}; border-radius: 4px; padding: 4px 6px; selection-background-color: {c['sel']}; }}
    QComboBox QAbstractItemView {{ background: {c['panel']}; selection-background-color: {c['sel']}; }}
    QTextBrowser {{ background: {c['panel']}; border: 1px solid {c['border']}; border-radius: 6px; padding: 8px; }}
    QTableWidget, QTableView, QTreeWidget, QListWidget, QListView {{ background: {c['panel']}; border: 1px solid {c['border']}; gridline-color: {c['border']}; alternate-background-color: {c['panel2']}; selection-background-color: {c['sel']}; selection-color: {c['text']}; }}
    QHeaderView::section {{ background: {c['panel2']}; border: none; border-bottom: 1px solid {c['border']}; padding: 5px; font-weight: 600; }}
    QTabWidget::pane {{ border: 1px solid {c['border']}; border-radius: 6px; top: -1px; }}
    QTabBar::tab {{ background: transparent; padding: 7px 14px; border-bottom: 2px solid transparent; color: {c['muted']}; }}
    QTabBar::tab:selected {{ color: {c['text']}; border-bottom: 2px solid {c['accent']}; }}
    QGroupBox {{ border: 1px solid {c['border']}; border-radius: 6px; margin-top: 14px; padding-top: 6px; }}
    QGroupBox::title {{ subcontrol-origin: margin; left: 10px; padding: 0 4px; color: {c['muted']}; }}
    QSplitter::handle {{ background: {c['border']}; }}
    QSplitter::handle:horizontal {{ width: 1px; }}
    QSplitter::handle:vertical {{ height: 1px; }}
    QScrollArea {{ border: none; }}
    QProgressBar {{ border: 1px solid {c['border']}; border-radius: 4px; text-align: center; background: {c['panel']}; height: 14px; }}
    QProgressBar::chunk {{ background: {c['accent']}; border-radius: 3px; }}
    QSlider::groove:horizontal {{ height: 4px; background: {c['border']}; border-radius: 2px; }}
    QSlider::handle:horizontal {{ background: {c['accent']}; width: 14px; margin: -6px 0; border-radius: 7px; }}
    QStatusBar {{ background: {c['panel']}; border-top: 1px solid {c['border']}; color: {c['muted']}; }}
    QCheckBox, QRadioButton, QLabel {{ background: transparent; }}
    """


def apply(app: QApplication, name: str) -> None:
    app.setStyle("Fusion")
    c = colors(name)
    pal = QPalette()
    pal.setColor(QPalette.Window, QColor(c["bg"]))
    pal.setColor(QPalette.WindowText, QColor(c["text"]))
    pal.setColor(QPalette.Base, QColor(c["panel"]))
    pal.setColor(QPalette.AlternateBase, QColor(c["panel2"]))
    pal.setColor(QPalette.Text, QColor(c["text"]))
    pal.setColor(QPalette.Button, QColor(c["panel2"]))
    pal.setColor(QPalette.ButtonText, QColor(c["text"]))
    pal.setColor(QPalette.Highlight, QColor(c["sel"]))
    pal.setColor(QPalette.HighlightedText, QColor(c["text"]))
    pal.setColor(QPalette.Link, QColor(c["link"]))
    pal.setColor(QPalette.PlaceholderText, QColor(c["muted"]))
    app.setPalette(pal)
    app.setStyleSheet(stylesheet(name))
