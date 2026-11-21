"""Reference comparison view for the classical *Reference image difference* method.

Shows the real pipeline of one comparison, using only arrays produced by the algorithm itself
(``MethodResult.extras["stages"]``, see ``core.algorithms.classical.methods._prep_pair``):

  Reference image · Test image · Test aligned onto the reference (ECC) · Difference map · Detected differences

plus an information card (identifiers, global score vs. local differences, and a log of the
references tried on the current test image). Nothing is recomputed here except the display mask at
the Lab's current pixel threshold and its connected-region count.
"""

from __future__ import annotations

import cv2
import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QGridLayout, QHBoxLayout, QLabel, QPushButton, QSizePolicy, QTextBrowser,
                               QToolButton, QVBoxLayout, QWidget)

from app.i18n import tr
from app.widgets.common import colorize, mask_overlay
from app.widgets.experiment_kit import ImageDialog, ZoomImage
from core.datasets.base import UNKNOWN

PANEL_METHODS = {"reference_difference"}     # methods whose results carry extras["stages"] for this view


def labels_withheld(sample) -> bool:
    """True for images whose ground truth is not available locally (MVTec AD 2 test_private*)."""
    return sample is not None and (sample.label == UNKNOWN or sample.split.startswith("test_private"))


def describe(sample, dataset_name: str = "") -> str:
    """'dataset · category · split · id' — the exact identity of an image."""
    if sample is None:
        return "—"
    ds = dataset_name or sample.dataset_id
    return f"{ds} · {sample.category} · {sample.split} · {sample.sample_id}"


def display_mask(result, thr: float) -> tuple[np.ndarray, str]:
    """(mask, operator). At the method's own threshold this is the algorithm's mask (``map > threshold``);
    any other threshold (calibrated or slider) uses the Lab convention ``map ≥ threshold``
    (``MethodResult.with_threshold``). The difference matters: the map holds integer grey levels."""
    if result.mask is not None and result.threshold is not None and thr == result.threshold:
        return result.mask, ">"
    return result.anomaly_map >= thr, "≥"


def n_regions(mask: np.ndarray) -> int:
    n, _ = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
    return max(0, n - 1)


_TITLES = {   # key: (short tile title, full title)
    "ref": (("ref.t_ref", "Reference"), ("ref.p_ref", "Reference image")),
    "test": (("ref.t_test", "Test"), ("ref.p_test", "Test image")),
    "aligned": (("ref.t_aligned", "Aligned (ECC)"), ("ref.p_aligned", "Test aligned to reference (ECC)")),
    "diff": (("ref.t_diff", "Difference"), ("ref.p_diff", "Difference map |test − reference|")),
    "mask": (("ref.t_mask", "Detected"), ("ref.p_mask", "Detected differences")),
}


def _t(key: str, full: bool = False) -> str:
    return tr(*_TITLES[key][1 if full else 0])


def _set(panel, img, short: str, full: str) -> None:
    """Short caption on the tile (narrow 1280-px layouts); the full description as tool-tip."""
    panel.set(img, short)
    panel.caption.setToolTip(full)
    panel.view.setToolTip(full + "\n" + tr("kit.enlarge", "Enlarge (or double-click the image)"))


def _rgb(gray: np.ndarray) -> np.ndarray:
    return np.repeat(gray[..., None], 3, axis=2) if gray.ndim == 2 else gray


def explanation_md() -> str:
    return tr("ref.explain", """\
### How *reference image difference* works

1. **Choose one normal image** in `Reference (normal)` — a *train/good* image, never a test image.
2. **Grey levels.** Both images are converted to grey; if their sizes differ, the reference is resized to the test size.
3. **Alignment (ECC).** The *test image* is warped onto the reference (translation, rotation or affine) so that the same
   part of the object falls on the same pixels. `none` skips this step. If ECC does not converge, no warp is applied
   (the caption says so).
4. **Pre-blur.** Both images are smoothed (Gaussian, `blur` px) to suppress noise and sub-pixel misalignment.
5. **Difference map** = |test − reference| for every pixel, in grey levels (0–255).
6. **Threshold.** Pixels whose difference reaches the pixel threshold become *detected differences* (the mask).

**Global score vs. local differences.** The *image score* is **one number** for the whole image (the 99.9th percentile of
the difference map); an OK/NOK decision compares it with an image threshold calibrated on other normal images. The
*mask* is **local**: it tells *where* the two images differ. A small, strong difference can give a high score with very
few pixels; a broad, weak difference can give many pixels and a moderate score.

**Parameters.** `align` — ECC motion model; `blur` — odd kernel size (larger: less noise, but small defects fade);
`threshold` — grey levels (default 30). The pixel-threshold slider re-thresholds the cached map without re-running.

**Limitations.** The method detects *any* difference, not only defects: normal part-to-part variation, lighting changes,
shadows, reflections and residual misalignment all appear in the map. The result **depends on which normal image is the
reference** — step through several references with ◀ ▶, press *Compare*, and look at the table. Large shifts or
deformable objects make ECC fail. It works best on rigid, repeatable, well-registered scenes.

**Withheld labels.** For `test_private` images the ground truth is withheld by the benchmark: an OK/NOK shown there is
only the method's *prediction*, and no metrics are computed on them.""")


class ReferenceComparePanel(QWidget):
    compareRequested = Signal()
    stepRequested = Signal(int)          # -1 previous reference, +1 next reference

    def __init__(self, parent=None):
        super().__init__(parent)
        self.ref_sample = None
        self.ref_image: np.ndarray | None = None
        self.ref_dataset = ""
        self.test_sample = None
        self.test_image: np.ndarray | None = None
        self.result = None
        self.result_ref_id: str | None = None
        self.stale = False
        self.log: list[dict] = []
        self.shown_mask: np.ndarray | None = None

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)
        head = QHBoxLayout()
        head.setContentsMargins(0, 0, 0, 0)
        self.prev_btn = QToolButton()
        self.prev_btn.setText("◀")
        self.prev_btn.setToolTip(tr("ref.prev", "Previous normal reference"))
        self.next_btn = QToolButton()
        self.next_btn.setText("▶")
        self.next_btn.setToolTip(tr("ref.next", "Next normal reference"))
        self.ref_info = QLabel("—")
        self.ref_info.setWordWrap(True)
        self.ref_info.setTextFormat(Qt.RichText)
        self.ref_info.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.ref_info.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)   # never widens the page
        self.compare_btn = QPushButton("▶  " + tr("ref.compare", "Compare"))
        self.compare_btn.setObjectName("Primary")
        self.compare_btn.setToolTip(tr("ref.compare_tip", "Run the comparison of the test image with this reference"))
        head.addWidget(self.prev_btn)
        head.addWidget(self.next_btn)
        head.addWidget(self.ref_info, 1)
        head.addWidget(self.compare_btn)
        lay.addLayout(head)
        self.status = QLabel("")
        self.status.setWordWrap(True)
        self.status.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        lay.addWidget(self.status)

        self.grid = QGridLayout()
        self.grid.setContentsMargins(0, 0, 0, 0)
        self.grid.setSpacing(4)
        self.p_ref = ZoomImage(_t("ref"), 90, 70)
        self.p_test = ZoomImage(_t("test"), 90, 70)
        self.p_aligned = ZoomImage(_t("aligned"), 90, 70)
        self.p_diff = ZoomImage(_t("diff"), 90, 70)
        self.p_mask = ZoomImage(_t("mask"), 90, 70)
        self.panels = [self.p_ref, self.p_test, self.p_aligned, self.p_diff, self.p_mask]
        for p, k in zip(self.panels, ("ref", "test", "aligned", "diff", "mask")):
            p.caption.setToolTip(_t(k, full=True))
        self.info = QTextBrowser()
        self.info.setMinimumSize(90, 70)
        self.info.setOpenExternalLinks(False)
        self.columns = 0
        self.relayout(3)
        lay.addLayout(self.grid, 1)
        hint = QLabel(tr("kit.zoom_hint", "Mouse wheel: zoom · drag: pan · double-click: fit. All views move together."))
        hint.setObjectName("Muted")
        hint.setWordWrap(True)
        hint.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        lay.addWidget(hint)

        for p in self.panels:
            p.provider = self.all_images          # ⤢ / double-click: all stages, synchronised
        self._syncing = False
        for p in self.panels:
            v = p.view
            v.viewChanged.connect(lambda v=v: self._sync(v))
            v.horizontalScrollBar().valueChanged.connect(lambda _=0, v=v: self._sync(v))
            v.verticalScrollBar().valueChanged.connect(lambda _=0, v=v: self._sync(v))
        self.prev_btn.clicked.connect(lambda: self.stepRequested.emit(-1))
        self.next_btn.clicked.connect(lambda: self.stepRequested.emit(1))
        self.compare_btn.clicked.connect(self.compareRequested.emit)
        self._render_info()

    def relayout(self, columns: int) -> None:
        """3 columns for ordinary images; 2 columns for panoramic ones (e.g. MVTec AD 2 sheet_metal, 4:1),
        which would otherwise become thin strips on a 1280-px screen."""
        if columns == self.columns:
            return
        for w in self.panels + [self.info]:
            self.grid.removeWidget(w)
        for i in range(4):
            self.grid.setColumnStretch(i, 0)
            self.grid.setRowStretch(i, 0)
        for i, w in enumerate(self.panels + [self.info]):
            self.grid.addWidget(w, i // columns, i % columns)
        for c in range(columns):
            self.grid.setColumnStretch(c, 1)
        for r in range(-(-6 // columns)):
            self.grid.setRowStretch(r, 1)
        self.columns = columns

    # ------------------------------------------------------------ zoom / pan
    def _sync(self, src):
        if self._syncing:
            return
        self._syncing = True
        for p in self.panels:
            v = p.view
            if v is not src and v._img is not None:
                v.setTransform(src.transform())
                v._fitted = src._fitted
                v.horizontalScrollBar().setValue(src.horizontalScrollBar().value())
                v.verticalScrollBar().setValue(src.verticalScrollBar().value())
        self._syncing = False

    def all_images(self):
        return [(p.caption.text(), p._img) for p in self.panels if p._img is not None]

    def enlarge_all(self) -> ImageDialog | None:
        imgs = self.all_images()
        if not imgs:
            return None
        self.dialog = ImageDialog(tr("ref.tab", "Reference comparison"), imgs, self.window())
        self.dialog.show()
        return self.dialog

    # ------------------------------------------------------------ inputs
    def set_reference(self, sample, image: np.ndarray | None, dataset_name: str = "") -> None:
        """Show the reference exactly as the session will use it (``LabSession._ref_img``)."""
        self.ref_sample, self.ref_image, self.ref_dataset = sample, image, dataset_name
        self.ref_info.setText(f"<b>{tr('ref.reference', 'Reference')}:</b> {describe(sample, dataset_name)}")
        self.ref_info.setToolTip(str(sample.primary_path) if sample is not None else "")
        full = _t("ref", full=True)
        full = f"{full}: {describe(sample, dataset_name)}" if sample is not None else full
        if image is not None:
            full += f" ({image.shape[1]}×{image.shape[0]} px)"
        _set(self.p_ref, image, _t("ref"), full)
        if self.result is not None and sample is not None and sample.sample_id != self.result_ref_id:
            self.invalidate()
        self._render_info()

    def set_test(self, sample, image: np.ndarray | None) -> None:
        if sample is not None and (self.test_sample is None or sample.sample_id != self.test_sample.sample_id):
            self.log = []                                   # the log compares references for ONE test image
        self.test_sample, self.test_image = sample, image
        lbl = "" if sample is None else (tr("ref.withheld", "label withheld") if labels_withheld(sample)
                                         else (sample.defect_type or sample.label))
        full = _t("test", full=True)
        full = f"{full}: {describe(sample)} ({lbl})" if sample is not None else full
        _set(self.p_test, image, _t("test"), full)
        if image is not None:
            self.relayout(2 if image.shape[1] >= 2 * image.shape[0] else 3)
        self._clear_outputs()
        self.result, self.result_ref_id, self.stale = None, None, False
        self._render_info()

    def invalidate(self) -> None:
        """The reference changed: results computed with the previous one are no longer shown."""
        self.result, self.result_ref_id, self.stale = None, None, True
        self._clear_outputs()
        self._render_info()

    def _clear_outputs(self):
        self.shown_mask = None
        self.p_aligned.set(None, _t("aligned"))
        self.p_diff.set(None, _t("diff"))
        self.p_mask.set(None, _t("mask"))

    # ------------------------------------------------------------ result
    def set_result(self, result, ref_id: str, thr: float, thr_source: str, params: dict,
                   decision: str | None = None, log: bool = True) -> None:
        st = result.extras.get("stages") or {}
        self.result, self.result_ref_id, self.stale = result, ref_id, False
        amap = result.anomaly_map
        align = params.get("align", "none")
        aligned = bool(result.extras.get("aligned"))
        t_al = _t("aligned", full=True)
        if align == "none":
            cap_a = full_a = tr("ref.align_off", "Alignment off — test used as is")
        elif aligned:
            w = st.get("warp")
            shift = f"Δx {w[0, 2]:+.1f}, Δy {w[1, 2]:+.1f} px" if w is not None else ""
            cap_a = f"{_t('aligned')} · {shift}"
            full_a = f"{t_al} — {align}; {shift}. " + tr("ref.aligned_note", "The algorithm warps the TEST image into "
                                                                           "the reference frame; maps and masks are in that frame.")
        else:
            cap_a = full_a = tr("ref.ecc_failed", "ECC did not converge — no warp applied")
        _set(self.p_aligned, _rgb(st["test_aligned"]) if "test_aligned" in st else None, cap_a, full_a)
        vmax = max(2.0 * thr, 1.0)
        t_d = _t("diff", full=True)
        _set(self.p_diff, colorize(amap, 0.0, vmax), f"{_t('diff')} · 0–{vmax:.0f}",
             f"{t_d}: {tr('ref.scale_note', 'fixed colour scale 0 – 2 × threshold (grey levels), comparable between references')}"
             f" = 0–{vmax:.0f}; max {amap.max():.0f}; "
             + tr("ref.after_blur", "computed after alignment and pre-blur") + f" ({st.get('blur_kernel', '?')} px)")
        self.update_threshold(thr, thr_source)
        if log:
            m = self.shown_mask
            row = {"ref": ref_id, "score": result.score, "px": float(100 * m.mean()), "regions": n_regions(m),
                   "thr": thr, "align": align, "aligned": aligned, "blur": params.get("blur"), "decision": decision}
            self.log = [r for r in self.log if not (r["ref"] == ref_id and r["align"] == align
                                                    and r["blur"] == row["blur"])] + [row]
        self._render_info()

    def update_threshold(self, thr: float, thr_source: str) -> None:
        r = self.result
        if r is None or r.anomaly_map is None:
            return
        st = r.extras.get("stages") or {}
        m, op = display_mask(r, thr)
        self.shown_mask, self.shown_op = m, op
        base = _rgb(st["test_aligned"]) if "test_aligned" in st else self.test_image
        t_m = _t("mask", full=True)
        _set(self.p_mask, mask_overlay(base, m, color=(255, 200, 0)), f"{_t('mask')} {op} {thr:.4g} · {100 * m.mean():.2f} %",
             f"{t_m}: |test − ref| {op} {thr:.4g} ({thr_source}) · {100 * m.mean():.2f} % px · {n_regions(m)} "
             f"{tr('ref.regions', 'regions')}")
        self._thr, self._thr_source = thr, thr_source
        self._render_info()

    # ------------------------------------------------------------ info card
    def _render_info(self):
        t = []
        t.append(f"<p><b>{tr('ref.reference', 'Reference')}:</b> {describe(self.ref_sample, self.ref_dataset)}<br>"
                 f"<b>{tr('ref.test', 'Test')}:</b> {describe(self.test_sample, self.ref_dataset)}</p>")
        if self.test_sample is not None and self.ref_sample is not None \
                and self.test_sample.sample_id == self.ref_sample.sample_id:
            t.append(f"<p style='color:#e8590c'>⚠ {tr('ref.same', 'The test image IS the reference: the difference is ~0 '
                                                               'everywhere — choose another test image.')}</p>")
        if labels_withheld(self.test_sample):
            t.append(f"<p style='color:#e8590c'>⚠ {tr('ref.private', 'Labels withheld (test_private): any OK/NOK is a '
                                                                  'prediction, not ground truth; no metrics are computed.')}</p>")
        if self.stale:
            t.append(f"<p style='color:#e8590c'><b>⚠ {tr('ref.stale', 'Reference changed — the previous result was '
                                                                     'discarded. Press Compare.')}</b></p>")
        elif self.result is None:
            t.append(f"<p>{tr('ref.not_run', 'Not compared yet — press Compare.')}</p>")
        else:
            r = self.result
            m = self.shown_mask
            t.append(f"<p><b>{tr('ref.threshold', 'Pixel threshold')}:</b> {self._thr:.4g} ({self._thr_source})</p>")
            t.append(f"<p><b>{tr('ref.global', 'Global (one number per image)')}:</b> "
                     f"{tr('ref.score_is', 'score = 99.9th percentile of the map')} = <b>{r.score:.2f}</b><br>"
                     f"<b>{tr('ref.local', 'Local (where)')}:</b> "
                     f"{m.sum() if m is not None else 0:,} px ({100 * m.mean():.2f} %) {self.shown_op} {self._thr:.4g} · "
                     f"{n_regions(m)} {tr('ref.regions', 'regions')}</p>")
        if self.log:
            hdr = (tr("ref.col_ref", "Reference"), tr("ref.col_score", "Score"), "% px",
                   tr("ref.regions", "regions"), tr("ref.col_dec", "Prediction"))
            rows = "".join(
                f"<tr{' style=\"font-weight:bold\"' if r['ref'] == self.result_ref_id else ''}>"
                f"<td>{r['ref']}</td><td>{r['score']:.2f}</td><td>{r['px']:.2f}</td><td>{r['regions']}</td>"
                f"<td>{r['decision'] or '—'}</td></tr>" for r in self.log)
            t.append(f"<p><b>{tr('ref.log', 'References compared on this test image')}</b> "
                     f"({tr('ref.log_note', 'same test image; mask at the threshold of each run')}):</p>"
                     f"<table cellpadding=2><tr>{''.join(f'<th align=left>{h}</th>' for h in hdr)}</tr>{rows}</table>")
        self.info.setHtml("".join(t))
        self.status.setText(("⚠ " + tr("ref.stale", "Reference changed — the previous result was discarded. Press Compare."))
                            if self.stale else "")
