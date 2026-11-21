"""Educational Experiments — three guided, evidence-based experiments on MVTec AD 2.

* Resolution & geometry : PatchCore 256×256 (squashed) vs. 1024×256 (aspect kept) on sheet_metal.
* Illumination          : the same model on the same scenes under different lighting conditions.
* Understanding Evaluation Metrics : ranking vs. decision, thresholds, the PaDiM fruit_jelly case.

All numbers shown come from ``configs/workshop_verified.json`` (re-evaluated, reproduced campaign
results) or are computed live from the verified checkpoints; nothing is simulated.
"""

from __future__ import annotations

import cv2
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QScrollArea,
    QSlider,
    QSplitter,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from app.i18n import tr
from app.widgets.common import PageHeader, colorize, mask_overlay, muted, overlay
from app.widgets.experiment_kit import (
    CLASS_COLORS,
    OUTCOME_COLORS,
    ConditionGrid,
    ExplanationBox,
    MplCanvas,
    ProvenanceBar,
    ResultTable,
    SceneScoreChart,
    ZoomImage,
    is_nok,
    outcome,
    outcome_name,
    short_condition,
    style_ax,
)
from app.workers.task import start
from core.datasets import registry as dreg
from core.datasets.io import read_mask, read_rgb8
from core.evaluation import metrics as M
from core.inference import model_registry as MR
from core.inference.session import LabSession
from core.research import content

RES_PRESETS = ("d05-sheet_metal-patchcore-256", "d05-sheet_metal-patchcore-aspect")
METRICS_EXAMPLE = "d05-fruit_jelly-padim-256"
F = lambda v, d=3: "—" if v is None else f"{v:.{d}f}"  # noqa: E731


# ------------------------------------------------------------------------- helpers
class Sessions:
    """Verified checkpoints loaded once per page (GPU memory is released when the page closes)."""

    def __init__(self):
        self._s: dict[str, LabSession] = {}

    def get(self, entry: MR.CheckpointEntry, progress=None) -> LabSession:
        s = self._s.get(entry.key)
        if s is None:
            idx = dreg.load_index(entry.get("dataset_id"), entry.get("category"))
            s = LabSession(entry.get("method_id"), seed=int(entry.manifest["seed"]))
            s.bind(idx)
            s.load_checkpoint(entry, progress=progress)
            self._s[entry.key] = s
        return s


def network_input(entry: MR.CheckpointEntry, img: np.ndarray) -> np.ndarray:
    """The image exactly as resized for the network (plain resize, INTER_AREA — see torch_common.to_tensor)."""
    h, w = entry.manifest.get("input_hw") or [256, 256]
    return cv2.resize(img, (int(w), int(h)), interpolation=cv2.INTER_AREA)


def error_overlay(img: np.ndarray, pred: np.ndarray, gt: np.ndarray | None) -> np.ndarray:
    """Green = true-positive pixels, red = false positives, blue = missed defect pixels."""
    out = (img.astype(np.float32) * 0.55).astype(np.uint8)
    gt = np.zeros(pred.shape, bool) if gt is None else gt
    for m, col in ((pred & gt, (60, 220, 60)), (pred & ~gt, (240, 60, 60)), (~pred & gt, (70, 140, 255))):
        out[m] = col
    return out


def zoom_box(gt: np.ndarray | None, shape, min_side: int = 192) -> tuple[int, int, int, int]:
    h, w = shape[:2]
    if gt is None or not gt.any():
        return 0, 0, w, h
    n, lab, stats, _ = cv2.connectedComponentsWithStats(gt.astype(np.uint8))
    i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    x, y, bw, bh = stats[i, :4]
    side = int(max(min_side, 3 * max(bw, bh)))
    cx, cy = x + bw // 2, y + bh // 2
    x0, y0 = max(0, min(w - side, cx - side // 2)), max(0, min(h - side, cy - side // 2))
    return x0, y0, min(w, x0 + side), min(h, y0 + side)


def heat(img, amap, thr):
    """Heat-map overlay on a comparable scale: map / calibrated pixel threshold, clipped to [0, 2]."""
    cm = colorize(amap / thr, 0.0, 2.0) if thr else colorize(amap)
    return overlay(img, cm, 0.55)


def table(rows: list[list[str]], header: list[str]) -> QTableWidget:
    t = QTableWidget(len(rows), len(header))
    t.setHorizontalHeaderLabels(header)
    for i, r in enumerate(rows):
        for j, v in enumerate(r):
            it = QTableWidgetItem(str(v))
            it.setFlags(it.flags() & ~Qt.ItemIsEditable)
            t.setItem(i, j, it)
    h = t.horizontalHeader()
    h.setSectionResizeMode(QHeaderView.ResizeToContents)      # values keep their full width;
    h.setSectionResizeMode(0, QHeaderView.Stretch)            # the label column takes the rest and wraps:
    t.setWordWrap(True)                                       # no horizontal scroll bar in narrow panels
    t.verticalHeader().setVisible(False)
    t.verticalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)   # rows follow the wrapped text
    fit_rows(t)
    return t


def fit_rows(t: QTableWidget):
    """Full label in the tool-tip (row heights follow the wrapped text automatically)."""
    for i in range(t.rowCount()):
        it = t.item(i, 0)
        if it is not None:
            it.setToolTip(it.text())


def tile(title: str, min_w: int = 150, min_h: int = 110) -> ZoomImage:
    """Image tile with a wrapping caption; double-click or ⤢ opens it at full size (zoom & pan)."""
    return ZoomImage(title, min_w, min_h)


def shrinkable(w, min_width: int = 120):
    """Combos and buttons with long texts must not dictate the page's minimum width."""
    from PySide6.QtWidgets import QSizePolicy
    if isinstance(w, QComboBox):
        w.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        w.setMinimumContentsLength(18)
    w.setSizePolicy(QSizePolicy.Ignored, w.sizePolicy().verticalPolicy())
    w.setMinimumWidth(min_width)
    return w


def provenance(stored_text: str, reasons: list[str], live_text: str) -> ProvenanceBar:
    """Make the kind of evidence explicit: STORED verified results vs. LIVE inference on this computer."""
    p = ProvenanceBar()
    p.set_state(stored_text, not reasons,
                live_text if not reasons else
                tr("xp.prov_off", "Live inference not available on this computer — needs: {r}.").format(r="; ".join(reasons)))
    return p


def live_reasons(presets, dataset_id: str, category: str) -> list[str]:
    r = []
    if category not in dreg.installed_categories(dataset_id):
        r.append(tr("xp.need_data", "MVTec AD 2 '{c}' imported from the official source").format(c=category))
    if not all(x.live for x in presets):
        r.append(tr("xp.need_ckpt", "the instructor's fitted checkpoints (not distributed) or your own fitted model"))
    return r


def missing(msg: str) -> QWidget:
    w = QLabel("⚠ " + msg)
    w.setWordWrap(True)
    w.setObjectName("Muted")
    w.setAlignment(Qt.AlignTop | Qt.AlignLeft)
    return w


NAMES = {"patchcore": "PatchCore", "padim": "PaDiM", "efficientad": "EfficientAD", "anomalydino": "AnomalyDINO"}


def plabel(v: dict) -> str:
    """Translatable label of a verified preset (built from its data, not from the stored English label)."""
    hw = v.get("input_hw")
    res = f"{hw[1]}×{hw[0]}" if hw else ""
    asp = " " + tr("xp.aspect_kept", "(aspect kept)") if v["params"].get("keep_aspect") else ""
    return f"{NAMES.get(v['method_id'], v['method_id'])} · {v['category']} · {res}{asp} · {v['n_fit']} " \
           + tr("xp.train_images", "train images")


def suff_text(s: dict) -> str:
    if s["level"] == "insufficient":
        return tr("xp.suff_insufficient", "Only {g} normal and {b} anomalous scenes: confidence intervals are not "
                                          "reported. Treat every number as descriptive.").format(g=s["normal_scenes"], b=s["anomalous_scenes"])
    if s["level"] == "very_limited":
        return tr("xp.suff_limited", "{g} normal and {b} anomalous scenes: intervals are shown but are wide and coarse "
                                     "(few distinct resamples); with ≤ 5 normal scenes each one carries ≥ 20 % of all "
                                     "normal–anomalous comparisons. Exploratory only.").format(g=s["normal_scenes"], b=s["anomalous_scenes"])
    return tr("xp.suff_ok", "{g} normal and {b} anomalous scenes.").format(g=s["normal_scenes"], b=s["anomalous_scenes"])


def roc_points(y: np.ndarray, s: np.ndarray):
    """ROC curve (FPR, TPR) at every distinct score — NumPy only (no scikit-learn inside the GUI)."""
    order = np.argsort(-s, kind="stable")
    ys, ss = y[order], s[order]
    tp, fp = np.cumsum(ys == 1), np.cumsum(ys == 0)
    last = np.r_[ss[1:] != ss[:-1], True]
    tpr = np.r_[0.0, tp[last] / max(1, (y == 1).sum())]
    fpr = np.r_[0.0, fp[last] / max(1, (y == 0).sum())]
    return fpr, tpr


def latency_text(m: dict) -> str:
    b = m.get("latency_benchmark")
    if b:
        return f"{b['median_ms']:.0f} / {b['p95_ms']:.0f}"
    return f"{m['metrics']['latency']['mean_ms']:.0f} " + tr("xp.eval_mean", "(eval. mean)")


def results_markdown(kind: str) -> str:
    """Real, verified results used by the Guided Workshop activities (same source as this page)."""
    v = {m["preset_id"]: m for m in MR.load_verified().values()}
    if not v:
        return "_" + tr("xp.no_presets", "No verified presets found — run `python -m scripts.verify_workshop_models`.") + "_"
    if kind == "resolution":
        rows = ["| " + " | ".join([tr("xp.h_model", "Model"), tr("xp.m_img_auroc", "Image AUROC"), tr("xp.m_pix_auroc", "Pixel AUROC"),
                                   "AU-PRO@0.05", tr("xp.h_pix_f1", "Pixel F1 (cal.)"), tr("xp.h_latency", "Latency median / p95 (ms)") + "*",
                                   tr("xp.h_gpu", "GPU peak")]) + " |", "|---|---|---|---|---|---|---|"]
        for pid in RES_PRESETS:
            m = v.get(pid)
            if m:
                x = m["metrics"]
                rows.append(f"| {plabel(m)} | {F(x['image_auroc'])} | {F(x['pixel_auroc'])} | {F(x['au_pro_05'])} | "
                            f"{F((x.get('pixel_f1') or {}).get('f1'))} | {latency_text(m)} | {x['gpu_peak_mb']:.0f} MB |")
        rows += ["", "\\* " + tr("xp.latency_note", "RTX 4090 Laptop GPU, 30 images after 3 warm-up calls; the laptop power "
                                                         "state is not controlled, so latency is indicative only (GPU memory is stable).")]
        return "\n".join(rows)
    if kind == "illumination":
        from core.evaluation import scenes
        rows = ["| " + " | ".join([tr("xp.h_model", "Model"), tr("xp.h_pooled", "Pooled image AUROC (cluster 95 % CI)"),
                                   tr("xp.h_naive", "Naive CI (ignores scene correlation)"),
                                   tr("xp.h_scene_err", "Scenes with ≥1 error / total")]) + " |", "|---|---|---|---|"]
        for pid, m in sorted(v.items()):
            if m["dataset_id"] != "D05" or m["params"].get("keep_aspect"):
                continue
            a = scenes.analyse(m["samples"], m["calibration"]["image_threshold"], n_boot=1000)
            ci, nv = a.get("pooled_ci_cluster"), a.get("pooled_ci_naive")
            rows.append(f"| {plabel(m)} | {F(a['pooled_auroc'])} ({F(ci[0]) if ci else '—'}–{F(ci[1]) if ci else '—'}) | "
                        f"{F(nv[0]) if nv else '—'}–{F(nv[1]) if nv else '—'} | {a.get('scenes_with_errors')}/{a['n_scenes']} |")
        return "\n".join(rows)
    if kind == "metrics":
        m = v.get(METRICS_EXAMPLE)
        if not m:
            return "_" + tr("xp.no_padim", "PaDiM fruit_jelly preset not verified.") + "_"
        x, c = m["metrics"], m["calibration"]
        d = x.get("decision") or {}
        sc = [s["score"] for s in m["samples"]]
        return tr("xp.padim_case", "PaDiM · fruit_jelly (verified {date}): image AUROC **{auroc}**, AP {ap}; validation-calibrated "
                                   "image threshold {thr:.1f} (p99 of {n} validation normals) while the highest test score is "
                                   "{mx:.1f} → TP {tp}, FP {fp}, TN {tn}, FN {fn}, image F1 **{f1}**.").format(
            date=m["verified_date"], auroc=F(x["image_auroc"]), ap=F(x["image_ap"]), thr=c["image_threshold"], n=c["n_images"],
            mx=max(sc), tp=d.get("tp"), fp=d.get("fp"), tn=d.get("tn"), fn=d.get("fn"), f1=F(d.get("f1")))
    return ""


# ------------------------------------------------------------------------- page
class ExperimentsPage(QWidget):
    def __init__(self, main):
        super().__init__()
        self.main = main
        self.sessions = Sessions()
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(PageHeader(tr("nav.experiments", "Educational Experiments"),
                                 tr("xp.subtitle", "Reproducible experiments on real MVTec AD 2 data with Workshop "
                                                   "Verified models. Local educational results — not official benchmark numbers.")))
        self.tabs = QTabWidget()
        self.res = ResolutionTab(self)
        self.ill = IlluminationTab(self)
        self.met = MetricsTab(self)
        self.tabs.addTab(self.res, tr("xp.tab_res", "A · Why does image resolution matter?"))
        self.tabs.addTab(self.ill, tr("xp.tab_ill", "B · Can a model handle changing illumination?"))
        self.tabs.addTab(self.met, tr("xp.tab_met", "C · Understanding Evaluation Metrics"))
        lay.addWidget(self.tabs, 1)

    def apply_preset(self, tab: str | None = None, preset_id: str | None = None, **_):
        i = {"resolution": 0, "illumination": 1, "metrics": 2}.get(tab or "")
        if i is not None:
            self.tabs.setCurrentIndex(i)
        if tab == "metrics" and preset_id:
            self.met.select(preset_id)

    def shutdown(self):
        self.sessions = Sessions()


def _scroll(inner: QWidget) -> QScrollArea:
    sa = QScrollArea()
    sa.setWidgetResizable(True)
    sa.setFrameShape(QFrame.NoFrame)
    sa.setWidget(inner)
    return sa


# ------------------------------------------------------------------------- A. resolution
class ResolutionTab(QWidget):
    def __init__(self, page: ExperimentsPage):
        super().__init__()
        self.page = page
        self.task = None
        lay = QVBoxLayout(self)
        self.entries = [MR.find_stored(p) for p in RES_PRESETS]
        if not all(self.entries):
            lay.addWidget(missing(tr("xp.res_missing", "No stored results for d05-sheet_metal-patchcore-256 / -aspect "
                                                       "(configs/workshop_verified.json).")))
            lay.addStretch(1)
            self.ready = False
            return
        self.ready = True
        self.reasons = live_reasons(self.entries, "D05", "sheet_metal")
        self.live_ok = not self.reasons
        self.prov = provenance(
            tr("xp.prov_stored_a", "STORED RESULTS — the table of verified results was computed and verified on the "
                                   "instructor's computer ({date}) with real MVTec AD 2 data.").format(
                date=self.entries[0].verified.get("verified_date", "")),
            self.reasons,
            tr("xp.prov_live_a", "LIVE — 'Run both models' computes this image's scores and heat-maps on this computer "
                                 "with the same Workshop Verified checkpoints."))
        lay.addWidget(self.prov)
        sp = QSplitter(Qt.Horizontal)
        left_tabs = QTabWidget()
        left = QWidget()
        ll = QVBoxLayout(left)
        ll.setContentsMargins(4, 4, 4, 4)
        left_tabs.addTab(left, tr("xp.results_tab", "Run && results"))
        left_tabs.addTab(ExplanationBox(content.markdown("experiments", "resolution") or ""),
                         tr("xp.explain_tab", "Explanation"))
        self.idx = dreg.load_index("D05", "sheet_metal") if "sheet_metal" in dreg.installed_categories("D05") else None
        self.sample = QComboBox()
        tests = [s for s in self.idx.samples if s.split == "test_public"] if self.idx else []
        tests.sort(key=lambda s: (s.label != "bad", s.meta.get("lighting_condition") != "regular", s.sample_id))
        for s in tests:
            lab = tr("xp.anomalous", "anomalous") if s.label == "bad" else tr("xp.normal", "normal")
            self.sample.addItem(f"{lab} · {tr('xp.scene', 'scene')} {s.meta.get('scene_id')} · "
                                f"{s.meta.get('lighting_condition')}", s.sample_id)
        ll.addWidget(QLabel(tr("xp.sample", "Test image (test_public)")))
        ll.addWidget(shrinkable(self.sample))
        self.run = QPushButton("▶  " + tr("xp.run_both", "Run both PatchCore models on this image"))
        self.run.setObjectName("Primary")
        self.run.clicked.connect(self._run)
        self.run.setEnabled(self.live_ok)
        self.sample.setEnabled(self.live_ok)
        if not self.live_ok:
            self.run.setToolTip("; ".join(self.reasons))
        ll.addWidget(shrinkable(self.run))
        self.msg = muted("")
        ll.addWidget(self.msg)
        rows = self._stored_rows()
        hdr = QLabel("<b>" + tr("xp.stored", "Verified results on the whole test_public split (stored)") + "</b>"
                     + f" — {self.entries[0].verified.get('verified_date', '')}")
        hdr.setWordWrap(True)
        ll.addWidget(hdr)
        self.stored = table(rows, [tr("xp.metric", "Metric"), "256×256", "1024×256"])
        self.stored.setMinimumHeight(140)
        ll.addWidget(self.stored, 2)
        self.live = table([], [tr("xp.this_image", "This image"), "256×256", "1024×256"])
        self.live.setMinimumHeight(100)
        ll.addWidget(self.live, 1)
        sp.addWidget(left_tabs)

        right = QWidget()
        self.grid = QGridLayout(right)
        self.views = {}
        names = [("orig", "Original + ground truth (full resolution)"), ("zoom", "Zoom on the largest defect")]
        for c, (k, t) in enumerate(names):
            self.views[k] = tile(tr(f"xp.v_{k}", t), 240, 110)
            self.grid.addWidget(self.views[k], 0, c * 2, 1, 2)
        for r, e in enumerate(self.entries, start=1):
            hw = e.manifest["input_hw"]
            tag = f"{hw[1]}×{hw[0]}"
            for c, (k, t) in enumerate([("pre", f"{tag}: " + tr("xp.v_pre", "what the network sees")),
                                        ("heat", f"{tag}: " + tr("xp.v_heat", "anomaly heat-map")),
                                        ("err", f"{tag}: " + tr("xp.v_err", "predicted mask vs GT")),
                                        ("zheat", f"{tag}: " + tr("xp.v_zheat", "zoom"))]):
                v = tile(t, 130, 100)
                self.views[f"{k}{r}"] = v
                self.grid.addWidget(v, r, c)
        for r in range(3):
            self.grid.setRowStretch(r, 1)
        for c in range(4):
            self.grid.setColumnStretch(c, 1)
        self.grid.addWidget(muted(tr("xp.legend", "Heat-map scale: map ÷ calibrated pixel threshold, 0–2 (1 = threshold). "
                                                 "Mask colours: green = correct defect pixels, red = false alarms, blue = missed defect pixels.")), 3, 0, 1, 4)
        sp.addWidget(_scroll(right))
        sp.setSizes([460, 1060])
        lay.addWidget(sp, 1)

    def _stored_rows(self):
        a, b = (e.verified["metrics"] for e in self.entries)
        def r(name, f):
            return [name, f(a), f(b)]
        return [r(tr("xp.m_img_auroc", "Image AUROC"), lambda m: F(m["image_auroc"])),
                r(tr("xp.m_img_ap", "Image AP"), lambda m: F(m["image_ap"])),
                r(tr("xp.m_pix_auroc", "Pixel AUROC"), lambda m: F(m["pixel_auroc"])), r("AU-PRO@0.30", lambda m: F(m["au_pro_30"])),
                r("AU-PRO@0.05", lambda m: F(m["au_pro_05"])),
                r(tr("xp.m_pix_f1", "Pixel F1 (validation threshold)"), lambda m: F((m.get("pixel_f1") or {}).get("f1"))),
                r(tr("xp.m_img_f1", "Image F1 (validation threshold)"), lambda m: F((m.get("decision") or {}).get("f1"))),
                [tr("xp.m_latency", "Latency median / p95 (ms, indicative)")] + [latency_text(e.verified) for e in self.entries],
                r(tr("xp.m_gpu", "GPU peak (MB)"), lambda m: f"{m['gpu_peak_mb']:.0f}"),
                [tr("xp.m_input", "Network input (w×h)")] + [f"{e.manifest['input_hw'][1]}×{e.manifest['input_hw'][0]}" for e in self.entries],
                [tr("xp.m_coreset", "Coreset / all patch features")] + [
                    f"{e.manifest['fit_info'].get('memory_bank')} / {e.manifest['fit_info'].get('patch_features')}"
                    if e.manifest.get("fit_info") else "—" for e in self.entries]]

    def _run(self):
        sid = self.sample.currentData()
        s = next(x for x in self.idx.samples if x.sample_id == sid)
        entries, sessions = self.entries, self.page.sessions
        self.run.setEnabled(False)

        def job(progress, cancel):
            img = read_rgb8(s.primary_path)
            gt = read_mask(s.masks["rgb"], img.shape[:2]) if s.masks.get("rgb") else None
            out = []
            for k, e in enumerate(entries):
                progress(0.1 + 0.4 * k, tr("xp.loading", "loading") + f" {plabel(e.verified)}…")
                sess = sessions.get(e.entry)
                progress(0.3 + 0.4 * k, tr("xp.inference", "inference…"))
                out.append((e, sess.calibration, sess.predict_image(img)))
            return s, img, gt, out
        self.task = start(job, on_done=self._show, on_error=self._err,
                          on_progress=lambda f, m: self.msg.setText(m))

    def _err(self, m, tb):
        self.run.setEnabled(True)
        self.msg.setText("✗ " + m)

    def _show(self, payload):
        self.run.setEnabled(True)
        s, img, gt, out = payload
        self.msg.setText("✓ " + tr("xp.live_note", "Computed now with the verified checkpoints."))
        self.views["orig"].set(mask_overlay(img, gt) if gt is not None else img)
        x0, y0, x1, y1 = zoom_box(gt, img.shape)
        crop = img[y0:y1, x0:x1]
        self.views["zoom"].set(mask_overlay(crop, gt[y0:y1, x0:x1]) if gt is not None else crop,
                               tr("xp.zoom_caption", "Zoom ({w}×{h} px of {W}×{H})").format(
                                   w=x1 - x0, h=y1 - y0, W=img.shape[1], H=img.shape[0]))
        live = []
        for r, (e, cal, res) in enumerate(out, start=1):
            pred = res.anomaly_map >= cal.pixel_threshold
            self.views[f"pre{r}"].set(network_input(e, img))
            self.views[f"heat{r}"].set(heat(img, res.anomaly_map, cal.pixel_threshold))
            self.views[f"err{r}"].set(error_overlay(img, pred, gt))
            zh = heat(crop, res.anomaly_map[y0:y1, x0:x1], cal.pixel_threshold)
            self.views[f"zheat{r}"].set(mask_overlay(zh, gt[y0:y1, x0:x1], color=(80, 255, 80), alpha=0.0)
                                        if gt is not None else zh)
            pa = M.pixel_auroc([gt], [res.anomaly_map]) if gt is not None and gt.any() else None
            d, _ = M.dice_iou(pred, gt) if gt is not None else (None, None)
            nok = is_nok(res.score, cal.image_threshold)
            live.append({"score": res.score, "dec": "NOK" if nok else "OK", "code": outcome(s.label == "bad", nok),
                         "pa": pa, "dice": d, "lat": res.latency_ms, "gpu": res.gpu_memory_mb})
        # enlarge: compare the same view of both models next to the original (same image size → synced zoom)
        orig = (tr("xp.v_orig", "Original + ground truth (full resolution)"), mask_overlay(img, gt) if gt is not None else img)
        for k in ("heat", "err"):
            for r in (1, 2):
                self.views[f"{k}{r}"].provider = lambda k=k: [orig] + [(self.views[f"{k}{q}"].caption.text(),
                                                                        self.views[f"{k}{q}"]._img) for q in (1, 2)]
        for r in (1, 2):
            self.views[f"zheat{r}"].provider = lambda: [(self.views["zoom"].caption.text(), self.views["zoom"]._img)] + [
                (self.views[f"zheat{q}"].caption.text(), self.views[f"zheat{q}"]._img) for q in (1, 2)]
        gt_lab = tr("xp.anomalous", "anomalous").upper() if s.label == "bad" else tr("xp.normal", "normal").upper()
        rows = [[tr("xp.r_gt", "Ground truth"), gt_lab, ""],
                [tr("xp.r_score", "Image score / threshold")] + [f"{x['score']:.2f} / {c.image_threshold:.2f}" for x, (_, c, _r) in zip(live, out)],
                [tr("xp.r_decision", "Decision")] + [x["dec"] + (" ✓ " if x["code"] in ("TP", "TN") else " ✗ ") + x["code"] for x in live],
                [tr("xp.r_pix_auroc", "Pixel AUROC (this image)")] + [F(x["pa"]) for x in live],
                [tr("xp.r_dice", "Dice at pixel threshold")] + [F(x["dice"]) for x in live],
                [tr("xp.r_latency", "Latency now (ms)")] + [f"{x['lat']:.1f}" for x in live],
                [tr("xp.r_gpu", "GPU peak now (MB)")] + ["—" if x["gpu"] is None else f"{x['gpu']:.0f}" for x in live]]
        rows[0][2] = rows[0][1]
        self.live.setRowCount(len(rows))
        for i, r in enumerate(rows):
            for j, v in enumerate(r):
                self.live.setItem(i, j, QTableWidgetItem(str(v)))
        fit_rows(self.live)
        for j, x in enumerate(live, start=1):                      # same outcome colours as experiments B and C
            c = QColor(OUTCOME_COLORS[x["code"]])
            c.setAlphaF(0.35)
            self.live.item(2, j).setBackground(c)
            self.live.item(2, j).setToolTip(outcome_name(x["code"]))


# ------------------------------------------------------------------------- B. illumination
def scene_kind(lab: str) -> str:
    return tr("xp.anomalous", "anomalous") if lab == "bad" else tr("xp.normal", "normal")


def scene_item_text(key, st: dict) -> str:
    """Scene selector text: class (upper case), scene id and its stability under lighting."""
    flag = ("✓ " + tr("xp.stable", "stable")) if st["consistent"] and not st["errors"] else (
        "⚠ " + tr("xp.n_errors", "{n} error(s)").format(n=st["errors"])
        + ("" if st["consistent"] else ", " + tr("xp.flips", "decision flips")))
    return f"{scene_kind(key[0]).upper()} · {tr('xp.scene', 'scene')} {key[1]} · {flag}"


def scene_story(items: list[dict], thr: float, st: dict, score_span: float) -> str:
    """Basic mode: plain-language explanation of what the model decided for one scene (markdown)."""
    anomalous = items[0]["label"] == "bad"
    conds = [s["lighting_condition"] for s in items]
    codes = [outcome(anomalous, is_nok(s["score"], thr)) for s in items]
    n = len(items)
    wrong = [c for c, k in zip(conds, codes) if k in ("FP", "FN")]
    right = [c for c, k in zip(conds, codes) if k in ("TP", "TN")]
    cnt = {k: codes.count(k) for k in ("TP", "TN", "FP", "FN")}
    L = ["#### " + tr("xp.story_title", "Scene {sid} — {kind} part").format(sid=items[0]["scene_id"],
                                                                          kind=scene_kind(items[0]["label"])),
         tr("xp.story_truth_bad", "This part **has a real defect** (ground truth), so the correct decision is **NOK** "
                                  "under every lighting condition.") if anomalous else
         tr("xp.story_truth_good", "This part is **defect-free** (ground truth), so the correct decision is **OK** under "
                                   "every lighting condition."),
         "",
         tr("xp.story_rule", "The model gives each image one anomaly score. A score **≥ {t:.2f}** — the threshold fixed "
                             "beforehand on normal validation images, without test labels — rejects the part (NOK); a "
                             "lower score accepts it (OK).").format(t=thr),
         "",
         "**" + tr("xp.story_count", "{c} of {n} conditions decided correctly").format(c=len(right), n=n) + "** — "
         + ", ".join(f"{k} {cnt[k]}" for k in ("TP", "TN", "FP", "FN") if cnt[k]) + ".", ""]
    lo_c = conds[int(np.argmin([s["score"] for s in items]))]
    hi_c = conds[int(np.argmax([s["score"] for s in items]))]
    if st["consistent"] and not wrong:
        L.append(tr("xp.story_robust", "✓ **Robust for this part:** the decision is the same, and correct, under all "
                                       "lighting conditions."))
    elif st["consistent"]:
        L.append((tr("xp.story_all_missed", "✗ **The defect is missed under every lighting condition:** all scores stay "
                                            "below the threshold.") if anomalous else
                  tr("xp.story_all_alarm", "✗ **False alarm under every lighting condition:** all scores are above the "
                                           "threshold.")) + " " +
                 tr("xp.story_not_light", "Lighting is not what causes this error — the model does not separate this "
                                          "part from the normal appearance under any condition (experiment A explores "
                                          "one reason: image resolution)."))
    else:
        L.append(tr("xp.story_flip", "⚠ **The decision changes with the lighting.** Correct under: {r}. Wrong under: {w}. "
                                     "The part is the same in every image — only the light changed — so these "
                                     "changes are caused by illumination.").format(r=", ".join(right) or "—",
                                                                                     w=", ".join(wrong) or "—"))
    L += ["", tr("xp.story_range", "Highest score under **{hi}**, lowest under **{lo}**; the lighting moves the score by "
                                   "{rng:.2f}. The closest score is {mg:.2f} away from the threshold.").format(
        hi=hi_c, lo=lo_c, rng=st["score_range"], mg=st["margin_min"])]
    if score_span and st["margin_min"] < 0.05 * score_span:
        L.append(tr("xp.story_close", "That is less than 5 % of the model's whole score range on this test set: a "
                                      "slightly different light could flip the decision."))
    L += ["", "_" + tr("xp.story_train", "The model was fitted only on `regular` images; every other condition is new "
                                         "to it.") + "_",
          "", "_" + tr("xp.story_tip", "Click a condition card or a point of the chart to select it; double-click an "
                                       "image or press ⤢ to enlarge it.") + "_"]
    return "\n".join(L)


class IlluminationTab(QWidget):
    """Experiment B. Basic: one scene under every lighting condition (images, maps, scores, threshold,
    OK/NOK, TP/TN/FP/FN, a plain explanation). Advanced: all scenes, decisions that change,
    AUROC per illumination with cluster-bootstrap CIs, statistical limitations and references.

    Scores, decisions and statistics always come from the STORED verified per-image scores; LIVE
    inference (when available) only adds the anomaly maps and re-computes the scores for comparison."""

    THUMB_CACHE = 48

    def __init__(self, page: ExperimentsPage):
        super().__init__()
        self.page = page
        self.task = None
        self.entry: MR.StoredPreset | None = None
        self.analysis: dict | None = None
        self.level = "basic"
        self.cond_idx = 0
        self.cur, self.cur_key = [], None
        self.live_delta = None
        self._pix_thr = None
        self._thumbs: dict = {}
        self._live_maps: dict = {}                         # (preset key, scene key) → {sample_id: (map, live score)}
        lay = QVBoxLayout(self)
        presets = [e for e in MR.stored_presets("D05") if not e.verified["params"].get("keep_aspect")]
        if not presets:
            lay.addWidget(missing(tr("xp.ill_missing", "No stored MVTec AD 2 results (configs/workshop_verified.json).")))
            lay.addStretch(1)
            self.ready = False
            return
        self.ready = True
        self.presets = presets
        lay.setContentsMargins(4, 4, 4, 4)
        lay.setSpacing(4)
        self.prov = ProvenanceBar()
        lay.addWidget(self.prov)

        bar = QHBoxLayout()
        bar.addWidget(QLabel(tr("xp.model_short", "Model")))
        self.model = shrinkable(QComboBox(), 150)
        for e in presets:
            self.model.addItem(plabel(e.verified), e.key)
        bar.addWidget(self.model, 3)
        bar.addWidget(QLabel(tr("xp.scene_short", "Scene")))
        self.prev_btn, self.next_btn = QToolButton(), QToolButton()
        self.prev_btn.setText("◀")
        self.next_btn.setText("▶")
        self.prev_btn.setToolTip(tr("xp.prev_scene", "Previous scene"))
        self.next_btn.setToolTip(tr("xp.next_scene", "Next scene"))
        self.scenes = shrinkable(QComboBox(), 150)
        bar.addWidget(self.prev_btn)
        bar.addWidget(self.scenes, 3)
        bar.addWidget(self.next_btn)
        self.basic_btn = QPushButton(tr("xp.basic", "Basic"))
        self.adv_btn = QPushButton(tr("xp.advanced", "Advanced"))
        self.levels = QButtonGroup(self)
        for b in (self.basic_btn, self.adv_btn):
            b.setCheckable(True)
            self.levels.addButton(b)
            bar.addWidget(b)
        self.basic_btn.setToolTip(tr("xp.basic_tip", "One scene: images, maps, scores, threshold and decisions"))
        self.adv_btn.setToolTip(tr("xp.adv_tip", "All scenes: interactive chart, decisions that change, AUROC per "
                                                 "illumination, confidence intervals, limitations, references"))
        self.maps_btn = QPushButton("▶  " + tr("xp.compute_maps_short", "Compute anomaly maps"))
        self.maps_btn.setObjectName("Primary")
        bar.addWidget(shrinkable(self.maps_btn, 150), 2)
        lay.addLayout(bar)
        self.msg = muted("")
        lay.addWidget(self.msg)

        self.vsplit = QSplitter(Qt.Vertical)
        top = QWidget()
        tl = QVBoxLayout(top)
        tl.setContentsMargins(0, 0, 0, 0)
        tl.setSpacing(3)
        self.scene_title = QLabel("")
        self.scene_title.setWordWrap(True)
        self.scene_title.setObjectName("CardTitle")
        tl.addWidget(self.scene_title)
        self.grid = ConditionGrid()
        tl.addWidget(self.grid, 1)
        self.vsplit.addWidget(top)

        self.stack = QStackedWidget()
        basic = QSplitter(Qt.Horizontal)
        self.scene_chart = SceneScoreChart(min_height=120)
        basic.addWidget(self.scene_chart)
        self.basic_tabs = QTabWidget()
        self.story = ExplanationBox()
        self.basic_tabs.addTab(self.story, tr("xp.what_happened", "What happened?"))
        self.basic_tabs.addTab(ExplanationBox(content.markdown("experiments", "illumination") or ""),
                               tr("xp.about_exp", "About this experiment"))
        basic.addWidget(self.basic_tabs)
        basic.setSizes([560, 440])
        self.stack.addWidget(basic)

        adv = QSplitter(Qt.Horizontal)
        left = QWidget()
        al = QVBoxLayout(left)
        al.setContentsMargins(0, 0, 0, 0)
        self.overview = SceneScoreChart(min_height=110)
        al.addWidget(self.overview, 1)
        fl = QHBoxLayout()
        self.only_flips = QCheckBox(tr("xp.only_flips", "Only scenes whose decision changes"))
        self.dim = QCheckBox(tr("xp.dim_others", "Dim the other scenes"))
        self.dim.setChecked(True)
        fl.addWidget(self.only_flips)
        fl.addWidget(self.dim)
        fl.addStretch(1)
        al.addLayout(fl)
        adv.addWidget(left)
        self.adv_tabs = QTabWidget()
        mw = QWidget()
        ml = QVBoxLayout(mw)
        ml.setContentsMargins(0, 0, 0, 0)
        ml.setSpacing(2)
        self.matrix = ResultTable()
        self.matrix.cellClicked.connect(self._matrix_cell)
        ml.addWidget(self.matrix, 1)
        self.matrix_detail = muted("")                 # range, margin and the clicked cell of the selected scene
        self.matrix_detail.setTextFormat(Qt.RichText)
        ml.addWidget(self.matrix_detail)
        self.adv_tabs.addTab(mw, tr("xp.tab_matrix", "Scenes × lighting"))
        au = QWidget()
        aul = QVBoxLayout(au)
        aul.setContentsMargins(0, 0, 0, 0)
        self.auroc_chart = MplCanvas(5.0, 2.2, 70)
        aul.addWidget(self.auroc_chart, 1)
        self.auroc_tbl = ResultTable()
        self.auroc_tbl.setMinimumHeight(50)
        aul.addWidget(self.auroc_tbl, 1)
        self.adv_tabs.addTab(au, tr("xp.tab_auroc", "AUROC per lighting"))
        self.report = ExplanationBox()
        self.adv_tabs.addTab(self.report, tr("xp.tab_stats", "Statistics"))
        self.refs = ExplanationBox()
        self.adv_tabs.addTab(self.refs, tr("xp.tab_refs", "References"))
        for w in (self.matrix, self.report, self.refs):      # contents scroll inside; never force the window taller
            w.setMinimumHeight(70)
        adv.addWidget(self.adv_tabs)
        adv.setSizes([450, 610])
        self.stack.addWidget(adv)
        self.vsplit.addWidget(self.stack)
        self.vsplit.setChildrenCollapsible(False)
        lay.addWidget(self.vsplit, 1)

        self.model.currentIndexChanged.connect(self._model)
        self.scenes.currentIndexChanged.connect(self._scene)
        self.prev_btn.clicked.connect(lambda: self._step(-1))
        self.next_btn.clicked.connect(lambda: self._step(1))
        self.basic_btn.clicked.connect(lambda: self.set_level("basic"))
        self.adv_btn.clicked.connect(lambda: self.set_level("advanced"))
        self.maps_btn.clicked.connect(self._maps)
        self.grid.conditionClicked.connect(self.select_condition)
        self.scene_chart.conditionClicked.connect(self.select_condition)
        self.overview.conditionClicked.connect(lambda x: self.select_condition(self._cond_to_item(x)))
        self.overview.sceneClicked.connect(self.select_scene)
        self.matrix.rowSelected.connect(self.select_scene)
        self.only_flips.toggled.connect(self._filter_changed)
        self.dim.toggled.connect(lambda _: self._draw_charts())
        self._model()
        self.set_level("basic")

    # ------------------------------------------------------------------ public API (also used by tests)
    def set_level(self, level: str):
        self.level = "advanced" if level == "advanced" else "basic"
        adv = self.level == "advanced"
        self.stack.setCurrentIndex(1 if adv else 0)
        (self.adv_btn if adv else self.basic_btn).setChecked(True)
        h = max(400, self.vsplit.height())
        self.vsplit.setSizes([int(h * (0.40 if adv else 0.56)), int(h * (0.60 if adv else 0.44))])
        self._draw_charts()

    def select_scene(self, key):
        i = self.scenes.findData(key)
        if i < 0 and self.only_flips.isChecked():          # selecting a hidden scene clears the filter
            self.only_flips.setChecked(False)
            i = self.scenes.findData(key)
        if i >= 0 and i != self.scenes.currentIndex():
            self.scenes.setCurrentIndex(i)

    def select_condition(self, i: int):
        if not self.cur or not 0 <= i < len(self.cur):
            return
        self.cond_idx = i
        self.grid.select(i)
        self._draw_charts()
        self._matrix_info()

    # ------------------------------------------------------------------ model / scene
    def _model(self):
        from core.evaluation import scenes
        self.entry = next(e for e in self.presets if e.key == self.model.currentData())
        v = self.entry.verified
        self.thr = v["calibration"]["image_threshold"]
        self.analysis = scenes.analyse(v["samples"], self.thr, n_boot=1000)
        self.groups = scenes.group(v["samples"])
        self.conds = []
        for items in self.groups.values():
            for s in items:
                if s["lighting_condition"] not in self.conds:
                    self.conds.append(s["lighting_condition"])
        sc = [s["score"] for s in v["samples"]]
        self.span = float(max(sc) - min(sc)) if sc else 0.0
        installed = v["category"] in dreg.installed_categories("D05")
        self.idx = dreg.load_index("D05", v["category"]) if installed else None
        self.by_id = {s.sample_id: s for s in self.idx.samples} if self.idx else {}
        self._thumbs = {}
        self.reasons = live_reasons([self.entry], "D05", v["category"])
        date = v.get("verified_date", "")
        self.prov.set_state(
            tr("xp.prov_stored_b", "STORED RESULTS — scores, decisions, AUROC and intervals: verified run on real "
                                   "MVTec AD 2 data ({date}).").format(date=date),
            not self.reasons,
            tr("xp.prov_live_b", "LIVE — anomaly maps computed on this computer with the same checkpoint; live "
                                 "scores are checked against the stored ones.") if not self.reasons else
            tr("xp.prov_off", "Live inference not available on this computer — needs: {r}.").format(r="; ".join(self.reasons)))
        self.maps_btn.setEnabled(not self.reasons)
        self.maps_btn.setToolTip("; ".join(self.reasons) or tr("xp.compute_maps", "Compute anomaly maps for this scene"))
        self._report()
        self._auroc()
        self._references()
        self._matrix()
        self._fill_scenes()

    def _fill_scenes(self):
        flips = self.only_flips.isChecked()
        cur = self.scenes.currentData()
        self.scenes.blockSignals(True)
        self.scenes.clear()
        for key in self.groups:
            st = self.analysis["stability"][f"{key[0]}:{key[1]}"]
            if flips and st["consistent"]:
                continue
            self.scenes.addItem(scene_item_text(key, st), key)
        self.scenes.blockSignals(False)
        i = max(0, self.scenes.findData(cur))
        if self.scenes.count():
            self.scenes.setCurrentIndex(i)
            self._scene(i)

    def _filter_changed(self, _=None):
        self._fill_scenes()
        self._matrix()
        self._draw_charts()

    def _step(self, d: int):
        n = self.scenes.count()
        if n:
            self.scenes.setCurrentIndex((self.scenes.currentIndex() + d) % n)

    def _cond_to_item(self, x: int) -> int:
        c = self.conds[x] if 0 <= x < len(self.conds) else None
        return next((i for i, s in enumerate(self.cur) if s["lighting_condition"] == c), -1)

    def _thumb(self, sid: str):
        """(¼-resolution image with the GT contour, plain ¼-resolution image, aspect w/h) — cached;
        None when the dataset is not installed on this computer."""
        if sid in self._thumbs:
            return self._thumbs[sid]
        sm = self.by_id.get(sid)
        if sm is None:
            return None
        img = read_rgb8(sm.primary_path)
        gt = read_mask(sm.masks["rgb"], img.shape[:2]) if sm.masks.get("rgb") else None
        small = cv2.resize(img, (img.shape[1] // 4, img.shape[0] // 4), interpolation=cv2.INTER_AREA)
        shown = mask_overlay(small, gt[::4, ::4][:small.shape[0], :small.shape[1]]) if gt is not None else small
        if len(self._thumbs) >= self.THUMB_CACHE:
            self._thumbs.pop(next(iter(self._thumbs)))
        self._thumbs[sid] = (shown, small, img.shape[1] / img.shape[0])
        return self._thumbs[sid]

    def _full(self, s: dict):
        """Full-resolution views for the enlarge dialog: original + GT and, if computed, the live map."""
        sm = self.by_id.get(s["sample_id"])
        if sm is None:
            return []
        img = read_rgb8(sm.primary_path)
        gt = read_mask(sm.masks["rgb"], img.shape[:2]) if sm.masks.get("rgb") else None
        out = [(f"{s['lighting_condition']} · " + tr("xp.v_orig", "Original + ground truth (full resolution)"),
                mask_overlay(img, gt) if gt is not None else img)]
        m = self._live_maps.get((self.entry.key, self.cur_key), {}).get(s["sample_id"])
        if m is not None:
            out.append((f"{s['lighting_condition']} · " + tr("xp.map_live", "anomaly map (LIVE)"),
                        heat(img, m[0], self._pix_thr)))
        return out

    def _scene(self, row):
        if row < 0 or row >= self.scenes.count():
            return
        key = self.scenes.itemData(row)
        items = self.groups[key]
        self.cur, self.cur_key = items, key
        self.cond_idx = min(self.cond_idx, len(items) - 1)
        st = self.analysis["stability"][f"{key[0]}:{key[1]}"]
        cards = self.grid.set_count(len(items))
        sc = [s["score"] for s in items] + [self.thr]
        pad = 0.15 * (max(sc) - min(sc) or 1)
        lo, hi = min(sc) - pad, max(sc) + pad
        self.tiles, self.maptiles = [], []
        for card, s in zip(cards, items):
            card.set_result(s["lighting_condition"], s["score"], self.thr, s["label"] == "bad", lo, hi)
            th = self._thumb(s["sample_id"])
            card.set_aspect(th[2] if th is not None else 2.0)
            if th is None:
                card.image.set(None, tr("xp.no_image", "(image not installed on this computer)"))
            else:
                card.image.set(th[0], tr("xp.image_gt", "image · ground truth outlined"))
            card.image.provider = card.map.provider = (lambda s=s: self._full(s))
            card.zoom_btn.setEnabled(th is not None)
            card.map.set(None, tr("xp.map_pending", "anomaly map: press 'Compute'") if not self.reasons
                         else tr("xp.map_unavailable", "anomaly map: live inference not available here"))
            self.tiles.append(card.image)
            self.maptiles.append(card.map)
        self._apply_maps()
        self.grid.select(self.cond_idx)
        self.scene_title.setText(tr("xp.scene_title", "Scene {sid} ({kind}) — score range {rng:.2f}, closest distance to "
                                                      "threshold {mg:.2f}, {cons}, {n} error(s)").format(
            kind=scene_kind(key[0]), sid=key[1], rng=st["score_range"], mg=st["margin_min"], n=st["errors"],
            cons=tr("xp.same_decision", "same decision under all conditions") if st["consistent"]
            else tr("xp.decision_changes", "decision changes with lighting")))
        self.story.set_markdown(scene_story(items, self.thr, st, self.span))
        self.matrix.select_payload(key)
        self._matrix_info()
        self._draw_charts()

    def _draw_charts(self):
        if not self.cur:
            return
        if self.level == "basic":
            self.scene_chart.plot_scene(self.cur, self.thr, self.cond_idx)
        else:
            only = None
            if self.only_flips.isChecked():
                only = {k for k in self.groups if not self.analysis["stability"][f"{k[0]}:{k[1]}"]["consistent"]}
            cond = self.cur[self.cond_idx]["lighting_condition"]
            self.overview.plot_overview(self.groups, self.conds, self.thr, selected=self.cur_key,
                                        sel_cond=self.conds.index(cond) if cond in self.conds else None,
                                        dim=self.dim.isChecked(), only=only)
            self._auroc_plot()

    # ------------------------------------------------------------------ live maps
    def _maps(self):
        if self.reasons or not self.cur:
            return
        items, e, sessions, by_id = self.cur, self.entry, self.page.sessions, self.by_id
        key = (e.key, self.cur_key)
        self.maps_btn.setEnabled(False)

        def job(progress, cancel):
            sess = sessions.get(e.entry)
            out = {}
            for i, s in enumerate(items):
                progress(i / len(items), f"{s['lighting_condition']}…")
                r = sess.predict_image(read_rgb8(by_id[s["sample_id"]].primary_path))
                out[s["sample_id"]] = (r.anomaly_map, float(r.score))
            return key, sess.calibration.pixel_threshold, out
        self.task = start(job, on_done=self._show_maps,
                          on_error=lambda m, tb: (self.maps_btn.setEnabled(True), self.msg.setText("✗ " + m)),
                          on_progress=lambda f, m: self.msg.setText(m))

    def _show_maps(self, payload):
        self.maps_btn.setEnabled(True)
        key, thr, out = payload
        self._pix_thr = thr
        while len(self._live_maps) >= 3:                      # full-resolution maps of the last 3 scenes only
            self._live_maps.pop(next(iter(self._live_maps)))
        self._live_maps[key] = out
        self._apply_maps()
        self.msg.setText("✓ " + tr("xp.same_scale", "Same colour scale for all conditions (map ÷ calibrated pixel threshold).")
                         + " " + tr("xp.live_match", "Live scores vs stored: max |Δ| = {d:.2g}.").format(d=self.live_delta or 0.0))

    def _apply_maps(self):
        maps = self._live_maps.get((self.entry.key, self.cur_key)) if self.cur_key else None
        self.live_delta = None
        if not maps:
            return
        deltas = []
        for card, t, s in zip(self.grid.cards, self.maptiles, self.cur):
            m = maps.get(s["sample_id"])
            th = self._thumb(s["sample_id"])
            if m is None or th is None:
                continue
            small = th[1]
            amap = cv2.resize(m[0], (small.shape[1], small.shape[0]), interpolation=cv2.INTER_AREA)
            d = abs(m[1] - s["score"])
            deltas.append(d)
            same = d <= 1e-6 * max(1.0, abs(s["score"]))
            cap = tr("xp.map_caption", "LIVE map · live score {l:.3f} {eq} stored {s:.3f}").format(
                l=m[1], s=s["score"], eq="=" if same else "≠")
            t.set(heat(small, amap, self._pix_thr), cap)
            card.set_live(cap, same)
        self.live_delta = max(deltas) if deltas else 0.0

    # ------------------------------------------------------------------ advanced panels
    def _matrix(self):
        flips = self.only_flips.isChecked()
        rows, keys = [], []
        for key, items in self.groups.items():
            st = self.analysis["stability"][f"{key[0]}:{key[1]}"]
            if flips and st["consistent"]:
                continue
            by_c = {s["lighting_condition"]: s for s in items}
            tip = tr("xp.matrix_row_tip", "{kind} · errors {e} · decision changes: {ch} · score range {rng:.2f} · min. "
                                          "distance to threshold {mg:.2f}").format(
                kind=scene_kind(key[0]), e=st["errors"], ch=tr("xp.yes", "yes") if not st["consistent"] else tr("xp.no", "no"),
                rng=st["score_range"], mg=st["margin_min"])
            r = [{"text": key[1], "bold": True, "tip": tip},
                 {"text": tr("xp.anom_short", "anom.") if key[0] == "bad" else tr("xp.norm_short", "norm."),
                  "fg": CLASS_COLORS[key[0]], "tip": tip}]
            for c in self.conds:
                s = by_c.get(c)
                if s is None:
                    r.append("—")
                    continue
                code = outcome(key[0] == "bad", is_nok(s["score"], self.thr))
                r.append({"text": code, "bg": OUTCOME_COLORS[code], "center": True, "bold": code in ("FP", "FN"),
                          "alpha": 0.55 if code in ("FP", "FN") else 0.22,
                          "tip": f"{c}: {tr('xp.score', 'score')} {s['score']:.3f} — {outcome_name(code)}"})
            r += [{"text": str(st["errors"]), "center": True, "tip": tip},
                  {"text": "—" if st["consistent"] else "⚠", "center": True, "tip": tip}]
            rows.append(r)
            keys.append(key)
        hdr = [tr("xp.h_scene", "Scene"), tr("xp.h_class", "Class")] + [short_condition(c) for c in self.conds] + [
            tr("xp.h_errors", "Errors"), "⇄"]
        self.matrix.set_rows(rows, hdr, keys)
        self.matrix.horizontalHeaderItem(len(hdr) - 1).setToolTip(tr("xp.h_changes", "Decision changes"))
        for j, c in enumerate(self.conds, start=2):
            self.matrix.horizontalHeaderItem(j).setToolTip(c)
        self.matrix.select_payload(self.cur_key)
        self._matrix_info()

    def _matrix_cell(self, row: int, col: int):
        """A click on a condition cell selects that scene AND that lighting condition everywhere."""
        key = self.matrix.payload(row)
        if key is None:
            return
        self.select_scene(key)
        k = col - 2
        if 0 <= k < len(self.conds):
            self.select_condition(self._cond_to_item(k))

    def _matrix_info(self):
        """Detail line under the matrix: every per-scene metric of the selected scene and its selected cell."""
        if not self.cur or self.cur_key not in self.groups:      # e.g. while switching models
            self.matrix_detail.setText("")
            return
        st = self.analysis["stability"][f"{self.cur_key[0]}:{self.cur_key[1]}"]
        s = self.cur[min(self.cond_idx, len(self.cur) - 1)]
        nok = is_nok(s["score"], self.thr)
        code = outcome(self.cur_key[0] == "bad", nok)
        self.matrix_detail.setText(tr("xp.matrix_detail", "<b>{scene} {sid}</b> ({kind}) · errors {e} · decision changes: "
                                                          "{ch} · {rng_l} {rng:.2f} · {mg_l} {mg:.2f}<br><b>{cond}</b>: "
                                                          "{score:.3f} {op} {thr:.3f} → {d} — {o}").format(
            scene=tr("xp.h_scene", "Scene"), sid=self.cur_key[1], kind=scene_kind(self.cur_key[0]), e=st["errors"],
            ch=tr("xp.yes", "yes") if not st["consistent"] else tr("xp.no", "no"),
            rng_l=tr("xp.h_range", "Score range").lower(), rng=st["score_range"],
            mg_l=tr("xp.h_margin", "Min. distance to threshold").lower(), mg=st["margin_min"],
            cond=s["lighting_condition"], score=s["score"], op="≥" if nok else "&lt;", thr=self.thr,
            d="NOK" if nok else "OK", o=outcome_name(code)))

    def _auroc(self):
        a = self.analysis
        ci = lambda c: tr("xp.not_reported", "not reported") if not c else f"{c[0]:.3f}–{c[1]:.3f}"  # noqa: E731
        rows = []
        for c, p in a["per_condition"].items():
            note = tr("xp.train_light", "training lighting") if c == "regular" else tr("xp.unseen_light", "unseen in training")
            rows.append([{"text": c, "bold": True}, f"{p['n_normal']} / {p['n_anomalous']}", F(p["auroc"]), ci(p["ci"]), note])
        s = a["sufficiency"]
        rows.append([{"text": tr("xp.r_pooled", "All images (pooled)"), "bold": True},
                     f"{s['normal_scenes']} / {s['anomalous_scenes']} ({a['n_images']} img.)", F(a["pooled_auroc"]),
                     ci(a.get("pooled_ci_cluster")),
                     tr("xp.naive_ci", "naive CI {c} (ignores correlation)").format(c=ci(a.get("pooled_ci_naive")))])
        rows.append([{"text": tr("xp.r_scene_mean", "Scene level (mean score)"), "bold": True},
                     f"{s['normal_scenes']} / {s['anomalous_scenes']}", F(a["scene_auroc_mean"]),
                     ci(a.get("scene_auroc_mean_ci")), ""])
        rows.append([{"text": tr("xp.r_scene_max", "Scene level (max score)"), "bold": True},
                     f"{s['normal_scenes']} / {s['anomalous_scenes']}", F(a["scene_auroc_max"]), "—", ""])
        self.auroc_tbl.set_rows(rows, [tr("xp.h_condition", "Condition"), tr("xp.h_scenes", "normal / anomalous scenes"),
                                       "AUROC", tr("xp.h_cluster_ci", "95 % cluster CI"), tr("xp.h_note", "Note")])

    def _auroc_plot(self):
        a, w = self.analysis, self.auroc_chart
        w.fig.clear()
        ax = w.fig.add_subplot(111)
        style_ax(ax, w.dark)
        cs = list(a["per_condition"])
        sel = self.cur[self.cond_idx]["lighting_condition"] if self.cur else None
        for i, c in enumerate(cs):
            p = a["per_condition"][c]
            col = "#ffd43b" if c == sel else "#e599f7"
            if p["ci"]:
                ax.errorbar([i], [p["auroc"]], yerr=[[p["auroc"] - p["ci"][0]], [p["ci"][1] - p["auroc"]]], fmt="o",
                            color=col, ms=6, capsize=4, lw=1.6)
            else:
                ax.plot([i], [p["auroc"]], "o", color=col, ms=6)
        ax.axhline(a["pooled_auroc"], color="#adb5bd", ls="--", lw=1)
        ax.axhline(0.5, color="#868e96", ls=":", lw=1)
        ax.set_xticks(range(len(cs)), cs, fontsize=7.5)
        ax.set_ylim(0, 1.02)
        ax.set_ylabel("AUROC", fontsize=8)
        ax.set_title(tr("xp.auroc_plot_title", "AUROC per condition, 95 % cluster CI (dashed: pooled, dotted: chance)"),
                     fontsize=8, color=w.fg)
        w.draw()

    def _references(self):
        from core.research.catalog import get_catalog
        cat = get_catalog()
        ids = ["hecklerkram2026mvtecad2", "vo2026beyondbenchmark"]
        alg = cat.algorithms.get(self.entry.verified["method_id"]) or {}
        ids += [p for p in alg.get("publication_ids", [])[:1] if p not in ids]
        L = ["#### " + tr("xp.refs_title", "Scientific references"), ""]
        for pid in ids:
            p = cat.publication(pid)
            if not p:
                continue
            au = p.get("authors") or []
            who = (au[0] + (" et al." if len(au) > 1 else "")) if au else ""
            link = cat.doi_url(p.get("doi")) or p.get("paper_url")
            L.append(f"- **{p['title']}** — {who} ({p.get('year')}). *{p.get('venue', '')}*."
                     + (f" [{p.get('doi') or link}]({link})" if link else ""))
        L += ["", "_" + tr("xp.refs_note", "The statistics on this page (scene grouping, per-condition and scene-level "
                                            "AUROC, percentile cluster bootstrap over scenes) are implemented in "
                                            "`core/evaluation/scenes.py`. More references: Dataset Research Atlas → "
                                            "MVTec AD 2.") + "_"]
        self.refs.set_markdown("\n".join(L))

    def _report(self):
        a = self.analysis
        s = a["sufficiency"]
        ci = lambda c: tr("xp.not_reported", "not reported") if not c else f"{c[0]:.3f}–{c[1]:.3f}"  # noqa: E731
        L = [f"### {tr('xp.scene_analysis', 'Scene-aware analysis')} — {plabel(self.entry.verified)}",
             f"> {suff_text(s)}", "",
             "- " + tr("xp.rep_images", "Images: {n} = {sc} scenes × lighting conditions. Lighting variants of one scene "
                                        "are **not independent**.").format(n=a["n_images"], sc=a["n_scenes"]),
             "- " + tr("xp.rep_pooled", "Pooled image AUROC (all images): **{auc}** — 95 % CI resampling *scenes* (respects "
                                        "correlation; unreliable with so few scenes): {cc}; resampling *images* as if "
                                        "independent (ignores correlation, usually too narrow): {nc}.").format(
                 auc=F(a["pooled_auroc"]), cc=ci(a.get("pooled_ci_cluster")), nc=ci(a.get("pooled_ci_naive")))]
        cc, nc = a.get("pooled_ci_cluster"), a.get("pooled_ci_naive")
        if cc and nc:
            cw, nw = cc[1] - cc[0], nc[1] - nc[0]
            if cw > nw:
                L.append("- " + tr("xp.rep_wider", "Here the cluster interval is wider than the naive one ({cw:.3f} vs "
                                                   "{nw:.3f}), as expected when lighting variants are correlated.").format(cw=cw, nw=nw))
            else:
                L.append("- " + tr("xp.rep_narrower", "⚠ Here the cluster interval is **not** wider than the naive one "
                                                      "({cw:.3f} vs {nw:.3f}). With only {g} normal scenes the cluster "
                                                      "bootstrap has few distinct resamples ({d} of 1000) and itself "
                                                      "underestimates the uncertainty: read both intervals as rough "
                                                      "indications.").format(cw=cw, nw=nw, g=s["normal_scenes"], d=cc[2]))
        L += ["- " + tr("xp.rep_scene", "Scene-level AUROC (mean score of each scene): {m} (CI {c}); with max over "
                                        "conditions: {x}.").format(m=F(a["scene_auroc_mean"]), c=ci(a.get("scene_auroc_mean_ci")),
                                                                   x=F(a["scene_auroc_max"])),
              "- " + tr("xp.rep_decisions", "Decisions at the validation threshold ({t:.2f}): {k}/{n} scenes get the same "
                                            "decision under every lighting condition; {e} scenes have ≥ 1 error.").format(
                  t=self.thr, k=a["consistent_scenes"], n=a["n_scenes"], e=a["scenes_with_errors"]),
              "- " + tr("xp.rep_percond", "Per-condition AUROC uses one image per scene, so the images of one condition "
                                          "are independent; differences between conditions are exploratory and are not "
                                          "a ranking of lighting conditions."),
              "- " + tr("xp.rep_min", "With fewer than 3 scenes per class no interval is reported at all."),
              "- " + tr("xp.rep_split", "Local educational results on `test_public`; the official robustness evaluation "
                                        "uses the private `test_private_mixed` split on MVTec's server."),
              "- " + tr("xp.rep_live", "All scores, decisions, AUROCs and intervals on this page come from the stored "
                                       "per-image scores of the verified run ({date}). LIVE inference only adds anomaly "
                                       "maps; its scores are compared with the stored ones on each map.").format(
                  date=self.entry.verified.get("verified_date", "")),
              "", "_" + tr("xp.rep_footer", "Training and validation images are all `regular`; every other condition is "
                                            "unseen during fitting. Percentile bootstrap, 1000 resamples, seed 0.") + "_"]
        self.report.set_markdown("\n".join(L))


# ------------------------------------------------------------------------- C. metrics
class MetricsTab(QWidget):
    def __init__(self, page: ExperimentsPage):
        super().__init__()
        self.page = page
        lay = QVBoxLayout(self)
        self.presets = MR.stored_presets()
        sp = QSplitter(Qt.Horizontal)
        sp.addWidget(ExplanationBox(content.markdown("experiments", "metrics") or ""))
        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(4, 0, 0, 0)
        if not self.presets:
            rl.addWidget(missing(tr("xp.met_missing", "No stored results (configs/workshop_verified.json).")))
            sp.addWidget(right)
            lay.addWidget(sp, 1)
            self.ready = False
            return
        self.ready = True
        self.prov = ProvenanceBar()
        self.prov.set_state(
            tr("xp.prov_stored_c", "STORED RESULTS — every score, histogram and count comes from the stored per-image "
                                   "scores and validation calibration of the verified runs."),
            False, tr("xp.prov_none_c", "No inference here: moving the threshold only re-counts the decisions on the "
                                        "stored scores."))
        rl.addWidget(self.prov)
        hb = QHBoxLayout()
        self.model = QComboBox()
        for e in self.presets:
            self.model.addItem(plabel(e.verified), e.verified["preset_id"])
        hb.addWidget(shrinkable(self.model), 2)
        ex = QPushButton(tr("xp.example", "Real example: PaDiM on fruit_jelly"))
        ex.clicked.connect(lambda: self.select(METRICS_EXAMPLE))
        hb.addWidget(shrinkable(ex, 280), 1)
        rl.addLayout(hb)
        self.chart = MplCanvas(7.5, 3.4, 200)
        self.canvas, self.fig = self.chart.canvas, self.chart.fig
        rl.addWidget(self.chart, 1)
        sl = QHBoxLayout()
        sl.addWidget(QLabel(tr("xp.explore_thr", "Exploratory threshold")))
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, 1000)
        sl.addWidget(self.slider, 1)
        self.reset = QPushButton(tr("xp.reset_thr", "Back to the valid protocol threshold"))
        sl.addWidget(self.reset)                      # natural width: an 'Ignored' policy collapsed it to 0 px
        rl.addLayout(sl)
        self.banner = QLabel("")
        self.banner.setWordWrap(True)
        rl.addWidget(self.banner)
        self.tbl = table([], ["", tr("xp.valid", "Valid: validation p99"),
                              tr("xp.paper_rule", "Valid: paper rule (val. mean+3σ, map max)"),
                              tr("xp.explore", "Exploration (test-informed)")])
        self.tbl.setMinimumHeight(150)
        low = QWidget()
        lw = QVBoxLayout(low)
        lw.setContentsMargins(0, 0, 0, 0)
        lw.addWidget(self.tbl, 1)
        self.summary = muted("")
        lw.addWidget(self.summary)
        vs = QSplitter(Qt.Vertical)              # plot | decision table: resizable, no page scroll bar
        vs.addWidget(right)
        vs.addWidget(low)
        vs.setChildrenCollapsible(False)
        vs.setSizes([560, 420])
        sp.addWidget(vs)
        sp.setSizes([340, 1100])
        lay.addWidget(sp, 1)
        self.model.currentIndexChanged.connect(self._model)
        self.slider.valueChanged.connect(self._update)
        self.reset.clicked.connect(self._reset)
        self.select(METRICS_EXAMPLE)

    def select(self, preset_id: str):
        if not self.ready:
            return
        i = self.model.findData(preset_id)
        if i >= 0:
            if self.model.currentIndex() == i:
                self._model()
            else:
                self.model.setCurrentIndex(i)

    def _model(self):
        v = next(e for e in self.presets if e.verified["preset_id"] == self.model.currentData()).verified
        self.v = v
        self.y = np.array([1 if s["label"] == "bad" else 0 for s in v["samples"]])
        self.s = np.array([s["score"] for s in v["samples"]], float)
        self.mx = np.array([s["map_max"] if s["map_max"] is not None else np.nan for s in v["samples"]], float)
        self.val = np.array(v["calibration"]["normal_scores"], float)
        self.thr0 = v["calibration"]["image_threshold"]
        lo = float(min(self.s.min(), self.val.min() if self.val.size else self.s.min()))
        hi = float(max(self.s.max(), self.thr0))
        pad = 0.05 * (hi - lo or 1)
        self.lo, self.hi = lo - pad, hi + pad
        self._reset()

    def _thr(self):
        return self.lo + (self.hi - self.lo) * self.slider.value() / 1000

    def _reset(self):
        self.slider0 = int(round(1000 * (self.thr0 - self.lo) / (self.hi - self.lo)))
        self.slider.blockSignals(True)
        self.slider.setValue(self.slider0)
        self.slider.blockSignals(False)
        self._update()

    def _update(self):
        exploring = self.slider.value() != self.slider0
        v = self.v
        thr = self._thr() if exploring else self.thr0
        if exploring:
            self.banner.setText("⚠ " + tr("xp.retro", "RETROSPECTIVE EXPLORATION ON TEST LABELS — this threshold was chosen "
                                                      "while looking at the test set. Its numbers are NOT a valid evaluation "
                                                      "result and must not be reported."))
            self.banner.setStyleSheet("background:#8a5a00;color:white;padding:6px;border-radius:4px;font-weight:bold")
        else:
            self.banner.setText("✓ " + tr("xp.valid_note", "Valid protocol: threshold fixed on validation normal images only "
                                                           "(99th percentile), before looking at any test label."))
            self.banner.setStyleSheet("background:#1f6f3f;color:white;padding:6px;border-radius:4px")
        d0 = M.decision_metrics(self.y, self.s, self.thr0, "validation")
        de = M.decision_metrics(self.y, self.s, thr, "exploration", threshold_from_test=True)
        pthr = v["calibration"].get("paper_pixel_threshold")
        ok_mx = ~np.isnan(self.mx)
        dp = M.decision_metrics(self.y[ok_mx], self.mx[ok_mx], pthr, "paper rule") if pthr is not None and ok_mx.all() else None
        def col(d, t):
            if d is None:
                return ["—"] * 9
            return [f"{t:.4g}", str(d.tp), str(d.fp), str(d.tn), str(d.fn), F(d.precision), F(d.recall), F(d.f1),
                    f"{F(d.false_acceptance_rate)} / {F(d.false_rejection_rate)}"]
        names = [tr("xp.t_thr", "Threshold"), tr("xp.t_tp", "TP (defect → NOK)"), tr("xp.t_fp", "FP (good → NOK, false alarm)"),
                 tr("xp.t_tn", "TN (good → OK)"), tr("xp.t_fn", "FN (defect → OK, missed)"), tr("xp.t_prec", "Precision"),
                 tr("xp.t_rec", "Recall"), "F1", tr("xp.t_far", "FAR / FRR")]
        cols = [col(d0, self.thr0), col(dp, pthr if pthr is not None else 0), col(de, thr)]
        self.tbl.setRowCount(len(names))
        for i, n in enumerate(names):
            it = QTableWidgetItem(n)
            code = {1: "TP", 2: "FP", 3: "TN", 4: "FN"}.get(i)
            if code:                                         # same outcome colours as experiments A and B
                c = QColor(OUTCOME_COLORS[code])
                c.setAlphaF(0.35)
                it.setBackground(c)
            self.tbl.setItem(i, 0, it)
            for j in range(3):
                self.tbl.setItem(i, j + 1, QTableWidgetItem(cols[j][i]))
        fit_rows(self.tbl)
        m = v["metrics"]
        self.summary.setText(tr("xp.met_summary", "{label} — threshold-free: image AUROC {auc}, AP {ap}, pixel AUROC {pauc}, "
                                                  "AU-PRO@0.05 {aupro}. Paper rule uses the max of the anomaly map as image score "
                                                  "and the validation pixel threshold mean+3σ = {pthr} (MVTec AD 2 paper, Sec. "
                                                  "IV-B2). Verified {date}.").format(
            label=plabel(v), auc=F(m["image_auroc"]), ap=F(m["image_ap"]), pauc=F(m["pixel_auroc"]),
            aupro=F(m.get("au_pro_05")), pthr=F(pthr, 4), date=v["verified_date"]))
        self._plot(thr, exploring)

    def _plot(self, thr, exploring):
        self.fig.clear()
        a1, a2 = self.fig.add_subplot(121), self.fig.add_subplot(122)
        for ax in (a1, a2):
            style_ax(ax, self.canvas.dark)
        bins = np.linspace(self.lo, self.hi, 40)
        if self.val.size:
            a1.hist(self.val, bins=bins, alpha=0.55, color="#adb5bd",
                    label=tr("xp.lg_val", "validation normals (threshold source)"))
        a1.hist(self.s[self.y == 0], bins=bins, alpha=0.65, color="#4dabf7", label=tr("xp.lg_test_normal", "test normal"))
        a1.hist(self.s[self.y == 1], bins=bins, alpha=0.65, color="#ff6b6b", label=tr("xp.lg_test_anom", "test anomalous"))
        a1.axvline(self.thr0, color="#51cf66", lw=2, label=tr("xp.lg_valid_thr", "valid threshold (validation p99)"))
        if exploring:
            a1.axvline(thr, color="#ffa94d", lw=2, ls="--", label=tr("xp.lg_explore", "exploration (test-informed)"))
        a1.set_xlabel(tr("xp.ax_score", "image anomaly score"))
        a1.set_ylabel(tr("xp.ax_images", "images"))
        a1.legend(fontsize=7, framealpha=0.3)
        fpr, tpr = roc_points(self.y, self.s)
        a2.plot(fpr, tpr, color="#e599f7", lw=2)
        a2.plot([0, 1], [0, 1], color="#868e96", ls=":")
        for t, c, mk in ((self.thr0, "#51cf66", "o"),) + (((thr, "#ffa94d", "s"),) if exploring else ()):
            p = self.s >= t
            tp = (p & (self.y == 1)).sum() / max(1, (self.y == 1).sum())
            fp = (p & (self.y == 0)).sum() / max(1, (self.y == 0).sum())
            a2.plot([fp], [tp], mk, color=c, ms=9)
        a2.set_xlabel(tr("xp.ax_fpr", "false-positive rate"))
        a2.set_ylabel(tr("xp.ax_tpr", "true-positive rate"))
        a2.set_title(tr("xp.roc_title", "ROC — AUROC {auc} (ranking, no threshold)").format(auc=F(self.v["metrics"]["image_auroc"])),
                     fontsize=8, wrap=True)
        self.canvas.draw_idle()
