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
from PySide6.QtWidgets import (QComboBox, QFrame, QGridLayout, QHBoxLayout, QHeaderView, QLabel,
                               QListWidget, QListWidgetItem, QPushButton, QScrollArea, QSlider,
                               QSplitter, QTableWidget, QTableWidgetItem, QTabWidget, QVBoxLayout,
                               QWidget)

from app.i18n import tr
from app.widgets.common import PageHeader, colorize, markdown_browser, mask_overlay, muted, overlay
from app.widgets.image_viewer import TitledImage
from app.workers.task import start
from core import settings
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
    t.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
    t.horizontalHeader().setStretchLastSection(True)
    t.verticalHeader().setVisible(False)
    return t


def tile(title: str, min_w: int = 150, min_h: int = 110) -> TitledImage:
    """TitledImage whose caption wraps (long captions must not force a huge minimum width)."""
    v = TitledImage(title)
    v.caption.setWordWrap(True)
    v.setMinimumSize(min_w, min_h)
    return v


def shrinkable(w, min_width: int = 120):
    """Combos and buttons with long texts must not dictate the page's minimum width."""
    from PySide6.QtWidgets import QSizePolicy
    if isinstance(w, QComboBox):
        w.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        w.setMinimumContentsLength(18)
    w.setSizePolicy(QSizePolicy.Ignored, w.sizePolicy().verticalPolicy())
    w.setMinimumWidth(min_width)
    return w


def mode_label(live: bool, reasons: list[str], date: str = "") -> QLabel:
    """Make the kind of evidence explicit: live inference on this computer vs. stored verified results."""
    if live:
        txt = "● " + tr("xp.mode_live", "LIVE — inference runs on this computer with the Workshop Verified checkpoints.")
        style = "background:#1f6f3f;color:white;padding:5px;border-radius:4px"
    else:
        txt = ("◆ " + tr("xp.mode_stored", "STORED RESULTS — computed and verified on the instructor's computer "
                                           "({date}) with real MVTec AD 2 data; nothing is computed on this computer.").format(date=date)
               + ("  " + tr("xp.mode_needs", "Live inference needs: {r}.").format(r="; ".join(reasons)) if reasons else ""))
        style = "background:#3b4a6b;color:white;padding:5px;border-radius:4px"
    lab = QLabel(txt)
    lab.setWordWrap(True)
    lab.setStyleSheet(style)
    return lab


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


def mpl_canvas(width=6.0, height=3.2):
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
    from matplotlib.figure import Figure
    dark = settings.get().theme == "dark"
    fig = Figure(figsize=(width, height), tight_layout=True, facecolor="#1e2228" if dark else "white")
    c = FigureCanvasQTAgg(fig)
    c.setMinimumHeight(int(height * 80))
    c.dark = dark
    return c, fig


def style_ax(ax, dark: bool):
    fg = "#d8dde3" if dark else "#222"
    ax.set_facecolor("#1e2228" if dark else "white")
    for s in ax.spines.values():
        s.set_color(fg)
    ax.tick_params(colors=fg, labelsize=8)
    ax.xaxis.label.set_color(fg)
    ax.yaxis.label.set_color(fg)
    ax.title.set_color(fg)


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
        lay.addWidget(mode_label(self.live_ok, self.reasons, self.entries[0].verified.get("verified_date", "")))
        sp = QSplitter(Qt.Horizontal)
        left_tabs = QTabWidget()
        left = QWidget()
        ll = QVBoxLayout(left)
        ll.setContentsMargins(4, 4, 4, 4)
        left_tabs.addTab(left, tr("xp.results_tab", "Run && results"))
        left_tabs.addTab(markdown_browser(content.markdown("experiments", "resolution") or ""),
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
        self.stored.setMinimumHeight(260)
        ll.addWidget(self.stored, 2)
        self.live = table([], [tr("xp.this_image", "This image"), "256×256", "1024×256"])
        self.live.setMinimumHeight(170)
        ll.addWidget(self.live, 1)
        sp.addWidget(left_tabs)

        right = QWidget()
        self.grid = QGridLayout(right)
        self.views = {}
        names = [("orig", "Original + ground truth (full resolution)"), ("zoom", "Zoom on the largest defect")]
        for c, (k, t) in enumerate(names):
            self.views[k] = tile(tr(f"xp.v_{k}", t), 300, 150)
            self.grid.addWidget(self.views[k], 0, c * 2, 1, 2)
        for r, e in enumerate(self.entries, start=1):
            hw = e.manifest["input_hw"]
            tag = f"{hw[1]}×{hw[0]}"
            for c, (k, t) in enumerate([("pre", f"{tag}: " + tr("xp.v_pre", "what the network sees")),
                                        ("heat", f"{tag}: " + tr("xp.v_heat", "anomaly heat-map")),
                                        ("err", f"{tag}: " + tr("xp.v_err", "predicted mask vs GT")),
                                        ("zheat", f"{tag}: " + tr("xp.v_zheat", "zoom"))]):
                v = tile(t, 160, 150)
                self.views[f"{k}{r}"] = v
                self.grid.addWidget(v, r, c)
        for r in range(3):
            self.grid.setRowStretch(r, 1)
        for c in range(4):
            self.grid.setColumnStretch(c, 1)
        self.grid.addWidget(muted(tr("xp.legend", "Heat-map scale: map ÷ calibrated pixel threshold, 0–2 (1 = threshold). "
                                                 "Mask colours: green = correct defect pixels, red = false alarms, blue = missed defect pixels.")), 3, 0, 1, 4)
        sp.addWidget(_scroll(right))
        sp.setSizes([420, 1100])
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
            live.append({"score": res.score, "dec": "NOK" if res.score >= cal.image_threshold else "OK",
                         "pa": pa, "dice": d, "lat": res.latency_ms, "gpu": res.gpu_memory_mb})
        gt_lab = tr("xp.anomalous", "anomalous").upper() if s.label == "bad" else tr("xp.normal", "normal").upper()
        rows = [[tr("xp.r_gt", "Ground truth"), gt_lab, ""],
                [tr("xp.r_score", "Image score / threshold")] + [f"{x['score']:.2f} / {c.image_threshold:.2f}" for x, (_, c, _r) in zip(live, out)],
                [tr("xp.r_decision", "Decision")] + [x["dec"] + (" ✓" if (x["dec"] == "NOK") == (s.label == "bad") else " ✗") for x in live],
                [tr("xp.r_pix_auroc", "Pixel AUROC (this image)")] + [F(x["pa"]) for x in live],
                [tr("xp.r_dice", "Dice at pixel threshold")] + [F(x["dice"]) for x in live],
                [tr("xp.r_latency", "Latency now (ms)")] + [f"{x['lat']:.1f}" for x in live],
                [tr("xp.r_gpu", "GPU peak now (MB)")] + ["—" if x["gpu"] is None else f"{x['gpu']:.0f}" for x in live]]
        rows[0][2] = rows[0][1]
        self.live.setRowCount(len(rows))
        for i, r in enumerate(rows):
            for j, v in enumerate(r):
                self.live.setItem(i, j, QTableWidgetItem(str(v)))


# ------------------------------------------------------------------------- B. illumination
class IlluminationTab(QWidget):
    def __init__(self, page: ExperimentsPage):
        super().__init__()
        self.page = page
        self.task = None
        self.entry: MR.CheckpointEntry | None = None
        self.analysis: dict | None = None
        lay = QVBoxLayout(self)
        presets = [e for e in MR.stored_presets("D05") if not e.verified["params"].get("keep_aspect")]
        if not presets:
            lay.addWidget(missing(tr("xp.ill_missing", "No stored MVTec AD 2 results (configs/workshop_verified.json).")))
            lay.addStretch(1)
            self.ready = False
            return
        self.ready = True
        self.presets = presets
        self.mode_holder = QVBoxLayout()
        lay.addLayout(self.mode_holder)
        sp = QSplitter(Qt.Horizontal)
        left = QWidget()
        ll = QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 0, 0)
        ll.addWidget(markdown_browser(content.markdown("experiments", "illumination") or ""), 2)
        self.model = QComboBox()
        for e in presets:
            self.model.addItem(plabel(e.verified), e.key)
        ll.addWidget(QLabel(tr("xp.model", "Workshop Verified model")))
        ll.addWidget(shrinkable(self.model))
        self.scenes = QListWidget()
        ll.addWidget(QLabel(tr("xp.scenes", "Scenes (one physical part, several lighting conditions)")))
        ll.addWidget(self.scenes, 2)
        sp.addWidget(left)

        right = QWidget()
        rl = QVBoxLayout(right)
        self.scene_title = QLabel("")
        self.scene_title.setWordWrap(True)
        self.scene_title.setObjectName("CardTitle")
        rl.addWidget(self.scene_title)
        self.strip = QGridLayout()
        rl.addLayout(self.strip)
        self.maps_btn = QPushButton("▶  " + tr("xp.compute_maps", "Compute anomaly maps for this scene"))
        self.maps_btn.setObjectName("Primary")
        self.maps_btn.clicked.connect(self._maps)
        hb = QHBoxLayout()
        hb.addWidget(self.maps_btn)
        hb.addStretch(1)
        rl.addLayout(hb)
        self.msg = muted("")                      # own line: long (translated) messages must not cover the button
        rl.addWidget(self.msg)
        self.canvas, self.fig = mpl_canvas(7.5, 3.0)
        rl.addWidget(self.canvas)
        self.report = markdown_browser("", min_height=260)
        rl.addWidget(self.report, 1)
        sp.addWidget(_scroll(right))
        sp.setSizes([380, 1050])
        lay.addWidget(sp, 1)
        self.model.currentIndexChanged.connect(self._model)
        self.scenes.currentRowChanged.connect(self._scene)
        self._model()

    def _model(self):
        from core.evaluation import scenes
        self.entry = next(e for e in self.presets if e.key == self.model.currentData())
        v = self.entry.verified
        self.thr = v["calibration"]["image_threshold"]
        self.analysis = scenes.analyse(v["samples"], self.thr, n_boot=1000)
        self.groups = scenes.group(v["samples"])
        installed = v["category"] in dreg.installed_categories("D05")
        self.idx = dreg.load_index("D05", v["category"]) if installed else None
        self.by_id = {s.sample_id: s for s in self.idx.samples} if self.idx else {}
        self.reasons = live_reasons([self.entry], "D05", v["category"])
        while self.mode_holder.count():
            w = self.mode_holder.takeAt(0).widget()
            if w:
                w.deleteLater()
        self.mode_holder.addWidget(mode_label(not self.reasons, self.reasons, v.get("verified_date", "")))
        self.maps_btn.setEnabled(not self.reasons)
        self.maps_btn.setToolTip("; ".join(self.reasons))
        self.scenes.blockSignals(True)
        self.scenes.clear()
        for (lab, sid), items in self.groups.items():
            st = self.analysis["stability"][f"{lab}:{sid}"]
            flag = ("✓ " + tr("xp.stable", "stable")) if st["consistent"] and not st["errors"] else (
                "⚠ " + tr("xp.n_errors", "{n} error(s)").format(n=st["errors"])
                + ("" if st["consistent"] else ", " + tr("xp.flips", "decision flips")))
            kind = (tr("xp.anomalous", "anomalous") if lab == "bad" else tr("xp.normal", "normal")).upper()
            it = QListWidgetItem(f"{kind} · {tr('xp.scene', 'scene')} {sid} · {len(items)} {tr('xp.conditions', 'conditions')}\n      {flag}")
            it.setData(Qt.UserRole, (lab, sid))
            self.scenes.addItem(it)
        self.scenes.blockSignals(False)
        self._plot()
        self._report()
        self.scenes.setCurrentRow(0)

    def _scene(self, row):
        if row < 0:
            return
        key = self.scenes.item(row).data(Qt.UserRole)
        items = self.groups[key]
        self.cur = items
        while self.strip.count():
            w = self.strip.takeAt(0).widget()
            if w:
                w.hide()
                w.setParent(None)
                w.deleteLater()
        self.tiles, self.maptiles = [], []
        for c, s in enumerate(items):
            sm = self.by_id.get(s["sample_id"])
            if sm is not None:
                img = read_rgb8(sm.primary_path)
                gt = read_mask(sm.masks["rgb"], img.shape[:2]) if sm.masks.get("rgb") else None
                small = cv2.resize(img, (img.shape[1] // 4, img.shape[0] // 4), interpolation=cv2.INTER_AREA)
            else:
                gt = small = None              # dataset not installed: stored score and decision only
            dec = "NOK" if s["score"] >= self.thr else "OK"
            ok = (dec == "NOK") == (s["label"] == "bad")
            err = "" if ok else (" · " + (tr("xp.false_alarm", "FALSE ALARM") if s["label"] == "good"
                                          else tr("xp.missed", "MISSED DEFECT")))
            t = tile(f"{s['lighting_condition']}: {tr('xp.score', 'score')} {s['score']:.2f} → {dec} {'✓' if ok else '✗'}{err}", 115, 110)
            if small is None:
                t.set(None, t.caption.text() + "\n" + tr("xp.no_image", "(image not installed on this computer)"))
            else:
                t.set(mask_overlay(small, gt[::4, ::4][:small.shape[0], :small.shape[1]]) if gt is not None else small)
            row, col = 2 * (c // 3), c % 3          # 3 columns: 6 lighting conditions fit 1280-px screens
            self.strip.addWidget(t, row, col)
            m = tile(tr("xp.map_pending", "anomaly map: press 'Compute'") if not self.reasons
                     else tr("xp.map_unavailable", "anomaly map: live inference not available here"), 115, 110)
            self.strip.addWidget(m, row + 1, col)
            self.maptiles.append(m)
        st = self.analysis["stability"][f"{key[0]}:{key[1]}"]
        self.scene_title.setText(tr("xp.scene_title", "Scene {sid} ({kind}) — score range {rng:.2f}, closest distance to "
                                                      "threshold {mg:.2f}, {cons}, {n} error(s)").format(
            kind=tr("xp.anomalous", "anomalous") if key[0] == "bad" else tr("xp.normal", "normal"), sid=key[1],
            rng=st["score_range"], mg=st["margin_min"], n=st["errors"],
            cons=tr("xp.same_decision", "same decision under all conditions") if st["consistent"]
            else tr("xp.decision_changes", "decision changes with lighting")))
        self._plot(highlight=key)

    def _maps(self):
        items, e, sessions = self.cur, self.entry, self.page.sessions
        by_id = self.by_id
        if self.reasons:
            return
        self.maps_btn.setEnabled(False)

        def job(progress, cancel):
            sess = sessions.get(e.entry)
            out = []
            for i, s in enumerate(items):
                progress(i / len(items), f"{s['lighting_condition']}…")
                img = read_rgb8(by_id[s["sample_id"]].primary_path)
                out.append((img, sess.predict_image(img)))
            return sess.calibration.pixel_threshold, out
        self.task = start(job, on_done=self._show_maps, on_error=lambda m, tb: (self.maps_btn.setEnabled(True), self.msg.setText("✗ " + m)),
                          on_progress=lambda f, m: self.msg.setText(m))

    def _show_maps(self, payload):
        self.maps_btn.setEnabled(True)
        thr, out = payload
        self.msg.setText("✓ " + tr("xp.same_scale", "Same colour scale for all conditions (map ÷ calibrated pixel threshold)."))
        for t, (img, r), s in zip(self.maptiles, out, self.cur):
            small = cv2.resize(img, (img.shape[1] // 4, img.shape[0] // 4), interpolation=cv2.INTER_AREA)
            amap = cv2.resize(r.anomaly_map, (small.shape[1], small.shape[0]), interpolation=cv2.INTER_AREA)
            t.set(heat(small, amap, thr), f"{tr('xp.map', 'map')} · {s['lighting_condition']} · {tr('xp.score', 'score')} {r.score:.2f}")

    def _plot(self, highlight=None):
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        style_ax(ax, self.canvas.dark)
        conds = []
        for items in self.groups.values():
            for s in items:
                if s["lighting_condition"] not in conds:
                    conds.append(s["lighting_condition"])
        for key, items in self.groups.items():
            x = [conds.index(s["lighting_condition"]) for s in items]
            y = [s["score"] for s in items]
            col = "#ff6b6b" if key[0] == "bad" else "#4dabf7"
            lw, al = (2.6, 1.0) if key == highlight else (1.0, 0.45)
            ax.plot(x, y, "-o", color=col, lw=lw, alpha=al, ms=3)
        ax.axhline(self.thr, color="#51cf66", ls="--", lw=1.5)
        ax.set_xticks(range(len(conds)), conds)
        ax.set_ylabel(tr("xp.ax_score", "image anomaly score"))
        ax.set_title(tr("xp.ill_plot_title", "One line per scene — red: anomalous, blue: normal, green dashed: validation threshold"),
                     fontsize=8, wrap=True)
        self.canvas.draw_idle()

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
                 auc=F(a["pooled_auroc"]), cc=ci(a.get("pooled_ci_cluster")), nc=ci(a.get("pooled_ci_naive"))),
             "- " + tr("xp.rep_scene", "Scene-level AUROC (mean score of each scene): {m} (CI {c}); with max over "
                                       "conditions: {x}.").format(m=F(a["scene_auroc_mean"]), c=ci(a.get("scene_auroc_mean_ci")),
                                                                  x=F(a["scene_auroc_max"])),
             "- " + tr("xp.rep_decisions", "Decisions at the validation threshold ({t:.2f}): {k}/{n} scenes get the same "
                                           "decision under every lighting condition; {e} scenes have ≥ 1 error.").format(
                 t=self.thr, k=a["consistent_scenes"], n=a["n_scenes"], e=a["scenes_with_errors"]), "",
             "| " + " | ".join([tr("xp.h_condition", "Condition"), tr("xp.h_scenes", "normal / anomalous scenes"), "AUROC",
                                tr("xp.h_cluster_ci", "95 % cluster CI")]) + " |", "|---|---|---|---|"]
        for c, p in a["per_condition"].items():
            L.append(f"| {c} | {p['n_normal']} / {p['n_anomalous']} | {F(p['auroc'])} | {ci(p['ci'])} |")
        L += ["", "_" + tr("xp.rep_footer", "Training and validation images are all `regular`; every other condition is "
                                            "unseen during fitting. Percentile bootstrap, 1000 resamples, seed 0.") + "_"]
        self.report.setMarkdown("\n".join(L))


# ------------------------------------------------------------------------- C. metrics
class MetricsTab(QWidget):
    def __init__(self, page: ExperimentsPage):
        super().__init__()
        self.page = page
        lay = QVBoxLayout(self)
        self.presets = MR.stored_presets()
        sp = QSplitter(Qt.Horizontal)
        sp.addWidget(markdown_browser(content.markdown("experiments", "metrics") or ""))
        right = QWidget()
        rl = QVBoxLayout(right)
        if not self.presets:
            rl.addWidget(missing(tr("xp.met_missing", "No stored results (configs/workshop_verified.json).")))
            sp.addWidget(right)
            lay.addWidget(sp, 1)
            self.ready = False
            return
        self.ready = True
        hb = QHBoxLayout()
        self.model = QComboBox()
        for e in self.presets:
            self.model.addItem(plabel(e.verified), e.verified["preset_id"])
        hb.addWidget(shrinkable(self.model), 2)
        ex = QPushButton(tr("xp.example", "Real example: PaDiM on fruit_jelly"))
        ex.clicked.connect(lambda: self.select(METRICS_EXAMPLE))
        hb.addWidget(shrinkable(ex, 280), 1)
        rl.addLayout(hb)
        self.canvas, self.fig = mpl_canvas(7.5, 3.4)
        rl.addWidget(self.canvas)
        sl = QHBoxLayout()
        sl.addWidget(QLabel(tr("xp.explore_thr", "Exploratory threshold")))
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, 1000)
        sl.addWidget(self.slider, 1)
        self.reset = QPushButton(tr("xp.reset_thr", "Back to the valid protocol threshold"))
        sl.addWidget(shrinkable(self.reset, 260))
        rl.addLayout(sl)
        self.banner = QLabel("")
        self.banner.setWordWrap(True)
        rl.addWidget(self.banner)
        self.tbl = table([], ["", tr("xp.valid", "Valid: validation p99"),
                              tr("xp.paper_rule", "Valid: paper rule (val. mean+3σ, map max)"),
                              tr("xp.explore", "Exploration (test-informed)")])
        self.tbl.setMinimumHeight(330)
        rl.addWidget(self.tbl, 1)
        self.summary = muted("")
        rl.addWidget(self.summary)
        sp.addWidget(_scroll(right))
        sp.setSizes([400, 1100])
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
            self.tbl.setItem(i, 0, QTableWidgetItem(n))
            for j in range(3):
                self.tbl.setItem(i, j + 1, QTableWidgetItem(cols[j][i]))
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
