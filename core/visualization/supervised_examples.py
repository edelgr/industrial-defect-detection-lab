"""Worked examples for group B — supervised learning (classification, detection, segmentation).

IDD Lab does not ship trained CNN / YOLO / U-Net defect models (they need labelled defect data and GPU
training; see research_catalog/algorithms/BC_supervised_reconstruction.yaml). These examples therefore
re-enact the three supervised SET-UPS with small models that ARE trained here, in about a second, on
LABELLED procedural synthetic images (``core.datasets.synthetic`` generators, fixed seeds):

* classification — image-level labels (OK / NOK) → global pooling of feature maps → logistic regression
  → one probability per image, no location;
* detection      — box labels with a class → per-cell objectness, class and box regression on an S×S grid
  (the YOLO formulation) → confidence threshold + non-maximum suppression → boxes, classes, confidences;
* segmentation   — pixel masks → per-pixel logistic regression on coarse "context" features (the
  encoder/bottleneck) with and without fine full-resolution features (the skip connections) → mask.

Hand-crafted, explainable features replace the learned convolutions, so the numbers are not those of a
real CNN, YOLO or U-Net — only the kind of supervision, the mechanism and the kind of output are.
Everything is deterministic (fixed seeds) and needs only NumPy, OpenCV and scikit-learn.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import cv2
import numpy as np

from core.datasets import synthetic as syn
from core.visualization.method_examples import (SIZE, Example, Stage, Text, _u8, binary, chart, heat, montage,
                                                outline, overlay_mask, to_rgb)

CLASSES = ("scratch", "stain", "spot")                  # synthetic defect kinds used as supervised classes
CLASS_COLORS = {"scratch": (230, 80, 70), "stain": (80, 150, 245), "spot": (240, 190, 40)}
GT_COLOR = (60, 220, 110)
CELL = 16                                               # detection grid: SIZE / CELL = 10 × 10 cells
S = SIZE // CELL


# =========================================================================== labelled synthetic data
@dataclass
class Labelled:
    image: np.ndarray                     # RGB uint8
    mask: np.ndarray                      # union of all defect pixels (uint8 0/1)
    parts: list[tuple[str, np.ndarray]]   # (class, mask) per defect
    boxes: list[tuple[str, tuple[int, int, int, int]]]   # (class, (x0, y0, x1, y1)) per connected component

    @property
    def nok(self) -> bool:
        return bool(self.mask.any())


def _boxes(parts) -> list:
    out = []
    for kind, m in parts:
        n, _, st, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), connectivity=8)
        for i in range(1, n):
            x, y, w, h, area = st[i]
            if area >= 6:
                out.append((kind, (int(x), int(y), int(x + w), int(y + h))))
    return out


def sample(seed: int, kinds: tuple[str, ...] = ()) -> Labelled:
    """One synthetic image with the given defect kinds (empty → a good part)."""
    rng = np.random.default_rng(seed)
    img = syn._texture(rng, SIZE)
    parts, union = [], np.zeros((SIZE, SIZE), np.uint8)
    for k in kinds:
        img, m = syn._defect(rng, img, k)
        m = m & (1 - union)                                # a pixel belongs to one defect only
        if m.any():
            parts.append((k, m))
            union |= m
    return Labelled(_u8(img), union, parts, _boxes(parts))


def dataset(n: int, seed: int, p_good: float = 0.4, max_defects: int = 2) -> list[Labelled]:
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        if rng.uniform() < p_good:
            kinds: tuple = ()
        else:
            k = int(rng.integers(1, max_defects + 1))
            kinds = tuple(rng.choice(CLASSES, size=k, replace=False))
        out.append(sample(seed * 1000 + i, kinds))
    return out


# =========================================================================== hand-crafted feature maps
def residual(img: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(dark, bright, edge) maps: deviation from the local background (a stand-in for learned filters)."""
    g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    g = cv2.GaussianBlur(g, (0, 0), 1.0)
    r = g - cv2.GaussianBlur(g, (0, 0), 8.0)                       # local background removed
    gx, gy = cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1)
    return np.maximum(-r, 0), np.maximum(r, 0), np.sqrt(gx ** 2 + gy ** 2)


def pool(a: np.ndarray, cell: int, fn=np.mean) -> np.ndarray:
    h, w = a.shape[0] // cell, a.shape[1] // cell
    return fn(a[: h * cell, : w * cell].reshape(h, cell, w, cell), axis=(1, 3))


def upsample(a: np.ndarray, size: int = SIZE) -> np.ndarray:
    return cv2.resize(a.astype(np.float32), (size, size), interpolation=cv2.INTER_LINEAR)


# ---------------------------------------------------------------- classification (global pooling)
def cls_feature_maps(img) -> list[np.ndarray]:
    d, b, e = residual(img)
    return [pool(d, 8, np.max), pool(b, 8, np.mean), pool(e, 8, np.mean)]     # 20 × 20 feature maps


def cls_vector(img) -> np.ndarray:
    """Global pooling: max and mean of every feature map → one short vector per image."""
    maps = cls_feature_maps(img)
    return np.array([f(m) for m in maps for f in (np.max, np.mean)], np.float32)


# ---------------------------------------------------------------- segmentation (encoder–decoder)
def seg_coarse(img) -> np.ndarray:
    """Bottleneck: context computed at 1/8 resolution (20 × 20), then upsampled (decoder without skips)."""
    d, b, e = residual(img)
    feats = [pool(d, 8), pool(b, 8), pool(d, 8, np.max), pool(b, 8, np.max), pool(e, 8)]
    return np.stack([upsample(f) for f in feats], -1)


def seg_fine(img) -> np.ndarray:
    """Skip connections: full-resolution detail (where exactly the contrast is)."""
    d, b, e = residual(img)
    return np.stack([d, b, e, cv2.GaussianBlur(d, (0, 0), 1.5), cv2.GaussianBlur(b, (0, 0), 1.5)], -1)


# ---------------------------------------------------------------- detection (S × S grid)
NF = 13


def _smoothed(img):
    d, b, _ = residual(img)
    return cv2.GaussianBlur(d, (0, 0), 1.5), cv2.GaussianBlur(b, (0, 0), 1.5)


def regions(img, hd: float, hb: float) -> np.ndarray:
    """Unusual regions: hysteresis (strong seeds grown over weaker pixels) on the dark and bright maps, with
    thresholds set on GOOD training images (99.9th percentile of the texture's own response)."""
    from skimage.filters import apply_hysteresis_threshold
    d, b = _smoothed(img)
    return apply_hysteresis_threshold(d, 0.7 * hd, hd) | apply_hysteresis_threshold(b, 0.7 * hb, hb)


def det_cell_features(img, hd: float, hb: float) -> np.ndarray:
    """Per grid cell: the unusual region whose centre lies in the cell (size, extent, offset of its centre,
    dark/bright contrast, elongation, compactness) → (S, S, NF); zeros when no region is centred there.
    The detection head LEARNS from labelled boxes which regions are defects, their class and their box."""
    d, b = _smoothed(img)
    on = regions(img, hd, hb)
    n, lab, st, cen = cv2.connectedComponentsWithStats(on.astype(np.uint8), connectivity=8)
    F = np.zeros((S, S, NF), np.float32)
    best: dict = {}
    for k in range(1, n):
        cx, cy = cen[k]
        cell = (min(S - 1, int(cy // CELL)), min(S - 1, int(cx // CELL)))
        if cell not in best or st[k, 4] > st[best[cell], 4]:
            best[cell] = k
    for (i, j), k in best.items():
        x, y, w, h, area = st[k]
        m = lab == k
        dm, bm = float(d[m].mean()) / hd, float(b[m].mean()) / hb
        ys, xs = np.nonzero(m)
        ev = np.sort(np.linalg.eigvalsh(np.cov(np.stack([xs, ys])) + 0.25 * np.eye(2))) if area > 2 else np.array([.25, .25])
        F[i, j] = [np.log1p(area), w / CELL, h / CELL, (x + w / 2 - (j + .5) * CELL) / CELL,
                   (y + h / 2 - (i + .5) * CELL) / CELL, dm, bm, max(dm, bm), np.log(ev[1] / ev[0]), area / (w * h),
                   float(d[m].max()) / hd, float(b[m].max()) / hb, 1.0]
    return F


def _box_targets(lab: Labelled):
    """YOLO-style targets: the cell containing a box centre is responsible for it."""
    obj = np.zeros((S, S), np.int32)
    cls = np.full((S, S), -1, np.int32)
    box = np.zeros((S, S, 4), np.float32)
    for kind, (x0, y0, x1, y1) in lab.boxes:
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        i, j = min(S - 1, int(cy // CELL)), min(S - 1, int(cx // CELL))
        obj[i, j] = 1
        cls[i, j] = CLASSES.index(kind)
        box[i, j] = [(cx - (j + 0.5) * CELL) / CELL, (cy - (i + 0.5) * CELL) / CELL, (x1 - x0) / CELL, (y1 - y0) / CELL]
    return obj, cls, box


# =========================================================================== the three trained models
N_TRAIN, N_VAL = 80, 40


@lru_cache(maxsize=1)
def train_set() -> tuple[list[Labelled], list[Labelled]]:
    return dataset(N_TRAIN, seed=7), dataset(N_VAL, seed=8)


@lru_cache(maxsize=1)
def classifier():
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    tr_, va = train_set()
    X = np.stack([cls_vector(s.image) for s in tr_])
    y = np.array([s.nok for s in tr_], int)
    m = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000)).fit(X, y)
    Xv = np.stack([cls_vector(s.image) for s in va])
    yv = np.array([s.nok for s in va], int)
    pv = m.predict_proba(Xv)[:, 1]
    return m, X, y, pv, yv


@lru_cache(maxsize=1)
def segmenter():
    """Two per-pixel models trained on the SAME pixel masks: context only, and context + skip details."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    tr_, _ = train_set()
    rng = np.random.default_rng(0)
    Xc, Xf, Y = [], [], []
    for s in tr_[:40]:
        c, f = seg_coarse(s.image).reshape(-1, 5), seg_fine(s.image).reshape(-1, 5)
        m = s.mask.ravel() > 0
        pos = np.flatnonzero(m)
        neg = rng.choice(np.flatnonzero(~m), size=max(300, 30 * len(pos)), replace=False)   # ~ real class balance
        idx = np.concatenate([pos, neg])
        Xc.append(c[idx]); Xf.append(f[idx]); Y.append(m[idx].astype(int))
    Xc, Xf, Y = np.concatenate(Xc), np.concatenate(Xf), np.concatenate(Y)
    ctx = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(Xc, Y)
    full = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(np.hstack([Xc, Xf]), Y)
    return ctx, full


@lru_cache(maxsize=1)
def segmenter_validation() -> dict:
    """Mean Dice on the held-out labelled NOK images, with and without the skip-connection features."""
    _, va = train_set()
    ds, dn = [], []
    for s in va:
        if s.nok:
            p_ctx, p_full = segment(s.image)
            ds.append(dice(p_full >= 0.5, s.mask))
            dn.append(dice(p_ctx >= 0.5, s.mask))
    return {"dice_skip": float(np.mean(ds)), "dice_noskip": float(np.mean(dn)), "n": len(ds)}


def segment(img) -> tuple[np.ndarray, np.ndarray]:
    """(probability without skip connections, probability with skip connections), each SIZE × SIZE."""
    ctx, full = segmenter()
    c, f = seg_coarse(img).reshape(-1, 5), seg_fine(img).reshape(-1, 5)
    p_ctx = ctx.predict_proba(c)[:, 1].reshape(SIZE, SIZE)
    p_full = full.predict_proba(np.hstack([c, f]))[:, 1].reshape(SIZE, SIZE)
    return p_ctx, p_full


@lru_cache(maxsize=1)
def detector():
    """Objectness, class and box heads trained on the labelled boxes (cells without a region are skipped)."""
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    tr_, _ = train_set()
    goods = [s for s in tr_ if not s.nok]
    hd = float(np.percentile(np.concatenate([_smoothed(s.image)[0].ravel() for s in goods]), 99.9))
    hb = float(np.percentile(np.concatenate([_smoothed(s.image)[1].ravel() for s in goods]), 99.9))
    F, O, C, B = [], [], [], []
    for s in tr_:
        f = det_cell_features(s.image, hd, hb)
        o, c, b = _box_targets(s)
        F.append(f.reshape(-1, NF)); O.append(o.ravel()); C.append(c.ravel()); B.append(b.reshape(-1, 4))
    F, O, C, B = np.concatenate(F), np.concatenate(O), np.concatenate(C), np.concatenate(B)
    has = F[:, -1] > 0
    pos = has & (O == 1)
    obj = make_pipeline(StandardScaler(), LogisticRegression(max_iter=4000)).fit(F[has], O[has])
    cls = make_pipeline(StandardScaler(), LogisticRegression(max_iter=4000)).fit(F[pos], C[pos])
    box = make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit(F[pos], B[pos])
    return obj, cls, box, hd, hb


@dataclass
class Detection:
    cls: str
    conf: float
    box: tuple[int, int, int, int]
    cell: tuple[int, int]


def _iou(a, b) -> float:
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def nms(dets: list[Detection], iou: float = 0.3) -> list[Detection]:
    keep: list[Detection] = []
    for d in sorted(dets, key=lambda d: -d.conf):
        if all(not (k.cls == d.cls and _iou(k.box, d.box) > iou) for k in keep):
            keep.append(d)
    return keep


DET_CONF = 0.5


def detect(img) -> tuple[np.ndarray, np.ndarray, list[Detection]]:
    """(objectness grid, class-index grid, every candidate box before thresholding and NMS)."""
    obj, cls, box, hd, hb = detector()
    F = det_cell_features(img, hd, hb).reshape(-1, NF)
    po = np.zeros(S * S)
    ci = np.zeros(S * S, int)
    cands = []
    idx = np.flatnonzero(F[:, -1] > 0)
    if len(idx):
        p_o = obj.predict_proba(F[idx])[:, 1]
        p_c = cls.predict_proba(F[idx])
        bx = box.predict(F[idx])
        for q, n in enumerate(idx):
            i, j = divmod(int(n), S)
            k = int(np.argmax(p_c[q]))
            po[n], ci[n] = p_o[q], k
            dx, dy, w, h = bx[q]
            cx, cy = (j + 0.5 + dx) * CELL, (i + 0.5 + dy) * CELL
            w, h = max(3.0, w * CELL), max(3.0, h * CELL)
            b = tuple(int(round(v)) for v in (max(0, cx - w / 2), max(0, cy - h / 2), min(SIZE - 1, cx + w / 2),
                                              min(SIZE - 1, cy + h / 2)))
            cands.append(Detection(CLASSES[k], float(p_o[q] * p_c[q, k]), b, (i, j)))
    return po.reshape(S, S), ci.reshape(S, S), cands


@lru_cache(maxsize=1)
def detector_validation() -> dict:
    """Precision / recall of the toy detector on the held-out labelled images (box IoU ≥ 0.3, same class)."""
    _, va = train_set()
    tp = fp = fn = 0
    for s in va:
        kept = nms([c for c in detect(s.image)[2] if c.conf >= DET_CONF])
        used: set = set()
        for gk, gb in s.boxes:
            hit = next((q for q, k in enumerate(kept) if q not in used and k.cls == gk and _iou(k.box, gb) >= 0.3), None)
            if hit is None:
                fn += 1
            else:
                tp += 1
                used.add(hit)
        fp += len(kept) - len(used)
    return {"precision": tp / max(1, tp + fp), "recall": tp / max(1, tp + fn), "n_boxes": tp + fn}


# =========================================================================== drawing
def _scale(img: np.ndarray, f: int = 2) -> np.ndarray:
    return cv2.resize(to_rgb(img), None, fx=f, fy=f, interpolation=cv2.INTER_NEAREST)


def ascii_text(text: str) -> str:
    """OpenCV's Hershey fonts draw ASCII only: strip accents (raya, mancha…) so labels stay readable."""
    import unicodedata
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")


def _label(out, text, x, y, color, scale=0.5):
    text = ascii_text(text)
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    y = max(th + 4, y)
    cv2.rectangle(out, (x, y - th - 4), (x + tw + 4, y + 1), color, -1)
    cv2.putText(out, text, (x + 2, y - 2), cv2.FONT_HERSHEY_SIMPLEX, scale, (15, 15, 15), 1, cv2.LINE_AA)


def draw_boxes(img, boxes, f: int = 2, gt: bool = False, t: Text | None = None) -> np.ndarray:
    """``boxes``: Detection list (predictions) or [(class, box)] (labels)."""
    out = _scale(img, f)
    for d in boxes:
        kind, b, conf = (d.cls, d.box, d.conf) if isinstance(d, Detection) else (d[0], d[1], None)
        col = GT_COLOR if gt else CLASS_COLORS[kind]
        x0, y0, x1, y1 = (int(v * f) for v in b)
        cv2.rectangle(out, (x0, y0), (x1, y1), col, 2)
        name = t("cls." + kind, kind) if t else kind
        _label(out, name if conf is None else f"{name} {conf:.2f}", x0, y0 - 2, col)
    return out


def banner(img, text: str, nok: bool | None, f: int = 2) -> np.ndarray:
    """Image-level output or label: one tag for the whole image (no location)."""
    out = _scale(img, f)
    col = (220, 80, 70) if nok else (70, 190, 110) if nok is not None else (200, 200, 200)
    h = 30
    over = out.copy()
    cv2.rectangle(over, (0, 0), (out.shape[1], h), col, -1)
    out = cv2.addWeighted(over, 0.85, out, 0.15, 0)
    cv2.putText(out, ascii_text(text), (8, h - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def grid(img, f: int = 2, color=(80, 200, 255)) -> np.ndarray:
    out = _scale(img, f)
    for k in range(1, S):
        out[k * CELL * f, :] = color
        out[:, k * CELL * f] = color
    return out


def cell_map(values: np.ndarray, f: int = 2, lo=0.0, hi=1.0) -> np.ndarray:
    return cv2.resize(heat(values, lo, hi), (SIZE * f, SIZE * f), interpolation=cv2.INTER_NEAREST)


def class_cells(img, cls_idx: np.ndarray, po: np.ndarray, f: int = 2, min_obj: float = 0.2) -> np.ndarray:
    out = _scale(img, f).astype(np.float32)
    for i in range(S):
        for j in range(S):
            if po[i, j] >= min_obj:
                col = np.array(CLASS_COLORS[CLASSES[cls_idx[i, j]]], np.float32)
                a = 0.25 + 0.5 * float(po[i, j])
                sl = (slice(i * CELL * f, (i + 1) * CELL * f), slice(j * CELL * f, (j + 1) * CELL * f))
                out[sl] = out[sl] * (1 - a) + col * a
    return out.astype(np.uint8)


def mask_compare(img, gt, pred, f: int = 2) -> np.ndarray:
    """Overlay: yellow = correct (both), red = predicted only (false positive), green = missed (GT only)."""
    out = _scale(img, f).astype(np.float32)
    g = cv2.resize(gt.astype(np.uint8), None, fx=f, fy=f, interpolation=cv2.INTER_NEAREST) > 0
    p = cv2.resize(pred.astype(np.uint8), None, fx=f, fy=f, interpolation=cv2.INTER_NEAREST) > 0
    for sel, col in ((g & p, (240, 200, 40)), (p & ~g, (230, 70, 60)), (g & ~p, (60, 220, 110))):
        out[sel] = out[sel] * 0.35 + np.array(col, np.float32) * 0.65
    return out.astype(np.uint8)


def dice(a, b) -> float:
    a, b = a > 0, b > 0
    s = a.sum() + b.sum()
    return float(2 * (a & b).sum() / s) if s else 1.0


# =========================================================================== examples
def ex_cnn_classification(t: Text) -> Example:
    m, X, y, pv, yv = classifier()
    tr_, _ = train_set()
    good = [s for s in tr_ if not s.nok][:2]
    bad = [s for s in tr_ if s.nok][:2]
    train_tiles = [banner(s.image, t("cls.tag_ok", "OK"), False, 1) for s in good] + \
                  [banner(s.image, t("cls.tag_nok", "NOK"), True, 1) for s in bad]
    test = sample(4245, ("scratch",))           # representative: P(NOK) close to the validation NOK images
    # the same picture rotated by 180°: the scratch is elsewhere, nothing else changes (no seam)
    shifted = Labelled(np.ascontiguousarray(np.rot90(test.image, 2)), np.ascontiguousarray(np.rot90(test.mask, 2)), [], [])
    ok_img = sample(4343)
    p = lambda img: float(m.predict_proba(cls_vector(img)[None])[0, 1])
    p_test, p_shift, p_ok = p(test.image), p(shifted.image), p(ok_img.image)
    maps = cls_feature_maps(test.image)
    vec = cls_vector(test.image)
    mu, sd = X[y == 0].mean(0), X[y == 0].std(0) + 1e-6
    names = [t("chart.f_dark_max", "dark max"), t("chart.f_dark_mean", "dark mean"),
             t("chart.f_bright_max", "bright max"), t("chart.f_bright_mean", "bright mean"),
             t("chart.f_edge_max", "edge max"), t("chart.f_edge_mean", "edge mean")]

    def draw_pool(ax):
        z = (vec - mu) / sd
        ax.bar(range(len(z)), z, color=["#e5534b" if v > 3 else "#7f8c99" for v in z])
        ax.axhline(0, color="k", lw=0.6)
        ax.set_xticks(range(len(z)))
        ax.set_xticklabels(names, fontsize=6, rotation=35, ha="right")
        ax.set_ylabel(t("chart.z_vs_ok", "vs. OK images (z)"), fontsize=7)

    def draw_scores(ax):
        bins = np.linspace(0, 1, 21)
        ax.hist(pv[yv == 0], bins=bins, color="#4e9f55", alpha=0.75, label=t("chart.val_ok", "validation OK"))
        ax.hist(pv[yv == 1], bins=bins, color="#d9534f", alpha=0.75, label=t("chart.val_nok", "validation NOK"))
        ax.axvline(0.5, color="k", ls="--", lw=1)
        ax.axvline(p_test, color="#9b6cd0", lw=2, label=t("chart.this_image", "this image"))
        ax.set_xlabel(t("chart.p_nok", "P(NOK)"), fontsize=8)
        ax.legend(fontsize=6, frameon=False)
    acc = float(((pv >= 0.5) == (yv == 1)).mean())
    tag = lambda pr: f"{t('cls.tag_nok', 'NOK') if pr >= 0.5 else t('cls.tag_ok', 'OK')}  P(NOK) = {pr:.2f}"
    return Example("cnn_classification", "toy", [
        Stage("train", montage(train_tiles, cols=2, more=N_TRAIN - 4), "train"),
        Stage("test", outline(test.image, test.mask), "input"),
        Stage("features", montage([heat(upsample(a)) for a in maps], cols=2)),
        Stage("pooling", chart(draw_pool, 320, 220)),
        Stage("classifier", chart(draw_scores, 320, 220)),
        Stage("decision", banner(test.image, tag(p_test), p_test >= 0.5), "output")],
        [Stage("out_input", _scale(outline(test.image, test.mask)), "input"),
         Stage("out_decision", banner(test.image, tag(p_test), p_test >= 0.5), "output"),
         Stage("out_shifted", banner(shifted.image, tag(p_shift), p_shift >= 0.5), "output")],
        {"n_train": N_TRAIN, "n_val": N_VAL, "p_test": p_test, "p_shift": p_shift, "p_ok": p_ok,
         "val_acc": acc * 100, "n_features": len(vec)})


def ex_unet(t: Text) -> Example:
    tr_, _ = train_set()
    train_tiles = [overlay_mask(s.image, s.mask, color=GT_COLOR) for s in tr_ if s.nok][:4]
    test = sample(5114, ("scratch", "spot"))    # representative: Dice close to the validation median
    d, b, e = residual(test.image)
    p_ctx, p_full = segment(test.image)
    m_ctx, m_full = p_ctx >= 0.5, p_full >= 0.5
    val = segmenter_validation()
    enc1 = np.maximum(d, b)
    enc2 = pool(enc1, 4, np.max)
    bott = pool(enc1, 8)
    return Example("unet", "toy", [
        Stage("train", montage(train_tiles, cols=2, more=N_TRAIN - 4), "train"),
        Stage("test", test.image, "input"),
        Stage("enc1", heat(enc1)),
        Stage("enc2", cv2.resize(heat(enc2), (SIZE, SIZE), interpolation=cv2.INTER_NEAREST)),
        Stage("bottleneck", cv2.resize(heat(bott), (SIZE, SIZE), interpolation=cv2.INTER_NEAREST)),
        Stage("dec_noskip", heat(p_ctx, 0, 1)),
        Stage("dec_skip", heat(p_full, 0, 1)),
        Stage("mask", binary(m_full), "output")],
        [Stage("out_input", _scale(test.image), "input"),
         Stage("out_gt", _scale(binary(test.mask)), "input"),
         Stage("out_pred", _scale(binary(m_full)), "output"),
         Stage("out_overlay", mask_compare(test.image, test.mask, m_full), "output")],
        {"n_train": N_TRAIN, "dice_skip": dice(m_full, test.mask), "dice_noskip": dice(m_ctx, test.mask),
         "gt_px": int(test.mask.sum()), "pred_px": int(m_full.sum()), "val_dice_skip": val["dice_skip"],
         "val_dice_noskip": val["dice_noskip"], "n_val": val["n"]})


def ex_yolo(t: Text) -> Example:
    tr_, _ = train_set()
    with_boxes = [s for s in tr_ if s.boxes][:4]
    train_tiles = [draw_boxes(s.image, s.boxes, 1, t=t) for s in with_boxes]
    test = sample(6215, ("scratch", "stain"))   # representative: one faint stain missed, as in validation
    po, ci, cands = detect(test.image)
    raw = [c for c in cands if c.conf >= 0.15]
    kept = nms([c for c in cands if c.conf >= DET_CONF])
    matched = sum(any(k.cls == gk and _iou(k.box, gb) >= 0.3 for k in kept) for gk, gb in test.boxes)
    _, _, _, hd, hb = detector()
    reg = regions(test.image, hd, hb)
    val = detector_validation()
    return Example("yolo", "toy", [
        Stage("train", montage(train_tiles, cols=2, more=N_TRAIN - 4), "train"),
        Stage("grid", grid(test.image), "input"),
        Stage("features", grid(overlay_mask(test.image, reg, color=(80, 200, 255)), 1)),
        Stage("objectness", cell_map(po)),
        Stage("classes", class_cells(test.image, ci, po)),
        Stage("raw", draw_boxes(test.image, raw, t=t)),
        Stage("nms", draw_boxes(test.image, kept, t=t), "output")],
        [Stage("out_input", draw_boxes(test.image, test.boxes, gt=True, t=t), "input"),
         Stage("out_boxes", draw_boxes(test.image, kept, t=t), "output")],
        {"n_train": N_TRAIN, "grid": S, "n_raw": len(raw), "n_kept": len(kept), "n_gt": len(test.boxes),
         "n_matched": matched, "conf": DET_CONF,
         "best": max((k.conf for k in kept), default=0.0), "val_precision": val["precision"] * 100,
         "val_recall": val["recall"] * 100, "val_boxes": val["n_boxes"]})


# =========================================================================== cross-method comparison
def comparison(t: Text) -> dict:
    """Same synthetic image, three supervised set-ups: the LABEL each needs and the OUTPUT each gives."""
    s = sample(7373, ("scratch", "spot"))
    m = classifier()[0]
    pr = float(m.predict_proba(cls_vector(s.image)[None])[0, 1])
    _, _, cands = detect(s.image)
    kept = nms([c for c in cands if c.conf >= DET_CONF])
    _, p_full = segment(s.image)
    tag = f"{t('cls.tag_nok', 'NOK') if pr >= 0.5 else t('cls.tag_ok', 'OK')}  P(NOK) = {pr:.2f}"
    return {
        "label_cls": banner(s.image, t("cls.tag_nok", "NOK"), True),
        "label_det": draw_boxes(s.image, s.boxes, t=t),
        "label_seg": overlay_mask(_scale(s.image), cv2.resize(s.mask, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST),
                                  color=GT_COLOR, alpha=0.7),
        "out_cls": banner(s.image, tag, pr >= 0.5),
        "out_det": draw_boxes(s.image, kept, t=t),
        "out_seg": mask_compare(s.image, s.mask, p_full >= 0.5),
        "facts": {"p": pr, "n_boxes": len(kept), "dice": dice(p_full >= 0.5, s.mask)},
    }


BUILDERS = {"cnn_classification": ex_cnn_classification, "unet": ex_unet, "yolo": ex_yolo}
