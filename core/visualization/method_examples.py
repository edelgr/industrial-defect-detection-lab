"""Visual worked examples for the Methods Library ("Example" tab).

Every example is computed at run time on PROCEDURAL SYNTHETIC images (``core.datasets.synthetic``
texture/defect generators, fixed seeds), so it needs no dataset, no download and no checkpoint, and
the pictures are freely redistributable with the Student Edition.

Two kinds of provenance, always shown to the student:

* ``implementation`` (classical group A): the final anomaly map / mask is produced by the REAL IDD
  Lab implementation (``core.algorithms.registry``); the intermediate panels replay exactly the same
  OpenCV operations with the method's default parameters (``tests/test_method_visuals.py`` checks
  that the replayed final stage equals the real output).
* ``toy`` (feature-based, foundation and 3D/multimodal methods): a small NumPy re-enactment of the
  method's MECHANISM (memory bank + nearest neighbour, per-position Gaussian, teacher/student
  regression, window aggregation, ICP registration, late fusion) using hand-crafted patch features
  instead of a pretrained network.  These pictures explain how the method decides; they are NOT
  outputs of the real model.

This module never changes an algorithm: it only calls the registry or re-enacts mechanisms.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import cv2
import numpy as np

from core.datasets import synthetic as syn

SIZE = 160
Text = Callable[[str, str], str]


def _t_default(key: str, default: str) -> str:
    return default


@dataclass
class Stage:
    key: str                  # caption key (education/<lang>/method_visuals.yaml → example.captions)
    image: np.ndarray         # RGB uint8
    role: str = "step"        # input | train | step | output


@dataclass
class Example:
    method_id: str
    provenance: str                       # implementation | toy
    stages: list[Stage]
    outputs: list[Stage]
    facts: dict = field(default_factory=dict)   # numbers interpolated into captions / interpretation


# ----------------------------------------------------------------------------- drawing helpers
def to_rgb(a: np.ndarray, lo: float | None = None, hi: float | None = None) -> np.ndarray:
    """Grey (uint8 or float) → RGB uint8 (floats are min-max scaled unless lo/hi are given)."""
    if a.ndim == 3:
        return a.astype(np.uint8)
    if a.dtype != np.uint8:
        lo = float(np.min(a)) if lo is None else lo
        hi = float(np.max(a)) if hi is None else hi
        a = (np.clip((a.astype(np.float32) - lo) / max(hi - lo, 1e-9), 0, 1) * 255).astype(np.uint8)
    return np.stack([a] * 3, -1)


def heat(a: np.ndarray, lo: float | None = None, hi: float | None = None) -> np.ndarray:
    a = a.astype(np.float32)
    lo = float(np.min(a)) if lo is None else lo
    hi = float(np.max(a)) if hi is None else hi
    n = np.clip((a - lo) / max(hi - lo, 1e-9), 0, 1)
    return cv2.cvtColor(cv2.applyColorMap((n * 255).astype(np.uint8), cv2.COLORMAP_INFERNO), cv2.COLOR_BGR2RGB)


def overlay_heat(img: np.ndarray, a: np.ndarray, alpha: float = 0.55, lo=None, hi=None) -> np.ndarray:
    return cv2.addWeighted(to_rgb(img), 1 - alpha, heat(a, lo, hi), alpha, 0)


def overlay_mask(img: np.ndarray, mask: np.ndarray, color=(255, 60, 60), alpha=0.5) -> np.ndarray:
    out = to_rgb(img).copy()
    m = mask.astype(bool)
    out[m] = (out[m] * (1 - alpha) + np.array(color) * alpha).astype(np.uint8)
    cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(out, cs, -1, color, 1)
    return out


def outline(img: np.ndarray, mask: np.ndarray, color=(60, 220, 110)) -> np.ndarray:
    """Draw the (synthetic) ground-truth outline on an image."""
    out = to_rgb(img).copy()
    cs, _ = cv2.findContours((mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(out, cs, -1, color, 1)
    return out


def binary(mask: np.ndarray) -> np.ndarray:
    return to_rgb((mask > 0).astype(np.uint8) * 255)


def montage(imgs: list[np.ndarray], cols: int = 2, size: int = SIZE, pad: int = 4, more: int = 0) -> np.ndarray:
    """Small grid of thumbnails (e.g. 'the normal training images'); ``more`` adds a '+N' tile."""
    tiles = [cv2.resize(to_rgb(i), ((size - pad * (cols - 1)) // cols,) * 2, interpolation=cv2.INTER_AREA) for i in imgs]
    t = tiles[0].shape[0]
    rows = (len(tiles) + cols - 1) // cols
    out = np.full((rows * t + pad * (rows - 1), size, 3), 40, np.uint8)
    for k, tile in enumerate(tiles):
        r, c = divmod(k, cols)
        out[r * (t + pad): r * (t + pad) + t, c * (t + pad): c * (t + pad) + t] = tile
    if more:
        r, c = divmod(len(tiles) - 1, cols)
        y, x = r * (t + pad) + t - 18, c * (t + pad) + 4
        cv2.rectangle(out, (x - 2, y - 13), (x + 40, y + 5), (20, 20, 20), -1)
        cv2.putText(out, f"+{more}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def grid_lines(img: np.ndarray, cell: int, color=(80, 200, 255)) -> np.ndarray:
    out = to_rgb(img).copy()
    for v in range(0, out.shape[0], cell):
        out[v, :] = (out[v, :] * 0.4 + np.array(color) * 0.6).astype(np.uint8)
    for u in range(0, out.shape[1], cell):
        out[:, u] = (out[:, u] * 0.4 + np.array(color) * 0.6).astype(np.uint8)
    return out


def up(grid: np.ndarray, size: int = SIZE, smooth: float = 0.0) -> np.ndarray:
    """Patch grid → image resolution (nearest, optionally Gaussian-smoothed like the real methods)."""
    a = cv2.resize(grid.astype(np.float32), (size, size), interpolation=cv2.INTER_NEAREST if smooth <= 0 else cv2.INTER_LINEAR)
    return cv2.GaussianBlur(a, (0, 0), smooth) if smooth > 0 else a


def chart(draw: Callable, w: int = 320, h: int = 220) -> np.ndarray:
    """Render a small matplotlib chart (Agg, no GUI) to an RGB array."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    fig = Figure(figsize=(w / 100, h / 100), dpi=100)
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)
    draw(ax)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(labelsize=7)
    fig.tight_layout(pad=0.5)
    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()


# ------------------------------------------------------------------------- synthetic images
def _u8(img01: np.ndarray) -> np.ndarray:
    g = (np.clip(img01, 0, 1) * 255).astype(np.uint8)
    return np.stack([g, g, g], -1)


def metal(seed: int, defect: str | None = None, size: int = SIZE) -> tuple[np.ndarray, np.ndarray]:
    """Synthetic brushed-metal image (RGB uint8) + defect mask, from the synthetic dataset generator."""
    rng = np.random.default_rng(seed)
    img = syn._texture(rng, size)
    mask = np.zeros((size, size), np.uint8)
    if defect:
        img, mask = syn._defect(rng, img, defect)
    return _u8(img), mask


def normals(n: int, seed: int = 100) -> list[np.ndarray]:
    return [metal(seed + i)[0] for i in range(n)]


def gray(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if img.ndim == 3 else img


# ------------------------------------------------------------------------- toy patch features
CELL = 8


def patch_features(img: np.ndarray, cell: int = CELL) -> np.ndarray:
    """Hand-crafted stand-in for CNN / ViT patch embeddings: (H/cell, W/cell, 6).

    Per patch (3×3-patch neighbourhood, like PatchCore's local aggregation): mean grey, contrast,
    mean |∂x|, mean |∂y|, darkest and brightest value.  Deliberately simple and explainable.
    """
    g = gray(img).astype(np.float32) / 255.0
    gx, gy = np.abs(cv2.Sobel(g, cv2.CV_32F, 1, 0)), np.abs(cv2.Sobel(g, cv2.CV_32F, 0, 1))
    h, w = g.shape[0] // cell, g.shape[1] // cell

    def pool(a, fn):
        return fn(a[: h * cell, : w * cell].reshape(h, cell, w, cell), axis=(1, 3))
    f = np.stack([pool(g, np.mean), pool(g, np.std), pool(gx, np.mean), pool(gy, np.mean),
                  pool(g, np.min), pool(g, np.max)], -1)
    return cv2.blur(f, (3, 3))                      # locally aware (3×3 neighbourhood)


# =============================================================================== classical (A)
def _real(method_id: str, img, reference=None, **params):
    from core.algorithms import registry
    return registry.create(method_id, **params).predict(img, reference)


def _outputs(img, gt, amap, mask, lo=None, hi=None):
    return [Stage("out_input", outline(img, gt), "input"),
            Stage("out_map", overlay_heat(img, amap, lo=lo, hi=hi), "output"),
            Stage("out_mask", overlay_mask(img, mask), "output")]


def ex_reference_difference(t: Text) -> Example:
    from core.algorithms.classical.methods import align_to_reference
    rng = np.random.default_rng(7)
    big, m = SIZE + 32, 16
    base = syn._texture(rng, big)                  # rendered larger, then cropped: no artificial borders
    M = cv2.getRotationMatrix2D((big / 2, big / 2), 1.5, 1.0)
    M[:, 2] += (3.0, -2.0)                          # the part sits slightly displaced in the test shot
    moved = cv2.warpAffine(base.astype(np.float32), M, (big, big))[m:-m, m:-m]
    base = base[m:-m, m:-m]
    ref01 = np.clip(base + rng.normal(0, 0.004, base.shape), 0, 1)
    test01, gt = syn._defect(rng, moved, "scratch")
    test01 = np.clip(test01 + rng.normal(0, 0.004, base.shape), 0, 1)
    ref, test = _u8(ref01), _u8(test01)
    res = _real("reference_difference", test, ref)          # REAL implementation (default parameters)
    p = {"blur": 5, "threshold": 30.0, "align": "euclidean"}
    g, r = gray(test), gray(ref)
    blur = lambda a: cv2.GaussianBlur(a, (p["blur"], p["blur"]), 0)
    d_raw = cv2.absdiff(blur(g), blur(r)).astype(np.float32)
    g_al, _, ok = align_to_reference(g, r, p["align"])
    d_al = cv2.absdiff(blur(g_al), blur(r)).astype(np.float32)
    hi = 2 * p["threshold"]
    return Example("reference_difference", "implementation", [
        Stage("reference", ref, "input"), Stage("test", outline(test, gt), "input"),
        Stage("diff_unaligned", heat(d_raw, 0, hi)), Stage("aligned", to_rgb(g_al)),
        Stage("diff_aligned", heat(d_al, 0, hi)), Stage("mask", binary(res.mask), "output")],
        _outputs(test, gt, res.anomaly_map, res.mask, 0, hi),
        {"score": res.score, "threshold": p["threshold"],
         "unaligned_px": float((d_raw > p["threshold"]).mean() * 100), "aligned_px": float(res.mask.mean() * 100),
         "_replay": d_al, "_real_map": res.anomaly_map})


def _stain_image(seed=3, dark=0.22):
    rng = np.random.default_rng(seed)
    img = syn._texture(rng, SIZE)
    gt = np.zeros((SIZE, SIZE), np.uint8)
    cv2.ellipse(gt, (62, 92), (30, 18), 25, 0, 360, 1, -1)
    cv2.ellipse(gt, (108, 50), (14, 10), -30, 0, 360, 1, -1)
    soft = cv2.GaussianBlur(gt.astype(np.float32), (0, 0), 1.5)
    return _u8(img * (1 - soft) + dark * soft), gt


def ex_otsu(t: Text) -> Example:
    img, gt = _stain_image()
    res = _real("otsu", img)
    g = cv2.GaussianBlur(gray(img), (5, 5), 0)
    thr = res.extras["otsu_threshold_grey"]
    hist = np.asarray(res.extras["histogram"])

    def draw(ax):
        ax.bar(np.arange(256), hist, width=1.0, color="#7f8c99")
        ax.bar(np.arange(int(thr) + 1), hist[: int(thr) + 1], width=1.0, color="#e5534b")
        ax.axvline(thr, color="#2bb3a3", lw=2)
        ax.text(thr + 4, hist.max() * 0.9, f"t = {thr:.0f}", color="#107d72", fontsize=8)
        ax.set_xlabel(t("chart.grey_level", "grey level"), fontsize=8)
        ax.set_ylabel(t("chart.pixels", "pixels"), fontsize=8)
    return Example("otsu", "implementation", [
        Stage("input", outline(img, gt), "input"), Stage("blurred", to_rgb(g)),
        Stage("histogram", chart(draw)), Stage("mask", binary(res.mask), "output")],
        _outputs(img, gt, res.anomaly_map, res.mask),
        {"otsu_t": thr, "flagged_pct": float(res.mask.mean() * 100), "_replay": 255.0 - g.astype(np.float32), "_real_map": res.anomaly_map})


def polished(seed: int) -> np.ndarray:
    """Finer, lower-contrast variant of the synthetic texture (a polished rather than brushed part)."""
    img = syn._texture(np.random.default_rng(seed), SIZE)
    return img.mean() + 0.4 * (img - img.mean())


def _pits_image(seed=11, depth=0.3):
    img = polished(seed)
    gt = np.zeros((SIZE, SIZE), np.uint8)
    for c, r in (((40, 50), 3), ((100, 40), 4), ((120, 115), 3)):
        cv2.circle(gt, c, r, 1, -1)
    cv2.line(gt, (30, 120), (85, 100), 1, 2)
    return _u8(np.where(gt > 0, img - depth, img)), gt


def ex_morphology(t: Text) -> Example:
    img, gt = _pits_image()
    res = _real("morphology", img)                       # defaults: black-hat, ellipse 15, thr 25, open 3
    g = gray(img)
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    closing = cv2.morphologyEx(g, cv2.MORPH_CLOSE, se)
    bh = cv2.morphologyEx(g, cv2.MORPH_BLACKHAT, se).astype(np.float32)
    raw = bh > 25.0
    return Example("morphology", "implementation", [
        Stage("input", outline(img, gt), "input"), Stage("closing", to_rgb(closing)),
        Stage("blackhat", heat(bh, 0, 80)), Stage("threshold", binary(raw)),
        Stage("cleaned", binary(res.mask), "output")],
        _outputs(img, gt, res.anomaly_map, res.mask, 0, 80),
        {"score": res.score, "raw_px": int(raw.sum()), "clean_px": int(res.mask.sum()), "_replay": bh, "_real_map": res.anomaly_map})


def ex_canny(t: Text) -> Example:
    rng = np.random.default_rng(21)
    base = syn._texture(rng, SIZE)
    gt = np.zeros((SIZE, SIZE), np.uint8)
    cv2.line(gt, (25, 40), (120, 70), 1, 2)
    pts = np.array([[105, 100], [80, 140], [135, 140]], np.int32)
    cv2.fillPoly(gt, [pts], 1)
    img = _u8(np.where(gt > 0, base - 0.28, base))
    res = _real("canny", img)                            # defaults: low 40, high 120, blur 5, window 15
    g = cv2.GaussianBlur(gray(img), (5, 5), 0)
    mag = np.hypot(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1))
    edges = cv2.Canny(g, 40, 120)
    density = cv2.blur((edges > 0).astype(np.float32), (15, 15))
    ed = img.copy()
    ed[edges > 0] = (255, 40, 40)
    return Example("canny", "implementation", [
        Stage("input", outline(img, gt), "input"), Stage("blurred", to_rgb(g)),
        Stage("gradient", heat(mag)), Stage("edges", ed), Stage("density", heat(density, 0, 0.4)),
        Stage("mask", binary(res.mask), "output")],
        _outputs(img, gt, res.anomaly_map, res.mask, 0, 0.4),
        {"edge_pct": float((edges > 0).mean() * 100), "_replay": density, "_real_map": res.anomaly_map})


def ex_gabor(t: Text) -> Example:
    img, gt = metal(5, "texture")
    from core.algorithms import registry
    m = registry.create("gabor")
    res = m.predict(img)
    bank = m._bank()                                     # the real filter bank (3 λ × 6 θ)
    g = gray(img).astype(np.float32) / 255.0
    n = int(m.params["n_orientations"])
    tiles = []
    for k in bank[n: 2 * n]:                             # λ = 8 px, all orientations
        kk = cv2.resize(k, (48, 48), interpolation=cv2.INTER_CUBIC)
        tiles.append(to_rgb(kk, -np.abs(kk).max(), np.abs(kk).max()))
    energies = [np.log(cv2.blur(cv2.filter2D(g, cv2.CV_32F, bank[n + i]) ** 2, (15, 15)) + 1e-8) for i in range(n)]
    return Example("gabor", "implementation", [
        Stage("input", outline(img, gt), "input"), Stage("bank", montage(tiles, cols=3)),
        Stage("energy_0", heat(energies[0])), Stage("energy_90", heat(energies[n // 2])),
        Stage("zscore", heat(res.anomaly_map, 0, 8)), Stage("mask", binary(res.mask), "output")],
        _outputs(img, gt, res.anomaly_map, res.mask, 0, 8),
        {"n_filters": len(bank), "score": res.score})


def ex_adaptive_threshold(t: Text) -> Example:
    base = polished(31)
    yy, xx = np.mgrid[0:SIZE, 0:SIZE].astype(np.float32) / SIZE
    lit = base * (0.45 + 0.9 * xx)                       # strong illumination gradient (left dark → right bright)
    gt = np.zeros((SIZE, SIZE), np.uint8)
    for c, r in (((30, 50), 4), ((80, 110), 5), ((135, 60), 4)):
        cv2.circle(gt, c, r, 1, -1)
    cv2.line(gt, (95, 30), (140, 35), 1, 2)
    img = _u8(np.where(gt > 0, lit * 0.6, lit))
    res = _real("adaptive_threshold", img)               # defaults: gaussian, block 51, C 12, dark
    g = gray(img)
    _, glob = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    local = cv2.GaussianBlur(g.astype(np.float32), (51, 51), 0)
    diff = local - g
    return Example("adaptive_threshold", "implementation", [
        Stage("input", outline(img, gt), "input"), Stage("global_otsu", binary(glob)),
        Stage("local_mean", to_rgb(local.astype(np.uint8))), Stage("deviation", heat(diff, 0, 40)),
        Stage("mask", binary(res.mask), "output")],
        _outputs(img, gt, res.anomaly_map, res.mask, 0, 40),
        {"global_pct": float((glob > 0).mean() * 100), "local_pct": float(res.mask.mean() * 100), "_replay": diff, "_real_map": res.anomaly_map})


# ======================================================================= toy mechanisms (D–G)
N_TRAIN = 16


def _standardise(train_feats: np.ndarray):
    flat = train_feats.reshape(-1, train_feats.shape[-1])
    mu, sd = flat.mean(0), flat.std(0) + 1e-6
    return lambda f: (f - mu) / sd


def _calibrated(score_fn, held_out: list[np.ndarray], q: float = 99.5) -> float:
    """Pixel threshold = high percentile of the map on held-out NORMAL images (never the test image)."""
    return float(np.percentile(np.concatenate([score_fn(h).ravel() for h in held_out]), q))


def ex_padim(t: Text) -> Example:
    train = normals(N_TRAIN)
    held = normals(2, seed=300)
    img, gt = metal(52, "geometric")
    F = np.stack([patch_features(x) for x in train])               # (N, h, w, d)
    z = _standardise(F)
    F = z(F)
    mu = F.mean(0)
    d = F.shape[-1]
    X = F - mu
    cov = np.einsum("nhwi,nhwj->hwij", X, X) / (len(F) - 1) + 0.05 * np.eye(d)
    icov = np.linalg.inv(cov)

    def maha(im):
        x = z(patch_features(im)) - mu
        return np.sqrt(np.einsum("hwi,hwij,hwj->hw", x, icov, x))
    grid = maha(img)
    amap = up(grid, smooth=4)
    thr = _calibrated(lambda h: up(maha(h), smooth=4), held)
    mask = amap > thr
    return Example("padim", "toy", [
        Stage("train", montage(train[:4], more=N_TRAIN - 4), "train"),
        Stage("features", grid_lines(img, CELL)),
        Stage("mean", heat(up(mu[..., 0]))), Stage("variance", heat(up(np.sqrt(cov[..., 1, 1])))),
        Stage("test", outline(img, gt), "input"), Stage("mahalanobis", heat(up(grid))),
        Stage("mask", binary(mask), "output")],
        _outputs(img, gt, amap, mask), {"n_train": N_TRAIN, "threshold": thr, "score": float(amap.max())})


def _kcenter(X: np.ndarray, k: int, seed: int = 0) -> np.ndarray:
    """Greedy k-center coreset (PatchCore §3.2): repeatedly add the point farthest from the set."""
    rng = np.random.default_rng(seed)
    sel = [int(rng.integers(len(X)))]
    dmin = np.linalg.norm(X - X[sel[0]], axis=1)
    for _ in range(k - 1):
        j = int(np.argmax(dmin))
        sel.append(j)
        dmin = np.minimum(dmin, np.linalg.norm(X - X[j], axis=1))
    return np.array(sel)


def _nn_dist(Q: np.ndarray, M: np.ndarray) -> np.ndarray:
    from scipy.spatial import cKDTree
    return cKDTree(M).query(Q)[0]


def ex_patchcore(t: Text) -> Example:
    train = normals(N_TRAIN)
    held = normals(2, seed=300)
    img, gt = metal(52, "scratch")
    F = np.stack([patch_features(x) for x in train])
    z = _standardise(F)
    bank = z(F).reshape(-1, F.shape[-1])
    core = bank[_kcenter(bank, len(bank) // 10)]                    # 10 % coreset

    def nn_map(im):
        q = z(patch_features(im))
        return _nn_dist(q.reshape(-1, q.shape[-1]), core).reshape(q.shape[:2])
    grid = nn_map(img)
    amap = up(grid, smooth=4)
    thr = _calibrated(lambda h: up(nn_map(h), smooth=4), held)
    mask = amap > thr
    # 2-D view of the memory bank (PCA of the coreset) with the test patches on top
    mu = bank.mean(0)
    _, _, Vt = np.linalg.svd(bank - mu, full_matrices=False)
    P = lambda a: (a - mu) @ Vt[:2].T
    q = z(patch_features(img)).reshape(-1, bank.shape[-1])
    gt_cells = cv2.resize(gt.astype(np.float32), grid.shape[::-1], interpolation=cv2.INTER_AREA).ravel() > 0.2

    def draw(ax):
        pb, pc, pq = P(bank[::4]), P(core), P(q)
        ax.scatter(pb[:, 0], pb[:, 1], s=2, c="#c5ccd3", label=t("chart.all_patches", "normal patches"))
        ax.scatter(pc[:, 0], pc[:, 1], s=5, c="#2bb3a3", label=t("chart.coreset", "coreset (memory)"))
        ax.scatter(pq[~gt_cells, 0], pq[~gt_cells, 1], s=6, c="#3b6fd6", marker="x", lw=0.8,
                   label=t("chart.test_normal", "test: normal patch"))
        ax.scatter(pq[gt_cells, 0], pq[gt_cells, 1], s=14, c="#e5534b", marker="x", lw=1.2,
                   label=t("chart.test_defect", "test: defect patch"))
        ax.set_xticks([]); ax.set_yticks([])
        ax.legend(fontsize=6, loc="best", framealpha=0.8, markerscale=1.5)
    return Example("patchcore", "toy", [
        Stage("train", montage(train[:4], more=N_TRAIN - 4), "train"),
        Stage("patches", grid_lines(train[0], CELL), "train"),
        Stage("memory", chart(draw, 320, 260)),
        Stage("test", outline(img, gt), "input"), Stage("nn_distance", heat(up(grid))),
        Stage("smoothed", heat(amap)), Stage("mask", binary(mask), "output")],
        _outputs(img, gt, amap, mask),
        {"bank": len(bank), "coreset": len(core), "threshold": thr, "score": float(grid.max())})


def _teacher(g: np.ndarray, kernels: np.ndarray) -> np.ndarray:
    """Frozen 'teacher': fixed random 7×7 filters + ReLU + 4×4 pooling (a non-linear feature map)."""
    out = [np.maximum(cv2.filter2D(g, cv2.CV_32F, k), 0) for k in kernels]
    out = [cv2.resize(o, (g.shape[1] // 4, g.shape[0] // 4), interpolation=cv2.INTER_AREA) for o in out]
    return np.stack(out, -1)


def _student_inputs(g: np.ndarray) -> np.ndarray:
    """'Student' sees 9×9 raw pixel neighbourhoods at the same 4×4 grid (+ bias)."""
    s = cv2.resize(g, (g.shape[1] // 4, g.shape[0] // 4), interpolation=cv2.INTER_AREA)
    p = np.pad(s, 1, mode="reflect")
    cols = [p[i:i + s.shape[0], j:j + s.shape[1]] for i in range(3) for j in range(3)]
    return np.stack(cols + [np.ones_like(s)], -1)


def ex_efficientad(t: Text) -> Example:
    rng = np.random.default_rng(5)
    kernels = rng.normal(0, 1, (8, 7, 7)).astype(np.float32)
    kernels -= kernels.mean(axis=(1, 2), keepdims=True)
    train = normals(N_TRAIN)
    held = normals(2, seed=300)
    img, gt = metal(41, "geometric")
    G = lambda im: gray(im).astype(np.float32) / 255.0
    T = np.concatenate([_teacher(G(x), kernels).reshape(-1, 8) for x in train])
    S = np.concatenate([_student_inputs(G(x)).reshape(-1, 10) for x in train])
    W = np.linalg.solve(S.T @ S + 1e-3 * np.eye(10), S.T @ T)      # student "training" on normal images only

    def discrepancy(im):
        te = _teacher(G(im), kernels)
        st = (_student_inputs(G(im)).reshape(-1, 10) @ W).reshape(te.shape)
        return te, st, ((te - st) ** 2).mean(-1)
    te, st, dmap = discrepancy(img)
    amap = up(dmap, smooth=3)
    thr = _calibrated(lambda h: up(discrepancy(h)[2], smooth=3), held)
    mask = amap > thr
    c = int(np.argmax(te.reshape(-1, 8).var(0)))
    lo, hi = float(te[..., c].min()), float(te[..., c].max())
    return Example("efficientad", "toy", [
        Stage("train", montage(train[:4], more=N_TRAIN - 4), "train"),
        Stage("test", outline(img, gt), "input"),
        Stage("teacher", heat(up(te[..., c]), lo, hi)), Stage("student", heat(up(st[..., c]), lo, hi)),
        Stage("discrepancy", heat(up(dmap))), Stage("mask", binary(mask), "output")],
        _outputs(img, gt, amap, mask), {"threshold": thr, "score": float(dmap.max())})


def ex_winclip(t: Text) -> Example:
    img, gt = metal(74, "scratch")
    g = gray(img).astype(np.float32) / 255.0
    cell = 16
    n = SIZE // cell
    # Stand-in for the CLIP image embedding of a window: a 2-D unit vector whose angle grows with the
    # window's strongest local contrast (no training, no normal images — zero-shot).
    r = g - cv2.GaussianBlur(g, (0, 0), 12)
    mad = float(np.median(np.abs(r)) * 1.4826 + 1e-6)
    t_n, t_a = np.array([0.0, 1.0]), np.array([1.0, 0.0])         # stand-ins for 'normal' / 'damaged' text embeddings
    tau = 1.0

    def window_score(ys, xs):
        a = 2.0 * max(float(np.percentile(np.abs(r[ys, xs]), 98)) / mad - 2.6, 0.0)
        e = np.array([a, 1.0]) / np.hypot(a, 1.0)
        logits = np.array([e @ t_n, e @ t_a]) / tau
        p = np.exp(logits - logits.max())
        return float(p[1] / p.sum())                              # softmax → P(anomalous | window)

    scores, harm_sum, count = {}, np.zeros((SIZE, SIZE)), np.zeros((SIZE, SIZE))
    for k in (2, 3):                                              # window sizes in patches (small, mid)
        sm = np.zeros((SIZE, SIZE))
        cnt = np.zeros((SIZE, SIZE))
        for i in range(n - k + 1):
            for j in range(n - k + 1):
                ys, xs = slice(i * cell, (i + k) * cell), slice(j * cell, (j + k) * cell)
                s = max(window_score(ys, xs), 1e-4)
                sm[ys, xs] += s
                cnt[ys, xs] += 1
                harm_sum[ys, xs] += 1.0 / s
                count[ys, xs] += 1
        scores[k] = sm / np.maximum(cnt, 1)
    amap = (count / np.maximum(harm_sum, 1e-9)).astype(np.float32)   # harmonic mean over all covering windows
    mask = amap > 0.45

    ys, xs = np.nonzero(gt)
    ci, cj = int(ys.mean()) // cell, int(xs.mean()) // cell

    def windows(k):
        out = img.copy()
        on_defect = (min(max(ci - k // 2, 0), n - k), min(max(cj - k // 2, 0), n - k))
        for (i, j), col in zip(((1, 1), on_defect), ((80, 200, 255), (255, 200, 80))):
            cv2.rectangle(out, (j * cell, i * cell), ((j + k) * cell - 1, (i + k) * cell - 1), col, 2)
        return grid_lines(out, cell, (120, 120, 120))
    return Example("winclip", "toy", [
        Stage("test", outline(img, gt), "input"), Stage("windows_small", windows(2)),
        Stage("windows_mid", windows(3)), Stage("scores_small", heat(scores[2], 0.2, 0.75)),
        Stage("scores_mid", heat(scores[3], 0.2, 0.75)), Stage("harmonic", heat(amap, 0.2, 0.75)),
        Stage("mask", binary(mask), "output")],
        _outputs(img, gt, amap, mask, 0.2, 0.75), {"threshold": 0.45, "score": float(amap.max())})


def ex_anomalydino(t: Text) -> Example:
    ref, _ = metal(85)
    img, gt = metal(86, "geometric")
    fr = patch_features(ref)
    sd = fr.reshape(-1, 6).std(0) + 1e-6                          # scaled, NOT centred (like ViT tokens)
    norm = lambda f: (f / sd).reshape(-1, 6)
    unit = lambda a: a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-9)
    R = unit(norm(fr))
    fq = patch_features(img)
    Q = unit(norm(fq))
    dist = (1 - (Q @ R.T).max(1)).reshape(fq.shape[:2])          # cosine distance to nearest reference patch
    k = max(1, int(round(dist.size * 0.01)))
    top = np.sort(dist.ravel())[-k:]
    score = float(top.mean())
    held = [metal(87 + i)[0] for i in range(2)]

    def dmap(im):
        f = patch_features(im)
        return up(1 - (unit(norm(f)) @ R.T).max(1).reshape(f.shape[:2]), smooth=2)
    amap = up(dist, smooth=2)
    thr = _calibrated(dmap, held, q=100.0)            # above every held-out normal patch
    mask = amap > thr
    topm = dist >= top.min()
    hl = overlay_mask(img, up(topm.astype(np.float32)) > 0.5, (255, 200, 40))
    return Example("anomalydino", "toy", [
        Stage("reference", grid_lines(ref, CELL), "train"), Stage("test", outline(img, gt), "input"),
        Stage("cosine", heat(up(dist))), Stage("top1", hl), Stage("mask", binary(mask), "output")],
        _outputs(img, gt, amap, mask), {"score": score, "k_top": k, "threshold": thr})


# ------------------------------------------------------------------------------- 3D (G)
def _surface(rng, n=60, dent: tuple | None = None):
    x, y = np.meshgrid(np.linspace(-20, 20, n), np.linspace(-20, 20, n))
    z = 0.002 * (x ** 2 - y ** 2) + 0.004 * x * y + rng.normal(0, 0.02, x.shape)
    lab = np.zeros_like(z)
    if dent:
        cx, cy, depth = dent
        d = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        prof = np.clip(1 - d / 4.0, 0, 1) ** 2
        z = z - depth * prof
        lab = (prof > 0.05).astype(float)
    return np.c_[x.ravel(), y.ravel(), z.ravel()], lab.ravel()


def _rigid(P, deg, tx, ty, tz=0.0):
    a = np.deg2rad(deg)
    R = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    return P @ R.T + np.array([tx, ty, tz])


def _kabsch(A, B):
    ca, cb = A.mean(0), B.mean(0)
    U, _, Vt = np.linalg.svd((A - ca).T @ (B - cb))
    D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ D @ U.T
    return R, cb - R @ ca


def icp(src, ref, iters=40):
    """Point-to-point ICP (Besl & McKay 1992) — NumPy/SciPy re-enactment for the explanation."""
    from scipy.spatial import cKDTree
    tree = cKDTree(ref)
    P = src.copy()
    for _ in range(iters):
        _, j = tree.query(P)
        R, tvec = _kabsch(P, ref[j])
        P = P @ R.T + tvec
    return P


def scatter_top(sets: list[tuple[np.ndarray, object]], lim=27, w=SIZE, h=SIZE, values=None, vlim=None) -> np.ndarray:
    """Top view (x, y) of point sets. ``values`` colours the LAST set by a scalar (heat)."""
    out = np.full((h, w, 3), 30, np.uint8)
    to_px = lambda P: np.c_[(P[:, 0] + lim) / (2 * lim) * (w - 1), (lim - P[:, 1]) / (2 * lim) * (h - 1)].astype(int)
    for k, (P, col) in enumerate(sets):
        px = to_px(P)
        ok = (px[:, 0] >= 0) & (px[:, 0] < w) & (px[:, 1] >= 0) & (px[:, 1] < h)
        if values is not None and k == len(sets) - 1:
            lo, hi = vlim if vlim else (float(values.min()), float(values.max()))
            cols = heat(values.reshape(-1, 1), lo, hi).reshape(-1, 3)
        else:
            cols = np.tile(np.array(col, np.uint8), (len(P), 1))
        for (u, v), c in zip(px[ok], cols[ok]):
            out[max(0, v - 1): v + 1, max(0, u - 1): u + 1] = c
    return out


def point_to_surface(P: np.ndarray, ref: np.ndarray, k: int = 12) -> np.ndarray:
    """|n·(p − q)|: distance to the reference tangent plane (normals by local PCA of k neighbours)."""
    from scipy.spatial import cKDTree
    tree = cKDTree(ref)
    _, nb = tree.query(ref, k)
    C = ref[nb] - ref[nb].mean(1, keepdims=True)
    normals = np.linalg.eigh(np.einsum("nki,nkj->nij", C, C))[1][:, :, 0]
    _, j = tree.query(P)
    return np.abs(np.einsum("ij,ij->i", P - ref[j], normals[j]))


def ex_pc_deviation(t: Text) -> Example:
    rng = np.random.default_rng(9)
    ref, _ = _surface(rng)
    test0, lab = _surface(rng, dent=(6.0, -5.0, 0.9))
    test = _rigid(test0, 12.0, 4.0, -3.0, 0.5)               # the part lies rotated / shifted on the sensor
    coarse = test - test.mean(0) + ref.mean(0)                # centroid pre-alignment
    fine = icp(coarse, ref)
    d_before = point_to_surface(coarse, ref)
    d = point_to_surface(fine, ref)
    thr = 0.25
    REF, TST = (150, 150, 150), (255, 170, 60)
    zc = scatter_top([(ref, None)], values=ref[:, 2])
    return Example("pc_deviation", "toy", [
        Stage("reference", zc, "train"), Stage("test", scatter_top([(ref, REF), (test, TST)]), "input"),
        Stage("coarse", scatter_top([(ref, REF), (coarse, TST)])),
        Stage("icp", scatter_top([(ref, REF), (fine, TST)])),
        Stage("deviation_before", scatter_top([(coarse, None)], values=d_before, vlim=(0, 0.8))),
        Stage("deviation", scatter_top([(fine, None)], values=d, vlim=(0, 0.8))),
        Stage("mask", scatter_top([(fine[d <= thr], (110, 110, 110)), (fine[d > thr], (255, 60, 60))]), "output")],
        [Stage("out_gt", scatter_top([(fine[lab == 0], (110, 110, 110)), (fine[lab > 0], (60, 220, 110))]), "input"),
         Stage("out_map", scatter_top([(fine, None)], values=d, vlim=(0, 0.8)), "output"),
         Stage("out_mask", scatter_top([(fine[d <= thr], (110, 110, 110)), (fine[d > thr], (255, 60, 60))]), "output")],
        {"threshold": thr, "score": float(np.percentile(d, 99)), "before": float(np.percentile(d_before, 99)),
         "rmse": float(np.sqrt(np.mean(d[lab == 0] ** 2)))})


def ex_multimodal_fusion(t: Text) -> Example:
    rng = np.random.default_rng(13)
    n = 48

    def height(dent):
        x, y = np.meshgrid(np.linspace(-20, 20, n), np.linspace(-20, 20, n))
        z = 0.002 * (x ** 2 - y ** 2) + rng.normal(0, 0.02, x.shape)
        if dent:
            d = np.sqrt((x - 5) ** 2 + (y + 4) ** 2)
            z = z - 0.25 * np.clip(1 - d / 4.0, 0, 1) ** 2
        return z

    z_ref = 0.002 * (np.meshgrid(np.linspace(-20, 20, n), np.linspace(-20, 20, n))[0] ** 2
                     - np.meshgrid(np.linspace(-20, 20, n), np.linspace(-20, 20, n))[1] ** 2)

    def rgb_score(img):
        return float(_real("morphology", img).score)          # REAL classical detector on the RGB view

    def geo_score(z):
        return float(np.percentile(np.abs(z - z_ref), 99))    # 3D deviation to the reference surface
    held = [(_u8(polished(400 + i)), height(False)) for i in range(10)]   # held-out NORMAL samples
    R = np.array([rgb_score(a) for a, _ in held])
    G = np.array([geo_score(b) for _, b in held])
    mr, sr, mg, sg = R.mean(), R.std() + 1e-6, G.mean(), G.std() + 1e-6
    cases = [("good", False, False), ("rgb", True, False), ("geo", False, True), ("both", True, True)]
    rows, rgbs, deps = [], [], []
    for k, (name, a, b) in enumerate(cases):
        img = _pits_image(500 + k, depth=0.12)[0] if a else _u8(polished(500 + k))
        z = height(b)
        zr, zg = (rgb_score(img) - mr) / sr, (geo_score(z) - mg) / sg
        rows.append((name, zr, zg, (zr + zg) / 2, max(zr, zg)))
        rgbs.append(img)
        deps.append(heat(z - z_ref, -0.3, 0.3))     # height minus nominal shape
    thr = 3.0
    labels = [t("chart.case_" + r[0], r[0]) for r in rows]

    def draw(ax):
        x = np.arange(len(rows))
        w = 0.2
        for i, (key, col) in enumerate((("chart.z_rgb", "#3b6fd6"), ("chart.z_3d", "#b56d00"),
                                        ("chart.fused_mean", "#7f8c99"), ("chart.fused_max", "#2bb3a3"))):
            ax.bar(x + (i - 1.5) * w, [r[i + 1] for r in rows], w, color=col,
                   label=t(key, {"chart.z_rgb": "RGB z", "chart.z_3d": "3D z", "chart.fused_mean": "mean fusion",
                                 "chart.fused_max": "max fusion"}[key]))
        ax.axhline(thr, color="#e5534b", lw=1.2, ls="--")
        ax.text(len(rows) - 0.5, thr, f"z = {thr:g}", color="#e5534b", fontsize=7, ha="right", va="bottom")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=7)
        ax.set_ylabel(t("chart.zscore", "standardised score z"), fontsize=8)
        ax.legend(fontsize=6, ncol=4, loc="lower left", bbox_to_anchor=(0, 1.0), frameon=False)

    def draw_norm(ax):
        ax.hist(R, bins=6, color="#3b6fd6", alpha=0.8)
        ax.axvline(mr, color="k", lw=1)
        ax.set_xlabel(t("chart.raw_rgb", "raw RGB score (held-out normals)"), fontsize=8)
        ax.set_yticks([])
    fused = chart(draw, 420, 260)
    return Example("multimodal_fusion", "toy", [
        Stage("rgb_views", montage(rgbs, cols=2), "input"), Stage("depth_views", montage(deps, cols=2), "input"),
        Stage("normalise", chart(draw_norm, 300, 200)), Stage("fusion", fused, "output")],
        [Stage("out_fusion", fused, "output")],
        {"threshold": thr, "rows": rows})


BUILDERS: dict[str, Callable[[Text], Example]] = {
    "reference_difference": ex_reference_difference, "otsu": ex_otsu, "morphology": ex_morphology,
    "canny": ex_canny, "gabor": ex_gabor, "adaptive_threshold": ex_adaptive_threshold,
    "padim": ex_padim, "patchcore": ex_patchcore, "efficientad": ex_efficientad,
    "winclip": ex_winclip, "anomalydino": ex_anomalydino,
    "pc_deviation": ex_pc_deviation, "multimodal_fusion": ex_multimodal_fusion,
}

_cache: dict[tuple, Example] = {}
_cmp_cache: dict[tuple, dict] = {}


def _builders() -> dict[str, Callable[[Text], Example]]:
    """Group B (supervised) lives in its own module, which imports this one: register it lazily."""
    from core.visualization import supervised_examples as sup
    return {**BUILDERS, **sup.BUILDERS}


def available(method_id: str) -> bool:
    return method_id in _builders()


def build(method_id: str, t: Text = _t_default, lang: str = "en") -> Example | None:
    """Example for ``method_id`` (cached per language, because chart labels are translated)."""
    builders = _builders()
    if method_id not in builders:
        return None
    key = (method_id, lang)
    if key not in _cache:
        _cache[key] = builders[method_id](t)
    return _cache[key]


def comparison_images(key: str, t: Text = _t_default, lang: str = "en") -> dict:
    """Pictures of a cross-method comparison (``supervised``: the label each method needs, its output)."""
    if (key, lang) not in _cmp_cache:
        if key != "supervised":
            raise KeyError(key)
        from core.visualization import supervised_examples as sup
        _cmp_cache[(key, lang)] = sup.comparison(t)
    return _cmp_cache[(key, lang)]
