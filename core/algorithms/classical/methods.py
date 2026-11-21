"""Group A — classical computer-vision baselines (runnable, CPU).

Implementations use OpenCV / scikit-image.  OpenCV and scikit-image are *implementation
libraries*; the original research for each operator is referenced in the research catalog.
"""

from __future__ import annotations

import cv2
import numpy as np

from core.algorithms.base import AnomalyMethod, MethodResult, ParamSpec


def _gray(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if img.ndim == 3 else img


def _ver(lib: str) -> str:
    if lib == "opencv":
        return f"OpenCV {cv2.__version__}"
    import skimage
    return f"scikit-image {skimage.__version__}"


# --------------------------------------------------------------------------- Otsu
class OtsuThreshold(AnomalyMethod):
    method_id = "otsu"
    display_name = "Otsu thresholding"

    @classmethod
    def param_specs(cls):
        return [
            ParamSpec("blur", "int", 5, "Gaussian blur (px, odd)", 1, 31, 2),
            ParamSpec("polarity", "choice", "dark", "Defect polarity", choices=["dark", "bright"],
                      help="Whether defects are expected darker or brighter than the background."),
        ]

    def model_version(self):
        return _ver("opencv")

    def _predict(self, image, reference):
        g = _gray(image)
        k = int(self.params["blur"]) | 1
        g = cv2.GaussianBlur(g, (k, k), 0) if k > 1 else g
        thr, _ = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        dark = self.params["polarity"] == "dark"
        amap = (255.0 - g) if dark else g.astype(np.float32)
        t = 255.0 - thr if dark else float(thr)
        mask = amap > t
        hist = np.bincount(g.ravel(), minlength=256)
        return MethodResult(self.method_id, amap, float(mask.mean()), mask, t,
                            score_semantics="map: (inverted) grey level; score: fraction of pixels on "
                                            "the 'defect' side of the Otsu threshold",
                            extras={"otsu_threshold_grey": float(thr), "histogram": hist.tolist()})


# ---------------------------------------------------------------- adaptive threshold
class AdaptiveThreshold(AnomalyMethod):
    method_id = "adaptive_threshold"
    display_name = "Adaptive thresholding"

    @classmethod
    def param_specs(cls):
        return [
            ParamSpec("method", "choice", "gaussian", "Local statistic", choices=["mean", "gaussian"]),
            ParamSpec("block_size", "int", 51, "Neighbourhood size (px, odd)", 3, 301, 2),
            ParamSpec("C", "float", 12.0, "Offset C (grey levels)", -50, 50, 1),
            ParamSpec("polarity", "choice", "dark", "Defect polarity", choices=["dark", "bright"]),
        ]

    def model_version(self):
        return _ver("opencv")

    def _predict(self, image, reference):
        g = _gray(image)
        b = max(3, int(self.params["block_size"]) | 1)
        C = float(self.params["C"])
        if self.params["method"] == "gaussian":
            local = cv2.GaussianBlur(g.astype(np.float32), (b, b), 0)
        else:
            local = cv2.blur(g.astype(np.float32), (b, b))
        diff = (local - g) if self.params["polarity"] == "dark" else (g - local)
        mask = diff > C            # equivalent to cv2.adaptiveThreshold(..., THRESH_BINARY_INV)
        return MethodResult(self.method_id, diff.astype(np.float32), float(mask.mean()), mask, C,
                            score_semantics="map: deviation from the local mean (grey levels); score: "
                                            "fraction of flagged pixels")


# ---------------------------------------------------------------------------- Canny
class CannyEdges(AnomalyMethod):
    method_id = "canny"
    display_name = "Canny edge detection"

    @classmethod
    def param_specs(cls):
        return [
            ParamSpec("low", "int", 40, "Low hysteresis threshold", 0, 500, 1),
            ParamSpec("high", "int", 120, "High hysteresis threshold", 0, 800, 1),
            ParamSpec("blur", "int", 5, "Pre-blur (px, odd)", 1, 31, 2),
            ParamSpec("density_window", "int", 15, "Edge-density window (px)", 3, 101, 2),
        ]

    def model_version(self):
        return _ver("opencv")

    def _predict(self, image, reference):
        g = _gray(image)
        k = int(self.params["blur"]) | 1
        g = cv2.GaussianBlur(g, (k, k), 0) if k > 1 else g
        edges = cv2.Canny(g, int(self.params["low"]), int(self.params["high"]))
        w = int(self.params["density_window"]) | 1
        density = cv2.blur((edges > 0).astype(np.float32), (w, w))
        vis = image.copy()
        vis[edges > 0] = (255, 40, 40)
        thr = 0.15
        return MethodResult(self.method_id, density, float((edges > 0).mean()), density > thr, thr,
                            visual=vis,
                            score_semantics="map: local edge density (0..1); score: global edge-pixel "
                                            "fraction. Edges are not defects per se: texture also "
                                            "produces edges.")


# ------------------------------------------------------------------------- Gabor
class GaborTexture(AnomalyMethod):
    method_id = "gabor"
    display_name = "Gabor texture analysis"

    @classmethod
    def param_specs(cls):
        return [
            ParamSpec("n_orientations", "int", 6, "Orientations", 2, 12, 1),
            ParamSpec("wavelengths", "choice", "4,8,16", "Wavelengths λ (px)",
                      choices=["4,8", "4,8,16", "8,16,32"]),
            ParamSpec("sigma_ratio", "float", 0.56, "σ/λ", 0.2, 1.5, 0.02),
            ParamSpec("energy_window", "int", 15, "Energy smoothing window (px)", 3, 61, 2),
        ]

    def model_version(self):
        return _ver("opencv")

    def _bank(self):
        kernels = []
        lams = [float(v) for v in str(self.params["wavelengths"]).split(",")]
        n = int(self.params["n_orientations"])
        for lam in lams:
            sigma = self.params["sigma_ratio"] * lam
            ks = int(6 * sigma) | 1
            for i in range(n):
                theta = np.pi * i / n
                k = cv2.getGaborKernel((ks, ks), sigma, theta, lam, 0.5, 0, ktype=cv2.CV_32F)
                k -= k.mean()
                kernels.append(k / (np.abs(k).sum() + 1e-8))
        return kernels

    def _predict(self, image, reference):
        g = _gray(image).astype(np.float32) / 255.0
        w = int(self.params["energy_window"]) | 1
        feats = []
        for k in self._bank():
            r = cv2.filter2D(g, cv2.CV_32F, k)
            feats.append(cv2.blur(r * r, (w, w)))
        F = np.log(np.stack(feats, -1) + 1e-8)
        med = np.median(F.reshape(-1, F.shape[-1]), 0)
        mad = np.median(np.abs(F.reshape(-1, F.shape[-1]) - med), 0) * 1.4826 + 1e-6
        z = np.abs((F - med) / mad)                   # robust z-score per filter response
        amap = z.max(-1).astype(np.float32)
        thr = 4.0
        return MethodResult(self.method_id, amap, float(np.percentile(amap, 99.5)), amap > thr, thr,
                            score_semantics="map: max robust z-score of Gabor energies w.r.t. the "
                                            "image's own texture statistics; score: 99.5th percentile",
                            extras={"n_filters": len(feats)})


# -------------------------------------------------------------------- morphology
class MorphologyTopHat(AnomalyMethod):
    method_id = "morphology"
    display_name = "Mathematical morphology (top-hat)"

    @classmethod
    def param_specs(cls):
        return [
            ParamSpec("operation", "choice", "black_hat", "Operation",
                      choices=["black_hat", "white_hat", "both"],
                      help="Black-hat highlights small dark structures, white-hat small bright ones."),
            ParamSpec("se_size", "int", 15, "Structuring element size (px)", 3, 101, 2),
            ParamSpec("se_shape", "choice", "ellipse", "Structuring element", choices=["ellipse", "rect", "cross"]),
            ParamSpec("threshold", "float", 25.0, "Response threshold (grey levels)", 1, 255, 1),
            ParamSpec("clean_size", "int", 3, "Mask opening (px, 1=off)", 1, 15, 2),
        ]

    def model_version(self):
        return _ver("opencv")

    def _predict(self, image, reference):
        g = _gray(image)
        s = int(self.params["se_size"]) | 1
        shape = {"ellipse": cv2.MORPH_ELLIPSE, "rect": cv2.MORPH_RECT, "cross": cv2.MORPH_CROSS}[self.params["se_shape"]]
        se = cv2.getStructuringElement(shape, (s, s))
        op = self.params["operation"]
        bh = cv2.morphologyEx(g, cv2.MORPH_BLACKHAT, se).astype(np.float32)
        wh = cv2.morphologyEx(g, cv2.MORPH_TOPHAT, se).astype(np.float32)
        amap = bh if op == "black_hat" else wh if op == "white_hat" else np.maximum(bh, wh)
        thr = float(self.params["threshold"])
        mask = (amap > thr).astype(np.uint8)
        c = int(self.params["clean_size"])
        if c > 1:
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (c, c)))
        return MethodResult(self.method_id, amap, float(amap.max()), mask > 0, thr,
                            score_semantics="map: top-hat response (grey levels); score: max response")


# ------------------------------------------------------------ reference alignment
def align_to_reference(image_g: np.ndarray, ref_g: np.ndarray, motion: str = "euclidean", max_side: int = 512):
    """ECC alignment (Evangelidis & Psarakis 2008) of ``image`` onto ``ref``.

    The warp is estimated on copies down-scaled to ``max_side`` (speed) and the translation is
    rescaled to full resolution (rotation/shear are scale-invariant).
    """
    mode = {"translation": cv2.MOTION_TRANSLATION, "euclidean": cv2.MOTION_EUCLIDEAN,
            "affine": cv2.MOTION_AFFINE}[motion]
    warp = np.eye(2, 3, dtype=np.float32)
    s = min(1.0, max_side / max(ref_g.shape))
    small = (lambda a: cv2.resize(a, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)) if s < 1 else (lambda a: a)
    try:
        crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-5)
        cc, warp = cv2.findTransformECC(small(ref_g).astype(np.float32), small(image_g).astype(np.float32),
                                        warp, mode, crit, None, 5)
        warp[:, 2] /= s
    except cv2.error:
        return image_g, np.eye(2, 3, dtype=np.float32), False
    h, w = ref_g.shape
    out = cv2.warpAffine(image_g, warp, (w, h), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
                         borderMode=cv2.BORDER_REFLECT)
    return out, warp, True


def _prep_pair(image, reference, align: str, blur: int, stages: dict | None = None):
    """Grey → (resize reference to the test size) → ECC-warp the TEST onto the reference → pre-blur.

    ``stages`` (optional) receives the intermediate arrays exactly as used, so the GUI can show the
    real pipeline (no recomputation): ``reference_gray``, ``test_gray``, ``test_aligned``, ``warp``,
    ``reference_pre``, ``test_pre`` (the two arrays that are subtracted / compared).
    """
    g, r = _gray(image), _gray(reference)
    if r.shape != g.shape:
        r = cv2.resize(r, (g.shape[1], g.shape[0]), interpolation=cv2.INTER_AREA)
    if stages is not None:
        stages.update(reference_gray=r, test_gray=g, reference_resized=r.shape != _gray(reference).shape)
    aligned, warp = False, None
    if align != "none":
        g, warp, aligned = align_to_reference(g, r, align)
    if stages is not None:
        stages.update(test_aligned=g, warp=None if warp is None else np.asarray(warp).copy())
    k = int(blur) | 1
    if k > 1:
        g, r = cv2.GaussianBlur(g, (k, k), 0), cv2.GaussianBlur(r, (k, k), 0)
    if stages is not None:
        stages.update(reference_pre=r, test_pre=g, blur_kernel=k)
    return g, r, aligned


class ReferenceDifference(AnomalyMethod):
    method_id = "reference_difference"
    display_name = "Reference image difference"
    requires_reference = True

    @classmethod
    def param_specs(cls):
        return [
            ParamSpec("align", "choice", "euclidean", "Alignment (ECC)", choices=["none", "translation", "euclidean", "affine"]),
            ParamSpec("blur", "int", 5, "Pre-blur (px, odd)", 1, 31, 2),
            ParamSpec("threshold", "float", 30.0, "Difference threshold (grey levels)", 1, 255, 1),
        ]

    def model_version(self):
        return _ver("opencv")

    def _predict(self, image, reference):
        stages: dict = {}
        g, r, aligned = _prep_pair(image, reference, self.params["align"], self.params["blur"], stages)
        amap = cv2.absdiff(g, r).astype(np.float32)
        thr = float(self.params["threshold"])
        return MethodResult(self.method_id, amap, float(np.percentile(amap, 99.9)), amap > thr, thr,
                            score_semantics="map: |test − reference| (grey levels); score: 99.9th percentile",
                            extras={"aligned": aligned, "stages": stages})


class SSIMComparison(AnomalyMethod):
    method_id = "ssim"
    display_name = "SSIM-based comparison"
    requires_reference = True

    @classmethod
    def param_specs(cls):
        return [
            ParamSpec("align", "choice", "euclidean", "Alignment (ECC)", choices=["none", "translation", "euclidean", "affine"]),
            ParamSpec("win_size", "int", 11, "SSIM window (px, odd)", 3, 51, 2),
            ParamSpec("blur", "int", 1, "Pre-blur (px, odd)", 1, 31, 2),
            ParamSpec("threshold", "float", 0.5, "Dissimilarity threshold (1−SSIM)", 0.0, 2.0, 0.01),
        ]

    def model_version(self):
        return _ver("skimage")

    def _predict(self, image, reference):
        from skimage.metrics import structural_similarity
        g, r, aligned = _prep_pair(image, reference, self.params["align"], self.params["blur"])
        w = int(self.params["win_size"]) | 1
        mssim, smap = structural_similarity(r, g, win_size=w, data_range=255, full=True,
                                            gaussian_weights=True, sigma=1.5, use_sample_covariance=False) \
            if w == 11 else structural_similarity(r, g, win_size=w, data_range=255, full=True)
        amap = (1.0 - smap).astype(np.float32)
        thr = float(self.params["threshold"])
        return MethodResult(self.method_id, amap, float(1 - mssim), amap > thr, thr,
                            score_semantics="map: 1 − local SSIM; score: 1 − mean SSIM",
                            extras={"aligned": aligned, "mean_ssim": float(mssim)})


CLASSICAL_METHODS = [OtsuThreshold, AdaptiveThreshold, CannyEdges, GaborTexture, MorphologyTopHat,
                     ReferenceDifference, SSIMComparison]
