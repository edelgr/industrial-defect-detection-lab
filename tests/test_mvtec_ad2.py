"""MVTec AD 2 support: lighting-condition metadata, localisation metrics (AU-PRO, pixel F1),
aspect-preserving input, cache-key stability and — when the real data are installed — index
integrity and reloading of fitted models."""

import cv2
import numpy as np
import pytest

from core.datasets.base import BAD, GOOD, UNKNOWN
from core.datasets.io import write_image
from core.datasets.mvtec import MVTecAD2Adapter
from core.evaluation import metrics as M
from tests.conftest import cuda_available


# ---------------------------------------------------------------- adapter metadata (fixture)
def test_ad2_lighting_condition_metadata(tmp_path):
    c = tmp_path / "fruit_jelly"
    img = np.full((8, 8, 3), 100, np.uint8)
    for rel in ("train/good/000_regular.png", "test_public/good/000_overexposed.png",
                "test_public/bad/003_shift_2.png", "test_private_mixed/007_mixed.png"):
        write_image(c / rel, img)
    write_image(c / "test_public/ground_truth/bad/003_shift_2_mask.png", np.full((8, 8), 255, np.uint8))
    idx = MVTecAD2Adapter().index(tmp_path, "fruit_jelly")
    by = {s.sample_id: s for s in idx.samples}
    assert by["test_public/bad/003_shift_2"].meta["lighting_condition"] == "shift_2"
    assert by["test_public/bad/003_shift_2"].meta["scene_id"] == "003"
    assert by["test_public/good/000_overexposed"].meta["lighting_condition"] == "overexposed"
    assert by["test_private_mixed/007_mixed"].label == UNKNOWN
    assert MVTecAD2Adapter.eval_pixel_max_side >= 1024


# ---------------------------------------------------------------- AU-PRO and pixel F1
def _brute_au_pro(masks, maps, lim):
    """Direct definition: for each distinct threshold, FPR on normal pixels and mean per-region overlap."""
    s_all = np.concatenate([a.ravel() for a in maps])
    regions = []
    for m, a in zip(masks, maps):
        k, lab = cv2.connectedComponents(m.astype(np.uint8), connectivity=8)
        regions += [a[lab == r] for r in range(1, k)]
    neg = np.concatenate([a[~m] for m, a in zip(masks, maps)])
    pts = [(0.0, 0.0)] + [((neg >= t).mean(), np.mean([(r >= t).mean() for r in regions]))
                          for t in np.unique(s_all)[::-1]]
    f, p = np.array([x for x, _ in pts]), np.array([y for _, y in pts])
    j = np.searchsorted(f, lim, side="right")
    if j < len(f):
        pl = p[j - 1] + (p[j] - p[j - 1]) * (lim - f[j - 1]) / (f[j] - f[j - 1]) if f[j] > f[j - 1] else p[j - 1]
        f, p = np.r_[f[:j], lim], np.r_[p[:j], pl]
    return np.trapezoid(p, f) / lim


def test_au_pro_matches_direct_definition():
    rng = np.random.default_rng(0)
    for _ in range(4):
        masks, maps = [], []
        for i in range(3):
            m = np.zeros((20, 30), bool)
            if i:
                y, x = rng.integers(0, 15), rng.integers(0, 24)
                m[y:y + 4, x:x + 5] = True
            if i == 2:
                m[2:4, 2:6] = True                     # a second region in the same image
            a = np.round(rng.random((20, 30)) + m * rng.random() * 1.5, 2).astype(np.float32)   # with ties
            masks.append(m)
            maps.append(a)
        for lim in (0.3, 0.05, 1.0):
            assert M.au_pro(masks, maps, lim) == pytest.approx(_brute_au_pro(masks, maps, lim), abs=1e-9)


def test_au_pro_extremes():
    m = np.zeros((10, 10), bool)
    m[2:4, 2:4] = True
    assert M.au_pro([m], [m.astype(np.float32)], 0.3) == pytest.approx(1.0)
    assert M.au_pro([np.zeros((5, 5), bool)], [np.zeros((5, 5), np.float32)]) is None


def test_pixel_f1_pooled():
    g = np.zeros((4, 4), bool)
    g[0, :2] = True
    p = np.zeros((4, 4), bool)
    p[0, 1:3] = True
    r = M.pixel_f1([g], [p])
    assert r["precision"] == pytest.approx(0.5) and r["recall"] == pytest.approx(0.5) and r["f1"] == pytest.approx(0.5)


# ---------------------------------------------------------------- input shape and cache keys
def test_keep_aspect_input_shape():
    from core.algorithms.memory_based.anomalib_models import PatchcoreMethod
    assert PatchcoreMethod()._input_hw(np.zeros((1056, 4224))) == (256, 256)
    m = PatchcoreMethod(keep_aspect=True)
    assert m._input_hw(np.zeros((1056, 4224))) == (256, 1024)        # sheet_metal
    assert m._input_hw(np.zeros((1520, 2100, 3))) == (256, 352)      # fruit_jelly
    assert m._input_hw(np.zeros((2000, 500, 3))) == (1024, 256)      # portrait, capped at 4x


def test_new_default_parameter_does_not_change_cache_key(synthetic_index, deep):
    from core.inference.session import LabSession
    keys = {}
    for name, params in {"implicit": {}, "explicit": {"keep_aspect": False}, "on": {"keep_aspect": True}}.items():
        s = LabSession("patchcore", params)
        s.bind(synthetic_index)
        keys[name] = s.cache_path(s._split_train(5)[0]).name
    assert keys["implicit"] == keys["explicit"] != keys["on"]


# ---------------------------------------------------------------- real data (if installed)
EXPECTED = {   # counts of the official archives (verified against the installed files)
    "sheet_metal": {"train": {"good": 137}, "validation": {"good": 19}, "test_public": {"good": 24, "bad": 90},
                    "test_private": {"unknown": 142}, "test_private_mixed": {"unknown": 142}},
    "fruit_jelly": {"train": {"good": 263}, "validation": {"good": 37}, "test_public": {"good": 20, "bad": 60},
                    "test_private": {"unknown": 255}, "test_private_mixed": {"unknown": 255}},
}


def _real_index(cat):
    from core import settings
    from core.datasets import registry as dreg
    settings.load()
    if cat not in dreg.installed_categories("D05"):
        pytest.skip(f"MVTec AD 2 {cat} not installed")
    return dreg.load_index("D05", cat, refresh=True)


@pytest.mark.parametrize("cat", sorted(EXPECTED))
def test_real_ad2_index_integrity(cat):
    idx = _real_index(cat)
    assert idx.summary() == EXPECTED[cat]
    assert not idx.warnings
    assert len({s.sample_id for s in idx.samples}) == len(idx.samples)
    for s in idx.samples:
        if s.label == BAD:
            (mask,) = s.masks["rgb"]
            assert mask.name == f"{s.primary_path.stem}_mask.png" and mask.is_file()
            assert s.split == "test_public" and s.meta.get("lighting_condition")
        else:
            assert not s.masks
    assert {s.split for s in idx.labelled_test()} == {"test_public"}
    assert all(s.label == GOOD for s in idx.train_normals())


@pytest.mark.real_checkpoints
@pytest.mark.skipif(not cuda_available(), reason="CUDA not available")
@pytest.mark.parametrize("cat", sorted(EXPECTED))
def test_real_ad2_fitted_models_reload(cat):
    """The campaign's fitted models (all train/good images) reload from checkpoints/fitted and predict."""
    from core.inference.session import LabSession
    from core.datasets.io import read_rgb8
    idx = _real_index(cat)
    test = next(s for s in idx.samples if s.split == "test_public" and s.label == BAD)
    img = read_rgb8(test.primary_path)
    found = 0
    for mid in ("patchcore", "padim", "efficientad", "anomalydino"):
        s = LabSession(mid)
        s.bind(idx)
        if s.cached_fit(1000) is None:
            continue
        found += 1
        info = s.fit(max_images=1000)
        assert info.get("cache") and s.method.fit_info.get("loaded_from_cache")
        assert len(s.fit_ids) == EXPECTED[cat]["train"]["good"]
        assert not set(s.fit_ids) & {x.sample_id for x in idx.samples if x.split != "train"}
        r = s.predict_image(img)
        assert r.anomaly_map.shape == img.shape[:2] and np.isfinite(r.score)
    if not found:
        pytest.skip("no fitted D05 models cached yet (run scripts.run_experiments --dataset D05)")
