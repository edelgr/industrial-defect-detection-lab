"""Inference output contracts for classical, deep and 3D methods."""

import numpy as np
import pytest

from core.algorithms import registry
from core.algorithms.base import MethodResult
from core.algorithms.classical.methods import CLASSICAL_METHODS
from tests.conftest import cuda_available


def _check(res: MethodResult, shape):
    assert isinstance(res, MethodResult)
    assert res.anomaly_map is not None and res.anomaly_map.shape == shape
    assert res.anomaly_map.dtype == np.float32
    assert np.isfinite(res.anomaly_map).all()
    assert res.score is None or np.isfinite(res.score)
    assert res.latency_ms > 0
    assert res.score_semantics, "every method must document what its scores mean"


@pytest.mark.parametrize("cls", CLASSICAL_METHODS, ids=lambda c: c.method_id)
def test_classical_contract(cls, rgb_image):
    m = cls()
    ref = rgb_image.copy()
    ref[40:50, 60:90] = ref[30:40, 60:90]
    res = m.predict(rgb_image, ref if m.requires_reference else None)
    _check(res, rgb_image.shape[:2])
    if res.mask is not None:
        assert res.mask.shape == rgb_image.shape[:2] and res.mask.dtype == bool
    # re-thresholding uses cached maps
    r2 = res.with_threshold(float(np.median(res.anomaly_map)))
    assert r2.mask.shape == rgb_image.shape[:2]


def test_ecc_alignment_recovers_translation():
    import cv2
    from core.algorithms.classical.methods import align_to_reference
    rng = np.random.default_rng(0)
    ref = cv2.normalize(cv2.GaussianBlur(rng.random((480, 640)).astype(np.float32), (0, 0), 5),
                        None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    img = cv2.warpAffine(ref, np.float32([[1, 0, 9], [0, 1, -5]]), (640, 480), borderMode=cv2.BORDER_REFLECT)
    out, warp, ok = align_to_reference(img, ref, "translation", max_side=256)
    assert ok and abs(warp[0, 2] - 9) < 0.5 and abs(warp[1, 2] + 5) < 0.5
    assert np.abs(out.astype(int) - ref)[40:-40, 40:-40].mean() < 2


def test_reference_methods_require_reference(rgb_image):
    for mid in ("reference_difference", "ssim"):
        with pytest.raises(ValueError):
            registry.create(mid).predict(rgb_image)


def test_dark_defect_scores_higher_than_background(rgb_image):
    res = registry.create("morphology", operation="black_hat", se_size=15).predict(rgb_image)
    assert res.anomaly_map[45, 75] > np.median(res.anomaly_map)


def test_fit_required_before_predict(rgb_image):
    ok, _ = registry.availability("padim")
    if not ok:
        pytest.skip("anomalib unavailable")
    with pytest.raises(RuntimeError):
        registry.create("padim").predict(rgb_image)


@pytest.mark.parametrize("mid,params", [("padim", {}), ("patchcore", {"backbone": "resnet18"}),
                                        ("anomalydino", {"k_shot": 2, "image_size": 224})])
def test_deep_methods_contract_on_synthetic(mid, params, synthetic_index):
    ok, why = registry.availability(mid)
    if not ok:
        pytest.skip(why)
    from core.datasets.io import read_rgb8
    m = registry.create(mid, **params)
    m.fit([read_rgb8(s.primary_path) for s in synthetic_index.train_normals()[:6]])
    img = read_rgb8(next(s for s in synthetic_index.samples if s.label == "bad").primary_path)
    res = m.predict(img)
    _check(res, img.shape[:2])
    assert res.device == ("cuda" if cuda_available() else "cpu")
    if cuda_available():
        assert res.gpu_memory_mb is not None and res.gpu_memory_mb > 0


def test_registration_recovers_rigid_motion():
    ok, why = registry.availability("pc_deviation")
    if not ok:
        pytest.skip(why)
    rng = np.random.default_rng(0)
    # an asymmetric 3D object: two boxes of different size
    a = rng.uniform([0, 0, 0], [40, 20, 10], (6000, 3))
    b = rng.uniform([30, 15, 10], [40, 20, 25], (2000, 3))
    ref = np.vstack([a, b])
    ang = np.deg2rad(25)
    R = np.array([[np.cos(ang), -np.sin(ang), 0], [np.sin(ang), np.cos(ang), 0], [0, 0, 1]])
    test = ref @ R.T + np.array([150.0, -40.0, 12.0])
    m = registry.create("pc_deviation")
    m.fit([ref])
    res = m.predict(test)
    assert res.steps["registration_ok"] and res.fitness > 0.9
    assert np.percentile(res.deviation, 95) < 2 * res.steps["voxel"]


def test_pointcloud_deviation_detects_bump(synthetic_root):
    ok, why = registry.availability("pc_deviation")
    if not ok:
        pytest.skip(why)
    from core.datasets.io import read_point_cloud
    from core.datasets.synthetic import SyntheticAdapter
    idx = SyntheticAdapter().index(synthetic_root, "surface_patch_3d")
    m = registry.create("pc_deviation", coarse=False)
    m.fit([read_point_cloud(s.modalities["pointcloud"])[0] for s in idx.train_normals()])
    good = next(s for s in idx.samples if s.split == "test" and s.label == "good")
    bad = next(s for s in idx.samples if s.label == "bad")
    rg = m.predict(read_point_cloud(good.modalities["pointcloud"])[0])
    rb = m.predict(read_point_cloud(bad.modalities["pointcloud"])[0])
    assert rb.deviation.shape == (len(rb.points),)
    assert rb.score > rg.score
