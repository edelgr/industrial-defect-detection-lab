"""Fitted-model cache: a reloaded model must give identical predictions without re-fitting."""

import numpy as np
import pytest

from core.algorithms import registry
from core.datasets.io import read_rgb8
from core.inference.session import LabSession


@pytest.mark.parametrize("mid,params", [("padim", {}), ("patchcore", {"backbone": "resnet18"}),
                                        ("anomalydino", {"k_shot": 2, "image_size": 224})])
def test_cache_roundtrip(mid, params, synthetic_index, tmp_path, monkeypatch):
    ok, why = registry.availability(mid)
    if not ok:
        pytest.skip(why)
    from core import paths
    monkeypatch.setattr(paths, "CHECKPOINTS", tmp_path)
    from core.inference import model_registry as MR
    monkeypatch.setattr(MR, "FITTED", tmp_path / "fitted")     # never write test models into checkpoints/
    s1 = LabSession(mid, params)
    s1.bind(synthetic_index)
    info1 = s1.fit(max_images=5)
    assert "cache" not in info1
    img = read_rgb8(next(s for s in synthetic_index.samples if s.label == "bad").primary_path)
    r1 = s1.predict_image(img)

    s2 = LabSession(mid, params)
    s2.bind(synthetic_index)
    info2 = s2.fit(max_images=5)
    assert info2["cache"].endswith(".pt") and info2.get("loaded_from_cache")
    r2 = s2.predict_image(img)
    assert np.allclose(r1.anomaly_map, r2.anomaly_map, atol=1e-4)
    assert abs(r1.score - r2.score) < 1e-4

    s4 = LabSession(mid, params)            # re-fit without cache must reproduce exactly (seeded)
    s4.bind(synthetic_index)
    s4.fit(max_images=5, use_cache=False)
    r4 = s4.predict_image(img)
    assert np.allclose(r1.anomaly_map, r4.anomaly_map, atol=1e-4) and abs(r1.score - r4.score) < 1e-4

    s3 = LabSession(mid, params)            # different fit set -> different cache entry
    s3.bind(synthetic_index)
    assert "cache" not in s3.fit(max_images=4)
