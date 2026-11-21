import os
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("TORCH_HOME", str(ROOT / "checkpoints" / "torch"))
os.environ.setdefault("HF_HOME", str(ROOT / "checkpoints" / "huggingface"))


@pytest.fixture(scope="session")
def synthetic_root(tmp_path_factory):
    from core.datasets.synthetic import generate, generate_pointclouds
    root = tmp_path_factory.mktemp("syn")
    generate(root, n_train=12, n_test_good=4, n_test_per_defect=3, size=128, seed=1)
    generate_pointclouds(root, n_train=2, n_test=3, seed=1)
    return root


@pytest.fixture(scope="session")
def synthetic_index(synthetic_root):
    from core.datasets.synthetic import SyntheticAdapter
    return SyntheticAdapter().index(synthetic_root, "brushed_metal")


@pytest.fixture
def rgb_image():
    rng = np.random.default_rng(0)
    img = (rng.random((96, 128, 3)) * 40 + 100).astype(np.uint8)
    img[40:50, 60:90] = 20          # a dark 'defect'
    return img


def cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


# ------------------------------------------------------------------ fitted-model cache isolation
# Every test writes fitted models to a temporary folder. Only tests marked ``real_checkpoints``
# (read-only use of the real Model Registry / Workshop Verified presets) see checkpoints/fitted/,
# and a session guard fails the run if any test created or modified a file there.
from core.inference import model_registry as _MR  # noqa: E402

REAL_FITTED = _MR.FITTED


def _fitted_snapshot() -> dict:
    if not REAL_FITTED.is_dir():
        return {}
    return {p.name: (p.stat().st_size, p.stat().st_mtime_ns) for p in REAL_FITTED.iterdir() if p.is_file()}


def pytest_configure(config):
    config.addinivalue_line("markers", "real_checkpoints: test reads the real checkpoints/fitted/ (must not write)")
    config._fitted_before = _fitted_snapshot()


def pytest_sessionfinish(session, exitstatus):
    import sys
    if "PySide6.QtCore" in sys.modules:          # let GUI tests' background tasks finish before teardown
        from PySide6.QtCore import QThreadPool
        QThreadPool.globalInstance().waitForDone(60000)
    before, after = session.config._fitted_before, _fitted_snapshot()
    changed = sorted(set(after) - set(before)) + sorted(k for k in before if k in after and after[k] != before[k])
    removed = sorted(set(before) - set(after))
    if changed or removed:
        print(f"\nERROR: tests modified checkpoints/fitted/: new/changed {changed}, removed {removed}")
        session.exitstatus = 1


@pytest.fixture(autouse=True)
def _isolate_fitted_cache(request, tmp_path_factory, monkeypatch):
    if request.node.get_closest_marker("real_checkpoints") is None:
        monkeypatch.setattr(_MR, "FITTED", tmp_path_factory.mktemp("fitted"))


@pytest.fixture
def deep():
    """Skip tests that fit deep models when PyTorch/Anomalib are not installed (Student core profile, CI)."""
    from core.algorithms import registry
    ok, why = registry.availability("padim")
    if not ok:
        pytest.skip(why)
