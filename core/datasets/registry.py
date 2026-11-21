"""Dataset adapter registry (plugin-style discovery of supported local dataset layouts)."""

from __future__ import annotations

from pathlib import Path

from core import settings
from core.datasets.base import DatasetAdapter, DatasetIndex

_ADAPTERS: dict[str, DatasetAdapter] = {}


def register(adapter: DatasetAdapter) -> DatasetAdapter:
    _ADAPTERS[adapter.dataset_id] = adapter
    return adapter


def _bootstrap() -> None:
    if _ADAPTERS:
        return
    from core.datasets.mvtec import MVTecAD2Adapter, MVTecADAdapter, MVTecLOCOAdapter
    from core.datasets.synthetic import SyntheticAdapter
    from core.datasets.mulsen import MulSenADAdapter
    from core.datasets.real3d import Real3DADAdapter
    for a in (MVTecADAdapter(), MVTecLOCOAdapter(), MVTecAD2Adapter(), MulSenADAdapter(),
              Real3DADAdapter(), SyntheticAdapter()):
        register(a)


def adapters() -> dict[str, DatasetAdapter]:
    _bootstrap()
    return dict(_ADAPTERS)


def get(dataset_id: str) -> DatasetAdapter | None:
    _bootstrap()
    return _ADAPTERS.get(dataset_id)


def dataset_root(dataset_id: str, data_root: Path | None = None) -> Path | None:
    a = get(dataset_id)
    if a is None:
        return None
    custom = settings.get().recent.get("dataset_roots", {}).get(dataset_id)
    if custom:
        return Path(custom)
    return a.default_root(data_root or settings.get().data_root_path)


def installed_categories(dataset_id: str) -> list[str]:
    a, root = get(dataset_id), dataset_root(dataset_id)
    if a is None or root is None:
        return []
    try:
        return a.list_categories(root)
    except OSError:
        return []


def installation_status(dataset_id: str) -> str:
    cats = installed_categories(dataset_id)
    return "installed" if cats else "not_installed"


_INDEX_CACHE: dict[tuple[str, str, str], DatasetIndex] = {}


def load_index(dataset_id: str, category: str, refresh: bool = False) -> DatasetIndex:
    a, root = get(dataset_id), dataset_root(dataset_id)
    if a is None or root is None:
        raise KeyError(f"No adapter for dataset {dataset_id}")
    key = (dataset_id, category, str(root))
    if refresh or key not in _INDEX_CACHE:
        _INDEX_CACHE[key] = a.index(root, category)
    return _INDEX_CACHE[key]
