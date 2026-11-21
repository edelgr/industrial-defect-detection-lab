"""Portable project paths. All paths are derived from the repository root with pathlib."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RESEARCH_CATALOG = PROJECT_ROOT / "research_catalog"
CATALOG_DATASETS = RESEARCH_CATALOG / "datasets"
CATALOG_ALGORITHMS = RESEARCH_CATALOG / "algorithms"
CATALOG_PUBLICATIONS = RESEARCH_CATALOG / "publications"
CATALOG_SCHEMAS = RESEARCH_CATALOG / "schemas"
LINK_AUDIT_FILE = RESEARCH_CATALOG / "link_audit.json"

EDUCATION = PROJECT_ROOT / "education"
CONFIGS = PROJECT_ROOT / "configs"
ASSETS = PROJECT_ROOT / "assets"
DOCS = PROJECT_ROOT / "docs"

# Default locations for git-ignored, locally prepared assets.  The data root can be
# relocated to another drive in Settings (see core.settings).
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data"
CHECKPOINTS = PROJECT_ROOT / "checkpoints"
CACHE = PROJECT_ROOT / "cache"
EXPERIMENTS = PROJECT_ROOT / "experiments"
USER_SETTINGS = CONFIGS / "user_settings.json"


def ensure_runtime_dirs() -> None:
    for p in (DEFAULT_DATA_ROOT, CHECKPOINTS, CACHE, EXPERIMENTS):
        p.mkdir(parents=True, exist_ok=True)
