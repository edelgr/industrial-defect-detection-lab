"""Loader for educational content files (Markdown/YAML) with language fallback to English."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml

from core import paths, settings


def _lang() -> str:
    return settings.get().language or "en"


def markdown(kind: str, item_id: str, lang: str | None = None) -> str | None:
    """education/<lang>/<kind>/<item_id>.md, falling back to English."""
    lang = lang or _lang()
    for lg in (lang, "en"):
        f = paths.EDUCATION / lg / kind / f"{item_id}.md"
        if f.exists():
            return f.read_text(encoding="utf-8")
    return None


@lru_cache(maxsize=64)
def _yaml(path: Path) -> dict | list | None:
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def yaml_content(relpath: str, lang: str | None = None):
    lang = lang or _lang()
    for lg in (lang, "en"):
        d = _yaml(paths.EDUCATION / lg / relpath)
        if d is not None:
            return d
    return _yaml(paths.EDUCATION / relpath)


def algorithm_steps(method_id: str, lang: str | None = None) -> dict | None:
    """Step-by-step 'How this algorithm works' definition for a method."""
    data = yaml_content("steps.yaml", lang) or {}
    return data.get(method_id)


def method_visual(method_id: str, lang: str | None = None) -> dict | None:
    """Didactic visual content for the Methods Library (summary, pipeline diagram, example captions,
    strengths/limitations): education/<lang>/method_visuals.yaml, falling back to English."""
    lang = lang or _lang()
    for lg in dict.fromkeys((lang, "en")):
        rec = (_yaml(paths.EDUCATION / lg / "method_visuals.yaml") or {}).get(method_id)
        if rec:
            return rec
    return None


def workshop(lang: str | None = None) -> dict:
    return yaml_content("workshop.yaml", lang) or {}
