"""Minimal UI string translation backed by education/i18n/ui_<lang>.yaml."""

from __future__ import annotations

import yaml

from core import paths, settings

_strings: dict[str, str] = {}
_fallback: dict[str, str] = {}
_lang = "en"


def _read(lang: str) -> dict[str, str]:
    """ui_<lang>.yaml plus any feature-specific ui_<lang>_<part>.yaml files (merged, sorted by name)."""
    d = paths.EDUCATION / "i18n"
    out: dict[str, str] = {}
    for f in [d / f"ui_{lang}.yaml", *sorted(d.glob(f"ui_{lang}_*.yaml"))]:
        if f.exists():
            with f.open(encoding="utf-8") as fh:
                out.update(yaml.safe_load(fh) or {})
    return out


def init(lang: str | None = None) -> None:
    global _strings, _fallback, _lang
    _lang = lang or settings.get().language or "en"
    _fallback = _read("en")
    _strings = _read(_lang) if _lang != "en" else _fallback


def tr(key: str, default: str | None = None) -> str:
    if not _fallback:
        init()
    return _strings.get(key) or _fallback.get(key) or default or key


def current() -> str:
    return _lang
