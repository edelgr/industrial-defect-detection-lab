"""Minimal UI string translation backed by education/i18n/ui_<lang>.yaml."""

from __future__ import annotations

import yaml

from core import paths, settings

_strings: dict[str, str] = {}
_fallback: dict[str, str] = {}
_lang = "en"


def _read(lang: str) -> dict[str, str]:
    f = paths.EDUCATION / "i18n" / f"ui_{lang}.yaml"
    if not f.exists():
        return {}
    with f.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


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
