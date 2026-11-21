"""Persistent user settings (JSON, git-ignored)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from core import paths


@dataclass
class Settings:
    language: str = "en"                 # "en" | "es"
    theme: str = "dark"                  # "dark" | "light"
    data_root: str = ""                  # empty -> paths.DEFAULT_DATA_ROOT
    min_free_space_gb: float = 15.0      # safety margin kept free on the destination drive
    device: str = "auto"                 # "auto" | "cuda" | "cpu"
    seed: int = 42
    # Workshop offline demo profile: prefer Workshop Verified checkpoints, confirm before any training,
    # never download. Also enabled for one session by the IDD_WORKSHOP=1 environment variable.
    workshop_mode: bool = False
    # Distribution profile: "auto" (instructor if Workshop Verified checkpoints are present, else
    # student), "instructor" or "student" — see core/profile.py.
    profile: str = "auto"
    first_run_done: bool = False         # the first-run assistant was shown (or not needed)
    recent: dict = field(default_factory=dict)

    @property
    def data_root_path(self) -> Path:
        return Path(self.data_root) if self.data_root else paths.DEFAULT_DATA_ROOT


_settings: Settings | None = None
_saved_workshop_mode: bool | None = None    # value in the file when IDD_WORKSHOP forces the profile on


def load(path: Path = paths.USER_SETTINGS) -> Settings:
    global _settings
    s = Settings()
    if path.exists():
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            for k, v in raw.items():
                if hasattr(s, k):
                    setattr(s, k, v)
        except (json.JSONDecodeError, OSError):
            pass
    import os
    global _saved_workshop_mode
    _saved_workshop_mode = None
    if os.environ.get("IDD_WORKSHOP") == "1":        # session-only: not written back by save()
        _saved_workshop_mode = s.workshop_mode
        s.workshop_mode = True
    s._file_existed = path.exists()      # an existing settings file = an existing installation
    _settings = s
    return s


def get() -> Settings:
    return _settings if _settings is not None else load()


def save(s: Settings | None = None, path: Path = paths.USER_SETTINGS) -> None:
    s = s or get()
    path.parent.mkdir(parents=True, exist_ok=True)
    d = asdict(s)
    if _saved_workshop_mode is not None and s.workshop_mode:
        d["workshop_mode"] = _saved_workshop_mode
    path.write_text(json.dumps(d, indent=2), encoding="utf-8")
