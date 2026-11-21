"""Distribution profile: one codebase, two editions.

* ``instructor`` — the full verified installation: real datasets, Workshop Verified checkpoints,
  instructor readiness (SHA-256 of every verified checkpoint).
* ``student``   — the GitHub edition: works without real datasets or pretrained/fitted models;
  stored, verified results are shown as *stored results*, never as live inference; readiness only
  checks what the chosen learning path needs.

``Settings.profile`` may be "instructor", "student" or "auto" (default). "auto" resolves to
instructor when at least one Workshop Verified checkpoint is present in ``checkpoints/fitted/``,
otherwise to student — so the existing instructor installation keeps its behaviour unchanged.
"""

from __future__ import annotations

PROFILES = ("instructor", "student")


def resolve(setting: str | None = None) -> str:
    from core import settings
    p = setting if setting is not None else getattr(settings.get(), "profile", "auto")
    if p in PROFILES:
        return p
    try:
        from core.inference import model_registry as MR
        verified = MR.load_verified()
        if verified and any((MR.FITTED / m["file"]).exists() for m in verified.values()):
            return "instructor"
    except Exception:  # noqa: BLE001 — a broken registry must not prevent start-up
        pass
    return "student"


def is_student() -> bool:
    return resolve() == "student"
