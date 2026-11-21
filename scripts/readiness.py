"""Offline readiness report for the Instructor or Student profile.

  python -m scripts.readiness                      # resolved profile (auto)
  python -m scripts.readiness --profile student    # Student Edition: core learning path required only
  python -m scripts.readiness --profile instructor # full verified installation (SHA-256 of presets)
  python -m scripts.readiness --profile instructor --quick   # size check instead of SHA-256

Exit status 1 if a *required* check fails. Never downloads anything.
"""

from __future__ import annotations

import argparse
import sys


def main(argv=None) -> int:
    from core import paths, profile, settings
    settings.load()
    paths.ensure_runtime_dirs()
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", choices=["auto", "instructor", "student"], default="auto")
    ap.add_argument("--quick", action="store_true", help="instructor: compare checkpoint sizes, not SHA-256")
    a = ap.parse_args(argv)
    prof = profile.resolve(None if a.profile == "auto" else a.profile)
    from core.system.readiness import run_checks
    checks = run_checks(deep=not a.quick, profile=prof)
    print(f"IDD Lab offline readiness — profile: {prof}")
    for c in checks:
        mark = "OK  " if c.ok else ("FAIL" if c.required else "info")
        print(f"[{mark}] {c.name}: {c.detail}")
    failing = [c for c in checks if c.required and not c.ok]
    print(f"\n{len(failing)} required check(s) failing" + (": " + "; ".join(c.name for c in failing) if failing else ""))
    return 1 if failing else 0


if __name__ == "__main__":
    sys.exit(main())
