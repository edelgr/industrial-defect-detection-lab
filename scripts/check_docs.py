"""Documentation checks (offline): run by CI and before a release.

  python -m scripts.check_docs

* every relative Markdown link in tracked/untracked (not ignored) .md files points to an existing file;
* PowerShell scripts and Python modules mentioned as commands exist;
* no GitHub URL for IDD Lab itself is invented before the repository exists;
* every exercise listed by the Learning Center exists.
Exit status 1 on any problem.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

from core import paths

LINK = re.compile(r"\[[^\]]*\]\(([^)\s]+)\)")
PS1 = re.compile(r"\.\\([A-Za-z_]+\.ps1)")
MOD = re.compile(r"python(?:\.exe)? -m (scripts\.[a-z0-9_]+)")
OWN_REPO = re.compile(r"github\.com/[^/\s)]+/(idd[-_]?lab)\b", re.IGNORECASE)


SKIP_DIRS = {".venv", "data", "checkpoints", "cache", "experiments", ".git", "node_modules"}


def md_files() -> list[Path]:
    try:
        out = subprocess.run(["git", "ls-files", "-co", "--exclude-standard", "*.md"], cwd=paths.PROJECT_ROOT,
                             capture_output=True, text=True, check=True).stdout.split()
        return [paths.PROJECT_ROOT / f for f in out]
    except (subprocess.CalledProcessError, FileNotFoundError):     # not a Git checkout (e.g. a release copy)
        root = paths.PROJECT_ROOT
        return sorted(p for p in root.rglob("*.md")
                      if not (set(p.relative_to(root).parts[:-1]) & SKIP_DIRS)
                      or p.relative_to(root).as_posix() in ("data/README.md", "checkpoints/README.md", "experiments/README.md"))


def check() -> list[str]:
    problems = []
    for f in md_files():
        txt = f.read_text(encoding="utf-8", errors="ignore")
        rel = f.relative_to(paths.PROJECT_ROOT).as_posix()
        for m in LINK.finditer(txt):
            target = m.group(1)
            if target.startswith(("http://", "https://", "mailto:", "#")):
                continue
            target = target.split("#")[0]
            if target and not (f.parent / target).exists():
                problems.append(f"{rel}: broken link -> {target}")
        for m in PS1.finditer(txt):
            if not (paths.PROJECT_ROOT / m.group(1)).exists():
                problems.append(f"{rel}: script not found -> {m.group(1)}")
        for m in MOD.finditer(txt):
            if not (paths.PROJECT_ROOT / (m.group(1).replace(".", "/") + ".py")).exists():
                problems.append(f"{rel}: module not found -> {m.group(1)}")
        for m in OWN_REPO.finditer(txt):
            problems.append(f"{rel}: GitHub URL for IDD Lab before a repository exists -> {m.group(0)}")
    from app.pages.learning import EXERCISE_FILES  # noqa: WPS433
    for ex, name in EXERCISE_FILES.items():
        if not (paths.DOCS / "exercises" / name).exists():
            problems.append(f"exercise sheet missing: {ex} -> docs/exercises/{name}")
    return problems


def main() -> int:
    problems = check()
    for p in problems:
        print("PROBLEM:", p)
    print(f"{len(md_files())} Markdown files checked; {len(problems)} problem(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
