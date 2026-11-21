"""Documentation checks (offline): run by CI and before a release.

  python -m scripts.check_docs

* every relative Markdown link in the project documentation (see ``md_files``) points to an existing file;
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


NESTED_SKIP = {".git", "node_modules", "__pycache__", ".ipynb_checkpoints"}

# Scope of the project documentation. Personal notes or drafts that are not tracked by Git (e.g. a
# README draft next to the project) are NOT project documentation and are not checked; they are
# listed by ``main()`` as "not checked" so nothing is hidden silently.
DOC_DIRS = ("docs", "education", "workshop")          # new, still untracked docs here ARE checked
# Root documents of this distribution (the workshop guides live in workshop/CIARP2026/guide/).
ROOT_DOCS = ("ARCHITECTURE.md", "CHANGELOG.md", "CONTRIBUTING.md", "DATASETS.md", "EXPERIMENT_PROTOCOLS.md",
             "INSTALL_WINDOWS.md", "METHODS.md", "README.md", "RESEARCH_REFERENCES.md", "STUDENT_QUICK_START.md",
             "THIRD_PARTY_NOTICES.md", "TROUBLESHOOTING.md")
PLACEHOLDERS = ("data/README.md", "checkpoints/README.md", "experiments/README.md")


def _git_md(*args: str, root: Path) -> list[str] | None:
    try:
        out = subprocess.run(["git", "ls-files", *args, "--", "*.md"], cwd=root, capture_output=True,
                             text=True, check=True).stdout
    except (subprocess.CalledProcessError, FileNotFoundError, NotADirectoryError):
        return None
    return [ln for ln in out.splitlines() if ln.strip()]


def _in_doc_dirs(rel: str) -> bool:
    return rel.split("/", 1)[0] in DOC_DIRS


def md_files(root: Path | None = None, use_git: bool = True) -> list[Path]:
    """Project Markdown documentation.

    * Git checkout: every tracked ``.md`` file, plus untracked (not ignored) ``.md`` files inside
      ``DOC_DIRS`` (new documentation being written before its first commit).
    * No Git (a ZIP download or a release copy): ``ROOT_DOCS``, every ``.md`` under ``DOC_DIRS`` and the
      data/checkpoint/experiment placeholders — the same set as the tracked files (guarded by a test).
    """
    root = root or paths.PROJECT_ROOT
    if use_git and (root / ".git").exists():
        tracked = _git_md(root=root)
        untracked = _git_md("-o", "--exclude-standard", root=root)
        if tracked is not None and untracked is not None:
            rels = set(tracked) | {u for u in untracked if _in_doc_dirs(u)}
            return sorted(root / r for r in rels if (root / r).exists())
    out = [root / f for f in ROOT_DOCS + PLACEHOLDERS if (root / f).exists()]
    for d in DOC_DIRS:
        if (root / d).is_dir():
            # only caches are skipped inside DOC_DIRS: education/<lang>/experiments/ IS documentation
            out += [p for p in (root / d).rglob("*.md") if not (set(p.relative_to(root / d).parts[:-1]) & NESTED_SKIP)]
    return sorted(set(out))


def unchecked_md(root: Path | None = None) -> list[str]:
    """Untracked Markdown files outside the documentation scope (personal notes, drafts): reported, not checked."""
    root = root or paths.PROJECT_ROOT
    if not (root / ".git").exists():
        return []
    untracked = _git_md("-o", "--exclude-standard", root=root) or []
    return sorted(u for u in untracked if not _in_doc_dirs(u))


def check(root: Path | None = None) -> list[str]:
    root = root or paths.PROJECT_ROOT
    problems = []
    for f in md_files(root):
        txt = f.read_text(encoding="utf-8", errors="ignore")
        rel = f.relative_to(root).as_posix()
        for m in LINK.finditer(txt):
            target = m.group(1)
            if target.startswith(("http://", "https://", "mailto:", "#")):
                continue
            target = target.split("#")[0]
            if target and not (f.parent / target).exists():
                problems.append(f"{rel}: broken link -> {target}")
        for m in PS1.finditer(txt):
            if not (root / m.group(1)).exists():
                problems.append(f"{rel}: script not found -> {m.group(1)}")
        for m in MOD.finditer(txt):
            if not (root / (m.group(1).replace(".", "/") + ".py")).exists():
                problems.append(f"{rel}: module not found -> {m.group(1)}")
        for m in OWN_REPO.finditer(txt):
            problems.append(f"{rel}: GitHub URL for IDD Lab before a repository exists -> {m.group(0)}")
    from app.pages.learning import EXERCISE_FILES  # noqa: WPS433
    for ex, name in EXERCISE_FILES.items():
        if not (root / "docs" / "exercises" / name).exists():
            problems.append(f"exercise sheet missing: {ex} -> docs/exercises/{name}")
    return problems


def main() -> int:
    problems = check()
    for p in problems:
        print("PROBLEM:", p)
    print(f"{len(md_files())} Markdown files checked; {len(problems)} problem(s)")
    skipped = unchecked_md()
    if skipped:
        print(f"note: {len(skipped)} untracked Markdown file(s) outside {', '.join(DOC_DIRS)}/ are not project "
              f"documentation and were not checked: {', '.join(skipped)}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
