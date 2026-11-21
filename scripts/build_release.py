"""Stage a Student Edition source release in a SEPARATE directory (never modifies the repository).

  python -m scripts.build_release --out C:\\IDD_release            # from the last commit (git archive HEAD)
  python -m scripts.build_release --out C:\\IDD_release --worktree # tracked + untracked, not ignored (pre-commit check)
  python -m scripts.build_release --out C:\\IDD_release --zip

Policy (docs/REDISTRIBUTION_POLICY.md):
* only files that Git would publish are considered; then datasets, checkpoints, experiment databases, caches,
  virtual environments, archives and model-weight formats are excluded even if tracked;
* the local user name / home folder is redacted in the staged copy of text files (the repository is untouched);
* the staged tree is re-scanned: any forbidden extension, large file, home path or user name fails the build;
* RELEASE_MANIFEST.json lists every file with SHA-256, the source commit and the assets that need the
  owner's review before publication (``conditional``).
Nothing is uploaded or published.
"""

from __future__ import annotations

import argparse
import fnmatch
import getpass
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path

from core import paths

POLICY_VERSION = 1
EXCLUDE = ["data/*", "checkpoints/*", "experiments/*", "cache/*", ".venv/*", "*/__pycache__/*", "configs/user_settings.json",
           "*.pt", "*.pth", "*.ckpt", "*.onnx", "*.safetensors", "*.bin", "*.sqlite", "*.db", "*.zip", "*.tar", "*.tar.gz",
           "*.tgz", "*.7z", "*.npz", "*.npy", "*.h5", "*.partial", "*quarantine*"]
KEEP = ["data/README.md", "checkpoints/README.md", "experiments/README.md"]
FORBIDDEN_EXT = {".pt", ".pth", ".ckpt", ".onnx", ".safetensors", ".sqlite", ".db", ".zip", ".tar", ".7z", ".npz", ".npy", ".h5"}
MAX_FILE_BYTES = 5 * 2 ** 20
TEXT_EXT = {".md", ".py", ".ps1", ".yaml", ".yml", ".json", ".txt", ".toml", ".bib", ".cfg", ".ini", ".gitignore",
            ".gitattributes", ".svg", ".csv"}
# Assets that are included but need the owner's explicit review before publication.
CONDITIONAL = {
    "configs/workshop_verified.json": "Derived evaluation results (metrics, thresholds, per-image scores; no image data) "
                                      "on MVTec AD 2 test_public (CC BY-NC-SA 4.0) and MulSen-AD (licence not stated). "
                                      "Publishing derived scores is common scientific practice, but the MulSen-AD part "
                                      "needs the owner's decision (or removal of the d06-* entries).",
    "docs/LOCAL_RESULTS.md": "Local educational results on MulSen-AD (licence not stated by the authors).",
    "docs/LOCAL_RESULTS_D05.md": "Local educational results on MVTec AD 2 (CC BY-NC-SA 4.0, attribution given).",
    "docs/screenshots/": "Screenshots of IDD Lab with catalog content and SYNTHETIC data only (no third-party images).",
}


def _files_from_git(worktree: bool) -> list[tuple[str, bytes]]:
    root = paths.PROJECT_ROOT
    if worktree:
        out = subprocess.run(["git", "ls-files", "-co", "--exclude-standard", "-z"], cwd=root, capture_output=True, check=True)
        names = [n for n in out.stdout.decode("utf-8").split("\0") if n]
        return [(n, (root / n).read_bytes()) for n in names if (root / n).is_file()]
    arch = subprocess.run(["git", "archive", "--format=tar", "HEAD"], cwd=root, capture_output=True, check=True).stdout
    files = []
    with tarfile.open(fileobj=io.BytesIO(arch)) as tf:
        for m in tf.getmembers():
            if m.isfile():
                files.append((m.name, tf.extractfile(m).read()))
    return files


def _excluded(name: str) -> bool:
    if name in KEEP:
        return False
    return any(fnmatch.fnmatch(name, pat) for pat in EXCLUDE)


def _redactions() -> list[tuple[str, str]]:
    user = getpass.getuser()
    home = str(Path.home())
    pairs = [(home, "<HOME>"), (home.replace("\\", "/"), "<HOME>"), (home.replace("\\", "\\\\"), "<HOME>")]
    if len(user) >= 3:
        pairs.append((user, "<user>"))
    return pairs


def _redact(name: str, data: bytes, pairs) -> tuple[bytes, int]:
    if Path(name).suffix.lower() not in TEXT_EXT and Path(name).name not in (".gitignore", ".gitattributes"):
        return data, 0
    try:
        txt = data.decode("utf-8")
    except UnicodeDecodeError:
        return data, 0
    n = 0
    for a, b in pairs:
        if a and a in txt:
            n += txt.count(a)
            txt = txt.replace(a, b)
    return txt.encode("utf-8"), n


def build(out: Path, worktree: bool = False, make_zip: bool = False) -> dict:
    out = out.resolve()
    if paths.PROJECT_ROOT.resolve() in [out, *out.parents]:
        raise SystemExit("The staging directory must be outside the repository.")
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} is not empty — choose a new staging directory (nothing is deleted automatically).")
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=paths.PROJECT_ROOT, capture_output=True, text=True).stdout.strip()
    pairs = _redactions()
    files, excluded, redacted = [], [], {}
    for name, data in _files_from_git(worktree):
        if _excluded(name):
            excluded.append(name)
            continue
        data, n = _redact(name, data, pairs)
        if n:
            redacted[name] = n
        dst = out / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(data)
        files.append({"path": name, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()})
    # ---- re-scan the staged tree
    problems = []
    user = getpass.getuser()
    for f in files:
        p = out / f["path"]
        if p.suffix.lower() in FORBIDDEN_EXT:
            problems.append(f"forbidden file type: {f['path']}")
        if f["bytes"] > MAX_FILE_BYTES:
            problems.append(f"file larger than {MAX_FILE_BYTES // 2**20} MB: {f['path']}")
        if p.suffix.lower() in TEXT_EXT:
            txt = p.read_text(encoding="utf-8", errors="ignore")
            if str(Path.home()) in txt or (len(user) >= 3 and user in txt):
                problems.append(f"user name or home path still present: {f['path']}")
    conditional = {k: v for k, v in CONDITIONAL.items() if any(f["path"].startswith(k) for f in files)}
    manifest = {"product": "IDD Lab — Student Edition (source release staging)", "policy_version": POLICY_VERSION,
                "source": "worktree" if worktree else f"git archive HEAD ({commit})", "commit": commit,
                "created": time.strftime("%Y-%m-%d %H:%M:%S"), "files": files, "n_files": len(files),
                "total_bytes": sum(f["bytes"] for f in files), "excluded": excluded,
                "redactions": {k: v for k, v in redacted.items()}, "conditional_review": conditional,
                "problems": problems, "published": False}
    (out / "RELEASE_MANIFEST.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    if make_zip and not problems:
        shutil.make_archive(str(out), "zip", out)
    return manifest


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="new, empty staging directory OUTSIDE the repository")
    ap.add_argument("--worktree", action="store_true", help="stage tracked + untracked (not ignored) files instead of HEAD")
    ap.add_argument("--zip", action="store_true")
    a = ap.parse_args(argv)
    m = build(Path(a.out), a.worktree, a.zip)
    print(f"staged {m['n_files']} files ({m['total_bytes'] / 2**20:.1f} MB) from {m['source']} into {a.out}")
    print(f"excluded {len(m['excluded'])}; redacted {sum(m['redactions'].values())} occurrence(s) in {len(m['redactions'])} file(s)")
    for k, v in m["conditional_review"].items():
        print(f"  needs owner review: {k} — {v[:90]}…")
    for p in m["problems"]:
        print("  PROBLEM:", p)
    print("NOT published (staging only).")
    return 1 if m["problems"] else 0


if __name__ == "__main__":
    sys.exit(main())
