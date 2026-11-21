"""Dataset Manager: disk-space guard, resumable downloads, safe selective extraction, import.

Nothing here downloads automatically.  The GUI must show the resource, source, access conditions,
size estimate and free space, and obtain explicit user approval for transfers > 1 GB
(see :func:`requires_approval`).
"""

from __future__ import annotations

import hashlib
import json
import shutil
import tarfile
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Callable

from core.algorithms.base import CancelFn, CancelledError, ProgressFn

GB = 1024 ** 3
APPROVAL_THRESHOLD_BYTES = 1 * GB


class UnsafeArchiveError(RuntimeError):
    pass


class InsufficientSpaceError(RuntimeError):
    pass


def free_bytes(path: Path) -> int:
    p = Path(path)
    while not p.exists() and p.parent != p:
        p = p.parent
    return shutil.disk_usage(p).free


def check_space(dest: Path, needed_bytes: int, margin_gb: float) -> tuple[bool, str]:
    free = free_bytes(dest)
    remaining = free - needed_bytes
    ok = remaining >= margin_gb * GB
    msg = (f"Free: {free / GB:.1f} GB, needed: {needed_bytes / GB:.2f} GB, "
           f"would leave {remaining / GB:.1f} GB (required margin {margin_gb:.0f} GB)")
    return ok, msg


def requires_approval(size_bytes: int | None) -> bool:
    return size_bytes is None or size_bytes > APPROVAL_THRESHOLD_BYTES


# ----------------------------------------------------------------- download
@dataclass
class DownloadResult:
    path: Path
    bytes: int
    sha256: str
    resumed: bool


def sha256_file(path: Path, chunk: int = 4 * 2 ** 20, progress: ProgressFn | None = None) -> str:
    h = hashlib.sha256()
    total = path.stat().st_size or 1
    done = 0
    with path.open("rb") as f:
        while b := f.read(chunk):
            h.update(b)
            done += len(b)
            if progress:
                progress(done / total, "verifying checksum")
    return h.hexdigest()


def download(url: str, dest: Path, *, expected_sha256: str | None = None, progress: ProgressFn | None = None,
             cancel: CancelFn | None = None, timeout: float = 30.0, session=None) -> DownloadResult:
    """HTTP(S) download with resume (Range requests) and cancellation.

    Writes to ``dest.part`` and renames on success. Only call after user approval.
    """
    import requests
    s = session or requests.Session()
    s.headers.setdefault("User-Agent", "IDD-Lab/0.1 (educational dataset manager)")
    part = dest.with_suffix(dest.suffix + ".part")
    dest.parent.mkdir(parents=True, exist_ok=True)
    start = part.stat().st_size if part.exists() else 0
    headers = {"Range": f"bytes={start}-"} if start else {}
    with s.get(url, stream=True, headers=headers, timeout=timeout, allow_redirects=True) as r:
        if r.status_code == 416:          # already complete
            pass
        else:
            r.raise_for_status()
            resumed = start > 0 and r.status_code == 206
            if start and not resumed:
                start = 0                  # server ignored Range; restart
            total = int(r.headers.get("Content-Length", 0)) + start
            mode = "ab" if resumed else "wb"
            done, t0 = start, time.time()
            with part.open(mode) as f:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    if cancel and cancel():
                        raise CancelledError("download cancelled (partial file kept for resume)")
                    f.write(chunk)
                    done += len(chunk)
                    if progress and total:
                        rate = (done - start) / max(time.time() - t0, 1e-6) / 2 ** 20
                        progress(done / total, f"{done / 2 ** 20:.0f}/{total / 2 ** 20:.0f} MB @ {rate:.1f} MB/s")
    digest = sha256_file(part, progress=progress)
    if expected_sha256 and digest.lower() != expected_sha256.lower():
        raise IOError(f"Checksum mismatch for {url}: got {digest}")
    part.replace(dest)
    return DownloadResult(dest, dest.stat().st_size, digest, start > 0)


# ------------------------------------------------------------- safe extraction
def _safe_member(name: str) -> PurePosixPath:
    p = PurePosixPath(name.replace("\\", "/"))
    if p.is_absolute() or ".." in p.parts or (p.parts and ":" in p.parts[0]):
        raise UnsafeArchiveError(f"Unsafe path in archive: {name!r}")
    return p


def list_archive(archive: Path) -> list[str]:
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as z:
            return z.namelist()
    with tarfile.open(archive) as t:
        return t.getnames()


def detect_categories(members: list[str], known: list[str] | None = None) -> dict[str, str]:
    """Map category name -> archive prefix, for archives laid out as ``[root/]<category>/...``."""
    found: dict[str, str] = {}
    for m in members:
        parts = _safe_member(m).parts
        for depth in range(min(3, len(parts) - 1)):
            name = parts[depth]
            if known is not None and name in known:
                found.setdefault(name, "/".join(parts[:depth + 1]))
    return found


def extract(archive: Path, dest: Path, *, include_prefixes: list[str] | None = None,
            strip_prefix: str | None = None, progress: ProgressFn | None = None,
            cancel: CancelFn | None = None) -> list[Path]:
    """Extract an archive safely (no path traversal, no links), optionally only some prefixes.

    ``include_prefixes``: archive paths (posix) to extract, e.g. ["mvtec_ad_2/sheet_metal"].
    ``strip_prefix``: leading archive path removed from extracted names.
    ``archive`` may also be an open binary file object (used by the tests, so that hostile test
    archives never exist on disk where endpoint protection would quarantine them).
    """
    dest = dest.resolve()
    dest.mkdir(parents=True, exist_ok=True)
    out: list[Path] = []

    def wanted(name: str) -> bool:
        if not include_prefixes:
            return True
        return any(name == p or name.startswith(p.rstrip("/") + "/") for p in include_prefixes)

    def target_for(name: str) -> Path | None:
        rel = _safe_member(name)
        if strip_prefix:
            sp = PurePosixPath(strip_prefix)
            try:
                rel = rel.relative_to(sp)
            except ValueError:
                return None
        if not rel.parts:
            return None
        t = (dest / Path(*rel.parts)).resolve()
        if dest not in t.parents and t != dest:
            raise UnsafeArchiveError(f"Path escapes destination: {name!r}")
        return t

    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as z:
            infos = [i for i in z.infolist() if wanted(i.filename.rstrip("/"))]
            for k, info in enumerate(infos):
                if cancel and cancel():
                    raise CancelledError("extraction cancelled")
                t = target_for(info.filename)
                if t is None:
                    continue
                if info.is_dir():
                    t.mkdir(parents=True, exist_ok=True)
                    continue
                t.parent.mkdir(parents=True, exist_ok=True)
                with z.open(info) as src, t.open("wb") as dst:
                    shutil.copyfileobj(src, dst, 1 << 20)
                out.append(t)
                if progress and k % 25 == 0:
                    progress((k + 1) / len(infos), info.filename)
    else:
        if hasattr(archive, "seek"):
            archive.seek(0)                   # is_zipfile() moved the stream position
        with (tarfile.open(fileobj=archive) if hasattr(archive, "read") else tarfile.open(archive)) as tf:
            members = [m for m in tf.getmembers() if wanted(m.name.rstrip("/"))]
            for k, m in enumerate(members):
                if cancel and cancel():
                    raise CancelledError("extraction cancelled")
                if m.issym() or m.islnk() or m.isdev():
                    continue              # never materialise links/devices from third-party archives
                t = target_for(m.name)
                if t is None:
                    continue
                if m.isdir():
                    t.mkdir(parents=True, exist_ok=True)
                    continue
                t.parent.mkdir(parents=True, exist_ok=True)
                src = tf.extractfile(m)
                if src is None:
                    continue
                with src, t.open("wb") as dst:
                    shutil.copyfileobj(src, dst, 1 << 20)
                out.append(t)
                if progress and k % 25 == 0:
                    progress((k + 1) / len(members), m.name)
    if progress:
        progress(1.0, f"extracted {len(out)} files")
    return out


def write_install_record(dataset_root: Path, record: dict) -> None:
    """Append provenance of an import (archive name, checksum, selected categories, date)."""
    f = dataset_root / "idd_install_log.json"
    log = json.loads(f.read_text(encoding="utf-8")) if f.exists() else []
    log.append({**record, "date": time.strftime("%Y-%m-%d %H:%M:%S")})
    f.write_text(json.dumps(log, indent=2), encoding="utf-8")


def import_archive(archive: Path, dataset_root: Path, categories: list[str] | None, known_categories: list[str],
                   progress: ProgressFn | None = None, cancel: CancelFn | None = None,
                   compute_checksum: bool = True) -> dict:
    """Import a manually downloaded archive; extracts only the selected categories."""
    members = list_archive(archive)
    found = detect_categories(members, known_categories)
    if categories:
        missing = [c for c in categories if c not in found]
        if missing:
            raise ValueError(f"Categories not found in archive: {missing}. Found: {sorted(found)}")
        chosen = {c: found[c] for c in categories}
    else:
        chosen = found
    if not chosen:
        raise ValueError("No known dataset categories were recognised in this archive.")
    files = []
    for cat, prefix in chosen.items():
        parent = str(PurePosixPath(prefix).parent)
        files += extract(archive, dataset_root, include_prefixes=[prefix],
                         strip_prefix=None if parent == "." else parent, progress=progress, cancel=cancel)
    rec = {"archive": archive.name, "archive_bytes": archive.stat().st_size,
           "sha256": sha256_file(archive) if compute_checksum else None,
           "categories": sorted(chosen), "files": len(files)}
    write_install_record(dataset_root, rec)
    return rec
