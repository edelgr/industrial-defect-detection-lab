"""Selective extraction from a remote ZIP archive using HTTP Range requests.

This is a generic HTTP/ZIP technique, NOT a per-category download service offered by any dataset's
authors: the ZIP central directory (at the end of the file) is read with Range requests, and
only the compressed bytes of the selected members are fetched and decompressed locally.  It
works only when the server honours ``Range`` (e.g. Hugging Face / S3) and the archive is a ZIP
(not RAR/TAR).  The transferred volume is reported before anything is downloaded so the user can
approve it (> 1 GB requires explicit approval, see core.datasets.manager).
"""

from __future__ import annotations

import io
import struct
import zipfile
import zlib
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from core.algorithms.base import CancelFn, CancelledError, ProgressFn
from core.datasets.manager import _safe_member


class RangeFile(io.RawIOBase):
    """Seekable read-only file over HTTP Range requests (with a small read-ahead cache)."""

    def __init__(self, url: str, session=None, block: int = 1 << 20):
        import requests
        self.url = url
        self.s = session or requests.Session()
        self.s.headers.setdefault("User-Agent", "IDD-Lab/0.1 (educational dataset manager)")
        r = self.s.head(url, allow_redirects=True, timeout=30)
        r.raise_for_status()
        self.final_url = r.url
        self.size = int(r.headers["Content-Length"])
        if r.headers.get("Accept-Ranges", "").lower() != "bytes":
            # many CDNs omit the header on HEAD; probe with a 1-byte range
            p = self.s.get(self.final_url, headers={"Range": "bytes=0-0"}, timeout=30)
            if p.status_code != 206:
                raise IOError("Server does not support HTTP Range requests")
        self.pos = 0
        self.block = block
        self._cache_start, self._cache = -1, b""
        self.transferred = 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = {0: off, 1: self.pos + off, 2: self.size + off}[whence]
        return self.pos

    def fetch(self, start: int, end: int) -> bytes:
        r = self.s.get(self.final_url, headers={"Range": f"bytes={start}-{end - 1}"}, timeout=120)
        if r.status_code != 206:
            raise IOError(f"Range request failed: HTTP {r.status_code}")
        self.transferred += len(r.content)
        return r.content

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        n = min(n, self.size - self.pos)
        if n <= 0:
            return b""
        cs, ce = self._cache_start, self._cache_start + len(self._cache)
        if not (cs <= self.pos and self.pos + n <= ce):
            start = self.pos
            end = min(self.size, start + max(n, self.block))
            self._cache_start, self._cache = start, self.fetch(start, end)
            cs = start
        out = self._cache[self.pos - cs:self.pos - cs + n]
        self.pos += len(out)
        return out

    def readinto(self, b):
        data = self.read(len(b))
        b[:len(data)] = data
        return len(data)


@dataclass
class RemoteMember:
    name: str
    compressed: int
    size: int


def list_remote(url: str, session=None) -> tuple[RangeFile, zipfile.ZipFile]:
    rf = RangeFile(url, session)
    return rf, zipfile.ZipFile(rf)


def select(zf: zipfile.ZipFile, prefixes: list[str]) -> list[zipfile.ZipInfo]:
    out = []
    for i in zf.infolist():
        if i.is_dir():
            continue
        if any(i.filename.startswith(p.rstrip("/") + "/") for p in prefixes):
            out.append(i)
    return out


def plan(infos: list[zipfile.ZipInfo]) -> dict:
    return {"files": len(infos), "compressed_bytes": sum(i.compress_size for i in infos),
            "uncompressed_bytes": sum(i.file_size for i in infos)}


def extract_members(rf: RangeFile, infos: list[zipfile.ZipInfo], dest: Path, strip_prefix: str = "",
                    progress: ProgressFn | None = None, cancel: CancelFn | None = None) -> list[Path]:
    """Download and decompress only ``infos`` (stored or deflated members), verifying CRC-32."""
    dest = dest.resolve()
    out = []
    total = sum(i.compress_size for i in infos) or 1
    done = 0
    for info in infos:
        if cancel and cancel():
            raise CancelledError("remote extraction cancelled")
        rel = _safe_member(info.filename)
        if strip_prefix:
            rel = rel.relative_to(PurePosixPath(strip_prefix))
        target = (dest / Path(*rel.parts)).resolve()
        if dest not in target.parents:
            raise IOError(f"unsafe path {info.filename}")
        if target.exists() and target.stat().st_size == info.file_size:
            done += info.compress_size
            out.append(target)
            continue                                     # resume: already extracted
        hdr = rf.fetch(info.header_offset, info.header_offset + 30)
        sig, *_rest = struct.unpack("<IHHHHHIIIHH", hdr)
        if sig != 0x04034B50:
            raise IOError(f"bad local header for {info.filename}")
        n_len, x_len = struct.unpack("<HH", hdr[26:30])
        start = info.header_offset + 30 + n_len + x_len
        raw = b""
        chunk = 8 << 20
        pos = start
        while pos < start + info.compress_size:
            end = min(start + info.compress_size, pos + chunk)
            raw += rf.fetch(pos, end)
            pos = end
            if cancel and cancel():
                raise CancelledError("remote extraction cancelled")
        if info.compress_type == zipfile.ZIP_STORED:
            data = raw
        elif info.compress_type == zipfile.ZIP_DEFLATED:
            data = zlib.decompress(raw, -15)
        else:
            raise IOError(f"unsupported compression {info.compress_type} for {info.filename}")
        if zlib.crc32(data) & 0xFFFFFFFF != info.CRC:
            raise IOError(f"CRC mismatch for {info.filename}")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        out.append(target)
        done += info.compress_size
        if progress:
            progress(done / total, f"{len(out)}/{len(infos)} files · {done / 2 ** 20:.0f}/{total / 2 ** 20:.0f} MB")
    return out
