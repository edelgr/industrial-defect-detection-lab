"""Dataset preparation CLI (selective, approval-gated).

Examples
--------
  python -m scripts.prepare_datasets synthetic
  python -m scripts.prepare_datasets mulsen --categories capsule nut flat_pad          # shows plan only
  python -m scripts.prepare_datasets mulsen --categories capsule nut flat_pad --yes    # downloads
  python -m scripts.prepare_datasets import --dataset D05 --archive C:/Downloads/sheet_metal.tar.xz --categories sheet_metal
  python -m scripts.prepare_datasets status

Rules: any transfer > 1 GB requires --approve-large in addition to --yes; at least
`min_free_space_gb` (default 15 GB) must remain free on the destination drive.
"""

from __future__ import annotations

import argparse
import json
import sys
import time

from core import settings
from core.datasets import manager, registry
from core.datasets.manager import GB

MULSEN_URL = "https://huggingface.co/datasets/orgjy314159/MulSen_AD/resolve/main/MulSen_AD_new.zip"


def _bar(f: float, msg: str) -> None:
    n = int(f * 30)
    sys.stdout.write(f"\r[{'#' * n}{'.' * (30 - n)}] {100 * f:5.1f}% {msg[:70]:70s}")
    sys.stdout.flush()


def cmd_synthetic(_a) -> None:
    from core.datasets.synthetic import ensure_generated
    root = ensure_generated(settings.get().data_root_path)
    print(f"SYNTHETIC demo data ready at {root}")


def cmd_mulsen(a) -> None:
    from core.datasets import remote_zip as rz
    dest = registry.dataset_root("D06")
    print(f"Source  : {MULSEN_URL}\n          (Hugging Face dataset orgjy314159/MulSen_AD, linked from the official repo "
          f"https://github.com/MapleBoat/MulSen-AD)")
    print("Access  : public, not gated. Dataset licence: NOT stated by the authors -> local research/teaching use "
          "only, never redistribute.")
    print("Method  : HTTP Range requests on the ZIP central directory; only the selected categories are fetched.")
    rf, zf = rz.list_remote(MULSEN_URL)
    infos = rz.select(zf, [f"MulSen_AD/{c}" for c in a.categories])
    if not infos:
        raise SystemExit("No files matched the requested categories.")
    p = rz.plan(infos)
    ok, msg = manager.check_space(dest, p["uncompressed_bytes"], settings.get().min_free_space_gb)
    print(f"Plan    : {p['files']} files, transfer ≈ {p['compressed_bytes'] / GB:.2f} GB, extracted ≈ "
          f"{p['uncompressed_bytes'] / GB:.2f} GB -> {dest}\nDisk    : {msg}")
    if not ok:
        raise SystemExit("Insufficient free space (safety margin).")
    if manager.requires_approval(p["compressed_bytes"]) and not a.approve_large:
        raise SystemExit("Transfer > 1 GB: re-run with --yes --approve-large after reviewing the plan.")
    if not a.yes:
        print("Dry run. Re-run with --yes to download.")
        return
    t0 = time.time()
    files = rz.extract_members(rf, infos, dest, strip_prefix="MulSen_AD", progress=_bar)
    print(f"\nExtracted {len(files)} files in {time.time() - t0:.0f} s; transferred {rf.transferred / 2 ** 20:.0f} MB")
    manager.write_install_record(dest, {"source": MULSEN_URL, "method": "http-range-zip", "categories": a.categories,
                                        "files": len(files), "transferred_bytes": rf.transferred,
                                        "crc32_verified": True})


def cmd_import(a) -> None:
    from pathlib import Path
    adapter = registry.get(a.dataset)
    if adapter is None:
        raise SystemExit(f"No local adapter for {a.dataset}")
    rec_cats = None
    from core.research.catalog import get_catalog
    rec = get_catalog().datasets.get(a.dataset)
    rec_cats = (rec or {}).get("categories") or a.categories
    dest = registry.dataset_root(a.dataset)
    arch = Path(a.archive)
    ok, msg = manager.check_space(dest, arch.stat().st_size * 2, settings.get().min_free_space_gb)
    print(msg)
    if not ok:
        raise SystemExit("Insufficient free space (safety margin).")
    rec = manager.import_archive(arch, dest, a.categories, rec_cats, progress=_bar)
    print("\n" + json.dumps(rec, indent=2))


def cmd_status(_a) -> None:
    for did, ad in registry.adapters().items():
        print(f"{did:4s} {type(ad).__name__:20s} {registry.installation_status(did):14s} "
              f"{registry.dataset_root(did)}  {registry.installed_categories(did)}")


def main(argv=None) -> None:
    settings.load()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("synthetic").set_defaults(fn=cmd_synthetic)
    m = sub.add_parser("mulsen")
    m.add_argument("--categories", nargs="+", default=["capsule", "nut", "flat_pad"])
    m.add_argument("--yes", action="store_true")
    m.add_argument("--approve-large", action="store_true")
    m.set_defaults(fn=cmd_mulsen)
    i = sub.add_parser("import")
    i.add_argument("--dataset", required=True)
    i.add_argument("--archive", required=True)
    i.add_argument("--categories", nargs="*")
    i.set_defaults(fn=cmd_import)
    sub.add_parser("status").set_defaults(fn=cmd_status)
    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
