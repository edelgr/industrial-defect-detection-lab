"""Build a reproducible workshop teaching subset with a selection manifest.

  python -m scripts.build_subset --dataset D06 --category capsule --name ws_capsule --seed 0
  python -m scripts.build_subset --dataset D05 --category sheet_metal --name ws_sheet_metal

The subset is written to <data_root>/subsets/<name>/ in the original layout. Point IDD Lab to it
with Settings → Dataset Manager → "Use existing extracted folder…".
"""

from __future__ import annotations

import argparse
from collections import Counter

from core import settings
from core.datasets import registry
from core.datasets.subset import build, select


def main(argv=None):
    settings.load()
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--category", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--n-train", type=int, default=40)
    ap.add_argument("--n-test-good", type=int, default=8)
    ap.add_argument("--per-defect", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    idx = registry.load_index(a.dataset, a.category)
    params = {"n_train": a.n_train, "n_test_good": a.n_test_good, "per_defect": a.per_defect, "seed": a.seed}
    sel = select(idx, a.n_train, a.n_test_good, a.per_defect, seed=a.seed)
    from core.research.catalog import get_catalog
    rec = get_catalog().datasets.get(a.dataset, {})
    out = build(idx, settings.get().data_root_path / "subsets", a.name, sel, params, rec.get("download_url"))
    print(f"subset '{a.name}': {len(sel)} samples -> {out}")
    for reason, n in Counter(r.split(" '")[0] if "anomaly type" in r else r.split(" (")[0] for _, r in sel).items():
        print(f"  {n:3d} × {reason}")


if __name__ == "__main__":
    main()
