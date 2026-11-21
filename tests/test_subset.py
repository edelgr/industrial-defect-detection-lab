"""Teaching subset builder: provenance, determinism, original layout preserved."""

import json

from core.datasets.subset import build, select
from core.datasets.synthetic import SyntheticAdapter


def test_subset_is_deterministic_and_traceable(synthetic_index, tmp_path):
    a = select(synthetic_index, n_train=5, n_test_good=2, per_defect=1, seed=3)
    b = select(synthetic_index, n_train=5, n_test_good=2, per_defect=1, seed=3)
    assert [s.sample_id for s, _ in a] == [s.sample_id for s, _ in b]
    reasons = {r for _, r in a}
    assert any("anomaly type 'scratch'" in r for r in reasons)
    assert any(r.startswith("difficult") for r in reasons)
    out = build(synthetic_index, tmp_path, "ws", a, {"seed": 3})
    man = json.loads((out / "subset_manifest.json").read_text(encoding="utf-8"))
    assert man["n_samples"] == len(a) and all("selection_reason" in s for s in man["samples"])
    # the subset is readable by the same adapter with identical ids, splits and masks
    idx2 = SyntheticAdapter().index(out, "brushed_metal")
    orig = {s.sample_id: s for s, _ in a}
    assert {s.sample_id for s in idx2.samples} == set(orig)
    for s in idx2.samples:
        assert s.split == orig[s.sample_id].split and s.label == orig[s.sample_id].label
        assert bool(s.masks) == bool(orig[s.sample_id].masks)
    # the source dataset is untouched (copy, not move)
    assert all(s.primary_path.exists() for s, _ in a)
