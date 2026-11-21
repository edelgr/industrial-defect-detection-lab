"""Model Registry — catalogue of fitted-model checkpoints in ``checkpoints/fitted/``.

Every checkpoint ``<method>_<dataset>_<category>_<key>.pt`` gets a JSON *manifest* next to it
(``<same name>.json``) that records its effective configuration and provenance: method,
all parameters, dataset/category/root, the ordered fit sample ids, seed, Anomalib version,
network input size, fit date and how the manifest was obtained.  The 16-hex ``key`` in the file
name is recomputed from the manifest (``session.fit_key``) before anything is loaded, so a
checkpoint is identified by its effective configuration, never by its file name alone.

Tiers
  workshop_verified  key listed in ``configs/workshop_verified.json`` — re-loaded and re-evaluated
                     by ``scripts/verify_workshop_models.py``, metrics reproduced, SHA-256 stored
  quick_experiment   default fit parameters and ≤ 60 fit images (the Lab / Compare defaults)
  custom             any other identified checkpoint
  unidentified       no manifest and no configuration reproduces its key (kept, never loaded)

Manifests for checkpoints created before the registry existed are *reconstructed*: candidate
configurations (default parameters and the known variants, every fit size, the seed) are hashed
until one reproduces the key.  A match is exact — the key covers the ordered fit ids.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path

from core import paths

FITTED = paths.CHECKPOINTS / "fitted"
VERIFIED_FILE = paths.CONFIGS / "workshop_verified.json"
TIERS = ("workshop_verified", "quick_experiment", "custom", "unidentified")
TIER_LABEL = {"workshop_verified": "Workshop Verified", "quick_experiment": "Quick Experiment",
              "custom": "Custom", "unidentified": "Unidentified"}
QUICK_MAX_FIT = 60
# parameter variants tried when reconstructing manifests of pre-registry checkpoints
VARIANTS = {"padim": [{}, {"keep_aspect": True}], "patchcore": [{}, {"keep_aspect": True}]}
_NEG: dict[str, float] = {}     # checkpoints whose configuration could not be reconstructed
_NAME = re.compile(r"^(?P<m>[a-z0-9]+)_(?P<d>D\d\d|SYN)_(?P<c>.+)_(?P<k>[0-9a-f]{16})\.pt$")


def manifest_path(ckpt: Path) -> Path:
    return ckpt.with_suffix(".json")


def parse_name(path: Path) -> dict | None:
    m = _NAME.match(path.name)
    return None if m is None else {"method_id": m["m"], "dataset_id": m["d"], "category": m["c"], "key": m["k"]}


def write_manifest(session, ckpt: Path, provenance: str, max_images: int | None = None) -> dict:
    from core.algorithms.torch_common import anomalib_version
    meth = session.method
    st = ckpt.stat()
    man = {
        "schema": 1, "file": ckpt.name, "key": parse_name(ckpt)["key"], "method_id": session.method_id,
        "model_version": meth.model_version(), "params": dict(meth.params),
        "fit_params": {p.name: meth.params[p.name] for p in meth.param_specs() if p.affects_fit},
        "dataset_id": session.index.dataset_id, "category": session.index.category,
        "dataset_root": str(session.index.root), "fit_ids": list(session.fit_ids),
        "n_fit": len(session.fit_ids), "max_images_requested": max_images, "seed": session.seed,
        "anomalib_version": anomalib_version(), "input_hw": meth.fit_info.get("input_hw") or _square(meth),
        "fit_info": {k: v for k, v in meth.fit_info.items() if isinstance(v, (int, float, str, bool, list))
                     and k != "loaded_from_cache"},
        "fit_date": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(st.st_mtime)),
        "size_bytes": st.st_size, "created_by": getattr(session, "created_by", "lab"), "provenance": provenance,
    }
    manifest_path(ckpt).write_text(json.dumps(man, indent=1), encoding="utf-8")
    return man


def _square(meth):
    s = meth.params.get("image_size")
    return [int(s), int(s)] if s else None


# ------------------------------------------------------------------------------ entries
@dataclass
class CheckpointEntry:
    path: Path
    manifest: dict | None
    tier: str
    verified: dict | None = None

    @property
    def key(self) -> str:
        return (parse_name(self.path) or {}).get("key", "")

    def get(self, k, default=None):
        return (self.manifest or parse_name(self.path) or {}).get(k, default)

    def label(self) -> str:
        if self.verified and self.verified.get("label"):
            return self.verified["label"]
        m = self.manifest or {}
        hw = m.get("input_hw")
        res = f"{hw[1]}×{hw[0]}" if hw else "?"
        return f"{self.get('method_id')} · {self.get('category')} · {res} · {m.get('n_fit', '?')} img"

    def describe(self) -> list[tuple[str, str]]:
        m = self.manifest or {}
        hw = m.get("input_hw")
        rows = [("Tier", TIER_LABEL[self.tier]), ("Dataset", f"{self.get('dataset_id')} · {self.get('category')}"),
                ("Algorithm", m.get("model_version") or self.get("method_id")),
                ("Configuration", ", ".join(f"{k}={v}" for k, v in (m.get("fit_params") or {}).items()) or "—"),
                ("Network input", f"{hw[1]}×{hw[0]} (w×h)" if hw else "—"),
                ("Fit images", f"{m.get('n_fit', '?')} train/good"), ("Seed", str(m.get("seed", "?"))),
                ("Fit date", m.get("fit_date", "?")), ("Anomalib", m.get("anomalib_version", "?")),
                ("Provenance", m.get("provenance", "no manifest")), ("File", self.path.name)]
        if self.verified:
            rows.append(("Verified", f"{self.verified.get('verified_date')} — metrics reproduced "
                                     f"(max |Δ| {self.verified.get('reference_check', {}).get('max_abs_diff', '—')})"))
        return rows


def load_verified() -> dict[str, dict]:
    if not VERIFIED_FILE.exists():
        return {}
    data = json.loads(VERIFIED_FILE.read_text(encoding="utf-8"))
    return {m["key"]: m for m in data.get("models", [])}


def _tier(man: dict | None, verified: dict | None, size: int) -> str:
    if man is None:
        return "unidentified"
    if verified is not None and verified.get("size_bytes") == size:
        return "workshop_verified"
    from core.algorithms import registry as areg
    cls = areg.runtime_class(man["method_id"])
    defaults = {p.name: p.default for p in cls.param_specs() if p.affects_fit} if cls else {}
    if all(man["fit_params"].get(k) == v for k, v in defaults.items()) and man["n_fit"] <= QUICK_MAX_FIT:
        return "quick_experiment"
    return "custom"


def scan(reconstruct: bool = True) -> list[CheckpointEntry]:
    """All checkpoints in checkpoints/fitted, with manifests (reconstructed if missing)."""
    verified = load_verified()
    out = []
    if not FITTED.is_dir():
        return out
    for p in sorted(FITTED.glob("*.pt")):
        mp = manifest_path(p)
        man = json.loads(mp.read_text(encoding="utf-8")) if mp.exists() else None
        if man is None and reconstruct and _NEG.get(p.name) != p.stat().st_mtime:
            man = reconstruct_manifest(p)
            if man is None:
                _NEG[p.name] = p.stat().st_mtime      # do not search again in this process
        v = verified.get((parse_name(p) or {}).get("key"))
        out.append(CheckpointEntry(p, man, _tier(man, v, p.stat().st_size), v))
    order = {t: i for i, t in enumerate(TIERS)}
    out.sort(key=lambda e: (order[e.tier], str(e.get("dataset_id")), str(e.get("category")), str(e.get("method_id")),
                            e.path.name))
    return out


def reconstruct_manifest(path: Path, seeds: tuple[int, ...] | None = None) -> dict | None:
    """Search the configuration that reproduces ``path``'s key; write and return its manifest."""
    info = parse_name(path)
    if info is None:
        return None
    from core import settings
    from core.algorithms import registry as areg
    from core.datasets import registry as dreg
    from core.inference.session import LabSession, fit_key
    if info["category"] not in dreg.installed_categories(info["dataset_id"]) or areg.runtime_class(info["method_id"]) is None:
        return None
    idx = dreg.load_index(info["dataset_id"], info["category"])
    for seed in seeds or tuple(dict.fromkeys((settings.get().seed, 42))):
        for params in VARIANTS.get(info["method_id"], [{}]):
            s = LabSession(info["method_id"], params, seed=seed)
            s.bind(idx)
            full, _ = s._split_train(10 ** 9)
            for n in range(1, len(full) + 1):
                ids = [x.sample_id for x in full[:n]]
                if fit_key(s.method, info["method_id"], ids, seed, idx.root) == info["key"]:
                    s.fit_ids = ids
                    if info["method_id"] in ("padim", "patchcore") and params.get("keep_aspect"):
                        from core.datasets.io import read_rgb8
                        s.method.fit_info = {"input_hw": list(s.method._input_hw(read_rgb8(full[0].primary_path)))}
                    return write_manifest(s, path, provenance="reconstructed by exact cache-key match "
                                                              "(fit date = file time)", max_images=n)
    return None


# ------------------------------------------------------------------------------ queries
def compatibility(entry: CheckpointEntry, index) -> tuple[bool, list[str]]:
    """Can ``entry`` be loaded for this dataset index?  Returns (ok, reasons-if-not)."""
    from core.algorithms import registry as areg
    from core.algorithms.torch_common import anomalib_version
    from core.inference.session import fit_key
    r: list[str] = []
    m = entry.manifest
    if m is None:
        return False, ["no manifest — configuration unknown"]
    if not entry.path.exists():
        r.append("checkpoint file missing")
    if (m["dataset_id"], m["category"]) != (index.dataset_id, index.category):
        r.append(f"fitted on {m['dataset_id']}/{m['category']}, current data is {index.dataset_id}/{index.category}")
    if m["dataset_root"] != str(index.root):
        r.append("dataset folder differs from the one used for fitting")
    if m["anomalib_version"] != anomalib_version():
        r.append(f"Anomalib {m['anomalib_version']} ≠ installed {anomalib_version()}")
    ok, why = areg.availability(m["method_id"])
    if not ok:
        r.append(why)
    train = {s.sample_id for s in index.train_normals()}
    if not set(m["fit_ids"]) <= train:
        r.append("some fit images are not train/good images of the current index")
    if not r:
        meth = areg.create(m["method_id"], **m["params"])
        if fit_key(meth, m["method_id"], m["fit_ids"], int(m["seed"]), index.root) != entry.key:
            r.append("recomputed cache key differs from the file key")
    return not r, r


def for_index(index, method_id: str | None = None) -> list[CheckpointEntry]:
    """Checkpoints for this dataset/category (and method), verified first."""
    return [e for e in scan() if e.get("dataset_id") == index.dataset_id and e.get("category") == index.category
            and (method_id is None or e.get("method_id") == method_id)]


def verified_presets(dataset_id: str | None = None, category: str | None = None) -> list[CheckpointEntry]:
    return [e for e in scan() if e.tier == "workshop_verified"
            and (dataset_id is None or e.get("dataset_id") == dataset_id)
            and (category is None or e.get("category") == category)]


def find_preset(preset_id: str) -> CheckpointEntry | None:
    return next((e for e in scan() if e.verified and e.verified.get("preset_id") == preset_id), None)


# ------------------------------------------------------------------------------ stored results
@dataclass
class StoredPreset:
    """A Workshop Verified result set from ``configs/workshop_verified.json`` (metrics, thresholds and
    per-sample scores of the public test split, no image data). ``entry`` is the matching checkpoint
    when it is present on this computer (instructor installation); otherwise only the *stored* results
    can be shown — never as live inference."""
    verified: dict
    entry: CheckpointEntry | None

    @property
    def key(self) -> str:
        return self.verified["key"]

    @property
    def live(self) -> bool:
        return self.entry is not None and self.entry.tier == "workshop_verified"

    @property
    def manifest(self) -> dict:
        if self.entry is not None and self.entry.manifest:
            return self.entry.manifest
        return {"input_hw": self.verified.get("input_hw"), "fit_info": {}, "seed": self.verified.get("seed", 42)}

    def get(self, k, default=None):
        return self.verified.get(k, default)

    def label(self) -> str:
        return self.verified.get("label", self.key)


def stored_presets(dataset_id: str | None = None, category: str | None = None) -> list[StoredPreset]:
    entries = {e.key: e for e in scan()} if FITTED.is_dir() else {}
    out = []
    for key, v in load_verified().items():
        if (dataset_id and v["dataset_id"] != dataset_id) or (category and v["category"] != category):
            continue
        e = entries.get(key)
        out.append(StoredPreset(v, e if (e is not None and e.tier == "workshop_verified") else None))
    return out


def find_stored(preset_id: str) -> StoredPreset | None:
    return next((p for p in stored_presets() if p.verified.get("preset_id") == preset_id), None)
