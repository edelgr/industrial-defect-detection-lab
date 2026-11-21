"""Offline readiness check for the workshop (run before the event, while offline if possible)."""

from __future__ import annotations

from dataclasses import dataclass

from core import paths
from core.algorithms import registry as areg
from core.datasets import registry as dreg

# substrings of Hugging Face hub cache folders (models--timm--<name>) holding each backbone
WEIGHTS = {
    "patchcore": ("wide_resnet50_2", "≈270 MB"),
    "padim": ("resnet18", "≈45 MB"),
    "anomalydino": ("vit_small_patch14_dinov2", "≈90 MB"),
    "winclip": ("plus_clip_240", "≈800 MB"),
}


@dataclass
class Check:
    name: str
    ok: bool
    detail: str
    required: bool = True


def weights_cached(method_id: str) -> bool | None:
    spec = WEIGHTS.get(method_id)
    if spec is None:
        return None
    hub = paths.CHECKPOINTS / "huggingface" / "hub"
    if not hub.is_dir():
        return False
    for d in hub.iterdir():
        if d.is_dir() and spec[0] in d.name:
            if any(f.stat().st_size > 1_000_000 for f in d.rglob("*") if f.is_file()):
                return True
    return False


def verified_checks(deep: bool = True) -> list[Check]:
    """Workshop Verified presets: present, same size and (deep) same SHA-256 as when verified,
    and loadable for the installed data (manifest + recomputed cache key)."""
    import hashlib
    from core.inference import model_registry as MR
    v = MR.load_verified()
    if not v:
        return [Check("Workshop Verified presets", False, "configs/workshop_verified.json missing — run "
                      "python -m scripts.verify_workshop_models")]
    entries = {e.key: e for e in MR.scan()}
    bad, ok_n = [], 0
    for key, m in v.items():
        e = entries.get(key)
        if e is None or not e.path.exists():
            bad.append(f"{m['preset_id']}: checkpoint missing")
            continue
        if e.path.stat().st_size != m["size_bytes"]:
            bad.append(f"{m['preset_id']}: size differs")
            continue
        if deep:
            h = hashlib.sha256()
            with open(e.path, "rb") as f:
                for b in iter(lambda: f.read(1 << 22), b""):
                    h.update(b)
            if h.hexdigest() != m["sha256"]:
                bad.append(f"{m['preset_id']}: SHA-256 differs")
                continue
        if m["category"] in dreg.installed_categories(m["dataset_id"]):
            okc, why = MR.compatibility(e, dreg.load_index(m["dataset_id"], m["category"]))
            if not okc:
                bad.append(f"{m['preset_id']}: {'; '.join(why)}")
                continue
        ok_n += 1
    return [Check("Workshop Verified presets (checkpoints, SHA-256, cache keys)", not bad,
                  f"{ok_n}/{len(v)} ready" + ("" if deep else " (size check only)") + ("; " + "; ".join(bad[:4]) if bad else ""))]


def _have(mod: str) -> tuple[bool, str]:
    import importlib.util
    if importlib.util.find_spec(mod) is None:
        return False, "not installed"
    try:
        import importlib.metadata as md
        return True, md.version({"cv2": "opencv-python-headless", "skimage": "scikit-image", "sklearn": "scikit-learn",
                                 "yaml": "PyYAML"}.get(mod, mod))
    except Exception:  # noqa: BLE001
        return True, "installed"


def student_checks() -> list[Check]:
    """Student Edition readiness: only the core learning path is required. Deep learning, 3D, real
    datasets, pretrained weights and fitted checkpoints are optional and reported as information —
    instructor-only Workshop Verified checkpoints are never required."""
    from core import settings
    from core.inference import model_registry as MR
    from core.research.catalog import load_catalog
    out: list[Check] = []
    cat = load_catalog()
    out.append(Check("Research catalog (Atlas, Methods Library, references; offline)", not cat.problems,
                     f"{len(cat.datasets)} datasets, {len(cat.algorithms)} methods, {len(cat.publications)} publications"))
    for mod, label in (("PySide6", "Qt GUI (PySide6)"), ("numpy", "NumPy"), ("cv2", "OpenCV (classical methods)"),
                       ("skimage", "scikit-image (SSIM)"), ("sklearn", "scikit-learn (metrics)"),
                       ("matplotlib", "matplotlib (experiment plots)"), ("yaml", "PyYAML")):
        ok, v = _have(mod)
        out.append(Check(label, ok, v))
    syn = settings.get().data_root_path / "synthetic_demo"
    out.append(Check("Synthetic Educational Data (generated locally, fixed seed)", (syn / "brushed_metal").is_dir(),
                     str(syn) if (syn / "brushed_metal").is_dir() else "not generated yet — run "
                     "'prepare_datasets.ps1 -Synthetic' or start the app (generated on first run)"))
    stored = MR.load_verified()
    out.append(Check("Stored verified results (Educational Experiments, no data needed)", bool(stored),
                     f"{len(stored)} stored model results" if stored else "configs/workshop_verified.json missing"))
    # ---- optional learning paths (information only)
    ok_t, v_t = _have("torch")
    if ok_t:
        try:
            import torch
            v_t += " · CUDA " + (torch.cuda.get_device_name(0) if torch.cuda.is_available() else "not available (CPU)")
        except Exception as e:  # noqa: BLE001
            v_t += f" · {e}"
    out.append(Check("Optional: deep learning (PyTorch)", ok_t, v_t if ok_t else "not installed — setup_student.ps1 -Profile full", False))
    ok_a, v_a = _have("anomalib")
    out.append(Check("Optional: Anomalib (PaDiM, PatchCore, EfficientAD, AnomalyDINO, WinCLIP)", ok_a, v_a, False))
    if ok_t and ok_a:
        for mid, (_, size) in WEIGHTS.items():
            out.append(Check(f"Optional: {mid} pretrained backbone cached", bool(weights_cached(mid)),
                             "cached" if weights_cached(mid) else f"not downloaded ({size}) — Model Manager or prepare_models.ps1", False))
    ok_3, v_3 = _have("open3d")
    out.append(Check("Optional: 3D (Open3D, PyVistaQt)", ok_3 and _have("pyvistaqt")[0], v_3, False))
    for did, label in (("D05", "Optional: MVTec AD 2 (import from the official source)"),
                       ("D06", "Optional: MulSen-AD (import from the official source)")):
        cats = dreg.installed_categories(did)
        out.append(Check(label, bool(cats), ", ".join(cats) if cats else "not installed — see DATASETS.md", False))
    fitted = [e for e in MR.scan() if e.manifest]
    out.append(Check("Optional: fitted models in checkpoints/fitted", bool(fitted),
                     f"{len(fitted)} available" if fitted else "none — fit your own in Methods Laboratory, or use stored results", False))
    return out


def run_checks(deep: bool = True, profile: str | None = None) -> list[Check]:
    """Offline readiness for the resolved profile (instructor checks are unchanged)."""
    from core import profile as _profile
    if (profile or _profile.resolve()) == "student":
        return student_checks()
    return instructor_checks(deep=deep)


def instructor_checks(deep: bool = True) -> list[Check]:
    from core.research.catalog import load_catalog
    out: list[Check] = []
    cat = load_catalog()
    out.append(Check("Research catalog valid", not cat.problems,
                     f"{len(cat.datasets)} datasets, {len(cat.algorithms)} methods, {len(cat.publications)} publications"
                     + (f"; {len(cat.problems)} problems" if cat.problems else "")))
    try:
        import torch
        cuda = torch.cuda.is_available()
        out.append(Check("CUDA GPU", cuda, torch.cuda.get_device_name(0) if cuda else "CPU fallback will be used", False))
    except ImportError:
        out.append(Check("PyTorch", False, "not installed"))
    for did, label, need in (("D05", "MVTec AD 2 (sheet_metal, fruit_jelly) for Activities 2-5, 7, A-C", True),
                             ("D06", "MulSen-AD subset for Activity 6", True),
                             ("D04", "Real3D-AD (optional, Activity 6)", False),
                             ("SYN", "SYNTHETIC fallback data", False)):
        cats = dreg.installed_categories(did)
        out.append(Check(label, bool(cats), ", ".join(cats) if cats else f"not installed at {dreg.dataset_root(did)}", need))
    for mid, (_, size) in WEIGHTS.items():
        ok, why = areg.availability(mid)
        cached = weights_cached(mid)
        out.append(Check(f"{mid}: code + weights cached offline", ok and bool(cached),
                         ("weights cached" if cached else f"weights not cached ({size}) — run prepare_models.ps1 online")
                         if ok else why, mid in ("patchcore", "padim")))
    try:
        from core.algorithms.teacher_student.efficientad import assets_present
        ea = assets_present()
        out.append(Check("EfficientAD teacher weights + Imagenette2-320 cached", ea,
                         "cached" if ea else "≈ 0.38 GB downloaded on first fit (or run prepare_models.ps1 -WithEfficientAD)",
                         False))
    except Exception as e:
        out.append(Check("EfficientAD", False, str(e), False))
    out += verified_checks(deep=deep)
    try:
        import matplotlib  # noqa: F401
        out.append(Check("matplotlib (Educational Experiments plots)", True, matplotlib.__version__))
    except ImportError as e:
        out.append(Check("matplotlib (Educational Experiments plots)", False, str(e)))
    from core import settings
    out.append(Check("Workshop offline demo profile", settings.get().workshop_mode,
                     "active — Lab/Compare prefer verified models and ask before training" if settings.get().workshop_mode
                     else "off (Settings → General → Profile, or start.ps1 -Workshop)", False))
    ok, why = areg.availability("pc_deviation")
    out.append(Check("Open3D (3D baseline)", ok, why))
    try:
        import pyvistaqt  # noqa: F401
        out.append(Check("PyVistaQt 3D viewer", True, "ok"))
    except Exception as e:
        out.append(Check("PyVistaQt 3D viewer", False, str(e)))
    return out
