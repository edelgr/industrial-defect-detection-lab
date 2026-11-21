"""Student-facing model availability: what can run on this computer, from which source, under which
licence, and what is missing.  Nothing here downloads or trains unless ``prepare()`` is called
explicitly (Model Manager button / prepare_models.ps1).

Availability classes (one method can have several):
  included          code included with IDD Lab; no pretrained weights needed (classical methods)
  downloadable      official pretrained backbone available for download from its original source
  student_trained   fitted on this computer (Model Registry tiers Quick Experiment / Custom)
  instructor_only   Workshop Verified checkpoints — instructor installation only, never redistributed
  unavailable       explanation only, or missing dependencies
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Pretrained resources downloaded by the user from the original source (IDD Lab never redistributes
# them). Licence values verified 2026-11-21 from the Hugging Face model-card API / GitHub release API
# (see docs/LICENSE_AUDIT.md for URLs and caveats).
WEIGHTS = {
    "padim": {"resource": "timm/resnet18.a1_in1k", "url": "https://huggingface.co/timm/resnet18.a1_in1k",
              "size": "≈ 45 MB", "licence": "Apache-2.0 (model card); trained on ImageNet-1k — ImageNet terms not verified"},
    "patchcore": {"resource": "timm/wide_resnet50_2.racm_in1k", "url": "https://huggingface.co/timm/wide_resnet50_2.racm_in1k",
                  "size": "≈ 270 MB", "licence": "Apache-2.0 (model card); trained on ImageNet-1k — ImageNet terms not verified"},
    "anomalydino": {"resource": "timm/vit_small_patch14_dinov2.lvd142m",
                    "url": "https://huggingface.co/timm/vit_small_patch14_dinov2.lvd142m",
                    "size": "≈ 90 MB", "licence": "Apache-2.0 (model card, DINOv2)"},
    "winclip": {"resource": "timm/vit_base_patch16_plus_clip_240.laion400m_e31",
                "url": "https://huggingface.co/timm/vit_base_patch16_plus_clip_240.laion400m_e31",
                "size": "≈ 800 MB", "licence": "MIT (model card, OpenCLIP); trained on LAION-400M"},
    "efficientad": {"resource": "Anomalib release 'efficientad_pretrained_weights' + Imagenette2-320",
                    "url": "https://github.com/open-edge-platform/anomalib/releases/tag/efficientad_pretrained_weights",
                    "size": "≈ 40 MB + 342 MB", "licence": "teacher weights: licence not stated in the release; "
                                                           "Imagenette repository: Apache-2.0 (image terms not stated)"},
}
FITTED_REDISTRIBUTION = ("Not distributable without permission: fitted on dataset images (e.g. MVTec AD 2: "
                         "CC BY-NC-SA 4.0; MulSen-AD: licence not stated)")


@dataclass
class MethodStatus:
    method_id: str
    name: str
    group: str
    implementation: str
    references: list[str]
    runnable: bool
    reason: str
    classes: list[str] = field(default_factory=list)
    needs_fit: bool = False
    cpu: str = ""
    weights: dict | None = None
    weights_cached: bool | None = None
    fitted: dict = field(default_factory=dict)       # tier -> count (present on disk)
    instructor_presets: int = 0                       # Workshop Verified entries (stored results)
    gpu_inference_mb: float | None = None             # measured on the instructor GPU, if verified
    redistribution: str = ""

    def summary_key(self) -> tuple[str, str]:
        """(i18n key, English text) of the status shown in the Model Manager."""
        if not self.runnable:
            return "mm.s_unavailable", "unavailable"
        if "included" in self.classes:
            return "mm.s_included", "ready (included, no weights needed)"
        if self.method_id == "winclip":
            return (("mm.s_zero_shot", "ready — zero-shot ('Fit' only prepares the text prompts)") if self.weights_cached
                    else ("mm.s_needs_weights", "needs pretrained weights (download)"))
        if self.needs_fit:
            if self.fitted:
                return "mm.s_fitted", "ready — fitted model(s) available"
            return (("mm.s_cached", "weights cached — fit on normal images to use") if self.weights_cached
                    else ("mm.s_needs_weights_fit", "needs pretrained weights (download) and fitting"))
        return ("mm.s_ready", "ready") if self.weights_cached else ("mm.s_needs_weights", "needs pretrained weights (download)")

    @property
    def summary(self) -> str:
        key, text = self.summary_key()
        return text + (" — " + self.reason if not self.runnable else "")


def statuses() -> list[MethodStatus]:
    from core.algorithms import registry as areg
    from core.inference import model_registry as MR
    from core.research.catalog import get_catalog
    from core.system.readiness import weights_cached
    cat = get_catalog()
    entries = MR.scan()
    verified = MR.load_verified()
    out = []
    for mid, rec in sorted(cat.algorithms.items(), key=lambda kv: (kv[1].get("group", ""), kv[0])):
        cls = areg.runtime_class(mid)
        ok, why = areg.availability(mid) if cls else (False, "explanation only in IDD Lab (no runnable implementation)")
        st = MethodStatus(mid, rec["name"], rec.get("group", ""), rec.get("implementation_type") or "none",
                          list(rec.get("publication_ids", [])), bool(ok), why)
        if not cls:
            st.classes = ["unavailable"]
            out.append(st)
            continue
        st.needs_fit = bool(getattr(cls, "requires_fit", False))
        w = WEIGHTS.get(mid)
        st.weights = w
        if w is None and not getattr(cls, "supports_gpu", False):
            st.classes.append("included")
            st.cpu = "yes (CPU only)"
        elif w is None:
            st.classes.append("included")
            st.cpu = "yes"
        else:
            st.classes.append("downloadable")
            if mid == "efficientad":
                try:
                    from core.algorithms.teacher_student.efficientad import assets_present
                    st.weights_cached = bool(assets_present())
                except Exception:  # noqa: BLE001 — torch/anomalib may be absent in the student core install
                    st.weights_cached = False
            else:
                st.weights_cached = bool(weights_cached(mid))
            st.cpu = ("fitting/inference on CPU is slow; 10 000-step training is impractical without a GPU"
                      if mid == "efficientad" else "yes (slower than GPU)")
        tiers: dict[str, int] = {}
        for e in entries:
            if e.get("method_id") == mid and e.manifest:
                tiers[e.tier] = tiers.get(e.tier, 0) + 1
        st.fitted = tiers
        if any(t in tiers for t in ("quick_experiment", "custom")):
            st.classes.append("student_trained")
        vs = [m for m in verified.values() if m["method_id"] == mid]
        st.instructor_presets = len(vs)
        if vs:
            st.classes.append("instructor_only")
            peaks = [m["metrics"].get("gpu_peak_mb") for m in vs if m["metrics"].get("gpu_peak_mb")]
            st.gpu_inference_mb = max(peaks) if peaks else None
        if not ok:
            st.classes.append("unavailable")
        st.redistribution = ("IDD Lab code (licence pending owner decision) on top of " + (rec.get("source_license") or "—")
                             + ("; weights downloaded by the user from the original source, not redistributed" if w else "")
                             + ("; fitted checkpoints: " + FITTED_REDISTRIBUTION if st.needs_fit else ""))
        out.append(st)
    return out


def prepare(method_id: str, progress=None, cancel=None) -> str:
    """Explicitly download/cache the pretrained resources of one method (online). Never fits on data."""
    import numpy as np
    from core.algorithms import registry as areg
    from core.algorithms.torch_common import configure_model_cache
    configure_model_cache()
    if method_id == "efficientad":
        from core.algorithms.teacher_student.efficientad import assets_present, prepare_assets
        prepare_assets(progress=progress, cancel=cancel)
        return f"EfficientAD assets present: {assets_present()}"
    params = {"anomalydino": {"k_shot": 1}, "winclip": {"k_shot": 0}}.get(method_id, {})
    m = areg.create(method_id, **params)
    if progress:
        progress(0.2, f"{method_id}: downloading / loading pretrained backbone…")
    dummy = [np.full((256, 256, 3), 128, np.uint8) for _ in range(2)]
    m.fit(dummy)            # builds the network (downloads weights); dummy grey images, nothing is saved
    if progress:
        progress(1.0, "done")
    return f"{method_id}: {m.model_version()} ready"
