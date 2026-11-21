"""Group F — WinCLIP (Jeong et al., CVPR 2023) via Anomalib's WinClipModel.

There is no official code from the authors; Anomalib's re-implementation (Apache-2.0) is used,
with OpenCLIP ViT-B-16-plus-240 (LAION-400M) weights.  Zero-shot mode needs only a class name
for the text prompts; few-shot mode additionally uses k normal reference images.
"""

from __future__ import annotations

from core.algorithms.base import AnomalyMethod, MethodResult, ParamSpec
from core.algorithms.torch_common import (CLIP_MEAN, CLIP_STD, anomalib_version, configure_model_cache,
                                          device, have, to_tensor, upsample_map)
import cv2


class WinCLIPMethod(AnomalyMethod):
    method_id = "winclip"
    display_name = "WinCLIP"
    requires_fit = True          # "fit" = set the class prompt and (optionally) k reference images
    supports_gpu = True

    @classmethod
    def param_specs(cls):
        return [
            ParamSpec("class_name", "choice", "auto", "Object name for prompts",
                      choices=["auto", "object", "metal sheet", "metal plate", "surface", "fabric", "bottle",
                               "screw", "printed circuit board", "capsule"], affects_fit=True,
                      help="'auto' uses the dataset category name, e.g. 'sheet metal'."),
            ParamSpec("k_shot", "int", 0, "Normal reference images (0 = zero-shot)", 0, 8, 1, affects_fit=True),
        ]

    @classmethod
    def availability(cls):
        return have("torch", "anomalib", "open_clip")

    def model_version(self):
        return f"Anomalib {anomalib_version()} WinClipModel (OpenCLIP ViT-B-16-plus-240, laion400m_e31, k={self.params['k_shot']})"

    def set_category(self, category: str):
        self.category = category

    def fit(self, images, progress=None, cancel=None):
        import torch
        from anomalib.models.image.winclip.torch_model import WinClipModel
        configure_model_cache()
        self.device = device()
        cname = self.params["class_name"]
        if cname == "auto":
            cname = getattr(self, "category", "object").replace("_", " ")
        if progress:
            progress(0.05, "loading OpenCLIP weights (≈0.8 GB on first use, cached in checkpoints/)…")
        k = int(self.params["k_shot"])
        refs = to_tensor(images[:k], (240, 240), CLIP_MEAN, CLIP_STD, cv2.INTER_CUBIC) if k > 0 else None
        m = WinClipModel(class_name=None).to(self.device)
        m.setup(cname, refs.to(self.device) if refs is not None else None)
        m.eval()
        self.model = m
        self.fitted = True
        self.fit_info = {"class_name": cname, "k_shot": k}

    def _predict(self, image, reference):
        import torch
        with torch.no_grad():
            out = self.model(to_tensor([image], (240, 240), CLIP_MEAN, CLIP_STD, cv2.INTER_CUBIC).to(self.device))
        amap = out.anomaly_map.squeeze().float().cpu().numpy()
        return MethodResult(self.method_id, upsample_map(amap, image.shape[:2]),
                            float(out.pred_score.squeeze().float().cpu()), device=self.device,
                            score_semantics="map: window-level CLIP image–text anomaly similarity (harmonic "
                                            "aggregation over windows/scales); score: zero/few-shot image score. "
                                            f"Prompts use class name '{self.fit_info.get('class_name')}'.")
