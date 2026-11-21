"""Group F — AnomalyDINO (Damm et al., WACV 2025) via Anomalib's AnomalyDINOModel.

Few-shot: the memory bank is built from k normal reference images (default k=4) using DINOv2
ViT-S/14 patch features; test patches are scored by cosine distance to their nearest normal
patch, image score = mean of the top-1 % patch distances (as in the paper).
"""

from __future__ import annotations

from core.algorithms.base import AnomalyMethod, MethodResult, ParamSpec
from core.algorithms.torch_common import (anomalib_version, configure_model_cache, device, have,
                                          to_tensor, upsample_map)


class AnomalyDINOMethod(AnomalyMethod):
    method_id = "anomalydino"
    display_name = "AnomalyDINO"
    requires_fit = True
    supports_gpu = True

    @classmethod
    def param_specs(cls):
        return [
            ParamSpec("k_shot", "int", 4, "Normal reference images (k-shot)", 1, 64, 1, affects_fit=True,
                      help="Only the first k of the provided normal images are used (few-shot)."),
            ParamSpec("encoder", "choice", "vit_small_patch14_dinov2", "DINOv2 encoder (timm)",
                      choices=["vit_small_patch14_dinov2", "vit_base_patch14_dinov2"], affects_fit=True),
            ParamSpec("image_size", "int", 448, "Input size (px, multiple of 14)", 224, 672, 14, affects_fit=True),
            ParamSpec("masking", "bool", False, "Zero-shot background masking (PCA)", affects_fit=True,
                      help="Paper uses masking for some object categories; off for textures."),
        ]

    @classmethod
    def availability(cls):
        return have("torch", "anomalib", "timm")

    def model_version(self):
        return f"Anomalib {anomalib_version()} AnomalyDINOModel ({self.params['encoder']}, k={self.params['k_shot']})"

    cacheable = True

    def _construct(self):
        from anomalib.models.image.anomaly_dino.torch_model import AnomalyDINOModel
        configure_model_cache()
        self.device = device()
        return AnomalyDINOModel(num_neighbours=1, encoder_name=self.params["encoder"],
                                masking=bool(self.params["masking"])).to(self.device)

    def save_state(self, path):
        import torch
        torch.save({"memory_bank": self.model.memory_bank.cpu(), "fit_info": self.fit_info}, path)

    def load_state(self, path):
        import torch
        ck = torch.load(path, map_location="cpu", weights_only=False)
        self.model = self._construct()
        self.model.memory_bank = ck["memory_bank"].to(self.device)
        self.model.eval()
        self.fit_info = {**ck["fit_info"], "loaded_from_cache": True}
        self.fitted = True

    def fit(self, images, progress=None, cancel=None):
        import torch
        if progress:
            progress(0.05, "loading DINOv2 (cached in checkpoints/)…")
        m = self._construct()
        m.train()
        s = int(self.params["image_size"])
        shots = images[: int(self.params["k_shot"])]
        with torch.no_grad():
            for i, im in enumerate(shots):
                self._check_cancel(cancel)
                m(to_tensor([im], (s, s)).to(self.device))
                if progress:
                    progress(0.1 + 0.85 * (i + 1) / len(shots), f"embedding reference {i + 1}/{len(shots)}")
        m.fit()
        m.eval()
        self.model = m
        self.fitted = True
        self.fit_info = {"k_shot": len(shots), "memory_bank": int(m.memory_bank.shape[0])}

    def _predict(self, image, reference):
        import torch
        s = int(self.params["image_size"])
        with torch.no_grad():
            out = self.model(to_tensor([image], (s, s)).to(self.device))
        amap = out.anomaly_map.squeeze().float().cpu().numpy()
        return MethodResult(self.method_id, upsample_map(amap, image.shape[:2]),
                            float(out.pred_score.squeeze().float().cpu()), device=self.device,
                            score_semantics="map: cosine distance of each DINOv2 patch to its nearest normal "
                                            "reference patch; score: mean of top-1 % patch distances")
