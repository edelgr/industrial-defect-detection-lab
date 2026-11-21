"""Group D — PaDiM and PatchCore via Anomalib's PyTorch model classes.

IDD Lab uses ``anomalib.models.image.{padim,patchcore}.torch_model`` directly (no Lightning
Engine) so that fitting and inference run on individual images inside the GUI.  The scoring
logic (Gaussian/Mahalanobis for PaDiM, coreset memory bank + nearest neighbour re-weighting for
PatchCore) is Anomalib's implementation; IDD Lab only provides preprocessing and plumbing.
"""

from __future__ import annotations

import numpy as np

from core.algorithms.base import AnomalyMethod, MethodResult, ParamSpec
from core.algorithms.torch_common import (anomalib_version, configure_model_cache, device, have,
                                          to_tensor, upsample_map)


class _AnomalibMemoryMethod(AnomalyMethod):
    requires_fit = True
    supports_gpu = True
    cacheable = True
    batch_size = 8

    def save_state(self, path):
        import torch
        torch.save({"state": self.model.state_dict(), "fit_info": self.fit_info, "params": self.params}, path)

    def load_state(self, path):
        import torch
        configure_model_cache()
        self.device = device()
        ck = torch.load(path, map_location=self.device, weights_only=False)
        self.model = self._build().to(self.device)
        self.model.load_state_dict(ck["state"])
        self.model.eval()
        self.fit_info = {**ck["fit_info"], "loaded_from_cache": True}
        self.fitted = True

    @classmethod
    def availability(cls):
        return have("torch", "anomalib", "timm")

    def _build(self):
        raise NotImplementedError

    def _finalize(self, model):
        raise NotImplementedError

    def _size(self) -> tuple[int, int]:
        hw = self.fit_info.get("input_hw")
        if hw:
            return int(hw[0]), int(hw[1])
        s = int(self.params["image_size"])
        return s, s

    def _input_hw(self, image) -> tuple[int, int]:
        """Network input size: square ``image_size``, or — with ``keep_aspect`` — shorter side =
        ``image_size`` and longer side scaled by the image aspect ratio (multiple of 32, ≤ 4×)."""
        s = int(self.params["image_size"])
        if not self.params.get("keep_aspect"):
            return s, s
        h, w = image.shape[:2]
        r = min(4.0, max(h, w) / min(h, w))
        long = int(round(s * r / 32)) * 32
        return (s, long) if w >= h else (long, s)

    def fit(self, images, progress=None, cancel=None):
        import torch
        configure_model_cache()
        self.device = device()
        if progress:
            progress(0.02, "loading pretrained backbone (cached in checkpoints/)…")
        model = self._build().to(self.device)
        self.fit_info = {"input_hw": list(self._input_hw(images[0]))}
        model.train()            # anomalib torch models collect embeddings in training mode
        model.feature_extractor.eval()
        n = len(images)
        with torch.no_grad():
            for i in range(0, n, self.batch_size):
                self._check_cancel(cancel)
                x = to_tensor(images[i:i + self.batch_size], self._size()).to(self.device)
                model(x)
                if progress:
                    progress(0.1 + 0.6 * min(1.0, (i + self.batch_size) / n), f"extracting features {min(n, i + self.batch_size)}/{n}")
        if progress:
            progress(0.75, "building normality model…")
        hw = self.fit_info["input_hw"]
        self._finalize(model)
        self.fit_info["input_hw"] = hw
        model.eval()
        self.model = model
        self.fitted = True
        if progress:
            progress(1.0, "done")

    def _predict(self, image, reference):
        import torch
        with torch.no_grad():
            x = to_tensor([image], self._size()).to(self.device)
            out = self.model(x)
        amap = out.anomaly_map.squeeze().detach().float().cpu().numpy()
        score = float(out.pred_score.squeeze().detach().float().cpu())
        return MethodResult(self.method_id, upsample_map(amap, image.shape[:2]), score, device=self.device,
                            score_semantics=self.semantics)


class PadimMethod(_AnomalibMemoryMethod):
    method_id = "padim"
    display_name = "PaDiM"
    semantics = ("map: Mahalanobis distance of each patch embedding to the Gaussian fitted at that "
                 "location (Gaussian-smoothed); score: max of the map")

    @classmethod
    def param_specs(cls):
        return [
            ParamSpec("backbone", "choice", "resnet18", "Backbone (ImageNet-pretrained)",
                      choices=["resnet18", "wide_resnet50_2"], affects_fit=True),
            ParamSpec("image_size", "int", 256, "Input size (px)", 128, 512, 32, affects_fit=True),
            ParamSpec("keep_aspect", "bool", False, "Keep aspect ratio", affects_fit=True, key_omit_default=True,
                      help="Shorter side = input size, longer side scaled with the image (≤ 4×). "
                           "Avoids squashing wide images such as MVTec AD 2 sheet_metal (4224×1056)."),
        ]

    def model_version(self):
        return f"Anomalib {anomalib_version()} PadimModel ({self.params['backbone']}, layers 1-3)"

    def _build(self):
        from anomalib.models.image.padim.torch_model import PadimModel
        return PadimModel(backbone=self.params["backbone"], layers=["layer1", "layer2", "layer3"], pre_trained=True)

    def _finalize(self, model):
        model.fit()
        self.fit_info = {"n_features": int(model.n_features)}


class PatchcoreMethod(_AnomalibMemoryMethod):
    method_id = "patchcore"
    display_name = "PatchCore"
    semantics = ("map: Euclidean distance of each patch to its nearest neighbour in the coreset memory "
                 "bank (Gaussian-smoothed); score: re-weighted max patch distance (Roth et al., Eq. 7)")

    @classmethod
    def param_specs(cls):
        return [
            ParamSpec("backbone", "choice", "wide_resnet50_2", "Backbone (ImageNet-pretrained)",
                      choices=["wide_resnet50_2", "resnet18"], affects_fit=True),
            ParamSpec("coreset_ratio", "float", 0.1, "Coreset sampling ratio", 0.01, 1.0, 0.01, affects_fit=True,
                      help="Fraction of patch features kept in the memory bank (k-center greedy)."),
            ParamSpec("num_neighbors", "int", 9, "Neighbours for score re-weighting", 1, 50, 1, affects_fit=True),
            ParamSpec("image_size", "int", 256, "Input size (px)", 128, 512, 32, affects_fit=True),
            ParamSpec("keep_aspect", "bool", False, "Keep aspect ratio", affects_fit=True, key_omit_default=True,
                      help="Shorter side = input size, longer side scaled with the image (≤ 4×). "
                           "Avoids squashing wide images such as MVTec AD 2 sheet_metal (4224×1056)."),
        ]

    def model_version(self):
        return f"Anomalib {anomalib_version()} PatchcoreModel ({self.params['backbone']}, layer2+layer3)"

    def _build(self):
        from anomalib.models.image.patchcore.torch_model import PatchcoreModel
        return PatchcoreModel(layers=["layer2", "layer3"], backbone=self.params["backbone"], pre_trained=True,
                              num_neighbors=int(self.params["num_neighbors"]))

    def _finalize(self, model):
        n_all = sum(e.shape[0] for e in model.embedding_store)
        model.subsample_embedding(float(self.params["coreset_ratio"]))
        self.fit_info = {"patch_features": int(n_all), "memory_bank": int(model.memory_bank.shape[0])}
