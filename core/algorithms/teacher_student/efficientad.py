"""Group E — EfficientAD (Batzner et al., WACV 2024) via Anomalib's ``EfficientAdModel``.

No official code exists; Anomalib's re-implementation (Apache-2.0) provides the networks and
losses. IDD Lab supplies a compact training loop that mirrors Anomalib's Lightning module:

* distilled PDN teacher weights from Anomalib's GitHub release (≈ 40 MB);
* ImageNet penalty images from **Imagenette2-320** (≈ 342 MB, fast.ai) instead of Anomalib's
  default full-size Imagenette (≈ 1.56 GB, would require approval) — resized ×2 and centre-cropped
  as in Anomalib, so only the source resolution differs;
* teacher channel mean/std on the training images; Adam (lr 1e-4, wd 1e-5), batch size 1,
  StepLR ×0.1 at 95 % of the steps;
* **short educational schedule** (default 10 000 steps instead of the paper's 70 000) — results are
  therefore below the paper's and are reported as local educational results;
* map-normalisation quantiles (q 0.9 / 0.995) on held-out normal images not used for training.
"""

from __future__ import annotations

import random

import numpy as np

from core import paths
from core.algorithms.base import AnomalyMethod, MethodResult, ParamSpec
from core.algorithms.torch_common import anomalib_version, configure_model_cache, device, have, upsample_map

TEACHER_URL = ("https://github.com/open-edge-platform/anomalib/releases/download/efficientad_pretrained_weights/"
               "efficientad_pretrained_weights.zip")
IMAGENETTE_URL = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz"
ROOT = paths.CHECKPOINTS / "efficientad"


def assets_present() -> bool:
    return (ROOT / "efficientad_pretrained_weights").is_dir() and (ROOT / "imagenette2-320").is_dir()


def prepare_assets(progress=None, cancel=None) -> None:
    """Download teacher weights (40 MB) and Imagenette2-320 (342 MB) — both < 1 GB."""
    from core.datasets.manager import download, extract
    ROOT.mkdir(parents=True, exist_ok=True)
    for url, name in ((TEACHER_URL, "efficientad_pretrained_weights.zip"), (IMAGENETTE_URL, "imagenette2-320.tgz")):
        target_dir = ROOT / name.split(".")[0]
        if target_dir.is_dir():
            continue
        arch = ROOT / name
        if not arch.exists():
            download(url, arch, progress=progress, cancel=cancel)
        extract(arch, ROOT, progress=progress, cancel=cancel)
        arch.unlink(missing_ok=True)


def _to01(images, size):
    import cv2
    import torch
    arr = np.stack([cv2.resize(im, (size, size), interpolation=cv2.INTER_AREA) for im in images]).astype(np.float32) / 255.0
    return torch.from_numpy(arr.transpose(0, 3, 1, 2).copy())   # EfficientAD normalises internally


class EfficientADMethod(AnomalyMethod):
    method_id = "efficientad"
    display_name = "EfficientAD"
    requires_fit = True
    supports_gpu = True

    @classmethod
    def param_specs(cls):
        return [
            ParamSpec("model_size", "choice", "small", "Model size", choices=["small", "medium"], affects_fit=True),
            ParamSpec("steps", "int", 10000, "Training steps (paper: 70 000)", 100, 70000, 100, affects_fit=True,
                      help="Short educational schedule by default; more steps = better accuracy, longer training."),
            ParamSpec("image_size", "int", 256, "Input size (px)", 128, 512, 32, affects_fit=True),
        ]

    @classmethod
    def availability(cls):
        ok, why = have("torch", "anomalib", "torchvision")
        if not ok:
            return ok, why
        if not assets_present():
            return True, "teacher weights + Imagenette2-320 (≈ 0.38 GB) are downloaded on first fit"
        return True, "ok"

    def model_version(self):
        return (f"Anomalib {anomalib_version()} EfficientAdModel ({self.params['model_size']}, "
                f"{self.params['steps']} steps, Imagenette2-320 penalty)")

    cacheable = True

    def _construct(self):
        import torch
        from anomalib.models.image.efficient_ad.torch_model import EfficientAdModel, EfficientAdModelSize
        configure_model_cache()
        self.device = dev = device()
        msize = EfficientAdModelSize.S if self.params["model_size"] == "small" else EfficientAdModelSize.M
        model = EfficientAdModel(teacher_out_channels=384, model_size=msize, padding=False, pad_maps=True).to(dev)
        tpath = ROOT / "efficientad_pretrained_weights" / f"pretrained_teacher_{self.params['model_size']}.pth"
        model.teacher.load_state_dict(torch.load(tpath, map_location=dev, weights_only=True))
        model.teacher.eval()
        for p in model.teacher.parameters():
            p.requires_grad_(False)
        return model

    def save_state(self, path):
        import torch
        torch.save({"state": self.model.state_dict(), "fit_info": self.fit_info}, path)

    def load_state(self, path):
        import torch
        ck = torch.load(path, map_location="cpu", weights_only=False)
        self.model = self._construct()
        self.model.load_state_dict(ck["state"])
        self.model.eval()
        self.fit_info = {**ck["fit_info"], "loaded_from_cache": True}
        self.fitted = True

    def fit(self, images, progress=None, cancel=None):
        import torch
        from torchvision.datasets import ImageFolder
        from torchvision.transforms import CenterCrop, Compose, RandomGrayscale, Resize, ToTensor
        if not assets_present():
            prepare_assets(progress=(lambda f, m: progress(0.05 * f, "downloading EfficientAD assets: " + m)) if progress else None,
                           cancel=cancel)
        model = self._construct()
        dev = self.device
        size = int(self.params["image_size"])

        rng = random.Random(0)
        imgs = list(images)
        n_q = max(2, len(imgs) // 10) if len(imgs) >= 6 else 0      # held-out normals for map quantiles
        q_imgs, train_imgs = imgs[:n_q], imgs[n_q:]
        X = _to01(train_imgs, size)

        with torch.no_grad():                                         # teacher channel statistics
            ys = [model.teacher(X[i:i + 8].to(dev)) for i in range(0, len(X), 8)]
            y = torch.cat(ys)
            model.mean_std.update({"mean": y.mean(dim=(0, 2, 3), keepdim=True),
                                   "std": y.std(dim=(0, 2, 3), keepdim=True)})

        tf = Compose([Resize((size * 2, size * 2)), RandomGrayscale(p=0.3), CenterCrop((size, size)), ToTensor()])
        inet = ImageFolder(str(ROOT / "imagenette2-320" / "train"), transform=tf)
        steps = int(self.params["steps"])
        opt = torch.optim.Adam(list(model.student.parameters()) + list(model.ae.parameters()), lr=1e-4, weight_decay=1e-5)
        sched = torch.optim.lr_scheduler.StepLR(opt, step_size=int(0.95 * steps), gamma=0.1)
        model.train()
        model.teacher.eval()
        losses = []
        for step in range(steps):
            if cancel and cancel():
                raise RuntimeError("training cancelled")
            x = X[rng.randrange(len(X))][None].to(dev)
            xi = inet[rng.randrange(len(inet))][0][None].to(dev)
            l_st, l_ae, l_stae = model(batch=x, batch_imagenet=xi)
            loss = l_st + l_ae + l_stae
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            losses.append(loss.detach().item())
            if progress and step % 50 == 0:
                progress(0.05 + 0.85 * step / steps, f"training step {step}/{steps} · loss {np.mean(losses[-50:]):.4f}")
        model.eval()
        if q_imgs:
            with torch.no_grad():
                st, ae = [], []
                for im in q_imgs:
                    a, b = model.get_maps(_to01([im], size).to(dev), normalize=False)
                    st.append(a.flatten())
                    ae.append(b.flatten())
                st, ae = torch.cat(st), torch.cat(ae)
                model.quantiles.update({"qa_st": torch.quantile(st, 0.9), "qb_st": torch.quantile(st, 0.995),
                                        "qa_ae": torch.quantile(ae, 0.9), "qb_ae": torch.quantile(ae, 0.995)})
        self.model = model
        self.fitted = True
        self.fit_info = {"train_images": len(train_imgs), "quantile_images": len(q_imgs), "steps": steps,
                         "final_loss": float(np.mean(losses[-100:]))}
        if progress:
            progress(1.0, "done")

    def _predict(self, image, reference):
        import torch
        size = int(self.params["image_size"])
        with torch.no_grad():
            out = self.model(_to01([image], size).to(self.device))
        amap = out.anomaly_map.squeeze().float().cpu().numpy()
        return MethodResult(self.method_id, upsample_map(amap, image.shape[:2]),
                            float(out.pred_score.squeeze().float().cpu()), device=self.device,
                            score_semantics="map: 0.5·(student–teacher) + 0.5·(student–autoencoder) squared "
                                            "differences, quantile-normalised on held-out normals; score: max")
