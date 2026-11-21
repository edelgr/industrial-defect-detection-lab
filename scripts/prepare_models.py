"""Download & cache pretrained backbones into checkpoints/ (run once while online).

  python -m scripts.prepare_models                 # patchcore, padim, anomalydino (≈ 0.4 GB)
  python -m scripts.prepare_models --with-winclip  # + OpenCLIP ViT-B-16-plus-240 (≈ 0.83 GB)

Each resource is < 1 GB.  Weights stay in the git-ignored checkpoints/ folder and are never
redistributed with the repository.  Licences: torchvision/timm ImageNet weights (see timm model
cards), DINOv2 (Apache-2.0), OpenCLIP LAION-400M checkpoint (see OpenCLIP model card).
"""

from __future__ import annotations

import argparse

import numpy as np

from core.algorithms import registry
from core.algorithms.torch_common import configure_model_cache


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--with-winclip", action="store_true")
    ap.add_argument("--with-efficientad", action="store_true",
                    help="download EfficientAD teacher weights (40 MB) + Imagenette2-320 (342 MB)")
    a = ap.parse_args(argv)
    configure_model_cache()
    if a.with_efficientad:
        from core.algorithms.teacher_student.efficientad import assets_present, prepare_assets
        prepare_assets(progress=lambda f, m: print(f"\r  efficientad assets {100 * f:5.1f}% {m[:60]}", end=""))
        print("\nEfficientAD assets present:", assets_present())
    dummy = [np.full((256, 256, 3), 128, np.uint8) for _ in range(2)]
    todo = [("padim", {}), ("patchcore", {}), ("anomalydino", {"k_shot": 1})]
    if a.with_winclip:
        todo.append(("winclip", {"k_shot": 0}))
    for mid, params in todo:
        print(f"== {mid}: instantiating (downloads weights if not cached) …", flush=True)
        m = registry.create(mid, **params)
        m.fit(dummy)
        r = m.predict(dummy[0])
        print(f"   ok — {m.model_version()} · device={r.device} · {r.latency_ms:.0f} ms")
    from core.system.readiness import weights_cached
    for mid, _ in todo:
        print(f"{mid:12s} weights cached: {weights_cached(mid)}")


if __name__ == "__main__":
    main()
