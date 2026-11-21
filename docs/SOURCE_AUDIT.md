# Source Audit

Generated 2026-11-20T23:53:03 by `scripts/verify_research_links.py`.

A `blocked` status (HTTP 401/403/405/418/429/503) means the site refuses automated clients; it is **not** evidence that the URL is invalid. Such links are verified manually.

## Summary

**manual**: 2, **ok**: 109, **redirect**: 41

## Links

| Status | HTTP | URL | Final URL | Used by |
|---|---|---|---|---|
| manual | 403 | https://doi.org/10.1080/2151237X.2007.10129236 | https://www.tandfonline.com/doi/full/10.1080/2151237X.2007.10129236 | publication:bradley2007.paper_url, publication:bradley2007.doi |
| manual | 403 | https://doi.org/10.1126/science.1127647 | https://www.science.org/doi/10.1126/science.1127647 | publication:hinton2006.paper_url, publication:hinton2006.doi |
| ok | 200 | https://amazon-visual-anomaly.s3.us-west-2.amazonaws.com/VisA_20220922.tar |  | dataset:D03.download_url |
| ok | 200 | https://anomalib.readthedocs.io/en/latest/markdown/guides/reference/models/image/anomaly_dino.html |  | algorithm:anomalydino.docs_url |
| ok | 200 | https://anomalib.readthedocs.io/en/latest/markdown/guides/reference/models/image/dinomaly.html |  | algorithm:dinomaly.docs_url |
| ok | 200 | https://anomalib.readthedocs.io/en/latest/markdown/guides/reference/models/image/efficient_ad.html |  | algorithm:efficientad.docs_url |
| ok | 200 | https://anomalib.readthedocs.io/en/latest/markdown/guides/reference/models/image/index.html |  | algorithm:draem.docs_url |
| ok | 200 | https://anomalib.readthedocs.io/en/latest/markdown/guides/reference/models/image/padim.html |  | algorithm:padim.docs_url |
| ok | 200 | https://anomalib.readthedocs.io/en/latest/markdown/guides/reference/models/image/patchcore.html |  | algorithm:patchcore.docs_url |
| ok | 200 | https://anomalib.readthedocs.io/en/latest/markdown/guides/reference/models/image/winclip.html |  | algorithm:winclip.docs_url |
| ok | 200 | https://arxiv.org/abs/1505.04597 |  | algorithm:unet.paper_url, publication:ronneberger2015unet.paper_url |
| ok | 200 | https://arxiv.org/abs/1801.09847 |  | publication:zhou2018open3d.paper_url |
| ok | 200 | https://arxiv.org/abs/1807.02011 |  | algorithm:conv_autoencoder.paper_url, publication:bergmann2019ssimae.paper_url |
| ok | 200 | https://arxiv.org/abs/2011.08785 |  | algorithm:padim.paper_url, publication:defard2021padim.paper_url |
| ok | 200 | https://arxiv.org/abs/2202.08341 |  | publication:akcay2022anomalib.paper_url |
| ok | 200 | https://arxiv.org/abs/2207.14315 |  | dataset:D03.paper_url, publication:zou2022visa.paper_url |
| ok | 200 | https://arxiv.org/abs/2503.21622 |  | dataset:D05.paper_url |
| ok | 200 | https://arxiv.org/abs/2507.07838 |  | dataset:D08.paper_url |
| ok | 200 | https://arxiv.org/abs/2508.10104 |  | algorithm:dino_features.paper_url, publication:simeoni2025dinov3.paper_url |
| ok | 200 | https://arxiv.org/pdf/1505.04597 |  | algorithm:unet.paper_pdf_url, publication:ronneberger2015unet.pdf_url |
| ok | 200 | https://arxiv.org/pdf/1801.09847 |  | publication:zhou2018open3d.pdf_url |
| ok | 200 | https://arxiv.org/pdf/2011.08785 |  | algorithm:padim.paper_pdf_url, publication:defard2021padim.pdf_url |
| ok | 200 | https://arxiv.org/pdf/2202.08341 |  | publication:akcay2022anomalib.pdf_url |
| ok | 200 | https://arxiv.org/pdf/2207.14315 |  | dataset:D03.paper_pdf_url, publication:zou2022visa.pdf_url |
| ok | 200 | https://arxiv.org/pdf/2304.07193 |  | publication:oquab2023dinov2.pdf_url |
| ok | 200 | https://arxiv.org/pdf/2507.07838 |  | dataset:D08.paper_pdf_url, publication:mchard2026adam3d.pdf_url |
| ok | 200 | https://arxiv.org/pdf/2508.10104 |  | algorithm:dino_features.paper_pdf_url, publication:simeoni2025dinov3.pdf_url |
| ok | 200 | https://creativecommons.org/licenses/by-nc-sa/4.0/ |  | dataset:D01.license_url, dataset:D02.license_url, dataset:D05.license_url, dataset:D07.license_url, dataset:D08.license_url |
| ok | 200 | https://creativecommons.org/licenses/by/4.0/ |  | dataset:D03.license_url, dataset:D04.license_url |
| ok | 200 | https://docs.ultralytics.com/ |  | algorithm:yolo.docs_url |
| ok | 200 | https://drive.google.com/file/d/1oM4qjhlIMsQc_wiFIFIVBvuuR8nyk2k0/view |  | dataset:D04.download_url |
| ok | 200 | https://dspace.isical.ac.in/items/dabb573e-3ce2-4bef-add0-04efa29d0317/full |  | publication:mahalanobis1936.paper_url |
| ok | 200 | https://github.com/KaimingHe/deep-residual-networks |  | algorithm:cnn_classification.official_repository_url, publication:he2016resnet.code_url |
| ok | 200 | https://github.com/M-3LAB/Real3D-AD |  | dataset:D04.official_website, dataset:D04.official_repository_url, dataset:D04.implementation_repository_url, algorithm:reg3dad.official_repository_url, publication:liu2023real3dad.code_url |
| ok | 200 | https://github.com/MapleBoat/MulSen-AD |  | dataset:D06.official_repository_url, dataset:D06.implementation_repository_url, publication:li2025mulsenad.code_url |
| ok | 200 | https://github.com/VitjanZ/DRAEM |  | algorithm:draem.official_repository_url, publication:zavrtanik2021draem.code_url |
| ok | 200 | https://github.com/amazon-science/patchcore-inspection |  | algorithm:patchcore.official_repository_url, publication:roth2022patchcore.code_url |
| ok | 200 | https://github.com/amazon-science/spot-diff |  | dataset:D03.official_repository_url, dataset:D03.implementation_repository_url, publication:zou2022visa.code_url |
| ok | 200 | https://github.com/dammsi/AnomalyDINO |  | algorithm:anomalydino.official_repository_url, publication:damm2025anomalydino.code_url |
| ok | 200 | https://github.com/facebookresearch/dinov2 |  | algorithm:dino_features.implementation_repository_url, publication:oquab2023dinov2.code_url |
| ok | 200 | https://github.com/facebookresearch/dinov3 |  | algorithm:dino_features.official_repository_url, algorithm:dino_features.docs_url, publication:simeoni2025dinov3.code_url |
| ok | 200 | https://github.com/guojiajeremy/Dinomaly |  | algorithm:dinomaly.official_repository_url, publication:guo2025dinomaly.code_url |
| ok | 200 | https://github.com/isl-org/Open3D |  | algorithm:pc_deviation.implementation_repository_url, publication:zhou2018open3d.code_url |
| ok | 200 | https://github.com/mala-lab/WinCLIP |  | publication:jeong2023winclip.code_url |
| ok | 200 | https://github.com/milesial/Pytorch-UNet |  | algorithm:unet.implementation_repository_url |
| ok | 200 | https://github.com/nomewang/M3DM |  | algorithm:m3dm.official_repository_url, publication:wang2023m3dm.code_url |
| ok | 200 | https://github.com/open-edge-platform/anomalib |  | dataset:D08.implementation_repository_url, algorithm:draem.implementation_repository_url, algorithm:padim.implementation_repository_url, algorithm:patchcore.implementation_repository_url, algorithm:efficientad.implementation_repository_url, algorithm:anomalydino.implementation_repository_url, algorithm:winclip.implementation_repository_url, algorithm:dinomaly.implementation_repository_url, publication:akcay2022anomalib.code_url |
| ok | 200 | https://github.com/openai/CLIP |  | publication:radford2021clip.code_url |
| ok | 200 | https://github.com/opencv/opencv |  | algorithm:otsu.implementation_repository_url, algorithm:adaptive_threshold.implementation_repository_url, algorithm:canny.implementation_repository_url, algorithm:gabor.implementation_repository_url, algorithm:morphology.implementation_repository_url, algorithm:reference_difference.implementation_repository_url |
| ok | 200 | https://github.com/pjreddie/darknet |  | algorithm:yolo.official_repository_url, publication:redmon2016yolo.code_url |
| ok | 200 | https://github.com/scikit-image/scikit-image |  | algorithm:ssim.implementation_repository_url |
| ok | 200 | https://github.com/taikiinoue45/PaDiM |  | publication:defard2021padim.code_url |
| ok | 200 | https://github.com/ultralytics/ultralytics |  | algorithm:yolo.implementation_repository_url |
| ok | 200 | https://huggingface.co/datasets/Real-IAD/Real-IAD_D3 |  | dataset:D07.download_url |
| ok | 200 | https://huggingface.co/datasets/orgjy314159/MulSen_AD |  | dataset:D06.download_url |
| ok | 200 | https://huggingface.co/datasets/pmchard/3D-ADAM |  | dataset:D08.official_website, dataset:D08.download_url |
| ok | 200 | https://huggingface.co/timm/resnet18.a1_in1k |  | algorithm:padim.weights_url |
| ok | 200 | https://huggingface.co/timm/vit_base_patch16_plus_clip_240.laion400m_e31 |  | algorithm:winclip.weights_url |
| ok | 200 | https://huggingface.co/timm/vit_small_patch14_dinov2.lvd142m |  | algorithm:anomalydino.weights_url |
| ok | 200 | https://huggingface.co/timm/wide_resnet50_2.racm_in1k |  | algorithm:patchcore.weights_url |
| ok | 200 | https://ieeexplore.ieee.org/abstract/document/11697283 |  | publication:mchard2026adam3d.paper_url |
| ok | 200 | https://link.springer.com/article/10.1007/s11263-022-01578-9 |  | dataset:D02.paper_url, publication:bergmann2022loco.paper_url |
| ok | 200 | https://link.springer.com/content/pdf/10.1007/s11263-022-01578-9.pdf |  | dataset:D02.paper_pdf_url, publication:bergmann2022loco.pdf_url |
| ok | 200 | https://link.springer.com/content/pdf/10.1007/s11263-026-02743-0.pdf |  | dataset:D05.paper_pdf_url, publication:hecklerkram2026mvtecad2.pdf_url |
| ok | 200 | https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/ |  | algorithm:unet.official_repository_url, publication:ronneberger2015unet.code_url |
| ok | 200 | https://mapleboat.github.io/MulSen_AD/ |  | dataset:D06.official_website |
| ok | 200 | https://openaccess.thecvf.com/content/CVPR2022/html/Roth_Towards_Total_Recall_in_Industrial_Anomaly_Detection_CVPR_2022_paper.html |  | algorithm:patchcore.paper_url, publication:roth2022patchcore.paper_url |
| ok | 200 | https://openaccess.thecvf.com/content/CVPR2022/papers/Roth_Towards_Total_Recall_in_Industrial_Anomaly_Detection_CVPR_2022_paper.pdf |  | algorithm:patchcore.paper_pdf_url, publication:roth2022patchcore.pdf_url |
| ok | 200 | https://openaccess.thecvf.com/content/CVPR2023/html/Jeong_WinCLIP_Zero-Few-Shot_Anomaly_Classification_and_Segmentation_CVPR_2023_paper.html |  | algorithm:winclip.paper_url, publication:jeong2023winclip.paper_url |
| ok | 200 | https://openaccess.thecvf.com/content/CVPR2023/html/Wang_Multimodal_Industrial_Anomaly_Detection_via_Hybrid_Fusion_CVPR_2023_paper.html |  | algorithm:multimodal_fusion.paper_url, algorithm:m3dm.paper_url, publication:wang2023m3dm.paper_url |
| ok | 200 | https://openaccess.thecvf.com/content/CVPR2023/papers/Jeong_WinCLIP_Zero-Few-Shot_Anomaly_Classification_and_Segmentation_CVPR_2023_paper.pdf |  | algorithm:winclip.paper_pdf_url, publication:jeong2023winclip.pdf_url |
| ok | 200 | https://openaccess.thecvf.com/content/CVPR2023/papers/Wang_Multimodal_Industrial_Anomaly_Detection_via_Hybrid_Fusion_CVPR_2023_paper.pdf |  | algorithm:m3dm.paper_pdf_url, publication:wang2023m3dm.pdf_url |
| ok | 200 | https://openaccess.thecvf.com/content/CVPR2025/html/Guo_Dinomaly_The_Less_Is_More_Philosophy_in_Multi-Class_Unsupervised_Anomaly_CVPR_2025_paper.html |  | algorithm:dinomaly.paper_url, publication:guo2025dinomaly.paper_url |
| ok | 200 | https://openaccess.thecvf.com/content/CVPR2025/html/Li_Multi-Sensor_Object_Anomaly_Detection_Unifying_Appearance_Geometry_and_Internal_Properties_CVPR_2025_paper.html |  | dataset:D06.paper_url, publication:li2025mulsenad.paper_url |
| ok | 200 | https://openaccess.thecvf.com/content/CVPR2025/html/Zhu_Real-IAD_D3_A_Real-World_2DPseudo-3D3D_Dataset_for_Industrial_Anomaly_Detection_CVPR_2025_paper.html |  | dataset:D07.paper_url, publication:zhu2025realiadd3.paper_url |
| ok | 200 | https://openaccess.thecvf.com/content/CVPR2025/papers/Guo_Dinomaly_The_Less_Is_More_Philosophy_in_Multi-Class_Unsupervised_Anomaly_CVPR_2025_paper.pdf |  | algorithm:dinomaly.paper_pdf_url, publication:guo2025dinomaly.pdf_url |
| ok | 200 | https://openaccess.thecvf.com/content/CVPR2025/papers/Li_Multi-Sensor_Object_Anomaly_Detection_Unifying_Appearance_Geometry_and_Internal_Properties_CVPR_2025_paper.pdf |  | dataset:D06.paper_pdf_url, publication:li2025mulsenad.pdf_url |
| ok | 200 | https://openaccess.thecvf.com/content/CVPR2025/papers/Zhu_Real-IAD_D3_A_Real-World_2DPseudo-3D3D_Dataset_for_Industrial_Anomaly_Detection_CVPR_2025_paper.pdf |  | dataset:D07.paper_pdf_url, publication:zhu2025realiadd3.pdf_url |
| ok | 200 | https://openaccess.thecvf.com/content/ICCV2021/html/Zavrtanik_DRAEM_-_A_Discriminatively_Trained_Reconstruction_Embedding_for_Surface_Anomaly_ICCV_2021_paper.html |  | algorithm:draem.paper_url, publication:zavrtanik2021draem.paper_url |
| ok | 200 | https://openaccess.thecvf.com/content/ICCV2021/papers/Zavrtanik_DRAEM_-_A_Discriminatively_Trained_Reconstruction_Embedding_for_Surface_Anomaly_ICCV_2021_paper.pdf |  | algorithm:draem.paper_pdf_url, publication:zavrtanik2021draem.pdf_url |
| ok | 200 | https://openaccess.thecvf.com/content/WACV2024/html/Batzner_EfficientAD_Accurate_Visual_Anomaly_Detection_at_Millisecond-Level_Latencies_WACV_2024_paper.html |  | algorithm:efficientad.paper_url, publication:batzner2024efficientad.paper_url |
| ok | 200 | https://openaccess.thecvf.com/content/WACV2024/papers/Batzner_EfficientAD_Accurate_Visual_Anomaly_Detection_at_Millisecond-Level_Latencies_WACV_2024_paper.pdf |  | algorithm:efficientad.paper_pdf_url, publication:batzner2024efficientad.pdf_url |
| ok | 200 | https://openaccess.thecvf.com/content/WACV2025/html/Damm_AnomalyDINO_Boosting_Patch-Based_Few-Shot_Anomaly_Detection_with_DINOv2_WACV_2025_paper.html |  | algorithm:anomalydino.paper_url, publication:damm2025anomalydino.paper_url |
| ok | 200 | https://openaccess.thecvf.com/content/WACV2025/papers/Damm_AnomalyDINO_Boosting_Patch-Based_Few-Shot_Anomaly_Detection_with_DINOv2_WACV_2025_paper.pdf |  | algorithm:anomalydino.paper_pdf_url, publication:damm2025anomalydino.pdf_url |
| ok | 200 | https://openaccess.thecvf.com/content_CVPR_2019/html/Bergmann_MVTec_AD_--_A_Comprehensive_Real-World_Dataset_for_Unsupervised_Anomaly_CVPR_2019_paper.html |  | dataset:D01.paper_url, publication:bergmann2019mvtecad.paper_url |
| ok | 200 | https://openaccess.thecvf.com/content_CVPR_2019/papers/Bergmann_MVTec_AD_--_A_Comprehensive_Real-World_Dataset_for_Unsupervised_Anomaly_CVPR_2019_paper.pdf |  | dataset:D01.paper_pdf_url, publication:bergmann2019mvtecad.pdf_url |
| ok | 200 | https://openaccess.thecvf.com/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html |  | algorithm:cnn_classification.paper_url, publication:he2016resnet.paper_url |
| ok | 200 | https://openaccess.thecvf.com/content_cvpr_2016/html/Redmon_You_Only_Look_CVPR_2016_paper.html |  | algorithm:yolo.paper_url, publication:redmon2016yolo.paper_url |
| ok | 200 | https://openaccess.thecvf.com/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf |  | algorithm:cnn_classification.paper_pdf_url, publication:he2016resnet.pdf_url |
| ok | 200 | https://openaccess.thecvf.com/content_cvpr_2016/papers/Redmon_You_Only_Look_CVPR_2016_paper.pdf |  | algorithm:yolo.paper_pdf_url, publication:redmon2016yolo.pdf_url |
| ok | 200 | https://openlibrary.org/search?q=image+analysis+and+mathematical+morphology+serra |  | publication:serra1982.paper_url |
| ok | 200 | https://people.idsia.ch/~ciresan/data/icann2011.pdf |  | publication:masci2011.pdf_url |
| ok | 200 | https://proceedings.mlr.press/v139/radford21a.html |  | publication:radford2021clip.paper_url |
| ok | 200 | https://proceedings.mlr.press/v139/radford21a/radford21a.pdf |  | publication:radford2021clip.pdf_url |
| ok | 200 | https://proceedings.neurips.cc/paper_files/paper/2023/file/611b896d447df43c898062358df4c114-Paper-Datasets_and_Benchmarks.pdf |  | dataset:D04.paper_pdf_url, algorithm:reg3dad.paper_pdf_url, publication:liu2023real3dad.pdf_url |
| ok | 200 | https://proceedings.neurips.cc/paper_files/paper/2023/hash/611b896d447df43c898062358df4c114-Abstract-Datasets_and_Benchmarks.html |  | dataset:D04.paper_url, algorithm:reg3dad.paper_url, publication:liu2023real3dad.paper_url |
| ok | 200 | https://realiad4ad.github.io/Real-IAD_D3/ |  | dataset:D07.official_website |
| ok | 200 | https://registry.opendata.aws/visa/ |  | dataset:D03.official_website |
| ok | 200 | https://scikit-image.org/docs/stable/api/skimage.metrics.html#skimage.metrics.structural_similarity |  | algorithm:ssim.docs_url |
| ok | 200 | https://www.cs.toronto.edu/~hinton/absps/science.pdf |  | publication:hinton2006.pdf_url |
| ok | 200 | https://www.ece.uwaterloo.ca/~z70wang/publications/ssim.html |  | algorithm:ssim.paper_url, publication:wang2004ssim.paper_url |
| ok | 200 | https://www.ece.uwaterloo.ca/~z70wang/publications/ssim.pdf |  | algorithm:ssim.paper_pdf_url, publication:wang2004ssim.pdf_url |
| ok | 200 | https://www.ece.uwaterloo.ca/~z70wang/research/ssim/ |  | algorithm:ssim.official_repository_url, publication:wang2004ssim.code_url |
| ok | 200 | https://www.mvtec.com/research-teaching/datasets/mvtec-ad |  | dataset:D01.official_website |
| ok | 200 | https://www.mvtec.com/research-teaching/datasets/mvtec-ad-2 |  | dataset:D05.official_website |
| ok | 200 | https://www.mvtec.com/research-teaching/datasets/mvtec-ad-2/downloads |  | dataset:D05.download_url |
| ok | 200 | https://www.mvtec.com/research-teaching/datasets/mvtec-ad/downloads |  | dataset:D01.download_url |
| ok | 200 | https://www.mvtec.com/research-teaching/datasets/mvtec-loco-ad |  | dataset:D02.official_website |
| ok | 200 | https://www.mvtec.com/research-teaching/datasets/mvtec-loco-ad/downloads |  | dataset:D02.download_url |
| ok | 200 | https://www.open3d.org/docs/release/tutorial/pipelines/icp_registration.html |  | algorithm:pc_deviation.docs_url |
| ok | 200 | https://www.scitepress.org/Papers/2019/73645/73645.pdf |  | algorithm:conv_autoencoder.paper_pdf_url, publication:bergmann2019ssimae.pdf_url |
| redirect | 200 | https://docs.opencv.org/4.x/ | https://docs.opencv.org/4.13.0/ | algorithm:reference_difference.docs_url |
| redirect | 200 | https://docs.opencv.org/4.x/d4/d86/group__imgproc__filter.html#ga67493776e3ad1a3df63883829375201f | https://docs.opencv.org/4.13.0/d4/d86/group__imgproc__filter.html#ga67493776e3ad1a3df63883829375201f | algorithm:morphology.docs_url |
| redirect | 200 | https://docs.opencv.org/4.x/d4/d86/group__imgproc__filter.html#gae84c92d248183bd92fa713ce51cc3599 | https://docs.opencv.org/4.13.0/d4/d86/group__imgproc__filter.html#gae84c92d248183bd92fa713ce51cc3599 | algorithm:gabor.docs_url |
| redirect | 200 | https://docs.opencv.org/4.x/d7/d1b/group__imgproc__misc.html#ga72b913f352e4a1b1b397736707afcde3 | https://docs.opencv.org/4.13.0/d7/d1b/group__imgproc__misc.html#ga72b913f352e4a1b1b397736707afcde3 | algorithm:adaptive_threshold.docs_url |
| redirect | 200 | https://docs.opencv.org/4.x/d7/d1b/group__imgproc__misc.html#gae8a4a146d1ca78c626a53577199e9c57 | https://docs.opencv.org/4.13.0/d7/d1b/group__imgproc__misc.html#gae8a4a146d1ca78c626a53577199e9c57 | algorithm:otsu.docs_url |
| redirect | 200 | https://docs.opencv.org/4.x/dd/d1a/group__imgproc__feature.html#ga04723e007ed888ddf11d9ba04e2232de | https://docs.opencv.org/4.13.0/dd/d1a/group__imgproc__feature.html#ga04723e007ed888ddf11d9ba04e2232de | algorithm:canny.docs_url |
| redirect | 200 | https://doi.org/10.1007/978-3-030-68799-1_35 | https://link.springer.com/10.1007/978-3-030-68799-1_35 | algorithm:padim.doi, publication:defard2021padim.doi |
| redirect | 200 | https://doi.org/10.1007/978-3-031-20056-4_23 | https://link.springer.com/chapter/10.1007/978-3-031-20056-4_23 | dataset:D03.doi, publication:zou2022visa.doi |
| redirect | 200 | https://doi.org/10.1007/978-3-319-24574-4_28 | https://link.springer.com/10.1007/978-3-319-24574-4_28 | algorithm:unet.doi, publication:ronneberger2015unet.doi |
| redirect | 200 | https://doi.org/10.1007/978-3-642-21735-7_7 | https://link.springer.com/10.1007/978-3-642-21735-7_7 | publication:masci2011.paper_url, publication:masci2011.doi |
| redirect | 200 | https://doi.org/10.1007/s11263-020-01400-4 | https://link.springer.com/10.1007/s11263-020-01400-4 | publication:bergmann2021mvtecad_ijcv.paper_url, publication:bergmann2021mvtecad_ijcv.doi |
| redirect | 200 | https://doi.org/10.1007/s11263-022-01578-9 | https://link.springer.com/10.1007/s11263-022-01578-9 | dataset:D02.doi, publication:bergmann2022loco.doi |
| redirect | 200 | https://doi.org/10.1007/s11263-026-02743-0 | https://link.springer.com/10.1007/s11263-026-02743-0 | dataset:D05.doi, publication:hecklerkram2026mvtecad2.paper_url, publication:hecklerkram2026mvtecad2.doi |
| redirect | 200 | https://doi.org/10.1016/0031-3203(91)90143-S | https://linkinghub.elsevier.com/retrieve/pii/003132039190143S | algorithm:gabor.paper_url, algorithm:gabor.doi, publication:jain1991.paper_url, publication:jain1991.doi |
| redirect | 200 | https://doi.org/10.1016/0262-8856(92)90066-C | https://linkinghub.elsevier.com/retrieve/pii/026288569290066C | publication:chen1992pointplane.paper_url, publication:chen1992pointplane.doi |
| redirect | 200 | https://doi.org/10.1016/S0031-3203(99)00055-2 | https://linkinghub.elsevier.com/retrieve/pii/S0031320399000552 | algorithm:adaptive_threshold.paper_url, algorithm:adaptive_threshold.doi, publication:sauvola2000.paper_url, publication:sauvola2000.doi |
| redirect | 202 | https://doi.org/10.1109/34.121791 | https://ieeexplore.ieee.org/document/121791/ | algorithm:pc_deviation.paper_url, algorithm:pc_deviation.doi, publication:besl1992icp.paper_url, publication:besl1992icp.doi |
| redirect | 202 | https://doi.org/10.1109/CVPR.2016.90 | https://ieeexplore.ieee.org/document/7780459/ | algorithm:cnn_classification.doi, publication:he2016resnet.doi |
| redirect | 202 | https://doi.org/10.1109/CVPR.2016.91 | https://ieeexplore.ieee.org/document/7780460/ | algorithm:yolo.doi, publication:redmon2016yolo.doi |
| redirect | 202 | https://doi.org/10.1109/CVPR.2019.00982 | https://ieeexplore.ieee.org/document/8954181/ | dataset:D01.doi, publication:bergmann2019mvtecad.doi |
| redirect | 202 | https://doi.org/10.1109/CVPR52688.2022.01392 | https://ieeexplore.ieee.org/document/9879738/ | algorithm:patchcore.doi, publication:roth2022patchcore.doi |
| redirect | 202 | https://doi.org/10.1109/CVPR52729.2023.00776 | https://ieeexplore.ieee.org/document/10203503/ | algorithm:multimodal_fusion.doi, algorithm:m3dm.doi, publication:wang2023m3dm.doi |
| redirect | 202 | https://doi.org/10.1109/CVPR52729.2023.01878 | https://ieeexplore.ieee.org/document/10204096/ | algorithm:winclip.doi, publication:jeong2023winclip.doi |
| redirect | 202 | https://doi.org/10.1109/CVPR52734.2025.00933 | https://ieeexplore.ieee.org/document/11094456/ | dataset:D06.doi, publication:li2025mulsenad.doi |
| redirect | 202 | https://doi.org/10.1109/CVPR52734.2025.01417 | https://ieeexplore.ieee.org/document/11093434/ | dataset:D07.doi, publication:zhu2025realiadd3.doi |
| redirect | 202 | https://doi.org/10.1109/CVPR52734.2025.01900 | https://ieeexplore.ieee.org/document/11092978/ | algorithm:dinomaly.doi, publication:guo2025dinomaly.doi |
| redirect | 202 | https://doi.org/10.1109/ICCV48922.2021.00822 | https://ieeexplore.ieee.org/document/9710329/ | algorithm:draem.doi, publication:zavrtanik2021draem.doi |
| redirect | 202 | https://doi.org/10.1109/ICRA57385.2026.11697283 | https://ieeexplore.ieee.org/document/11697283/ | dataset:D08.doi, publication:mchard2026adam3d.doi |
| redirect | 200 | https://doi.org/10.1109/ROBOT.2009.5152473 | https://ieeexplore.ieee.org/document/5152473/ | publication:rusu2009fpfh.paper_url, publication:rusu2009fpfh.doi |
| redirect | 202 | https://doi.org/10.1109/TIP.2003.819861 | https://ieeexplore.ieee.org/document/1284395/ | algorithm:ssim.doi, publication:wang2004ssim.doi |
| redirect | 202 | https://doi.org/10.1109/TPAMI.1986.4767851 | https://ieeexplore.ieee.org/document/4767851 | algorithm:canny.paper_url, algorithm:canny.doi, publication:canny1986.paper_url, publication:canny1986.doi |
| redirect | 202 | https://doi.org/10.1109/TPAMI.1987.4767941 | https://ieeexplore.ieee.org/document/4767941 | algorithm:morphology.paper_url, publication:haralick1987.paper_url, publication:haralick1987.doi |
| redirect | 202 | https://doi.org/10.1109/TPAMI.2008.113 | https://ieeexplore.ieee.org/document/4515873/ | algorithm:reference_difference.paper_url, publication:evangelidis2008ecc.paper_url, publication:evangelidis2008ecc.doi |
| redirect | 202 | https://doi.org/10.1109/TSMC.1979.4310076 | https://ieeexplore.ieee.org/document/4310076/ | algorithm:otsu.paper_url, algorithm:otsu.doi, publication:otsu1979.paper_url, publication:otsu1979.doi |
| redirect | 202 | https://doi.org/10.1109/WACV57701.2024.00020 | https://ieeexplore.ieee.org/document/10484326/ | algorithm:efficientad.doi, publication:batzner2024efficientad.doi |
| redirect | 202 | https://doi.org/10.1109/WACV61041.2025.00136 | https://ieeexplore.ieee.org/document/10944159/ | algorithm:anomalydino.doi, publication:damm2025anomalydino.doi |
| redirect | 200 | https://doi.org/10.1364/JOSAA.2.001160 | https://opg.optica.org/josaa/abstract.cfm?uri=josaa-2-7-1160 | publication:daugman1985.paper_url, publication:daugman1985.doi |
| redirect | 200 | https://doi.org/10.5220/0007364503720380 | https://www.scitepress.org/Link.aspx?doi=10.5220/0007364503720380 | algorithm:conv_autoencoder.doi, publication:bergmann2019ssimae.doi |
| redirect | 200 | https://doi.org/10.52202/075280-1324 | https://www.proceedings.com/075280-1324.html | dataset:D04.doi, algorithm:reg3dad.doi, publication:liu2023real3dad.doi |
| redirect | 200 | https://github.com/open-edge-platform/anomalib/releases/download/efficientad_pretrained_weights/efficientad_pretrained_weights.zip | https://release-assets.githubusercontent.com/github-production-release-asset/423775360/d2a14009-95fa-4bac-aa8f-c380c948a298?… | algorithm:efficientad.weights_url |
| redirect | 200 | https://openreview.net/forum?id=a68SUt6zFt | https://openreview.net/challenge?redirect=%2Fforum%3Fid%3Da68SUt6zFt | publication:oquab2023dinov2.paper_url |

## Publication metadata (Crossref / arXiv title match)

| Publication | DOI | arXiv |
|---|---|---|
| akcay2022anomalib | — 1.0 | match |
| batzner2024efficientad | match 1.0 | match |
| bergmann2019mvtecad | match 1.0 | — |
| bergmann2019ssimae | match 1.0 | match |
| bergmann2021mvtecad_ijcv | match 1.0 | — |
| bergmann2022loco | match 1.0 | — |
| besl1992icp | match 1.0 | — |
| bradley2007 | match 1.0 | — |
| canny1986 | match 1.0 | — |
| chen1992pointplane | match 1.0 | — |
| damm2025anomalydino | match 1.0 | match |
| daugman1985 | match 1.0 | — |
| defard2021padim | match 1.0 | match |
| evangelidis2008ecc | match 1.0 | — |
| guo2025dinomaly | match 1.0 | match |
| haralick1987 | match 1.0 | — |
| he2016resnet | match 1.0 | match |
| hecklerkram2026mvtecad2 | match 1.0 | match |
| hinton2006 | match 1.0 | — |
| jain1991 | match 1.0 | — |
| jeong2023winclip | match 1.0 | match |
| li2025mulsenad | match 1.0 | match |
| liu2023real3dad | match 1.0 | match |
| mahalanobis1936 | —  | — |
| masci2011 | match 1.0 | — |
| mchard2026adam3d | match 1.0 | match |
| oquab2023dinov2 | — 1.0 | match |
| otsu1979 | match 1.0 | — |
| radford2021clip | — 1.0 | match |
| redmon2016yolo | match 1.0 | match |
| ronneberger2015unet | match 1.0 | match |
| roth2022patchcore | match 1.0 | match |
| rusu2009fpfh | match 1.0 | — |
| sauvola2000 | match 1.0 | — |
| serra1982 | —  | — |
| simeoni2025dinov3 | — 1.0 | match |
| wang2004ssim | match 1.0 | — |
| wang2023m3dm | match 1.0 | match |
| zavrtanik2021draem | match 1.0 | match |
| zhou2018open3d | — 1.0 | match |
| zhu2025realiadd3 | match 1.0 | match |
| zou2022visa | match 1.0 | match |

## Structural notes

- algorithm otsu: no official authors' repository (stated as such)
- algorithm adaptive_threshold: no official authors' repository (stated as such)
- algorithm canny: no official authors' repository (stated as such)
- algorithm gabor: no official authors' repository (stated as such)
- algorithm morphology: no official authors' repository (stated as such)
- algorithm reference_difference: no official authors' repository (stated as such)
- algorithm conv_autoencoder: no official authors' repository (stated as such)
- algorithm padim: no official authors' repository (stated as such)
- algorithm efficientad: no official authors' repository (stated as such)
- algorithm winclip: no official authors' repository (stated as such)
- algorithm pc_deviation: no official authors' repository (stated as such)
- algorithm multimodal_fusion: no official authors' repository (stated as such)
