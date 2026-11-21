# Dataset integrity validation

Generated 2026-11-21 07:01 by `python -m scripts.validate_dataset` (read-only; no file is modified). Checks: complete and unique indexing, image ↔ mask pairing by name, mask size equal to image size, binary non-empty masks, no masks on normal/private samples, no orphan masks, every image decodes, private splits unlabelled and excluded from evaluation, no byte-identical image shared between train / validation / test.

## D05 · fruit_jelly — ✅ OK

- Image files on disk: 890 · indexed samples: 890 · mask files: 60
- Splits: `test_private` unknown 255; `test_private_mixed` unknown 255; `test_public` bad 60, good 20; `train` good 263; `validation` good 37
- Labelled test used for evaluation: 80 (20 good / 60 bad)
- Image format per split: `test_private` 2100x1520x3 uint8 ×255; `test_private_mixed` 2100x1520x3 uint8 ×255; `test_public` 2100x1520x3 uint8 ×80; `train` 2100x1520x3 uint8 ×263; `validation` 2100x1520x3 uint8 ×37
- Defect area per anomalous image: min 0.0227 % · median 0.3959 % · max 5.8 %
- Lighting conditions: `test_private` unknown:regular 255; `test_private_mixed` unknown:mixed 255; `test_public` bad:overexposed 15, bad:regular 15, bad:shift_1 15, bad:underexposed 15, good:overexposed 5, good:regular 5, good:shift_1 5, good:underexposed 5; `train` good:regular 263; `validation` good:regular 37
- `test_public` distinct scenes: 5 good, 15 bad (each under several lighting conditions)

## D05 · sheet_metal — ✅ OK

- Image files on disk: 554 · indexed samples: 554 · mask files: 90
- Splits: `test_private` unknown 142; `test_private_mixed` unknown 142; `test_public` bad 90, good 24; `train` good 137; `validation` good 19
- Labelled test used for evaluation: 114 (24 good / 90 bad)
- Image format per split: `test_private` 4224x1056x1 uint8 ×142; `test_private_mixed` 4224x1056x1 uint8 ×142; `test_public` 4224x1056x1 uint8 ×114; `train` 4224x1056x1 uint8 ×137; `validation` 4224x1056x1 uint8 ×19
- Defect area per anomalous image: min 0.0117 % · median 0.2464 % · max 4.115 %
- Lighting conditions: `test_private` unknown:regular 142; `test_private_mixed` unknown:mixed 142; `test_public` bad:overexposed 15, bad:regular 15, bad:shift_1 15, bad:shift_2 15, bad:shift_3 15, bad:underexposed 15, good:overexposed 4, good:regular 4, good:shift_1 4, good:shift_2 4, good:shift_3 4, good:underexposed 4; `train` good:regular 137; `validation` good:regular 19
- `test_public` distinct scenes: 4 good, 15 bad (each under several lighting conditions)

