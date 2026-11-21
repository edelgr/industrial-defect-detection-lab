# Redistribution policy

IDD Lab distributes **source code, original educational content, bibliographic metadata and tools**.
It never distributes research datasets, pretrained weights, or models fitted on third-party data.
Users obtain those from the original source, under the original terms, with an explicit action.

## What a public repository / Student release may contain

| Included | Condition |
|---|---|
| `app/`, `core/`, `scripts/`, `tests/`, `*.ps1`, `pyproject.toml`, `requirements*/` | after the owner has chosen the project licence (see LICENSE_AUDIT §1) |
| `education/` (texts, YAML, generated SVG diagrams), `docs/` (guides, exercises, audits) | idem; `docs/` files with local paths are redacted in the staged copy |
| `research_catalog/`, `CITATIONS.bib`, `RESEARCH_REFERENCES.md` | bibliographic metadata; links point to the original sources |
| `docs/screenshots/` | synthetic data, catalog content and stored metric plots (no dataset images) |
| `configs/workshop_verified.json`, `docs/LOCAL_RESULTS*.md` | **conditional**: derived evaluation results; attribution to the datasets; owner decision for MulSen-AD-derived entries |
| `data/README.md`, `checkpoints/README.md`, `experiments/README.md` | placeholders only |

## What is never included

- Any file under `data/` (MVTec AD/AD 2, MulSen-AD, VisA, …) or generated synthetic images
  (users generate them locally with a fixed seed).
- Any checkpoint or weight file (`*.pt`, `*.pth`, `*.ckpt`, `*.safetensors`, `*.onnx`, `*.bin`):
  pretrained backbones, EfficientAD assets, the instructor's fitted / Workshop Verified checkpoints,
  quarantined artefacts.
- Experiment databases (`*.sqlite`), caches (`cache/`, screenshots of real data), virtual environments,
  archives, `configs/user_settings.json`.
- Personal information (local user names, home paths); secrets.

## How it is enforced

1. `.gitignore` keeps local assets out of Git.
2. `python -m scripts.build_release --out <new dir outside the repo>` stages a release **from
   `git archive HEAD`**, re-applies the exclusion list even to tracked files, redacts the local user name
   and home path in the staged text files, re-scans the result (forbidden extensions, files > 5 MB, user
   name / home path) and writes `RELEASE_MANIFEST.json` (SHA-256 of every file, source commit, excluded
   files, redactions, items needing review). It never modifies the repository and never uploads anything.
3. `python -m scripts.check_docs` and the CI workflow check links and commands; CI downloads no datasets
   and no weights (`IDD_OFFLINE_WEIGHTS=1`).

## Downloads performed by users (never by IDD Lab automatically)

| Resource | How | Shown to the user before download |
|---|---|---|
| Pretrained backbones | Model Manager → *Download pretrained weights*, `prepare_models.ps1`, or the Lab's confirmation dialog when fitting | source URL, size, licence |
| MVTec AD 2 / MVTec AD / VisA / … | official websites (Atlas → *Research Sources & Downloads*), then *Settings → Dataset Manager → Import* | licence and access route in the Atlas |
| MulSen-AD | official Hugging Face route; selective category fetch requires explicit confirmation | licence *not stated* warning |
