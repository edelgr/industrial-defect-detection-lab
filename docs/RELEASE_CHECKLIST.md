# Release checklist (first public GitHub release)

Nothing in this list is automated: each step needs the project owner. CI never publishes.

## A. Owner decisions (blockers — see PUBLICATION_READINESS.md)
- [ ] B1 Choose the project licence. If Apache-2.0 is confirmed: copy
      `docs/licensing/LICENSE-Apache-2.0.proposed.txt` to `LICENSE`, rename `NOTICE.proposed` to `NOTICE`,
      update `pyproject.toml` (`license = ...`) and the README *Licence* section.
- [ ] B2 Confirm code ownership and any institutional authorisation.
- [x] B3 Git history / author identity — resolved: fresh public history from the GitHub initial commit,
      `noreply` identity, internal maintenance notes not included.
- [ ] B4 Decide on MulSen-AD-derived results (`configs/workshop_verified.json` `d06-*`, `docs/LOCAL_RESULTS.md`).
- [x] B5 GitHub repository created (private): https://github.com/edelgr/industrial-defect-detection-lab
- [ ] Add the original workshop presentation to `workshop/CIARP2026/slides/` after its IP review.

## B. Technical verification (run on the instructor computer)
- [ ] `.\test.ps1` → all tests pass.
- [ ] `python -m scripts.check_docs` → 0 problems.
- [ ] `python -m scripts.readiness --profile instructor` → 0 required failing (24/24 SHA-256).
- [ ] `python -m scripts.qt_flow_check --lang en` and `--lang es` → all checks pass.
- [ ] `python -m scripts.qt_student_check --lang en` and `--lang es` → all checks pass.
- [ ] Clean install of the Student Edition (core and full) in a separate short folder → readiness 0 failing.

## C. Stage the release (outside the repository)
- [ ] Commit everything to be released.
- [ ] `python -m scripts.build_release --out C:\IDD_release_<version>` → *0 PROBLEM* lines.
- [ ] Read the generated `RELEASE_MANIFEST.json` (in the staging folder, not committed): excluded files, redactions, items needing review.
- [ ] Spot-check that no `data/`, `checkpoints/*.pt`, `*.sqlite`, `cache/` or personal paths are present.

## D. Publish (only after explicit authorisation)
- [ ] Push to the private repository; let CI run (no GPU, no downloads).
- [ ] Review the rendered README and links on GitHub.
- [ ] Make the repository public / create the release manually; attach no datasets or weights.
- [ ] Update CHANGELOG.md with the release date.
