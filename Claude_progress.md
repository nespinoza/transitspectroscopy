# Claude progress log — executing Prompts 0–5 of `Claude_code_report.md`

Branch: `refactor/consolidation` (from `dev-codex-upgrades` @ `fd0cc30`).
Test env: `/Users/newen/anaconda3/envs/transitspectroscopy` (Python 3.12.14, jwst 3.0.0) until Prompt 1 rebuilds it on 3.13.
Baseline: **157 passed, 13 skipped, 2 warnings** (`python -m pytest tests -q`).

**How to resume:** read this file top to bottom; the last unchecked item under the
current prompt is where to continue. Every commit is listed with its hash.

Decisions taken where the report left a choice open (edit if you disagree):
- Version `0.5.0.dev0`.
- Docs tool: Sphinx + myst-nb (astropy ecosystem), hosted on Read the Docs.
- Large files: only `validation/references/soss/jwst_niriss_bkg_0033.fits` (103 MB) exceeds 5 MB; it is already git-ignored and is NOT committed.

---

## Prompt 0 — Secure the work and clean the repository

- [x] Branch `refactor/consolidation` created.
- [ ] Commit untracked source/tests/validation scripts.
- [ ] Remove build artefacts from working tree; verify `.gitignore`.
- [ ] Version bump + CHANGELOG 0.5.0 section.
- [ ] Move CODEX_*.md to `docs/dev-notes/`.
- [ ] `residual_correction=False` default + test.
