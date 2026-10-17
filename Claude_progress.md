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
- [x] Commit untracked source/tests/validation scripts — `dac707d`.
- [x] Remove build artefacts (root `CCF.cpython-311-darwin.so`, `build/`, egg-info, swap files, caches, empty `ts_outputs/`); `.gitignore` extended.
- [x] Version `0.5.0.dev0`; CHANGELOG `[0.5.0] - Unreleased` section (Added/Changed/Fixed).
- [x] CODEX_*.md moved to `docs/dev-notes/`; links in README, tests/README, validation/README updated.
- [x] `residual_correction=False` default, docs updated, new test; one existing test now passes `residual_correction=True` explicitly.
- Commit `f712a91`. Tests: **158 passed, 13 skipped** (157 + 1 new test).

**Prompt 0 DONE.**

---

## Prompt 1 — Packaging (pyproject, Python ≥3.13, jwst required), env, CI, README install

- [x] jwst 3.0.0 metadata: `Requires-Python: >=3.11,<3.15` → package declares `>=3.13,<3.15`.
- [x] 3.13 availability (conda-forge osx-arm64 + linux-64, PyPI): numpy 2.2.6, scipy 1.17.1, celerite 0.4.3, george 0.4.4, h5py, pandas all have py313 builds; pymultinest, dynesty, matplotlib, jdcal are noarch; multinest/gsl have no Python dependency; jwst 3.0.0 / ray 2.58.0 ship cp313 wheels; astropy 7.2.2 ships an abi3 wheel; stdatamodels/crds/juliet/astroquery are pure Python. **No pin needs to change.**
- [x] `pyproject.toml` holds all metadata; `setup.cfg` removed; `setup.py` reduced to the optional CCF extension (setuptools cannot call `numpy.get_include()` from pyproject alone). `upgrade.sh` uses `python -m build`.
- [~] `environment.yml` → `python=3.13`, `setuptools>=77`. Building it as a NEW env `transitspectroscopy-py313` so the working 3.12 env is not destroyed (log: scratchpad `env313.log`). Once verified, you can remove the old env and rename: `conda env remove -n transitspectroscopy && conda rename -n transitspectroscopy-py313 transitspectroscopy`.
- [x] `.github/workflows/tests.yml`: `unit` job (pip `.[test]`, ubuntu+macos, py3.13; py3.14 non-blocking) and `full` job (environment.yml via setup-miniconda). Markers `crds`/`network` registered in pyproject; CI runs `-m "not crds and not network"`. (Current tests already avoid CRDS by using local synthetic references.)
- [x] README install section shortened; full details in `docs/installation.md`.
- [x] Verified: `pip install` on Python 3.12 is refused (`requires a different Python: 3.12.14 not in '<3.15,>=3.13'`).
- Commit `83fd01e`.
- [x] New conda env `transitspectroscopy-py313` (Python 3.13.16, jwst 3.0.0): `pip install --no-build-isolation --no-deps .` OK, `pip check` clean, CCF built for cp313. Full suite: **158 passed, 13 skipped** (3rd warning is upstream photutils deprecation inside jwst).
- [x] Fresh venv on 3.13 with plain `pip install ".[test]"`: pulls jwst 3.0.0, imports, **156 passed, 15 skipped** (2 extra skips need juliet/ray/C extras not in `[test]`).
- [x] yamllint on the workflow: clean.
- The docs toolchain (sphinx 9.1, myst-nb 1.4, pydata-sphinx-theme 0.23, numpydoc, sphinx-copybutton) and seaborn were pip-installed into `transitspectroscopy-py313`.

**Prompt 1 DONE.**

---

## Prompt 2 — Docs, README, tutorials

- [ ] Move validation READMEs/reports + SHARED_PROFILE.md/RAMP_FITTING.md into docs/, fix links.
- [ ] Sphinx (myst-nb, pydata theme) skeleton: index, installation, quickstart, concepts, guides, API, validation, changelog.
- [ ] README ≤ 400 words.
- [ ] Synthetic executed tutorials: optimal extraction, shared profile, ramp fitting. Real-data (not executed): SOSS HAT-P-14b, MIRI LTT 1445A.
- [ ] Existing 4 tutorials: static API check + smoke run with local data (full rerun needs full PID 1118/1177 downloads).
- [ ] `.readthedocs.yaml`; docs build without warnings.
- Note: `jwst` is still imported lazily inside the package (allowed by the prompt; keeps `import transitspectroscopy` fast and the existing "blocked optional imports" test valid).
