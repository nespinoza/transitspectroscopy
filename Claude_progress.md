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

- [x] Moved `validation/{README,SOSS_README,MIRI_README}.md` and all `*_REPORT.md` to `docs/validation/` (renamed: nirspec-nrs1, soss, miri, soss-report, soss-optimal-report, soss-overlap-report, miri-report, miri-ramp-comparison-report, gp-sweep-report); `SHARED_PROFILE.md`/`RAMP_FITTING.md` → `docs/guides/`. Links rewritten: cross-doc links relative, repo files → GitHub `blob/master` URLs, git-ignored `validation/runs/...` products → plain code paths. New short `validation/README.md` index of scripts/configs.
- [x] Sphinx docs (myst-nb, pydata theme, mermaid): index, installation, quickstart, concepts, guides (nirspec, optimal-extraction, shared-profile, soss, miri, ramp-fitting, lightcurve-fitting), tutorials, API (autodoc), validation index with headline table, changelog (includes CHANGELOG.md). `docs/dev-notes/` excluded from the build.
- [x] README ≈ 290 words (author voice + citation kept).
- [x] Synthetic tutorials executed at build (each ~3 s): `optimal-extraction` (shows box vs polynomial vs GP; also shows that 3 polynomial terms are biased (pull std 7.2) where 6 terms are fine (1.05) on a bright spectrum — a real property of the Marsh polynomial model, not a bug), `shared-profile` (recovers 1.007 ± 0.022 % vs injected 1 %), `ramp-fitting` (pull std 0.995). Real-data `realdata-soss-hatp14b`, `realdata-miri-ltt1445a` (not executed).
- [x] Original 4 notebooks: static check — every `ts.*` call exists and accepts its keywords (0 problems). Fixed Python 3.12+ invalid-escape `SyntaxWarning`s (6 LaTeX label strings → raw strings). Not changed: a label in WhiteLightFitting says NRS1 while plotting `rms_nrs2` (author's content).
- [x] `.readthedocs.yaml` (py3.13, fail_on_warning); docs CI job. `sphinx -W` build: **succeeded, 0 warnings**. Docstring markup fixes only (no code changes) in jwst.py/spectroscopy.py/utils.py; CHANGELOG versions are now `##` headings.
- Commits `6609e55`, `cfe4db3`. (Note: commit `83fd01e` accidentally included a temporary `condaenv.*.requirements.txt` written by conda; removed in `6609e55`.)
- [x] Smoke run of the Quicklook flow (load → detector_calibration → fit_ramps → stage2, 2-pixel box) on the single local NRS1 segment with Python 3.13 / jwst 3.0.0 / CRDS_PATH=/Users/newen/crds_cache: **passed** — 391 integrations, white-light RMS of first 100 integrations 252.7 ppm (undetrended), 671 s. White-light curve kept as `equiv/smoke_whitelight_before.npy` to re-check after the refactor. Full re-execution of the four original notebooks needs the full PID 1118 (NRS1+NRS2, all segments) and PID 1177 downloads, which I did not do.
- Env note: set `CRDS_PATH=/Users/newen/crds_cache` on `transitspectroscopy-py313` (the old env had no CRDS_PATH; validation runs recorded that path). Registered the env's own `python3` Jupyter kernel (`ipykernel install --sys-prefix`), since base Anaconda's kernel was shadowing it.
- Note: `jwst` is still imported lazily inside the package (allowed by the prompt; keeps `import transitspectroscopy` fast and the existing "blocked optional imports" test valid).

**Prompt 2 DONE** (pending only the background NRS1 smoke run result, see above).

---

## Prompt 3 — `transitspectroscopy.core` (no jwst/stdatamodels/gwcs imports)

Equivalence baseline: scratchpad `equiv/record.py` records 956 arrays from every function
being moved (Marsh poly/GP, box, simple, tracing/CCF, utils binning, outlier detector,
all detector algorithms incl. side refpix / group 1/f / TSO jump on fake models, ramp
fitting, shared profiles incl. hyperparameter fit, SOSS helpers/optimal/overlap,
photometry incl. GP PSF). `equiv/before.npz` recorded at commit `cfe4db3`; verified
deterministic (two runs identical). After refactor: rerun and `compare.py before after`.

Layout decision (deviation from the report's suggested file list, for readability):
`core/profiles.py` holds the Marsh profile fit and the profile bases; `SharedProfile`
lives in `core/shared_profile.py`, and joint multi-trace (overlap) fitting in
`core/overlap.py`, rather than one ~1,200-line `profiles.py`.

- [x] core/kernels.py — one covariance (Matérn 3/2, 5/2, squared-exponential) + `inducing_cholesky` + `whitened_basis`, used by Marsh GP profiles, SharedProfile and the 2-D GP PSF (NumPy and SciPy Cholesky verified bitwise equal before unifying on SciPy).
- [x] core/geometry.py (legacy Marsh geometry + native-pixel windows + `aperture_weights`), core/profiles.py (`power_basis`, `legendre_basis`, `gp_basis`, `marsh_gp_options`, `fit_marsh_profile`), core/extraction.py (`box_spectrum` = your getSimpleSpectrum algorithm, `fractional_box`, `marsh_box`, `marsh_optimal`, `frozen_profile`, `solve_blended`).
- [x] core/shared_profile.py (SharedProfile, fit_shared_profile, public `fit_coefficients`, `pixel_profile`, `profile_basis`, `MODEL_VERSION`), core/overlap.py (`JointProfileFit`, `train_joint`, `extract_blended`, `unmodeled_pixels`).
- [x] core/tracing.py (your trace_spectrum/CCF/FWHM functions + SOSS `refine_trace`, `trace_motion`), core/ccf.py (was `_ccf.py`), core/backends.py (`select_backend`), core/photometry.py (`_precision` → public `sky_precision`), core/ramps.py (`_weights` → `difference_weights`).
- [x] core/detector.py: your array functions moved unchanged (cc_uniluminated_outliers, get_roeba, get_loom, get_last_minus_first, spill_filter, get_uniluminated_mask, correct_1f) + new kernels `subtract_side_pedestal`, `group_1f_correct`, `tso_jump_flags`. `jwst.side_refpix_correction` / `group_1f_correction` / `tso_jumpstep` keep their signatures and become thin datamodel wrappers.
- [x] core/timeseries.py (`outlier_detector`, `mad_sigma`), core/binning.py (`bin_at_resolution`), core/provenance.py (`file_sha256`, `option_record`); `src/jwst_models.py` (`open_model`, `integration_times`) replaces the private `soss._model/_times` that miri imported.
- [x] Shims: spectroscopy.py (1,159 → 485 lines; getP/getOptimalSpectrum/getFastSimpleSpectrum/getSimpleSpectrum keep signatures and C dispatch; tracing names re-exported), shared_profile.py, photometry.py, soss_overlap.py, timeseries.py, ramp_fitting.py (JWST layer only), utils.py (re-exports bin_at_resolution, get_mad_sigma). Deleted `_optimal_extraction.py` (Codex-internal, never released). `reduce_soss` source-hash list updated to the new file locations (Prompt 5 replaces that mechanism).
- [x] All moved Codex code formatted with `ruff format` (AST-preserving; quote style preserved; ≤100 columns, no semicolons); `[tool.ruff]` added to pyproject. Your moved functions keep their logic and comments.
- [x] Tests: legacy-API test now uses `inspect` on the imported objects (same contract: names, order, defaults, **kwargs) instead of parsing source files; tests that used private helpers now use the public core names. New `tests/test_core_layering.py`: core imports no jwst/stdatamodels/gwcs (checked in a subprocess and on the source), and no module imports another module's underscore names.
- [x] **Equivalence: 956 arrays, 955 bit-for-bit identical; 1 (cached GP PSF evaluator grid) differs at 3e-16 relative** (Cholesky now via SciPy) — within rtol=1e-12.
- [x] Suite: **161 passed, 13 skipped** (158 + 3 layering tests). `sphinx -W`: succeeded (API page now documents core).
- [~] Post-refactor NRS1 smoke run in background (scratchpad `smoke_after/`), to compare with `equiv/smoke_whitelight_before.npy`.
