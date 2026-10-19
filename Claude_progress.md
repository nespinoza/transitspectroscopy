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
- [x] Post-refactor NRS1 smoke run (real data, full load → stage2 flow): white-light curve **bit-for-bit identical** to the pre-refactor run.
- Commits `396b01d`, `425bb38` (gp_sweep: uses core geometry; warns instead of refusing on changed sources).

**Prompt 3 DONE.**

---

## Prompt 4 — Mode registry, ExtractionConfig, SOSS/MIRI in the main flow

Decisions (the report's decision lines were removed; I used its recommendations):
- Detector stage for every mode = `ts.jwst.load(...).detector_calibration()` with the
  author's custom steps chosen by the Mode (as before); the official calwebb_detector1
  remains available through `miri.extract_photometry(input_product='uncal')` / `stage1`.
- SOSS: the JWST background step (zodiacal light, CRDS bkg template) runs on a median
  rate image after ramp fitting and before tracing/extraction (unchanged algorithm,
  now recorded as the mode's `background_strategy='jwst_background_step'`). Tracing is
  still PASTASOSS + the double-Gaussian refinement (I did not switch it to the CCF
  tracer: that would change validated SOSS results; noted as a follow-up).
- Mode names are unchanged (`nirspec/prism`, ..., `miri/photometry`, alias `miri/imaging`)
  because `load.mode` values are user-visible.

- [x] `src/instruments/` (`modes.py`: Mode, register/unregister, identify, identify_model,
  list_modes, UnknownModeError; `jwst_modes.py`: built-in modes with the historical values).
- [x] jwst.py: all `self.mode == ...` / `mode == ...` branches replaced by Mode lookups
  (check_status, fill_calibration_parameters, detector_calibration, stage1, stage2 tracing
  and extraction defaults, NRS1/NRS2 suffixes). `cds_stage1` (experimental, takes its own
  `instrument` string) left as is.
- [x] `src/config.py`: ExtractionConfig + `from_stage2_arguments` + mode-aware `validate`.
  stage2 gains `config=` and dispatches on `mode.stage2` ('spectral' | 'soss' | 'imaging' |
  callable plug-in). Deprecated keywords: extraction_backend, profile_method, gp_options,
  extraction_options, shared_profile, shared_profile_options, soss_options, miri_options.
  The author's original stage2 arguments do not warn.
- [x] soss.reduce_soss → `extract_soss` (+ deprecated wrapper); miri.reduce_miri →
  `extract_photometry` (+ deprecated wrapper). Callers in tests/validation updated.
- [x] tests/test_instruments.py (11 tests): registry, historical defaults, synthetic
  stage2 for PRISM, G395H NRS1/NRS2, G395M NRS1/NRS2, SOSS & MIRI dispatch equivalence
  (config vs keywords), plug-in mode with no jwst.py edits, config validation.
- Suite: **172 passed, 13 skipped**. Commit `18df404`.
- [x] NRS1 real-data smoke run on Prompt 4 code: white-light curve **bit-for-bit identical** to the pre-refactor run (mode identified through the registry).
- [x] `load.sort_filenames` sorts by (exposure, segment) instead of segment only (same order within one exposure; correct for multi-exposure visits).
- [x] Docs: `docs/guides/adding-a-mode.md`; API pages for config/instruments; SOSS/MIRI guides now lead with load → detector_calibration → fit_ramps → stage2 (commit `c807d31`).
- [x] **SOSS validation re-run** (`validation/runs/soss_prompt4_check`, same command/pinned references as `soss_overlap_20260925_final`): status complete; every non-timing number in reduction.json identical (orders 96, optimal 216, extraction_comparison 207, order_overlap 16, background 9); **all 9 saved spectra files (box/polynomial/GP × orders 1–3) bit-for-bit identical**; order-1 white-light depth 6779.4 ppm as reported.
- [x] **MIRI validation re-run** (`validation/runs/miri_prompt4_check`, full rateints → calwebb_image2 → cutouts → photometry, both visits, 164 s): status complete; cutouts/centroids/sky/flags/times bit-for-bit identical; aperture-sum fluxes bit-for-bit identical; GP fluxes differ by ≤1.1e-15 relative per integration (headline GP RMS 637.976148959680 → 637.976148959682 ppm; max relative change of any reported number 1.9e-11), from the SciPy-vs-NumPy Cholesky unification in Prompt 3. Standard products written to `visit*/products/`.
- Not testable here: the uncal → load → stage2 path for SOSS/MIRI on real data (only rateints are on disk for PIDs 1541/9234); covered by synthetic tests only.

**Prompt 4 DONE.**

---

## Prompt 5 — Products, provenance, runner, validation ports

- [x] `core/products.py`: SpectroscopicTimeSeries, PhotometricTimeSeries, LightCurveSet; FITS (multi-extension, METADATA JSON extension, provenance summary in header) + ECSV I/O; `write_products/read_products`; white-light and wavelength binning; `resolution_edges`.
- [x] `core/provenance.py`: `build_manifest` (package/git commit+dirty/Python/numpy/jwst/crds versions, CRDS context, input names+sizes+SHA-256, resolved config + its SHA-256, seeds, timestamps), `check_manifest` (warn; `strict=True` raises), manifest sidecars `<file>.provenance.json`.
- [x] Every reduction path returns `result['products']` and writes them: spectral stage2 (`pipeline_outputs/products<suffix>/`: `<det>_corrected`, `<det>_original`, `<det>_whitelight`), SOSS (`products/`: `order<N>_<box|polynomial|gp>` + fixed-band `_whitelight`), MIRI (`products/`: `sum`, `gp`). `stage2(mode=..., write_products=...)` added.
- [x] `soss.load_products(strict=False)` warns instead of refusing when source files changed.
- [x] `ts.jwst.download(..., exposure=, segments=, detectors=, skip_existing=True)` returns the selected paths (+ test).
- [x] `src/runner.py` (`ts.run`, `load_config`, step folders `<step>-<hash>`, force/from_step, per-exposure loading, CRDS context pinning, keep_stage2) and `src/cli.py` (`transitspectroscopy run config.toml`, `transitspectroscopy modes`); console script in pyproject.
- [x] Runner tests (`tests/test_runner.py`: products/manifests, step reuse, recompute on change, from_step, TOML + CLI, config errors, warn/strict). Suite **177 passed, 13 skipped**. Commit `c67b9d5`.
- [x] Runner docs: `docs/guides/running.md`; quickstart/README mention `transitspectroscopy run`; API pages for runner/products.
- [x] Validation ports (commit `5d24149`, docs `1fe6ac9`): `validation/configs/{nrs1-hatp14b,soss-hatp14b,miri-ltt1445a}.toml` (MAST via ts.jwst.download, CRDS context pinned for NRS1, pinned SOSS references, [validation] diagnostic settings). `validate.py --config X.toml` (or `--mode`) runs `ts.run` then the diagnostics; soss/miri drivers keep their plots/metrics; drivers use the installed package and refuse a stale install. Old JSONs → `validation/configs/legacy/`. Runner loads data only when needed; download takes exposure lists and treats `.fits.gz` as present. Old `--reuse-*` options replaced by the runner's step caching.
- [~] Re-running the three ported validations (background batch, scratchpad `validations_p5.log`, outputs `validation/runs/{miri,soss,nrs1}_prompt5_runner`), to compare with the reviewed runs.
  - [x] **MIRI (ported, from config alone): reproduces the report** — sum RMS 659.297653/646.667056 ppm, GP RMS 637.976149/632.427043 ppm (visits 1/2), all integrations and segments selected; sum fluxes, times and centroids bit-for-bit identical, GP fluxes ≤1.1e-15 relative (182 s).
  - [x] **SOSS (ported, from config alone): bit-for-bit identical to the reviewed run** — every number in orders/optimal/extraction_comparison/order_overlap/background (544) identical, all 9 spectra files identical; data via ts.jwst.download (MAST query, files reused), rateints through ts.jwst.load → stage2 (13 min, fits in 8.6 GB RAM).
  - [x] **NRS1 (ported, from config alone): reproduces the report.** First attempt was killed (exit 137) because the disk was 98% full and swap could not grow; after I deleted ~16 GB of my own scratchpad smoke outputs and test venv, the rerun (`--from-step calibrate`, 1295 s) completed: white-light RMS **simple 264.1 / polynomial 273.0 / GP 265.6 ppm**, **1537** common columns (2.674–3.717 µm), trace-motion RMS **0.001906** px, FWHM-change RMS **0.0004471** px, median FWHM **1.8017** px, median simple−polynomial difference **0.8294** DN/s — all as reported in docs/validation/nirspec-nrs1.md. (The original reference folder `nrs1_segment001_20260922_final` is no longer on disk, so this is a comparison with the reported numbers, not file-by-file.)
  - Disk note for you: the volume is at ~90% after cleanup; `validation/runs/` holds ~12 GB of runs (mine: `*_prompt4_check`, `*_prompt5_runner`) that you may delete once reviewed.



**Prompt 5 DONE.** Not done: the optional item 5 (embedding SharedProfile/PSF models *inside*
the product FITS). The models are saved next to the products (as before), and shared-profile
fingerprints are recorded in the product metadata.

---

## Final state (Prompts 0–5)

- Branch `refactor/consolidation`, all work committed (not pushed). Suite: **178 passed, 13 skipped** on Python 3.13 / jwst 3.0.0 (`transitspectroscopy-py313` env). Docs: `sphinx -W` clean.
- Real-data checks: NRS1 smoke run bit-identical after Prompts 3 and 4; SOSS bit-identical after Prompt 4 and through the runner; MIRI aperture sums bit-identical and GP ≤1e-15 relative; NRS1 validation reproduces every reported headline number through the runner.
- Things for you to do / decide:
  - Swap the conda env: `conda env remove -n transitspectroscopy && conda rename -n transitspectroscopy-py313 transitspectroscopy` (the new env also has CRDS_PATH set and the docs toolchain installed).
  - Disk is ~90% full: `validation/runs/*_prompt4_check` and `*_prompt5_runner` (~10 GB) can be deleted after review.
  - Not tested on real data: uncal → load → stage2 for SOSS and MIRI (only rateints on disk); the GitHub Actions workflows (only linted locally).
  - SOSS tracing still uses the PASTASOSS + double-Gaussian refinement (switching to your CCF tracer would change validated results).

---

## SOSS uncal validation — PID 1541 obs 1, the 6.1-hour TSO (2026-10-17)

Request: download with transitspectroscopy, run the pipeline from uncal, white-light curve,
RMS about a 100-point median filter.

- Exposure `jw01541001001_04101_00001` (572 ints × 6 groups, SUBSTRIP256, EXPEND−EXPSTART = 6.11 h).
- [x] Downloaded with `ts.jwst.download(1541, 1, outputfolder='validation/data/soss_uncal', data_product='uncal', exposure=...)`: 4 segments, 3.6 GB.
- Memory: this Mac has 8.6 GB RAM; `ts.run` would hold all 4 segments' ramps at once (~16 GB).
  New `validation/soss_uncal_calibrate.py` runs load → detector_calibration → fit_ramps (mode defaults,
  TSO jump window 10) per segment, keeps `stage1/<seg>_rateints.fits`, deletes the GB-sized ramps.
- [x] seg001: 265 s, all steps COMPLETE, CRDS jwst_1584.pmap, peak swap ~7.4 GB.
- [x] seg002–004: 185 / 198 / 194 s. Four rateints in `validation/runs/soss-hatp14b-uncal/stage1/`.
- [x] Stage 2 (`validate.py --config validation/configs/soss-hatp14b-uncal.toml`): complete, 900 s, 572 integrations.
- Result (100-pt median-filter RMS, ppm; table in `validation/runs/soss-hatp14b-uncal/white_light_rms_summary.txt`):
  order 1 box 484.6 (139.5 without integration 60), polynomial 447.1 (130.0), GP 445.2 (132.9);
  archive rateints: 192.3 / 120.2 / 122.8.
- Integration 60 (seg001) is +11,100 ppm: a detector-wide event adds ~300 DN to the first group difference of
  every illuminated pixel. The JWST jump step (archive) flags 32 % of trace pixels (>=2000 DN/group); the TSO
  jump step at nsigma=10 flags 9 %, since 300 DN < 10 sigma of photon noise there. Not fixed (needs a decision).
- Transit fits: order-1 depth 6782 ± 229 ppm (reference 6750 ± 40, 0.14 sigma).
- [ ] Stage 2: `python validation/validate.py --config validation/configs/soss-hatp14b-uncal.toml`
  (same extraction settings as soss-hatp14b.toml; `validate.py` now also resolves `data.glob` against the repo root).
- Reference (archive rateints, soss_prompt5_runner), order-1 white light, 100-pt median-filter RMS:
  box 192.0, polynomial 120.0, GP 122.6 ppm; order 2: 244.7 / 143.8 ppm.

### Why the TSO jump step misses integration 60 (investigation, no code changed)

- The event: in integration 60 (seg001), columns 1024–2047 of the active pixels shift by an additive,
  illumination-independent amount between groups 0 and 1 (group 0 about 200 DN low, groups 1–5 about 115 DN high).
  Only the first group difference changes, by 280–390 DN (smooth along columns, constant along rows).
  The reference rows and columns don't show it, so refpix can't remove it. It is the only such event in the 572 integrations
  (332 DN against a 3 DN scatter in the half-detector difference statistic).
- `tso_jump_flags` tests each pixel separately: |r| > nsigma·σ_pixel, with σ_pixel from the MAD of that pixel's
  group-difference time series. Faint pixels have σ≈11–13 DN (z≈22, 97 % flagged). On the trace σ≈49 DN (photon noise), so z≈6.3:
  only 9 % are flagged at nsigma=10. The unflagged 95 % of the trace flux keeps +309 DN in one of 5 differences,
  which adds 309/(5·5.494 s) = 11.2 DN/s to the rate (11.4 measured), i.e. +1.1 %.
- Lowering nsigma doesn't fix it: at nsigma 7/5/4/3 the flagged fraction of trace pixels is 0.39/0.72/0.86/0.95, but the unflagged
  ones stay biased (274/233/200/157 DN, because of selection).
- Subtracting a per-column offset estimated from low-signal pixels (for that integration and group) removes it:
  on the trace, the median z goes from 6.3 to −0.40 and no pixel exceeds 10σ.
- Proposed change: a common-mode stage in the TSO jump step (detect and correct, or flag, coherent per-column
  offsets per integration and group before the per-pixel test). See the chat answer, 2026-10-17.

### Culprit in our Stage 1: `clean_flicker_noise` is never run (2026-10-17)

- Archive seg001 header: `S_CLNFNS = COMPLETE` (jwst 2.0.1, jwst_1535.pmap); every reference file is the same as ours.
- CRDS `pars-detector1pipeline` for this exposure: `clean_flicker_noise` skip=False, background_method='median_image',
  mask_science_regions=True, n_sigma=0 (and jump rejection_threshold=6). Our `detector_calibration` calls the steps
  one by one with `Step.call`, so pipeline-level CRDS parameters never apply, and `clean_flicker_noise` isn't in the
  mode's step list.
- Test (integrations 40–80 of seg001; our Stage 1, then RampFitStep, sum over trace pixels):
  without it, int 60 = +12,220 ppm and the others' scatter is 182 ppm; with `CleanFlickerNoiseStep` (CRDS parameters)
  before ramp fitting, int 60 = +138 ppm and the others' scatter is 145 ppm.
- Scratch test: `<scratchpad>/cfn_test.log`. No library code changed yet.

### clean_flicker_noise in Stage 1; TSO jump vs STScI jump (2026-10-17, in progress)

Plan:
- [x] Mode registry: SOSS `detector_steps['post_jump_steps'] = ('clean_flicker_noise',)` (other modes: none, so no change).
  `detector_calibration` runs these after the jump step and before ramp fitting, with `Step.call` (CRDS step pars apply).
- [x] TSO jump intermediate ramps saved only when `save=True` (currently always).
- [x] Bug: `use_tso_jump=False` passes `jump_threshold` (not a JumpStep parameter), so the STScI default rejection_threshold=4 is used;
  map it to `rejection_threshold` like `stage1()` does (explicit `rejection_threshold` wins).
- [x] `validation/soss_uncal_calibrate.py --jump {tso,jwst}`, save=False and write only the rateints.
- Code commit `1201ea1` (tests 179 passed). Old no-CFN stage1 rateints deleted.
- Note: the first variant A run used a STALE install (non-editable; no CFN). Discarded. The calibration script now
  refuses a stale install (validate.installed_package). Reinstalled with `pip install --no-build-isolation --no-deps .`.
- [~] Variant A (Stage 1 rerunning, output validation/runs/soss-hatp14b-uncal-tsojump/): TSO jump + CFN → stage1-tsojump → Stage 2 (configs/soss-hatp14b-uncal.toml).
- [ ] Variant B: STScI jump (rejection_threshold=6, the CRDS pipeline value for SOSS) + CFN → stage1-jwstjump → Stage 2
  (configs/soss-hatp14b-uncal-jwstjump.toml).
- Disk: about 10 GB free; each variant's rateints are 6 GB. I'm deleting the old no-CFN stage1 rateints (its Stage 2
  results and RMS table are kept), and A's rateints after A's Stage 2 if needed.
- Variant A rerun: seg001/002 done (505/481 s; with clean_flicker_noise, swap grew ~10 GB and filled the disk). Killed (137)
  in seg003 at 303 MB free. Freed 4.8 GB by deleting the two regenerable NRS1 intermediate ramps in
  `validation/runs/nrs1_prompt5_runner/reduction/calibrate-715b63a75a7c/exposure1/ts_outputs/` (linearitystep, tsojumpstep;
  the results are kept). Resumed seg003–004 with a guard that stops below 1.5 GB free (log stage1_resume.log).
- Resumed run STOPPED by Claude Code (system critically low on memory) during seg003. Not restarted (needs the user's OK).
  Done: variant A seg001 + seg002 rateints (stage1-tsojump/). To do: A seg003–004, A Stage 2, then variant B.
  Resume: `python validation/soss_uncal_calibrate.py validation/data/soss_uncal/*_uncal.fits --output validation/runs/soss-hatp14b-uncal-tsojump/stage1 --jump tso`
  (skips finished segments).
- Partial check (seg001 ints 40–80, trace-pixel sum): int 60 = +142 ppm (TSO jump + CFN), against −177 ppm for the archive;
  scatter of the other integrations 147 ppm (ours) vs 219 ppm (archive). The event is fixed by CFN with the TSO jump.

### One-segment comparison: CFN vs group 1/f, TSO jump vs STScI jump (2026-10-17)

User: just seg001; 4 reductions run serially; white-light RMS plots for each.
Folder `validation/runs/soss-seg1-comparison/<variant>/` (stage1 rateints + Stage 2 via
`validate.py --config validation/configs/soss-hatp14b-uncal.toml --input-glob <stage1 rateints> --output <variant> --skip-transit`).
- [x] 1_cfn_tsojump: Stage 1 reused from the variant A run (current code); seg002 rateints deleted.
- [x] 2_group1f_tsojump: `--group-1f --skip-clean-flicker-noise` (262 s, peak RSS 4.8 GB)
- [x] 3_cfn_jwstjump: `--jump jwst` (rejection_threshold 6) (560 s, 4.4 GB)
- [x] 4_group1f_jwstjump: `--jump jwst --group-1f --skip-clean-flicker-noise` (246 s, 4.7 GB)
- [x] Stage 2 for all 4 (about 560 s each, `--skip-transit`).
- [x] `validation/soss_segment_comparison.py` written and tested (references: archive_rateints = soss_prompt5_runner, uncal_no_correction = soss-hatp14b-uncal; first 144 ints).
  Run: `python validation/soss_segment_comparison.py validation/runs/soss-seg1-comparison --reference archive_rateints=validation/runs/soss_prompt5_runner --reference uncal_no_correction=validation/runs/soss-hatp14b-uncal`
- Results (seg001, 144 integrations; order 1 box/polynomial/GP; tables in validation/runs/soss-seg1-comparison/white_light_rms*.{json,txt}, plots white_light_rms_*.png):
  - All integrations: 1 CFN+TSO 120.4/117.0/119.9; 2 g1f+TSO 213.9/209.6/214.0; 3 CFN+STScI 132.1/123.2/125.8;
    4 g1f+STScI 219.7/210.6/215.4; archive 193.6/123.0/125.6; uncal with neither 934.7/862.6/857.5.
  - Without int 60: 1 120.8/117.1/120.3; 2 122.7/120.6/123.1; 3 132.4/122.5/126.1; 4 132.1/124.6/126.9.
  - Group 1/f overcorrects int 60 (−2,100 ppm). The event offset is constant AFTER linearity (raw offset +320 DN on dark pixels, 306→266 DN
    over the groups on the trace); group_1f runs before linearity and subtracts the dark-pixel raw value from the trace, giving
    −10 DN/group ≈ −1.8 DN/s (measured −1.83). Moving group_1f after linearity should fix it (not tested).
  - The TSO jump step beats the STScI jump step (rejection 6) for the box: order 1 121 vs 132, order 2 144 vs 152 ppm.

## 2026-10-18: group 1/f vs clean_flicker_noise (jwst 3.0.0 source review)
- CFN (SOSS CRDS pars: background_method=median_image, mask_science_regions, n_sigma=0, fit_method=median):
  builds a draft rateints, subtracts the SOSS background (BackgroundStep), box-extracts order 1 for a
  white-light scale, takes the median ramp over integrations, and subtracts `scale_i*median_ramp +
  background_ramp` from every group (group 0 included) before the masked column median; adds it back.
  Mask: pastasoss traces of all orders, +-16 px. Runs on linearized, dark-subtracted ramps after jump.
- group_1f: no scene subtraction; one |LMF|-weighted centroid per column, +-10 px. On PID 1541 seg 1 the
  centroid misses order 1 by 15-45 rows for columns <1800 (pulled by order 2/background); the mask holds
  1-7% of the column flux there (73% only at columns >1900, where order 2 is absent).
- Int-60 offset is compressed by nonlinearity (raw 320 dark vs 266-306 trace): pixel-level, not
  amplifier-level (absent in ref pixels). Genuine 1/f is additive -> pre-linearity is the right place.
- Suggested: scene-subtracted (scaled median ramp) group_1f, multi-order mask, DQ/NaN exclusion,
  vectorization; keep it before linearity; treat outlier column events as pixel-level (correct in
  linearized space, x' = f^-1(f(x) - delta)). Untested; test plan in the chat answer.
- Correction (user meant `core.detector.correct_1f`, Stage 2): the SOSS Stage 2 path (`src/soss.py`) does
  not call correct_1f; with scale_1f=True it subtracts a plain per-column median of background-subtracted
  rates outside a static mask (all orders, aperture_radius+12 px; no scaled template). correct_1f is only
  used by the NIRSpec/legacy Stage 2 (jwst.py ~L2600) and the CDS path (~L1557). correct_1f = rate minus
  white-light-scaled median rate, then median of an annulus r_in<|dy|<r_out around the trace.
- Rate-level vs group-level: JWST ramp fit uses SNR-dependent weights (power 0 dark .. 10 bright), so a
  column offset projects onto the slope differently for trace and background pixels: a groups-1..5 step
  gives 0.143*delta/t_group (uniform) vs 0.20*delta/t_group (end-weighted). Measured int-60 trace excess
  11.2 DN/s matches 0.2*320/5.494 = 11.6 DN/s. For iid per-group 1/f, a rate-level correction from dark
  pixels leaves ~0.15 sigma_g of the ~0.28 sigma_g on the brightest pixels.
- correct_1f edge bug: `signal[:int(y - outer)]` with y < outer gives a negative stop index and masks
  almost the whole column, keeping only far rows.
- int-60 JUMP flags in variant 1 rateints: 31% of bright trace pixels, 0% of dark pixels (cols 1100-2000).

### Stage 2 1/f test on the group 1/f rates (2026-10-18)
User asked to run it. `src/soss.py`: new `onef_method` ('median' default, unchanged; 'template' =
correct_1f-style scaled median-rate template, current mask; 'template_ring' = template + only the
pixels between aperture_radius and aperture_radius+12 of the traces). Helper `soss.column_levels`.
Runs (Stage 2 only, on validation/runs/soss-seg1-comparison/2_group1f_tsojump/stage1 rateints),
folder validation/runs/soss-seg1-stage2-1f/g1f_stage2_{template,template_ring,none}; configs are
the uncal toml with `onef_method` / `scale_1f = false` (scratchpad copies; command:
`validate.py --config <toml> --input-glob <rates> --output <folder> --skip-transit`).
'median' = the existing 2_group1f_tsojump run.
- [x] template, template_ring, none (about 870 s each, peak RSS 1.8 GB); comparison plots/json in the
  folder (`soss_segment_comparison.py ... --reference g1f_stage2_median=... --reference cfn_stage2_median=...`),
  table without int 60 in `white_light_rms_without_int60.txt`.
- Without int 60 (box/poly/GP, ppm): order 1 none 122.1/117.0/118.2, median 123.1/120.8/123.4,
  template 120.1/118.0/120.0, ring 128.5/123.9/126.1, CFN+median 121.0/117.3/120.3.
  Order 2 none 168.0/174.8/174.8, median 143.0/140.6/139.6, template 141.7/136.1/135.7,
  ring 166.7/150.8/151.4, CFN+median 143.5/138.4/137.8.
- Int 60 (order 1 box): none -2115, median -2106, template -2046, ring -2046 ppm: no Stage 2 variant fixes it.
- [x] User: activate the correct_1f-style Stage 2 1/f for SOSS with masks around orders 1, 2 and 3:
  `extract_soss(onef_method='template')` is now the default (masks +-(aperture_radius+12) px around
  each of orders 1-3, plus 4 edge columns). Checked on seg001: all three orders' apertures fully masked
  (orders 1/2/3 cover columns 4-2043/4-1813/4-1141), >=91 unmasked pixels per column (median 133),
  0.9% of the median-rate light unmasked. This changes default SOSS outputs (requested).

### Group-level 1/f with median group-difference templates (2026-10-18)
User: like correct_1f but per group; template for group k = median over integrations of
(group k+1 - group k), scaled to group k (last group uses the last difference); same white-light
factor; done in Stage 2 (traces, white light), then ramps refitted.
- Code: `core.detector.group_difference_1f` (template factor = sum of median group k / sum of the
  difference over the order-1 aperture); `soss.onef_pixels`, `soss.white_light_scale` (shared with
  onef_method='template'; default numerics unchanged), `soss.correct_ramps_1f(ramps, rateints, result)`
  (valid = not DNU/saturated, not at/after a jump, not reference/DNU pixels). Test in tests/test_soss.py.
  Note: the median difference carries the median 1/f of the differences, times the template factor:
  a static (same in every integration) per-group, per-column offset; not a time-series noise.
- Experiment: `validation/soss_group_1f_experiment.py` (seg001; Stage 1 TSO jump, no CFN/group 1/f;
  first-pass rates -> box-only extract_soss -> correct_ramps_1f -> refit). Output
  validation/runs/soss-seg1-group-diff-1f/{no_group_1f,group_diff_1f}/stage1.
- First experiment run (6 min, peak RSS 4.9 GB): template factors 0.99/1.99/2.99/4.00/4.99/5.99 (groups
  1-6); white-light factors 0.9997-1.0102. But int 60 stayed (+9261 ppm, bright-pixel sum): the TSO jump
  flags 97.6% of background pixels in int 60 (31% of bright trace pixels), and the levels excluded groups
  at/after a jump. Background rates show no int-60 offset (ramps split at the jump); CFN does not exclude
  jump flags, so it measures the step. Fix: `template_valid` (jumps excluded) separate from the level
  validity (only DNU/saturated/reference excluded); new test. Stale outputs deleted; rerun started.
- [x] Rerun: int 60 bright-pixel sum +76 ppm (CFN +94, uncorrected +12684); bright trace -0.06 DN/s.
- [x] Stage 2 (template default) s2_group_diff_1f, s2_group_diff_1f_no_stage2_1f, s2_no_group_1f, s2_cfn;
  tables in validation/runs/soss-seg1-group-diff-1f/white_light_rms_without_int60.txt (both cases), plots there.
  RMS ppm (box/poly/GP), all ints | without int 60:
  order 1: group-diff+template 117.8/114.0/116.4 | 118.1/114.2/117.0; group-diff only 119.4/116.7/118.4 |
  119.7/115.0/118.1; CFN+template 118.8/115.2/117.8 | 119.3/115.0/118.2; g1f+template 208.2/202.4/206.5 |
  120.1/118.0/120.0; none+template 928.9/864.0/859.5 | 132.8/130.4/135.1.
  order 2: group-diff+template 144.5/137.3/136.0 | 139.1/132.4/131.1; group-diff only 140.9/134.7/134.1 |
  140.9/135.2/134.5; CFN+template 141.4/133.3/132.6 | 140.2/132.7/131.7; g1f+template 211.7/209.3/207.5 |
  141.7/136.1/135.7; none+template 1344.8/1360.2/1353.2 | 162.4/162.4/160.3.
- Open: order 2, group-diff + Stage 2 template leaves int 60 at about -480 ppm (not with group-diff alone,
  nor CFN + template); per-column-range sums too noisy to localize. Unexplained.

### Full exposure (4 segments): group-difference 1/f vs group 1/f, both TSO jump + Stage 2 template
- Stage 1 per segment (seg001 reused from the one-segment runs): group-diff via
  `soss_group_1f_experiment.py --drop-first-pass` (6 min/segment, peak footprint 10.4 GB);
  group 1/f via `soss_uncal_calibrate.py --group-1f --skip-clean-flicker-noise` (5 min/segment).
  Stage 2: validate.py with onef_method='template' (12 and 16 min). First Stage 2 attempt died on a
  full disk (swap files on the same volume); seg1 experiment FITS/NPZ intermediates deleted (approved).
- Output validation/runs/soss-hatp14b-full-1f/{s2_group_diff_template,s2_g1f_template,comparison}.
  Comparison: `validation/soss_full_comparison.py` (archive = soss_prompt5_runner as reference).
- 100-pt median-filter RMS ppm, all ints (without int 60), poly: order 1 group-diff 118.5 (117.2),
  group 1/f 145.3 (119.9), archive 120.2; order 2 138.9 (137.5), 162.6 (141.8), 144.0.
  Int 60: group-diff +436/-477 ppm (o1/o2 poly; o1 worst is now #255), group 1/f -1965/-1909.
  Transit-fit residual RMS (after the 80-integration trim) poly: o1 120.9 vs 121.8; o2 134.7 vs 138.0.
  Red noise (binning) the same for all three, archive included.
- Depths (poly): o1 6820±234 (group-diff) vs 6776±226 (group 1/f), archive 6846; o2 6694 vs 6633,
  archive 6651. group 1/f / group-diff white-light ratio rises ~+50 ppm in segment 3 (transit bottom):
  consistent with group 1/f (no scene subtraction) removing a flux-proportional part of the trace
  wings, diluting the transit (not verified further).
- Order-2 transmission median uncertainty (poly/GP): 689/741 group-diff vs 800/810 group 1/f.

## Group-difference 1/f as the overall default (user request, 2026-10-18)
- `detector_calibration(group_difference_1f=True)` (new argument, default True; runner
  `[calibration] group_difference_1f`, default written into the resolved config so cached calibrate
  steps get a new hash). `fit_ramps` = `run_ramp_fit` (first pass) -> `group_difference_1f` (corrects
  ramps, refits without loading/saving STScI ramp-fit products; save=True writes
  `*_groupdiff1f_rateints.fits`). Status key `group_difference_1f`; for SOSS it sets
  clean_flicker_noise SKIPPED. Only modes with `detector_steps['group_difference_1f']` (NIRSpec
  PRISM/G395H/G395M, SOSS); MIRI SKIPPED. `fill_calibration_parameters` default False (so fit_ramps
  without detector_calibration is unchanged). Legacy `stage1`/CDS paths untouched.
- `jwst.group_difference_1f_correction(ramps, rateints, mode, **options)`: SOSS -> box-only
  extract_soss + correct_ramps_1f (options to extract_soss); spectral -> median-rate trace (stage2
  start columns, outlier fix, spline), white-light factors as Stage 2 scale_1f (rows/columns/window,
  after its background subtraction), 1/f levels from correct_1f's inner..outer annulus without its
  negative-index wrap, untraced columns masked around the trace extended at its edge values.
  `jwst_models.ramp_1f_validity` shared by both.
- Tests: tests/test_group_difference_1f.py (PRISM synthetic stripes; sparta end-to-end fit_ramps;
  SOSS CFN swap; defaults). Suite 188 passed, 13 skipped. Commits 3370425 (code), a0b230c (docs).
- [x] Real-data NIRSpec check: validation/group_difference_1f_check.py on PID 1118 NRS1 seg001
  (first pass vs corrected, Stage 2 simple/polynomial/gp) -> validation/runs/nrs1-group-diff-1f.
  First run: polynomial optimal extraction failed ('profile rejection removed an entire column'):
  cosmic-ray showers covering most background rows of a column (int 40, cols 617-619; 24 such steps
  in 10 ints) shifted the column level by ~2000 DN, trace included, because jump-flagged pixels were
  kept in the levels (SOSS int-60 fix). Fix (baee093): `shared_fraction=0.5` - jump-flagged pixels
  excluded from a group's levels unless >50% of its valid background pixels are flagged; new test.
  Rerun, 100-pt median-filter RMS ppm (simple/polynomial/gp): first pass 263.8/272.4/297.2 (archived
  reference simple 264.1), group-difference 159.3/178.9/210.4. Seg001 is pre-transit; the ratio
  corrected/first pass drifts ~65 ppm over 2 h. Shared groups: 0. Peak memory footprint 12 GB.
- [x] SOSS seg001 through the default path (CRDS references for the first-pass box extraction),
  validation/runs/soss-seg1-group-diff-check: shared groups only int 60 groups 2-6; RMS (all ints)
  order 1 117.9/114.0/116.5, order 2 144.5/137.4/135.9 - same as the earlier manual run
  (117.8/114.0/116.4, 144.5/137.3/136.0); first pass = 'none + template' (928.9/864.0/859.5).
- [x] User: no double 1/f correction. `load.group_difference_1f` marks the rates (HISTORY, <72 chars so
  FITS keeps one card); stage2 (spectral + SOSS) skips the rate-level 1/f for marked rates
  (`_rate_level_1f`; metadata `rate_level_1f`). Commits 5c8e095, 65ecc17. Cost, 100-pt RMS ppm:
  NRS1 seg1 simple/poly/gp group-diff + Stage 2 159.3/178.9/210.4 -> group-diff only 166.3/185.9/215.4
  (first pass + Stage 2: 263.8/272.4/297.2). SOSS seg1 (earlier runs): order 1 117.8/114.0/116.4 ->
  119.4/116.7/118.4; order 2 144.5/137.3/136.0 -> 140.9/134.7/134.1.
- [x] User: compare 1/f on groups (standard) vs on group differences (CDS) + SPARTA rates.
  Review of the SPARTA fitter (core/ramps.py) vs ideasrule/sparta calibrate.py: same covariance
  (s + 2R^2 diagonal, -R^2 off) and weights (tridiagonal solve instead of the arccosh closed
  form), one-at-a-time rejection; SPARTA's median-residual template stays opt-in. Changes
  (ca3e68a): per-ramp active set (refit only while rejecting or estimate moves > 1e-3 sigma of one
  difference; results no longer depend on chunk neighbours), float32 rates/variances.
  New `core.detector.difference_1f` (563f014): templates = median difference over ints (jumped
  differences excluded), levels = column median of background residual of each difference
  (shared_fraction rule), removed from groups as cumulative sums. `domain='differences'` in
  group_difference_1f_correction / soss.correct_ramps_1f; flags in detector_calibration
  `group_difference_1f_domain`, `ramp_fit_method`, `ramp_fit_options` (54455b0); runner key
  `[calibration] group_difference_1f_domain`. Validation: validation/cds_1f_comparison.py (one
  Stage 1, first-pass JWST rates for traces, 4 variants, Stage 2 with no rate-level 1/f).
  100-pt RMS ppm. NRS1 seg1 simple/poly/gp: groups_jwst 166.3/185.9/215.4, groups_sparta
  167.4/189.0/242.6, differences_jwst 167.6/187.5/217.3, differences_sparta 168.7/187.8/242.0.
  SOSS seg1 o1 box/poly/gp, o2 box/poly/gp: groups_jwst 119.3/116.5/118.1, 140.9/134.9/134.1;
  groups_sparta 114.7/121.1/122.9, 139.0/134.8/135.0; differences_jwst 120.6/119.1/120.5,
  140.4/135.2/134.7; differences_sparta 118.2/124.0/123.7, 138.5/135.9/135.8. No significant
  white-light gain (RMS sampling error ~6-7 ppm). NRS1 levels: domains disagree by 5-14 DN per
  column, growing with group -> check spectroscopic channels next. Second jump pass on corrected
  differences: deferred by user.
