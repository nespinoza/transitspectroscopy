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
