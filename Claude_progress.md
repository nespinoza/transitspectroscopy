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
- [x] Spectroscopic light curves, NRS1 seg1 (validation/runs/nrs1-cds-1f-spec, 351b195): per-channel
  100-pt RMS at columns (R0) and R=30/100/300, 'corrected' products. All variants within +-0.6% of
  groups_jwst in median per-channel ratio (16-84%: +-1.5-4%), ~half the channels better; excess over
  the error ~1.01 (R0), 1.04-1.14 (binned) for all. SOSS rerun killed by Claude Code during Stage 1
  (low memory); not restarted.
- [x] Images of the group-level 1/f correction (validation/group_1f_panels.py; runs nrs1-1f-panels,
  soss-1f-panels): per group (a) group, (b) minus template, (c) levels, (d) corrected, plus column
  medians over the background pixels. NRS1 seg1 ints 10/100/300; SOSS seg1 has 144 ints, so 10/70/130.
  The recorded call matches the levels recomputed from the templates (1e-4 DN) and (a)-(c) = (d)
  (float32 rounding). Split-half test (background band below vs above the trace, after the template):
  SOSS (133 bkg px/column) bands correlate 0.5 per column, 0.8-0.9 averaged over 4-64 columns;
  NRS1 (10 bkg px/column) 0.1 per column (0.0 in the untraced columns < 450), 0.2/0.4/0.6 over
  4/16/64 columns: the per-column NRS1 level is mostly pixel noise (~20 DN vs ~10 DN coherent);
  rows 0-8 (unused) get worse after the correction (high-pass 24 -> 30 DN). Untraced NRS1 columns
  with clusters of unflagged negative pixels bias levels by up to ~100 DN. Not acted on.
- [x] Literature sweep + data check for a time-domain (pixel-clock) GP 1/f model (no pipeline code;
  scratch scripts in the session scratchpad gp1f/). Timing from headers: TSAMPLE 10 us/pixel (not
  1 us), 12-tick line overhead (jwst NSCleanSubarray nloh=12); NRS1 FASTAXIS=+2 SLOWAXIS=+1
  (t = c*44 + r), SOSS -2/-1 (t = (2047-c)*268 + (255-r)); RAPID, NFRAMES=1, GROUPGAP=0. Groups
  (minus template): pixel-fixed part (kTC/superbias, constant over groups) is 75% (NRS1, sigma 50
  DN) / 50% (SOSS, 31 DN) of the variance, plus line-periodic structure: not stationary in time.
  Group differences or groups centred per pixel over the integration: stationary, PSD white +
  f^-alpha, alpha 1.03 (NRS1) / 0.96 (SOSS), knees ~2 / ~6 kHz; ACF fitted by white + 7-8
  exponentials (celerite real terms) to <1 / 3 DN^2. Hold-out (fake trace band) band-mean rms,
  DN, none / current column median / median on centred groups / GP: NRS1 8.05/7.50/6.20/5.81
  (floor 4.56), SOSS 9.80/5.75/5.58/4.51 (floor 3.05). Neighbouring columns add nothing for SOSS,
  little for NRS1. Awaiting user decision.
- [x] Difference-domain hold-out (scratch gp1f/predict_diff.py): band-mean rms of the group
  differences, DN, none / median on groups then differenced / median on differences / GP per
  difference / GP on centred groups then differenced: NRS1 11.90/10.62/8.80/8.22/8.22 (floor
  6.45), SOSS 14.45/8.36/8.13/6.47/6.47 (floor 4.32). GP per difference = joint model (the
  linear predictor commutes with differencing when masks are shared and frames independent).
- [x] Why SPARTA + GP profiles was worse on NRS1 (242.6 vs 215.4 ppm; runs/nrs1-sparta-gp, scratch
  sparta_gp/). (1) GP optimal extraction returns NaN for a column whose whole aperture is masked
  (mask_dq masks any DQ != 0, e.g. the TSO jump step flags the full trace of a column in an
  integration although its rate is fine); the white light is a nansum, so each NaN is a 600-2500
  ppm dip (JWST rates: ints 6, 64, 109, 354). Polynomial/simple give finite values that the Stage
  2 cleaning replaces. (2) SPARTA writes JUMP_DET for its own rejected differences (~150k extra
  pixel-ints, ~13 trace px/int) -> more masked pixels -> 14 NaN columns instead of 6. Clearing
  those flags: GP 219.1 ppm; dropping the NaN columns in all: jwst 174.7, sparta 179.1, nojump
  178.6 (+3.9 +- 1.0 ppm, block bootstrap); polynomial +1.7 +- 1.7, simple +0.2 +- 0.6. Variances
  of both fits equally calibrated. No pipeline change yet.
- [x] User: no second jump detection in SPARTA after the TSO jump step. 0d4cc25:
  `ramp_fitting.jump_step_options` sets `sigma=None` (unless given) when the TSO jump step ran
  (load.run_ramp_fit via `self.use_tso_jump` from detector_calibration; stage1). Test added; suite
  194 passed, 13 skipped. NRS1 seg1 rerun through the pipeline (runs/nrs1-sparta-gp/sparta_tso):
  provenance sigma None, 0 rejections, JUMP flags identical to JWST's (32964). White light
  (as is / NaN columns dropped) vs JWST: simple 166.9 vs 166.3 (+0.6+-0.6), polynomial 188.0 vs
  185.9 (+2.1+-1.6), GP 219.5/179.4 vs 215.4/174.7 (+4.7+-1.0). GP gap not from jump flags;
  unexplained. Remaining DQ differences: JWST sets NO_GAIN_VALUE (non-trace) and SATURATED on 42
  trace pixel-ints that SPARTA does not. Stage 2 NaN-column issue (mask_dq + GP) still open.
- [x] User approved the Stage 2 NaN fix (GP gap of ~5 ppm: not a concern for now). 9eddb7c:
  `jwst_models.dq_masked` (any DQ but JUMP_DET) used by mask_dq in stage2 (image filling and
  the GP mask) and soss.reduce_soss; the Stage 2 cleaning also replaces non-finite spectra in
  `corrected`. Suite 195 passed, 13 skipped. NRS1 seg1 Stage 2 rerun (runs/nrs1-sparta-gp,
  stage2fix_*): no NaN spectra; white light old -> new, JWST rates: simple 166.3 -> 165.2,
  polynomial 185.9 -> 185.0, GP 215.4 -> 174.6; SPARTA (sigma None): 166.9 -> 165.5,
  188.0 -> 186.9, 219.5 -> 179.2. SPARTA - JWST: +0.4+-0.5, +1.9+-1.6, +4.6+-1.2.
- [~] GP (readout-time) 1/f experiment (user: run it, watch memory, delete regenerable run
  products). validation/runs cleaned 21 GB -> 294 MB (kept png/json/log/txt). celerite2 0.3.3
  installed in the py313 env for the experiment only (not a package dependency).
  validation/gp_1f.py: `gp_difference_1f` replaces core.detector.difference_1f (same templates,
  pixels and shared rule; 5-sigma column clipping + 4-neighbour dilation); kernel white + NNLS
  exponentials fitted to the masked ACF (timescales to `kernel_lines`=200 columns) of 16 ints;
  celerite2 predictive mean at every pixel (exact vs dense to 1e-14; ~5 ms NRS1, ~40 ms SOSS
  frame; never return_var: dense covariance); hold-out band (NRS1: band below the trace; SOSS
  rows 200-215) with per-column and band-sum errors. validation/gp_1f_comparison.py: one variant
  per process (groups_jwst, differences_sparta, gp_sparta), ramps freed before Stage 2, rates
  deleted, --summarize. NRS1 seg1 (runs/nrs1-gp-1f): groups_jwst 165.2/185.0/174.6 ppm,
  gp_sparta 172.2/194.0/185.0 (+7.1+-1.8, +9.0+-2.2, +10.5+-2.0) but per-channel RMS better:
  R0 ratio 0.979-0.984, R100 0.989-0.995, R300 0.983-0.987, R30 ~1.00. Kernel white 73.4,
  correlated 43.0 DN^2; hold-out band-mean rms none 7.74 / median 6.98 / GP 5.41 (floor 3.85).
  Peak memory 4.6 GB per variant. Restarted on request (scratch run_gp1f_rest.sh).
  differences_sparta 167.8/187.7/180.1 (+2.7+-2.5, +2.7+-2.8, +5.5+-2.3 vs groups_jwst), channel
  ratios ~1.00. GP vs differences_sparta: white +4.4+-1.6/+6.4+-1.9/+5.0+-1.6 ppm, but R0
  ratio 0.984-0.987 (92-96% of columns < 1), R100/R300 0.989-0.992. Hold-out (no correction,
  every 5th int, runs/nrs1-gp-1f/holdout): kernel 50/200/1000/4000 columns identical: band mean
  median 6.96 / GP 5.41 (floor 3.84); band sum per pixel none 0.264 / median 0.220 / GP 0.216
  (floor 0.086). So the hold-out does not reproduce the white-light loss; the per-integration
  difference (24-30 ppm rms, lag-1 corr 0.1-0.2) is slow. Hypothesis (untested): GP carries
  flux-correlated residuals of the background pixels nearest the trace (PSF wings, template x
  smoothed white-light factor) into the trace edge pixels. SOSS gp_sparta killed by Claude Code
  (system low memory) during its first ramp fit; not restarted.
- [x] GP detector-effect exploration (user: GP cost not worth it; try modelling other Rauscher
  effects). Scratch gp_explore.py on NRS1 seg1 (runs/nrs1-gp-explore: diagnostics.json,
  variants.json, proxy.npz, delta.npz, logs). Diagnostics of the difference background residuals:
  ACN small (even-odd variance +19% over white; within-column ACF -1.2/-4.2 DN^2 at odd lags vs
  73 white); no readout-asymmetric ringing/crosstalk (corr ~0.005 before and after the trace);
  a per-row pattern common to all columns (sigma ~1.1 DN/row/difference, rows independent, corr
  ~0.2 with the reference columns). gp_1f.fit_kernel gained `frequencies` (ComplexTerm
  exp(-t/T)cos(wt); ACN at w=pi) and `periodic` (line harmonics 2pi m/L, row pattern) and a
  fitted non-negative white term. Hold-out band sum (median 0.231, floor 0.086): gp 0.224,
  gp_acn 0.175, gp_rows 0.188, hybrid (median + GP on its residual) 0.199. White-light proxy
  (aperture sum of the per-group 1/f change / flux) validated on the measured gp_sparta vs
  differences_sparta: corr 0.82 (simple), predicts 167.8 -> 173.5 vs measured 172.2. Proxy
  white light (simple, vs differences_sparta 167.8): median_clipped -0.6+-0.3, gp +5.7+-1.9,
  gp_acn +6.3+-1.9, gp_rows +7.3+-1.9, hybrid -0.6+-0.3, gp_gap2 -7.4+-3.4, hybrid_gap2
  -7.7+-2.5. So the detector terms improve the noise model but not the white light; the only
  white-light gain is a 2-row gap between the trace and the background (hybrid_gap2 ~ its
  median, so the gap, not the GP). Next (not run): median with gapped background through Stage 2.
- [x] White-noise floor of the NRS1 seg1 white lights (user hunch: GP may improve the scatter
  around the slow systematics). Hann periodogram of the white light minus a 100-point median,
  band rms (ppm, simple; vs differences_sparta, 40-point block bootstrap): high (0.25-0.5/int,
  <76 s) 130.3; gp_sparta -0.1+-2.1 (polynomial +4.6+-2.9, gp -0.1+-2.9); point-to-point
  +1.0+-1.3. The GP excess is at low frequency (<0.065/int: +15+-9) and in a periodic feature:
  all variants show a ~89 ppm, 10.75-integration (204 s) oscillation (band 0.065-0.1/int rms
  ~300 ppm, the largest PSD feature). Proxy variants: no change of the floor (gp_acn +3,
  gp_rows +7, i.e. proxy noise ~+3); the gap-2 variants lower mainly the 204 s band (-23+-14,
  -31+-12). Plot runs/nrs1-gp-1f/psd_white_light.png; scratch psd_check.py, psd_bands.py.
- [x] The 204 s white-light oscillation (user: study it). Scratch osc_frames.py reads the NRS1
  seg1 uncal memory-mapped (no Stage 1; 15 s): column-median background (rows >= 8 from a fixed
  trace), 6-sigma group-difference clipping, per-integration spectra, trace y centroid and width,
  white light for aperture half-widths 1-9 rows, row profile, reference columns, bias.
  runs/nrs1-oscillation/frames.npz (10 MB) and oscillation.png; osc_analyse.py. Result: PSF
  breathing. Trace width oscillates at 10.76 int (204 s = 3.4 min; 0.29 mpix, 10 sigma, LS power
  56x median); flux moves between the core row and rows +-2..3 in anti-phase. y centroid
  oscillates at 20.5 int (389 s = 6.5 min, 0.9 mpix, 8 sigma); no x shift, bias, reference
  column or background signal. These are the 3.4 and 6.5 min periods of the NIRSpec
  commissioning paper (arXiv:2211.01459), attributed to IEC heater cycling (astigmatism,
  STPSF jwst_iec_wfe). Amplitude vs half-width (raw): 1 row 203, 2 rows 50, 3 rows 15, 4-9 rows
  25-30 ppm; raw white rms (100-pt) min at 3 rows (134 vs 139 at 2). Pipeline (radius 2) 89-111
  ppm; regression on the encircled-energy ratio EE(1)/EE(9) cuts it to 40-46 ppm and the rms
  172 -> 164 (simple), 198 -> 185 (polynomial), 189 -> 177 (gp); width alone only to 64-79.
  Side finding (open): the raw half-width-2 white light is 139 ppm vs pipeline simple 172 ppm
  on the same metric (difference 101 ppm rms, lag-1 corr 0.35, not cosmic rays).
- [x] Source of the ~100 ppm extra white-light noise of the pipeline vs the raw (uncal) box
  extraction (user: find it). NRS1 seg1, box r=2, 100-pt rms (ppm): raw 140, pipeline 172-175.
  Scratch: source_stage1.py, source_bisect.py, source_bands.py, source_pixels.py,
  source_emulate.py, source_pedestal.py, source_linearity.py, source_rawlin.py; small npz in
  runs/nrs1-oscillation (rates deleted, 8 min to regenerate). Findings:
  (1) Stage 2 adds nothing: the same box on the Stage 1 rates gives 175, corr 1.00 with the
  pipeline. The extra is a fractional mode common to all columns (corr 0.94), not DQ/jumps/CRs,
  not the aperture shape, not the 1/f background rows (raw with rows 3-8: 143).
  (2) Offsets removed after linearity: the per-integration pedestal (frame-wide, 9-11 DN rms,
  not tracked by the refpix constant) goes through the linearity curvature; measured +7.0
  ppm/DN, predicted +6.9 ppm/DN from jwst_nirspec_linearity_0024 and the trace counts. Scratch
  patch removing a pedestal (median of columns 100-450 per group) after refpix: 173 -> 163.
  The per-column, per-group levels L x (f'(c)-1) predict the rest of this term (corr 0.63);
  subtracting the prediction: 163 -> 153.
  (3) Linearity itself on clean raw data: 140 -> 158 (r=2), 133 -> 144 (r=3); the fast band
  rises 110 -> 120-124 for all apertures (unexplained); the 204 s band rises at r=2 (197 ->
  238; uncorrected nonlinearity was hiding part of the breathing loss) but falls at r=3-4.
  (4) The 1/f template white-light scale (whole-frame sum of the first-pass rates) is mostly
  1/f: 830-1380 ppm rms, corr 0.14 with the true white light; ~12 ppm effect (emulation).
  (5) Group-difference background rows are asymmetric about the trace ([int(c)-3, int(c)+3)
  excluded; e.g. centre 13.95 keeps row 16 at +2.05 px).
  Proposed (not done): remove the 1/f levels and pedestal before linearity (or subtract
  level x f'(counts) after it), aperture radius 3, scale from the aperture of 1/f-corrected
  rates, symmetric background rows; then check the linearity fast-band increase.
- [x] 1/f before linearity (user: run the test; if it lowers the white-light rms, migrate).
  Test (scratch prelinear_test.py, variant groups_jwst_prelinear of gp_1f_comparison, NRS1
  seg1, default groups domain + JWST ramp fit, r=2): levels measured as before, then each
  group rebuilt as f(f^-1(data + L) - L) (Newton inversion of the CRDS polynomial; 0.07% of
  group pixels not inverted, kept). White light 165.2/185.0/174.6 -> 154.7/177.6/165.9 ppm
  (-10.5+-2.9, -7.3+-2.6, -8.6+-2.8; simple/polynomial/gp); channel ratios 0.995-1.001; peak
  3.2 GB; correction +66 s. Migrated (0e01420): core.detector.linearity_polynomial and
  offsets_before_linearity, jwst_models.linearity_coefficients (CRDS file from
  meta.ref_file.linearity, subarray cut, NO_LIN_CORR/non-finite -> identity),
  group_difference_1f_correction(before_linearity=True default; both NIRSpec and SOSS paths;
  diagnostics before_linearity, linearity_not_inverted; skipped without a linearity
  reference). Tests: exact inversion, 30x lower slope error on a curved synthetic scene,
  no-reference None. Suite 198 passed, 13 skipped. Docs: CHANGELOG, nirspec and soss guides.
  Not validated on SOSS (memory). Still open: aperture radius 3, the 1/f template scale,
  symmetric background rows, linearity fast-band increase.
- [x] Literal move of the 1/f step before the linearity step (user: test whether it gives the
  same results as the inversion). Scratch moved_test.py: first pass (Stage 1 + JWST ramp fit;
  rates and groupdq kept in validation/runs/nrs1-gp-1f/first_pass), then Stage 1 again with
  LinearityStep.call wrapped to run group_difference_1f_correction(before_linearity=False) on
  the refpix-corrected ramp (first-pass rates and groupdq), then linearity, dark, final fit.
  White light 152.8/175.1/163.2 ppm (moved - inversion: -1.8+-2.0, -2.5+-2.7, -2.7+-1.7).
  But the rates gain +0.78 DN/s in every pixel (rows uniform; pixel-level difference after
  removing column offsets 0.037 DN/s; per-integration column scatter 0.49 DN/s; per-column
  light curves differ by ~930 ppm; DQ and NaNs identical; not the dark, corr 0.02). Cause:
  core.detector.group_difference_1f scales one difference template per group by a factor
  fitted in the aperture (group_scale); on raw counts the bright aperture is non-linear
  (group_scale 1.009 ... 20.94 for groups 1-20), so the template over-subtracts the fainter
  1/f background pixels by a growing amount, the levels get a -0.80 DN/s slope, and removing
  them adds it to every column. Stage 2 background subtraction (static median) hides the
  constant part. The literal move needs a template valid on raw counts (e.g., the median of
  every group, no fitted factor), and two Stage 1 passes (peak 4.7 GB vs 3.2 GB).
- [x] Literal move with a per-group median template (user: try it). Scratch moved_test.py
  `moved median keep`: as the literal move, but core.detector.group_difference_1f replaced
  by a copy whose template of every group is the median over integrations of that group
  divided by the white-light factors (no fitted group_scale). The uniform offset is gone
  (levels flat over groups, slope 0.001 DN/s; background row medians as the first pass).
  White light 153.1/174.9/164.7 ppm (vs inversion -1.5+-1.5, -2.7+-2.4, -1.3+-1.4). Per-
  channel RMS ratio to the inversion: R0 0.991-0.993 (16-84% 0.976-1.004), R100 0.994-0.998.
  Far-background (rows 0-5, 28-31) per-integration column scatter: first pass 0.29, inversion
  0.52, moved median 0.41 DN/s. Pixels vs inversion: identical DQ/NaNs, per-integration
  column offsets differ by 0.50 DN/s robust sd (level measurement), pixel level 0.024 DN/s.
  Not separated yet: the median template alone (after linearity) vs the move.
  Rates kept: first_pass, groups_jwst_prelinear, groups_jwst_moved_median (~1.5 GB).
- [x] Median template after linearity (user: run it). Scratch moved_test.py
  `inversion_median`: the committed correction (levels after linearity, removed before it by
  inversion) with the per-group median template; variant groups_jwst_prelinear_median.
  White light 154.2/175.4/165.4 ppm. Same template, inversion vs literal move: rates agree
  (column offsets 0.018 DN/s robust sd, pixels 0.001 DN/s), white light +1.0+-0.6,
  +0.5+-1.1, +0.7+-0.6 ppm, channel RMS ratios 0.998-1.000. Median template vs scaled
  difference template (both by inversion): white -0.5+-1.2, -2.3+-2.3, -0.5+-1.2 ppm;
  channel RMS R0 0.991-0.992 (16-84% 0.975-1.003), R100 0.995-0.996; far-background column
  scatter 0.52 -> 0.40 DN/s. So the move and the inversion are equivalent; the channel gain
  is the template. Peak memory of this run 4.6 GB (run_variant, as the literal move's 4.7 GB),
  so the earlier 3.2 GB of groups_jwst_prelinear is not a like-for-like comparison.
- [x] Median template + 1/f before linearity as the default (user: adopt, make default,
  commit). f57f497: core.detector.group_difference_1f(template='median' default |
  'difference'; scale_pixels optional); jwst.load.group_difference_1f rebuilds the input
  ramps up to linearity (_ramps_before_linearity: steps whose input status was None and are
  COMPLETE, products not saved; first-pass groupdq kept), corrects, then linearity, post-
  linearity and post-jump steps (_ramps_after_linearity), refit; falls back to correcting
  the calibrated ramps if the inputs were linearized, linearity did not run, or
  template='difference'. diagnostics['before_linearity']. _run_step helper used by
  detector_calibration. Removed the inversion (offsets_before_linearity,
  linearity_polynomial, linearity_coefficients, before_linearity). Tests: median template on
  non-linear ramps, class flow with stand-in steps, fallback; suite 198 passed, 13 skipped.
  NRS1 seg1 package run (variant package_default): rates identical to the scratch
  groups_jwst_moved_median (0.000 DN/s), white light 153.1/174.9/164.7; Stage 1 peak 5.1 GB.
  a833620: _reload_status keeps group_1f, group_difference_1f and SKIPPED statuses when
  detector_calibration reloads saved linearity/jump products (a save=True rerun skipped the
  correction). Not validated: SOSS (memory). Rate files of the variants deleted.

## Self non-linearity correction, NRS1 HAT-P-14 b seg1 (2026-10-22/23, in progress)

Scratch scripts nl_*.py; products in validation/runs/nrs1-nonlinearity/seg001 (not committed).
- [x] Per-pixel non-linearity of the median ramps (391 integrations, raw = 1/f-corrected ramps
  before linearity, dark-subtracted). Late/early slope change of trace pixels > 500 DN: raw
  +0.08% (robust sd 1.9%), CRDS linearity +1.35% (2.1%). Raw cores sublinear (to -10% at
  26k DN), shoulders superlinear at the same counts; CRDS residual = -2.66 own/1e4 + 2.60
  neighbours/1e4 (%): charge sharing. Aperture sums: CRDS +0.94% superlinear; predicted
  transit-depth factor of CRDS 1.0056 (16-84% 1.0030-1.0083), ~35 ppm on HAT-P-14 b.
- [x] Self correction (h(x) = x + a2 x^2 [+ a3 x^3 > 5000 DN], median ramp, OLS, outlier mask,
  CRDS c1 as gain): slope-change sd 0.33% in sample, 0.75% even/odd cross-validated.
- [ ] Light curves of the self/residual variants, seg2/3 out of sample: nl_chain_all.sh was
  killed for low memory (raw copy); fixed (memmap of raw.npy), not restarted.
- [x] Modelling study (user: Chebyshev, bias per integration, up-the-ramp correlations).
  nl_study_extract.py + nl_study.py {noise, neighbours, bias, fits, sim}; logs in
  scratchpad/nl_logs/study_*.log. Cross-validation: fit on even integrations, test on odd.
  Noise: read 7.08 DN/group; difference covariance = read (lag 1, -sigma^2) + Poisson, lags
  2-4 < 1% of the variance. Photon transfer (var/mean of differences, late/early): core
  -7.1% vs mean -8.9% (>15k DN); shoulders var/mean -0.4 to -2.6% while the mean is +1.4 to
  +1.5%: the shoulders' superlinearity is charge arriving from neighbours. Horizontal
  neighbour correlation 1.6-4% (IPC), no early-late growth; vertical 6-17% (shared column
  noise). Bias: per-pixel offset per integration ~30 DN rms (reset-like), common mode 1.2 DN
  rms; zero charge at group -0.99 (reset one frame before group 0). Impact on the white light
  -3.5 ppm per DN of common offset: 4 ppm (common) and 3 ppm (per-pixel) per integration.
  Fits (held-out chi2/18 by 500-2k/2-5k/5-15k/>15k DN; null 1.03): CRDS 1.7/6.1/54/20; median
  OLS cubic 1.8/1.7/1.6/1.5; mean GLS cubic 1.6/1.5/1.4/1.3. Chebyshev vs power: same fit
  (held-out metrics identical), condition 1.3-16 vs 160-4e7, coefficient correlation
  0.13-0.82 vs 0.99. Degree: 2 leaves curvature > 15k DN (chi2 2.5), 4-6 double the
  rate-factor scatter and worsen the held-out chi2. Joint fit (all ramps, free intercept and
  slope per integration) biased for faint pixels (errors in variables): slope change -0.47%
  at 500-2000 DN; simulation -0.7% rate, depth factor 0.9936. Mean ramp + GLS (read +
  Poisson covariance) unbiased in the simulation, scatter 0.75/0.30/0.17/0.11% (median OLS
  1.12/0.47/0.27/0.16%); empirical covariance = parametric. Background pixels: held-out
  chi2 1.6 from a static per-pixel group pattern (1.2 DN rms, reproducible between halves)
  missing from the dark reference: the chi2 floor, not non-linearity. Depth factor per
  column (truth: the other half): CRDS 1.0064, self 1.0000 +- 0.0010 (median OLS 0.0015).
- [x] Self correction switched (user: covariance-weighted Chebyshev, degree 3 for every pixel,
  so coefficients are comparable): nl_analyse.py fits the mean ramp (GLS, read 7.08 DN +
  Poisson 1.44 e/DN covariance), h = (x + a2 (top/2) T2 + a3 (top/2) T3 - zero)/slope on
  [0, top = 1.02 x final], h(0) = 0, h'(0) = 1; rate-factor outlier mask kept, CRDS c1 gain.
  Seg1: reliable 10937/10970 pixels > 500 DN; slope change in sample +0.01% (sd 0.16%),
  cross-validated -0.00% (sd 0.45%; median OLS 0.75%). Aperture: self +0.01%, CRDS +0.93%;
  CRDS depth factor 1.0056 (unchanged). Old fit kept as scratch nl_analyse_median_ols.py.
- [ ] nl_chain_all.sh restarted 2026-10-23 with these coefficients (seg1-3 variants, Stage 2,
  own fits of seg2/3). First attempt: seg1 self Stage 2 failed ("profile rejection removed an
  entire column", integration 336): a cosmic-ray jump took pixel (19, 658), mean ramp 780 DN,
  to 16000 DN, where the cubic h gave -2e5. h is now extended outside its fitted range [0,
  top] by the straight line with h's value and derivative at the end; variants rerun.
  Chain killed by Claude Code for low memory during seg3 variants (seg1, seg2 complete; seg3
  has raw.npy and rates_jwst only).
- [x] Seg1+2 results (user: run on seg 1 and 2). White light (100-pt rms ppm, simple/poly/gp):
  seg1 jwst 153.1/174.9/164.7, self +3.8+-1.2/+9.2+-3.4/+7.2+-2.8, residual +4.7/+8.2/+8.2;
  seg2 jwst 147.3/173.5/163.1, self +3.4+-0.8/+3.9+-3.7/+9.4+-1.7, residual +3.7/+9.4/+8.0.
  Channel rms ratios to jwst 0.996-1.006 (R0/R100/R300 slightly below 1, R30 slightly above).
  self/jwst white-light ratio: rms 14-22 ppm, slow (lag-1 0.75-0.83, point-to-point 7-10
  ppm); seg1 integrations 42-46 dip 100-180 ppm (not yet explained). Depth / jwst (seg1+2,
  nl_depth.py): self 0.9925+-0.0031 (simple), 0.9947+-0.0037 (poly), 0.9909+-0.0046 (gp);
  residual 0.9951/0.9970/0.9940 (+-0.003-0.004); consistent with CRDS 0.5-0.6% too deep.
  Transit shape poorly constrained without seg3 (reference depth 4562-5861 ppm by method).
  Transfer (nl_transfer.py, seg1 h on seg2 mean ramps and back): slope change median -0.03%,
  sd 0.35% (own fit 0.16%, uncorrected 1.4%); rate other/own fit +0.01% (sd 0.5%).
  Coefficients seg1 vs seg2 (same pixels): a2 correlation 0.991 (1.000 above 10k DN, 0.56
  below 3k DN), a3 0.45 (0.96 above 10k DN), slope 0.95, zero/top 0.98.
- [x] Seg3 run (user: run seg 3); stage 1 peak 4.6 GB. Seg3 white light (100-pt rms ppm,
  simple/poly/gp): jwst 164.5/181.9/178.8, self +3.7+-0.9/+7.6+-4.3/+7.3+-1.5, residual
  +4.5/+13.4/+7.5; channel rms ratios 0.996-1.008. Own fit: slope change self +0.00% (sd
  0.16%), crossed sd 0.49%. Transfer seg1 or seg2 h -> seg3: sd 0.33-0.34%, rate +0.004%/
  -0.009%; coefficient correlations as seg1-seg2 (a2 0.991, a3 0.45; > 10k DN 1.000/0.96).
- [x] nl_depth.py transit fit fixed: unbounded and in absolute BJD it did not converge (seg1-3:
  rp < 0; seg1+2 numbers above unreliable). Now bounded, time relative to the data middle,
  best of 6 starts: rp 0.0823, a 9.15, inc 84.3, b 0.91, residual rms 156 ppm. Depth / jwst,
  seg1-3, simple/poly/gp: self 0.9954+-0.0011 / 0.9964+-0.0022 / 0.9944+-0.0017; residual
  0.9969+-0.0010 / 0.9971+-0.0020 / 0.9962+-0.0015. Self agrees with the aperture-sum CRDS
  factor 1.0056 (1/1.0056 = 0.9944): CRDS makes the transit ~0.45% (~30 ppm) too deep. R100
  self factors flat with wavelength (quartiles 0.9948-0.9966; slope -2.0+-1.2 per mille/um).
- [ ] NRS2 study (user: same study on NRS2; NRS1 raw/groupdq/rates deleted for space, 11.5
  GB). Scratch scripts take NL_DETECTOR / NL_COLUMNS; nl_chain_nrs2.sh. NRS2 Stage 2 failed
  first ("trace/aperture extends outside the detector"): the trace rises from row 12.6 to
  30.5 of the 32-row subarray, the optimal aperture leaves it at column 1968, and the range
  detector (a port of the original C RangeDetector) only handles a rising trace leaving
  through row 0. Package unchanged; nl_stage2.py re-registers G395H with NRS2 xend 1900 (open
  question for the package: NRS2 G395H default xend 2043 fails on this target).
  Seg1 (columns 5-1899): raw slope change -0.74% (aperture sum -1.85%), CRDS +0.51% (sum
  +0.41%; core +0.3%, shoulders +0.6 to +1.7%), self -0.02% (sd 0.20%, crossed sd 0.52%);
  reliable 10347/10360. Predicted CRDS depth factor (aperture) 1.0030 (NRS1 1.0056). White
  light (simple/poly/gp): jwst 208.4/215.0/215.0, self -1.0+-0.6/-1.4+-0.7/-1.2+-0.8,
  residual -0.8/-1.0/-0.8; channel rms ratios 0.994-0.998. Chain killed by Claude Code for
  low memory during the seg2 extraction (seg1 complete).
  Restart (user: yes): seg2 complete, killed again for low memory during the seg3
  extraction. Seg2: raw slope change -0.66%, CRDS +0.55%, self -0.00% (sd 0.20%, crossed
  0.53%). White light (simple/poly/gp): jwst 203.6/209.7/209.9, self -1.1+-0.5/-0.7+-0.5/
  -0.6+-0.5, residual -0.7/-0.3/-0.5; channel rms ratios 0.995-0.998. Transfer seg1 h ->
  seg2: slope change +0.03% (sd 0.39%; own 0.20%), rate other/own fit -0.09% (sd 0.57%;
  NRS1 +0.01%); a2 correlation 0.958 (0.994 above 10k DN), a3 0.44.
- [x] NRS2 seg3 (user: yes), chain complete. Seg3: raw -0.58%, CRDS +0.63%, self +0.02% (sd
  0.20%, crossed 0.54%). White light: jwst 195.7/202.3/203.0, self -1.1+-0.7/-0.7+-0.6/
  -1.1+-0.6, residual -0.4/-0.1/-0.6; channel rms ratios 0.995-0.998 (NRS2 self lowers the
  noise slightly on every segment; NRS1 raised the white light by 3-9 ppm).
  Depth / jwst, NRS2 seg1-3 (rp 0.0833, a 9.21, inc 84.35), simple/poly/gp: self
  0.9962+-0.0007 / 0.9961+-0.0007 / 0.9962+-0.0007; residual 0.9967 / 0.9966 / 0.9967 (+-
  0.0005). Predicted from the aperture sums 1/1.0030 = 0.9970. R100 quartiles 0.9951-0.9974,
  slope +1.2+-0.7 per mille/um. Transfer: seg1 h on seg3 rate -0.07% vs own (seg1 on seg2
  -0.09%), seg2 h on seg3 +0.01%: the seg1 (out-of-transit, first) fit differs from the later
  ones by ~0.08% in rate (uniform per segment, so absorbed by the per-segment baselines);
  a2 correlation 0.957, a3 0.43.
- [x] Figures (scratch): nl_levels.py (bright / half / faint pixels: CRDS and self corrections,
  line-fit residuals; seg1 of each detector) and nl_tso.py (white light seg1-3, simple
  extraction, CRDS vs self, transit fits, residuals, ratio, rms vs bin). Transit residual rms /
  pipeline error (white-light error from the R0 column errors, 1/sqrt(sum 1/sigma^2)): NRS1
  CRDS 155.4/106.5 = 1.46, self 159.2/107.5 = 1.48; NRS2 CRDS 202.7/143.1 = 1.42, self
  201.9/143.6 = 1.41 (columns: excess ~0.99, so the white-light excess is common to the
  columns). The self/CRDS dip (100-180 ppm) is at integrations 42-48 of seg1 on both
  detectors: an event in the exposure, not in one detector's fit (not yet explained).
- [x] Viewer nl_inspect.py (scratch; user: image + light-curve residuals with integration and
  group sliders). The self/CRDS dip (seg1 integrations 42-48) found with it on NRS2: a
  two-row band of positive offset in raw - dark, the same in every group (an additive offset of
  the whole ramp, not charge), sweeping down one row per integration from row ~29 (integration
  30) to row 0 (~60) and growing from ~80 to ~290 DN; it crosses the trace rows 12-17 at
  integrations 41-47. Trace position unchanged (shift < 0.01 px). An offset c changes the
  linearized slope by about c x the curvature of the correction, which differs between CRDS
  and self, hence the dip in their ratio (and a bias in both). Not corrected by the reference
  pixel step. Same integrations on NRS1 (raw deleted, not inspected).
- [x] Ramp viewer in the package (user: utils and the dataset object; groups and integrations;
  user-selected ramps): `src/inspection.py` (`RampViewer`, `inspect_ramps`, `estimate_trace`;
  exported from `ts.utils`), `ts.jwst.load.inspect_ramps(stage, segments)` with stage
  'calibrated', 'input' or any detector step (input ramps calibrated again up to it, without
  saving; `_apply_step` factored out of `_run_step`). Middle panel: aperture mean and
  background median of the image mode against the integrations or the groups ('a'), with new
  jumps. Guide docs/guides/inspecting-ramps.md, API entry, changelog. tests/test_inspection.py
  (moving band: in the aperture only while it crosses, in group 0 of the increments only).
  Real NRS2 seg1 raw (391 x 20 x 32 x 2048, memory map): open 7 s, groups axis first time 10 s,
  1.5 GB peak.
- [x] Moving row band, correction study (user: row term in the 1/f correction, training from
  integrations off the trace, literature). Literature: Albert et al. 2023 (NIRISS/SOSS,
  arXiv:2306.04572, sec. VII.2.1): two noisier rows (sigma ~100 adu) moving one row down per
  integration, linked to the full-detector reset after each subarray integration; not
  corrected there. NRS2 seg1 (scratch nl_band*.py): two rows, offset +20 DN (int 29) to +360
  DN (int 59), smooth in time (~11 DN/int; trained quadratic predicts 41-48 within 5-8%), the
  same in every group, gone at 61; uniform along the row in its median (+-15 DN) but with
  pixel scatter 115-205 DN (normal 25), not repeatable between integrations (correlation
  ~0.05), so only the row mean can be trained. Per-pixel reset level (line through groups 0-3
  to group -1) removes it to 6-8 DN. Physics: within-row regression (slopes from groups 6-19,
  disjoint from the reset estimate) of the raw-slope change on the change predicted by the
  CRDS curve from each pixel's own offset: 0.80 +- 0.16 (0 if electronic; ~0.67 for the self
  curvature): the offset is in the pixel (reset level), so linearity must see it. Hence
  subtracting it before linearity (row, trained, reset variants) is wrong: lin/raw at 41-47
  goes from +219 (CRDS) / +147 (self) to ~0 and leaves a -180..-240 ppm white-light dip.
  The band biases the 1/f column levels (median of rows 3-8 from the trace): other rows -2
  DN (-4 max) during the band; re-measuring without the band rows: +53 ppm raw at 41-47, +23
  ppm through linearity. Row term as its variation over groups only (keeps the band): white
  rms 198 -> 174 ppm. 1/f levels from all rows > 6 from the trace (~19 vs ~10 pixels per
  column): white rms 198 -> 160 ppm on clean integrations (not band related). self/CRDS at
  41-47 stays -73..-82 ppm with the correct treatments (the two curves differ at offset counts).
- [x] 1/f background rows sweep, NRS2 seg1 (user: rows 3-10 from the trace, a few to all
  rows beyond; scratch nl_onef_rows.py, nl_onef_rows_plot.py, figure
  onef_rows_nrs2_seg001.png). Levels re-measured on raw.npy (already package-corrected; the
  median being shift-equivariant, this replaces the package's levels) from inner <= |row -
  trace| < inner + width, inner 3..10, width 1, 2, 3, 4, 6, 8, 12, all. CRDS white rms, ints
  25-65 excluded: package pixels (3-8, ~10 px/col) 164.5; needs >= ~10 px/col (width 1: ~900-
  1160, width 2: 360-680, width 3: 190-440); best 4-all 158.5 and 3-all 159.3 (-6.0 +- 2.9 and
  -5.2 +- 2.7 ppm against the package pixels, bootstrap); inner >= 8 worse (one-sided, fewer
  pixels: 10-all 170). Binned (10) rms 51.4 -> 49.2. Correction of the previous entry: the
  198 -> 160 was mostly not the rows. The package correction is not idempotent: applying
  core.detector.group_difference_1f again to its own output (same pixels, white-light
  factors, masks; nl_onef_package.py) gives 198.1 -> 165.3 ppm (second-pass levels vary by
  0.4-0.6 DN per column in time); template from corrected ramps. Not explained yet. Package
  pixels are asymmetric (int(center) +- 3: at column 0, centre 12.63, row 15 at 2.4 px is a
  background pixel and row 9 at 3.6 px is not); package white-light factors differ from
  aperture-sum ones by 570 ppm rms (correlation 0.31), with no rms effect.
- [x] Why the second 1/f pass helped (user: template noise or white-light factors? passes
  3-5?). NRS2 seg1 Stage 1 rerun with the checkout (scratch onef_extract.py; 440 s, 4.7 GB):
  the uncorrected input of core.detector.group_difference_1f saved
  (validation/runs/nrs2-onef/seg001/orig.npy, 2 GB, and onef_inputs.npz: masks, pixels,
  factors, levels). Pass n = one pass on the input with the template of the pass n-1 output
  (median shift-equivariant), scratch onef_passes.py; pass 1 reproduces the package levels
  exactly. CRDS white rms (ints 25-65 excluded): package factors, passes 1-5: 198.1, 198.3,
  198.3, 198.3, 198.4 (levels change 0.41, 0.17, 0.08, 0.04 DN): more passes do not help;
  template perturbations (T1 - T2 column-common or per-pixel, shuffled) change nothing.
  Factors: aperture-sum 165.8, unity 163.7 (seg1 is out of transit), package factors median
  filtered over 51: 164.0 (25: 167.9), iterated (factors from the white light of the
  previous pass, window 5; onef_scales.py): 164.9, 165.0, 165.0. Cause: the package factors
  (first-pass rates, full frame, window 5) scatter by 2678 ppm about a smooth curve; the
  white light moves by 0.040 ppm per ppm of factor error (correlation 0.9997): 0.04 x 2678 =
  107 ppm, in quadrature with 164 gives 196. The earlier "second pass" gain came from
  factors of the final (1/f-corrected) rates (541 ppm scatter), and the earlier "factors do
  not matter" test used those, not the package's. Full-frame last-first sums: 9238 ppm
  scatter uncorrected, 5471 ppm after the 1/f levels.
- [x] Group 1/f study for PRISM (PID 3969 obs 1, Kepler-12, NRS1 SUB512, 5827 x 5 per
  segment; seg1 and seg3 downloaded to validation/data/prism_3969) and SOSS (PID 1541 obs 1,
  seg1 out of transit, seg3 in transit). Scratch onef_extract.py (both 1/f code paths; saves
  the uncorrected input, masks, pixels, factors, levels, geometry), onef_study.py (passes,
  factors, sweep with iterated factors), onef_summary.py (rms about a 51-integration running
  median, 5-sigma clipped; bootstrap errors). Pass 1 reproduces the package levels exactly
  in all four. Uncorrected white light (no linearity). Inputs deleted after each study.
  Passes: no effect anywhere (template converged). Factors: PRISM (package window 200)
  775/786 ppm for package, iterated and unity alike (coupling 0.027 wide aperture, 0.007
  tight); SOSS (package factors from the order-1 aperture of the first-pass rates, no
  filter) package 111.6/109.0, iterated 112.4/109.7, unity 110.6/108.0 (seg1 clipped at
  integration 60, +1% outlier; coupling 0.015, so unity in transit would lower the depth by
  ~1.5%): iterating does not help SOSS (feeds white-light noise back). Pixels: SOSS flat (all
  inner 15-47 x widths >= 8 within +-1.5 ppm of the package's 27-px mask); PRISM all rows
  beyond 3 vs the package's 4-15: tight aperture (2.5) -9.3 +- 1.6 / -8.0 +- 1.7 ppm (1.5%),
  wide aperture (the mode's 10) -58 +- 5 / -62 +- 6 ppm (7.5%), but there the 1/f rows lie
  inside the aperture (as the package's 4-15 do), which also pulls each integration toward
  s_i T (self-subtraction): the wide gain is not all noise reduction. Open: PRISM default 1/f
  rows (4-15) overlap its default aperture (10).
- [x] Bug fixed (f6e22bd): soss.correct_ramps_1f overwrote its template argument with the
  median rate, so the default SOSS fit_ramps raised since f57f497; regression test added.
- [x] One function for the group 1/f correction (871890e): jwst.group_difference_1f_from_traces
  (traces, median rate, background -> distances, 1/f pixels, aperture, white-light factors,
  core correction) and jwst.group_1f_settings (the per-mode table). NIRSpec and SOSS only
  find the traces. Outputs identical on synthetic PRISM, G395H and SOSS scenes, both
  templates and domains (scratch onef_equivalence.py; SOSS up to apertures at exactly
  half-integer distances; scipy's median_filter is not reproducible for windows longer than
  twice the data, so the check uses a numpy filter). The mode differences, now explicit:
  NIRSpec 1/f rows from floor(trace) - 0.5 (0.5-1.5 rows below the trace), traces extended
  to all columns, factors from a fixed rectangle (G395H the full 32 rows) of the first-pass
  rates, NaNs as zero, normalised by their median, filtered over 5 (G395H) or 200 (PRISM)
  integrations; SOSS 1/f pixels > 27.5 px from the nearest order (center), 4 edge columns
  out, factors from the order-1 aperture relative to the median rate, unfiltered.
  Template question: the median template is normalised (median_i of x_i/s_i), but the whole
  pixel is divided by s_i, also what does not transit (bias residual, dark, sky): with
  x = A + s_i S + f, the background residual is A (1 - s_i) + f, so A (1 - s_i) is taken as
  1/f and removed from the aperture; noise in s_i leaks the same way. That is the measured
  coupling (G395H 0.040, PRISM 0.027/0.007, SOSS 0.015 = (A + S) of the 1/f pixels x
  aperture rows / aperture flux), and the A part of it biases the depth by that fraction.
  The 'differences' domain template (core.detector.difference_1f) is not normalised at all
  (median of the raw differences, then times s_i). G395H gained most from iterating only
  because its factors were the noisiest (2678 ppm x 0.040 = 107 ppm). Proposed next:
  per-pixel two-term templates (x = a + s_i b, robust), one factor definition (aperture
  white light, window in time) and 1/f pixels from the trace center, for all modes; depth
  bias on a full transit.
- [x] Restart after the 2026-10-26 machine freeze (full-segment Stage 1 of G395H NRS2 seg1
  reached a ~10 GB footprint with a PRISM download alongside; swap filled the disk). jwst
  steps called alone deep-copy their input; patching that (make_copy=False) was not enough.
  New extraction (scratch onef_extract3.py, chain.sh): each segment in 3 chunks of
  integrations through the package flow up to the group 1/f call, 2-3 GB peak; the saved
  inputs reproduce the package's chunk levels to 0.003 DN rms (levels 14.7 DN rms; float16).
  Memory watchdog (job footprint incl. compressed, swap, disk). Deleted the G395H NRS1 uncal
  (re-downloadable); auto mode refused deleting the nrs2-nonlinearity intermediates (12 GB).
- [x] Homogeneous group 1/f method, G395H NRS2 (PID 1118 obs 5, segs 1-3, transit in 2-3,
  ~6600 ppm; scratch onef_homog.py, onef_homog_summary.py; results
  validation/runs/onef-full/g395h/homog_seg00?.npz). Per segment, as the package. Variants
  template/pixels/factors: med = median_i(x_i/s_i) (package), two = robust per-pixel fit x =
  a + s_i b (3 x 5-sigma clipping); pkg pixels (floor - 0.5, 3-8 rows, 9.4 px/col) vs new
  (trace centre, >= 3 rows, all; 26 px/col); pkg factors (32-row rectangle, window 5;
  2400-2800 ppm scatter) vs new (aperture of the first-pass rates / median rate, window 300 s
  = 17 ints; 170-190 ppm). White light = aperture-2 sums of per-pixel OLS slopes (no
  linearity). rms about a 51 median, all segments: raw 819, med/pkg/pkg 198.1, two/pkg/pkg
  259.9, med/new/pkg 176.5, med/new/new 175.3, two/new/new 176.0 (-22 +- 3 vs package). Depth
  change vs raw (regression of corrected - raw on s - 1, block bootstrap): all within +-1%
  (+-0.8); paired vs two/new/new: med/pkg/pkg -0.16 +- 0.18 %, med/new/pkg -0.67 +- 0.21,
  med/new/new -0.76 +- 0.23 (median template shallower), two/pkg/pkg +2.61 +- 0.44 (two-term
  with noisy factors: attenuated b, worse rms). So the two-term template needs clean factors;
  with them it matches the median template's rms and differs in depth by ~0.7%.
- [x] Homogeneous group 1/f method, PRISM (PID 3969 obs 1, Kepler-12, segs 1-5, all 5
  downloaded; transit ~1.7%; results validation/runs/onef-full/prism/homog_seg00?.npz; inputs
  deleted after each segment). New pixels: trace centre, >= 11 rows (one beyond aperture 10;
  10 px/col vs the package's 21 in rows 4-15); new factors: aperture-10 rates, window 300 s =
  265 ints (unfiltered 7600 ppm scatter, filtered 110; package 900-1000). rms all segments,
  aperture 10 / 2.5: med/pkg/pkg 753.6 / 571.9, two/pkg/pkg 993.8 / 625.1, med/new/pkg 883.6
  / 579.0, med/new/new 883.6 / 579.0, two/new/new 883.3 / 578.9. Factors and template: no
  rms effect; pixels outside the aperture: +130 ppm (wide; half the pixels, and the
  package's in-aperture rows may self-subtract) / +7 (tight). Depth, paired vs two/new/new:
  med/pkg/pkg -0.36 +- 0.09 % / -0.09 +- 0.02, med/new/new -0.24 +- 0.02 / -0.06 +- 0.00
  (median template shallower, as G395H's -0.76). Open: PRISM pixels (inside the aperture or
  not); the depth shifts are small (< 1%) in both NIRSpec modes.
