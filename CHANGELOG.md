# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.5.0] - Unreleased
Detailed development notes for the work below are in `docs/dev-notes/`.

### Added
- Pure-Python Marsh optimal extraction (`backend='python'`), used automatically when the
  `Marsh` C extension is not installed. Matches the archived C outputs within tested tolerances.
- NumPy implementation of the CCF routines; the `CCF` C extension is optional.
- Experimental Gaussian-process spatial profiles for optimal extraction
  (`profile_method='gp'`, `gp_options=...`). Always opt-in.
- `fit_shared_profile` / `SharedProfile`: train one intrinsic spatial profile on selected
  integrations, save/load it, and evaluate it through every integration's trace.
- NIRISS/SOSS three-order reduction (`ts.soss.reduce_soss`, or `ts.jwst.stage2` on
  `NIS_SOSS` data): PASTASOSS traces, official background subtraction, box extraction, and
  optional joint (deblended) polynomial/GP optimal extraction of overlapping orders.
- MIRI imaging time-series photometry (`ts.miri.reduce_miri`): official Detector1/Image2
  calibration, aperture photometry and a shared 2-D GP PSF.
- Alternative correlated adjacent-read ramp fitter (`ramp_fit_method='sparta'`, see
  `docs/guides/ramp-fitting.md`), selectable in `ts.jwst.load`, `stage1` and MIRI calibration.
- `ramps_err_available` in `ts.jwst.load`, flagging which integrations carry ramp errors.
- `stage2` options `extraction_backend`, `profile_method`, `gp_options`,
  `extraction_options`, `shared_profile`, `shared_profile_options`, `soss_options`,
  `miri_options`.
- `transitspectroscopy.core`: the instrument-independent algorithms (tracing, profiles,
  extraction, shared profiles, overlapping traces, photometry, ramp fitting, detector
  corrections, binning, products, provenance), with no JWST-pipeline imports. Existing modules
  keep their names and signatures as thin layers over it.
- `transitspectroscopy.instruments`: a registry of observing modes (`Mode`, `register`,
  `identify`, `list_modes`) holding every mode-specific default; plug-in modes need no edits
  to `jwst.py` (see `docs/guides/adding-a-mode.md`).
- `ExtractionConfig` (`transitspectroscopy.config`): all Stage 2 settings in one validated
  object, `ts.jwst.stage2(..., config=...)`; Stage 2 dispatches on the observing mode.
- Standard products (`core.products`): `SpectroscopicTimeSeries`, `PhotometricTimeSeries` and
  `LightCurveSet`, with FITS/ECSV I/O and JSON provenance manifests (versions, git commit,
  CRDS context, input hashes, resolved configuration). Every Stage 2 path returns them in
  `result['products']` and writes them.
- `ts.run(config)` / `transitspectroscopy run config.toml`: configuration-driven reductions
  with per-step caching (`--force`, `--from-step`); `transitspectroscopy modes`.
- `ts.jwst.download(..., exposure=, segments=, detectors=)` selects files and returns their paths.
- Validation configurations in `validation/configs/*.toml`, run by `validation/validate.py`.
- A pytest suite (`tests/`), including a snapshot of the 0.4 public API
  (`tests/legacy_api.json`), and real-data validation scripts/reports in `validation/`.
- Conda `environment.yml` for the full pipeline and dependency extras
  (`fitting`, `parallel`, `download`, `test`).

### Changed
- Group-level 1/f correction with median group-difference templates
  (`group_difference_1f_correction`, `load.group_difference_1f`) is the default for NIRSpec
  and NIRISS/SOSS (`detector_calibration(group_difference_1f=True)`, `[calibration]
  group_difference_1f = true`): `fit_ramps` fits the ramps, corrects every group using the
  traces and white-light factors of those rates, and fits them again. For SOSS it replaces
  `clean_flicker_noise`. The corrected rates are marked (HISTORY), and Stage 2 then skips its
  rate-level 1/f correction (`scale_1f`). Numerical outputs of reductions from `uncal` files
  change. With `parameters={'group_difference_1f': {'domain': 'differences'}}` the 1/f levels
  are measured on the group differences (CDS frames), free of offsets that change between
  integrations, and removed from the groups as their cumulative sums
  (`core.detector.difference_1f`); the default is `'groups'`. `detector_calibration` takes
  the flags `group_difference_1f_domain` and `ramp_fit_method`/`ramp_fit_options` (replacing
  the ramp fit selected in `load`); `[calibration] group_difference_1f_domain` in `ts.run`.
- The group-difference 1/f levels, measured on the ramps corrected for linearity, are removed
  from the raw counts, i.e. before the linearity correction (`before_linearity=True`, the
  default of `group_difference_1f_correction`; `core.detector.offsets_before_linearity`,
  `jwst_models.linearity_coefficients`). Removed after it, the per-integration pedestal and
  1/f offsets went through the curvature of the correction, larger for brighter pixels: on
  NIRSpec/G395H NRS1 (HAT-P-14 b, segment 1) the white-light 100-point RMS goes from 165.2,
  185.0 and 174.6 ppm to 154.7, 177.6 and 165.9 ppm (simple, polynomial and GP extraction),
  with the spectroscopic light curves unchanged. Numerical outputs of reductions from `uncal`
  files change; `parameters={'group_difference_1f': {'before_linearity': False}}` restores
  the previous correction.
- After the TSO jump step, the correlated (`'sparta'`) ramp fit rejects no differences unless
  `sigma` is given in `ramp_fit_options` (the TSO jump flags define the excluded differences;
  its own clipping flagged ~4e-4 of the differences of faint NIRSpec pixels as `JUMP_DET`).
- The correlated (`'sparta'`) ramp fit refits a ramp only while it rejects a difference or its
  estimate changes (by more than 1e-3 of the noise of one difference), so a ramp's rate no
  longer depends on the other ramps of its pixel chunk; rates and variances are float32.
- NIRISS/SOSS Stage 2 removes rate-level 1/f with the median rate scaled by the white light
  (`onef_method='template'`, orders 1-3 masked) by default.
- Compatibility with JWST pipeline 3.0 (`stdatamodels` datamodels, ramp models without `ERR`).
- Optional dependencies (juliet, Ray, astroquery, pandas) are imported only when used, so
  `import transitspectroscopy` no longer fails when one of them is missing.
- `ts.jwst.load` allocates only the integrations actually loaded when a subset of segments
  is given (previously used the full-exposure `NINTS`).
- The Python extraction checks the lower and upper aperture edges independently for
  masked (`-9999`) pixels (the C reference keeps its historical behavior).
- Reading saved products made by different code warns instead of refusing (`strict=True`
  restores the error), e.g. `soss.load_products`.
- `soss.reduce_soss` and `miri.reduce_miri` are deprecated in favour of `ts.jwst.stage2` (or
  `soss.extract_soss` / `miri.extract_photometry`); the 0.5-development `stage2` keywords
  `extraction_backend`, `profile_method`, `gp_options`, `extraction_options`,
  `shared_profile(_options)`, `soss_options` and `miri_options` are deprecated in favour of
  `config=ExtractionConfig(...)`.
- `ts.jwst.load` orders files by exposure, then segment.
- The package requires Python >= 3.13 and installs `jwst >= 3.0.0` (`pyproject.toml`).
- `ramp_fitting.fit_ramps` now defaults to `residual_correction=False`. The empirical
  residual template is trained on all integrations, including in-transit ones, so it must be
  enabled explicitly with suitable training integrations.

### Fixed
- Stage 2 `mask_dq` no longer masks pixels flagged only `JUMP_DET` (their rates are fitted
  around the jump). When the TSO jump step flagged a column's whole trace, the GP optimal
  extraction of that column was NaN and the `nansum` white light dipped (600-2500 ppm on
  NRS1); non-finite spectra are now also replaced in the `corrected` spectra.
- Parallel `fit_lightcurves` (Ray) returned `None` for every fit.
- `download(..., data_product='ramps')` could not select ramp products.
- `trace_spectrum(xend=None)`, `get_pm_ccf`, `load.interpolate_nans`, `fit_spline` without
  knots, and `get_uniluminated_mask` without DQ raised name/undefined-variable errors.
- `stage2(zero_nans=False)` used an undefined median-error image.
- MIRI mode detection compared strings by identity; a missing filter crashed `load`.
- Running a `load` step without a prior calibration step failed on suffix initialization.
- `spill_filter` divided by zero on empty boxes.

## [0.4.1] - 2026-09-13
### Added
- Import of `Marsh` and `CCF` is now optional.
- Removed `Marsh` setup. Plan is to convert it to Python in the summer.

### Fixed
- Error because jwst build 11.3 removes the .err array in the 4D ramp products (see https://jwst-docs.stsci.edu/jwst-science-calibration-pipeline/jwst-operations-pipeline-build-information/jwst-operations-pipeline-build-11-3-release-notes?utm_source=chatgpt.com#gsc.tab=0). Now if products were created with build 11.3 and above, 4D ramps have error arrays of zeros. If below, save the error arrays.

## [0.4.0] - 2023-02-13 
### Added
- Initial a-la-`juliet` retrieval functionality.
- 'convolve' methodology for CCF's added by @tbainesUA in #2.
- Function to get background and 1/f-corrected CDS to `ts.jwst`.
- Re-vamping `stage1` into `ts.jwst`.
- Re-vamping `stage2` into `ts.jwst`.
- Requirement of `tqdm` to `setup.py`.
- Support for NIRSpec/G395M.
- Support for JWST data download via `ts.jwst.download`
- Implemented `tsojump` step.

## [0.3.13] - 2022-10-28
### Added
- Resolution binning can now consider data errorbars to compute binned errorbars.
- "Poor man's CCF" calculation in `ts.spectroscopy.get_pm_ccf`.

## [0.3.12] - 2022-08-24
### Added
- Function to compute FWHM: `ts.spectroscopy.get_fwhm(x,y)` and `ts.spectroscopy.trace_fwhm(tso, x, y)`.

## [0.3.11] - 2022-08-02
### Added
- Functions in `utils.py` to predict transits, perform calculations of atmospheric signals and scale-heights.
- Function in `utils.py` to bin in resolution (`bin_at_resolution`).
- Function in `utils.py` to convert wavelengths from vacuum to air and viceversa.
- Added `transitfitting.py` which allows to perform transit fitting in parallel.

### Changed
- `getSimpleSpectrum` has `correct_bkg=False` by default now.

## [0.3.10] - 2022-06-07
### Added
- Now getSimpleSpectrum can use background on both sides or on only one side of the spectrum.

## [0.3.9] - 2022-06-07
### Fixed
- Bug commit that prevented script from compiling.

## [0.3.8] - 2022-06-07
### Added
- New function in `jwst.py`, `get_cds` to get Correlated Double Sampling frames from a dataset or set of segmented datasets.
- The `jwst.stage` can now receive `rateints` products as inputs too.

## [0.3.7] - 2022-06-06
### Added
- Now allow background fits with LOOM.

## [0.3.6] - 2022-05-04
### Fixed
- Fixed bug that didn't allow to import `transitspectroscopy.jwst`.

## [0.3.5] - 2022-05-04
### Added
- New function in `utils.py` to fit splines.
- `quicklook` flag added for reduction.

## [0.3.4] - 2022-04-01
### Added
- New set of algorithms for creating uniluminated masks.

## [0.3.3] - 2022-03-31
### Fixed
- JWST ramp-fitting outputs can't be oppened with datamodels.RampModel. Changed that.

## [0.3.2] - 2022-03-30
### Fixed
- JWST stage 1 was forced in `transitspectroscopy` to use an input gain reference file. Now use the default one if none was added.

## [0.3.1] - 2022-03-18
### Fixed
- ROEBA model corrected thanks to inputs from Everett Schlawin.

## [0.3] - 2022-03-18
### Added
- Option to trace spectra using CCF instead; implemented gaussian and double gaussian (useful for SOSS).
- Various CCF utilities.
- Set by default no gaussian filtering on the tracing; variable median filter window (and optional outlier detection with it).
- Create c-functions that perform CCF against gaussian, double gaussian and arbitrary functions (x40, speed increase)
- Made c-functions the default for CCF while tracing.
- Added the `tests` folder; these could become unit tests.
### Changed
- Bumped version number.

## [0.2] - 2022-02-08
### Added
- The `jwst` class which can reduce JWST data using the JWST pipeline.
- Added `stage1` master function to `jwst` to reduce SOSS data.
- Added `LOOM` model to `stage1` --- this replaces the refpix step in this script.
- Now `jwst` checks for already existing outputs so steps don't run again if products already produced.
### Changed
- Name of simple extraction in `transitspectroscopy.spectroscopy` from `getSimpleExtraction` to `getSimpleSpectrum`.

## [0.1] - 2022-01-28
### Added
- A `tests.py` script showcasing the usability of the code for spectral extraction.
### Fixed
- Input problems for spectral extraction. Coda has been verified (but not validated extensively)

## [0.0] - 2022-01-28
### Added
- First (working but not tested) version of the spectroscopic algorithms.
- Installation files, `README`, this `CHANGELOG`, `LICENCE`.
### Upgraded
- `setup.cfg` file has metadata info with long description.
