# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

# [0.3.13] - 2022-10-28
### Added
- Resolution binning can now consider data errorbars to compute binned errorbars.

# [0.3.12] - 2022-08-24
### Added
- Function to compute FWHM.

# [0.3.11] - 2022-08-02
### Added
- Functions in `utils.py` to predict transits, perform calculations of atmospheric signals and scale-heights.
- Function in `utils.py` to bin in resolution (`bin_at_resolution`).
- Function in `utils.py` to convert wavelengths from vacuum to air and viceversa.
- Added `transitfitting.py` which allows to perform transit fitting in parallel.

### Changed
- `getSimpleSpectrum` has `correct_bkg=False` by default now.

# [0.3.10] - 2022-06-07
### Added
- Now getSimpleSpectrum can use background on both sides or on only one side of the spectrum.

# [0.3.9] - 2022-06-07
### Fixed
- Bug commit that prevented script from compiling.

# [0.3.8] - 2022-06-07
### Added
- New function in `jwst.py`, `get_cds` to get Correlated Double Sampling frames from a dataset or set of segmented datasets.
- The `jwst.stage` can now receive `rateints` products as inputs too.

# [0.3.7] - 2022-06-06
### Added
- Now allow background fits with LOOM.

# [0.3.6] - 2022-05-04
### Fixed
- Fixed bug that didn't allow to import `transitspectroscopy.jwst`.

# [0.3.5] - 2022-05-04
### Added
- New function in `utils.py` to fit splines.
- `quicklook` flag added for reduction.

# [0.3.4] - 2022-04-01
### Added
- New set of algorithms for creating uniluminated masks.

# [0.3.3] - 2022-03-31
### Fixed
- JWST ramp-fitting outputs can't be oppened with datamodels.RampModel. Changed that.

# [0.3.2] - 2022-03-30
### Fixed
- JWST stage 1 was forced in `transitspectroscopy` to use an input gain reference file. Now use the default one if none was added.

# [0.3.1] - 2022-03-18
### Fixed
- ROEBA model corrected thanks to inputs from Everett Schlawin.

# [0.3] - 2022-03-18
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
