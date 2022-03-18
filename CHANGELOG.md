# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

# [0.3.1] - 2021-03-18
### Fixed
- ROEBA model corrected thanks to inputs from Everett Schlawin.

# [0.3] - 2021-03-18
### Added
- Option to trace spectra using CCF instead; implemented gaussian and double gaussian (useful for SOSS).
- Various CCF utilities.
- Set by default no gaussian filtering on the tracing; variable median filter window (and optional outlier detection with it).
- Create c-functions that perform CCF against gaussian, double gaussian and arbitrary functions (x40, speed increase)
- Made c-functions the default for CCF while tracing.
- Added the `tests` folder; these could become unit tests.
### Changed
- Bumped version number.

## [0.2] - 2021-02-08
### Added
- The `jwst` class which can reduce JWST data using the JWST pipeline.
- Added `stage1` master function to `jwst` to reduce SOSS data.
- Added `LOOM` model to `stage1` --- this replaces the refpix step in this script.
- Now `jwst` checks for already existing outputs so steps don't run again if products already produced.
### Changed
- Name of simple extraction in `transitspectroscopy.spectroscopy` from `getSimpleExtraction` to `getSimpleSpectrum`.

## [0.1] - 2021-01-28
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
