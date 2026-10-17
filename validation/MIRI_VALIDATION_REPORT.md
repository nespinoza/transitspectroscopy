# MIRI LTT 1445 A validation — 2026-09-27

Both public F1500W visits of [program 9234](https://www.stsci.edu/jwst/science-execution/program-information?id=9234)
were downloaded and extracted using JWST calibration and local sum/2-D GP
photometry, without Eureka. Observation/exposure selection was verified through
the MAST program catalogue.

## Executed products

Final results are in
[`runs/miri_9234_validation_20260927_reviewed/`](runs/miri_9234_validation_20260927_reviewed/).
The [JSON manifest](runs/miri_9234_validation_20260927_reviewed/reduction.json)
records metrics, settings and provenance. Earlier first-segment and unreviewed
run directories are development diagnostics, not the final flux convention.

All eight archive **Stage 1 rateints** segments received local JWST 3.0.0
Image2 calibration with `jwst_1584.pmap`: WCS, flat field and photom. Raw-ramp
recalibration was not executed on this target. Raw input through official
Detector1 is implemented but this target's full raw-data route remains untested.
Prepared data/provenance are in
[`runs/miri_9234_calibration_20260927/`](runs/miri_9234_calibration_20260927/).
Final extraction reused those calibrated cutouts. Compressed source FITS remain
in `validation/data/miri/`; no deletion mode was used.

| Observation | Exposures / segments | Integrations | Duration (hours) | Sum valid | GP valid |
| --- | --- | ---: | ---: | ---: | ---: |
| 1 | 1 / 2 | 7,264 | 3.950 | 7,264 | 7,264 |
| 2 | 3 / 6 | 22,425 | 12.216 | 22,423 | 22,425 |

Two transient incomplete apertures invalidate sums in observation 2; GP uses
the remaining pixels. Each visit has its own shared PSF. Source flux is not
renormalized per exposure, gaps are retained, and the simple aperture has one
fixed detector position per visit.

- [Observation 1 binned light curves](runs/miri_9234_validation_20260927_reviewed/visit1/04_miri_binned_events.png).
- [Observation 2 binned light curves](runs/miri_9234_validation_20260927_reviewed/visit2/04_miri_binned_events.png).
- [Observation 1 unbinned curves/residuals](runs/miri_9234_validation_20260927_reviewed/visit1/03_miri_lightcurves_residuals.png).
- [Observation 2 unbinned curves/residuals](runs/miri_9234_validation_20260927_reviewed/visit2/03_miri_lightcurves_residuals.png).
- [Example fitted PSF and pixel residuals](runs/miri_9234_validation_20260927_reviewed/visit1/05_miri_psf_residuals.png).

## Precision and scientific scope

| Observation | Sum median-filter residual RMS (ppm) | GP median-filter residual RMS (ppm) | Median formal error, sum / GP (ppm) | Median PSF chi-square per pixel |
| --- | ---: | ---: | --- | ---: |
| 1 | 659.30 | 637.98 | 558.42 / 517.88 | 2.129 |
| 2 | 646.67 | 632.43 | 557.88 / 516.61 | 2.091 |

These are native-integration, 100-point running-median residuals, not residuals
from an astrophysical/systematics fit. The median filter does not modify saved
photometry. Observation 1 visibly contains a transit; observation 2 shows
shallower structure and a settling trend. The full requested visits are covered,
but unprovided event times are not assigned and planet-specific eclipse depths
are not reported. Configuration `events` accepts markers with `mid_bjd_tdb`
and `label`. Ephemerides and a simultaneous event/systematics fit are needed to
interpret depths.

GP and binned errors are conditional pixel/sky uncertainties. PSF residuals and
RMS exceeding formal errors show remaining model/time-correlated noise. Science
status remains `diagnostics_complete_not_certified`. Lower scatter alone is not
evidence of an unbiased eclipse measurement.

Spatial residuals motivated the 21×21 inducing grid. A preliminary 11×11 grid
left diffraction-ring residuals with first-segment median chi-square per pixel
about 63.5; the denser grid gives about 2.1. Length scale (1 pixel) and amplitude
(2) are fixed. Twelve independent sampled integrations train each visit; fluxes
are independent and the GP contains no time coordinate.

Calibration/cutout preparation took **173 seconds**, excluding downloads. Final
PSF fitting, extraction, saving and plotting from cutouts took **21.7 seconds**
with one BLAS/OpenMP thread. PSF training took 4.16 and 5.08 seconds. These are
workstation measurements, not scaling guarantees.

## Injection and regression checks

The suite passed **122 tests**, with **13 skipped** and two existing spectral
shared-profile boundary warnings. New tests cover dense-vs-rank-one sky
covariance propagation, GP derivatives/interpolation/persistence, noisy signal
recovery, permanent/transient masks, physical-unit conversion, integration-time
selection, prepared product round trips and official pipeline dispatch.

The final product audit passed: all eight input/preparation hashes, all 36
output hashes and current extraction source hashes; FITS/NPZ/CSV agreement;
complete visit counts and monotonic BJD_TDB; bin counts and reported RMS;
normalized PSFs; and flux/error reconstruction from saved optimal weights,
including correlated sky uncertainty. The current compressed-input calibration
wrapper was also exercised through official Image2 on a real segment.

The independent noisy Airy-PSF experiment is in
[`runs/miri_injection_20260927_v2/`](runs/miri_injection_20260927_v2/), generated by
[`tests/simulated_miri_photometry.py`](../tests/simulated_miri_photometry.py).
It includes pointing motion, photon/read-background noise, a persistent bad pixel,
a transient missing core and a zero-variance pixel. The generating PSF is not a GP.

| Synthetic visit | Injected depths (ppm) | GP recovered depths (ppm) | Formal errors (ppm) |
| --- | --- | --- | --- |
| Transit + eclipse | 8,000 / 300 | 7,968.2 / 213.6 | 34.8 / 33.2 |
| Two eclipses | 600 / 300 | 605.3 / 211.7 | 34.9 / 33.2 |

All GP differences are below three formal standard errors, but both 300-ppm
recoveries are low by about 2.6 sigma. This does not prove zero bias; larger
ensembles/profile-uncertainty tests remain appropriate before precision eclipse
inference. Plain sums show larger pointing/dead-pixel sensitivity, including a
six-sigma discrepancy for the first event in synthetic visit 2. That behavior is
retained rather than hidden by a model correction.

This test caught an error during development: scaling the GP flux by each
instantaneous masked-aperture fraction reproduced pointing-dependent losses.
Final GP output instead uses a fixed unmasked reference-aperture fraction. The
simple sum retains its observed-aperture meaning, so absolute offsets between
methods are expected. See [the guide](MIRI_README.md) for equations and flags.
