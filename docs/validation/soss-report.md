# HAT-P-14 b NIRISS/SOSS validation — 2026-09-24

**This is the historical box-only run.** The subsequent
[optimal-extraction validation](../validation/soss-optimal-report.md) adds both
polynomial and GP spectra/light curves and compares them on identical columns.
The default SOSS validation now uses that expanded workflow.

The implemented reduction successfully produces spectra, light curves and
transmission diagnostics for **all three orders**, using all **572 integrations**
of PID 1541 observation 1. Orders 1 and 2 recover white-light depths broadly
consistent with published values. This verifies the reduction and its products;
it does **not** certify an unbiased transmission spectrum across the entire
SOSS band. Masked/saturated apertures substantially reduce the conservative
comparison band, limb-darkening parameters reach bounds, and several faint
order 3 bins are poorly constrained.

## Executed run

The final manifest (`validation/runs/soss_hatp14_20260924_complete/reduction.json`) and products
are in `validation/runs/soss_hatp14_20260924_complete/`. This final plotting/fitting
pass reuses the independently checked extraction in
`soss_hatp14_20260924_verified` (`validation/runs/soss_hatp14_20260924_verified/reduction.json`).
The latter completed the full extraction and diagnostics in **63.6 seconds**,
excluding downloads. No Ray workers were used; BLAS/OpenMP threads were limited
to one.

The four public files are
`jw01541001001_04101_00001-seg001..004_nis_rateints.fits`, containing
144/144/144/140 integrations with shape 256 × 2048. Their total size is about
6 GB. Following the notebook, the run starts from **archive rateints**: upstream
Stage 1 provenance is **JWST 2.0.1, CRDS `jwst_1535.pmap`**. The new post-ramp
processing uses **JWST 3.0.0**. Full raw-data recalibration of this visit was not
executed. The validation driver also supports raw `uncal` inputs and the existing
local detector-calibration/ramp-fitting methods.

Reproduce the extraction using the local downloaded files and pinned references:

```sh
python validation/validate.py --mode soss \
  --input-glob 'validation/data/soss/*_rateints.fits' \
  --pastasoss-reference validation/references/soss/jwst_niriss_pastasoss_0003.asdf \
  --background-reference validation/references/soss/jwst_niriss_bkg_0033.fits \
  --output validation/runs/new_soss_validation
```

See [the guide](../validation/soss.md) for environment/CRDS setup, downloads and reuse.
The reference filenames, mapping versions and hashes are documented
[here](https://github.com/nespinoza/transitspectroscopy/blob/master/validation/references/soss/README.md). Actual PWCPOS is **245.898438 degrees**. The
PASTASOSS file is evaluated separately for orders 1/2/3; traces are refined with
bounded double-Gaussian correlation and an offset spline. Median fitted-minus-
reference offsets are approximately −0.006, −0.109 and −0.852 pixels, respectively.

## Processing and saved products

The order is ramp-fitted rates → official Stage 2 SOSS background fitting →
reference-based/refined tracing → background and residual column-offset
subtraction → three fractional-box extractions. The background is fitted to a
32-integration sampled median and applied as the same static image to every
integration. The aperture radius is 15 pixels. Inputs remain unchanged.

The final directory contains 36 products plus the JSON manifest, including:

- Background diagnostics (`validation/runs/soss_hatp14_20260924_complete/01_soss_background.png`),
  three traces (`validation/runs/soss_hatp14_20260924_complete/02_soss_traces.png`), and
  trace motion/profile widths (`validation/runs/soss_hatp14_20260924_complete/03_soss_motion.png`).
- All three extracted spectra (`validation/runs/soss_hatp14_20260924_complete/04_soss_spectra.png`)
  and 100-point-filter light-curve diagnostics (`validation/runs/soss_hatp14_20260924_complete/05_soss_white_light.png`).
- Transit fits (`validation/runs/soss_hatp14_20260924_complete/06_soss_transit_fits.png`) and
  transmission spectra (`validation/runs/soss_hatp14_20260924_complete/07_soss_transmission.png`).
- Per-order FITS/NPZ spectra and errors, micron wavelengths, BJD_TDB times,
  explicit DQ/white-light masks, trace tables and CSV light curves/residuals.
  The spectral FITS format is a custom layout, not an STScI `x1dints` model.
  Median rate, fitted background and background-subtracted median are also
  saved as JWST `ImageModel` FITS files.

There is no temporal flux replacement or order decontamination. Incomplete
apertures remain missing. Informational DQ flags such as successfully fitted
jumps remain usable; `DO_NOT_USE`, saturation and invalid science/errors are
excluded. White-light/transmission comparisons use fixed columns valid in
every integration and clear of adjacent orders. This retains only **81, 100
and 120 columns**, respectively. Their wavelength extents are approximately
0.850–2.116, 0.594–1.076 and 0.648–0.952 µm; coverage within these ranges is
fragmented, not continuous. Per-integration spectra retain the larger available
PASTASOSS range and flag unusable or overlapping apertures.

## Transit and transmission results

White-light fits use 492 points after discarding the first 80 integrations.
Radius ratio, quadratic limb darkening, a quadratic baseline and mid-time are
free. Orbital geometry is fixed, with the eccentricity convention documented
in the preset. Errors below are **local covariance estimates**, inflated by
residual variance and time-binning diagnostics. They exclude uncertainty in
orbital geometry, wavelength calibration, background and extraction profiles.
All three white-light fits put `q2` near its lower bound, so these errors should
not be read as reliable posterior credible intervals.

| Quantity | Order 1 | Order 2 | Order 3 |
|---|---:|---:|---:|
| Fitted `(Rp/Rs)^2`, ppm | 6779 ± 263 | 6598 ± 283 | 7611 ± 2066 |
| White-light fit residual RMS, ppm | 180.6 | 232.8 | 1556.6 |
| Median-filter residual RMS, ppm | 192.0 | 244.7 | 1559.9 |
| Trace displacement standard deviation, pixels | 0.0117 | 0.0128 | 0.0225 |
| Nominal R=30 transmission bins | 7 | 9 | 12 |
| Failed bins | 0 | 0 | 1 |
| Bins with missing error or error ≥ depth | 0 | 1 | 6 |

For context, [Liu et al.](https://arxiv.org/html/2504.08903v1) report white-light
depths of 6750 ± 40 and 6714 ± 68 ppm for orders 1 and 2. The differences are
about +0.11 and −0.40 combined **local** standard errors here, although the
bandpasses and treatment of systematics differ. Their aggregate SOSS
transmission depth, 6522.4 ± 9.8 ppm, is shown as a contextual line in the
transmission figure. It is not a wavelength-by-wavelength reference spectrum
and is not used to force the measured spectrum to a desired result.

The order 3 bin spanning **0.6481–0.6697 µm** has nonpositive baseline flux and
cannot support a normalized transit fit. It is retained with NaN depth/error,
`fit_failed=1`, and its failure reason in the JSON fit table. Other weak order 3
bins can converge while remaining essentially unconstrained; the full-range
and zoomed plots deliberately show both facts. One order 2 bin has a
rank-deficient local covariance, with no finite depth uncertainty reported.

The transmission fits keep each order's fitted mid-time and orbital geometry
fixed, but fit limb darkening independently in every bin. Several parameters
hit bounds (7/8/9 bins in orders 1/2/3). The output therefore supports inspecting
data quality and comparing future reductions, rather than claims about small
atmospheric features. A preferred numerical reference spectrum can be supplied
as CSV for an overlay. No independent numerical reference transmission table
was available in this execution; agreement with a full published spectrum has
not been established.

## Verification and assessment

The repository test suite passes: **103 passed, 13 skipped**, with two existing
shared-profile boundary warnings. New tests cover noisy double-Gaussian trace
recovery, subpixel motion, fractional-aperture variance, missing/edge pixels,
integration-number timestamp selection, Stage 2 dispatch, and enforced
ramp-fitting/background order. A noisy three-order reduction test recovers an
injected 0.7% flux decrement while preserving input arrays, handling permanent
bad pixels and informational jump flags, and round-tripping saved products.
A separate noisy physical-transit test recovers radius-ratio depth and mid-time;
failed-bin tests guard against silently discarding unusable measurements.

An independent audit checked every source/input/product hash, all 572 integration
numbers, per-order FITS/NPZ/CSV consistency, BJD_TDB timestamps, fixed masks,
integrated white-light fluxes and reported residual RMS. The diagnostic plots
were inspected. Existing NIRSpec tests still pass; its algorithms and GP defaults
were not changed.

The appropriate conclusion is **successful three-order reduction with scientific
limitations exposed**, not a blanket science-validation pass. Further work toward
a precision transmission benchmark should address the lost wavelength coverage
without changing the time-dependent bandpass, overlap decontamination, consistent
treatment of partially saturated ramps, and posterior inference with validated
limb-darkening information. Full ramp-level signal injections and a matched
numerical reference transmission spectrum would then provide a stronger test
of chromatic bias. The manifest records `diagnostics_complete_not_certified` to
make this distinction machine-readable.
