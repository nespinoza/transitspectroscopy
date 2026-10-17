# SOSS optimal extraction and validation completion

**Historical isolated-order comparison:** the subsequent
[overlap correction](../validation/soss-overlap-report.md) replaces the long-wavelength
exclusion with joint profile/flux fitting. The default validation now uses that
correction; the numbers below describe the earlier isolated-order run.

The original SOSS validation omitted optimal spectra because its reduction
implemented only box extraction. The reduction now supplies **box, polynomial
optimal and GP optimal products for orders 1, 2 and 3**, and the root spectrum
figure explicitly overlays all three methods. The new default validation preset
is `config.soss-hatp14-optimal.json`; the original box-only preset is unchanged.

## Functional gaps addressed

- Added shared intrinsic profile training and native-pixel optimal extraction
  for each isolated order. One coefficient set per order/method is fitted to
  four independent sampled integrations and then reused throughout the visit.
  Moving-trace extraction evaluates those coefficients at each integration's
  trace. No per-integration profile refit or temporal flux replacement occurs.
- Added portable fitted profile models, observed-profile fractions, independent
  optimal spectra/errors, FITS/NPZ/CSV light curves and method-specific transit
  and transmission products. Missing-pixel flux recovery requires at least
  half of the modeled profile to remain observed. Unusable measurements remain
  NaNs rather than silently filled observations.
- Added matched-wavelength comparisons. Each method's broader usable coverage
  is reported separately from its performance on the intersection of fixed
  valid wavelengths. The latter supplies like-for-like light-curve RMS, transit
  depth and median spectral-difference metrics and numerical series.
- Added visible profile-fit quality and relative spectral-difference plots,
  and expanded reuse verification to fitted profiles and extraction modules.
- Added noisy signal-injection regressions for trace motion, permanently masked
  peaks, pixels masked only during transit, completely missing columns,
  insufficient observed-profile support and sigma rejection. A three-order
  integration test exercises both models, the post-ramp background ordering,
  immutable input arrays and saved-product round trips.

The implementation uses the repository's **positive shared log-intensity
models**, with polynomial or GP variation along dispersion. These are distinct
from the legacy unregularized Marsh coefficient fit. The GP masked-column
fixes and defaults in `_optimal_extraction.py`, and existing NIRSpec behavior,
are unchanged. Unlike constructing an empirical normalized profile, the shared
model works directly with observed native pixels and independently fitted
column fluxes; missing pixels contribute no likelihood.

## Run and figures

The complete visit contains 572 integrations from the four previously downloaded
PID 1541 observation 1 `rateints` segments. Background and trace settings match
the [historical box run](../validation/soss-report.md). The new extraction is still
downstream of the archival Stage 1 calibration; this is not a new raw-ramp
calibration of the visit.

The final products are under
`soss_hatp14_optimal_20260925_reviewed` (`validation/runs/soss_hatp14_optimal_20260925_reviewed/reduction.json`):

- Three extraction methods in the median spectra (`validation/runs/soss_hatp14_optimal_20260925_reviewed/04_soss_spectra.png`).
- Identical-band light-curve comparison (`validation/runs/soss_hatp14_optimal_20260925_reviewed/08_soss_extraction_comparison.png`).
- Profile fit quality (`validation/runs/soss_hatp14_optimal_20260925_reviewed/09_soss_profile_fit_quality.png`).
- Relative spectral differences (`validation/runs/soss_hatp14_optimal_20260925_reviewed/10_soss_spectral_differences.png`).
- Polynomial transmission spectrum (`validation/runs/soss_hatp14_optimal_20260925_reviewed/optimal/polynomial/07_soss_transmission.png`)
  and GP transmission spectrum (`validation/runs/soss_hatp14_optimal_20260925_reviewed/optimal/gp/07_soss_transmission.png`).

Each method's `05_soss_white_light.png` and `06_soss_transit_fits.png` are in
the corresponding product directory. Root products retain box-extraction names
for compatibility; optimal products are under `optimal/polynomial/` and
`optimal/gp/`. All comparison series and spectral differences are saved as CSV.

## Measured results

The full extraction took **159.6 seconds**, excluding downloads, in serial execution
with one BLAS/OpenMP thread. The final saved-product reuse and fitting/plotting pass
took **14.8 seconds** and wrote **159 products** plus the manifest.

The table below uses identical wavelengths for all three methods within each order.
RMS is measured about the fitted transit and baseline, rather than the median filter.
Depth errors are conditional local covariance estimates, with the same limitations
as the original validation fits.

| Order | Comparison columns | Box RMS (ppm) | Polynomial RMS (ppm) | GP RMS (ppm) |
|---|---:|---:|---:|---:|
| 1 | 81 | 180.6 | 180.2 | 180.4 |
| 2 | 100 | 232.8 | 231.8 | 230.9 |
| 3 | 120 | 1556.6 | 1505.5 | 1516.5 |

| Order | Box depth (ppm) | Polynomial depth (ppm) | GP depth (ppm) |
|---|---:|---:|---:|
| 1 | 6779 ± 263 | 6765 ± 253 | 6769 ± 254 |
| 2 | 6598 ± 283 | 6612 ± 290 | 6609 ± 292 |
| 3 | 7611 ± 2066 | 7532 ± 1969 | 7578 ± 2009 |

Allowing each method its own fixed valid band yields:

| Order | Box columns | Optimal columns (both models) | Polynomial RMS (ppm) | GP RMS (ppm) |
|---|---:|---:|---:|---:|
| 1 | 81 | 1310 | 134.6 | 134.7 |
| 2 | 100 | 1074 | 142.6 | 142.0 |
| 3 | 120 | 1084 | 614.7 | 616.5 |

Optimal transmission products have 29/19/12 bins for orders 1/2/3. In both
methods, 29/18 bins in orders 1/2 reach at least one fitting bound. Two order 3
bins have missing uncertainties or uncertainty at least as large as depth;
one GP bin fails entirely and remains explicitly flagged. Near the weak/negative
order 3 signal below about 0.7 µm, relative spectral differences can exceed
20%; the 0.6–0.8% figure describes the median across the comparison bands.

Mean training chi² per valid pixel is approximately 182/93/4.4 for polynomial
and 171/89/4.3 for GP. These large residuals are not hidden by the improved
wider-band light-curve RMS.

## Interpretation and remaining scientific work

Recovering masked columns is the largest practical gain. The optimal spectra
have substantially wider usable coverage, but their wider-band precision is
not an equal-band comparison with the old box curves. On identical wavelengths
the improvement is modest. Differences between methods' fitted depths are
small compared with their conditional errors; these are highly correlated
measurements of the same exposure, not independent estimates.

The optimal median flux scale is about 0.6–0.8% below the box scale on the
comparison bands. This is an extraction/model offset, not a measurement of
transit-depth bias. Training residual chi² per valid pixel is substantially
above unity, especially for orders 1 and 2. The current profile family and/or
supplied pixel variance therefore do not fully describe the data. A lower
light-curve scatter does not prove unbiased absolute or chromatic flux recovery.
The new fit-quality figure and JSON diagnostics expose this discrepancy.

The functional omission of optimal extraction is fixed. The remaining work
toward a precision transmission benchmark is scientific validation, not a
missing optimal-spectrum plot:

1. Diagnose the profile residuals using held-out integrations and spatial
   residual maps; test spatial resolution, wavelength flexibility, trace
   accuracy and variance calibration with ramp-level injections. Hyperparameter
   optimization is available but is not a substitute for a correct noise/model
   specification. This validation deliberately fixes hyperparameters and does
   not tune on recovered depth.
2. Model overlapping orders if the excluded wavelength regions are required.
   These optimal extractions are **not order decontamination**. The pipeline
   continues to exclude overlap/edge regions rather than label their blended
   signal as independent optimal spectra.
3. Reprocess raw ramps and investigate partial saturation, detector response and
   the weak/negative short-wavelength order 3 signal. This run reuses archival
   rateints; no full raw-ramp recovery claim is made.
4. Perform posterior transit inference with validated limb-darkening and
   correlated-noise treatment, and compare against a numerical published
   transmission table with matched bins. The current local covariance errors
   exclude profile/background/trace uncertainty and can be unreliable at fit
   bounds. The literature aggregate depth remains context, not a complete
   transmission-spectrum benchmark.

The manifest therefore retains `diagnostics_complete_not_certified` rather than
declaring a blanket science-validation pass. The [usage guide](../validation/soss.md)
describes the API, controls, products and reproducible commands.

## Verification

The repository suite passes **109 tests**, with **13 skipped** and the two
existing shared-profile hyperparameter-boundary warnings. The added changing-
mask/moving-trace injection tests recover a 0.7% dimming within three propagated
standard errors for both methods, retain the coefficient fingerprint throughout
extraction, and recover the mean aperture flux to better than 0.1% in that
specified simulation. Those tests are necessary regressions, not substitutes
for real ramp-level injections or a proof of absence of chromatic bias.

An independent audit verified all input/code/product hashes, all nine FITS/NPZ/CSV
product sets, timestamps, fixed masks, profile normalization and matched-band RMS.
The original box spectra are numerically unchanged (including missing values).
The comparison, spectral-difference and profile-quality plots were inspected.
