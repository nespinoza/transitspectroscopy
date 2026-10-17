# GP white-light precision experiment — 2026-09-23

**Recommendation: retain the existing GP settings.** The best development-grid
setting does not improve held-out white-light precision. Its small development
gain is not evidence for a repeatable improvement. The full-segment corrected
RMS decreases from **265.623 to 264.298 ppm** (0.50%), but this descriptive gain
does not survive the held-out comparison.

## Data and reproducibility

This experiment starts from the completed
`20260923T042304707564Z` validation (`validation/runs/20260923T042304707564Z/reduction.json`),
with 391 integrations of the first NRS1 segment. It reuses the saved extraction
input, variance/DQ, traces, wavelengths and GP reference profile. The detector
calibration, background/1/f processing, two-pixel aperture radius, spatial
spacing, rejection thresholds and complete-column GP mask policy are fixed.

Every score uses the same 1,537 wavelength columns, spanning approximately
2.674–3.717 µm. Missing measurements cannot be dropped to improve a candidate's
score. White-light flux is the sum over this band, normalized by its time median.
Precision is the RMS about zero of residuals from the existing exactly 100-point
median filter with reflected padding. This is a precision diagnostic, not a
transit fit or a measurement of the photon-noise limit.

Re-extracting with the saved reference profile reproduces the source normalized
light curve to a maximum difference of **0.001223 ppm**. That is negligible
relative to the differences being tested, and checks the float32 FITS input and
mask reconstruction used for reuse.

Run with the pipeline environment:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python validation/gp_sweep.py \
  --run validation/runs/20260923T042304707564Z \
  --output validation/runs/20260923_gp_sweep
```

The output directory must be new. To continue an interrupted run, repeat the
command with `--resume`; input, extraction-code and sweep-setting checks prevent
mixing incompatible results. The recorded run additionally used a temporary
`MPLCONFIGDIR` for the sandbox's Matplotlib cache.

## Search and selection

The baseline is Matérn-3/2, length scale 100 dispersion pixels, amplitude 1 and
24 inducing points. The grid contains both Matérn-3/2 and Matérn-5/2, length
scales 30/100/300/1000 pixels and amplitudes 0.03/0.1/1/3: **32 settings**.
Two more settings test 12 and 48 inducing points at the best grid parameters.
Amplitude is the covariance amplitude parameter, with covariance proportional
to its square; it is not the spatial PSF width. Inducing-point count controls
the approximation's resolution. These are fixed-parameter extraction trials,
not marginal-likelihood hyperparameter fits.

The first 196 integrations are development data. For screening, profiles fitted
to 32 evenly spaced development integrations are combined using the existing
Stage 2 median-profile procedure and column normalization. The existing spectral
replacement step estimates its template and scatter from development data only.
Selection minimizes corrected development residual RMS, subject to:

- Uncorrected residual RMS no more than 2% above the corresponding baseline.
- At most 2% change at the 95th percentile of absolute fractional differences in
  median uncorrected spectra, on bright columns with complete apertures in at
  least 90% of development integrations.
- Finite flux and positive finite errors everywhere in the fixed comparison band.

The selected setting is **Matérn-5/2, length scale 1000, amplitude 0.1,
24 inducing points**. Its corrected development RMS is **267.751885 ppm**,
compared with **268.888077 ppm** for the baseline. Uncorrected RMS is
267.866908 versus 273.298224 ppm. The spectral guard is 0.2046%, below the 2%
threshold. The 12- and 48-inducing-point variants yield corrected RMS values of
283.250560 and 268.041810 ppm, respectively. The 12-point variant also fails the
uncorrected-RMS guard. A Matérn-5/2 model with length scale 30 and amplitude 1
is close, at 267.805174 ppm: the grid does not identify a sharply determined
unique optimum.

After locking these settings, both baseline and selected profiles are refitted
using all 196 development integrations, and tested on the remaining 195. The
spectral-replacement template remains fitted only to development data. Median
filtering is performed separately within each half. The paired moving-block
bootstrap uses 2,000 draws, 20-point blocks and seed 20260923; a positive RMS
difference means improvement. The holdout is not used to select another setting.

Finally the selected setting is fitted to all 391 integrations, with the
replacement template also fitted to all integrations, for the same full-segment
procedure used by the original validation. That final comparison is descriptive,
not another independent test.

## Held-out result

All 34 candidate fits succeeded; 33 passed the development guards. With the
selected parameters locked and both profiles trained on all development frames:

| Holdout metric, 195 integrations | Baseline | Selected |
|---|---:|---:|
| Corrected residual RMS, ppm | 262.375000 | 262.667187 |
| Uncorrected residual RMS, ppm | 264.119879 | 266.660648 |
| Corrected first-difference noise, ppm | 215.326169 | 219.751020 |
| Corrected 5-point binned residual RMS, ppm | 155.496373 | 154.156558 |
| Corrected 10-point binned residual RMS, ppm | 100.849351 | 98.733371 |
| Corrected 20-point binned residual RMS, ppm | 69.793394 | 71.062809 |
| Spectrally replaced samples in comparison band, % | 0.057722 | 0.052383 |

The corrected improvement, defined as baseline minus selected RMS, is
**−0.292187 ppm**, with an approximate paired-block 95% interval of
**[−6.274409, +2.234688] ppm**. The uncorrected difference is **−2.540770 ppm**,
with interval **[−7.369928, +1.234267] ppm**. Neither establishes an improvement;
neither establishes a statistically clear worsening from these intervals alone.
The mixed binned-residual results and worse first-difference noise also do not
support a consistent precision gain. These intervals describe scatter
differences, not transit-depth uncertainties.

The selected model replaces slightly fewer samples, so its development gain
was not obtained by simply increasing the number of replacements. However,
correction affects the selected model's holdout RMS more strongly than the
baseline. Reporting both variants is essential; corrected RMS alone would hide
the greater difference in the uncorrected extraction.

## Full-segment comparison and spectrum checks

| Metric, all 391 integrations | Source GP | Selected GP |
|---|---:|---:|
| Corrected residual RMS, ppm | 265.623056 | 264.298201 |
| Uncorrected residual RMS, ppm | 267.819218 | 266.727267 |
| Corrected first-difference noise, ppm | 221.105259 | 221.391461 |
| Corrected 5-point binned residual RMS, ppm | 148.830572 | 148.576861 |
| Corrected 10-point binned residual RMS, ppm | 113.752411 | 113.324136 |
| Corrected 20-point binned residual RMS, ppm | 91.565986 | 90.425886 |
| Spectrally replaced samples in comparison band, % | 0.056409 | 0.049920 |

The corrected RMS decrease is **1.324854 ppm**, or approximately **0.499%**.
The source values in this table come from the successful replay; the stored
original source RMS is 265.623092 ppm, effectively identical. For context, the
unchanged simple and polynomial extractions in the source run have corrected
RMS values of 264.083061 and 273.044555 ppm on this same band. The selected GP
therefore does not outperform the simple extraction's full-segment RMS either.
This descriptive ranking does not compare flux accuracy or transit-depth bias.

The median spectra agree closely over most wavelengths, but there are important
localized differences. The two largest changes are at detector columns **1245**
(3.17942 µm, −945.51 DN/s or −39.86%) and **1369** (3.26331 µm, −652.75 DN/s
or −18.98%). Both have incomplete apertures in every integration: their profiles
are predicted from other columns and extraction uses the remaining valid pixels.
Both remain in the fixed white-light band. There are also individual complete
columns near 2.974 µm with changes of approximately 2.7%; a 95th-percentile guard
does not bound every column.

These changes do not establish which profile is more accurate, but they show
why minimizing white-light scatter is insufficient for accepting new settings.
They merit targeted masked-pixel flux/injection checks before any adoption. The
existing mask semantics were preserved, and all selected profile columns remain
finite, nonnegative and normalized. The numerical checks and largest localized
differences are saved in
`validation/runs/20260923_gp_sweep/post_run_checks.json`.

## Execution and verification

The completed serial sweep took **840.8 seconds (14.0 minutes)** in the local
`transitspectroscopy` conda environment, with BLAS/OpenMP threads limited to one.
The 34 screening trials account for 541.2 seconds; the remainder includes replay,
confirmation/full refits, extraction, compression and plotting. These timings
apply to this saved 391 × 32 × 1543 extraction cube, not arbitrary detector sizes
or a complete recalibration. There is no requirement for Ray or new dependencies.

The standard test suite completed with **92 passed, 13 skipped**, and two
existing shared-profile boundary warnings. New tests exercise fixed comparison
bands, missing-data preservation, development-only spectral templates, separate
holdout filtering and the direction of the paired bootstrap. An independent
post-run audit checked the selected development minimum, disjoint/exhaustive
split, normalized profiles, unchanged source/sweep code hashes, saved band,
timestamps and agreement between saved spectra, light curves and RMS values.
All three generated comparison figures were visually inspected.

## Products

All numerical results, input hashes, candidate timings, eligibility and the
locked selection are in `validation/runs/20260923_gp_sweep/sweep.json`.
The output directory also contains:

- `validation/runs/20260923_gp_sweep/candidates.csv` and
  `validation/runs/20260923_gp_sweep/sweep.png`: development-grid comparisons.
- `validation/runs/20260923_gp_sweep/white_light_comparison.png`
  and `validation/runs/20260923_gp_sweep/spectrum_comparison.png`:
  full-segment baseline/selected comparisons.
- CSV light curves with times, integrated flux, normalized errors, median filters
  and residuals, for both original and spectrally corrected data.
- Candidate, holdout and selected-full NPZ products containing profiles, spectra,
  errors and replacement masks. Final products retain the common-band mask and
  timestamps; the selected-full product also includes wavelengths/columns.
- `validation/runs/20260923_gp_sweep/selected_config.json`: a separate
  proposed configuration. The user's configuration and library defaults remain
  unchanged.

The `runs/` directory is locally ignored by Git. Keep/archive these products
alongside the source run; this report alone does not contain the extracted data.

## Scope of the conclusion

This is one segment, with a contiguous development/holdout split. Upstream trace,
background and 1/f preprocessing were already estimated using the source run;
the holdout is therefore conditional on that fixed preprocessing, not an entirely
independent observation. Block-bootstrap intervals likewise condition on the
selected settings, fitted profile and detrending, and do not propagate all
calibration or profile uncertainties. Only about ten 20-point blocks fit in the
holdout, so the interval is an approximate diagnostic.

Smaller white-light RMS does not establish unbiased transit depths or improved
precision in every spectral bin. The spectral guard excludes large changes on
mostly complete bright columns; it is not a ground-truth flux test and does not
guarantee accurate interpolation at damaged columns. Confirmation on another
segment/visit and time-dependent signal injection/recovery would be needed
before adopting new general defaults. The selected length scale is also the
upper boundary of this grid: this is the best eligible tested setting, not a
claim of a global optimum.
