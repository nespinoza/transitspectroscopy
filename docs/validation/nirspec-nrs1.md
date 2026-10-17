# Pipeline validation

For **MIRI imaging/photometry**, use `validate.py --mode miri`; see the
[MIRI guide](../validation/miri.md) and [LTT 1445 A validation report](../validation/miri-report.md).
It compares sums and shared 2-D GP PSF photometry for both PID 9234 visits.

For **NIRISS/SOSS orders 1, 2 and 3**, use `validate.py --mode soss`; see the
[SOSS guide](../validation/soss.md) and [executed HAT-P-14 b report](../validation/soss-report.md).
Its default validation compares box, polynomial optimal and GP optimal extraction;
the [overlap-correction report](../validation/soss-overlap-report.md) describes the joint fit
that restores long-wavelength extraction. The [earlier optimal-extraction report](../validation/soss-optimal-report.md) includes
matched-wavelength precision, depth and profile-quality checks.
The default mode and the configuration described below remain NRS1.

`validate.py` reduces the HAT-P-14 commissioning exposure used by the original
script, with the current checkout and installed JWST pipeline. It produces
six diagnostic figures, JWST FITS models, portable numerical products, and one
`reduction.json` manifest. Its preset is for **NIRSpec NRS1/G395H**, not a
universal calibration recipe for every instrument or target.

## Run

The validation is a run configuration,
[`validation/configs/nrs1-hatp14b.toml`](https://github.com/nespinoza/transitspectroscopy/blob/master/validation/configs/nrs1-hatp14b.toml): the
reduction is done by `ts.run` (exactly what `transitspectroscopy run` does), and
`validate.py` adds the diagnostics. From the repository root, with this checkout installed in
the full environment:

```sh
python -m pip install --no-build-isolation --no-deps .
python validation/validate.py --config validation/configs/nrs1-hatp14b.toml
```

The configuration downloads only the first raw segment of exposure
`jw01118005001_04101_00001` (1.03 GB; 391 integrations × 20 groups × 32 rows × 2048
columns) with `ts.jwst.download`, pins the CRDS context of the reference run
(`jwst_1584.pmap`), and runs three extraction variants (simple, polynomial and GP
optimal). Reduction steps are written to `<output>/reduction/<step>-<hash>/`, diagnostics to
`<output>/` (default `validation/runs/nrs1-hatp14b/`, or `--output DIR`). Rerunning reuses the
calibration of an unchanged configuration; `--force` recomputes everything and
`--from-step extract` only the extractions. To reduce more segments, list them in
`data.segments`; to use other local files, set `data.source = "local"` and `data.glob`.

The custom TSO jump filter uses the integrations of all loaded segments together. The
historical JSON settings are in `validation/configs/legacy/`.

## Fixed settings and comparisons

[`nrs1-hatp14b.toml`](https://github.com/nespinoza/transitspectroscopy/blob/master/validation/configs/nrs1-hatp14b.toml) contains all configurable plotting/extraction values (the `[validation]` table and the extraction variants).
Copy it and use `--config my-validation.json` for another version of the preset.
Every run stores its complete configuration in the JSON report. Limits are
**fixed across runs**, never recalculated from each image's percentiles:

| Diagnostic | Fixed limits | Scale |
|---|---|---|
| Last minus first | −100 to 20,000 DN | asinh, linear width 100 DN |
| Rate image / median trace image | −10 to 1,500 DN/s | asinh, linear width 10 DN/s |
| Group signal | −1,000 to 40,000 DN | linear |
| Group residual | −150 to 150 DN | linear |

The colormap is `magma`; all three panels use the same normalization. These are
explicit starting limits for this target. Inspect a real run before approving
them as a permanent reference; saturation outside a plot's limits does not
change the saved data. Background and illuminated pixel curves also have fixed
limits so that different runs remain visually comparable.

An optional [config.segment001-measured.json](https://github.com/nespinoza/transitspectroscopy/blob/master/validation/configs/legacy/config.segment001-measured.json)
preset preserves the reduction settings and changes only three display ranges:
last-minus-first to −100…32,000 DN, rates to −10…1,900 DN/s, and group signals
to −150…3,000 DN. These ranges were chosen from real segment-1 measurements:
the selected image 99.9th percentiles are about 30,200 DN, while the three
requested pixel signals range from about −65 to 2,450 DN. The original config
is unchanged. Use the same chosen preset for subsequent comparisons rather
than rescaling each run.

Pixels are **zero-based (column, row)**: `(500,5)`, `(1000,20)`, `(1750,15)`.
The selected integrations have loaded-array indices `9`, `n//2`, and `n-10`.
Figure titles give the original exposure integration numbers. For fewer than
20 integrations selections are clamped, may repeat, and are flagged in JSON.

Default extraction uses a 2-pixel aperture, spacing 1 pixel, three polynomial
terms (quadratic), and a 10-sigma rejection threshold. The alternative profile
uses a Matérn-3/2 kernel, length scale 100 dispersion pixels, amplitude 1, and
24 inducing points. Both optimal extractions explicitly use the **Python**
backend. These are the existing Stage 2 polynomial/fixed-hyperparameter GP
paths: they fit integration profiles, form a median detector profile, and reuse
it in extraction. This run does not train the newer shared-intrinsic-profile
model or optimize hyperparameters. Settings are initial comparison choices,
not a claim that either model is optimal for this target.

The GP trains on **complete valid aperture columns**. A rejected training pixel
removes its whole column from training. The worst column is removed first and
the GP is refitted before rejecting another, limiting swamping of neighboring
good wavelengths by a pathological pixel. Valid extraction pixels in that column
remain available. The GP predicts profiles across omitted columns without
renormalizing a partial observed PSF. Nonfinite science, nonpositive/nonfinite
variance and the original Stage 2 DQ mask remain masked. If a wavelength has no
weighted extraction support, its GP flux is NaN and inverse variance is zero;
it is excluded from the common comparison band. Rejection that leaves no
training columns anywhere still fails explicitly. This does not change the
polynomial training/rejection rules or tune GP hyperparameters.

All methods use the same aperture, trace cache, DQ masking, median background
subtraction and scaled 1/f correction. A single median trace is used by default.
Both pre- and post-spectral-outlier-correction spectra are saved. The preset
plots the `corrected` variant to match the original Stage 2 light curve; change
`spectrum_variant` to `original` to inspect results before spectral replacement.
White-light curves use one fixed set of finite wavelength columns valid across
**all three methods and every integration**, avoiding variable bandpasses.

## Products

| File | Contents |
|---|---|
| `01_detector_calibration.png` | Three last-minus-first images and the three pixel ramps below each |
| `02_ramp_fitting.png` | Three rate images, pixel ramps with pipeline slopes, and group residuals |
| `03_trace.png` | Median rate image with the median fitted trace |
| `04_trace_fwhm_timeseries.png` | Trace displacement and FWHM change, with median filters and RMS |
| `05_median_spectra.png` | Median NRS1 spectrum for simple, polynomial optimal, and GP optimal extraction |
| `06_white_light.png` | Three white-light curves, 100-point filters, residuals and RMS in ppm |
| `ts_outputs/*tsojumpstep.fits` | Calibrated ramp datamodel for each segment |
| `ts_outputs/*rateints.fits` | Fitted rate/error/DQ datamodel for each segment |
| `wcs_first_segment_rateints.fits` | First segment with assigned WCS and calibration-reference provenance |
| `extraction_input_rateints.fits` | Combined simple-extraction input cube after background/1/f correction, with errors, original DQ and integration times |
| `ramp_samples.csv`, `.npz` | Selected groups, slopes, fitted signals, residuals, validity/DQ; NPZ also includes images |
| `detector_timeseries.csv`, `ramp_fit_timeseries.csv` | Per-integration image medians and ramp-residual RMS |
| `trace_shape.txt` | Detector column, median fitted row and median FWHM |
| `trace_timeseries.csv`, `trace_fwhm.npz` | Movement/FWHM time series; NPZ also includes every integration's trace and per-column FWHM |
| `spectra_{method}.npz` | Full time × wavelength original/corrected fluxes and errors, timestamps, wavelength/column axes, common-band mask and column RMS |
| `median_spectrum_{method}.csv` | Wavelength-dependent median flux and fractional time-series RMS |
| `white_light_{method}.csv` | Times, integrated flux, normalized flux/error, median filter and residual ppm |
| `simple_minus_{polynomial,gp}.csv` | Signed differences between median spectra and relative differences in ppm |
| `profile_{polynomial,gp}.npz` | Reused median detector profiles |
| `pipeline_outputs/` | Existing Stage 2 trace and extraction caches, scoped to this new run |
| `pipeline.log` | Python/JWST logging messages |
| `reduction.json` | Settings, input hashes, versions, source hashes, references, metrics, status and product paths |

NPZ files contain numerical arrays rather than pickled objects; load with
`np.load(path, allow_pickle=False)`. CSV/text headers state units. Time-series
files retain absolute BJD_TDB and/or detector indices; plotted elapsed time is
relative to the **first loaded** integration, not necessarily exposure start.

## Meaning of the metrics

Trace and FWHM conventions follow [Espinoza et al., Figures 2 and 6 and Appendix
B](https://arxiv.org/pdf/2211.01459): subtract each column's time median and
combine changes across columns by their median. FWHM comes from cubic-spline
half-maximum crossings. Ambiguous crossings are missing, not zero width. The
plot uses 11-point trace and 21-point FWHM median filters.

Ramp residuals use the **JWST output rate**, with a median intercept estimated
from usable groups. The diagnostic excludes DO_NOT_USE, SATURATED and JUMP_DET
groups and DO_NOT_USE pixels. A single intercept cannot represent distinct
segments separated by cosmic-ray jumps: those offsets can remain in residuals.
This metric is not JWST's segmented-fit chi-square or a new rate estimate.

`rms_about_zero` is `sqrt(mean(x**2))` over finite samples;
`rms_about_median` subtracts the finite median first. Null JSON values mean a
measurement was unavailable. Spectral fractional RMS includes astrophysical
variability. White-light residuals subtract an exactly **100-point** median
filter with reflected boundary padding (central offsets −50 through +49).
This is a diagnostic high-pass residual, not a transit fit or an unbiased noise
estimate. Full time series and filters are saved so later comparisons can use
the same time intervals or omit filter-affected boundaries.

Individual-column FWHM excursions can reflect bad/missing detector pixels and
undersampling; they are not automatically evidence for physical PSF changes.
The column-median summaries reduce their influence, and the saved arrays allow
subsequent checks against detector masks.

Differences have sign **simple minus optimal**: the JSON reports the median
over common columns of the difference of the two time-median spectra, in DN/s,
and relative to the simple spectrum in ppm. They are not absolute differences
or differences of separately normalized white-light curves. Extraction errors
are conditional on fitted profiles and omit inter-column/profile/background
covariance, as in the current library.

Relative spectral differences can become very large where the simple spectrum
is near zero, including masked-pixel losses and low-throughput columns. Inspect
the DN/s differences and saved column masks before interpreting those ratios.

The JSON is checkpointed after completed stages; exceptions produce
`status: failed` with a traceback, rather than a misleading successful report.
These are descriptive diagnostics: no universal science acceptance thresholds
or transit-depth bias claim is attached to a successful execution.

Offline regression checks:

```sh
python -m pytest tests/test_validation.py tests/test_jwst_compatibility.py -q
```

## GP precision sweep from a completed run

The executed experiment and its interpretation are documented in
[GP sweep report](../validation/gp-sweep-report.md). Its 34-setting search gave a small
full-segment RMS decrease (265.62 → 264.30 ppm), but no held-out improvement;
the report recommends retaining the existing settings.

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python validation/gp_sweep.py \
  --run validation/runs/20260923T042304707564Z \
  --output validation/runs/20260923_gp_sweep
```

This reads the saved extraction-input FITS cube, traces, errors, DQ, wavelengths
and reference profiles. It does not repeat detector calibration or tracing,
download data, modify the source run, or change library/configuration defaults.
The extraction code must match the source run. Before searching, re-extraction
with its saved profile must reproduce the original normalized light curve to
within 0.1 ppm; the actual difference is saved. This checks that float32 FITS
storage and mask reconstruction are adequate for the comparison.

The default search tests Matérn-3/2 and Matérn-5/2 kernels, length scales
30/100/300/1000 dispersion pixels and amplitudes 0.03/0.1/1/3, retaining the
source inducing-point count. It also tests 12 and 48 inducing points at the
best grid settings; these are model-resolution checks rather than additional
kernel hyperparameters. Change the grid through `--length-scales`,
`--amplitudes`, `--kernels` and `--resolution-check`.

The protocol holds aperture, spatial spacing, rejection thresholds, errors,
preprocessing, median-filter window, integrations and wavelength mask fixed.
A candidate that loses measurements in that band is rejected; it cannot lower
its RMS by discarding difficult wavelengths or exposures. Only GP settings vary.

- The first half of the integrations is the development set; the second half
  is held out. By default, 32 evenly distributed development integrations supply
  per-integration profiles, which are combined by the existing Stage 2 median-
  profile procedure. `--train-profiles` changes this screening sample size.
- Selection minimizes development RMS after the existing spectral replacement
  and median filtering. The replacement template/scatter is estimated only from
  development integrations. Median filtering is performed separately in each
  half so its windows cannot cross the split.
- Raw, pre-replacement RMS must stay within 2% of the development baseline.
  A spectrum guard also rejects large changes: the 95th percentile fractional
  median-spectrum change must stay below 2% on bright columns with complete
  apertures in at least 90% of development frames. These are explicit comparison
  guards, not universal science acceptance criteria.
- Settings are selected before inspecting holdout scores. Both selected and
  baseline profiles are then fitted using **all** development integrations and
  evaluated on the holdout. Paired 20-point moving-block bootstrap intervals
  summarize the RMS difference; raw/corrected RMS, first-difference noise,
  binned residual RMS and replacement fractions are retained.
- Finally the selected profile is fitted on every integration, reproducing the
  original full-run extraction procedure for a directly comparable figure and
  RMS. This full-run score is descriptive; it is not another independent test.

The holdout is conditional on the source run's fixed preprocessing: its trace
and background/1f estimates already used the full exposure. It is not a wholly
independent observation. A small RMS decrease alone does not establish preserved
transit depth or justify replacing the default settings. No ramp-level injection
or transit fit is performed by this sweep.

`sweep.json` records input/code hashes, the exact protocol, every candidate,
failed fits, locked selection, holdout intervals and full-refit metrics.
`candidates.csv`, `sweep.png`, `white_light_comparison.png` and
`spectrum_comparison.png` summarize results. Candidate and final NPZ files
contain profiles, raw/corrected spectra, errors and replacement masks; CSVs
contain light curves and residuals. `selected_config.json` is a separate proposed
configuration, not an automatic update of the user's config.

To continue an interrupted sweep, repeat the same command and add `--resume`.
Completed candidates are reused only with matching input/code hashes and sweep
settings. Regression checks for fixed bands, holdout isolation and bootstrap
direction are in `tests/test_gp_sweep.py`.

## Executed reference

The final local reference is
`runs/nrs1_segment001_20260922_final/reduction.json` (`validation/runs/nrs1_segment001_20260922_final/reduction.json`).
It covers all 391 integrations of segment 1 with JWST 3.0.0 and CRDS context
`jwst_1584.pmap`, using the display-only measured preset and original extraction
settings. Its calibrated FITS products are referenced in the earlier
`runs/nrs1_segment001_20260922/` directory; keep that directory too.

| Quantity | Simple | Polynomial | GP |
|---|---:|---:|---:|
| White-light residual RMS (ppm) | 264.1 | 273.0 | 265.6 |
| Median simple minus method (DN/s) | — | 0.8294 | 0.3201 |

All curves use the same 1537 columns. The final
median-spectrum comparison (`validation/runs/nrs1_segment001_20260922_final/05_median_spectra.png`)
and white-light comparison (`validation/runs/nrs1_segment001_20260922_final/06_white_light.png`)
are available locally; large run products are git-ignored. The run took 520 s
after reusing calibration. It tests execution and numerical products, not
transit-depth bias or the entire three-segment observation. See
[development notes](https://github.com/nespinoza/transitspectroscopy/blob/master/docs/dev-notes/CODEX_CHANGELOG.md) for the history of these changes.

## Alternative ramp fitting

The full real-data MIRI comparison is complete: see
[the ramp-fitting comparison report](../validation/miri-ramp-comparison-report.md) for both
visits' plots and FITS/NPZ/CSV products, the corrected short-tail batching, and
the measured precision of JWST versus the alternative fitter.

The NIRSpec, SOSS and MIRI drivers accept `"ramp_fit_method": "sparta"` plus
`"ramp_fit_options": {"residual_correction": false, "sigma": 5.0}` in their
JSON configurations. Use uncalibrated ramps; for SOSS/MIRI specify
`--input-product uncal`. Omitted settings retain the JWST fitter. Reuse checks
reject changed methods/options. See the [ramp-fitting guide](../guides/ramp-fitting.md) for
the optional empirical correction, supported read patterns and uncertainties.
