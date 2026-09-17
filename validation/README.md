# NRS1 pipeline validation

`validate.py` reduces the HAT-P-14 commissioning exposure used by the original
script, with the current checkout and installed JWST pipeline. It produces
six diagnostic figures, JWST FITS models, portable numerical products, and one
`reduction.json` manifest. Its preset is for **NIRSpec NRS1/G395H**, not a
universal calibration recipe for every instrument or target.

## Run

From the repository root, with the full environment installed:

```sh
conda activate transitspectroscopy
python validation/validate.py --download --segments 1
```

This downloads only the first raw segment (1.03 GB; 391 integrations × 20 groups
× 32 rows × 2048 columns). Existing downloaded files are reused. Direct MAST
product downloads avoid a full-program catalogue query. CRDS downloads the
necessary calibration references into `CRDS_PATH`; if unset, the script selects
`$HOME/crds_cache`. Set a compatible `CRDS_CONTEXT` before running if you need
to reproduce a specific calibration context.

To reduce all three segments of this exposure:

```sh
python validation/validate.py --download --segments 1 2 3
```

To use files downloaded by the previous script:

```sh
python validation/validate.py --input-glob 'JWSTdata/jw01118005001_04101*nrs1_uncal.fits'
```

`--input-glob` uses every matching file and overrides segment/download selection.
Use `--exposure` and `--data-dir` to change the public exposure identifier or
download directory. Input metadata must describe NRS1/G395H segments of one
exposure. The script retains exposure integration numbers and timestamps when
only a subset is loaded.

Each run creates a new directory in `validation/runs/`. You can choose a new
directory with `--output`; existing directories are refused to prevent silently
reusing stale reduction caches. Input and output data are git-ignored locally.
Detector calibration is performed one segment at a time to limit ramp memory;
Stage 2 still operates on the combined rate cube in memory. Full-exposure runs
need several GB of RAM and disk space. Saved calibrated ramps are larger than
raw ramps. No raw files or old runs are deleted automatically.

The driver loads this checkout explicitly. If you enable Ray through `nthreads`
in the configuration, reinstall the checkout first so workers run matching code:

```sh
python -m pip install --no-build-isolation --no-deps .
```

## Fixed settings and comparisons

[config.json](config.json) contains all configurable plotting/extraction values.
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

Differences have sign **simple minus optimal**: the JSON reports the median
over common columns of the difference of the two time-median spectra, in DN/s,
and relative to the simple spectrum in ppm. They are not absolute differences
or differences of separately normalized white-light curves. Extraction errors
are conditional on fitted profiles and omit inter-column/profile/background
covariance, as in the current library.

The JSON is checkpointed after completed stages; exceptions produce
`status: failed` with a traceback, rather than a misleading successful report.
These are descriptive diagnostics: no universal science acceptance thresholds
or transit-depth bias claim is attached to a successful execution.

Offline regression checks:

```sh
python -m pytest tests/test_validation.py tests/test_jwst_compatibility.py -q
```
