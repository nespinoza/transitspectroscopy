# MIRI ramp-fitting comparison — 2026-10-02

The paired reduction covers both LTT 1445 A F1500W visits in PID 9234:
**29,689 integrations, four exposures and eight raw segments**. Both the
official JWST ramp fitter and the new correlated-difference fitter completed.
In this test, **JWST produces lower photometric scatter**.

The final audited results are in
`runs/miri_ramp_comparison_20261002_corrected/` (`validation/runs/miri_ramp_comparison_20261002_corrected`).
The earlier `miri_ramp_comparison_20261001` run remains available for debugging
but contains a short-chunk calibration artifact and is superseded.

## Comparison figures and numerical products

- Visit 1 comparison (`validation/runs/miri_ramp_comparison_20261002_corrected/visit1_ramp_comparison.png`):
  120-second binned photometry, native-integration running-median residuals,
  and paired SPARTA-minus-JWST differences, for both sum and GP photometry.
- Visit 2 comparison (`validation/runs/miri_ramp_comparison_20261002_corrected/visit2_ramp_comparison.png`).
- Comparison metrics (`validation/runs/miri_ramp_comparison_20261002_corrected/comparison.json`),
  complete run/provenance manifest (`validation/runs/miri_ramp_comparison_20261002_corrected/comparison_manifest.json`),
  independent audit (`validation/runs/miri_ramp_comparison_20261002_corrected/audit.json`), and
  product hashes (`validation/runs/miri_ramp_comparison_20261002_corrected/product_manifest.json`).
- Paired numerical time series:
  visit 1 sum (`validation/runs/miri_ramp_comparison_20261002_corrected/visit1_sum_paired.csv`),
  visit 1 GP (`validation/runs/miri_ramp_comparison_20261002_corrected/visit1_gp_paired.csv`),
  visit 2 sum (`validation/runs/miri_ramp_comparison_20261002_corrected/visit2_sum_paired.csv`),
  visit 2 GP (`validation/runs/miri_ramp_comparison_20261002_corrected/visit2_gp_paired.csv`).

Each of `jwst/visit1`, `jwst/visit2`, `sparta/visit1` and `sparta/visit2`
contains the same product set:

| Product | Contents |
| --- | --- |
| `photometry.fits`, `photometry.npz` | Both flux/error time series, BJD_TDB, quality flags, PSF and weights; NPZ also has centroid/width/sky diagnostics. Flux units are Jy. |
| `sum_lightcurve.csv`, `gp_lightcurve.csv` | Physical and normalized fluxes/errors, flags and physical-segment indices. |
| `*_filtered_lightcurve.csv` | Relative flux/error, 100-point running median and residuals in ppm. |
| `*_binned_lightcurve.csv` | Inverse-variance weighted 120-second bins, conditional errors and contributing counts. |
| `gp_psf.npz`, `photometry_metadata.json` | Fitted shared spatial GP, training settings/indices and extraction conventions. |
| `01`–`05` diagnostic PNGs | Images/PSF/weights, centroids/widths/sky, unbinned residuals, binned curves and pixel-fit residuals. |
| `metrics.json`, `centroid_width_background.csv` | Per-method precision and saved geometric/background measurements. |

Calibrated image cutouts for every integration are in each backend's `prepared/`
directory. The corrected directory also contains ten one-integration calibrated
JWST `CubeModel` FITS samples from the repaired tails. Full detector cubes were
not retained because of disk limits. The preceding run retains first-integration
calibrated samples for every segment. Temporary raw downloads were removed only
after successful processing; input names and hashes remain in the manifests.

## Results

RMS values below use a 100-point running median independently within each
physical input segment and **the same valid integrations for both fitters**.
No running-median correction is applied to the saved physical photometry.

| Visit | Photometry | JWST RMS (ppm) | SPARTA RMS (ppm) | SPARTA RMS increase | Common integrations |
| --- | --- | ---: | ---: | ---: | ---: |
| 1 | Sum | 656.72 | 740.53 | 12.76% | 7,264 |
| 1 | Spatial GP optimal | 638.80 | 708.45 | 10.90% | 7,264 |
| 2 | Sum | 645.83 | 734.33 | 13.70% | 22,423 |
| 2 | Spatial GP optimal | 632.90 | 705.94 | 11.54% | 22,425 |

Two transient incomplete sum apertures remain invalid in visit 2 for both
fitters; GP photometry retains every integration. All four spatial GP fits
converged. The alternative ramp fitter reported zero nonconverged pixel ramps.

The robust scatter estimator `1.4826 * median(abs(residual - median(residual)))`
confirms that the difference is not driven solely by a few outliers. For GP
photometry it is 619.12/686.50 ppm (JWST/SPARTA) in visit 1 and
620.99/688.67 ppm in visit 2. Median formal GP errors are 517.90/530.00 ppm
and 516.60/528.75 ppm, respectively: both methods have excess scatter above
their conditional pixel/sky uncertainties.

The new fitter's absolute median flux is about 0.2% lower. The displayed curves
are normalized independently by their visit medians, so this constant offset
is removed from the relative comparison. These measurements do not determine
whether transit or eclipse depths are biased. No astrophysical/systematics
model or injection/recovery experiment was fit to these real-data light curves.

## What was held fixed

The target, exposures, extraction aperture, background annulus, spatial GP
settings and plot settings come from
`miri_9234_validation_20260927_reviewed/reduction.json`. That older run began
with archive rateints. This comparison instead downloaded the corresponding
**uncalibrated ramps** and repeated detector calibration.

- JWST 3.0.0 and CRDS `jwst_1584.pmap` are used for both branches.
- Official Detector1 executes through jump detection; the same calibrated
  ramp and masks then go to both fitters. JWST uses `OLS_C`, one core.
- The alternative uses `sigma=5` and **`residual_correction=False`**. This tests
  correlated-difference fitting and its clipping. It does **not** test the
  optional empirical shared ramp-residual correction, or reproduce every
  SPARTA processing choice.
- Each branch receives the official gain-scale and Image2 WCS, flat and
  photometric calibration. Neither branch is resampled.
- Both use the existing five-pixel fixed sum aperture, sky annulus of 12–20
  pixels, and shared spatial GP PSF trained on 12 integrations per visit.
  The 21×21 inducing grid and fixed GP hyperparameters match the previous run.
- Times and exposure gaps are preserved. Each branch uses its own flux/error
  arrays and independently fitted spatial PSF with otherwise identical settings.

The optional empirical ramp correction was deliberately disabled to avoid
training a shared temporal correction on a visit containing astrophysical
events. A future test of that option needs a justified homogeneous training
selection and injection/recovery checks. The present result is not evidence
against every possible configuration of the alternative method.

## Calibration artifact found and corrected

The 8-GB workstation and limited free disk required calibration in 128-integration
chunks. Five physical-segment tails contained only 64, 32, 58, 58 and 58
integrations. With `only_use_ints=True` and `minimum_sigclip_groups=100`, JWST
changes jump-detection algorithms for those smaller batches. The first pair of
plots exposed large differences at precisely these tails.

The validation driver now pads short final batches backwards to retain 128
calibration integrations, runs both fitters with that identical context, and
keeps only the originally requested tail integrations. Thus output counts and
times are unchanged. Only those five tails were recalibrated. Raw FITS byte-range
requests avoided downloading the entire files again; ranges, hashes and ETags
are recorded. The paired segment-end excursions disappear in the corrected
plots. Unaffected cutouts are reused without modifying the original run.

The corrections are in `miri_ramp_compare.py`; the one-time recovery procedure
is `miri_ramp_repair_tails.py`. This fixes the validation batching artifact,
not either ramp-fitting algorithm. Both still share chunk-based EMI estimation
and temporal jump statistics, which can differ from a full-segment Detector1
run. The result is a matched comparison under this explicitly documented
calibration setup, not a proof of invariance to calibration batch size.

## Verification and reproduction

`audit_miri_ramp_comparison.py` verified all 470 prepared-product hashes, source
provenance, contiguous integration coverage, equal backend times, successful GP
fits, PSF normalization, FITS/NPZ/CSV agreement, bin counts and independently
recomputed RMS metrics. The product manifest hashes all retained final products.

The initial full paired processing session took about 104 minutes, dominated
by detector calibration and compressed raw-file access. Summed ramp-fitting
times for the retained chunks are about 155 s for JWST and 447 s for SPARTA;
median per-chunk times are 0.65 s and 1.85 s. These are workstation timings,
not general scaling estimates; overlapping tail context is included.

To make a fresh full comparison with the corrected batching logic:

```sh
CRDS_CONTEXT=jwst_1584.pmap OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python validation/miri_ramp_compare.py \
  --source-run validation/runs/miri_9234_validation_20260927_reviewed \
  --output validation/runs/miri_ramp_comparison_new \
  --chunk-integrations 128
```

Use `--resume` with the same settings/source hashes if interrupted. Validate
saved products with:

```sh
python validation/audit_miri_ramp_comparison.py \
  validation/runs/miri_ramp_comparison_20261002_corrected
python -m pytest tests/test_miri_ramp_compare.py tests/test_ramp_fitting.py -q
```

The additional regression tests cover integration slicing and timestamps,
short-tail context/trimming, and validated seekable HTTP range access. Their
synthetic inputs require no MAST downloads or remote CRDS lookups.

The final full regression suite passed **157 tests**, with 13 skipped and two
existing shared-profile hyperparameter-bound warnings. Compilation and
`git diff --check` passed as well.
