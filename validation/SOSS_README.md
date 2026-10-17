# NIRISS/SOSS reduction and validation

This path follows the [reading/tracing notebook](https://github.com/nespinoza/how-to-soss/blob/main/Part%201%20-%20Reading%20data%20and%20tracing%20the%20spectrum.ipynb),
using PASTASOSS instead of manually chosen trace starting positions. It produces
separate spectral products and light curves for orders **1, 2 and 3**. The
three-order reduction requires NIRISS `CLEAR/GR700XD`, `SUBSTRIP256` science data
and JWST 3.0 in the repository's [full environment](../environment.yml).

## Run the HAT-P-14 b case

From the repository root, after activating the full environment and setting up
CRDS as described in the main README:

```sh
python validation/validate.py --mode soss --download \
  --output validation/runs/hatp14_soss
```

The preset selects PID 1541 observation 1, exposure
`jw01541001001_04101_00001`, all four science segments, 572 integrations.
Downloading its Stage 1 `rateints` products takes approximately **6 GB**. This is
the starting product used in the notebook. The manifest records the upstream
calibration version; this command does not claim to recalibrate raw ramps.
The default `config.soss-hatp14-optimal.json` runs box, polynomial optimal and
GP optimal extraction. The older `config.soss-hatp14.json` remains an explicit
box-only preset. Existing downloaded files are reused. Each validation output directory must be
new, and inputs remain unchanged. `--segments 1 --skip-transit` provides a smaller
extraction/plot smoke test; a single segment is insufficient for a transit fit.

To start from raw data and execute local detector calibration and ramp fitting:

```sh
python validation/validate.py --mode soss --input-product uncal \
  --input-glob 'path/to/jw01541001001_04101_00001-seg*_nis_uncal.fits' \
  --output validation/runs/hatp14_soss_from_raw
```

Add `--download` instead of `--input-glob` to download raw segments. Raw processing
requires additional memory, disk space and CRDS references. It saves locally
generated rateints and then enters exactly the same post-ramp reduction. The
executed reference below used archive rateints; raw recalibration of this entire
SOSS visit has not been executed as part of that reference run. Existing detector
calibration options are in the separate `config.soss-hatp14.json` preset.

PASTASOSS and background files are normally selected by CRDS using the **science
metadata**, including observation date, filter and exposure type. Explicit
`--pastasoss-reference file.asdf` and `--background-reference file.fits` overrides
support pinned/offline runs. See [reference provenance](references/soss/README.md).
Image limits and extraction/transit settings are in
[`config.soss-hatp14-optimal.json`](config.soss-hatp14-optimal.json); the original
box-only and NRS1 configs are unchanged.

## Library interface and processing order

```python
import transitspectroscopy as ts

dataset = ts.jwst.load(uncal_files, outputfolder="my_reduction")
dataset.detector_calibration()
dataset.fit_ramps()
products = ts.jwst.stage2(
    {"rampstep": dataset.rateints_per_segment, "times": dataset.times},
    aperture_radius=15,
    outputfolder="my_reduction",
    soss_options={"sample_integrations": 32, "optimal_methods": ["polynomial", "gp"]},
)
order1_flux = products["orders"][1]["flux"]
order3_lightcurve = products["orders"][3]["white_light"]
gp_order1_flux = products["optimal"]["gp"]["orders"][1]["flux"]
```

Alternatively, avoid holding every segment open with:

```python
products = ts.soss.reduce_soss(rateints_paths, outputfolder="my_products",
                              optimal_methods=("polynomial", "gp"))
```

The reduction performs these operations in order:

1. Requires ramp-fitted `CubeModel` inputs in DN/s and checks segment identity
   and time ordering. Reads BJD_TDB times using integration numbers, including
   tables that contain times for the entire exposure rather than one segment.
2. Samples integrations across the supplied segments and constructs a median
   rate image, retaining the science/error/DQ mask.
3. Runs the official **Stage 2 `BackgroundStep`**, after ramp fitting, on that
   image. Its SOSS reference-template algorithm supplies a static background
   image, subsequently subtracted from every integration. This intentionally
   uses a representative median rather than fitting a different background
   normalization to each transit integration. Failed/skipped background fits
   are fatal. Already background-subtracted inputs are rejected to prevent
   double subtraction. [JWST background documentation](https://jwst-pipeline.readthedocs.io/en/stable/jwst/background/description.html)
4. Evaluates all three PASTASOSS trace/wavelength models at the measured
   `PWCPOS`, then refines each trace with the notebook's double-Gaussian peaks
   at ±7.5 pixels with sigma 3. Correlation searches stay near the reference;
   rejected offsets are excluded from a smoothed offset spline. Wavelengths
   retain the PASTASOSS solution at native detector columns; there is no
   unmeasured dispersion shift or empirical wavelength recalibration.
   [PASTASOSS reference documentation](https://jwst-pipeline.readthedocs.io/en/latest/jwst/extract_1d/reference_files.html#pastasoss-reference-file)
5. Subtracts that static background, then optionally removes residual column
   offsets using pixels outside **all three** traces and their wings. This
   post-background column correction is the SOSS meaning of `scale_1f`.
6. Extracts fractional box apertures for each order, propagating squared pixel
   weights into the variance. Default radius is 15 pixels. Complete aperture
   support is required; missing/saturated pixels do not become zero-flux
   measurements. No temporal replacement of spectra is performed.
7. Forms white-light curves using a **fixed band** with valid flux/error in
   every supplied integration, sufficient detector-edge clearance, and trace
   separation greater than two aperture radii plus 10 pixels. The actual mask
   is saved. Inter-order contamination is flagged rather than modeled away.

The default excludes `DO_NOT_USE`, `SATURATED`, nonfinite science/errors and
nonpositive errors. `mask_dq=True` additionally excludes every nonzero DQ flag,
including informational flags such as successfully fitted jumps. This strict
option can leave no common band in long exposures. Excluding a whole wavelength
whenever any aperture pixel fails is conservative and sacrifices coverage; the
HAT-P-14 box run retains only 81/100/120 common columns in orders 1/2/3. Per-integration
spectra retain the larger available coverage with explicit missing values.

## Optimal extraction

Box extraction remains the default library behavior and retains its original
result keys and filenames. The validation additionally trains one **shared
intrinsic profile per order and method**, using four sampled independent
integrations spanning the exposure. Each column's flux is a free nuisance
parameter during training, so a transit is not assumed to be profile evolution.
The coefficients are frozen during extraction. With
`single_trace_extraction=False`, the same coefficients are evaluated through
each integration's measured trace geometry; they are not refitted.

These are the existing positive **log-intensity shared-profile** polynomial
and inducing-point GP models, not the legacy unregularized Marsh polynomial
fit. The SOSS preset uses spacing 1.5 pixels, five Legendre terms, GP 12 inducing
points and a broad initial Gaussian width of 8 pixels. The fitted spatial
profile is flexible enough to represent SOSS's two peaks; it is not constrained
to remain Gaussian. GP defaults in the underlying library are unchanged.
Hyperparameters are fixed in the default joint fit; `gp_options` can set their
values. Joint hyperparameter optimization is not implemented and is rejected
explicitly. The older isolated-order path retains its optional optimizer.

The native-pixel likelihood uses supplied rate variances and ignores invalid
pixels without constructing or renormalizing an incomplete empirical profile.
Flux estimates can recover columns with missing pixels, provided at least 50%
of the predicted profile remains observed (`minimum_profile_fraction`). A
column with no usable support returns NaNs. The default
`overlap_strategy='joint'` fits all order contributions simultaneously at each
detector column, including the formerly excluded long-wavelength region. A
20-pixel profile radius (for a 15-pixel extraction aperture) includes contaminating
wings. `profile_radius` can override this support. Profile coefficients are
initialized using uncontaminated pixels and then jointly refined with overlapping
pixels included. Each order's flux is independently profiled out of the shared
likelihood; no equality between different orders' spectra is imposed.

The flux solve uses `F=(PᵀV⁻¹P)⁻¹PᵀV⁻¹D`, retaining the **full inter-order
covariance** `(PᵀV⁻¹P)⁻¹`. A diagonally scaled normal-matrix condition threshold
(`condition_limit=1e6`) rejects indistinguishable profiles. Model support must
still fit on the detector. The old 40-pixel separation criterion remains a
geometric diagnostic and controls the unchanged box white-light band; it no
longer excludes identifiable optimal spectra. Set `overlap_strategy='isolated'`
and leave `profile_radius` unset to reproduce the previous extraction policy.

This is native-column multi-profile optimal extraction. It does **not** implement
ATOCA's wavelength/line-spread-function coupling across detector columns. See
the [official SOSS extraction description](https://jwst-pipeline.readthedocs.io/en/1.20.2/jwst/extract_1d/description.html)
for that separate algorithm. The resulting total native-aperture flux and error are scaled
by the model's fractional-box aperture fraction for comparison with box fluxes.
Errors remain conditional on the profile, with no profile covariance included.
Optional `nsigma` pixel rejection defaults to zero; supplied DQ/error masks
always apply. Training convergence failures are fatal, with no silent fallback.

`ts.jwst.stage2(..., optimal_extraction=True, profile_method='gp',
extraction_backend='python')` enables that single optimal method alongside box
extraction. Use `soss_options` to enable both methods. The C backend and the
single-order `shared_profile` argument are not supported for SOSS; each order
requires its own model.

Existing NIRSpec extraction behavior, GP defaults and masked-profile fixes
remain unchanged. SOSS processing is serial; the direct API accepts paths and
opens one segment at a time. It does not fit or change detector gain/read noise,
apply a flat-field/photometric calibration, or promise absolute flux accuracy.
Errors exclude uncertainty/covariance from fitted background, trace and column
offset corrections. The saved signal remains in DN/s.

## Products and validation plots

Each order has `orderN_spectra.npz`, `orderN_spectra.fits`,
`orderN_white_light.csv`, and `orderN_trace.csv`. They contain native detector
columns, reference/refined traces, micron wavelengths, spectra/errors, BJD_TDB
times and the fixed white-light mask. NPZ also stores trace displacement and
profile width versus time. Width is the **second-moment sigma**, not Gaussian
FWHM, because the SOSS profile is double peaked. Spectral FITS files use a
documented custom layout (`SCI`, `ERR`, `WAVELENGTH`, `TIME`, `DQ`, `WHITE_MASK`),
not a claim to be STScI `x1dints` datamodels. Their local DQ bits flag incomplete
apertures and geometric overlap/edge regions. With `DEBLEND=True`, geometric
overlap (DQ bit 1) is informational: consult finite flux/error and `WHITE_MASK`
for usable joint measurements. `order_flux_covariance.npz` stores covariance
with shape `(integration, detector_column, order, order)`, order labels,
timestamps, detector columns and scaled condition numbers. Unavailable
covariances are NaN, and the diagonal matches the squared spectral errors.
The diagnostic median/background images are
also saved as actual JWST `ImageModel` FITS files.

The root `04_soss_spectra.png` overlays **box, polynomial optimal and GP optimal**
median spectra for each order. Each optimal method also has a complete product
and diagnostic set under `optimal/polynomial/` and `optimal/gp/`, including
white-light/transmission fits, portable `orderN_profile.npz` coefficient models,
and `orderN_profile_support.npz` masks/observed-profile fractions. Profile training
settings, integration IDs, convergence and fit diagnostics are in metadata.
Reuse checks the source hashes of all extraction modules and saved model files.

`08_soss_extraction_comparison.png` compares white-light curves on the
**intersection of fixed valid wavelength columns across all methods**. Its
per-method CSVs, transit fits and JSON metrics use that identical band, while
individual-method products retain their wider available coverage. Median
spectral differences from box extraction are measured on the same intersection.
This avoids treating a different wavelength band as an extraction improvement.
`09_soss_profile_fit_quality.png` reports training residual chi² per valid
pixel; it is not a degrees-of-freedom-corrected goodness-of-fit statistic.
`10_soss_spectral_differences.png` shows the relative spectral differences that
are difficult to see in the overlaid spectra, with matching numerical CSVs.
`11_soss_order_overlap.png` shows order 1–2 flux correlation and the condition of
the simultaneous solve, with CSVs and recovered-overlap counts in the JSON report.
When the PASTASOSS reference is available, `12_soss_throughput_consistency.png`
compares order 2 with the signal predicted from order 1 at the same wavelengths,
using the reference throughput ratio and dispersion widths. This is a diagnostic,
not a constraint on extracted fluxes; aperture losses and calibration uncertainty
are not corrected by it. It exposes residual model sensitivity in the faint red
tail even when the linear deblending solve is well-conditioned.

The other figures cover background subtraction, reference/refined traces, trace
motion/profile width, all three spectra, three white-light curves with 100-point
median filters/residuals, transit fits/residuals, and transmission spectra.
Image scales are config-defined and constant between runs. Transmission figures
include both the full uncertainty range and a labeled zoom, so unconstrained
order 3 bins do not hide the useful order 1/2 measurements. All numerical series,
filters, fit residuals and transmission bins are saved to CSV/NPZ/JSON.

`reduction.json` combines input/product/code hashes, software/reference
provenance, background statistics, time coverage, trace metrics, white-light
scatter, transit/transmission fit diagnostics and limitations. A successful
execution is separate from a claim that the transmission spectrum is scientifically
validated. Failed spectral bins remain in the table with NaNs and `fit_failed=1`,
and an explicit reason in `orderN_transmission_fits.json`. Rank-deficient fits
have no reported finite depth error; poorly constrained and boundary bins are
counted separately.

## Transit comparison and reuse

The preset uses orbital parameters and reference depths from
[Liu et al.](https://arxiv.org/html/2504.08903v1). It drops the first 80 integrations,
fits radius ratio, quadratic limb darkening and a quadratic baseline, and also
fits mid-transit time in each white-light curve. Spectroscopic bins have nominal
R=30 and retain only fixed complete columns. Their mid-times and orbital geometry
are fixed while limb darkening remains free. `b_circular` means `a/Rs*cos(i)`;
the eccentric impact parameter at conjunction is different. Depth means
`(Rp/Rs)^2`, not simply one minus the observed minimum flux.

Reported uncertainties use a local least-squares covariance, residual variance
and time-binning inflation. They are conditional diagnostics, not posterior
credible intervals, especially at limb-darkening bounds or in faint order 3
bins. No common-mode correction is fitted. Do not interpret scatter about the
100-point median filter across transit as the photon-noise limit. White-light
fits and their residuals provide a separate diagnostic.

The published aggregate SOSS depth is plotted for context; it is not a full
reference transmission table. Provide `--reference-spectrum reference.csv` to
overlay a preferred numerical spectrum, with columns `wavelength_microns`,
`depth_ppm`, `depth_error_ppm`. Different bandpasses, limb-darkening treatment,
partial saturation and order contamination limit direct comparisons.

To rerun fits/plots using a completed extraction, without repeating calibration
or background/tracing:

```sh
python validation/validate.py --mode soss \
  --reuse-extraction validation/runs/your_completed_optimal_run \
  --output validation/runs/soss_refit
```

The source extraction's code, saved product hashes and extraction/background
settings must match. A failed validation can also be reused if its extraction
products were fully written; the manifest retains the failure. See
[`SOSS_OVERLAP_REPORT.md`](SOSS_OVERLAP_REPORT.md) for the current correction and validation.
All large inputs/runs/reference binaries are ignored by Git and should be
archived separately.
