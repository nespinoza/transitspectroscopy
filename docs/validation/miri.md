# MIRI imaging and photometry

`transitspectroscopy.miri` processes MIRI `MIR_IMAGE` time-series observations
using official JWST calibration and local sum/2-D spatial GP photometry. Eureka
is neither imported nor required. The supplied [MIRI notebook](https://github.com/spacetelescope/jwebbinar_prep/blob/main/jwebbinar41/MIRI_Photometry/Notebook1.ipynb)
informed the sequence of calibration, centroiding, sky estimation and photometry;
its pipeline wrappers are not used here.

## Run LTT 1445 A

Activate the repository's full environment and configure CRDS, then run:

```sh
python validation/validate.py --mode miri --download --compress-downloads \
    --output validation/runs/ltt1445_miri
```

The preset uses all eight public F1500W science segments in program 9234
observations 1 and 2: 7,264 integrations in the first visit and 22,425 in the
second. The second visit contains three consecutive exposures. Exposure gaps
are retained and acquisition images excluded. Inputs total about 2.74 GB
uncompressed; gzip downloads save space without losing information. Existing
inputs are reused and the output directory must be new.

Default input is **archive Stage 1 rateints**. Official `Image2Pipeline` runs
WCS assignment, flat field and photom; resampling is disabled. This mode does not
recalibrate raw ramps. `--input-product uncal --download` additionally runs
official `Detector1Pipeline`, including MIRI-specific ordering and current
reference-controlled settings. The historical `jwst.load` detector helper is
not used by this validation path. Raw processing needs more storage.

`--input-product calints --input-glob 'path/*_calints.fits'` starts from existing
imaging calibration; completed flat-field calibration is required. Local FITS
and gzip FITS are supported. `--visits 1` selects one visit. A narrower input glob
permits a segment smoke test; JSON distinguishes partial coverage from all
configured segments.

Settings are in [`config.miri-ltt1445.json`](https://github.com/nespinoza/transitspectroscopy/blob/master/validation/config.miri-ltt1445.json). Target
positions are zero-based `(x,y)` in the input subarray. WCS initializes bounded
Gaussian-core centroiding; its fitted amplitude is not the science flux. An
explicit `center` overrides WCS. `exclude_sources` accepts `[x,y,radius]` circles
for crowded fields; automatic multi-star deblending is not implemented.

## PSF, weights and flux conventions

The GP models the **2-D logarithm of the spatial PSF**, using an RBF kernel,
whitened inducing coefficients with normal priors and a Gaussian log-intensity
mean. Pixel quadrature precedes normalization. Validation uses 21×21 inducing
points, a 1-pixel length scale, amplitude 2, a 21×21 detector stamp and 12 training
integrations per visit. These settings are distinct from spectral GP defaults.

One coefficient set is fitted jointly with independent training-image fluxes,
using the repository's variable-projection Gauss–Newton optimizer. Coefficients
remain fixed during extraction; each integration evaluates the PSF at its own
centroid. Only the model is interpolated for faster evaluation. Its agreement
with direct GP evaluation is checked; science pixels are never resampled.
This is an inducing-point GP MAP fit with fixed kernel hyperparameters, not a
temporal GP or hyperparameter optimization.

For sky-subtracted pixels `d`, normalized profile `P` and covariance `C`:

```text
F = (Pᵀ C⁻¹ d) / (Pᵀ C⁻¹ P)
Var(F) = 1 / (Pᵀ C⁻¹ P)
C = diag(pixel_variance) + sky_mean_variance × 11ᵀ
```

The sky term propagates uncertainty shared by all source pixels. Sky is a
spatially clipped, inverse-variance annular mean; excess scatter inflates its
uncertainty. The source time series is not sigma-clipped or replaced.
`DO_NOT_USE`, `SATURATED`, nonfinite pixels and nonpositive errors carry no
likelihood. Default optimal extraction requires 80% observed PSF support. It
never renormalizes an incomplete empirical image to train the profile.

- **Sum:** fractional circular aperture fixed on the visit median detector
  centroid. Permanently invalid pixels receive one fixed detector exclusion
  mask for the visit. Further transient missing pixels produce a flagged NaN.
  There is no model-based flux repair. Errors include pixel and sky variance.
- **GP:** fitted stamp flux scaled by one fixed, unmasked reference-aperture
  fraction. A dead pixel moving through the PSF therefore does not introduce an
  artificial time-dependent aperture correction. This is not an infinite-aperture
  correction. Its absolute flux need not equal a sum excluding a dead pixel.

MJy/sr and errors are converted to Jy/pixel using `PIXAR_SR` and relative `AREA`.
Explicitly skipping `photom` retains DN/s; gain is not guessed. Errors exclude
PSF/centroid uncertainty, residual detector systematics and absolute calibration
or aperture-correction uncertainty.

## Products and reuse

Each visit directory includes:

- `sum_lightcurve.csv`, `gp_lightcurve.csv`: integration BJD_TDB, flux/error,
  normalized flux/error, flags and segment IDs.
- `*_binned_lightcurve.csv`: weighted 120-second bins, errors and sample counts.
- `*_filtered_lightcurve.csv`: the 100-point median diagnostic and residuals.
  This filter is never divided out of the delivered photometry.
- `photometry.fits`/`.npz`: both methods, errors, original times, flags, PSF and
  weights. The FITS layout is a local photometry product, not JWST `calints`.
- `gp_psf.npz`: portable spatial GP and fitted coefficients.
- `median_image.fits`: diagnostic JWST ImageModel. PNGs show the image, PSF,
  weights, centroids, widths, sky, light curves and pixel residuals.
- `photometry_metadata.json`, centroid/background and comparison CSVs.

Quality bits: bit 0 means centroid/sky failure; bit 1 means a transient
incomplete sum aperture; bit 2 means insufficient optimal support. Bit 1 alone
does not invalidate GP flux. Permanent exclusions are separate metadata.
`reduction.json` records config, software, input/source/product hashes, counts,
RMS, profile quality and coverage. Units and time conventions are explicit.

`prepared/` retains calibrated sky-subtracted cutouts, variances, sky covariance,
centroids, original times and provenance. Full detector cubes are not stacked
across the visit. `--save-calints` optionally retains full calibrated cubes.
Otherwise they are processed in memory and compact cutouts retained.

```sh
python validation/validate.py --mode miri --prepare-only \
    --output validation/runs/miri_calibration
python validation/validate.py --mode miri \
    --reuse-prepared validation/runs/miri_calibration/prepared \
    --output validation/runs/miri_new_extraction
```

Preserve both `.npz` and paired `.json` cache files. Reuse verifies hashes and
preparation/calibration settings and retains their original provenance;
extraction settings may change. Optional `--temporary-downloads` removes only
newly downloaded inputs after cutouts are saved; it never deletes pre-existing
files. It was not used for the executed validation.

## Library interface and tests

```python
import transitspectroscopy as ts

result = ts.miri.reduce_miri(
    rateints_paths, outputfolder="miri_visit",
    preparation_options={"annulus": (12, 20)},
    gp_options={"n_inducing": 21, "length_scale": 1.0},
)
sum_flux = result["flux"][:, 0]
gp_flux = result["flux"][:, 1]
```

`ts.jwst.stage2({"rampstep": rateints_models}, miri_options={...})` dispatches
MIRI imaging to this path and returns both methods. Other instruments and
spectral defaults are unchanged. `GPPSF.save/load`, `fit_gp_psf`, `optimal_flux`,
`prepare_segment` and `extract_visit` support controlled experiments.

```sh
python -m pytest tests/test_miri.py tests/test_photometry.py -q
python tests/simulated_miri_photometry.py --output validation/runs/miri_injection
```

The second command creates independent noisy Airy-PSF transit/eclipse simulations,
plots and recovery metrics. See [the executed report](../validation/miri-report.md).
