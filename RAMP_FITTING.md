# Alternative correlated ramp fitting

The default remains the official JWST `RampFitStep`. Select the alternative
**before detector calibration**:

```python
import transitspectroscopy as ts

dataset = ts.jwst.load(
    filenames, outputfolder="reduction",
    ramp_fit_method="sparta",
    ramp_fit_options={"residual_correction": True, "sigma": 5.0},
)
dataset.detector_calibration(save=True)
dataset.fit_ramps(save=True)
rates = dataset.rateints       # DN/s, integration x row x column
errors = dataset.rateints_err  # DN/s
```

`ramp_fit_options` configures the alternative; the existing
`parameters={'ramp_fit': ...}` dictionary configures the JWST backend. Existing
calls, JWST defaults and JWST cached product filenames retain their behavior.
The alternative always refits, so a saved JWST rate cube or an earlier choice
of alternative settings cannot silently replace the requested calculation.

The same selection is available in `ts.jwst.stage1(...)`,
`ts.miri.calibrate(..., input_product='uncal')`, and
`ts.miri.reduce_miri(..., input_product='uncal')`. MIRI uses a subclass of the
official Detector1 pipeline that replaces only `ramp_fit`: reset, first/last
group handling, dark correction and subsequent `gain_scale` retain their
official ordering. Do not request a new ramp-fitting method on already fitted
MIRI `rateints`/`calints` inputs.

## Method and relationship to the references

The scientific references are the Methods/Data reduction section of
[arXiv:2305.06240v1](https://arxiv.org/html/2305.06240v1) and
[SPARTA's calibration implementation](https://github.com/ideasrule/sparta/blob/master/calibrate.py).
This is an independent NumPy implementation of the equations, with no SPARTA,
CuPy or other pipeline dependency.

For adjacent-read differences `d` in electrons, expected electrons per group
interval `s`, and RMS noise `R` of one read, the covariance is

```text
C[j,j]   = max(s, 0) + 2 R²
C[j,j+1] = C[j+1,j] = -R²
w = solve(C, ones)
alpha = w / sum(w)
slope = sum(alpha * d)
variance = 1 / sum(w)
```

The estimate transitions between ordinary least squares in the read-noise
limit and first-minus-last differencing in the photon-noise limit. A first
fit can supply a median residual template per group/pixel across integrations;
subtracting it precedes a second fit with outlier rejection.

Implementation details specific to this library:

- The covariance solve is tridiagonal, avoiding exponential overflow in the
  analytic expression. The median valid difference supplies the initial signal
  estimate. Negative measured slopes are retained; only the photon variance is
  constrained nonnegative.
- Rejection removes the worst difference above `sigma` per ramp per iteration,
  monotonically. We solve the covariance of the retained differences again;
  merely zeroing weights from the full covariance is incorrect at gaps.
- `DO_NOT_USE`, nonfinite reads, and saturation exclude affected differences.
  Saturation propagates through later groups. `JUMP_DET` excludes the difference
  ending at the flagged group; subsequent valid differences can still be used.
  A single valid group cannot produce a slope. Unusable/nonconverged ramps get
  NaN rates/errors and `DO_NOT_USE`; detected jumps also set output `JUMP_DET`.
- Residual-template construction removes each ramp's arbitrary pedestal before
  taking the integration median. Jump-flagged ramps do not train that pixel's
  template. Missing template elements have a zero correction. This is not a
  bit-for-bit reproduction of SPARTA's clipping or template bookkeeping.
- No instrument-specific gain, last-group removal, border masking or flat-field
  correction is hardcoded in the fitter. Detector calibration and its DQ flags
  control those choices. Inclusive `firstgroup`/`lastgroup` limits are available.
- Per-integration rates are independent conditional on the template. Only the
  separate exposure-level `rate` image averages integrations, with inverse
  variance weights; light curves use the `rateints` cube.

## Options and units

| Option | Default | Meaning |
| --- | --- | --- |
| `residual_correction` | `True` | Train/subtract the empirical residual template before the clipped fit. |
| `template_integrations` | All integrations in this segment | Unique zero-based local indices to train the template; extraction still fits all integrations. |
| `residual_template` | `None` | Supply a finite `(group,row,column)` array in DN instead of training. |
| `sigma` | `5.0` | Difference rejection threshold; `None` disables clipping. |
| `max_iterations` | `50` | Rejection iteration cap; ramps still rejecting at the cap are flagged unusable. |
| `firstgroup`, `lastgroup` | All groups | Inclusive zero-based range; at least two groups must remain. |
| `gain` | JWST reference | Scalar or detector-shaped array, electrons/DN. |
| `read_noise` | JWST reference | Scalar or detector-shaped array, **RMS electrons per single read**. |
| `override_gain`, `override_readnoise` | CRDS selection | Local JWST reference files instead of CRDS; do not also supply the corresponding scalar/map. |
| `chunk_pixels` | `256` | Upper bound on simultaneous spatial pixels; automatically reduced for large integration/group counts. |

The direct array API is
`ts.ramp_fitting.fit_ramps(data, group_time=..., gain=..., read_noise=..., ...)`.
It accepts optional `groupdq` and `pixeldq`, and returns arrays and diagnostics
without requiring JWST. `fit_model(ramp_model, **options)` uses JWST datamodels
and returns `(rate, rateints, diagnostics)`; the caller owns both models.

Calibrated input is DN. Internally, the fitter converts to electrons and returns
DN/s by dividing by `gain * group_time`. Variance components divide by that
factor squared. Explicit `read_noise` is already per-read electrons. A JWST
READNOISE reference is converted as `reference * gain / sqrt(2)`, matching
stcal's convention for one frame per group. The paper's historical fixed gain
is not used. Reference subarrays follow the science model's detector coordinates.

`ERR² = VAR_POISSON + VAR_RNOISE`. For normalized difference weights `alpha`,
the components before unit conversion are `s * sum(alpha²)` and
`R² * (2 sum(alpha²) - 2 sum(alpha[j] alpha[j+1]))`. `INT_TIMES`, applicable
metadata and reference names survive into the rate datamodels.

## Applicability and uncertainty limits

This implementation requires **NFRAMES=1, uniform group timing, and at least
two usable adjacent reads**. Nonempty `read_times` metadata and averaged groups
are rejected; use JWST for those data. It does not bridge over missing groups,
fit a one-group/zero-frame slope, infer gain/read noise, model interpixel noise,
or propagate detector-reference uncertainties.

The empirical correction assumes a repeatable residual ramp shape within its
training block. For transits/eclipses or changing illumination, use a suitable
homogeneous training subset, split the data into appropriate blocks, supply an
independently trained template, or set `residual_correction=False`. Selecting
out-of-event training indices is a user decision; the fitter does not identify
events. Small training samples can absorb noise into the template.

Errors condition on the estimated covariance, clipping mask, and residual
template. Template-estimation uncertainty and the correlations it induces
across integrations are **not propagated**. Clipping and signal-dependent
weights can also affect very low-signal error coverage. Injection/recovery on
the actual observation is necessary before interpreting astrophysical depths.

Work per rejection pass scales with integrations × groups × pixels. Pixel
chunks target roughly 64 MiB of working temporaries, in addition to the input,
rate/variance/DQ cubes and group-template output. This is CPU NumPy code;
`maximum_cores` from the JWST step parameters does not parallelize it.

## Products, validation and tests

`load.fit_ramps(save=True)` writes distinct `*_sparta_rate.fits`,
`*_sparta_rateints.fits`, `*_sparta_ramp_diagnostics.npz` and
`*_sparta_ramp_fit.json` products. Diagnostics contain retained/rejected
difference counts, convergence flags, and the residual template in DN.
Provenance records the method, implementation revision, references and options;
array options are represented by shape/dtype/SHA-256. The same information is
available in `dataset.ramp_fit_diagnostics`, and method provenance is included
in FITS history. Legacy `stage1` returns diagnostics in its dictionary; official
Detector1 retains standard rate filenames and the ramp step's diagnostics.

All three validation drivers accept these additional JSON configuration keys:

```json
"ramp_fit_method": "sparta",
"ramp_fit_options": {"residual_correction": false, "sigma": 5.0}
```

Use raw `uncal` inputs (`--input-product uncal` for SOSS/MIRI). Omitted keys
preserve JWST behavior. Reuse checks reject a change in fitter or options;
existing files cannot be relabelled as a new ramp-fitting result. Validation
configuration and FITS metadata record the selection. Ramp diagnostic plots
compare fitted slopes against the original calibrated reads, so residuals can
still show the empirical correction when that correction is enabled.

Run `python -m pytest tests/test_ramp_fitting.py -q`. Tests cover dense and
analytic covariance agreement, both noise limits, unit/error propagation,
Poisson-plus-read-noise injection and coverage, jumps, saturation, missing
groups, negative slopes, template reuse and variable-slope preservation,
chunk invariance, reference conversion, FITS/timestamp round trips, and the
real Detector1 dispatch with other calibration steps skipped on synthetic data.
These are synthetic checks, not a claim of improved real-observation precision.
