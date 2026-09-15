# Shared intrinsic profile implementation

This opt-in API trains one intrinsic profile from independent integrations,
then evaluates it at new supplied trace positions. It does not impose a common
spectrum or retrain the profile during extraction. Existing Marsh/GP calls and
the Stage 2 median-of-independent-profiles strategy remain unchanged.

## Observation model and parameter meanings

For integration t, spatial row i, dispersion column j, and spatial component k:

```
z[j,k] = mean[k] + sum(a,m) contrast[k,a] * coefficient[a,m] * basis[j,m]
G[j,k] = exp(z[j,k])
P[t,j,i] = sum(k) Q[t,j,i,k] * G[j,k] / sum(i,k) Q[t,j,i,k] * G[j,k]
data[t,j,i] = flux[t,j] * P[t,j,i] + background[t,j,i] + noise[t,j,i]
```

`Q` uses the existing integrated triangular spatial basis. Only detector pixels
within the fixed aperture enter the likelihood. The shared coefficients describe
log intrinsic intensities relative to a fixed Gaussian mean (default width
1.5 detector pixels). An orthonormal Helmert contrast removes the redundant
common log-intensity mode. It has K-1 columns for K spatial components.
Exponentials are stabilized by subtracting a common maximum before normalization.

Polynomial functions use a Legendre basis on the trained dispersion interval.
`polynomial_order` counts terms. A Gaussian prior with amplitude
`polynomial_amplitude` replaces the legacy unregularized coefficient fit only in
this new model. GP functions use a whitened inducing-point Matern-3/2 or Matern-5/2
basis; the default is 16 equally spaced inducing points, or the column count if
smaller. The amplitude is a log-intensity prior scale. It is **not** the amplitude
of the existing fraction-fitting GP. Neither fitted amplitude is a flux scale.

The number of shared coefficients is `(K-1)*M`, where
`K = 2*int(aperture_radius/spacing + 0.5)+1 + 2*ceil(1/spacing)` and M is the number of polynomial
terms or inducing points. Aperture radius is an integer. Spatial support is fixed
across training and evaluation; an off-detector aperture raises instead of
silently changing the trained model. Extra intrinsic components cover the full
footprints of boundary pixels, extending at least one pixel beyond the nominal
radius. They do not expand the data aperture. This padding is essential for
unbiased narrow-aperture training with moving traces.

All training measurements use fixed supplied Gaussian variances. Nonfinite
measurements/variances, nonpositive variances, the legacy -9999 sentinel, and
explicit `mask=True` pixels are excluded. At least one training integration
must have two valid pixels in every column. An individual integration/column
with fewer than two valid pixels contributes no profile-training information.
Masks are frozen throughout optimization; there is no optimization-time sigma
clipping. Identify cosmic rays before training or provide a conservative mask.

Background means may be supplied as an integration-indexed sequence. Their
uncertainty, and any remaining photon noise from subtracted backgrounds, must
already be represented in the supplied variances. Temporal/spatial correlations
are not modeled by this diagonal likelihood.

## Training and optimization

`fit_shared_profile(data, data_variance, centroids, ...)` accepts cubes, memory
maps, or indexable sequences with `len` and integer indexing. It reads only the
selected `training_indices` and retains cropped pixels, variances, and traces.
The full science cube is never stacked internally. The cropped training set
remains resident: this is not an unbounded out-of-core optimizer. If the working
estimate exceeds `memory_limit_mb` (default 1024), select fewer integrations or
explicitly adjust the budget/model resolution.

Coordinates default to consecutive detector columns. Explicit
`dispersion_coordinates` must increase strictly and have one value per column.
The default GP length-scale initialization assumes detector-pixel units; set
`gp_options['length_scale']` explicitly when using other coordinate units.
Traces may be `(integration,column)` or a common `(column,)` vector. Training
IDs are recorded with the selected indices and a content fingerprint of cropped
data, effective variances, and traces.

The inner solver is a damped variable-projection Gauss-Newton method. It updates
the shared coefficients while analytically eliminating each integration's free
flux in each column. It includes the whitened Gaussian coefficient prior once.
An Armijo line search checks the actual nonlinear objective; convergence uses
the coefficient gradient scaled by local curvature or a squared Gauss-Newton
step below 1e-8. For large likelihood sums, the latter includes a floating-point
rounding floor capped at 1e-7 (a step smaller than 0.00032 in its local posterior
metric). Small-data fits retain the tighter threshold to keep finite-difference
evidence gradients stable. Stopping reasons and the decrement threshold are
recorded. Nonconvergence raises.

For hyperparameter fitting, the exact coefficient Hessian **conditional on the
optimized fluxes** is evaluated at the fitted coefficients. The objective is:

```
negative conditional log marginal likelihood (Laplace)
  = negative pixel log likelihood + 0.5 * dot(coefficients, coefficients)
    + 0.5 * logdet(conditional posterior Hessian)
```

Spatial derivatives are projected before forming curvature, and the Helmert
diagonal correction uses prefix sums rather than cubic dense basis transforms.
Gaussian pixel likelihood normalization is included. Coefficients are whitened,
so the prior normalization and Laplace integration constants cancel. The Hessian
must be positive definite; it is not silently repaired with fitted jitter.
This is an approximation to coefficient marginalization, conditional on fitted
fluxes. It is not full marginalization over the spectra or hyperparameters.

The outer optimizer uses bounded L-BFGS-B in log hyperparameters, finite
differences (central in the interior; one-sided at a bound), warm starts projected
into each new basis, and deterministic multiple starts (default two).
Only converged outer endpoints with converged inner fits are eligible for the
returned model. Failed trial fits receive a finite rejection penalty and are
recorded; total optimization failure raises rather than silently returning the
initial model. Set `optimize_hyperparameters=False` to fit only shared coefficients.

The default amplitude bounds are 0.03 to 10. GP length-scale bounds run from the
inducing-point spacing to ten times the dispersion span. Override with
`hyperparameter_bounds={'amplitude':(low,high), 'length_scale':(low,high)}`; for
polynomials only amplitude applies. A bound-hitting result emits a warning and
is recorded in diagnostics. Test increasing inducing-point count if length scale
hits its lower bound. Kernel family, polynomial degree, inducing-point count,
mean width, traces, and noise parameters remain fixed during a run.

## Reuse, outputs, and uncertainty

`SharedProfile.evaluate(trace, dispersion_coordinates=None)` returns a fresh
normalized detector array. Supplied coordinates, if given, must match training;
no wavelength extrapolation or implicit registration is performed. Without
explicit coordinates the caller is responsible for using the same grid.

`SharedProfile.extract(data, trace, data_variance, mask=None, background=None,
nsigma=0, return_P=False)` uses native, whole detector pixels in the aperture.
It returns the existing 3-by-column layout: column number, flux, inverse variance.
The flux convention is the total represented by the normalized aperture profile;
no external aperture correction is inferred. Invalid columns return NaN flux
and zero inverse variance. Optional positive `nsigma` iteratively rejects the
largest standardized residual in each column. This rejection is an extraction
option, not part of training or a fitted outlier likelihood.

`getOptimalSpectrum(..., P=model, data_variance=variance)` delegates to this
native-pixel extraction without fitting. Aperture and spacing arguments must
match the model, the backend must be auto/python, and legacy column bounds and
GP options cannot override a trained model's grid/settings. In contrast, passing
the **array** returned by `evaluate` into old extraction intentionally retains
the legacy fractional-edge data/variance behavior. Prefer the model interface
for consistency with its training likelihood.

`model.diagnose(...)` evaluates held-out chi-square by column with the profile
fixed and flux re-extracted. Inspect these residuals across time, wavelength,
and trace position. Changing intrinsic width/asymmetry needs new training
intervals or a future time-dependent model; there is no automatic refit.

`model.covariance` stores the local coefficient covariance conditional on fitted
training fluxes and hyperparameters, unless `store_covariance=False`. Extracted
errors remain conditional on the profile. Shared-profile uncertainty can correlate
spectra across integrations: a subsequent sampling calculation must use the same
coefficient draw for all spectra. Training/science data overlap adds dependence.
No automatic uncertainty propagation or hyperparameter posterior sampling is
claimed by this release.

`save(path)` and `SharedProfile.load(path)` use NPZ arrays plus JSON, with pickle
disabled on load and a checked numerical-model fingerprint. Diagnostics include
hyperparameter trials, failures/convergence, bounds, training selection, residual
summaries, timing components, and estimated working memory. Coefficient arrays
are read-only copies; evaluate/extract never invoke the fitting engine.

## Batching and Ray

`execution='serial'` requires no Ray installation. `execution='ray', n_workers=4`
uses resident actors with cropped training data. Contiguous integration batches
(`batch_size`, default maximum 8, capped to distribute small training sets) are
assigned round-robin. Geometry is cached within a
per-worker budget (`geometry_cache_mb`, default 64); dispersion work is chunked
(`column_batch_size`, default 32). No full pixel-by-coefficient matrix is formed.

All actors evaluate the **same** shared coefficients. They return likelihood,
gradient, and curvature contributions. The driver adds the prior once, reduces
in actor order, and performs the shared solve. Basis arrays are placed in the
object store once per basis change. A fixed worker count/partition is
deterministic; changing the partition can introduce floating-point differences.

Actor CPU usage is explicitly one CPU with common BLAS thread variables set to
one. Inner integration work is parallelized; optimizer starts/finite differences
are sequential to avoid nested resource oversubscription. If Ray is initialized
by the caller, it is reused and left running. Otherwise a local instance is started
and stopped by the fit. Training actors are cleaned up on success or fit failure.
Install the library and matching dependencies on every cluster worker.

Train on representative integrations spanning the trace positions and observing
conditions, then check residuals on held-out integrations before extracting the
whole visit. Increase the training selection and basis resolution to check
stability of the result. Thousands of science integrations do not require
thousands of training integrations when the intrinsic profile is stable.
The current Stage 2 extraction loop streams integrations serially; Ray
parallelizes training evaluations, not that inexpensive extraction loop.

Dense coefficient curvature/factorizations remain a scaling limitation. One
3200-by-3200 float64 matrix is about 82 MB, and workers can each return one.
The preflight estimate includes cropped data, several matrices, worker geometry
budgets, and copies for Ray. It is a conservative planning check, not a strict
operating-system memory limiter; driver/framework overhead and external input
arrays also occupy memory. Fine spatial sampling can be expensive even when
the number of integrations is small. Full GP kernels and sparse/banded
curvature solvers are not implemented in this milestone.

## Stage 2

```python
result = jwst.stage2(
    inputs, optimal_extraction=True, extraction_backend='python',
    profile_method='gp', gp_options={'n_inducing':16},
    aperture_radius=7, extraction_options={'polynomial_spacing':1.},
    single_trace_extraction=False,
    shared_profile='train',
    shared_profile_options={'training_indices':[0,10,20,30],
                            'execution':'ray', 'n_workers':4},
)
model = result['spectra']['shared_profile']
```

Alternatively pass `shared_profile=model` to reuse an existing fit. Additional
training options and replacement GP options are then rejected; omitted aperture
and spacing inherit the model's geometry, and its profile method is used.
`store_profiles=True` may be supplied through
`shared_profile_options` to retain the entire evaluated P_t cube; otherwise `Ps`
is None and `P` is a reference rendering at the median trace. Default extraction
streams model evaluations and does not retain that cube. The resulting model is
also saved beside the spectrum cache as `shared_profile_*.npz`.

New shared-profile cache identities include requested training configuration or
the supplied model fingerprint, input data/error/trace content, and relevant
preprocessing/trace-sharing flags. Existing cache identities are unchanged.
This does not repair the broader legacy trace-cache provenance scheme.

Original invalid/DQ pixels remain excluded in this new mode even when legacy
Stage 2 preprocessing fills their displayed values. This mask is reconstructed
per integration without retaining another full mask cube. Additional masks or
backgrounds in `shared_profile_options` apply to both training and extraction;
their images must match Stage 2's selected detector-column region.

Use appropriate supplied variances after preprocessing: existing Stage 2
background and 1/f corrections may introduce covariance absent from diagonal
error arrays. This feature does not recalibrate those errors. RON/gain fitting,
correlated ramp/group likelihoods, automatic temporal profile changes, and real
instrument validation remain later milestones.
