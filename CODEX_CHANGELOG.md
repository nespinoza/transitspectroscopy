# CODEX_CHANGELOG

## 2026-09-16 — CRDS cache setup during installation

- Added CRDS setup directly to the README installation commands, plus a setup
  snippet for existing installations. Preserve a nonempty `CRDS_PATH`; otherwise
  default to `$HOME/crds_cache`. Check whether the directory exists before
  creating it, and leave existing cache contents untouched.
- Export the selected path for immediate use and persist it with
  `conda env config vars set` for future activations. Document the setup beside
  the environment YAML instead of embedding an unexpanded shell expression or
  a machine-specific home path. The existing CRDS server URL is unchanged.

## 2026-09-15 — JWST 3 compatibility and complete Conda environments

Updated this checkout for **JWST 3.0.0**, the latest stable release verified on
2026-09-15 ([release metadata](https://pypi.org/project/jwst/3.0.0/)). The primary
environment uses NumPy 2 and modern JWST data models; it does not pin an older
pipeline to retain the removed ramp `ERR` field. Package version remains 0.4.1.

### Code and compatibility

- Import the lazy data-model alias from `stdatamodels.jwst.datamodels`, the
  maintained namespace. Existing `transitspectroscopy.jwst.datamodels` access
  and public function signatures/defaults remain available.
- `load.merge_ramps_segments()` inspects each segment's optional errors,
  preserving provided arrays even when other segments lack them. It retains
  linked science/DQ/error views and rejects malformed error shapes instead of
  masking failures with a broad exception. Absent, `None`, or empty error arrays
  are treated as unavailable. Modern models are not assigned an artificial
  `err` field.
- Preserve the existing zero-filled `ramps_err` compatibility array for missing
  errors. New `ramps_err_available` is a boolean vector indexed by integration;
  it distinguishes supplied errors from placeholders. Zero placeholders are
  **not uncertainties** and must not be used for weighting. No ramp/group noise
  model or RON/gain estimator was introduced. Rate-product `err` arrays remain
  independently populated and linked by `merge_rateints_segments()`.
- Remove the unused full-size error cube and unconditional `dm.err` read from
  `cds_stage1()`. Its CDS signal calculation and return convention are unchanged;
  input FITS models now close after their data and timestamps are copied.
- Update CCF's NumPy interface to use `PyArray_DATA` and `import_array()`.
  Its numerical equations are unchanged, and live C/Python comparisons pass
  under both NumPy 1 and NumPy 2. **Marsh.c is unchanged** and remains an optional
  scientific reference built separately with NumPy 1. Archived simulation
  results and their original source hashes were not rewritten.
- Real environment validation exposed a separate juliet adapter failure:
  the public default `starting_point={}` made juliet look up absent free
  parameters while saving `priors.dat`, before nested sampling began. Both
  serial and Ray adapters now translate an empty mapping to juliet's `None`.
  Supplied nonempty starting points and public defaults are preserved. All four
  combinations of GP/linear regressors are covered for both adapter paths.

### Environments and operating instructions

- Created/upgraded **`transitspectroscopy`** at
  `/Users/newen/anaconda3/envs/transitspectroscopy` from `environment.yml`, and
  installed this checkout with `pip install --no-build-isolation --no-deps .`.
  Validated Python 3.12.14, NumPy 2.2.6, SciPy 1.17.1, Astropy 7.2.2, JWST 3.0.0,
  stdatamodels 6.0.0, stcal 1.20.0, GWCS 1.0.3, and CRDS 14.0.0 on macOS arm64.
  Python 3.12 is the recommended full-pipeline target: current JWST support,
  working compiled fitting dependencies, and a longer support lifetime than
  the previous Python 3.11 baseline.
- The full environment supplies juliet 2.2.10, Ray 2.58.0, Astroquery,
  JupyterLab/ipykernel, plotting, test/build tools, GSL, native MultiNest 3.10,
  PyMultiNest 2.12, george 0.4.4, celerite 0.4.3, Dynesty 2.1.5 and emcee.
  Native MultiNest is included because it is the repository's default sampler.
  The installed Conda celerite build reports distribution metadata `0.0.0`, but
  its module and Conda package report 0.4.3; actual covariance solves pass.
- Created **`transitspectroscopy-c-reference`** from
  `environment-c-reference.yml`, with Python 3.11 and NumPy 1.26.4, for the
  unchanged Marsh reference. It is not a JWST runtime and contains no older
  JWST pin. Reference binaries were built under
  `/private/tmp/transitspectroscopy-jwst3-c-reference`; keep that directory off
  the main environment's `PYTHONPATH`.
- Persisted `CRDS_SERVER_URL=https://jwst-crds.stsci.edu` and the local
  `CRDS_PATH=/Users/newen/.cache/crds` for the primary environment. The portable
  YAML contains the server URL; README explains choosing/persisting a local
  cache directory and recording `CRDS_CONTEXT` with science results.
- Updated README installation, upgrade, activation, notebooks, validation and
  legacy-reference instructions. YAMLs pin key compatibility dependencies but
  are not full lockfiles; archive a resolved environment with an analysis.
  MultiNest 3.10's native 100-character filename limit was encountered during
  validation with macOS's long default temporary path. The smoke check now uses
  a short `/tmp` path; users should keep MultiNest output paths short as well.

### Validation

- New `tests/test_jwst_compatibility.py`: missing/mixed/legacy ramp errors,
  linked arrays and repeated merging, invalid error shapes, CDS ingestion
  without accessing errors, real installed JWST entry points, real FITS
  round-trip without adding an ERR field, and a real two-integration JWST ramp
  fit using local gain/read-noise reference FITS. The input slope of 50 DN/s is
  recovered to `rtol=1e-5`, and output rate uncertainties remain finite and
  positive. These tests need no network; installed-JWST checks skip when absent.
- On the upgraded stack, **76 distinct tests passed** across the standard suite,
  targeted adapter tests and opt-in Ray checks. The standard run passed 70 tests;
  both real Ray GP/polynomial comparisons passed separately, and the final
  28-test compatibility module included four additional adapter cases. The 11
  live Marsh-only cases run in the separate reference environment.
- With the freshly built NumPy 1 C references, **82 distinct tests passed**
  across the full suite (78 passed) and final expanded compatibility module
  (28 passed, including four additional cases). Five tests were skipped: three
  installed-JWST checks and two opt-in Ray cases. Numerical warnings were treated as errors
  (`-W error::RuntimeWarning`). Existing shared-profile warnings about GP length
  scale reaching a bound were retained rather than suppressed.
- `tests/check_environment.py` validates the installed package and optional
  dependencies, real george/celerite covariance solves, and noisy synthetic
  transit inference through the actual adapter with MultiNest and Dynesty.
  Injected radius ratio: 0.1; recovered medians in the validation run were
  approximately **0.09979** and **0.09981**. This is an installation smoke check,
  not a precision/bias study; sampler realizations can vary.
- Separately exercised the installed `RampFitStep.call()` with real CRDS
  configuration lookup and synthetic local gain/read-noise references.
  JWST 3.0.0 selected `jwst_1584.pmap` and recovered the expected slope.
  Only small CRDS configuration/mapping files were fetched for this check.
  `pip check` reported no broken requirements.

These checks establish the exercised interfaces and synthetic behavior; they
do not certify every observing mode or replace a reduction of a real exposure
with its instrument-specific WCS and calibration references. Existing extraction
defaults, scientific fixtures, profile inference algorithms, CDS timing
conventions, DQ policy and known unfinished features were not redesigned.

## 2026-09-13 — Unreleased: compatibility baseline and Python extraction

Implemented priorities 1 and 2 from `CODEX_IMPLEMENTATION_RECS.md`, starting from
commit `8458f6b` (package version remains `0.4.1`). This change adds a tested
Python polynomial extraction backend and an explicitly selected experimental GP
profile model. It also repairs core import and runtime failures. No release,
commit, publication, or change to the original C source files was performed.

`CODEX_REPORT.md` and `CODEX_IMPLEMENTATION_RECS.md` remain unchanged. The original
interactive scripts under `tests/` remain available. General cache manifests,
the broad `jwst.py` restructuring, and a completed atmospheric retrieval engine
are outside this implementation.

### Priority 1: compatibility and usable existing workflows

- Added `tests/legacy_api.json`, recording 66 existing function/method signatures
  and defaults from the starting commit. Tests protect existing argument order,
  defaults, exported functions, `load`, and the Ray `.remote` fitting interface.
- Deferred imports of juliet, Ray, JWST pipeline components, GWCS, pandas,
  Astroquery/MAST, Marsh and CCF until their capabilities are needed. Core
  extraction and utilities can import when these packages are unavailable.
  Missing capabilities raise focused `ImportError`s on use. The existing base
  dependency list and top-level names are retained; dependency module aliases
  are now lazy proxies. `jwstversion` is resolved lazily when requested.
- Added fitting, parallel, download and test dependency extras. CCF compilation
  is optional, allowing the Python fallback to work when compilation fails.
  GSL is needed only for a separately built legacy Marsh reference, not for
  Python extraction. Updated README installation and usage guidance.
- Fixed the parallel fitting adapter's missing return value. Both serial and
  parallel adapters now return the fit result. Tested all four combinations of
  GP and linear regressors with deterministic juliet/Ray test doubles.
- Fixed `trace_spectrum(xend=None)` traversal initialization and the fallback
  for a bad first column. A bad column uses the most recent accepted center.
- Fixed undefined interpolation references in `get_pm_ccf`.
- Fixed Stage 2's undefined median-error image when `zero_nans=False`; missing
  error values now use their own NaN mask when constructing that image.
- Fixed absent-DQ handling in `get_uniluminated_mask`.
- Fixed the three undefined references in `load.interpolate_nans`; the helper
  works on a copy and fills the requested pixels from a zero-filled median-filter
  image. This does not complete the separate unfinished `trace_spectra` method.
- Fixed suffix initialization for a step run without a preceding calibration
  call and without an explicit suffix.
- Accepted `ramp` and `ramps` as aliases in MAST product selection.
- Handled a missing instrument filter and used value equality for the MIRI
  non-dispersive mode check.
- Replaced the undefined-knots failure in `fit_spline` with an actionable
  `ValueError` when neither supported knot specification is supplied.
- Skipped empty boxes in `spill_filter`. This removes a divide-by-zero warning
  while preserving the legacy result: the previous NaN comparison also skipped
  those boxes.
- Added tests for linked segment/combined arrays, Stage 2 dictionary schemas,
  legacy cache filenames and cache reads, array ownership, and missing optional
  dependencies. Actual JWST calibration/WCS and Ray workers are not exercised;
  their orchestration is tested with controlled substitutes.

### Priority 2a: Python polynomial extraction and CCF

New private modules:

- `src/_optimal_extraction.py`: array validation, aperture/range handling,
  fractional aperture edges, Marsh spatial overlaps, profile fitting,
  normalization, iterative rejection, weighted extraction, fractional simple
  extraction, and a matched second-image extraction helper.
- `src/_ccf.py`: NumPy equivalents of Gaussian, double-Gaussian and custom-function
  correlation. Built-in profiles process lag blocks to bound memory; custom
  functions still receive the complete lag grid, preserving that interface.
- `src/_optional.py`: deferred dependency access and the lazy Ray adapter.

The existing `getP`, `getOptimalSpectrum`, `getFastSimpleSpectrum` and `get_ccf`
functions accept an additive `backend='auto'|'c'|'python'` argument. `auto` keeps
C when installed and falls back to Python when it is unavailable. Selecting C
explicitly fails clearly if the extension is missing. No Python module named
`Marsh` or `CCF` is installed to shadow the original extensions.

The polynomial implementation deliberately retains:

- local dispersion-column powers and the same weighted normal equations,
  solved with unpivoted QR (SciPy/LAPACK in Python versus GSL in C);
- the meaning of `polynomial_order` as a **number of terms**, maximum degree
  `polynomial_order - 1`;
- the integer aperture conversion/shrinkage, legacy custom-bound semantics,
  fractional edge scaling, and full detector-width output with zero-filled
  unselected columns;
- the initial variance treatment, empirical profile-fraction variance formula,
  fixed initial column sums/fractions, and variance-update/rejection order;
- nonnegative clipping and column normalization of the fitted spatial profile;
- simultaneous profile-fit rejection and one-worst-pixel-at-a-time extraction
  rejection, including the legacy `-9999` sentinel convention;
- both supplied-variance and read-noise/gain extraction paths;
- `(3, ncolumns)` results with column, flux, and **inverse variance** rows;
  `return_P`, supplied-profile reuse, `return_flat`, floating-point aperture
  returns, and independent ownership of a returned supplied profile;
- CCF lag conventions, list return values and its historical `3.142857` pi
  approximation for built-in Gaussian normalizations.

Shape/geometry/noise validation now rejects unsafe or undefined inputs before
the public wrappers pass raw pointers to C. The Python implementation raises
exceptions for unusable variances, zero-sum profile apertures or absent weighted
support rather than treating those failures as valid spectra. These guards are
not a promise to reproduce C crashes, out-of-bounds reads, or invalid NaNs.
Zero variances outside the selected aperture are allowed; explicitly masked
pixels may have zero variance.

The matched second-image helper is compared with `BObtainSpectrum`, including
cosmic rejection. It remains private; the existing public API is not expanded
with a new background-extraction result schema. The public Marsh-mode paths are
ported; the unexposed C-only alternate Horne mode is not a new Python API.

The original `Marsh.c` and `CCF.c` remain byte-for-byte unchanged. Their source
hashes are recorded with the comparison artifacts. The historical C GSL
allocation leak and direct C-entry-point safety limitations remain in that
optional reference backend. The Python solver does not allocate those GSL
objects.

### Priority 2b: explicitly selected experimental GP profiles

`getP` and `getOptimalSpectrum` accept `profile_method='gp'` and `gp_options`.
Polynomial fitting remains the default. GP fitting requires Python; an explicit
C/GP request is rejected.

The GP replaces smooth polynomial amplitudes along dispersion while retaining
Marsh's trace-dependent, pixel-integrated spatial overlap operator. All spatial
components are fitted **jointly**, because a detector pixel receives contributions
from multiple components. This is not independent GP fitting of detector rows
and does not assume a Gaussian-shaped spatial profile.

Implemented options:

| Option | Meaning | Default |
|---|---|---|
| `kernel` | Matérn covariance along dispersion | `matern32`; also `matern52` |
| `length_scale` | Correlation length in selected dispersion-column pixels | `max(10, ncolumns / 5)` |
| `amplitude` | Prior standard deviation in unnormalized component-profile units | `1` |
| `n_inducing` | Equally spaced inducing locations per spatial component | `min(16, ncolumns)` |

At least two selected columns/inducing locations are needed. Unknown options,
nonpositive scales, unsupported kernels and invalid inducing counts raise errors.
The required legacy polynomial arguments remain accepted; polynomial term count
does not set GP complexity.

The approximation uses `G = Kxz Lzz^-T u`, with a standard-normal prior on the
whitened inducing coefficients `u`. After applying the shared spatial overlap
operator, the joint posterior mean is a regularized weighted solve. This is a
low-rank GP approximation with a zero mean and caller-fixed hyperparameters,
not a dense exact GP or automatic hyperparameter optimizer. Negative predictions
are clipped and each profile column is normalized, as in polynomial mode.

**Scientific limit:** extracted inverse variances remain conditional on the fitted
profile. They do not marginalize GP/profile uncertainty, propagate cross-column
covariance, or account for correlated detector noise. The normalized training
fractions also have correlated errors that the inherited diagonal likelihood
approximates. Kernel sensitivity, a count-space likelihood, positive latent
profiles, full uncertainty propagation and large real-data recovery studies
remain future work. The GP is suitable for explicit experimentation, not a claim
of production validation for arbitrary JWST observations.

### Stage 2 integration and compatibility

Added `extraction_backend`, `profile_method`, `gp_options` and
`extraction_options` to `jwst.stage2`. The latter can override
`polynomial_spacing`, `polynomial_order` and `nsigma`. Existing defaults remain
unchanged, including the legacy Stage 2 0.1-pixel spacing and median-profile
strategy. Serial and Ray-dispatched paths pass the same options to the backend.

Unusable-error pixels are marked with the legacy extraction sentinel on copies
of individual integrations. This avoids zero-variance weighting while preserving
returned TSO arrays and avoiding additional full-visit data/variance copies.
For inputs on which the old backend had valid finite weights, the numerical
selection is unchanged; previously undefined zero-error weighting is now masked.

Explicit nondefault extraction settings receive a deterministic spectrum-cache
suffix based on backend, profile method, supplied GP/extraction options and
aperture. Such products add `spectra['extraction_settings']`. Default calls retain
their existing filenames and schemas. Tests check that changing GP length scale
creates a separate cached spectrum. Trace caching remains unchanged.

This narrow cache separation is necessary to keep GP and polynomial experiments
distinct. It does **not** implement priority 3's general input/version/reference
provenance validation; legacy caches can still be stale when unrelated settings
or inputs change.

### Noisy simulations and measured C/Python differences

`tests/simulated_spectra.py` generates independent pixel-integrated profiles with
seeded Poisson photon noise and Gaussian read noise. `tests/compare_extraction.py`
runs the actual C backend, Python polynomial backend and Python GP backend on
identical noisy arrays. It fails if C is unavailable rather than substituting a
Python reference.

Saved under `tests/extraction_comparison/`:

- `smooth_noisy.png`, `.csv`, `.npz`: curved trace and constant-width profile,
  seed 123;
- `varying_profile_cosmics.png`, `.csv`, `.npz`: changing width, asymmetric wings,
  positive/negative impulses, seed 456;
- `summary.json`: configuration, versions, original C source hashes, timing and
  numerical comparison/recovery metrics.

The NPZ files include noisy input, supplied variance, true trace, true profile,
true flux, simple-extraction control, C/Python/GP spectra and fitted profiles.
The figures expose very small differences on separate axes instead of relying
on overlapping spectrum curves. The original minimal interactive simulation is
preserved; these additional tests and artifacts live in the same `tests` folder.

Measured with aperture 7, spacing 1, three polynomial terms and 8-sigma rejection:

| Simulation | Maximum absolute flux difference, Python minus C | Maximum difference in C error units | Maximum profile-fraction difference |
|---|---:|---:|---:|
| Smooth noisy | `6.06e-10` | `1.07e-11 sigma` | `2.13e-12` |
| Varying profile + impulses | `1.12e-9` | `1.96e-11 sigma` | `1.77e-12` |

For the smooth case, normalized residual scatter is about 1.06 for both polynomial
implementations and the GP. In the deliberately changing-profile case, the
three-term polynomial has normalized residual mean/scatter about -1.82/2.49;
the GP gives about -0.09/1.07. Thus C/Python agreement reproduces the polynomial
model's limitations as well as its successful cases. This specific example
supports investigating GP flexibility, not universal superiority over a tuned
polynomial model.

A separate live comparison covers the 0.1-pixel spacing used by Stage 2 over
441 columns. A diagnostic run found profile differences around `1.3e-6`; the
test requires flux differences below `0.001` of the C extraction uncertainty.
Legacy raw-power normal equations are less well conditioned in this regime.
The implementation promises tested tolerances, not bitwise agreement across
linear algebra libraries or reliable solutions to every rank-deficient model.

### Validation performed

- Built **unchanged** C Marsh and CCF using Python 3.11.4, NumPy 1.26.4 and GSL
  2.8 in an isolated temporary environment on macOS arm64. SciPy 1.17.1 supplies
  Python linear algebra. The user's installed environment was not altered.
- Full suite with live C available: **55 passed**.
- Python-only suite with C unavailable: **41 passed, 14 explicitly skipped**;
  archived C-fixture comparisons still run. This run also treats RuntimeWarnings
  as errors.
- Tests include supplied and detector-model variance, outlier injection,
  subpixel/reversed traces, custom bounds, aperture shrinkage, fine spacing,
  profile reuse/ownership, masked pixels, all CCF modes and second-image weights.
- Truth-based tests include eight independent noisy realizations per profile
  method, conditional-error coverage, known-profile cosmic recovery, and transit
  and null injections over 16 independently noisy integrations with a
  transit-correlated trace shift. These are initial scientific regression tests,
  not a full instrument-validation campaign.
- Tested Stage 2 simple, Python polynomial and GP paths, both NaN policies,
  serial and simulated parallel dispatch, output keys, filenames, GP cache
  separation and cache reuse. External WCS/calibration are deterministic test
  doubles; real JWST data and actual distributed Ray workers were not run.
- Built the Python package to a temporary directory and smoke-tested import and
  extraction from that built package without the optional pipeline/fitting stack.
- New numerical/helper/test modules pass a focused pyflakes check. Existing
  unused imports and star-import diagnostics remain in legacy modules.
- `git diff --check` passes. The comparison plots were generated and visually
  inspected. Reproduction commands are in `tests/README.md`.

### Deliberately retained behavior and remaining work

Characterization tests preserve, but do not endorse, the existing jump-DQ
overwrite, CDS seconds/days timestamp bug, simple-background input mutation,
and linked datamodel array ownership. The numerical quantile/binning quirks,
unfinished retrieval scaffold, experimental CDS orchestration and unfinished
object tracing workflow have not been broadly redesigned or completed.

Further work should include a separately reviewed scientific-defect release,
real observation replay, larger transit/feature/null ensembles, GP hyperparameter
sensitivity and uncertainty propagation, and resource/performance measurements
on full visits. The C reference is intentionally retained so future numerical
changes remain directly comparable.

## Shared intrinsic profiles and hyperparameter inference — 2026-09-14

Implemented the agreed first milestone: one intrinsic profile trained jointly on
selected independent integrations, reusable through integration-specific trace
geometry; optional GP or polynomial-prior hyperparameter inference; supplied
fixed variances; batched serial and optional Ray execution. This is an opt-in
statistical model. Noise calibration and correlated ramp/group inference remain
subsequent milestones.

### Model, optimization, and API

- Added `src/shared_profile.py`, exporting `fit_shared_profile` and `SharedProfile`
  at package level. Polynomial Legendre functions and inducing-point Matern-3/2
  or Matern-5/2 functions describe log intrinsic intensities relative to a fixed
  Gaussian mean. A spatial Helmert contrast removes the redundant common scale.
  Positive intensities are integrated through the existing Q geometry and
  normalized inside the forward model.
- Intrinsic basis support extends at least one pixel beyond the aperture to
  model complete boundary-pixel footprints. The data aperture itself stays
  fixed. A narrow-aperture, fine-spacing regression checks that trace motion
  does not introduce the boundary bias found during validation.
- One coefficient vector is shared by all training integrations. Each integration
  and wavelength has its own freely varying flux, eliminated analytically for
  each coefficient evaluation. Independent spectra therefore retain transits
  and spectral variability. Known trace motion changes Q, not the fitted
  coefficients. A trained model caches its intrinsic log intensities.
- Training uses original detector measurements and supplied variances, with a
  fixed mask. Invalid data/errors and -9999 sentinels are excluded. No changing
  sigma-clipping mask enters hyperparameter optimization. Explicit background
  means are supported; supplied variances must include background uncertainty.
- Coefficients use damped variable-projection Gauss-Newton updates with line
  search and convergence checks. This replaced an initially tested L-BFGS inner
  solve after full-size images exposed slow convergence. The outer optimizer
  remains bounded L-BFGS-B over log amplitude and log length scale (GP), or log
  prior amplitude (polynomial), using finite differences and deterministic
  multiple starts. Warm starts are projected into each new basis.
- Spatial Jacobians are projected before curvature assembly, and the Helmert
  diagonal transform uses analytic prefix sums. These remove unnecessary cubic
  spatial transforms. Hyperparameter caches retain objective scalars rather
  than one dense factorization per trial or restart.
- A 100-integration Ray benchmark exposed an overly strict convergence cutoff
  at the precision limit of the large likelihood sum. The squared Gauss-Newton
  step threshold retains its 1e-8 baseline with a rounding floor capped at 1e-7,
  equivalent to a step below 0.00032 in its local posterior metric. Small-data
  fits retain the tighter threshold needed for stable evidence gradients.
  Stopping reasons are recorded, and genuine line-search
  failures report the scaled gradient, step and decrement. The initial failed
  benchmark is retained separately for provenance; benchmark scripts now record
  failed fits as well as timeouts instead of leaving an old success file behind.
- The marginal-likelihood objective uses a Laplace approximation over shared
  coefficients **conditional on optimized training fluxes**. It includes the
  Gaussian pixel normalization, one coefficient prior, and the determinant of
  the exact conditional coefficient Hessian. This is not full marginalization
  over spectra/hyperparameters. Failed inner trials are recorded and rejected;
  only converged outer endpoints are eligible. Total failure raises.
- `evaluate(trace)` renders P_t without fitting. `extract(...)` returns the
  familiar [column, flux, inverse variance] layout. `diagnose(...)` reports
  held-out chi-square by column. NPZ/JSON save/load avoids executable pickle
  payloads and checks a numerical-model fingerprint. Coefficient/model fields
  are frozen; local conditional covariance is retained by default.
- `getOptimalSpectrum(P=model, data_variance=...)` dispatches to the model's
  native-pixel extraction, checking geometry and backend. Existing ndarray-P
  calls retain their historical fractional-edge behavior. For consistency with
  the new likelihood, use the model interface instead of passing its evaluated
  array back through legacy edge scaling. Existing C sources are unchanged.

### Parallel execution, memory, and Stage 2

- Training accepts cubes, memory maps, or indexable image sequences. It reads
  only selected integrations and retains aperture crops, not a stacked full
  detector cube. Cropped training pixels remain resident; this is bounded
  batching, not unlimited out-of-core training.
- Serial and Ray paths share the same batch evaluator. Actors retain cropped
  data; Q caching is bounded; columns are chunked; basis arrays are reused in
  Ray's object store. The driver reduces likelihood/gradient/curvature
  contributions in a deterministic actor order and adds the shared prior once.
  Actors do not train independent profiles.
- Worker count and BLAS threads are controlled, and small training sets are
  spread across workers. Training actors are cleaned up on success/failure;
  a caller-owned Ray instance is left running. The optional parallel extra now
  installs `ray[default]`, including the runtime-environment dependencies used
  by actors. No Ray import is needed for serial extraction/training.
- A configurable memory preflight accounts for cropped data, multiple dense
  curvature matrices, worker copies, and Q caches. It fails before costly
  matrix allocation rather than silently coarsening the scientific model.
  Dense coefficient factorization remains a limitation for fine spatial grids.
- Stage 2 accepts `shared_profile='train'` or a previously trained model, plus
  `shared_profile_options`. Per-integration trace evaluation uses
  `single_trace_extraction=False`. It streams extraction and stores the model
  and reference P; the P_t cube is optional (`store_profiles=True`, otherwise
  `Ps=None` in this new mode). Existing Stage 2 output behavior is unchanged
  when shared-profile extraction is not selected.
- Shared-mode spectrum caches include the model version, requested training
  settings or model fingerprint, input/error/trace content, and preprocessing
  flags. NumPy configuration arrays are content-hashed rather than expanded
  into large JSON lists. A portable model NPZ accompanies new spectrum products.
  General legacy trace-cache provenance is not changed.
- Original invalid/DQ pixels remain excluded even when legacy Stage 2
  preprocessing fills their displayed values. The validity mask is reconstructed
  per integration. User masks and backgrounds apply to training and extraction.

### Simulations and verification

Added tests for direct Gaussian evidence, numerical coefficient derivatives and
conditional Hessians, independent noisy recovery, shifted traces, freely varying
spectra, held-out transit recovery, fixed masks, selected-only image access,
serialization, no-refitting reuse, polynomial prior and GP hyperparameter
optimization, profile instability, memory/input failures, Stage 2 integration,
and cache reuse. Real Ray worker comparison is opt-in during test execution.

`tests/compare_shared_profile.py` saves the simulated detector data, truth,
evaluated profiles, extracted spectra, light curves, model files, metrics and a
four-panel figure in `tests/shared_profile_comparison/`. Seven selected
integrations train one intrinsic profile. Sixteen integrations have independently
simulated Poisson/read noise, trace motion, and an injected 1% transit. The
simulation's Gaussian-CDF pixel integration is independent of the fitted
triangular spatial basis. The deliberately challenging wavelength-dependent
shape exposes three-term polynomial misspecification; it is not a general
claim that GP extraction is superior.

The saved comparison uses injected aperture flux as extraction truth, retaining
full-source flux separately. The extraction does not infer an aperture correction.
GP held-out normalized residual mean/std are -0.021/0.995; the recovered transit
depth is 0.01052 for an injected 0.01. The polynomial shape mismatch produces
mean/std of -3.56/3.68 and a depth of 0.01056. Training, including hyperparameter
optimization with two starts on seven 96-column integrations, took 33.3 seconds
for the GP and 11.2 seconds for the polynomial in the final artifact run. Stable-profile
residual chi-square ranges from 1.01 to 1.17; a deliberately broadened held-out
profile produces 3.34 to 3.72. These are one seeded realization, with errors
conditional on the fitted profile, not an ensemble coverage assessment.

Final regression validation: **74 tests passed**, including the unchanged C
reference comparisons and real Ray actors for fixed polynomial and optimized GP
fits. The suite treats RuntimeWarnings as errors. Four expected UserWarnings
surface GP length-scale boundary solutions. The new numerical module and tests
also pass Pyflakes; `git diff --check` passes. Validation used an isolated Python
3.11 environment with NumPy 1.26.4, SciPy 1.17.1 and Ray 2.58.0; the user's main
Python environment was not modified.

`tests/benchmark_shared_profile.py` measures full nonlinear coefficient training,
curvature, and extraction on lazily generated 2048-by-256 images. Each benchmark
configuration runs in a separate process, records timing components and driver
peak RSS, and distinguishes fixed hyperparameters from optimized ones. The
recorded measurements replace the earlier conversation's extrapolations only
for the configurations actually run; no thousand-integration timing or universal
Ray speedup is claimed.

Final default-grid benchmark results (2048 columns by 256 rows, radius 7,
spacing 1 pixel, GP with 16 inducing points, fixed hyperparameters, one BLAS
thread per process) on the available macOS ARM machine:

| Training integrations | Execution | Training seconds | Extraction ms/integration | Driver peak RSS MB |
| ---: | --- | ---: | ---: | ---: |
| 1 | Serial | 0.98 | 8.5 | 206 |
| 10 | Serial | 7.76 | 8.9 | 270 |
| 100 | Serial | 130.16 | 8.7 | 431 |
| 100 | Ray, 4 workers | 146.57 | 12.5 | 416 |

These are single wall-time measurements, including preparation and Ray startup;
driver RSS excludes worker processes. Both 100-integration fits took 19 inner
iterations. Their evidence objectives differ by 2.24e-8, maximum per-integration
chi-square difference is 1.46e-11, and median extracted fluxes agree to machine
precision. Ray did **not** accelerate this final run; communication, reduction,
process overhead and available hardware must be assessed for each deployment.
Default-grid results correspond to implementation SHA-256
`2bb54ff34638d27ea7dcbc153609f7876aec2099a0c550064d40476ae477e55e`.

With hyperparameter optimization enabled, one full-size default-grid integration
converged in 20.02 seconds using one optimizer start, 7 outer iterations and 49
inner/evidence evaluations. Neither fitted hyperparameter was at a bound. This
is a measurement for one stable synthetic profile, not a scaling factor for
arbitrary visits. The opt-in fitter's default uses two optimizer starts.

At 0.1-pixel spatial spacing, even one full-size integration with only four
inducing points did not finish coefficient training within the explicit
60-second budget (driver peak RSS 456 MB). Its report records a timeout, not a
converged model or a 60-second training estimate. Dense curvature remains a
practical obstacle for fine grids; these measurements do not establish that
training hundreds or thousands of integrations at that resolution is practical.

### Limits and next milestones

The shared model assumes a stable intrinsic shape with known trace geometry and
a consistent dispersion grid. It does not fit trace positions, automatically
divide observations into stability intervals, or model time-dependent widths.
GP inducing-point and polynomial-prior sensitivity still require scientific
assessment. Boundary solutions are surfaced, not hidden.

The stored coefficient covariance conditions on training fluxes and fitted
hyperparameters. Extracted inverse variances condition on P; shared-profile
uncertainty and its correlations across time are not automatically propagated.
Real instrument replay and broader injection/recovery ensembles remain needed.

RON/gain fitting, effective noise-component calibration, temporal covariance of
group/ramp measurements, jump/pedestal inference, full GP kernels, sparse/banded
curvature solvers, and hyperparameter posterior sampling are not implemented in
this milestone. Existing Stage 2 preprocessing can introduce covariance absent
from its supplied diagonal errors; the new model does not recalibrate those
errors. See `SHARED_PROFILE.md` and `tests/README.md` for use and reproduction.
