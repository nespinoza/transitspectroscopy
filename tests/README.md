# Extraction validation and compatibility tests

Run from the repository root:

```bash
python -m pytest tests -q
```

The suite loads this checkout as `transitspectroscopy`; installation of the full
JWST pipeline is not needed. Required test dependencies are NumPy, SciPy,
Astropy, jdcal, tqdm, and pytest. Matplotlib is needed to regenerate figures.
The original interactive `spectral_extraction.py` and `ccf.py` demonstrations
remain available and are not collected as pytest tests.

## Inspect the simulations

Both cases contain Poisson photon noise plus Gaussian read noise (5 electrons,
gain 1). Spatial profiles are generated independently of the fitting basis by
integrating Gaussian mixtures over detector pixels. A curved, subpixel trace,
sloping spectrum and narrow absorption feature are included.

- [Smooth noisy profile](extraction_comparison/smooth_noisy.png): seed 123,
  constant width, curved trace.
- [Changing profile and cosmic rays](extraction_comparison/varying_profile_cosmics.png):
  seed 456, changing width, asymmetric wings, positive and negative impulses.

Each figure shows the detector image, true and recovered spectra, C/Python flux
and uncertainty differences, profile differences, and recovery residuals.
The polynomial C and Python curves overlap; the difference panels expose the
floating-point discrepancies. Colors are consistent across panels.

Matching `.csv` files contain readable spectra and differences. Matching `.npz`
files contain input data, variance, trace, injected spectrum/profile, simple
extraction, and all C/Python/GP spectra and fitted profiles. Load them with
`numpy.load(path)`; they contain no pickled objects. [summary.json](extraction_comparison/summary.json)
records seeds, parameters, numerical metrics, runtime, versions and hashes of
the unchanged C sources. Timings are single illustrative runs, not benchmarks.

The changing-profile example intentionally challenges a three-term polynomial.
Its GP improvement is evidence for this particular simulation, not a general
claim of GP superiority. The smooth-profile control shows why polynomial mode
remains the default.

## Regenerate with the actual C backend

Use a separate environment with NumPy < 2 for the unchanged legacy C sources:

```bash
python -m venv /tmp/ts-validation
/tmp/ts-validation/bin/pip install 'numpy<2' scipy astropy jdcal tqdm pytest matplotlib setuptools wheel
```

Provide an existing GSL installation prefix (containing `include`, `lib`, and
`bin/gsl-config`). The reference build compiles the repository C sources without
editing or installing them:

```bash
/tmp/ts-validation/bin/python tests/build_c_reference.py \
    --gsl-prefix /path/to/gsl --output /tmp/ts-c-reference
PYTHONPATH=/tmp/ts-c-reference /tmp/ts-validation/bin/python -m pytest tests -q
PYTHONPATH=/tmp/ts-c-reference /tmp/ts-validation/bin/python tests/compare_extraction.py
```

`compare_extraction.py` requires C and fails clearly if it is unavailable. It
never labels a Python result as C. Its default output is the checked-in
`tests/extraction_comparison` directory; use `--output /tmp/new-comparison` to
make a separate comparison without overwriting the reference fixtures.

Ordinary pytest runs without C still compare Python against the archived C
fixtures. Live C-only tests report explicit skips. Do not regenerate reference
fixtures just to make a failing test pass: first investigate and document the
numerical difference.

## What the suite establishes

- Original signatures/defaults and exported functions, taken from commit
  `8458f6b` and stored in `legacy_api.json`.
- Core import and Python fallbacks with optional packages unavailable.
- Serial/parallel fitting adapter results with test doubles for juliet/Ray.
- Trace defaults, a bad first column, CCF interpolation, masks without DQ,
  NaN replacement, suffix initialization, and linked segment arrays.
- Stage 2 output schemas, filenames, cache reuse, Python optimal extraction,
  GP dispatch, and unusable-error masking. External JWST WCS is replaced by a
  deterministic test double; these are orchestration tests, not calibration tests.
- Live and archived C/Python polynomial comparisons, including detector-model
  and supplied variances, outliers, subpixel positions, reversed traces, bounds,
  fractional apertures, fine spacing, supplied profiles and a second weighted image.
- All CCF modes, return types, and normalization conventions.
- Recovery from noisy independent simulations, conditional-error coverage,
  large impulses, simple invalid-input failures, and a small transit/null
  experiment with a transit-correlated trace shift and wavelength-varying PSF.

The main differential cases use strict absolute/relative tolerances. A separate
0.1-pixel spacing case assesses flux differences in units of the C uncertainty,
because the legacy normal equations become less well conditioned. The Python
polynomial path deliberately retains those equations rather than changing the
scientific baseline during the port.

## Deliberately characterized legacy behavior

Tests explicitly record that `tso_jumpstep` overwrites jump-pixel DQ with 4,
`get_cds` mixes seconds and days in its timestamps, and background-corrected
`getSimpleSpectrum` mutates the input image. These behaviors are known issues,
not assertions of scientific correctness. Changes to them need separately
reviewed compatibility decisions. No extraction uncertainty here includes
profile-estimation uncertainty or correlations in detector noise.

## Remaining validation before production use of the legacy GP

Real PRISM/G395/SOSS observations, broad kernel/hyperparameter sensitivity,
measured non-Gaussian PSFs, larger transit-depth ensembles, correlated detector
noise, saturated regions, performance on long visits, and propagation of profile
uncertainty remain future validation. The current GP is a fixed-hyperparameter,
low-rank approximation with a zero mean and a Gaussian profile-fraction likelihood;
normalizing the noisy training fractions induces covariance that this likelihood
approximates as diagonal, following the polynomial baseline.

## Shared intrinsic profiles

`test_shared_profile.py` tests the new positive, normalized shared model. It
checks coefficient gradients and exact conditional Hessians by finite differences,
linear evidence against a direct covariance solve, independent noisy recovery,
moving traces, held-out transit recovery, masks, source selection, serialization,
profile reuse, memory/input checks, profile-instability residuals, hyperparameter
optimization, and Stage 2 integration/cache reuse. Existing C comparisons remain
tests of the legacy polynomial port, not equivalence claims for the new model.

Real Ray processes are opt-in (the package must be installed for workers):

```sh
TRANSITSPECTROSCOPY_TEST_RAY=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m pytest tests/test_shared_profile.py -q
```

Generate the noisy shared-profile comparison with:

```sh
OPENBLAS_NUM_THREADS=1 python tests/compare_shared_profile.py
```

The saved [figure](shared_profile_comparison/comparison.png),
[metrics](shared_profile_comparison/summary.json), [CSV](shared_profile_comparison/lightcurves.csv),
and [simulation](shared_profile_comparison/simulation.npz) compare optimized GP
and polynomial shared models. Seven integrations train a single intrinsic profile;
all integrations have independently simulated Poisson/read noise, moving traces,
and freely varying spectra. This deliberately challenging wavelength-dependent
profile exposes polynomial misspecification; it is not evidence that a GP is
universally preferable. Model NPZ files retain the fitted settings/diagnostics.
Recovery uses the known simulated flux inside the selected whole-pixel aperture;
full-source truth is saved separately. The light-curve reference is simulation
truth, not an aperture correction estimated by the extraction routine. The saved
GP held-out residual mean/std are -0.021/0.995; its recovered depth is 0.01052 for
the injected 0.01. This single seeded run is not an uncertainty-coverage ensemble.

Benchmark one configuration per process, for example:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python tests/benchmark_shared_profile.py --integrations 10 --output /tmp/shared_10.json
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python tests/benchmark_shared_profile.py --integrations 10 --optimize --workers 4 --output /tmp/shared_10_ray.json
```

Default images are 2048 columns by 256 rows, aperture 7, spacing 1, GP with 16
inducing points. The first command fits coefficients at fixed hyperparameters;
`--optimize` adds hyperparameter inference. Reports separate training and
extraction time, inner objective/curvature time, convergence, and driver peak RSS
(excluding Ray workers). Synthetic images are generated lazily during preparation.
Saved `benchmark_*.json` files report measured configurations only. They do not
establish thousand-integration performance or a general Ray speedup factor.
In the final 100-integration default-grid run, serial took 130 seconds and four
Ray workers took 147 seconds, with fitted likelihoods and flux summaries agreeing
to numerical precision. One full-size integration with optimized GP
hyperparameters took 20 seconds (one optimizer start). These are distinct from
coefficient-only timings; see the measured table in [CODEX_CHANGELOG.md](../CODEX_CHANGELOG.md).
Each report records the implementation SHA-256. Training has a 300-second
wall-time limit by default (`--max-seconds`); a timeout is recorded explicitly
and is not reported as a converged fit. Numerical fit failures also receive a
distinct status and reason. This benchmark script uses Unix alarms and process
resource counters. Fine spatial sampling is substantially
more expensive than the default benchmark's one-pixel spacing.

See [SHARED_PROFILE.md](../SHARED_PROFILE.md) for the approximation, priors,
uncertainty conventions, batching limitations, and deferred noise/ramp work.
