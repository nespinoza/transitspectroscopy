# TransitSpectroscopy 🪐 --- a package for all your transit spectroscopy needs
-------------------------------------------------------------------------------

`transitspectroscopy` is a package containing algorithms and wrappers useful for performing transit (transmission and emission) spectroscopy of exoplanetary systems.

**Author**: Nestor Espinoza (nespinoza@stsci.edu)

## Statement of need
I might fill this in later, but basically: I needed a package to make available all the tools useful for transit spectroscopy: from tracing, spectral extraction to actual lightcurve fitting of wavelength-dependant transits.

## Installation

### Full pipeline environment (recommended)

Use [environment.yml](environment.yml) from this checkout on Linux or macOS:

```sh
conda env create -f environment.yml
conda activate transitspectroscopy

# Keep an existing CRDS_PATH; otherwise use a cache in your home directory.
export CRDS_PATH="${CRDS_PATH:-$HOME/crds_cache}"
if [ ! -d "$CRDS_PATH" ]; then
    mkdir -p "$CRDS_PATH"
fi
# Restore this path automatically on future environment activations.
conda env config vars set -n transitspectroscopy CRDS_PATH="$CRDS_PATH"

python -m pip install --no-build-isolation --no-deps .
python -m pip check
```

Run these commands from the repository root. The final installation step installs
this checkout, including the new Python/shared-profile extraction APIs. Repeat it
after updating the source. The YAML contains dependencies rather than a local
absolute path or an older PyPI release of this library. A recent Conda or
Miniforge installation is recommended; `--solver libmamba` can accelerate solving
when available.

**Python 3.12 is the recommended target for the full pipeline.** It supports
the current JWST stack and the compiled transit-fitting dependencies, while
providing a longer support lifetime than the previous Python 3.11 baseline.
The package's historical `python_requires='>=3.0'` is not a tested compatibility
matrix. See the [Python support schedule](https://devguide.python.org/versions/).

The environment uses **JWST 3.0.0**, the latest stable release verified on
2026-09-15, with stdatamodels 6.0, CRDS 14, NumPy 2.2.6, SciPy 1.17.1,
Astropy 7.2.2, juliet 2.2.10 and Ray 2.58.0. The JWST pin makes environment
creation repeatable; it does not automatically track future releases. See the
[JWST installation guidance](https://jwst-pipeline.readthedocs.io/en/stable/getting_started/install.html)
and [JWST release metadata](https://pypi.org/project/jwst/3.0.0/).

The code accepts modern ramp models without `ERR` as well as older ramps with
errors. `load.ramps_err` remains an array for compatibility, with zero placeholders
where ramp errors are absent. Check `load.ramps_err_available` (one boolean per
integration) before using it: missing errors are **not** zero measurement noise.
No artificial ramp uncertainties are inserted into JWST models. The CDS routine
uses signal differences, and fitted rate products retain their own `err` arrays.
Supply appropriate variances when extracting spectra from individual groups/CDS;
this upgrade does not infer correlated ramp noise.

To upgrade an existing environment, run the following, then repeat the local
installation and `pip check` commands above:

```sh
conda env update -n transitspectroscopy -f environment.yml
```

Restart Python/Jupyter processes after upgrading. Rebuild any separately
installed binary extensions when changing Python or NumPy versions.

Included capabilities cover JWST calibration and WCS, spectral extraction,
MAST downloads, juliet light-curve fitting, Ray workers, plotting, JupyterLab and
the regression tests. Conda supplies GSL, a C compiler, and the native MultiNest
library **and** its Python wrapper: this matters because this repository's fitting
functions default to `sampler='multinest'`. Dynesty 2.1.5, emcee and juliet's
UltraNest dependency provide alternative samplers. The Dynesty major version is
held at 2 to retain the API used by this juliet release.
Use short output-folder paths with MultiNest: its native 3.10 library has a
100-character filename limit, including filenames added by juliet.

The environment was installed and validated on macOS arm64 with Python 3.12.14.
Validation includes real JWST synthetic ramp fitting, both NumPy versions of
CCF, archived/live Marsh comparisons, real Ray workers, and noisy transit fits
with MultiNest and Dynesty. To reproduce the main checks after installation:

```sh
python -m pytest tests -q
python tests/check_environment.py
```

The normal JWST tests use local synthetic references and do not download
observations. Actual instrument calibration and WCS still require the relevant
CRDS references; a complete science-observation reduction was not part of this
validation. See [tests/README.md](tests/README.md) for additional checks.

### Calibration reference files and notebooks

The installation commands above configure the CRDS cache, preserving any
nonempty `CRDS_PATH` already set in your shell or Conda environment. If it is
unset or empty, they select `$HOME/crds_cache`. They create the selected folder
only if it does not already exist, then save the path in Conda for subsequent
activations. An existing cache and its contents are left untouched.

For an environment installed previously, run this once:

```sh
conda activate transitspectroscopy
export CRDS_PATH="${CRDS_PATH:-$HOME/crds_cache}"
if [ ! -d "$CRDS_PATH" ]; then
    mkdir -p "$CRDS_PATH"
fi
conda env config vars set -n transitspectroscopy CRDS_PATH="$CRDS_PATH"
```

The export makes the path available immediately; Conda restores it on future
activations. The conditional setup runs in the shell so `$HOME` is expanded for
the installing user instead of storing a literal shell expression in the YAML.
The environment created during earlier validation already uses `~/.cache/crds`;
these commands preserve that setting as well.

The YAML sets `CRDS_SERVER_URL=https://jwst-crds.stsci.edu` on activation.
Reference files are downloaded as needed and are not bundled with the
environment. Choose a cache location with sufficient disk space for your
instrument modes. For repeatable science reductions, record the JWST version
and the compatible `CRDS_CONTEXT` used alongside your outputs; the environment
file alone does not freeze calibration references. See the
[CRDS setup instructions](https://jwst-pipeline.readthedocs.io/en/stable/getting_started/install.html#calibration-references-data-system-crds-setup).
MAST access to proprietary observations additionally requires your own login.

Start notebooks with `jupyter lab` after activating this environment. To expose
its kernel to a Jupyter installation elsewhere, run:

```sh
python -m ipykernel install --user --name transitspectroscopy --display-name "Python (transitspectroscopy)"
```

### Other installations and legacy C comparisons

The published release can also be installed with
`python -m pip install transitspectroscopy`; it may differ from this checkout.

The Python extraction and correlation backends require NumPy/SciPy and do not
require GSL or a C compiler. The CCF extension is built when possible; Python is
used when it is unavailable. Existing installations with a separately built Marsh
extension continue to use it with `backend='auto'`.

Fitting, parallel execution and MAST downloads load their optional dependencies
when used. For this checkout, install the corresponding extras with
`pip install '.[fitting,parallel,download]'`. The base installation retains
its existing JWST dependency declaration.

## Python optimal extraction

Existing calls retain polynomial profile fitting. To select a backend explicitly:

```python
from transitspectroscopy import spectroscopy

spectrum, P = spectroscopy.getOptimalSpectrum(
    data, centroids, aperture_radius=7, ron=5., gain=1., nsigma=8.,
    polynomial_spacing=1., polynomial_order=3,
    data_variance=variance, backend='python', return_P=True,
)
# spectrum rows: detector column, flux, INVERSE variance
```

An experimental GP profile model is available by adding
`profile_method='gp'` and, optionally,
`gp_options={'kernel': 'matern32', 'length_scale': 20., 'n_inducing': 16}`.
It fits spatial components jointly along dispersion using Marsh's pixel geometry.
Hyperparameters are fixed by the caller; it uses an inducing-point approximation,
clips negative predictions, and normalizes the profile. Returned extraction
errors remain conditional on the fitted profile and exclude GP profile
uncertainty and cross-column covariance. GP selection is always explicit.

`jwst.stage2` accepts `extraction_backend`, `profile_method`, `gp_options`, and
`extraction_options` (spacing, polynomial term count, rejection sigma). Explicit
extraction settings use distinct spectrum cache names; legacy default names remain
unchanged. This is not general cache provenance validation.

See [tests/README.md](tests/README.md) for the noisy comparison figures, archived
C results, test instructions and optional GSL-based reference build. Implementation
details and limitations are recorded in [CODEX_CHANGELOG.md](CODEX_CHANGELOG.md).

The unchanged Marsh C reference requires a separate NumPy 1 environment; it is
not needed for normal pipeline operation. [environment-c-reference.yml](environment-c-reference.yml)
provides the compiler, GSL and numerical dependencies for these comparisons:

```sh
conda env create -f environment-c-reference.yml
conda activate transitspectroscopy-c-reference
python tests/build_c_reference.py --gsl-prefix "$CONDA_PREFIX" --output /tmp/transitspectroscopy-c-reference
PYTHONPATH=/tmp/transitspectroscopy-c-reference python -m pytest tests -q
conda activate transitspectroscopy
```

Keep that reference directory off `PYTHONPATH` in the NumPy 2 pipeline environment.
CCF itself supports both NumPy ABIs when compiled against the environment's
headers; its numerical calculation is unchanged.

The standalone Marsh reference is opt-in; normal extraction can use the Python
backend. Run `python -m pytest tests -q` for the standard suite. Real Ray worker
checks additionally use `TRANSITSPECTROSCOPY_TEST_RAY=1` (see the test README).
The YAML pins the key compatibility dependencies but is not a complete lockfile;
archive `conda env export` and `python -m pip freeze` with a scientific analysis
if exact installed dependency versions are needed.

## Shared intrinsic profiles: train once, extract many

The new opt-in shared model fits one intrinsic spatial profile jointly across
selected independent integrations. It evaluates that model through each trace's
pixel overlaps, so detector profiles follow trace motion without retraining.
Each integration keeps its own freely varying spectrum. Both polynomial and GP
models support Gaussian coefficient priors and optional hyperparameter fitting.

```python
from transitspectroscopy import fit_shared_profile, SharedProfile

model = fit_shared_profile(
    images, variances, traces,                 # (time,row,column), (time,column)
    training_indices=[0, 10, 20, 30],
    aperture_radius=7, spacing=1.,
    profile_method='gp', gp_options={'n_inducing':16},
    optimize_hyperparameters=True,
    execution='serial',                       # or 'ray', n_workers=4
)
model.save('intrinsic_profile.npz')
model = SharedProfile.load('intrinsic_profile.npz')
P_t = model.evaluate(traces[t])                # geometry evaluation only
spectrum = model.extract(images[t], traces[t], variances[t])
# spectrum rows: column, flux, INVERSE variance conditional on the profile
```

This model fits positive, normalized log-intensity profiles on native detector
pixels. It uses a Laplace approximation over the shared coefficients conditional
on optimized training fluxes. Its GP amplitude and polynomial prior differ from
the legacy models above. RON/gain fitting and correlated ramp groups are not
supported in this milestone. Supplied variances must include the applicable
noise and background uncertainty.

Ray remains optional. Training reads only selected integrations and retains
cropped aperture data; a memory budget guards dense curvature allocations.
See [SHARED_PROFILE.md](SHARED_PROFILE.md) for geometry conventions, Stage 2,
optimization, parallel execution, uncertainty limitations and reproducibility.
Noisy simulations, a comparison figure, and measured benchmarks are in
[tests/shared_profile_comparison](tests/shared_profile_comparison).

## Licence and attribution

Read the `LICENCE` file for licencing details on how to use the code.

## Citation

You can cite this software via its DOI: 10.5281/zenodo.6960923 (see https://zenodo.org/record/6960924#.YutMaezMLUI). Here's the citation snippet:

    @software{espinoza_nestor_2022_6960924,
      author       = {Espinoza, Nestor},
      title        = {TransitSpectroscopy},
      month        = aug,
      year         = 2022,
      publisher    = {Zenodo},
      version      = {0.3.11},
      doi          = {10.5281/zenodo.6960924},
      url          = {https://doi.org/10.5281/zenodo.6960924}
    }
