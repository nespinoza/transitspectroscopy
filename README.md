# TransitSpectroscopy 🪐 --- a package for all your transit spectroscopy needs
-------------------------------------------------------------------------------

`transitspectroscopy` is a package containing algorithms and wrappers useful for performing transit (transmission and emission) spectroscopy of exoplanetary systems.

**Author**: Nestor Espinoza (nespinoza@stsci.edu)

## Statement of need
I might fill this in later, but basically: I needed a package to make available all the tools useful for transit spectroscopy: from tracing, spectral extraction to actual lightcurve fitting of wavelength-dependant transits.

## Installation

`transitspectroscopy` requires **Python 3.13 or 3.14** and the
[JWST calibration pipeline](https://github.com/spacetelescope/jwst) (`jwst>=3.0.0`),
which is installed automatically as a dependency.

**Conda (recommended).** From the repository root, create the full environment
(JWST pipeline, juliet with MultiNest/Dynesty, Ray, JupyterLab) and install this checkout:

```sh
conda env create -f environment.yml
conda activate transitspectroscopy
python -m pip install --no-build-isolation --no-deps .
```

**pip.** Install the package, optionally with extras for light-curve fitting
(`fitting`), parallel execution (`parallel`), MAST downloads (`download`), plotting
(`plots`), tests (`test`) or everything (`all`):

```sh
python -m pip install .                 # or: python -m pip install transitspectroscopy
python -m pip install ".[fitting,parallel,download]"
```

Calibrating JWST data needs reference files from CRDS: set `CRDS_PATH` to a cache
folder and `CRDS_SERVER_URL=https://jwst-crds.stsci.edu`. See
[docs/installation.md](docs/installation.md) for CRDS setup, MultiNest notes, the
optional C extensions and the legacy C-reference environment.

## NIRISS/SOSS

NIRISS `CLEAR/GR700XD` rateints now have a three-order reduction path through
`ts.soss.reduce_soss(rateints_paths)` or `ts.jwst.stage2(...)`. It runs SOSS
background subtraction **after ramp fitting**, starts traces/wavelengths from
PASTASOSS at the measured pupil-wheel position, and saves separate spectra and
light curves for orders 1, 2 and 3. Box extraction remains the library default;
`optimal_methods=('polynomial', 'gp')` adds shared-profile Python optimal
extraction. Missing apertures and order
overlap are flagged explicitly. Optimal extraction now fits overlapping order
profiles jointly, including the long-wavelength regions; box spectra retain
their original blended-aperture interpretation.

Run the full HAT-P-14 b PID 1541 observation 1 validation with
`python validation/validate.py --mode soss --download`. See
[the SOSS guide](validation/SOSS_README.md) for raw-data processing, reference
overrides, extraction comparisons, transmission fits and saved-product reuse.
The default SOSS validation now runs all three extraction methods, including
matched-wavelength precision and transit-depth comparisons.
The [overlap-correction report](validation/SOSS_OVERLAP_REPORT.md) distinguishes
successful reduction from the remaining scientific limitations of the spectrum.

## MIRI imaging/photometry

`ts.miri.reduce_miri(...)` uses official JWST Detector1/Image2 calibration and
local fractional-aperture sums plus optimal photometry from a shared 2-D spatial
GP PSF. It preserves masks and integration times and propagates sky uncertainty.
No Eureka dependency is used.

Run both LTT 1445 A F1500W visits with
`python validation/validate.py --mode miri --download --compress-downloads`.
See the [MIRI guide](validation/MIRI_README.md) and
[executed report](validation/MIRI_VALIDATION_REPORT.md) for the 29,689
integrations, products, injection tests and scientific limits.

## Python optimal extraction

For a real NRS1 reduction with saved detector/ramp-fit plots, trace/FWHM
diagnostics, three extraction comparisons, FITS datamodels, numerical time
series and a JSON report, see [validation/README.md](validation/README.md).
The default validation downloads only the first segment of the HAT-P-14
commissioning exposure; plot limits are fixed in `validation/config.json`.

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
details and limitations are recorded in [docs/dev-notes/CODEX_CHANGELOG.md](docs/dev-notes/CODEX_CHANGELOG.md).

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

## Alternative ramp fitting

JWST ramp fitting remains the default. The correlated adjacent-read method from
[arXiv:2305.06240](https://arxiv.org/html/2305.06240v1), also used by SPARTA, can
be selected before detector calibration:

```python
dataset = ts.jwst.load(filenames, outputfolder="reduction",
    ramp_fit_method="sparta",
    ramp_fit_options={"residual_correction": False, "sigma": 5.0})
dataset.detector_calibration()
dataset.fit_ramps()
```

Enable `residual_correction` to add the empirical two-pass ramp-template
correction, using appropriate training integrations for variable-flux targets.
The alternative supports uniform, single-frame groups (`NFRAMES=1`), preserves
DN/s rates and separate variance components, and is also selectable in the
NIRSpec/SOSS/MIRI validation configurations. See [RAMP_FITTING.md](RAMP_FITTING.md)
for options, reference-noise conventions, saved diagnostics and uncertainty limits.

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
