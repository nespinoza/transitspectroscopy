# TransitSpectroscopy 🪐 --- a package for all your transit spectroscopy needs
-------------------------------------------------------------------------------

`transitspectroscopy` is a package containing algorithms and wrappers useful for performing transit (transmission and emission) spectroscopy of exoplanetary systems.

**Author**: Nestor Espinoza (nespinoza@stsci.edu)

## Statement of need
I might fill this in later, but basically: I needed a package to make available all the tools useful for transit spectroscopy: from tracing, spectral extraction to actual lightcurve fitting of wavelength-dependant transits.

## Installation
Installation is as simple as:

        python setup.py install

Or via PyPi:

        pip install transitspectroscopy

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
