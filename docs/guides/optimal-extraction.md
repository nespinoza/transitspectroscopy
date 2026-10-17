# Optimal extraction

`transitspectroscopy` implements the optimal-extraction algorithm of
[Marsh (1989)](https://ui.adsabs.harvard.edu/abs/1989PASP..101.1032M): a spatial profile
`P` is fitted to the data, and each column's flux is the profile-weighted, inverse-variance
weighted sum of its pixels.

## Polynomial profiles (default)

The default profile model follows Marsh: the spatial profile is described by a set of
components spaced `polynomial_spacing` pixels apart along the cross-dispersion direction,
each varying smoothly along the dispersion direction as a polynomial with
`polynomial_order` **terms** (maximum degree `polynomial_order - 1`).

```python
from transitspectroscopy import spectroscopy

spectrum, P = spectroscopy.getOptimalSpectrum(
    data, centroids, aperture_radius=7, ron=5., gain=1., nsigma=8.,
    polynomial_spacing=1., polynomial_order=3,
    data_variance=variance, return_P=True,
)
column, flux, inverse_variance = spectrum
```

`centroids` holds one trace position per detector column. If `data_variance` is given it
is used as the pixel variance; otherwise the variance is modelled from `ron` and `gain`.
Masked pixels are marked with `-9999` in `data`. To speed up a time series, fit `P` once
(`getP`) and pass it to `getOptimalSpectrum(..., P=P)` for every integration.

## Backends

`backend='auto'` (default) uses the original C implementation if a compiled `Marsh` module
is installed, and the NumPy/SciPy implementation otherwise. `backend='python'` forces the
NumPy implementation, which reproduces the C results within tested tolerances (see
`tests/README.md`). `backend='c'` requires the C module.

## Gaussian-process profiles (experimental)

`profile_method='gp'` replaces the polynomials along the dispersion direction with a
Gaussian process with a Matérn kernel and an inducing-point approximation:

```python
spectrum = spectroscopy.getOptimalSpectrum(
    data, centroids, 7, 5., 1., 8., 1., 3, data_variance=variance,
    profile_method='gp',
    gp_options={'kernel': 'matern32', 'length_scale': 20., 'n_inducing': 16},
)
```

Hyperparameters are fixed by the caller. Negative predictions are clipped and each column is
normalized. Only complete, valid aperture columns train the profile. The returned errors are
conditional on the fitted profile: they exclude the GP profile uncertainty and
cross-column covariance. GP selection is always explicit.

## In `ts.jwst.stage2`

`stage2(..., optimal_extraction=True)` accepts `extraction_backend`, `profile_method`,
`gp_options` and `extraction_options` (`polynomial_spacing`, `polynomial_order`, `nsigma`).
Non-default extraction settings are cached under distinct file names.

For a profile trained once and reused across a whole time series as the trace moves, see
{doc}`shared-profile`. A worked example comparing box, polynomial and GP extraction is in
{doc}`../tutorials/optimal-extraction`.
