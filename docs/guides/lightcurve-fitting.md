# Light-curve fitting

`ts.transitfitting.fit_lightcurves` fits many light curves (for example, one per wavelength
bin) with [juliet](https://juliet.readthedocs.io). It needs the `fitting` extra
(`pip install "transitspectroscopy[fitting]"`); the default `multinest` sampler also needs the
MultiNest library, which the Conda environment provides.

```python
data = {'bin1': {'times': t, 'flux': f1, 'error': e1},
        'bin2': {'times': t, 'flux': f2, 'error': e2}}
priors = {'bin1': priors_bin1, 'bin2': priors_bin2}   # juliet prior dictionaries

results = ts.transitfitting.fit_lightcurves(data, priors, sampler='dynesty', nthreads=2)
```

Each data dictionary must contain `times`, `flux` and `error`, and may add
`GP_external_parameters` and/or `linear_external_parameters` (arrays of shape
`(len(times), n_regressors)`). With `nthreads`, fits run in parallel with Ray. Each fit is
written to an output folder named after its key.

The light curves are passed to juliet under the instrument name `SOSS`, whatever the actual
instrument, so instrument-specific juliet priors must use that suffix (e.g. `mflux_SOSS`,
`sigma_w_SOSS`, `q1_SOSS`).
