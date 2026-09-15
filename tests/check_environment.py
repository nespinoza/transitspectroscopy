"""Smoke-test the installed full environment, including actual native samplers.

Run ``python tests/check_environment.py`` after installing the checkout. No
MAST/CRDS downloads are made. Outputs are confined to a temporary directory.
JWST algorithm tests live in test_jwst_compatibility.py; Ray has opt-in pytest
checks. This script deliberately imports the installed package, not _support.
"""
import importlib
from importlib.metadata import version
from pathlib import Path
import tempfile

import numpy as np


def main():
    for module in ['transitspectroscopy', 'jwst', 'stdatamodels', 'gwcs', 'crds',
                   'astroquery', 'juliet', 'pymultinest', 'dynesty', 'emcee',
                   'ultranest', 'batman', 'george', 'celerite', 'ray',
                   'jupyterlab', 'ipykernel']:
        loaded = importlib.import_module(module)
        distribution = 'batman-package' if module == 'batman' else module
        print(module, version(distribution),
              '(module '+str(getattr(loaded, '__version__', 'n/a'))+')', flush=True)

    # Imports alone do not establish that these compiled GP solvers work with
    # the installed NumPy ABI. Factor and evaluate a small covariance matrix.
    import george
    import celerite
    x = np.linspace(0., 1., 30)
    y = np.sin(x)
    gp = george.GP(george.kernels.ExpSquaredKernel(.1))
    gp.compute(x, .1)
    assert np.isfinite(gp.log_likelihood(y))
    gp = celerite.GP(celerite.terms.RealTerm(log_a=0., log_c=0.))
    gp.compute(x, .1)
    assert np.isfinite(gp.log_likelihood(y))
    print('george and celerite covariance solves passed', flush=True)

    from transitspectroscopy.transitfitting import notremote_fit_data
    import batman
    parameters = batman.TransitParams()
    parameters.t0, parameters.per, parameters.rp = 0., 1., .1
    parameters.a, parameters.inc, parameters.ecc, parameters.w = 10., 88., 0., 90.
    parameters.u, parameters.limb_dark = [.3, .2], 'quadratic'
    times = np.linspace(-.05, .05, 60)
    flux = batman.TransitModel(parameters, times).light_curve(parameters)
    flux += np.random.default_rng(922).normal(0., .0003, len(times))
    fixed = dict(P_p1=1., t0_p1=0., a_p1=10., b_p1=10*np.cos(np.deg2rad(88)),
                 ecc_p1=0., omega_p1=90., q1_SOSS=.25, q2_SOSS=.3,
                 mdilution_SOSS=1., mflux_SOSS=0., sigma_w_SOSS=0.)
    priors = {k: dict(distribution='fixed', hyperparameters=v) for k, v in fixed.items()}
    priors['p_p1'] = dict(distribution='uniform', hyperparameters=[.08, .12])
    data = dict(times=times, flux=flux, error=np.full(len(times), .0003))
    # MultiNest 3.10 has a 100-character native filename limit. macOS's usual
    # /var/folders temporary prefix can exceed it once juliet adds filenames.
    with tempfile.TemporaryDirectory(prefix='ts-fit-', dir='/tmp') as temp:
        for sampler in ['multinest', 'dynesty']:
            # Leave starting_point at its public default: this also exercises
            # juliet prior-file serialization through the real adapter.
            result = notremote_fit_data(
                data, priors, output_folder=str(Path(temp)/sampler),
                sampler=sampler, n_live_points=50, verbose=False)
            samples = result.posteriors['posterior_samples']['p_p1']
            recovered = float(np.median(samples))
            assert abs(recovered-.1) < .005, (sampler, recovered)
            print(sampler, 'recovered radius ratio', recovered, flush=True)
    print('Installed environment smoke checks passed', flush=True)


if __name__ == '__main__':
    main()
