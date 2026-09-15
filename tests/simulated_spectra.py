"""Seeded, pixel-integrated detector simulations independent of Marsh's basis."""
import numpy as np
from scipy.special import ndtr


def simulate(seed=123, nrows=32, ncolumns=96, evolving=False, cosmic=False,
             flux_scale=3000., noise=True, phase=0., reverse=False):
    rng = np.random.default_rng(seed)
    x = np.arange(ncolumns, dtype=float)
    t = x / max(1, ncolumns - 1)
    centers = nrows / 2. + phase + 2. * (t - .5) + .4 * np.sin(2. * np.pi * t)
    if reverse:
        centers = centers[::-1].copy()
    width = np.full(ncolumns, 1.5)
    if evolving:
        width += .35 * np.sin(4. * np.pi * t) + .25 * np.exp(-((t - .7) / .08)**2)
    edges = np.arange(nrows + 1)[:, None] - .5
    profile = np.diff(ndtr((edges - centers) / width), axis=0)
    if evolving:
        wing = np.diff(ndtr((edges - centers - 1.5) / (width * 1.5)), axis=0)
        profile = .9 * profile + .1 * wing
    profile /= profile.sum(axis=0)
    truth = flux_scale * (1. + .2 * t - .25 * np.exp(-((t - .43) / .025)**2))
    expected = profile * truth
    ron = 5.
    variance = expected + ron**2
    data = rng.poisson(expected).astype(float) + rng.normal(0., ron, expected.shape) if noise else expected.copy()
    if cosmic:
        for column, amplitude in ((ncolumns // 3, 1200.), (2 * ncolumns // 3, -700.)):
            data[int(round(centers[column])) + 2, column] += amplitude
    return dict(data=data, variance=variance, truth=truth, profile=profile,
                centers=centers, ron=ron, gain=1., x=x)


def simulate_sequence(nintegrations=16, ncolumns=96, nrows=32, seed=700,
                      flux_scale=10000., depth=.01, evolving=True):
    """Stable intrinsic shape, moving trace, independent noise, and a transit.

    `evolving` changes shape along dispersion, identically in every integration;
    it does not change the intrinsic shape with time. Pixel integration uses
    Gaussian CDFs, independently of the fitter's triangular spatial basis.
    """
    transit = (np.arange(nintegrations) >= nintegrations//3) & (
        np.arange(nintegrations) < 2*nintegrations//3)
    cases = [simulate(seed=seed+i, nrows=nrows, ncolumns=ncolumns,
                      phase=.6*np.sin(i*1.4), evolving=evolving,
                      flux_scale=flux_scale*(1-depth*transit[i])) for i in range(nintegrations)]
    result = {key:np.stack([case[key] for case in cases]) for key in
              ('data','variance','centers','truth','profile')}
    result.update(transit=transit, depth=depth, seed=seed)
    return result
