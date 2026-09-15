"""Differential C tests plus truth-based tests independent of either fitter."""
from pathlib import Path

import numpy as np
import pytest

from simulated_spectra import simulate
from transitspectroscopy import spectroscopy as s
from transitspectroscopy import _optimal_extraction as optimal


def run(v, backend='python', spacing=1., order=3, bounds=(None, None), method='polynomial'):
    return s.getOptimalSpectrum(v['data'], v['centers'], 7, v['ron'], v['gain'],
        8., spacing, order, min_column=bounds[0], max_column=bounds[1],
        data_variance=v.get('variance'), backend=backend, return_P=True,
        profile_method=method)


@pytest.mark.parametrize('seed,spacing,order,reverse,cosmic,explicit,bounds', [
    (11, 1., 3, False, False, True, (None, None)),
    (12, .5, 2, False, False, True, (None, None)),
    (13, .8, 3, True, True, True, (None, None)),
    (14, 1., 2, False, True, False, (None, None)),
    (15, 1., 3, True, False, False, (4, 88)),
    (16, 1., 2, False, False, True, (0, 0)),
])
def test_live_c_equivalence(seed, spacing, order, reverse, cosmic, explicit, bounds):
    pytest.importorskip('Marsh', reason='Build tests/build_c_reference.py to enable live C comparisons')
    v = simulate(seed=seed, reverse=reverse, cosmic=cosmic)
    if not explicit:
        v.pop('variance')
    c, cp = run(v, 'c', spacing, order, bounds)
    p, pp = run(v, 'python', spacing, order, bounds)
    np.testing.assert_allclose(p, c, rtol=2e-7, atol=2e-7)
    np.testing.assert_allclose(pp, cp, rtol=2e-6, atol=2e-8)
    np.testing.assert_array_equal(p[0], np.arange(v['data'].shape[1]))
    # Extraction alone, with EXACTLY the same profile, separates fitting errors.
    args = (v['data'], v['centers'], 7, 5., 1., 8., spacing, order)
    a = s.getOptimalSpectrum(*args, P=cp, data_variance=v.get('variance'), backend='c',
                             min_column=bounds[0], max_column=bounds[1])
    b = s.getOptimalSpectrum(*args, P=cp, data_variance=v.get('variance'), backend='python',
                             min_column=bounds[0], max_column=bounds[1])
    np.testing.assert_allclose(a, b, rtol=2e-12, atol=2e-9)


@pytest.mark.parametrize('phase,radius,bounds', [(0., 7, (None, None)), (.5, 7.9, (3, 88)),
                                                (.25, 14, (None, None))])
def test_fast_simple_c(phase, radius, bounds):
    pytest.importorskip('Marsh')
    v = simulate(phase=phase)
    a = s.getFastSimpleSpectrum(v['data'], v['centers'], radius, *bounds, True, backend='c')
    b = s.getFastSimpleSpectrum(v['data'], v['centers'], radius, *bounds, True, backend='python')
    np.testing.assert_allclose(a[0], b[0], rtol=1e-14, atol=1e-10)
    assert a[1] == b[1]
    assert type(a[1]) is type(b[1]) is float


@pytest.mark.parametrize('function,parameters', [('gaussian', [1., 2.]),
    ('double gaussian', [-3., 1., 4., 2.]), (lambda x: np.exp(-x*x/4.), None)])
def test_ccf_c(function, parameters):
    pytest.importorskip('CCF')
    rng = np.random.default_rng(100)
    x = np.arange(50)[::2]
    y = rng.normal(size=50)[::2]
    a = s.get_ccf(x, y, function, parameters, lag_step=.1, backend='c')
    b = s.get_ccf(x, y, function, parameters, lag_step=.1, backend='python')
    assert isinstance(a[1], list) and isinstance(b[1], list)
    np.testing.assert_array_equal(a[0], b[0])
    np.testing.assert_allclose(a[1], b[1], rtol=1e-12, atol=1e-14)


def test_archived_c_outputs():
    """Always runs, even when no C compiler/GSL is installed."""
    paths = sorted((Path(__file__).parent / 'extraction_comparison').glob('*.npz'))
    assert paths, 'Run compare_extraction.py with C to create the archived fixtures'
    for path in paths:
        with np.load(path) as v:
            result, profile = run(v)
            np.testing.assert_allclose(result, v['c_spectrum'], rtol=2e-7, atol=2e-7)
            np.testing.assert_allclose(profile, v['c_profile'], rtol=2e-6, atol=2e-8)


def test_shapes_bounds_reuse_and_no_mutation():
    v = simulate()
    original = v['data'].copy()
    result, profile = run(v, bounds=(4, 88))
    np.testing.assert_array_equal(v['data'], original)
    assert result.shape == (3, 96) and profile.shape == original.shape
    assert np.all(result[1:, :5] == 0) and np.all(result[1:, 88:] == 0)
    np.testing.assert_allclose(profile[:, 5:88].sum(axis=0), 1.)
    flat = s.getP(original, v['centers'], 7, 5., 1., 8., 1., 3,
                 return_flat=True, data_variance=v['variance'], backend='python')
    assert flat.shape == (original.size,)
    full, p = run(v)
    reused = s.getOptimalSpectrum(original, v['centers'], 7, 5., 1., 8., 1., 3,
                                  P=p, data_variance=v['variance'], backend='python')
    np.testing.assert_array_equal(full, reused)
    _, returned = s.getOptimalSpectrum(original, v['centers'], 7, 5., 1., 8., 1., 3,
                    P=p, data_variance=v['variance'], backend='python', return_P=True)
    assert not np.shares_memory(returned, p)


@pytest.mark.parametrize('method', ['polynomial', 'gp'])
def test_noise_recovery(method):
    residuals = []
    for seed in range(8):
        v = simulate(seed=seed, ncolumns=64)
        result, profile = run(v, method=method)
        np.testing.assert_allclose(profile.sum(axis=0), 1., atol=1e-14)
        assert np.all(profile >= 0.)
        residuals.extend((result[1] - v['truth']) * np.sqrt(result[2]))
    residuals = np.asarray(residuals)
    assert abs(residuals.mean()) < .2
    assert .8 < residuals.std() < 1.2
    assert .60 < np.mean(abs(residuals) < 1.) < .76


def test_known_profile_noise_and_cosmic_recovery():
    v = simulate(cosmic=True)
    result = s.getOptimalSpectrum(v['data'], v['centers'], 7, 5., 1., 8., 1., 3,
                P=v['profile'], data_variance=v['variance'], backend='python')
    # Deliberately injected large impulses must not become spectral features.
    normalized = (result[1] - v['truth']) * np.sqrt(result[2])
    assert np.max(abs(normalized)) < 4.
    assert abs(normalized.mean()) < .3


@pytest.mark.parametrize('depth', [0., .01])
def test_gp_transit_injection(depth):
    """Independent noisy integrations, including a transit-correlated trace shift."""
    estimates, variances = [], []
    for i in range(16):
        in_transit = 4 <= i < 12
        v = simulate(seed=1000+i, ncolumns=48, flux_scale=30000.*(1.-depth*in_transit),
                     phase=.15*in_transit, evolving=True)
        result, _ = run(v, method='gp')
        baseline = v['truth'] / (1.-depth*in_transit)
        estimates.append(np.mean(result[1] / baseline))
        variances.append(np.sum(1./result[2]/baseline**2) / len(baseline)**2)
    mask = (np.arange(16) >= 4) & (np.arange(16) < 12)
    estimates, variances = np.asarray(estimates), np.asarray(variances)
    outside = estimates[~mask].mean()
    recovered = 1. - estimates[mask].mean()/outside
    uncertainty = np.sqrt(variances[mask].sum()/mask.sum()**2 +
                          variances[~mask].sum()/(~mask).sum()**2) / outside
    assert abs(recovered-depth) < max(4.*uncertainty, .001)


@pytest.mark.parametrize('change', ['centers', 'variance', 'profile', 'gain', 'spacing', 'bounds', 'zero'])
def test_invalid_inputs_fail_safely(change):
    v = simulate()
    args = dict(data=v['data'], centroids=v['centers'], aperture_radius=7, ron=5.,
                gain=1., nsigma=8., polynomial_spacing=1., polynomial_order=3,
                data_variance=v['variance'], backend='python')
    if change == 'centers': args['centroids'] = np.ones(2)
    if change == 'variance': args['data_variance'] = -v['variance']
    if change == 'profile': args['P'] = np.ones((2, 2))
    if change == 'gain': args.update(gain=0., data_variance=None)
    if change == 'spacing': args['polynomial_spacing'] = 0.
    if change == 'bounds': args['min_column'] = 200
    if change == 'zero': args['data'] = np.zeros_like(v['data'])
    with pytest.raises(ValueError):
        s.getOptimalSpectrum(**args)


def test_gp_options_and_selection():
    v = simulate(ncolumns=32)
    args = (v['data'], v['centers'], 7, 5., 1., 8., 1., 3)
    for options in ({'length_scale': 0}, {'n_inducing': 100}, {'kernel': 'unknown'}, {'typo': 2}):
        with pytest.raises(ValueError):
            s.getP(*args, backend='python', profile_method='gp', gp_options=options)
    with pytest.raises(ValueError, match='Python'):
        s.getP(*args, backend='c', profile_method='gp')
    with pytest.raises(ValueError, match='gp_options'):
        s.getP(*args, backend='python', gp_options={})
    p = s.getP(*args, backend='python', profile_method='gp',
              gp_options={'kernel':'matern52', 'n_inducing':12})
    assert np.all(np.isfinite(p))


def test_second_image_weights():
    v = simulate()
    spectrum, p = run(v)
    a = optimal.extract(v['data'], v['centers'], 7, 5., 1., 0., 1., p,
                        variance=v['variance'], background=v['data'])
    np.testing.assert_array_equal(a[1], a[3])


def test_second_image_c_equivalence():
    marsh = pytest.importorskip('Marsh')
    v = simulate(cosmic=True)
    _, p = run(v)
    b = np.full_like(v['data'], 2.)
    raw, size = marsh.BObtainSpectrum(v['data'].ravel(), v['centers'], p.ravel(),
        b.ravel(), 32, 96, 96, 7., 5., 1., 1., 8., 0, 96)
    result = optimal.extract(v['data'], v['centers'], 7, 5., 1., 8., 1., p, background=b)
    np.testing.assert_allclose(result, np.asarray(raw).reshape(4,size), rtol=1e-12, atol=2e-9)


def test_masked_pixels_and_unused_zero_variance():
    v = simulate()
    v['data'][16,40] = -9999
    v['variance'][16,40] = 0.
    v['variance'][:3] = 0.  # outside the extraction aperture
    result, p = run(v)
    assert np.all(np.isfinite(result))
    assert abs(result[1,40]-v['truth'][40]) < 4./np.sqrt(result[2,40])
    try:
        s._select_backend('Marsh', 'c')
    except ImportError:
        return
    c, cp = run(v, 'c')
    np.testing.assert_allclose(c, result, rtol=2e-7, atol=2e-7)
    np.testing.assert_allclose(cp, p, atol=2e-8)


def test_fine_spacing_used_by_stage2():
    pytest.importorskip('Marsh')
    v = simulate(ncolumns=441)
    c, cp = run(v, 'c', spacing=.1)
    p, pp = run(v, 'python', spacing=.1)
    # Raw-power normal equations are less well conditioned at this spacing.
    # Assess both model differences and their effect in scientific error units.
    assert np.max(abs(cp-pp)) < 2e-5
    assert np.max(abs(c[1]-p[1])*np.sqrt(c[2])) < .001
    np.testing.assert_allclose(c[2],p[2],rtol=1e-5)
