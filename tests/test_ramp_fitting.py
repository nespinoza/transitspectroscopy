"""Independent covariance checks, noisy injections and JWST interoperability."""
import json
from types import SimpleNamespace
import numpy as np
import pytest

from transitspectroscopy import ramp_fitting as rf
from transitspectroscopy import jwst


def linear_ramps(nints=8, ngroups=12, shape=(2, 3), slope=40., pedestal=1000.):
    return np.broadcast_to(pedestal + slope*np.arange(ngroups)[None, :, None, None],
                           (nints, ngroups, *shape)).copy()


@pytest.mark.parametrize('signal,read_noise', [(0., 12.), (.01, 8.), (50., 4.), (1.e8, 2.), (50., 0.)])
@pytest.mark.parametrize('holes', [False, True])
def test_tridiagonal_matches_dense_covariance(signal, read_noise, holes):
    good = np.ones((2, 11), bool)
    if holes:
        good[0, [0, 4, 5]] = False
        good[1, [2, 9, 10]] = False
    weights = rf.difference_weights(np.full(2, signal), np.full(2, read_noise**2), good)
    c = np.eye(11)*(signal+2*read_noise**2)
    c += (np.eye(11, k=1)+np.eye(11, k=-1))*(-read_noise**2)
    for i in range(2):
        expected = np.linalg.solve(c[np.ix_(good[i], good[i])], np.ones(good[i].sum()))
        np.testing.assert_allclose(weights[i, good[i]], expected, rtol=1.e-12)
        assert np.all(weights[i, ~good[i]] == 0)


def test_paper_analytic_weights_and_limits():
    # Equivalent stable hyperbolic solution to the constant tridiagonal system.
    n, signal, read = 12, 6., 9.
    ell = np.arccosh(1+signal/(2*read**2))
    j = np.arange(1, n+1)
    analytic = (1-np.cosh((j-(n+1)/2)*ell)/np.cosh((n+1)*ell/2))/signal
    np.testing.assert_allclose(rf.difference_weights(signal, read**2, np.ones(n, bool)), analytic, rtol=1.e-12)
    # Read noise only: parabolic weights give the ordinary least-squares slope.
    np.testing.assert_allclose(rf.difference_weights(0., read**2, np.ones(n, bool)),
                               j*(n+1-j)/(2*read**2), rtol=1.e-12)
    # Photon noise only: uniform differences telescope to (last-first)/N.
    np.testing.assert_allclose(rf.difference_weights(signal, 0., np.ones(n, bool)), 1/signal)


def test_units_variances_and_nonmutation():
    data = linear_ramps()
    original = data.copy()
    out = rf.fit_ramps(data, group_time=2., gain=3., read_noise=7., residual_correction=False)
    np.testing.assert_allclose(out['data'], 20.)
    np.testing.assert_allclose(out['err']**2, out['var_poisson']+out['var_rnoise'])
    weights = rf.difference_weights(120., 49., np.ones(11, bool))
    np.testing.assert_allclose(out['err']**2, 1/weights.sum()/6**2)
    np.testing.assert_array_equal(data, original)
    assert np.all(out['n_used'] == 11)
    assert not out['dq'].any()


def test_flags_jumps_missing_groups_and_bad_references():
    data = linear_ramps(shape=(1, 7))
    flags = np.zeros(data.shape, np.uint32)
    flags[:, 5, 0, 0] = 4
    data[:, 5:, 0, 0] += 1.e5
    flags[:, 8, 0, 1] = 2  # Saturation persists even if later flags are absent.
    data[:, 8:, 0, 1] = 1.e8
    data[:, 4, 0, 2] = np.nan
    flags[:, :, 0, 3] = 1
    # An unflagged cosmic ray should be removed by the local difference clipping.
    data[:, 6:, 0, 4] += 1.e5
    pixel_flags = np.zeros((1, 7), np.uint32)
    pixel_flags[0, 5] = 1
    gain = np.full((1, 7), 2.); gain[0, 6] = np.nan
    out = rf.fit_ramps(data, group_time=1., gain=gain, read_noise=5.,
                       groupdq=flags, pixeldq=pixel_flags, residual_correction=False)
    np.testing.assert_allclose(out['data'][:, 0, [0, 1, 2, 4]], 40.)
    assert np.isnan(out['data'][:, 0, [3, 5, 6]]).all()
    assert np.all(out['dq'][:, 0, [3, 5, 6]] & 1)
    np.testing.assert_array_equal(out['n_used'][0, 0], [10, 7, 9, 0, 10, 0, 0])
    assert np.all(out['n_rejected'][:, 0, 4] == 1)
    assert np.all(out['dq'][:, 0, 4] & 4)


def test_negative_slopes_and_two_group_ramps():
    out = rf.fit_ramps(linear_ramps(ngroups=2, slope=-10.), group_time=2,
                       gain=2., read_noise=5., residual_correction=False)
    np.testing.assert_allclose(out['data'], -5.)
    np.testing.assert_allclose(out['var_poisson'], 0.)
    np.testing.assert_allclose(out['var_rnoise'], 50./16)


def test_template_training_and_external_template_preserve_variable_slopes():
    data = linear_ramps(nints=40)
    slopes = np.r_[np.full(20, 40.), np.full(20, 38.)]
    data = data + (slopes-40)[:, None, None, None]*np.arange(12)[None, :, None, None]
    rng = np.random.default_rng(82)
    data += rng.normal(0, 200, (40, 1, 2, 3))  # arbitrary integration pedestals
    curve = 25*np.sin(np.arange(12)*np.pi/11)[None, :, None, None]
    contaminated = data+curve
    out = rf.fit_ramps(contaminated, group_time=1, gain=2, read_noise=10,
                       residual_correction=True, template_integrations=list(range(20)),
                       sigma=None)
    np.testing.assert_allclose(out['data'], np.broadcast_to(slopes[:, None, None], (40, 2, 3)), atol=1.e-12)
    # The inferred template can be reused without retraining on later segments.
    replay = rf.fit_ramps(contaminated, group_time=1, gain=2, read_noise=10,
                          residual_correction=True,
                          residual_template=out['residual_template'], sigma=None)
    np.testing.assert_allclose(replay['data'], out['data'])
    np.testing.assert_allclose(out['residual_template'], np.broadcast_to((curve-np.median(curve)).squeeze()[..., None, None], (12, 2, 3)), atol=1.e-12)


@pytest.mark.parametrize('signal,read', [(0., 20.), (80., 10.), (2000., 2.)])
def test_noisy_injection_bias_and_uncertainty_coverage(signal, read):
    rng = np.random.default_rng(51)
    n = 5000
    # Photon increments are independent; read differences share read noise.
    data = np.cumsum(rng.poisson(signal, (n, 12, 1, 1)), axis=1).astype(float)
    data += rng.normal(0., read, data.shape)
    out = rf.fit_ramps(data/2, group_time=2, gain=2, read_noise=read,
                       residual_correction=False, sigma=None)
    recovered, error = out['data'].ravel(), out['err'].ravel()
    assert abs(np.mean(recovered)-signal/4) < 4*np.std(recovered)/np.sqrt(n)
    assert .92 < np.std(recovered)/np.sqrt(np.mean(error**2)) < 1.08


@pytest.mark.parametrize('options', [{'nframes': 2}, {'group_time': 0}, {'sigma': 0},
    {'firstgroup': 3, 'lastgroup': 3}, {'max_iterations': 0},
    {'template_integrations': [-1]}, {'template_integrations': [1, 1]},
    {'residual_template': np.ones((2, 2, 3))}])
def test_invalid_inputs(options):
    kwargs = dict(group_time=1., gain=2., read_noise=10.)
    kwargs.update(options)
    with pytest.raises(ValueError):
        rf.fit_ramps(linear_ramps(), **kwargs)


def test_chunking_is_invariant():
    rng = np.random.default_rng(7)
    data = linear_ramps()+rng.normal(0, 2, (8, 12, 2, 3))
    a = rf.fit_ramps(data, group_time=1, gain=2, read_noise=10, chunk_pixels=1)
    b = rf.fit_ramps(data, group_time=1, gain=2, read_noise=10, chunk_pixels=64)
    for key in a:
        np.testing.assert_allclose(a[key], b[key])


def test_model_references_roundtrip_and_loader_selection(tmp_path):
    models = pytest.importorskip('stdatamodels.jwst.datamodels')
    pytest.importorskip('jwst.ramp_fitting')
    from test_jwst_compatibility import make_ramp
    with make_ramp(models) as ramp:
        ramp.int_times = np.array([(1, 0, 0, 0, 0, 60000., 0), (2, 0, 0, 0, 0, 60001., 0)],
            dtype=[('integration_number', 'i4'), ('int_start_MJD_UTC', 'f8'),
                   ('int_mid_MJD_UTC', 'f8'), ('int_end_MJD_UTC', 'f8'),
                   ('int_start_BJD_TDB', 'f8'), ('int_mid_BJD_TDB', 'f8'), ('int_end_BJD_TDB', 'f8')])
        for cls, kind, value in ((models.GainModel, 'gain', 2.), (models.ReadnoiseModel, 'readnoise', 3.)):
            with cls(data=np.full((8, 8), value, np.float32)) as ref:
                ref.meta.instrument.name = 'NIRSPEC'; ref.meta.instrument.detector = 'NRS1'
                ref.meta.subarray.xstart = ref.meta.subarray.ystart = 1
                ref.meta.subarray.xsize = ref.meta.subarray.ysize = 8
                ref.save(tmp_path/f'{kind}.fits')
        options = dict(override_gain=str(tmp_path/'gain.fits'), override_readnoise=str(tmp_path/'readnoise.fits'),
                       residual_correction=False)
        rate, cube, diag = rf.fit_model(ramp, **options)
        expected = rf.fit_ramps(ramp.data, group_time=1, gain=2, read_noise=3*2/np.sqrt(2), residual_correction=False)
        np.testing.assert_allclose(cube.err, expected['err'], rtol=1.e-6)
        np.testing.assert_allclose(cube.data, 50.)
        assert cube.meta.bunit_data == 'DN/s'
        cube.save(tmp_path/'rateints.fits')
        with models.open(tmp_path/'rateints.fits') as saved:
            np.testing.assert_array_equal(saved.int_times, ramp.int_times)
            assert saved.meta.cal_step.ramp_fit == 'COMPLETE'
            from astropy.io import fits
            assert 'sparta' in str(fits.getheader(tmp_path/'rateints.fits')['HISTORY'])
            np.testing.assert_allclose(saved.err**2, saved.var_poisson+saved.var_rnoise, rtol=2.e-7)
        rate.close(); cube.close()
        # Select the alternative before detector calibration; fitting calibrated
        # synthetic input exercises the real load/merge/save path without CRDS.
        ramp.save(tmp_path/'test-seg001_nrs1_uncal.fits')
    dataset = jwst.load([str(tmp_path/'test-seg001_nrs1_uncal.fits')], outputfolder=str(tmp_path),
                        ramp_fit_method='sparta', ramp_fit_options=options)
    dataset.fit_ramps(save=True)
    np.testing.assert_allclose(dataset.rateints, 50.)
    assert len(dataset.ramp_fit_diagnostics) == 1
    products = list((tmp_path/'ts_outputs').glob('*sparta_rateints.fits'))
    assert len(products) == 1
    assert len(list((tmp_path/'ts_outputs').glob('*ramp_diagnostics.npz'))) == 1
    assert json.loads(next((tmp_path/'ts_outputs').glob('*ramp_fit.json')).read_text())['method'] == 'sparta'
    for model in dataset.ramps_per_segment+dataset.rateints_per_segment:
        model.close()


def test_invalid_selection_fails_before_io(tmp_path):
    with pytest.raises(ValueError, match='ramp_fit_method'):
        jwst.load(['does_not_exist'], ramp_fit_method='typo')
    with pytest.raises(ValueError, match='Unknown'):
        jwst.load(['does_not_exist'], ramp_fit_method='sparta', ramp_fit_options={'sigam': 3})
    with pytest.raises(ValueError, match='requires ramp inputs'):
        jwst.load(['test-seg001_nrs1_rateints.fits'], outputfolder=str(tmp_path), ramp_fit_method='sparta')


def test_read_outlier_limits_nonconvergence_and_empty_template():
    data = linear_ramps(shape=(1, 3))
    data[:, 4, 0, 0] += 1.e5  # a bad read corrupts two differences
    flags = np.zeros(data.shape, np.uint32)
    flags[..., 0, 2] = 1
    options = dict(group_time=1, gain=2, read_noise=10, groupdq=flags,
                   firstgroup=1, lastgroup=10, residual_correction=False)
    out = rf.fit_ramps(data, **options)
    np.testing.assert_allclose(out['data'][:, 0, :2], 40.)
    assert np.all(out['n_rejected'][:, 0, 0] == 2)
    assert np.all(out['n_used'][:, 0, 0] == 7)
    assert np.all(out['n_used'][:, 0, 1] == 9)
    failed = rf.fit_ramps(data, max_iterations=1, **options)
    assert np.isnan(failed['data'][:, 0, 0]).all()
    assert np.all(failed['dq'][:, 0, 0] & 1)
    assert not failed['converged'][:, 0, 0].any()
    out = rf.fit_ramps(data, **dict(options, residual_correction=True))
    assert np.isfinite(out['residual_template']).all()
    assert np.all(out['residual_template'][:, 0, 2] == 0)
    assert np.all(out['dq'][:, 0, 2] & 1)


def test_official_detector1_replaces_only_ramp_step():
    pytest.importorskip('jwst.pipeline')
    from jwst.pipeline import Detector1Pipeline
    from test_jwst_compatibility import make_ramp
    from stdatamodels.jwst import datamodels
    cls = rf.detector1_class(dict(gain=2, read_noise=10, residual_correction=False))
    for key, step in Detector1Pipeline.step_defs.items():
        if key != 'ramp_fit':
            assert cls.step_defs[key] is step
    step = cls.step_defs['ramp_fit']()
    with make_ramp(datamodels) as ramp:
        image, cube = step.process(ramp)
        np.testing.assert_allclose(cube.data, 50.)
        image.close(); cube.close()


def test_detector1_control_flow_saves_alternative_rateints(tmp_path):
    pytest.importorskip('jwst.pipeline')
    from test_jwst_compatibility import make_ramp
    from stdatamodels.jwst import datamodels
    cls = rf.detector1_class(dict(gain=2, read_noise=10, residual_correction=False))
    # The input is already calibrated; exercise real pipeline dispatch and
    # saving, without needing CRDS for the detector correction steps.
    steps = {name: {'skip': name != 'ramp_fit'} for name in cls.step_defs}
    pipeline = cls(steps=steps, output_dir=str(tmp_path), output_file='synthetic.fits')
    with make_ramp(datamodels) as ramp:
        ramp.meta.filename = 'synthetic_uncal.fits'
        image = pipeline.process(ramp)
        np.testing.assert_allclose(image.data, 50.)
        image.close()
    with datamodels.open(tmp_path/'synthetic_rateints.fits') as rateints:
        np.testing.assert_allclose(rateints.data, 50.)
        assert rateints.meta.cal_step.ramp_fit == 'COMPLETE'


def test_residual_correction_is_off_by_default():
    # The empirical template is trained on all integrations, including any
    # in-transit ones, so it must be an explicit choice.
    import inspect
    default = inspect.signature(rf.fit_ramps).parameters['residual_correction'].default
    assert default is False


def test_tso_jump_turns_off_sparta_rejection(monkeypatch):
    assert rf.jump_step_options(None, True) == {'sigma': None}
    assert rf.jump_step_options({'sigma': 5.0}, True) == {'sigma': 5.0}
    assert rf.jump_step_options({'max_iterations': 3}, False) == {'max_iterations': 3}

    # load.run_ramp_fit passes the options of the jump step that detector_calibration used.
    seen = []

    class Stop(Exception):
        pass

    def fit_model(ramp, **options):
        seen.append(options)
        raise Stop

    monkeypatch.setattr(rf, 'fit_model', fit_model)
    for use_tso_jump, expected in ((True, {'sigma': None}), (False, {})):
        dataset = SimpleNamespace(ramp_fit_method='sparta', ramp_fit_options={},
                                  use_tso_jump=use_tso_jump, datanames=['x'],
                                  ramps_per_segment=[None])
        with pytest.raises(Stop):
            jwst.load.run_ramp_fit(dataset, save=False)
        assert seen[-1] == expected
