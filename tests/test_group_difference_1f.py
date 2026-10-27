"""Group-difference 1/f correction: NIRSpec path, the fit_ramps flow and the defaults."""
import numpy as np
import pytest

from transitspectroscopy import instruments, jwst, runner

NINTS, NGROUPS, NROWS, NCOLS = 24, 4, 32, 512
CENTER = 15.3


def prism_scene():
    """A Gaussian trace across the PRISM subarray, with a transit, column stripes and a bias."""
    rng = np.random.default_rng(3)
    rows = np.arange(NROWS)[:, None]
    amplitude = 200 + 100*np.sin(np.arange(NCOLS)/60.)
    scene = 0.2 + amplitude*np.exp(-0.5*((rows - CENTER)/1.2)**2)
    transit = 1 - 0.01*((np.arange(NINTS) > 8) & (np.arange(NINTS) < 16))
    stripes = rng.normal(0, 3, (NINTS, NGROUPS, NCOLS))
    bias = rng.normal(0, 5, (NROWS, NCOLS))
    clean = (transit[:, None, None, None]*scene*(np.arange(NGROUPS) + 1)[None, :, None, None]
             + bias).astype(np.float32)
    return scene, transit, stripes, clean, clean + stripes[:, :, None, :].astype(np.float32)


def prism_models(models, data, scene, transit):
    ramp = models.RampModel(data=data.copy())
    ramp.groupdq = np.zeros(data.shape, dtype=np.uint8)
    ramp.pixeldq = np.zeros(data.shape[2:], dtype=np.uint32)
    ramp.meta.instrument.name = 'NIRSPEC'
    ramp.meta.instrument.detector = 'NRS1'
    ramp.meta.instrument.filter = 'CLEAR'
    ramp.meta.instrument.grating = 'PRISM'
    ramp.meta.exposure.type = 'NRS_BRIGHTOBJ'
    for key, value in dict(frame_time=1., group_time=1., integration_time=NGROUPS*1.,
                           nframes=1, ngroups=NGROUPS,
                           nints=NINTS, groupgap=0, readpatt='NRSRAPID').items():
        setattr(ramp.meta.exposure, key, value)
    ramp.meta.observation.date = '2025-01-01'
    ramp.meta.observation.time = '00:00:00'
    ramp.meta.subarray.name = 'SUB512'
    ramp.meta.subarray.xstart = ramp.meta.subarray.ystart = 1
    ramp.meta.subarray.xsize, ramp.meta.subarray.ysize = NCOLS, NROWS
    rates = models.CubeModel(data=(transit[:, None, None]*scene).astype(np.float32))
    rates.dq = np.zeros(rates.data.shape, dtype=np.uint32)
    rates.err = np.ones(rates.data.shape, dtype=np.float32)
    return ramp, rates


def test_nirspec_group_difference_1f_traces_and_removes_column_stripes():
    models = pytest.importorskip('stdatamodels.jwst.datamodels')
    scene, transit, stripes, clean, data = prism_scene()
    ramp, rates = prism_models(models, data, scene, transit)
    mode = instruments.get_mode('nirspec/prism')
    out = jwst.group_difference_1f_correction([ramp], [rates], mode)
    np.testing.assert_allclose(out['trace'], CENTER, atol=0.1)
    assert 'group_scale' not in out
    # The 1/f pixels are all those at least 8 rows from the trace centre:
    rows = np.arange(NROWS)[:, None] - np.interp(np.arange(NCOLS), out['trace_columns'],
                                                 out['trace'])
    np.testing.assert_array_equal(out['background_pixels'], np.abs(rows) >= 8)
    # Up to an offset that is the same in every integration (the templates' bias and median
    # stripes), the removed levels are the stripes, and the corrected ramps the clean ones:
    levels = out['levels'] - out['levels'].mean(axis=0)
    np.testing.assert_allclose(levels, stripes - stripes.mean(axis=0), atol=0.3)
    residual = ramp.data - clean
    np.testing.assert_allclose(residual - residual.mean(axis=0), 0., atol=0.3)


def test_group_difference_1f_is_only_for_spectroscopic_modes():
    with pytest.raises(ValueError, match='spectroscopic'):
        jwst.group_difference_1f_correction([], [], instruments.get_mode('miri/photometry'))


def test_fit_ramps_corrects_and_refits_by_default(tmp_path):
    models = pytest.importorskip('stdatamodels.jwst.datamodels')
    pytest.importorskip('jwst.ramp_fitting')
    scene, transit, stripes, clean, data = prism_scene()
    ramp, _ = prism_models(models, data, scene, transit)
    ramp.int_times = np.array(
        [(i + 1, 0, 0, 0, 0, 60000. + i/1440., 0) for i in range(NINTS)],
        dtype=[('integration_number', 'i4'), ('int_start_MJD_UTC', 'f8'),
               ('int_mid_MJD_UTC', 'f8'), ('int_end_MJD_UTC', 'f8'),
               ('int_start_BJD_TDB', 'f8'), ('int_mid_BJD_TDB', 'f8'),
               ('int_end_BJD_TDB', 'f8')])
    for cls, kind, value in ((models.GainModel, 'gain', 2.), (models.ReadnoiseModel, 'readnoise', 3.)):
        with cls(data=np.full((NROWS, NCOLS), value, np.float32)) as reference:
            reference.meta.instrument.name = 'NIRSPEC'
            reference.meta.instrument.detector = 'NRS1'
            reference.meta.subarray.xstart = reference.meta.subarray.ystart = 1
            reference.meta.subarray.xsize, reference.meta.subarray.ysize = NCOLS, NROWS
            reference.save(tmp_path/f'{kind}.fits')
    ramp.save(tmp_path/'test-seg001_nrs1_uncal.fits')
    options = dict(override_gain=str(tmp_path/'gain.fits'),
                   override_readnoise=str(tmp_path/'readnoise.fits'), residual_correction=False)
    dataset = jwst.load([str(tmp_path/'test-seg001_nrs1_uncal.fits')], outputfolder=str(tmp_path),
                        ramp_fit_method='sparta', ramp_fit_options=options)
    assert dataset.mode == 'nirspec/prism'
    dataset.calibration_parameters = {}  # as detector_calibration does
    dataset.fill_calibration_parameters({}, group_difference_1f=True, save=False)
    dataset.fit_ramps(save=True)
    assert dataset.status['group_difference_1f'] == 'COMPLETE'
    saved = list((tmp_path/'ts_outputs').glob('*groupdiff1f_rateints.fits'))
    assert len(saved) == 1
    # The rates are marked, so that stage2 does not remove 1/f noise again:
    from transitspectroscopy.jwst_models import group_1f_corrected
    assert group_1f_corrected(dataset.rateints_per_segment[0]) and group_1f_corrected(saved[0])
    # Column stripes of the rates are gone (up to a static pattern), the transit is kept:
    residual = dataset.rateints - transit[:, None, None]*scene
    residual -= residual.mean(axis=0)
    uncorrected = (stripes - stripes.mean(axis=0)).std()
    assert np.std(residual) < 0.1*uncorrected
    # Without it (the default of fill_calibration_parameters), the ramps are fitted once:
    dataset = jwst.load([str(tmp_path/'test-seg001_nrs1_uncal.fits')], outputfolder=str(tmp_path),
                        ramp_fit_method='sparta', ramp_fit_options=options)
    dataset.fit_ramps(save=False)
    assert dataset.status['group_difference_1f'] == 'SKIPPED'


def bare_dataset(mode):
    dataset = object.__new__(jwst.load)
    dataset.status = {'jump': None, 'clean_flicker_noise': None, 'group_1f': None}
    dataset.calibration_parameters = {}
    dataset.mode_definition = instruments.get_mode(mode)
    return dataset


def test_group_difference_1f_replaces_clean_flicker_noise_for_soss():
    dataset = bare_dataset('niriss/soss')
    dataset.fill_calibration_parameters({}, group_difference_1f=True, save=False)
    assert dataset.status['group_difference_1f'] is None
    assert dataset.status['clean_flicker_noise'] == 'SKIPPED'
    dataset = bare_dataset('niriss/soss')
    dataset.fill_calibration_parameters({}, group_difference_1f=False, save=False)
    assert dataset.status['group_difference_1f'] == 'SKIPPED'
    assert dataset.status['clean_flicker_noise'] is None
    # Modes without the correction (MIRI) skip it:
    dataset = bare_dataset('miri/photometry')
    dataset.fill_calibration_parameters({}, group_difference_1f=True, save=False)
    assert dataset.status['group_difference_1f'] == 'SKIPPED'


def test_group_difference_1f_is_the_default():
    import inspect
    signature = inspect.signature(jwst.load.detector_calibration)
    assert signature.parameters['group_difference_1f'].default is True
    for name in ('nirspec/prism', 'nirspec/g395h', 'nirspec/g395m', 'niriss/soss'):
        assert instruments.get_mode(name).detector_steps['group_difference_1f']
    config = runner.load_config({'data': {'glob': '*.fits'}})
    assert config['calibration']['group_difference_1f'] is True


def test_stage2_skips_the_rate_level_1f_correction_of_group_corrected_rates():
    models = pytest.importorskip('stdatamodels.jwst.datamodels')
    from transitspectroscopy.config import ExtractionConfig
    from transitspectroscopy.jwst_models import mark_group_1f
    corrected, uncorrected = (models.CubeModel(data=np.zeros((2, 4, 4), np.float32))
                              for _ in range(2))
    mark_group_1f(corrected)
    rate_level = jwst._rate_level_1f
    assert not rate_level(ExtractionConfig(), {'rampstep': [corrected]})
    assert rate_level(ExtractionConfig(), {'rampstep': [uncorrected]})
    assert rate_level(ExtractionConfig(), {'rampstep': [corrected, uncorrected]})
    assert not rate_level(ExtractionConfig(scale_1f=False), {'rampstep': [uncorrected]})


def test_difference_1f_is_insensitive_to_offsets_that_change_between_integrations():
    # kTC-like offsets (one per pixel and integration) and read noise make the 1/f levels
    # measured on the groups noisy; on the group differences the offsets cancel:
    models = pytest.importorskip('stdatamodels.jwst.datamodels')
    scene, transit, stripes, clean, data = prism_scene()
    rng = np.random.default_rng(5)
    offsets = rng.normal(0, 30, (NINTS, 1, NROWS, NCOLS)).astype(np.float32)
    noise = offsets + rng.normal(0, 5, data.shape).astype(np.float32)
    mode = instruments.get_mode('nirspec/prism')
    scatter = {}
    for domain in ('groups', 'differences'):
        ramp, rates = prism_models(models, data + noise, scene, transit)
        out = jwst.group_difference_1f_correction([ramp], [rates], mode, domain=domain)
        error = np.diff(ramp.data - clean - noise, axis=1)
        scatter[domain] = np.std(error - error.mean(axis=0))
    assert out['levels'].shape == (NINTS, NGROUPS - 1, NCOLS) and 'group_scale' not in out
    uncorrected = np.std(np.diff(stripes, axis=1))
    # (the median templates of the groups remove the static part of the offsets only):
    assert scatter['differences'] < 0.6*scatter['groups'] and scatter['groups'] > 0.75*uncorrected
    with pytest.raises(ValueError, match='domain'):
        jwst.group_difference_1f_correction([ramp], [rates], mode, domain='frames')


def test_difference_1f_excludes_a_jump_from_one_difference_only():
    from transitspectroscopy.core.detector import difference_1f
    rng = np.random.default_rng(1)
    nints, ngroups, nrows, ncols = 30, 5, 40, 8
    stripes = rng.normal(0, 3, (nints, ngroups, 1, ncols))
    ramps = (np.arange(ngroups)[None, :, None, None]*10. + stripes).repeat(nrows, axis=2)
    jumps = np.zeros(ramps.shape, bool)
    # A cosmic ray on the lower half of column 2 of integration 4, at group 2; and an offset
    # of all pixels of integration 7 from group 3 on, with most of them flagged:
    ramps[4, 2:, :20, 2] += 500.
    jumps[4, 2, :20, 2] = True
    ramps[7, 3:] += 200.
    jumps[7, 3, :35] = True
    valid = np.ones(ramps.shape, bool)

    def corrected_differences(jumps):
        data = ramps.copy()
        out = difference_1f([data], [valid], np.ones((nrows, ncols), bool), np.ones(nints),
                            jumps=jumps)
        assert np.all(data[:, 0] == ramps[:, 0])  # the first group is kept
        differences = np.diff(data, axis=1)
        differences[4, 1, :20, 2] -= 500.
        return out, differences - np.median(differences, axis=0)

    # The corrected differences are those of the template, but for the cosmic ray (only in
    # the difference that ends at its group); the offset is shared, and removed:
    out, residual = corrected_differences([jumps])
    assert out['shared'][7, 2] and out['shared'].sum() == 1
    assert np.abs(residual).max() < 0.5 < np.diff(stripes, axis=1).std()/5
    # Unless the cosmic ray is excluded from the levels:
    out, residual = corrected_differences(None)
    assert np.abs(residual[4, 1, :, 2]).max() > 100


def test_detector_calibration_flags_select_the_1f_domain_and_the_ramp_fit():
    dataset = bare_dataset('nirspec/g395h')
    dataset.status.update({step: None for step in ('dq_init', 'saturation', 'superbias',
                                                   'refpix', 'linearity', 'dark_sub',
                                                   'ramp_fit', 'tracing')})
    dataset.ramp_fit_method, dataset.ramp_fit_options = 'jwst', {}
    dataset.outputfolder = 'unused/'

    def calibrate(**options):
        # The flags are applied before the data are touched; this dataset has none:
        with pytest.raises(AttributeError, match='ramps_per_segment'):
            dataset.detector_calibration(save=False, **options)
        return dataset.calibration_parameters['group_difference_1f']

    assert calibrate().get('domain', 'groups') == 'groups'
    assert calibrate(group_difference_1f_domain='differences')['domain'] == 'differences'
    parameters = {'group_difference_1f': {'domain': 'differences'}}
    assert calibrate(parameters=parameters)['domain'] == 'differences'
    with pytest.raises(ValueError, match='differ'):
        dataset.detector_calibration(parameters=parameters, group_difference_1f_domain='groups')
    with pytest.raises(ValueError, match="'groups' or 'differences'"):
        dataset.detector_calibration(group_difference_1f_domain='frames')
    calibrate(ramp_fit_method='sparta', ramp_fit_options={'sigma': 4.0})
    assert dataset.ramp_fit_method == 'sparta' and dataset.ramp_fit_options == {'sigma': 4.0}
    with pytest.raises(ValueError, match='requires ramp_fit_method'):
        dataset.detector_calibration(ramp_fit_options={'sigma': 4.0})


def nonlinear(counts):
    """A linearity correction with curvature: corrected counts of raw ``counts``."""
    return counts + 4e-4*counts**2


def test_median_template_holds_for_ramps_that_are_not_linear():
    # Raw (not linearity-corrected) ramps with column stripes: the median templates remove
    # the stripes; the group-difference templates, scaled in the bright aperture, over-subtract
    # the fainter background pixels by an amount that grows with the group.
    from transitspectroscopy.core.detector import group_difference_1f
    scene, transit, stripes, clean, _ = prism_scene()
    raw = transit[:, None, None, None]*scene*(np.arange(NGROUPS) + 1)[None, :, None, None]
    raw = raw/(1 + 0.5*raw/raw.max())  # the raw counts of a non-linear detector
    data = (raw + stripes[:, :, None, :]).astype(np.float32)
    rows = np.abs(np.arange(NROWS) - CENTER)
    background = np.repeat(((rows >= 2) & (rows < 5))[:, None], NCOLS, axis=1)
    aperture = np.repeat((rows <= 2)[:, None], NCOLS, axis=1)
    valid = np.ones(data.shape, bool)
    bias = {}
    for template in ('median', 'difference'):
        out = group_difference_1f([data.copy()], [valid], background, transit, aperture,
                                  template=template)
        bias[template] = np.abs(np.mean(out['levels'] - stripes, axis=(0, 2)))
    assert bias['median'].max() < 0.2
    assert bias['difference'].max() > 10*bias['median'].max()


def calibrated_dataset(tmp_path, models):
    """A dataset loaded from raw ramps (column stripes), calibrated with stand-in steps: an
    identity before linearity, `nonlinear` for linearity and a dark of 1 DN per group."""
    scene, transit, stripes, clean, data = prism_scene()
    ramp, rates = prism_models(models, data, scene, transit)
    ramp.int_times = np.array(
        [(i + 1, 0, 0, 0, 0, 60000. + i/1440., 0) for i in range(NINTS)],
        dtype=[('integration_number', 'i4'), ('int_start_MJD_UTC', 'f8'),
               ('int_mid_MJD_UTC', 'f8'), ('int_end_MJD_UTC', 'f8'),
               ('int_start_BJD_TDB', 'f8'), ('int_mid_BJD_TDB', 'f8'),
               ('int_end_BJD_TDB', 'f8')])
    ramp.save(tmp_path/'test-seg001_nrs1_uncal.fits')
    dataset = jwst.load([str(tmp_path/'test-seg001_nrs1_uncal.fits')], outputfolder=str(tmp_path))
    dataset.calibration_parameters, dataset.actual_suffix = {}, ''
    dataset.fill_calibration_parameters({}, group_difference_1f=True, save=True)
    calls = []

    def step(name, function=lambda data: data):
        def call(model, **parameters):
            calls.append((name, parameters.get('save_results', False)))
            model.data = function(model.data).astype(np.float32)
            return model
        return call

    dataset.step_calls = {name: step(name) for name in ('dq_init', 'saturation', 'superbias',
                                                        'refpix')}
    dataset.step_calls['linearity'] = step('linearity', nonlinear)
    dataset.step_calls['dark_sub'] = step('dark_sub', lambda data: data - np.arange(
        NGROUPS)[None, :, None, None])
    for name in ('dq_init', 'saturation', 'superbias', 'refpix', 'linearity', 'dark_sub'):
        dataset._run_step(name)
        dataset.status[name] = 'COMPLETE'
    calls.clear()
    jumps = np.zeros(data.shape, np.uint8)
    jumps[5, 2:, 20, 7] = 4
    dataset.ramps_per_segment[0].groupdq = jumps
    dataset.merge_ramps_segments()
    dataset.rateints_per_segment = [rates]
    dataset.run_ramp_fit = lambda save=True, use_saved=True: None
    expected = nonlinear(clean) - np.arange(NGROUPS)[None, :, None, None]
    return dataset, calls, jumps, expected


def test_group_difference_1f_is_applied_before_linearity(tmp_path):
    models = pytest.importorskip('stdatamodels.jwst.datamodels')
    dataset, calls, jumps, expected = calibrated_dataset(tmp_path, models)
    dataset.group_difference_1f(save=False)
    assert dataset.group_difference_1f_diagnostics['before_linearity']
    # The steps run again, without saving, but for the jump step, whose flags are kept:
    assert calls == [(name, False) for name in ('dq_init', 'saturation', 'superbias', 'refpix',
                                                'linearity', 'dark_sub')]
    np.testing.assert_array_equal(dataset.groupdq, jumps)
    # Up to an offset that is the same in every integration, the ramps are those of the raw
    # counts without stripes; removed after linearity, the stripes would leave a part
    # proportional to the counts:
    def error(ramps):
        slope = np.diff(ramps - expected, axis=1).mean(axis=1)
        return np.std(slope - slope.mean(axis=0))
    before = error(dataset.ramps)
    dataset, calls, jumps, expected = calibrated_dataset(tmp_path, models)
    dataset.input_status['linearity'] = 'COMPLETE'  # e.g., inputs corrected for linearity
    dataset.group_difference_1f(save=False)
    assert not dataset.group_difference_1f_diagnostics['before_linearity'] and calls == []
    assert before < 0.2*error(dataset.ramps)


def test_reloaded_products_keep_the_group_difference_1f_status(tmp_path):
    # detector_calibration reloads its saved linearity or jump products with _reload_status,
    # which must not drop the correction (a rerun with save=True would skip it):
    models = pytest.importorskip('stdatamodels.jwst.datamodels')
    dataset, calls, jumps, expected = calibrated_dataset(tmp_path, models)
    dataset.status['clean_flicker_noise'] = 'SKIPPED'
    dataset._reload_status(dataset.ramps_per_segment[0])
    assert dataset.status['group_difference_1f'] is None
    assert dataset.status['clean_flicker_noise'] == 'SKIPPED'
    assert dataset.status['group_1f'] == 'SKIPPED'


def test_white_light_factors_are_the_aperture_rates_filtered_over_a_time_window():
    # Every mode: aperture sums of the rates relative to the median rate, median filtered over
    # scale_window_time (12 s of 4-s integrations: 3 integrations).
    models = pytest.importorskip('stdatamodels.jwst.datamodels')
    from scipy.ndimage import median_filter
    scene, transit, stripes, clean, data = prism_scene()
    ramp, rates = prism_models(models, data, scene, transit)
    noise = 1 + np.random.default_rng(5).normal(0, 1e-3, NINTS)
    rates.data *= noise[:, None, None].astype(np.float32)
    mode = instruments.get_mode('nirspec/prism')
    settings = jwst.group_1f_settings(mode, scale_window_time=12.)
    assert (settings['scale_pixels'], settings['scale_normalization']) == ('aperture',
                                                                            'median_rate')
    traces = {1: (np.arange(NCOLS), np.full(NCOLS, CENTER))}
    out = jwst.group_difference_1f_from_traces([ramp], [rates], traces, scene,
                                               np.zeros(scene.shape), settings)
    aperture = np.abs(np.arange(NROWS) - CENTER) <= settings['aperture_radius']
    expected = (rates.data[:, aperture].sum(axis=(1, 2)) / scene[aperture].sum())
    np.testing.assert_allclose(out['white_light_scale'], median_filter(expected, 3), rtol=1e-6)
    ramp.meta.exposure.integration_time = None
    with pytest.raises(ValueError, match='integration time'):
        jwst.group_difference_1f_from_traces([ramp], [rates], traces, scene,
                                             np.zeros(scene.shape), settings)
