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
    for key, value in dict(frame_time=1., group_time=1., nframes=1, ngroups=NGROUPS,
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
    np.testing.assert_allclose(out['group_scale'], np.arange(NGROUPS) + 1, rtol=0.02)
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
