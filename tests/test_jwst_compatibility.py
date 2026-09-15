"""Ramp-model compatibility; synthetic JWST checks require the full environment."""
from contextlib import nullcontext
from types import SimpleNamespace as NS

import numpy as np
import pytest

from transitspectroscopy import jwst


@pytest.mark.parametrize('present', [(False, False), (True, False),
                                    (False, True), (True, True)])
def test_optional_ramp_errors_and_segment_views(present):
    dataset = jwst.load.__new__(jwst.load)
    dataset.nints, dataset.ngroups, dataset.nrows, dataset.ncols = 3, 2, 2, 3
    dataset.ints_per_segment = [1, 2]
    dataset.ramps_per_segment = []
    for n, has_error in zip(dataset.ints_per_segment, present):
        segment = NS(data=np.full((n, 2, 2, 3), n, dtype=np.float32),
                     groupdq=np.zeros((n, 2, 2, 3), dtype=np.uint8),
                     pixeldq=np.zeros((2, 3), dtype=np.uint32))
        if has_error:
            segment.err = np.full(segment.data.shape, n + .5, dtype=np.float32)
        dataset.ramps_per_segment.append(segment)
    dataset.merge_ramps_segments()
    np.testing.assert_array_equal(dataset.ramps_err_available,
                                  [present[0], present[1], present[1]])
    for i, (segment, has_error) in enumerate(zip(dataset.ramps_per_segment, present)):
        start = sum(dataset.ints_per_segment[:i])
        assert np.shares_memory(segment.data, dataset.ramps)
        assert np.shares_memory(segment.groupdq, dataset.groupdq)
        assert np.shares_memory(segment.pixeldq, dataset.pixeldq)
        if has_error:
            assert np.shares_memory(segment.err, dataset.ramps_err)
            np.testing.assert_array_equal(segment.err, dataset.ints_per_segment[i] + .5)
            segment.err[0, 0, 0, 0] = 12.
            assert dataset.ramps_err[start, 0, 0, 0] == 12.
        else:
            assert not hasattr(segment, 'err')
            assert not dataset.ramps_err[start].any()
    # Repeated merging must preserve error availability and mutations.
    dataset.merge_ramps_segments()
    np.testing.assert_array_equal(dataset.ramps_err_available,
                                  [present[0], present[1], present[1]])


def test_malformed_ramp_error_is_not_silently_discarded():
    dataset = jwst.load.__new__(jwst.load)
    dataset.ramps_per_segment = [NS(data=np.ones((1, 2, 3, 4)), err=np.ones((3, 4)))]
    with pytest.raises(ValueError, match='ERR shape'):
        dataset.merge_ramps_segments()


def test_cds_stage1_does_not_read_unused_ramp_errors(monkeypatch):
    # Stop at tracing to isolate ingestion/CDS from subsequent science methods.
    rows = np.arange(32.)
    signal = np.broadcast_to(np.exp(-.5*((rows-16)/2)**2)[:, None], (32, 2048))
    groups = 100 + np.arange(3)[None, :, None, None]*signal
    class Ramp:
        data = groups
        int_times = {'int_mid_BJD_TDB': np.array([60000.])}
        @property
        def err(self):
            pytest.fail('CDS must not access the removed, unused RampModel.err')
    monkeypatch.setattr(jwst, 'datamodels', NS(RampModel=lambda _: nullcontext(Ramp())))
    class ReachedTracing(Exception):
        pass
    def trace(frame, flags, **kwargs):
        np.testing.assert_allclose(frame, signal, atol=2e-14)
        raise ReachedTracing
    monkeypatch.setattr(jwst, 'trace_spectrum', trace)
    with pytest.raises(ReachedTracing):
        jwst.cds_stage1(['synthetic_nrs1_ramp.fits'], 1, 3)


def _models():
    return pytest.importorskip('stdatamodels.jwst.datamodels')


def make_ramp(models, nints=2):
    ramp = models.RampModel(data=np.broadcast_to(
        1200 + 50*np.arange(8, dtype=np.float32)[None, :, None, None],
        (nints, 8, 8, 8)).copy())
    ramp.groupdq = np.zeros(ramp.data.shape, dtype=np.uint8)
    ramp.pixeldq = np.zeros((8, 8), dtype=np.uint32)
    ramp.meta.instrument.name = 'NIRSPEC'
    ramp.meta.instrument.detector = 'NRS1'
    ramp.meta.instrument.filter = 'CLEAR'
    ramp.meta.instrument.grating = 'PRISM'
    ramp.meta.exposure.type = 'NRS_BRIGHTOBJ'
    ramp.meta.exposure.frame_time = 1.
    ramp.meta.exposure.group_time = 1.
    ramp.meta.exposure.nframes = 1
    ramp.meta.exposure.ngroups = 8
    ramp.meta.exposure.nints = nints
    ramp.meta.exposure.groupgap = 0
    ramp.meta.exposure.readpatt = 'NRSRAPID'
    ramp.meta.observation.date = '2025-01-01'
    ramp.meta.observation.time = '00:00:00'
    ramp.meta.subarray.name = 'GENERIC'
    ramp.meta.subarray.xstart = ramp.meta.subarray.ystart = 1
    ramp.meta.subarray.xsize = ramp.meta.subarray.ysize = 8
    return ramp


def test_installed_jwst_entry_points():
    pytest.importorskip('jwst')
    for module, names in [
        (jwst.calwebb_detector1, ['dq_init_step', 'emicorr_step', 'saturation_step',
                                 'superbias_step', 'dark_current_step', 'refpix_step',
                                 'linearity_step', 'jump_step', 'ramp_fit_step']),
        (jwst.calwebb_spec2, ['assign_wcs_step']),
        (jwst.assign_wcs, ['nrs_wcs_set_input']),
        (jwst.wcstools, ['grid_from_bounding_box']),
        (jwst.datamodels, ['RampModel', 'open']),
    ]:
        for name in names:
            assert getattr(module, name) is not None


def test_real_ramp_roundtrip_and_merge(tmp_path):
    models = _models()
    with make_ramp(models) as ramp:
        path = tmp_path/'synthetic_ramp.fits'
        ramp.save(path)
    with models.RampModel(path) as segment:
        dataset = jwst.load.__new__(jwst.load)
        dataset.nints, dataset.ngroups, dataset.nrows, dataset.ncols = segment.data.shape
        dataset.ints_per_segment = [dataset.nints]
        dataset.ramps_per_segment = [segment]
        has_errors = getattr(segment, 'err', None) is not None
        dataset.merge_ramps_segments()
        assert np.all(dataset.ramps_err_available == has_errors)
        dataset.ramps[0, 1, 0, 0] = 1251.
        assert segment.data[0, 1, 0, 0] == 1251.
        # Merging must not add a removed schema field when saving.
        segment.save(tmp_path/'merged_ramp.fits')
    with models.RampModel(tmp_path/'merged_ramp.fits') as saved:
        assert saved.data[0, 1, 0, 0] == 1251.
        assert (getattr(saved, 'err', None) is not None) == has_errors


def test_real_ramp_fitting_retains_rate_uncertainties(tmp_path):
    models = _models()
    pytest.importorskip('jwst.ramp_fitting')
    for kind, cls, level in [('gain', models.GainModel, 2.),
                              ('readnoise', models.ReadnoiseModel, 3.)]:
        with cls(data=np.full((8, 8), level, dtype=np.float32)) as ref:
            ref.meta.instrument.name = 'NIRSPEC'
            ref.meta.instrument.detector = 'NRS1'
            ref.meta.subarray.name = 'GENERIC'
            ref.meta.subarray.xstart = ref.meta.subarray.ystart = 1
            ref.meta.subarray.xsize = ref.meta.subarray.ysize = 8
            ref.save(tmp_path/(kind+'.fits'))
    # Direct process bypasses CRDS parameter lookup; real algorithm and local
    # reference FITS files are used, without a network or reference-cache dependency.
    step = jwst.calwebb_detector1.ramp_fit_step.RampFitStep(
        override_gain=str(tmp_path/'gain.fits'),
        override_readnoise=str(tmp_path/'readnoise.fits'), maximum_cores='none')
    with make_ramp(models) as ramp:
        rate, rateints = step.process(ramp)
    try:
        np.testing.assert_allclose(rate.data, 50., rtol=1e-5)
        np.testing.assert_allclose(rateints.data, 50., rtol=1e-5)
        assert rateints.meta.cal_step.ramp_fit == 'COMPLETE'
        assert np.all(np.isfinite(rateints.err)) and np.all(rateints.err > 0)
        dataset = jwst.load.__new__(jwst.load)
        dataset.nints, dataset.nrows, dataset.ncols = rateints.data.shape
        dataset.ints_per_segment = [dataset.nints]
        dataset.rateints_per_segment = [rateints]
        dataset.merge_rateints_segments()
        np.testing.assert_array_equal(dataset.rateints_err, rateints.err)
        assert np.shares_memory(dataset.rateints_err, rateints.err)
    finally:
        rate.close()
        rateints.close()
