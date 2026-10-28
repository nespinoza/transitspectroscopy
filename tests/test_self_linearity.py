"""Self non-linearity correction: the core fit and correction, and the detector_calibration flow."""
import numpy as np
import pytest

from transitspectroscopy import instruments, jwst
from transitspectroscopy.core.linearity import (SelfLinearity, fit_self_linearity,
                                                mean_ramp_sums)

NINTS, NGROUPS, NROWS, NCOLS = 400, 12, 4, 6
ALPHA = 6e-7  # raw counts x = q - ALPHA q^2 of the charge q (DN): 1.8% low at 30000 DN


def raw_counts(charge):
    return charge - ALPHA*charge**2


def scene(seed=1, rates=None, noise=True):
    """Raw ramps (integration, group, row, column) of pixels with rates of 10-2500 DN/group,
    with Poisson noise accumulated along the ramps (gain 1.4 e/DN) and 7 DN of read noise."""
    rng = np.random.default_rng(seed)
    if rates is None:
        rates = np.geomspace(10, 2500, NROWS*NCOLS).reshape(NROWS, NCOLS)
    shape = (NINTS, NGROUPS, NROWS, NCOLS)
    increments = np.broadcast_to(rates, shape)
    if noise:
        increments = increments + rng.normal(0, 1, shape)*np.sqrt(rates/1.4)
    charge = np.cumsum(increments, axis=1)
    read = rng.normal(0, 7., shape) if noise else 0.
    return rates, raw_counts(charge) + read


def test_the_fit_makes_the_mean_ramps_linear_and_recovers_the_charge():
    rates, data = scene(noise=False)
    sums, counts = mean_ramp_sums(data)
    correction = fit_self_linearity(sums/counts, counts, min_counts=500.)
    fitted = rates*NGROUPS > 500
    np.testing.assert_array_equal(correction.pixels, fitted)
    charge = rates*(np.arange(NGROUPS) + 1.)[:, None, None]
    corrected = correction.correct(raw_counts(charge))
    # h(0) = 0 and h'(0) = 1: the corrected counts are the charge (a cubic in x approximates
    # the inverse of the quadratic to < 0.01% here); the others are left as they are:
    np.testing.assert_allclose(corrected[:, fitted], charge[:, fitted], rtol=1e-4)
    np.testing.assert_array_equal(corrected[:, ~fitted], raw_counts(charge)[:, ~fitted])
    # With noise, the slopes (offsets of a fraction of a DN do not matter):
    rates, data = scene()
    sums, counts = mean_ramp_sums(data)
    noisy = fit_self_linearity(sums/counts, counts, min_counts=500.)
    # With noise, the slopes over the ramps (offsets do not matter): the fit's scatter is
    # ~1% for the faintest fitted pixels and a few 0.1% above 5000 DN:
    error = (lambda h: (h[-1] - h[0])/(charge[-1] - charge[0]) - 1)(
        noisy.correct(raw_counts(charge)))
    assert np.all(np.abs(error[fitted]) < 0.02)
    assert np.all(np.abs(error[fitted & (charge[-1] > 5000)]) < 0.005)
    # A gain scales the correction:
    np.testing.assert_allclose(correction.correct(raw_counts(charge), gain=0.98)[:, fitted],
                               0.98*corrected[:, fitted])


def test_outside_the_fitted_range_the_correction_continues_as_straight_lines():
    rates, data = scene()
    sums, counts = mean_ramp_sums(data)
    correction = fit_self_linearity(sums/counts, counts)
    row, column = NROWS - 1, NCOLS - 1
    top = correction.top[row, column]
    x = np.array([-200., -1e-6, 1e-6, top - 1e-3, top + 1e-3, top + 5000.])
    rows, columns = np.full(6, row), np.full(6, column)
    h = correction.correct(x, rows=rows, columns=columns)
    np.testing.assert_allclose(h[:3], [-200., -1e-6, 1e-6], rtol=1e-3)
    assert abs(h[4] - h[3]) < 0.01
    slope_above = (h[5] - h[4])/(5000. - 1e-3)
    slope_at_top = (correction.correct(np.array([top - 1.]), rows=rows[:1], columns=columns[:1])
                    - h[3])/(-1. + 1e-3)
    np.testing.assert_allclose(slope_above, slope_at_top[0], rtol=1e-3)


def test_mean_ramp_sums_exclude_jumps_flagged_groups_and_later_groups():
    rates, data = scene(noise=False)
    data = data + np.random.default_rng(5).normal(0, 3., data.shape)
    data[7, 5:, 1, 2] += 500.                    # a cosmic ray: the whole integration goes
    groupdq = np.zeros(data.shape, np.uint8)
    groupdq[3, 8, 2, 3] = 2                      # saturated from group 8 on
    sums, counts = mean_ramp_sums(data, groupdq)
    assert np.all(counts[:, 1, 2] == NINTS - 1)
    assert np.all(counts[:8, 2, 3] == NINTS) and np.all(counts[8:, 2, 3] == NINTS - 1)
    mean = sums/counts
    expected = raw_counts(rates*(np.arange(NGROUPS) + 1.)[:, None, None])
    assert np.max(np.abs(mean - expected)) < 3.


def test_mean_ramps_of_several_visits_are_fitted_together():
    # Two visits of different brightness share the correction; each has its own slope:
    rates, first = scene(seed=1)
    _, second = scene(seed=2, rates=0.8*rates)
    means, counts = [], []
    for data in (first, second):
        s, n = mean_ramp_sums(data)
        means.append(s/n)
        counts.append(n)
    joint = fit_self_linearity(means, counts)
    assert joint.metadata['sets'] == 2
    charge = 0.8*rates*(np.arange(NGROUPS) + 1.)[:, None, None]
    fitted = joint.pixels
    error = (lambda h: (h[-1] - h[0])/(charge[-1] - charge[0]) - 1)(
        joint.correct(raw_counts(charge)))
    assert np.all(np.abs(error[fitted]) < 0.02)
    assert np.all(np.abs(error[fitted & (charge[-1] > 5000)]) < 0.005)
    # A visit without some groups (e.g., saturated) still contributes the others:
    counts[1] = counts[1].copy()
    counts[1][9:] = 0
    partial = fit_self_linearity(means, counts)
    error = (lambda h: (h[-1] - h[0])/(charge[-1] - charge[0]) - 1)(
        partial.correct(raw_counts(charge)))
    assert np.all(np.abs(error[fitted & (charge[-1] > 5000)]) < 0.005)


def test_coefficients_are_saved_and_read_back(tmp_path):
    rates, data = scene()
    sums, counts = mean_ramp_sums(data)
    correction = fit_self_linearity(sums/counts, counts, metadata={'detector': 'NRS1',
                                                                   'xstart': 1})
    correction.save(tmp_path/'self.npz')
    read = SelfLinearity.load(tmp_path/'self.npz')
    for key in ('a2', 'a3', 'top'):
        np.testing.assert_array_equal(getattr(read, key), getattr(correction, key))
    assert read.metadata['detector'] == 'NRS1' and read.metadata['xstart'] == 1
    assert read.metadata['sets'] == 1
    np.testing.assert_allclose(read.mean_ramps[0], correction.mean_ramps[0], rtol=1e-6)
    np.testing.assert_array_equal(read.counts[0], correction.counts[0])
    # User coefficients: any file with the three maps.
    np.savez(tmp_path/'user.npz', a2=correction.a2, a3=correction.a3, top=correction.top)
    user = SelfLinearity.load(tmp_path/'user.npz')
    np.testing.assert_array_equal(user.pixels, correction.pixels)
    assert user.mean_ramps == [] and user.metadata == {}


def bare_dataset():
    dataset = object.__new__(jwst.load)
    dataset.status = {step: None for step in ('jump', 'clean_flicker_noise', 'group_1f',
                                              'dq_init', 'saturation', 'superbias', 'refpix',
                                              'linearity', 'dark_sub', 'ramp_fit', 'tracing')}
    dataset.calibration_parameters = {}
    dataset.mode_definition = instruments.get_mode('nirspec/g395h')
    dataset.ramp_fit_method, dataset.ramp_fit_options = 'jwst', {}
    dataset.outputfolder = 'unused/'
    return dataset


def test_detector_calibration_options_of_the_self_correction():
    dataset = bare_dataset()

    def calibrate(**options):
        # The options are checked before the data are touched; this dataset has none:
        with pytest.raises(AttributeError, match='ramps_per_segment'):
            dataset.detector_calibration(save=False, **options)

    calibrate(self_linearity=True)
    assert dataset.use_self_linearity and dataset.actual_suffix == 'selflin_'
    calibrate(self_linearity=True, suffix='run')
    assert dataset.actual_suffix == 'run_selflin_'
    calibrate(self_linearity_coefficients='coefficients.npz')
    assert dataset.use_self_linearity
    calibrate()
    assert not dataset.use_self_linearity and dataset.actual_suffix == ''
    with pytest.raises(ValueError, match='needs self_linearity'):
        dataset.detector_calibration(parameters={'self_linearity': {'min_counts': 100}})
    with pytest.raises(ValueError, match='Unknown'):
        dataset.detector_calibration(self_linearity=True,
                                     parameters={'self_linearity': {'degree': 4}})


def test_the_linearity_step_applies_the_self_correction(tmp_path):
    models = pytest.importorskip('stdatamodels.jwst.datamodels')
    rates, data = scene()
    ramp = models.RampModel(data=data.astype(np.float32))
    ramp.groupdq = np.zeros(data.shape, np.uint8)
    ramp.pixeldq = np.zeros(data.shape[2:], np.uint32)
    ramp.meta.instrument.detector = 'NRS1'
    ramp.meta.subarray.name, ramp.meta.subarray.xstart, ramp.meta.subarray.ystart = 'S', 1, 1
    ramp.meta.subarray.xsize, ramp.meta.subarray.ysize = NCOLS, NROWS
    dataset = bare_dataset()
    dataset.ramps_per_segment, dataset.datanames = [ramp], ['test']
    dataset.self_linearity_options = {'min_counts': 500.}
    dataset.fit_self_linearity(save=False)
    calls = []

    def crds(model, **parameters):
        # A stand-in for the CRDS step (a different correction), which works in place:
        calls.append(parameters)
        model.data = (model.data*1.01).astype(np.float32)
        return model

    dataset.step_calls = {'crds_linearity': crds}
    dataset._self_linearity_gain = lambda model: np.full(model.data.shape[2:], 0.98)
    output = dataset._self_linearity_step(ramp, suffix='x')
    assert calls == [{'suffix': 'x'}]
    fitted = dataset.self_linearity.pixels
    charge = rates*(np.arange(NGROUPS) + 1.)[:, None, None]
    mean = output.data.mean(axis=0)
    # The fitted pixels are linear, with the gain; the others keep the CRDS stand-in:
    error = (mean[-1] - mean[0])/(0.98*(charge[-1] - charge[0])) - 1
    assert np.all(np.abs(error[fitted]) < 0.02)
    assert np.all(np.abs(error[fitted & (charge[-1] > 5000)]) < 0.005)
    np.testing.assert_allclose(output.data[:, :, ~fitted], 1.01*data[:, :, ~fitted], rtol=1e-6)
    assert any('self non-linearity' in str(entry) for entry in output.history)
    # Coefficients of another subarray are refused:
    dataset.self_linearity = SelfLinearity(*(np.zeros((2, 2)),)*3)
    with pytest.raises(ValueError, match='pixels'):
        dataset._self_linearity_step(ramp)
