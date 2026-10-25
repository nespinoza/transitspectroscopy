"""Ramp viewer (transitspectroscopy.utils.inspect_ramps) and the dataset stages it shows."""
import numpy as np
import pytest

matplotlib = pytest.importorskip('matplotlib')
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

from transitspectroscopy import utils  # noqa: E402

NINTS, NGROUPS, NROWS, NCOLS = 40, 6, 32, 200
CENTER = 15.


def scene():
    """Ramps of a trace at row 15, with a two-row offset band that moves down one row per
    integration (rows 29-30 at integration 10), and a jump at integration 5, group 3."""
    rng = np.random.default_rng(1)
    rows = np.arange(NROWS)[:, None]
    rate = 50 + 500*np.exp(-0.5*((rows - CENTER)/1.5)**2)*np.ones(NCOLS)
    groups = np.arange(1, NGROUPS + 1)[None, :, None, None]
    ramps = (rate[None, None]*groups + rng.normal(0, 7, (NINTS, NGROUPS, NROWS, NCOLS)))
    for i in range(10, 39):
        band = 29 - (i - 10)
        ramps[i, :, max(band, 0):band + 2] += 100.
    groupdq = np.zeros(ramps.shape, np.uint8)
    ramps[5, 3:, 15, 40] += 1000.
    groupdq[5, 3:, 15, 40] = 4
    return ramps.astype(np.float32), groupdq


def test_viewer_traces_profiles_and_jumps(tmp_path):
    ramps, groupdq = scene()
    rateints = np.ones((NINTS, NROWS, NCOLS), np.float32)
    viewer = utils.inspect_ramps(ramps, groupdq, show=False, rateints=rateints,
                                 comparison_rateints=rateints, curves={'flat': np.zeros(NINTS)})
    assert np.allclose(viewer.trace, CENTER, atol=0.05)
    assert viewer.new_jumps.sum() == 1 and viewer.new_jumps[5, 3] == 1

    # Over the integrations, the band shows in the aperture when it crosses the trace rows
    # (rows 13-17: integrations 22-27), and not elsewhere:
    aperture, background = viewer.profile('integrations', group=NGROUPS - 1,
                                          mode='counts - median')
    crossing = (np.arange(NINTS) >= 22) & (np.arange(NINTS) <= 27)
    assert aperture[crossing].max() > 30 and np.abs(aperture[~crossing]).max() < 10
    assert np.abs(background).max() < 10

    # Over the groups: an offset is the same in every group (in the counts), and in the
    # increments it shows only in group 0:
    counts, _ = viewer.profile('groups', integration=25, mode='counts - median')
    increments, _ = viewer.profile('groups', integration=25, mode='increment - median')
    assert np.all(counts > 30)
    assert increments[0] > 30 and np.abs(increments[1:]).max() < 10

    for mode in viewer.modes:
        for axis in ('integrations', 'groups'):
            viewer.set(integration=25, group=2, mode=mode, axis=axis)
    viewer.save(tmp_path/'viewer.png', dpi=50)
    assert (tmp_path/'viewer.png').stat().st_size > 0
    plt.close(viewer.figure)


def test_viewer_options():
    ramps, groupdq = scene()
    viewer = utils.inspect_ramps(ramps, show=False, trace=False, first_integration=100,
                                 read_noise=7., gain=1.5, integration=3, axis='groups')
    assert viewer.aperture.all() and not viewer.background.any()
    assert viewer.integration_slider.val == 103 and viewer.new_jumps is None
    assert 'rate / median - 1' not in viewer.modes
    _, noise = viewer._statistics(2)['counts']
    assert np.allclose(noise[0, 0], np.sqrt(49 + max(np.median(ramps[:, 2, 0, 0]), 0)/1.5),
                       rtol=1e-4)
    plt.close(viewer.figure)
    with pytest.raises(ValueError):
        utils.inspect_ramps(ramps, show=False, mode='rate / median - 1')
    with pytest.raises(ValueError):
        utils.inspect_ramps(ramps, show=False, read_noise=7.)


def test_dataset_stages(tmp_path):
    models = pytest.importorskip('stdatamodels.jwst.datamodels')
    from test_group_difference_1f import calibrated_dataset, nonlinear
    dataset, calls, jumps, expected = calibrated_dataset(tmp_path, models)
    dataset.use_tso_jump = False
    calibrated = dataset.inspect_ramps(show=False, trace=False)
    assert calibrated.ramps is not None and calibrated.new_jumps.sum() == 1
    np.testing.assert_array_equal(calibrated.ramps, dataset.ramps)

    with models.RampModel(dataset.filenames[0]) as raw:
        data = np.array(raw.data)
    calls.clear()
    viewer = dataset.inspect_ramps('input', segments=0, show=False, trace=False)
    assert calls == []
    np.testing.assert_allclose(viewer.ramps, data)
    viewer = dataset.inspect_ramps('linearity', show=False, trace=False)
    # The steps run again up to linearity, without saving:
    assert calls == [(name, False) for name in ('dq_init', 'saturation', 'superbias',
                                                'refpix', 'linearity')]
    np.testing.assert_allclose(viewer.ramps, nonlinear(data), rtol=1e-6)
    for stage, segments in (('ramp_fit', None), ('linearity', 1), ('linearity', [0, 2])):
        with pytest.raises(ValueError):
            dataset.inspect_ramps(stage, segments=segments, show=False)
    plt.close('all')
