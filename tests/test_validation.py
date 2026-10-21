"""Scientific invariants for the validation products, without network access."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('validation_diagnostics', ROOT/'validation/diagnostics.py')
diag = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diag)


def test_pipeline_slope_residuals_and_dq():
    times = np.arange(1, 21)*.902
    rate = np.array([[3., 10.], [0., 15.]])
    data = 1200+times[:, None, None]*rate
    flags = np.zeros(data.shape, dtype=np.uint8)
    data[4, 0, 0] += 1000
    flags[4, 0, 0] = 4
    pixeldq = np.array([[0, 0], [1, 0]], dtype=np.uint32)
    residual, fitted, valid = diag.ramp_residuals(data, rate, times, flags, pixeldq)
    np.testing.assert_allclose(residual[valid], 0, atol=1e-12)
    assert not valid[4, 0, 0] and not valid[:, 1, 0].any()
    assert np.isnan(residual[4, 0, 0])
    np.testing.assert_allclose(np.diff(fitted[:, 0, 1]), .902*10)


def test_exact_even_median_filter_and_missing_values():
    x = np.arange(201.)
    smoothed = diag.smooth_series(x, 100)
    assert smoothed[100] == np.median(x[50:150])
    assert smoothed[0] == np.median(np.pad(x, (50, 49), mode='symmetric')[:100])
    x[100] = np.nan
    assert np.isfinite(diag.smooth_series(x, 100)[100])
    assert np.isnan(diag.smooth_series(np.full(3, np.nan), 100)).all()
    assert diag.selected_integrations(391) == [9, 195, 381]
    assert diag.selected_integrations(1) == [0, 0, 0]


def test_saved_spectra_share_a_fixed_band_and_preserve_differences(tmp_path):
    n, k = 120, 12
    time = 2459000+np.arange(n)/86400
    flux = np.full((n, k), 100.)
    flux[40:60] *= .99
    extractions = {}
    for method, offset in [('simple', 0), ('polynomial', 2), ('gp', -1)]:
        f = flux+offset
        if method == 'gp':
            f[0, 2] = np.nan
        extractions[method] = dict(wavelengths=np.linspace(2.8, 3.7, k), columns=np.arange(k),
            original=f.copy(), original_err=np.ones_like(f), corrected=f.copy(), corrected_err=np.ones_like(f))
    config = json.loads((ROOT/'validation/configs/legacy/config.json').read_text())
    report = diag.extraction_diagnostics(extractions, time, config, tmp_path)
    assert report['common_columns'] == k-1
    assert report['simple_minus_polynomial']['difference_dn_per_second']['median'] == -2.
    assert report['simple_minus_gp']['difference_dn_per_second']['median'] == 1.
    for method in extractions:
        with np.load(tmp_path/f'spectra_{method}.npz') as saved:
            assert not saved['common_white_light_columns'][2]
            assert saved['original_flux'].shape == (n, k)
        table = np.genfromtxt(tmp_path/f'white_light_{method}.csv', delimiter=',', names=True)
        assert np.isfinite(table['relative_flux']).all()
    assert (tmp_path/'06_white_light.png').stat().st_size > 1000
    diag.write_json(tmp_path/'metrics.json', {'metrics': report, 'missing': np.nan})
    assert json.loads((tmp_path/'metrics.json').read_text())['missing'] is None
    for value in extractions.values():
        value['corrected'][:] = np.nan
    with pytest.raises(ValueError, match='No common'):
        diag.extraction_diagnostics(extractions, time, config, tmp_path)


def test_fwhm_and_trace_motion_recovery(tmp_path):
    n, ny, nx = 30, 32, 8
    shift = .05*np.sin(np.arange(n))
    sigma = 1.5+.01*np.cos(np.arange(n))
    trace = np.full((n, nx), 16.)+shift[:, None]
    rows = np.arange(ny)[None, :, None]
    data = 1000*np.exp(-.5*((rows-trace[:, None, :])/sigma[:, None, None])**2)
    result = dict(traces=dict(x=np.arange(nx), y=trace, ycorrected=trace, ysmoothed=trace),
                  tso=data, tso_err=np.ones_like(data), validation_median_rate=np.median(data, axis=0))
    config = json.loads((ROOT/'validation/configs/legacy/config.json').read_text())
    report = diag.trace_diagnostics(result, 2459000+np.arange(n)/86400, config, tmp_path)
    saved = np.load(tmp_path/'trace_fwhm.npz')
    np.testing.assert_allclose(saved['fwhm_pixels'], np.broadcast_to(
        (2*np.sqrt(2*np.log(2))*sigma)[:, None], (n, nx)), atol=.02)
    series = np.genfromtxt(tmp_path/'trace_timeseries.csv', delimiter=',', names=True)
    np.testing.assert_allclose(series['trace_movement_pixels'], shift-np.median(shift), atol=1e-12)
    assert report['fwhm_valid_fraction'] == 1


@pytest.mark.parametrize('kind', ['ramp', 'rateints'])
def test_loader_allocates_only_supplied_integrations(tmp_path, monkeypatch, kind):
    from transitspectroscopy import jwst
    shape = (2, 3, 4, 5) if kind == 'ramp' else (2, 4, 5)
    model = NS(data=np.ones(shape), err=np.ones(shape),
        groupdq=np.zeros(shape, dtype=np.uint8), dq=np.zeros(shape, dtype=np.uint32),
        pixeldq=np.zeros((4, 5), dtype=np.uint32),
        int_times={'int_mid_BJD_TDB': np.array([59000., 59001.])},
        meta=NS(exposure=NS(nints=1139, ngroups=3)))
    monkeypatch.setattr(jwst, 'datamodels', NS(RampModel=lambda _: model, open=lambda _: model))
    monkeypatch.setattr(jwst.load, 'check_status', lambda *args: None)
    dataset = jwst.load([f'example-seg002_nrs1_{kind}.fits'], outputfolder=str(tmp_path))
    assert dataset.nints == len(dataset.times) == 2
    assert (dataset.ramps if kind == 'ramp' else dataset.rateints).shape[0] == 2
    assert model.meta.exposure.nints == 1139


def test_gp_stage2_recovers_original_mask_after_image_filling():
    from transitspectroscopy import jwst
    image = np.full((1, 8, 10), 100.)
    error = np.ones_like(image)
    dq = np.zeros_like(image, dtype=np.uint32)
    image[0, 3, 2] = np.nan
    error[0, 4, 3] = 0.
    dq[0, 5, 4] = 1
    inputs = dict(rampstep=[NS(data=image, err=error, dq=dq)], ints_per_segment=[1])
    mask = jwst._Stage2SharedMask(inputs, True, slice(1, 9))[0]
    # Legacy filling uses zero science and large finite errors for DQ pixels.
    filled = np.nan_to_num(image[0, :, 1:9])
    filled_errors = np.ones_like(filled)
    filled_errors[mask] = 1e9
    frame, variance = jwst._optimal_frame(filled, filled_errors, invalid_mask=mask)
    assert mask.sum() == 3
    assert np.all(frame[mask] == -9999.)
    assert np.all(frame[~mask] == 100.)
    assert np.all(variance[mask] == 1e18)


def test_mask_dq_keeps_jump_only_pixels():
    from transitspectroscopy import jwst
    from transitspectroscopy.jwst_models import dq_masked
    dq = np.zeros((1, 4, 6), dtype=np.uint32)
    dq[0, 1, 1], dq[0, 2, 2], dq[0, 3, 3] = 4, 4 | 2, 1
    assert dq_masked(dq[0]).tolist() == (dq[0] & ~np.uint32(4) != 0).tolist()
    assert dq_masked(dq[0].astype(float)).tolist() == dq_masked(dq[0]).tolist()
    inputs = dict(rampstep=[NS(data=np.ones(dq.shape), err=np.ones(dq.shape), dq=dq)],
                  ints_per_segment=[1])
    mask = jwst._Stage2SharedMask(inputs, True, slice(0, 6))[0]
    assert not mask[1, 1] and mask[2, 2] and mask[3, 3] and mask.sum() == 2


def _driver(name, monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT/'validation'))
    spec = importlib.util.spec_from_file_location(name, ROOT/f'validation/{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_validation_configs_are_run_configurations(monkeypatch):
    import transitspectroscopy as ts
    driver = _driver('validate', monkeypatch)
    for kind, name in driver.DEFAULT_CONFIGS.items():
        config = driver.read_config(ROOT/'validation/configs'/name)
        assert config['validation']['kind'] == kind
        assert Path(config['run']['output']).is_absolute() and Path(config['data']['folder']).is_absolute()
        for settings in config['extraction'].values():
            ts.ExtractionConfig(**settings)  # every variant is a valid Stage 2 configuration
    nrs1 = driver.nrs1_settings(driver.read_config(ROOT/'validation/configs/nrs1-hatp14b.toml'))
    assert nrs1['aperture_radius'] == 2. and nrs1['gp_options']['length_scale'] == 100.
    assert nrs1['extraction_options'] == {'polynomial_spacing': 1., 'polynomial_order': 3, 'nsigma': 10.}
    soss = driver.read_config(ROOT/'validation/configs/soss-hatp14b.toml')
    reference = Path(soss['extraction']['soss']['soss']['background_reference'])
    assert reference.is_absolute() and reference.parent == ROOT/'validation/references/soss'
    assert _driver('soss_validate', monkeypatch).soss_settings(soss)['extraction'] == {'aperture_radius': 15.}
    miri = _driver('miri_validate', monkeypatch).miri_settings(
        driver.read_config(ROOT/'validation/configs/miri-ltt1445a.toml'))
    assert set(miri['visits']) == {'1', '2'} and miri['events'] == {'1': [], '2': []}
