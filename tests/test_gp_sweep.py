"""Guard the comparison protocol against changing bands and holdout leakage."""
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'validation'))
spec = importlib.util.spec_from_file_location('validation_gp_sweep', ROOT/'validation/gp_sweep.py')
sweep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sweep)


def test_template_uses_only_training_spectra_and_keeps_missing():
    rng = np.random.default_rng(40)
    flux = 1000+rng.normal(size=(240, 30))
    error = np.ones_like(flux)
    training = np.arange(120)
    corrected, _, _ = sweep.correct_spectra(flux, error, training)
    changed = flux.copy()
    changed[120:, 4] += 1000.
    changed[121, 5] = np.nan
    result, _, replaced = sweep.correct_spectra(changed, error, training)
    np.testing.assert_array_equal(result[:120], corrected[:120])
    assert replaced[120:, 4].all()
    assert np.isnan(result[121, 5]) and not replaced[121, 5]


def test_fixed_band_rejects_missing_candidate_data():
    rng = np.random.default_rng(20)
    flux = 1000+rng.normal(size=(150, 6))
    error = np.ones_like(flux)
    band = np.array([True, True, True, False, False, False])
    metrics, curve = sweep.curve_statistics(flux, error, band, np.arange(150), 100)
    assert metrics['count'] == 150
    np.testing.assert_allclose(curve['residual_ppm'],
        (curve['relative_flux']-curve['median_filter'])*1e6)
    flux[7, 4] = np.nan
    assert sweep.curve_statistics(flux, error, band, np.arange(150), 100)[0] == metrics
    flux[7, 1] = np.nan
    with pytest.raises(ValueError, match='fixed comparison band'):
        sweep.curve_statistics(flux, error, band, np.arange(150), 100)


def test_paired_bootstrap_direction_and_identical_control():
    first = np.random.default_rng(5).normal(size=200)
    same = sweep.paired_bootstrap(first, first, draws=100)
    assert same['improvement_ppm'] == 0 and same['interval_95_ppm'] == [0., 0.]
    better = sweep.paired_bootstrap(first, first*.8, draws=100)
    assert better['interval_95_ppm'][0] > 0
    np.testing.assert_allclose(better['improvement_ppm'], .2*np.sqrt(np.mean(first**2)))


def test_holdout_filter_does_not_use_development_rows():
    rng = np.random.default_rng(15)
    flux = 1000+rng.normal(size=(240, 3))
    error = np.ones_like(flux)
    band = np.ones(3, bool)
    original = sweep.curve_statistics(flux, error, band, np.arange(120, 240), 100)
    flux[:120] *= 10
    changed = sweep.curve_statistics(flux, error, band, np.arange(120, 240), 100)
    np.testing.assert_array_equal(original[1]['residual_ppm'], changed[1]['residual_ppm'])
