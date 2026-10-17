"""Noisy signal recovery and coverage checks for the SOSS validation fitter."""
from pathlib import Path
import sys
import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'validation'))
from soss_transit import fit_transit


def simulated():
    batman=pytest.importorskip('batman')
    t0=2459738.962267
    times=np.linspace(t0-.15,t0+.12,572)
    p=batman.TransitParams()
    p.t0,p.per,p.a=t0,4.62767,8.111
    p.inc=np.degrees(np.arccos(1.014/p.a))
    p.ecc,p.w,p.rp=.11,106.1,.082
    p.u,p.limb_dark=[.2,.3],'quadratic'
    model=batman.TransitModel(p,times,supersample_factor=5,exp_time=np.median(np.diff(times)))
    flux=model.light_curve(p)*(1+.001*(times-t0)/np.ptp(times))
    error=np.ones_like(times)*100e-6
    flux+=np.random.default_rng(1541).normal(0,error)
    config=dict(t0_bjd_tdb=t0,period_days=p.per,a_rs=p.a,b_circular=1.014,eccentricity=p.ecc,omega_degrees=p.w)
    return times,flux,error,config,p.rp**2*1e6


def test_noisy_transit_recovery_with_free_limb_darkening():
    times,flux,error,config,depth=simulated()
    metrics,arrays=fit_transit(times,flux,error,config,fit_time=True)
    assert abs(metrics['depth_ppm']-depth) < 3*metrics['depth_error_ppm']
    assert abs(metrics['t0_bjd_tdb']-config['t0_bjd_tdb']) < .0002
    assert 70 < metrics['residual_rms_ppm'] < 130
    assert metrics['covariance_rank'] == 7
    np.testing.assert_allclose(arrays['residual_ppm'],(arrays['normalized_flux']-arrays['model'])*1e6)


def test_missing_transit_or_baseline_cannot_be_certified():
    times,flux,error,config,_=simulated()
    with pytest.raises(ValueError,match='both out-of-transit'):
        fit_transit(times[:100],flux[:100],error[:100],config)
    error[20]=0
    with pytest.raises(ValueError,match='positive errors'):
        fit_transit(times,flux,error,config)


def test_failed_spectral_bins_are_retained_and_identified(monkeypatch):
    import soss_transit
    def reject(*args,**kwargs):
        raise ValueError('Nonpositive baseline flux')
    monkeypatch.setattr(soss_transit,'fit_transit',reject)
    trace=dict(wavelength=np.linspace(.65,.7,30),white_light_columns=np.ones(30,bool),
               flux=-np.ones((100,30)),error=np.ones((100,30)))
    rows,details=soss_transit.transmission_spectrum(np.arange(100.),trace,
        dict(trim_first_integrations=0,resolving_power=30.))
    assert len(rows)>0 and np.isnan(rows[:,4:6]).all()
    assert np.all(rows[:,9]==1)
    assert all(d['status']=='failed' and 'Nonpositive' in d['reason'] for d in details)


def test_comparison_uses_identical_fixed_columns_and_saves_all_methods(tmp_path):
    from soss_validate import compare_extractions
    times=2459000.+np.arange(120)/1000
    rng=np.random.default_rng(34)
    result=dict(times=times,orders={},optimal={})
    for method in ('polynomial','gp'):
        result['optimal'][method]=dict(orders={},metadata={'profile_training':{}})
    for order in (1,2,3):
        flux=100+rng.normal(0,.1,(120,20))
        common=np.arange(20)<12
        trace=dict(columns=np.arange(20),wavelength=np.linspace(.8,1.5,20),
            flux=flux,error=np.ones_like(flux)*.1,white_light_columns=common)
        result['orders'][order]=trace
        for method in result['optimal']:
            different=flux.copy()
            different[:,12:]*=2  # deliberately different non-comparison band
            result['optimal'][method]['orders'][order]=dict(trace,flux=different,white_light_columns=np.ones(20,bool))
            result['optimal'][method]['metadata']['profile_training'][str(order)] = dict(
                training=[dict(chi2=10.,valid_pixels=10)],integration_ids=['1'],seconds=1.)
    report={}
    compare_extractions(result,{'white_light_filter_points':100},tmp_path,report,transit=False)
    for order in (1,2,3):
        m=report['extraction_comparison'][str(order)]
        assert m['common_columns']==12
        for method in ('polynomial','gp'):
            assert m['methods'][method]['median_relative_spectrum_difference_from_box']==0
            assert m['methods'][method]['median_filter_residual_rms_ppm']==m['methods']['box']['median_filter_residual_rms_ppm']
            assert (tmp_path/f'order{order}_{method}_matched_lightcurve.csv').is_file()
    assert (tmp_path/'08_soss_extraction_comparison.png').is_file()
    assert (tmp_path/'09_soss_profile_fit_quality.png').is_file()
    assert (tmp_path/'10_soss_spectral_differences.png').is_file()
    saved=np.genfromtxt(tmp_path/'order1_gp_matched_spectrum_difference.csv',delimiter=',',names=True)
    np.testing.assert_array_equal(saved['relative_difference'],0)
