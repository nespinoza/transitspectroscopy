"""Noisy double-peaked PSF recovery with changing traces and missing pixels."""
import numpy as np
import pytest

from transitspectroscopy import soss, soss_optimal


@pytest.mark.parametrize('method', ['polynomial', 'gp'])
def test_frozen_profile_recovers_signal_with_motion_and_masked_peaks(method):
    rng = np.random.default_rng(1541)
    nx, ny = 48, 60
    x = np.arange(nx)
    centers = 29.+.3*np.sin(x/30)
    def scene(shift, scale=1.):
        dy = np.arange(ny)[:,None]-centers-shift
        p = np.exp(-.5*((dy-7.5)/3.)**2)+np.exp(-.5*((dy+7.5)/3.)**2)
        return scale*(1+.2*x/nx)*100*p
    _, options = soss_optimal.settings([method], dict(training_integrations=2,
        polynomial_order=2, spacing=1., gp_options={'n_inducing':6, 'length_scale':80.}), 15)
    shifts = [-.15,.15]
    data = [scene(shift)+rng.normal(0,.5,(ny,nx)) for shift in shifts]
    variance = [np.full((ny,nx),.25) for _ in shifts]
    # A masked peak is absent from the likelihood, never renormalized as wings.
    for a in data:
        a[35:38,20] = np.nan
    model=soss_optimal.train(data,variance,centers+np.asarray(shifts)[:,None],x,15,method,options,[1,2])
    fingerprint=model.fingerprint()
    light, expected, errors = [], [], []
    for i in range(40):
        shift=.12*np.sin(i/7.)
        scale=.993 if 12 <= i < 28 else 1.
        truth=scene(shift,scale)
        frame=truth+rng.normal(0,.5,(ny,nx))
        valid=np.ones_like(frame,bool)
        valid[35:38,20]=False
        # A varying bad peak during transit must not produce a deeper transit.
        if scale < 1:
            valid[35:38,30]=False
        valid[:,5]=False
        frame[~valid]=1e10
        f,e,fraction=soss_optimal.extract(model,frame,variance[0],valid,x,
                                         centers+shift,15,options)
        assert np.isnan(f[5]) and np.isnan(e[5]) and fraction[5] == 0
        assert np.isfinite(f[[20,30]]).all()
        keep=np.arange(nx)!=5
        light.append(f[keep].sum())
        errors.append(np.sqrt((e[keep]**2).sum()))
        expected.append(np.sum(truth[:,keep]*soss.aperture_weights(ny,(centers+shift)[keep],15)))
    light, expected, errors = map(np.asarray,(light,expected,errors))
    outside=np.r_[np.arange(12),np.arange(28,40)]
    depth=1-np.mean(light[12:28])/np.mean(light[outside])
    uncertainty=np.sqrt(np.mean(errors[12:28]**2)/16+np.mean(errors[outside]**2)/24)/np.mean(light[outside])
    assert abs(depth-.007) < 3*uncertainty
    assert abs(np.mean(light/expected)-1) < .001
    assert model.fingerprint()==fingerprint


def test_optimal_settings_reject_unsupported_parameters():
    with pytest.raises(ValueError,match='unique'):
        soss_optimal.settings(['gp','gp'],{},15)
    with pytest.raises(ValueError,match='integer aperture'):
        soss_optimal.settings(['gp'],{},14.5)
    with pytest.raises(ValueError,match='minimum_profile_fraction'):
        soss_optimal.settings(['gp'],{'minimum_profile_fraction':0},15)


def test_profile_coverage_accounts_for_invalid_variance_and_sigma_rejection():
    # Exact known profile isolates weighting/coverage from fitting assumptions.
    p=np.zeros((12,3))
    p[3:9]=1/6
    class Profile:
        def evaluate(self, *args, **kwargs):
            return p.copy()
    data=60*p
    variance=np.ones_like(data)
    valid=np.ones_like(data,bool)
    data[4,0]=1000  # one cosmic ray; removal still leaves sufficient light
    variance[3:7,1]=np.nan  # only two of six profile pixels remain
    valid[:,2]=False
    _,options=soss_optimal.settings(['gp'],{'nsigma':5.},3)
    flux,error,fraction=soss_optimal.extract(Profile(),data,variance,valid,
        np.arange(3),np.full(3,5.5),3,options)
    np.testing.assert_allclose(flux[0],60)
    np.testing.assert_allclose(error[0],np.sqrt(36/5))
    np.testing.assert_allclose(fraction,[5/6,2/6,0])
    assert np.isnan(flux[1:]).all() and np.isnan(error[1:]).all()
