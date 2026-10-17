"""Noisy transit/eclipse recovery, masks, sky covariance and GP persistence."""
import numpy as np
import pytest
from scipy.special import ndtr
from transitspectroscopy import photometry as ph, miri


def gaussian(shape,center,sigma=1.6):
    y,x=np.indices(shape)
    p=(ndtr((x+.5-center[0])/sigma)-ndtr((x-.5-center[0])/sigma))*(
       ndtr((y+.5-center[1])/sigma)-ndtr((y-.5-center[1])/sigma))
    return p/p.sum()


def test_optimal_sky_covariance_matches_dense_solve():
    p=gaussian((11,11),(5.1,5.2));var=np.full(p.shape,4.)
    d=200*p;d[5,5]=np.nan
    good=np.isfinite(d);sv=.3
    cov=np.diag(var[good])+sv
    precision=np.linalg.inv(cov)
    expected=1/(p[good]@precision@p[good])
    f,e,frac,w=ph.optimal_flux(d,var,p,sky_variance=sv)
    assert f==pytest.approx(200)
    assert e**2==pytest.approx(expected)
    assert np.sum(w*p)==pytest.approx(1)
    assert w[5,5]==0 and frac<1
    d[:]=np.nan
    assert np.isnan(ph.optimal_flux(d,var,p)[0])


def test_gp_pixel_integrated_derivatives_and_persistence(tmp_path):
    z=np.array([[-2,-2],[-2,2],[2,-2],[2,2],[0,0.]])
    m=ph.GPPSF(z,np.array([.2,-.1,.3,.1,-.2]),quadrature=3)
    design=m.design((13,13),(6.2,5.9));p,jac=m.render(m.coefficients,design,True)
    for i in range(len(z)):
        a=m.coefficients.copy();a[i]+=1e-5
        b=m.coefficients.copy();b[i]-=1e-5
        numerical=(m.render(a,design)-m.render(b,design))/2e-5
        np.testing.assert_allclose(jac[...,i],numerical,atol=1e-10)
    assert p.sum()==pytest.approx(1) and np.all(p>0)
    fast=m.evaluator((13,13),[[6.2,5.9],[6,6]])
    np.testing.assert_allclose(fast([6.1,6]),m.evaluate((13,13),[6.1,6]),atol=1e-7)
    m.save(tmp_path/'psf.npz');loaded=ph.GPPSF.load(tmp_path/'psf.npz')
    np.testing.assert_array_equal(loaded.evaluate((13,13),[6.2,5.9]),p)


def simulated_visit(seed=9234):
    rng=np.random.default_rng(seed);n=160
    center=np.column_stack((8+.12*np.sin(np.arange(n)/20),8+.08*np.cos(np.arange(n)/25)))
    signal=np.ones(n);signal[35:65]-=.012;signal[100:130]-=.002
    data=np.array([1e5*f*gaussian((17,17),c) for f,c in zip(signal,center)])
    variance=np.full_like(data,9.)
    data+=rng.normal(0,3,data.shape)
    data[10,8,8]=np.nan;variance[20,5,5]=0
    return dict(data=data,variance=variance,centers=center,widths=np.full((n,2),1.6),
        times=2460000.+np.arange(n)*2/86400,sky=np.zeros(n),sky_variance=np.zeros(n),
        sky_pixels=np.full(n,500),flags=np.zeros(n,int),origin=np.zeros(2),median_image=np.nanmedian(data,axis=0),
        unit='DN/s',exposure='synthetic',metadata=dict(filter='F1500W',program='synthetic',observation='1',visit='1'))


def test_noisy_transit_eclipse_recovery_with_moving_psf_and_masks(tmp_path):
    s=simulated_visit();out=miri.extract_visit([s],aperture_radius=6.,training_integrations=6,
        gp_options=dict(n_inducing=5,extent=5.,width=1.6,quadrature=3,maxiter=600),outputfolder=tmp_path)
    for j in range(2):
        f,e=out['flux'][:,j],out['error'][:,j]
        baseline=np.r_[np.arange(0,35),np.arange(65,100),np.arange(130,160)]
        valid=baseline[np.isfinite(f[baseline])]
        normal=np.mean(f[valid])
        for start,end,truth in [(35,65,.012),(100,130,.002)]:
            depth=1-np.mean(f[start:end])/normal
            uncertainty=np.sqrt(np.sum(e[start:end]**2)/(end-start)**2+np.sum(e[valid]**2)/len(valid)**2)/normal
            assert abs(depth-truth)<5*uncertainty
    assert np.isnan(out['flux'][10,0]) and np.isfinite(out['flux'][10,1])
    assert out['flags'][10]&2
    assert out['metadata']['evaluator_max_absolute_error']<1e-6
    assert (tmp_path/'photometry.fits').exists()
    loaded=ph.GPPSF.load(tmp_path/'gp_psf.npz')
    np.testing.assert_array_equal(loaded.coefficients,out['profile_model'].coefficients)


def test_background_rejects_spatial_cosmic_ray_and_preserves_input():
    rng=np.random.default_rng(4);d=10+rng.normal(0,1,(51,51));d[25,40]=1e5
    before=d.copy();b,v,n=ph.background(d,np.ones_like(d),(25,25))
    assert abs(b-10)<4*np.sqrt(v) and n>500
    np.testing.assert_array_equal(d,before)
    with pytest.raises(ValueError,match='Insufficient'):
        ph.background(d,np.ones_like(d),(100,100))
