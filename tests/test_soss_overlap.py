"""Order mixing, covariance and noisy chromatic-transit recovery regressions."""
import numpy as np
import pytest
from scipy.special import ndtr

from transitspectroscopy import soss_optimal, soss_overlap, soss


def test_joint_flux_covariance_and_degenerate_profiles():
    profiles=np.zeros((4,12,3))
    profiles[:,2:6,0]=.25
    profiles[:,4:8,1]=.25
    profiles[:,9:12,2]=1/3
    truth=np.broadcast_to([100.,20.,7.],(4,3))
    image=np.einsum('jro,jo->jr',profiles,truth)
    variance=np.ones_like(image)*4
    valid=np.ones_like(image,bool)
    valid[1,3]=False
    valid[2,:]=False
    profiles[3,:,1]=profiles[3,:,0]  # indistinguishable orders must be flagged
    flux,cov,supported,condition,iv=soss_overlap.solve_flux(profiles,image,variance,valid)
    np.testing.assert_allclose(flux[:2],truth[:2])
    for j in (0,1):
        design=profiles[j,valid[j]]
        np.testing.assert_allclose(cov[j],np.linalg.inv(design.T@design/4))
        assert cov[j,0,1]<0
    assert not supported[2:].any()
    assert condition[3]>1e6


def test_unmodeled_order_support_is_masked_instead_of_assumed_zero():
    orders={1:dict(columns=np.arange(4),model_columns=np.array([True,False,True,True]))}
    centers={1:np.full(4,8.)}
    mask=soss_overlap.unmodeled_pixels(orders,centers,(20,4),3)
    np.testing.assert_array_equal(np.flatnonzero(mask[:,1]),np.arange(4,13))
    assert not mask[:,[0,2,3]].any()


def scene():
    x=np.arange(64)
    centers=[np.full(len(x),30.2),50.2+20*x/(len(x)-1)]
    def image(shift=0.,scales=(1.,1.)):
        parts=[]
        for i,y in enumerate(centers):
            row=np.arange(100)[:,None]-y-shift
            profile=sum(ndtr((row+.5-sign*7.5)/3)-ndtr((row-.5-sign*7.5)/3) for sign in (-1,1))/2
            spectrum=(3000. if i==0 else 200.)*(1+.1*np.sin(x/25+i))
            parts.append(profile*spectrum*scales[i])
        return parts
    return x,centers,image


@pytest.mark.parametrize('method',['polynomial','gp'])
def test_joint_training_and_extraction_recover_distinct_transits(method):
    x,centers,scene_image=scene()
    rng=np.random.default_rng(188)
    shifts=[-.07,.07]
    data=[sum(scene_image(s))+rng.normal(0,.7,(100,len(x))) for s in shifts]
    variance=[np.full_like(f,.49) for f in data]
    _,options=soss_optimal.settings([method],dict(spacing=1.,polynomial_order=2,
        gp_options={'n_inducing':5,'length_scale':100.}),15)
    models=[]
    traces=[np.array([y+s for s in shifts]) for y in centers]
    for i,y in enumerate(centers):
        seed=[f.copy() for f in data]
        for f,s in zip(seed,shifts):
            f[np.abs(np.arange(100)[:,None]-centers[1-i]-s)<=18]=np.nan
        models.append(soss_optimal.train(seed,variance,traces[i],x,20,method,options,[1,2]))
    models,diag=soss_overlap.train_joint(models,traces,[x,x],data,variance)
    assert diag['iterations']>0
    fingerprints=[m.fingerprint() for m in models]
    light,errors,naive=[],[],[]
    # Use only genuinely overlapping columns to prevent isolated pixels from
    # hiding cross-contamination in the aggregate diagnostic.
    keep=x<25
    for t in range(64):
        scales=(.993,.98) if 20<=t<44 else (1.,1.)
        shift=.06*np.sin(t/9)
        parts=scene_image(shift,scales)
        frame=sum(parts)+rng.normal(0,.7,parts[0].shape)
        valid=np.ones_like(frame,bool)
        valid[37,3]=False
        f,e,observed,cov,condition=soss_overlap.extract(models,[x,x],
            [y+shift for y in centers],frame,variance[0],valid,15,options)
        assert np.isfinite(f).all() and np.max(condition)<20
        light.append(f[keep].sum(axis=0));errors.append(np.sqrt((e[keep]**2).sum(axis=0)))
        weights=soss.aperture_weights(100,centers[1]+shift,15)
        naive.append(np.sum(frame[:,keep]*weights[:,keep]))
    light,errors=np.array(light),np.array(errors)
    out=np.r_[np.arange(20),np.arange(44,64)]
    depths=1-light[20:44].mean(axis=0)/light[out].mean(axis=0)
    uncertainty=np.sqrt((errors[20:44]**2).mean(axis=0)/24+(errors[out]**2).mean(axis=0)/40)/light[out].mean(axis=0)
    np.testing.assert_array_less(abs(depths-[.007,.02]),4*uncertainty)
    naive=np.array(naive)
    assert abs((1-naive[20:44].mean()/naive[out].mean())-.02)>10*uncertainty[1]
    assert [m.fingerprint() for m in models]==fingerprints


def test_joint_profile_gradient_matches_finite_differences():
    from transitspectroscopy.shared_profile import SharedProfile,_basis,_component_count
    from scipy.linalg import helmert
    x=np.arange(10)
    models=[]
    for _ in (0,1):
        count=_component_count(3,1.)
        models.append(SharedProfile(np.zeros((count-1,2)),_basis(x,'polynomial',2,{'amplitude':1.}),
            helmert(count).T,-.5*((np.arange(count)-(count-1)/2)/1.5)**2,
            x,20,3,1.,'polynomial',{'amplitude':1.},{'training':[]}))
    centers=[np.full((1,10),7.2),np.full((1,10),10.3)]
    data=[30*models[0].evaluate(centers[0][0])+10*models[1].evaluate(centers[1][0])]
    data[0]+=np.random.default_rng(99).normal(0,.1,data[0].shape)
    worker=soss_overlap._JointFit(models,centers,[x,x],data,[np.ones_like(data[0])])
    coefficients=np.random.default_rng(3).normal(0,.01,worker.size)
    value,gradient,hessian,_=worker.evaluate(coefficients,curvature=True)
    numeric=[]
    for i in range(len(coefficients)):
        d=np.zeros_like(coefficients);d[i]=1e-5
        numeric.append((worker.evaluate(coefficients+d)[0]-worker.evaluate(coefficients-d)[0])/2e-5)
    np.testing.assert_allclose(gradient,numeric,atol=2e-7,rtol=1e-5)
    np.testing.assert_allclose(hessian,hessian.T,atol=1e-10)
    assert np.linalg.eigvalsh(hessian).min()>-1e-8
    # Combining curvature before the dispersion-basis expansion must preserve
    # the likelihood of independent integrations, including unequal variances.
    repeated=[np.repeat(c,2,axis=0) for c in centers]
    combined=soss_overlap._JointFit(models,repeated,[x,x],[data[0],.8*data[0]],
                                    [np.ones_like(data[0]),np.ones_like(data[0])*2])
    second=soss_overlap._JointFit(models,centers,[x,x],[.8*data[0]],[np.ones_like(data[0])*2])
    total=combined.evaluate(coefficients,curvature=True)
    extra=second.evaluate(coefficients,curvature=True)
    np.testing.assert_allclose(total[0],value+extra[0])
    np.testing.assert_allclose(total[1],gradient+extra[1],atol=1e-10)
    np.testing.assert_allclose(total[2],hessian+extra[2],atol=1e-10)
