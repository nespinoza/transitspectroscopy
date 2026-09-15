"""Shared intrinsic profiles: independent simulations and likelihood checks."""
import numpy as np
import pytest
import os
from scipy.linalg import helmert
from scipy.optimize._numdiff import approx_derivative

from transitspectroscopy import fit_shared_profile, SharedProfile, getOptimalSpectrum
from transitspectroscopy import shared_profile as sp
from simulated_spectra import simulate


def sequence(count=8, ncolumns=32, flux=10000., evolving=False, depth=.01, nrows=32):
    cases = [simulate(seed=700+i, nrows=nrows, ncolumns=ncolumns,
                      phase=.6*np.sin(i*1.4), evolving=evolving,
                      flux_scale=flux*(1-depth*(count//3 <= i < 2*count//3))) for i in range(count)]
    return tuple(np.stack([v[key] for v in cases]) for key in
                 ('data','variance','centers','truth'))


def train(data, variance, trace, **kwargs):
    options = dict(aperture_radius=5, profile_method='polynomial', polynomial_order=2,
                   optimize_hyperparameters=False, store_covariance=True)
    options.update(kwargs)
    return fit_shared_profile(data, variance, trace, **options)


def test_linear_evidence_matches_direct_covariance():
    rng = np.random.default_rng(83)
    x, y, v = rng.normal(size=(50,8)), rng.normal(size=50), rng.uniform(.2,3,50)
    nll, mean, covariance = sp._linear_gaussian_evidence(y,x,v)
    c = np.diag(v)+x@x.T
    direct = .5*(y@np.linalg.solve(c,y)+np.linalg.slogdet(c)[1]+50*np.log(2*np.pi))
    np.testing.assert_allclose(nll,direct,rtol=1e-13)
    np.testing.assert_allclose(mean, x.T@np.linalg.solve(c,y),atol=1e-12)
    np.testing.assert_allclose(covariance,np.eye(8)-x.T@np.linalg.solve(c,x),atol=1e-12)


def test_profile_derivatives_and_conditional_curvature():
    data, var, trace, _ = sequence(1,8)
    rows, inside, k = sp._geometry(trace[0],32,4,1.)
    j = np.arange(8)[:,None]
    frame = dict(index=0, centers=trace[0],rows=rows,inside=inside,
                 data=data[0][rows,j],inverse=1/var[0][rows,j],log_normalization=0.)
    c = helmert(k).T
    mean = -.5*((np.arange(k)-(k-1)/2)/1.5)**2
    worker = sp._BatchWorker([frame],1.,k,c,mean,3,0.)
    basis = np.column_stack([np.ones(8),np.linspace(-1,1,8)])
    coef = np.random.default_rng(4).normal(0,.03,(k-1,2))
    value, gradient, hessian, _ = worker.evaluate(coef,basis,True)
    numeric = approx_derivative(lambda a: worker.evaluate(a.reshape(coef.shape),basis)[0],coef.ravel())
    np.testing.assert_allclose(gradient.ravel(), numeric.ravel(),rtol=1e-5,atol=2e-5)
    q = worker.geometry(0,frame)
    p,_ = sp._pixel_profile(q,basis@coef.T@c.T+mean)
    flux = np.sum(p*frame['data']*frame['inverse'],axis=1)/np.sum(p*p*frame['inverse'],axis=1)
    def conditional_gradient(a):
        p,w = sp._pixel_profile(q,basis@a.reshape(coef.shape).T@c.T+mean)
        gp = flux[:,None]*(flux[:,None]*p-frame['data'])*frame['inverse']
        gz = np.einsum('jrk,jr->jk',w,gp)-w.sum(axis=1)*np.sum(p*gp,axis=1)[:,None]
        return ((gz@c).T@basis).ravel()
    numeric_h = approx_derivative(conditional_gradient,coef.ravel())
    np.testing.assert_allclose(hessian,numeric_h,rtol=1e-5,atol=2e-5)


def test_structured_helmert_transform_matches_dense():
    values = np.random.default_rng(134).normal(size=(13,41))
    c = helmert(41).T
    expected = np.stack([c.T@np.diag(row)@c for row in values])
    np.testing.assert_allclose(sp._helmert_diagonal(values),expected,atol=1e-14)


@pytest.mark.parametrize('method',['polynomial','gp'])
def test_joint_recovery_reuse_and_roundtrip(tmp_path, monkeypatch, method):
    data, variance, trace, truth = sequence()
    original = data.copy()
    options = dict(profile_method=method)
    if method == 'gp':
        options['gp_options'] = {'n_inducing':4}
    model = train(data,variance,trace,training_indices=[0,1,6,7],**options)
    assert model.coefficients.shape[0] == 12
    assert model.covariance.shape == (model.coefficients.size,)*2
    np.testing.assert_array_equal(data,original)
    assert not model.coefficients.flags.writeable
    profiles = [model.evaluate(t) for t in trace]
    np.testing.assert_allclose(np.sum(profiles,axis=1),1.,atol=1e-14)
    assert np.max(abs(profiles[0]-profiles[1])) > .01
    model.save(tmp_path/'model.npz')
    restored = SharedProfile.load(tmp_path/'model.npz')
    assert restored.fingerprint() == model.fingerprint()
    monkeypatch.setattr(sp,'minimize',lambda *a,**kw: pytest.fail('reuse must not fit'))
    monkeypatch.setattr(sp,'_fit_coefficients',lambda *a,**kw: pytest.fail('reuse must not fit coefficients'))
    residuals = []
    for i in range(len(data)):
        result = restored.extract(data[i],trace[i],variance[i])
        via_old_api = getOptimalSpectrum(data[i],trace[i],5,1.,1.,0.,1.,2,
                                        P=restored,data_variance=variance[i])
        np.testing.assert_array_equal(result,via_old_api)
        aperture_truth = np.sum((variance[i]-25)*(profiles[i]>0),axis=0)
        residuals.extend((result[1]-aperture_truth)*np.sqrt(result[2]))
    assert abs(np.mean(residuals)) < .3
    assert .7 < np.std(residuals) < 1.3


def test_batches_masks_and_training_selection():
    data, variance, trace, _ = sequence(4,16)
    data[0,16,5] = np.nan
    variance[1,14,6] = 0.
    mask = np.zeros_like(data,dtype=bool)
    mask[2,17,8] = True
    a = train(data,variance,trace,mask=mask,column_batch_size=4,batch_size=1)
    b = train(data,variance,trace,mask=mask,column_batch_size=16,batch_size=3)
    np.testing.assert_allclose(a.evaluate(trace[0]),b.evaluate(trace[0]),atol=2e-6)
    class SelectedOnly:
        def __init__(self,array): self.array = array
        def __len__(self): return len(self.array)
        def __getitem__(self,i):
            assert i in [0,2], 'must not read unselected integrations'
            return self.array[i]
    train(SelectedOnly(data),SelectedOnly(variance),trace,training_indices=[0,2])


@pytest.mark.parametrize('count,ncolumns,evolving',[(3,24,True),(4,16,False)])
def test_hyperparameter_optimization(count,ncolumns,evolving):
    # The small stable case is sensitive to noisy finite differences if inner
    # coefficient convergence is loosened uniformly for large detector sums.
    data, variance, trace, _ = sequence(count,ncolumns,evolving=evolving)
    model = train(data,variance,trace,profile_method='gp',gp_options={'n_inducing':4},
                  optimize_hyperparameters=True, n_starts=1,hyper_maxiter=40)
    assert model.diagnostics['hyperparameters_optimized']
    assert any(r['success'] for r in model.diagnostics['outer_results'])
    assert len(model.diagnostics['evaluations']) > 3
    first = model.diagnostics['evaluations'][0]['objective']
    assert model.diagnostics['objective'] <= first
    assert model.options['amplitude'] != 1.


def test_validation_and_memory_budget():
    d,v,t,_ = sequence(2,8)
    for kw in [dict(training_indices=[0,0]),dict(training_indices=[2]),
               dict(memory_limit_mb=.0001),dict(gp_options={'bad':1},profile_method='gp'),
               dict(execution='threads'),dict(dispersion_coordinates=np.ones(8))]:
        with pytest.raises(ValueError): train(d,v,t,**kw)
    with pytest.raises(RuntimeError,match='converge'):
        train(d,v,t,maxiter=1)
    model = train(d,v,t)
    with pytest.raises(ValueError,match='outside detector'):
        model.evaluate(np.zeros(8))
    with pytest.raises(ValueError,match='coordinates'):
        model.evaluate(t[0],dispersion_coordinates=np.arange(8)+1)


def test_shared_stage2_helper_and_model_reuse(monkeypatch):
    from transitspectroscopy import jwst
    data,var,trace,truth = sequence(4,16)
    args = (data,np.sqrt(var),trace,np.median(trace,axis=0))
    settings = dict(optimize_hyperparameters=False, training_indices=[0,2])
    result = jwst._extract_shared_sequence(*args,'train',settings,5,1.,'polynomial',2,None,
                                          0.,None,np.arange(16))
    spectra, errors, p, ps, model = result
    assert spectra.shape == errors.shape == truth.shape
    assert ps is None and p.shape == data.shape[1:]
    monkeypatch.setattr(sp,'minimize',lambda *a,**k:pytest.fail('supplied model must not retrain'))
    monkeypatch.setattr(sp,'fit_shared_profile',lambda *a,**k:pytest.fail('supplied model must not train'))
    reused = jwst._extract_shared_sequence(*args,model,{},5,1.,'polynomial',2,None,
                                          0.,None,np.arange(16))
    np.testing.assert_array_equal(reused[0],spectra)


def test_transit_and_profile_instability():
    data,var,trace,truth = sequence(24,48,flux=30000.)
    # External/held-out training avoids sharing the science realization's noise.
    model = train(data,var,trace,training_indices=list(range(6)))
    values, uncertainties = [], []
    for i in range(6,24):
        result = model.extract(data[i],trace[i],var[i])
        reference = truth[i]/(1-.01*(8 <= i < 16))
        values.append(np.mean(result[1]/reference))
        uncertainties.append(np.sum(1/result[2]/reference**2)/48**2)
    values, uncertainties = np.asarray(values), np.asarray(uncertainties)
    transit = (np.arange(6,24) >= 8) & (np.arange(6,24) < 16)
    depth = 1-values[transit].mean()/values[~transit].mean()
    sigma = np.sqrt(uncertainties[transit].sum()/transit.sum()**2+
                    uncertainties[~transit].sum()/(~transit).sum()**2)
    assert abs(depth-.01) < max(4*sigma,.001)
    stable = simulate(seed=991,ncolumns=48,flux_scale=30000.)
    changing = simulate(seed=992,ncolumns=48,flux_scale=30000.,evolving=True)
    def reduced_chi2(v):
        result,p = model.extract(v['data'],v['centers'],v['variance'],return_P=True)
        inside = p > 0
        return np.mean(((v['data']-p*result[1])**2/v['variance'])[inside])
    assert reduced_chi2(changing) > 3*reduced_chi2(stable)


@pytest.mark.skipif(os.environ.get('TRANSITSPECTROSCOPY_TEST_RAY') != '1',
                    reason='explicit opt-in for real Ray processes')
@pytest.mark.parametrize('method',['polynomial','gp'])
def test_real_ray_matches_serial(method):
    ray = pytest.importorskip('ray')
    data,var,trace,_ = sequence(4,16)
    options = dict(profile_method=method)
    if method == 'gp':
        options.update(gp_options={'n_inducing':4},optimize_hyperparameters=True,n_starts=1)
    serial = train(data,var,trace,**options)
    parallel = train(data,var,trace,execution='ray',n_workers=2,batch_size=1,**options)
    np.testing.assert_allclose(parallel.evaluate(trace[0]),serial.evaluate(trace[0]),atol=2e-6)
    np.testing.assert_allclose(parallel.diagnostics['objective'],serial.diagnostics['objective'],atol=1e-5)
    assert not ray.is_initialized(), 'internally started Ray must be cleaned up'


def test_accumulation_is_one_shared_likelihood():
    d,v,t,_ = sequence(3,8)
    frames = []
    for index in range(3):
        rows, inside, k = sp._geometry(t[index],32,4,1.)
        cols = np.arange(8)[:,None]
        frames.append(dict(index=index,centers=t[index],rows=rows,inside=inside,
                           data=d[index][rows,cols],inverse=1/v[index][rows,cols],
                           log_normalization=3.))
    c = helmert(k).T
    mean = -.5*((np.arange(k)-(k-1)/2)/1.5)**2
    b = np.column_stack([np.ones(8),np.linspace(-1,1,8)])
    a = np.random.default_rng(92).normal(0,.02,(k-1,2))
    all_at_once = sp._BatchWorker(frames,1.,k,c,mean,8,0).evaluate(a,b,True)
    parts = [sp._BatchWorker([frame],1.,k,c,mean,3,0).evaluate(a,b,True) for frame in frames]
    for i in range(3):
        np.testing.assert_allclose(all_at_once[i],sum(part[i] for part in parts),rtol=1e-12,atol=1e-9)


def test_polynomial_prior_optimization_and_background():
    d,v,t,_ = sequence(3,16)
    background = np.full_like(d,200.)
    # Supply background variance, not just its subtracted mean.
    rng = np.random.default_rng(19)
    observed = d+rng.poisson(background)
    model = train(observed,v+background,t,background=background,
                  optimize_hyperparameters=True,n_starts=1)
    assert model.diagnostics['hyperparameters_optimized']
    report = model.diagnose(observed[0],t[0],v[0]+background[0],background=background[0])
    assert .5 < report['reduced_chi2'] < 1.5


@pytest.mark.parametrize('kernel',['matern32','matern52'])
def test_gp_coordinates_and_existing_kernel(kernel):
    options = dict(amplitude=.7,length_scale=9.,n_inducing=6,kernel=kernel)
    x = np.arange(24.)
    b = sp._basis(x,'gp',3,options)
    np.testing.assert_allclose(b,sp._legacy._gp_basis(24,options),atol=1e-12)
    scaled = sp._basis(3*x+100,'gp',3,dict(options,length_scale=27.))
    np.testing.assert_allclose(b,scaled,atol=1e-12)


def test_narrow_aperture_models_entire_boundary_pixels():
    from simulated_spectra import simulate_sequence
    v = simulate_sequence(nintegrations=5,ncolumns=8,flux_scale=100000.,evolving=False)
    model = fit_shared_profile(v['data'],v['variance'],v['centers'],aperture_radius=2,
                               spacing=.1,profile_method='polynomial',polynomial_order=1,
                               optimize_hyperparameters=False)
    # Without intrinsic support outside the nominal aperture, the same stable
    # Gaussian yielded chi2/dof up to 44 as trace phase changed.
    chi2 = [model.diagnose(d,t,var)['reduced_chi2'] for d,t,var in
            zip(v['data'],v['centers'],v['variance'])]
    assert max(chi2) < 2.5


def test_stage2_shared_cache(tmp_path, monkeypatch):
    from types import SimpleNamespace as NS
    from transitspectroscopy import jwst
    data,var,trace,_ = sequence(3,512)
    data[0,16,100] = np.nan
    meta = NS(instrument=NS(name='NIRSPEC',filter='CLEAR',grating='PRISM',detector='NRS1',
                           fixed_slit='S1600A1'),subarray=NS(name='SUB512'))
    rate = NS(data=data,err=np.sqrt(var),dq=np.zeros_like(data),meta=meta)
    inputs = dict(rampstep=[rate],ints_per_segment=[3],times=2450000.+np.arange(3)/86400.)
    class WCS:
        def __call__(self,x,y): return x*0,y*0,1+x/100.
    monkeypatch.setattr(jwst,'calwebb_spec2',NS(assign_wcs_step=NS(AssignWcsStep=NS(call=lambda x:x))))
    monkeypatch.setattr(jwst,'assign_wcs',NS(nrs_wcs_set_input=lambda *a:WCS()))
    monkeypatch.setattr(jwst,'wcstools',NS(grid_from_bounding_box=lambda b:np.meshgrid(np.arange(512),np.arange(32))))
    def tracing(image,flags,xstart,ystart,xend,**kwargs):
        x = np.arange(min(xstart,xend),max(xstart,xend)+1)
        return x,trace[0,x]
    monkeypatch.setattr(jwst,'trace_spectrum',tracing)
    options = dict(optimal_extraction=True,scale_1f=False,outputfolder=str(tmp_path),
                   aperture_radius=5,extraction_options={'polynomial_spacing':1.,'polynomial_order':2},
                   shared_profile='train',shared_profile_options={'optimize_hyperparameters':False})
    result = jwst.stage2(inputs,**options)
    assert result['spectra']['Ps'] is None
    assert isinstance(result['spectra']['shared_profile'],SharedProfile)
    reports = result['spectra']['shared_profile'].diagnostics['training']
    assert reports[0]['valid_pixels'] == reports[1]['valid_pixels']-1
    assert len(list((tmp_path/'pipeline_outputs').glob('shared_profile*.npz'))) == 1
    monkeypatch.setattr(sp,'minimize',lambda *a,**k:pytest.fail('cached fit must not retrain'))
    monkeypatch.setattr(sp,'fit_shared_profile',lambda *a,**k:pytest.fail('cache must not train'))
    cached = jwst.stage2(inputs,**options)
    np.testing.assert_array_equal(cached['spectra']['original'],result['spectra']['original'])
    reused_options = dict(options, shared_profile=result['spectra']['shared_profile'],
                          shared_profile_options=None, aperture_radius=None, extraction_options=None)
    reused = jwst.stage2(inputs,**reused_options)
    np.testing.assert_array_equal(reused['spectra']['original'],result['spectra']['original'])
