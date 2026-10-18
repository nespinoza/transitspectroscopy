"""SOSS mask, geometry, tracing, time and multi-order dispatch regressions."""
from types import SimpleNamespace as NS
import numpy as np
import pytest
from transitspectroscopy import soss
from transitspectroscopy.jwst_models import integration_times


def test_fractional_aperture_and_variance():
    data = np.ones((40, 5))*7
    variance = np.ones_like(data)*4
    centers = np.array([15., 15.2, 15.6, 16., 16.8])
    f, e = soss.extract_box(data, variance, np.ones_like(data, bool), np.arange(5), centers, 3.)
    np.testing.assert_allclose(f, 42.)
    weights = soss.aperture_weights(40, centers, 3.)
    np.testing.assert_allclose(e**2, np.sum(weights**2*4, axis=0))


def test_missing_aperture_is_not_renormalized_or_zero_filled():
    data = np.ones((40, 4))
    valid = np.ones_like(data, bool)
    valid[15, 1] = False
    f, e = soss.extract_box(data, data, valid, np.arange(4), np.array([15.,15.,1.,15.]), 3.)
    assert np.isfinite(f[[0,3]]).all()
    assert np.isnan(f[[1,2]]).all() and np.isnan(e[[1,2]]).all()


def test_column_levels_with_and_without_scaled_template():
    rng = np.random.default_rng(3)
    template = np.zeros((60, 8))
    template[25:35] = 100.
    template[38:45] = 5.  # light outside the aperture, inside the background pixels
    stripes = rng.normal(0, 2, 8)
    frame = 0.99*template + stripes[None, :]
    valid = np.ones_like(frame, bool)
    aperture = np.zeros_like(valid)
    aperture[25:35] = True
    background = ~aperture
    np.testing.assert_allclose(
        soss.column_levels(frame, valid, background, template, aperture), stripes, atol=0.02)
    # Without the template the median picks up part of the extra light only if it is the
    # majority; here it is not, and the level equals the stripes too:
    np.testing.assert_allclose(soss.column_levels(frame, valid, background), stripes)
    # Columns with no background pixels use the fallback pixels:
    ring = np.zeros_like(valid)
    ring[:, :4] = background[:, :4]
    np.testing.assert_allclose(soss.column_levels(frame, valid, ring, fallback_pixels=background),
                               stripes)
    np.testing.assert_allclose(soss.column_levels(frame, valid, ring)[4:], 0.)


def test_reference_seed_refines_noisy_double_peaked_profile():
    rng = np.random.default_rng(145)
    x = np.arange(160)
    reference = 40+.0002*(x-80)**2
    truth = reference+1.2+.2*np.sin(x/60.)
    dy = np.arange(90)[:, None]-truth
    data = 500*(np.exp(-.5*((dy-7.5)/3.)**2)+np.exp(-.5*((dy+7.5)/3.)**2))
    data += rng.normal(0, 3, data.shape)
    data[20:30, 32] = np.nan
    fitted, details = soss.refine_trace(data, x, reference)
    assert np.sqrt(np.mean((truth-fitted)**2)) < .08
    assert details['accepted'].sum() >= 10


def test_trace_motion_recovers_shift_without_following_flux():
    x = np.arange(80)
    centers = np.ones(80)*40
    y = np.arange(90)[:, None]-centers
    profile = lambda dy: np.exp(-.5*((dy-7.5)/3)**2)+np.exp(-.5*((dy+7.5)/3)**2)
    first = profile(y)
    valid = np.ones_like(first, bool)
    _, _, template = soss.trace_motion(first, valid, x, centers)
    shift, width, _ = soss.trace_motion(.97*profile(y-.7), valid, x, centers, template)
    assert abs(shift-.7) < .03 and 7 < width < 9


def test_segment_times_select_integration_numbers():
    table = np.array([(i, 59000.+i/1000) for i in range(1,7)],
                     dtype=[('integration_number', int), ('int_mid_BJD_TDB', float)])
    m = NS(int_times=table, data=np.zeros((2,3,4)),
           meta=NS(exposure=NS(integration_start=3, integration_end=4)))
    np.testing.assert_allclose(integration_times(m), [2459000.503,2459000.504])


def test_require_ramp_fitting_before_background():
    pytest.importorskip('jwst')
    model = NS(data=np.zeros((2,3,4)), meta=NS(exposure=NS(type='NIS_SOSS'),
               cal_step=NS(ramp_fit=None)))
    with pytest.raises(ValueError, match='ramp fitting'):
        soss.extract_soss([model])


def test_stage2_dispatch_preserves_three_order_result(monkeypatch, tmp_path):
    from transitspectroscopy import jwst
    first = NS(meta=NS(exposure=NS(type='NIS_SOSS')))
    inputs = dict(rampstep=[first])
    called = {}
    def fake(data, **kwargs):
        called.update(kwargs)
        return dict(orders={1:{},2:{},3:{}})
    monkeypatch.setattr(soss, 'extract_soss', fake)
    result = jwst.stage2(inputs, outputfolder=str(tmp_path), soss_options={'trace_max_shift': 3.})
    assert set(result['orders']) == {1,2,3}
    assert called['trace_max_shift'] == 3. and called['aperture_radius'] == 15.
    jwst.stage2(inputs, optimal_extraction=True, profile_method='gp', gp_options={'n_inducing':8})
    assert called['optimal_methods'] == ('gp',)
    assert called['optimal_options']['gp_options'] == {'n_inducing':8}
    with pytest.raises(ValueError, match='Python backend'):
        jwst.stage2(inputs, optimal_extraction=True, extraction_backend='c')


@pytest.mark.parametrize('optimal_methods', [(), ('polynomial', 'gp')])
def test_three_order_noisy_reduction_background_order_and_saved_products(monkeypatch, tmp_path, optimal_methods):
    pytest.importorskip('jwst')
    from stdatamodels.jwst.datamodels import CubeModel
    import jwst.background
    n, ny, nx = 24, 256, 128
    x = np.arange(nx)
    reference = {o:dict(columns=x.copy(), reference_trace=np.full(nx,y),
                       wavelength=np.linspace(2.5/o,1./o,nx)) for o,y in [(1,45.),(2,125.),(3,205.)]}
    scene = np.zeros((ny,nx))
    for t in reference.values():
        d = np.arange(ny)[:,None]-t['reference_trace'][None,:]-.6
        scene += 200*(np.exp(-.5*((d-7.5)/3)**2)+np.exp(-.5*((d+7.5)/3)**2))
    modulation = np.ones(n)
    modulation[8:16] = .993
    data = 3+modulation[:,None,None]*scene[None]+np.random.default_rng(14).normal(0,.2,(n,ny,nx))
    model=CubeModel(data=data.astype('float32'),err=np.full(data.shape,.2,dtype='float32'),
                    dq=np.zeros(data.shape,dtype='uint32'))
    model.meta.exposure.type='NIS_SOSS'
    model.meta.instrument.name='NIRISS'
    model.meta.instrument.filter='CLEAR'
    model.meta.instrument.pupil='GR700XD'
    model.meta.bunit_data='DN/s'
    model.meta.exposure.start_time=59000.
    model.meta.exposure.integration_start=1
    model.meta.exposure.integration_end=n
    model.meta.cal_step.ramp_fit='COMPLETE'
    model.meta.subarray.name='SUBSTRIP256'
    model.meta.instrument.pupil_position=245.76
    dtype=[('integration_number','int32')]+[(f'int_{position}_{scale}','float64')
        for scale in ('MJD_UTC','BJD_TDB') for position in ('start','mid','end')]
    table=np.zeros(n,dtype=dtype)
    table['integration_number']=np.arange(1,n+1)
    table['int_mid_BJD_TDB']=59000+np.arange(n)*.0004
    model.int_times=table
    model.dq[:,45,7]=1  # Permanently missing core: cannot enter a complete aperture.
    model.dq[0,45,8]=4  # A fitted jump is informational, not an invalid slope.
    original=model.data.copy()
    ref=tmp_path/'reference.txt'
    ref.write_text('mock background reference')
    monkeypatch.setattr(soss,'reference_traces',lambda *args: (reference,dict(path='mock',sha256='mock')))
    class Background:
        def __init__(self,**kwargs):
            assert kwargs['override_bkg']==str(ref)
        def run(self,m):
            assert m.meta.cal_step.ramp_fit=='COMPLETE' and m.data.ndim==2
            m.data-=3.
            m.meta.cal_step.bkg_subtract='COMPLETE'
            return m
    monkeypatch.setattr(jwst.background,'BackgroundStep',Background)
    result=soss.extract_soss([model],background_reference=ref,outputfolder=tmp_path/'products',sample_integrations=8,
        optimal_methods=optimal_methods,optimal_options=dict(training_integrations=2,
            polynomial_order=2,gp_options={'n_inducing':3}))
    np.testing.assert_array_equal(model.data,original)
    loaded=soss.load_products(tmp_path/'products')
    for o,t in result['orders'].items():
        np.testing.assert_allclose(loaded['orders'][o]['flux'],t['flux'],equal_nan=True)
        outside=np.r_[t['white_light'][:8],t['white_light'][16:]]
        assert abs(1-np.mean(t['white_light'][8:16])/np.mean(outside)-.007) < .0002
        assert (tmp_path/'products'/f'order{o}_spectra.fits').is_file()
    assert not result['orders'][1]['white_light_columns'][7]
    assert result['orders'][1]['white_light_columns'][8]
    for method in optimal_methods:
        np.testing.assert_allclose(loaded['optimal'][method]['joint_covariance'],
            result['optimal'][method]['joint_covariance'],equal_nan=True)
        for o,t in result['optimal'][method]['orders'].items():
            np.testing.assert_allclose(loaded['optimal'][method]['orders'][o]['flux'],t['flux'],equal_nan=True)
            assert loaded['optimal'][method]['orders'][o]['profile_model'].fingerprint()==t['profile_model'].fingerprint()
            outside=np.r_[t['white_light'][:8],t['white_light'][16:]]
            assert abs(1-np.mean(t['white_light'][8:16])/np.mean(outside)-.007) < .0002
            assert (tmp_path/'products'/'optimal'/method/f'order{o}_spectra.fits').is_file()
            covariance=result['optimal'][method]['joint_covariance']
            np.testing.assert_allclose(covariance[:,t['columns'],o-1,o-1],t['error']**2,equal_nan=True)
        assert result['optimal'][method]['orders'][1]['white_light_columns'][7]
    # Standard products are returned and written; changed code warns instead of refusing.
    names = {f'order{o}_{m}' for o in (1, 2, 3) for m in ('box', *optimal_methods)}
    assert names | {n+'_whitelight' for n in names} == set(result['products'])
    from transitspectroscopy.core import products
    standard = products.read_products(tmp_path/'products'/'products')
    np.testing.assert_array_equal(standard['order1_box'].flux, result['orders'][1]['flux'])
    assert standard['order2_box'].order == 2
    import json
    metadata_path = tmp_path/'products'/'soss_metadata.json'
    metadata = json.loads(metadata_path.read_text())
    metadata['soss_source_sha256'] = 'edited docstring'
    metadata_path.write_text(json.dumps(metadata))
    with pytest.warns(UserWarning, match='SOSS extraction code changed'):
        soss.load_products(tmp_path/'products')
    with pytest.raises(ValueError, match='regenerate products'):
        soss.load_products(tmp_path/'products', strict=True)
    model.close()
