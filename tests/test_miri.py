"""MIRI metadata, time mapping, units and prepared-product contracts."""
import numpy as np
import pytest
from transitspectroscopy import miri, photometry as ph
from test_photometry import gaussian, simulated_visit


def calibrated_model():
    dm=pytest.importorskip('stdatamodels.jwst.datamodels')
    d=50+10000*gaussian((51,51),(25.2,24.9))
    model=dm.CubeModel(data=np.repeat(d[None],3,axis=0).astype('float32'))
    model.err=np.full_like(model.data,2.)
    model.dq=np.zeros(model.data.shape,dtype='uint32');model.dq[:,25,25]=1
    model.meta.instrument.name='MIRI';model.meta.instrument.filter='F1500W'
    model.meta.exposure.type='MIR_IMAGE';model.meta.visit.tsovisit=True
    model.meta.cal_step.flat_field='COMPLETE';model.meta.bunit_data='DN/s'
    model.meta.filename='jw09234001001_03101_00001-seg002_mirimage_calints.fits'
    model.meta.observation.program_number='09234';model.meta.observation.observation_number='001'
    model.meta.observation.visit_number='001'
    model.meta.exposure.integration_start=4;model.meta.exposure.integration_end=6
    dtype=[('integration_number','int32')]+[(f'int_{position}_{scale}','float64')
        for scale in ('MJD_UTC','BJD_TDB') for position in ('start','mid','end')]
    table=np.zeros(6,dtype=dtype)
    table['integration_number']=np.arange(1,7);table['int_mid_BJD_TDB']=60000+np.arange(6)*.001
    model.int_times=table
    return model


def test_preparation_masks_times_units_and_roundtrip(tmp_path):
    with calibrated_model() as model:
        before=model.data.copy()
        s=miri.prepare_segment(model,center=(25,25))
        np.testing.assert_allclose(s['times'],2460000.5+np.arange(3,6)*.001)
        assert s['unit']=='DN/s' and (s['flags']==0).all()
        assert np.isnan(s['data'][:,25-s['origin'][1],25-s['origin'][0]]).all()
        np.testing.assert_allclose(s['sky'],50,atol=.001)
        np.testing.assert_array_equal(model.data,before)
        model.meta.bunit_data='MJy/sr';model.meta.photometry.pixelarea_steradians=1e-12
        model.area=np.full((51,51),1.1,dtype='float32')
        physical=miri.prepare_segment(model,center=(25,25))
        assert physical['unit']=='Jy'
        np.testing.assert_allclose(physical['data'],s['data']*1.1e-6,rtol=1e-4,atol=1e-9,equal_nan=True)
        np.testing.assert_allclose(physical['variance'],s['variance']*(1.1e-6)**2,rtol=1e-6,equal_nan=True)
        miri.save_prepared(physical,tmp_path/'prepared.npz')
        saved=miri.load_prepared(tmp_path/'prepared.npz')
        np.testing.assert_array_equal(saved['times'],s['times'])
        np.testing.assert_array_equal(saved['data'],physical['data'])


def test_reject_wrong_mode_missing_calibration_and_time_table():
    with calibrated_model() as model:
        model.meta.visit.tsovisit=False
        with pytest.raises(ValueError,match='TSO'):
            miri.prepare_segment(model,center=(25,25))
        model.meta.visit.tsovisit=True;model.meta.cal_step.flat_field='SKIPPED'
        with pytest.raises(ValueError,match='flat-field'):
            miri.prepare_segment(model,center=(25,25))
        model.meta.cal_step.flat_field='COMPLETE';model.meta.exposure.integration_end=5
        with pytest.raises(ValueError,match='INT_TIMES'):
            miri.prepare_segment(model,center=(25,25))


def test_permanent_bad_pixel_fixed_sum_support_and_no_profile_mutation(tmp_path):
    s=simulated_visit();s['data'][:,7,7]=np.nan
    result=miri.extract_visit([s],aperture_radius=6,training_integrations=4,
        gp_options=dict(n_inducing=5,extent=5.,width=1.6,quadrature=3),outputfolder=tmp_path)
    assert np.isfinite(result['flux'][:,0]).sum()==len(s['times'])-2
    assert [7,7] in result['metadata']['fixed_excluded_detector_pixels']
    m=result['profile_model'];before=m.coefficients.copy()
    m.evaluate(s['data'].shape[1:],[8,8]);m.evaluate(s['data'].shape[1:],[8.2,8.1])
    np.testing.assert_array_equal(m.coefficients,before)
    with pytest.raises(ValueError,match='Duplicate'):
        miri.extract_visit([s,s])


def test_calibration_uses_official_image2_without_mutating_options(monkeypatch,tmp_path):
    pipeline=pytest.importorskip('jwst.pipeline')
    seen={}
    def run(source,**kwargs):
        seen.update(kwargs);return ['calibrated']
    monkeypatch.setattr(pipeline.Image2Pipeline,'call',run)
    options={'resample':{'skip':False},'photom':{'skip':True}}
    assert miri.calibrate('rateints.fits',tmp_path,image_options=options,save_calints=False)=='calibrated'
    assert seen['steps']['resample']['skip'] is True
    assert options['resample']['skip'] is False
    assert seen['steps']['photom']['skip'] is True
