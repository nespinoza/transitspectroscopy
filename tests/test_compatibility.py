"""Public contracts and runtime regressions; scientific legacy quirks are explicit."""
import ast
import inspect
import json
from pathlib import Path
import pickle
import subprocess
import sys
from types import SimpleNamespace as NS

import numpy as np
import pytest

import transitspectroscopy as ts
from transitspectroscopy import jwst, spectroscopy, transitfitting, utils
from simulated_spectra import simulate


def test_legacy_signatures_and_top_level_exports():
    root = Path(__file__).resolve().parents[1]
    snapshot = json.loads((Path(__file__).parent / 'legacy_api.json').read_text())
    trees = {m: ast.parse((root / 'src' / (m+'.py')).read_text())
             for m in ['spectroscopy', 'transitfitting', 'utils', 'timeseries', 'jwst']}
    for name, previous in snapshot.items():
        module, *path = name.split('.')
        nodes = trees[module].body
        for part in path:
            node = next(n for n in nodes if getattr(n, 'name', None) == part)
            nodes = node.body
        names = [a.arg for a in node.args.args]
        assert names[:len(previous['args'])] == previous['args'], name
        defaults = dict(zip(names[-len(node.args.defaults):], map(ast.unparse, node.args.defaults)))
        old_defaults = dict(zip(previous['args'][-len(previous['defaults']):], previous['defaults']))
        for argument, value in old_defaults.items():
            assert defaults[argument] == value, (name, argument)
        if previous['kwarg']:
            assert node.args.kwarg.arg == previous['kwarg']
        if len(path) == 1:
            assert hasattr(getattr(ts, module), path[0])
            if module != 'timeseries':
                assert hasattr(ts, path[0]), name
    assert list(inspect.signature(transitfitting.fit_data).parameters)[:2] == ['data_dictionary', 'priors']
    assert ts.load is jwst.load


def test_core_import_with_optional_packages_blocked():
    code = '''
import sys, importlib.abc, importlib.util
blocked = {'jwst','gwcs','juliet','ray','astroquery','pandas','Marsh','CCF'}
find_spec = importlib.util.find_spec
importlib.util.find_spec = lambda name, *args: None if name.split('.')[0] in blocked else find_spec(name, *args)
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0] in {'jwst','gwcs','juliet','ray','astroquery','pandas','Marsh','CCF'}:
            raise ModuleNotFoundError('deliberately unavailable: '+fullname)
sys.meta_path.insert(0, Block())
import _support
import transitspectroscopy as ts
import numpy as np
lags, ccf = ts.get_ccf(np.arange(8.), np.ones(8), lag_step=1.)
assert len(lags) == len(ccf) == 7
assert ts.getFastSimpleSpectrum(np.ones((20,8)),np.full(8,10.),3).shape == (8,)
try:
    ts.transitfitting.notremote_fit_data({'times':[], 'flux':[], 'error':[]},{})
except ImportError as exc:
    assert 'juliet' in str(exc)
else:
    raise AssertionError('missing capability did not raise')
assert not any(x in sys.modules for x in ['jwst','juliet','ray','pandas','astroquery'])
'''
    result = subprocess.run([sys.executable, '-c', code], cwd=Path(__file__).parent,
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize('starting_point', [None, {'p_p1': .1}])
@pytest.mark.parametrize('regressors', [(), ('GP_external_parameters',),
    ('linear_external_parameters',), ('GP_external_parameters','linear_external_parameters')])
def test_serial_parallel_fits_return_results(monkeypatch, regressors, starting_point):
    calls = []
    def load(**kwargs):
        calls.append(kwargs)
        return NS(fit=lambda **options: {'folder':kwargs['out_folder'], 'sampler':options['sampler']})
    monkeypatch.setattr(transitfitting, 'juliet', NS(load=load))
    fake_ray = NS(init=lambda **kw: None, get=lambda values: values,
                  remote=lambda fn: NS(remote=fn))
    monkeypatch.setattr(transitfitting, 'ray', fake_ray)
    monkeypatch.setitem(sys.modules, 'ray', fake_ray)
    monkeypatch.setattr(transitfitting.fit_data, '_remote', None)
    data = {'times':np.arange(5.), 'flux':np.ones(5), 'error':np.full(5,.1)}
    data.update({name: np.arange(5.) for name in regressors})
    options = {} if starting_point is None else {
        'starting_points': {'a': starting_point, 'b': starting_point}}
    serial = transitfitting.fit_lightcurves({'a':data,'b':data}, {'a':{},'b':{}}, sampler='test', **options)
    parallel = transitfitting.fit_lightcurves({'a':data,'b':data}, {'a':{},'b':{}}, sampler='test', nthreads=2, **options)
    assert serial == parallel == {'a':{'folder':'a','sampler':'test'}, 'b':{'folder':'b','sampler':'test'}}
    for call in calls:
        assert call['starting_point'] == starting_point
        assert set(call['t_lc']) == {'SOSS'}
        assert ('GP_regressors_lc' in call) == ('GP_external_parameters' in regressors)
        assert ('linear_regressors_lc' in call) == ('linear_external_parameters' in regressors)


def test_trace_default_end_and_bad_first_column():
    rows = np.arange(32.)
    frame = np.tile(np.exp(-(rows-16.)**2/8.)[:,None], (1,8))
    dq = np.zeros_like(frame, dtype=int)
    explicit = spectroscopy.trace_spectrum(frame, dq, 7, 16., xend=0, method='centroid')
    default = spectroscopy.trace_spectrum(frame, dq, 7, 16., method='centroid')
    np.testing.assert_array_equal(default, explicit)
    dq[:,7] = 1
    x, y = spectroscopy.trace_spectrum(frame, dq, 7, 16., method='centroid')
    assert y[-1] == 16. and np.all(np.isfinite(y))


def test_pm_ccf_and_spline_error():
    x = np.arange(30.)
    y = np.exp(-(x-15.)**2/5.)
    shifts, score = spectroscopy.get_pm_ccf(x, y, x, y)
    assert abs(shifts[np.argmin(score)]) < 1e-12
    with pytest.raises(ValueError, match='nknots'):
        utils.fit_spline(x, y)


def test_mask_without_dq_and_nan_helper():
    frame = simulate()['data']
    a = jwst.get_uniluminated_mask(frame)
    b = jwst.get_uniluminated_mask(frame, pixeldq=np.zeros_like(frame))
    np.testing.assert_array_equal(a, b)
    dataset = jwst.load.__new__(jwst.load)
    dataset.calibration_parameters = {'tracing':{'row_window':3,'column_window':3}}
    frame = np.ones((7,7))
    frame[3,3] = np.nan
    result = dataset.interpolate_nans(frame, np.where(np.isnan(frame)))
    assert result[3,3] == 1. and np.isnan(frame[3,3])


def test_initialize_suffix_none(tmp_path):
    dataset = jwst.load.__new__(jwst.load)
    dataset.outputfolder = str(tmp_path)+'/'
    dataset.calibration_parameters = {'ramp_fit':{'suffix':'old'}}
    dataset.initialize_step('ramp_fit', {}, True, None, None)
    assert dataset.suffix == '' and dataset.actual_suffix == ''
    assert dataset.calibration_parameters['ramp_fit']['suffix'] == 'ramp_fitstep'


def test_segment_arrays_remain_linked():
    dataset = jwst.load.__new__(jwst.load)
    dataset.nints, dataset.ngroups, dataset.nrows, dataset.ncols = 3, 2, 2, 2
    dataset.ints_per_segment = [1, 2]
    dataset.ramps_per_segment = [NS(data=np.ones((n,2,2,2)), err=np.ones((n,2,2,2)),
        groupdq=np.zeros((n,2,2,2),dtype=np.uint32), pixeldq=np.zeros((2,2),dtype=np.uint32)) for n in (1,2)]
    dataset.merge_ramps_segments()
    dataset.ramps[1,0,0,0] = 42.
    assert dataset.ramps_per_segment[1].data[0,0,0,0] == 42.
    dataset.ramps_per_segment[0].groupdq[0,0,0,0] = 4
    assert dataset.groupdq[0,0,0,0] == 4


def test_legacy_jump_dq_and_input_ownership(monkeypatch):
    # This records existing overwrite behavior; changing DQ policy is separate.
    segment = NS(data=np.zeros((3,2,1,1)), groupdq=np.full((3,2,1,1),2,dtype=np.uint32))
    monkeypatch.setattr(jwst, 'outlier_detector', lambda *args, **kw: np.array([1]))
    output = jwst.tso_jumpstep([segment], window=3)
    assert output[0].groupdq[1,1,0,0] == 4
    assert segment.groupdq[1,1,0,0] == 2
    assert np.shares_memory(output[0].data, segment.data)


def test_legacy_cds_timestamp_units():
    segment = NS(data=np.arange(6.).reshape(2,3,1,1), shape=(2,3,1,1),
        meta=NS(exposure=NS(ngroups=3,frame_time=1.,group_time=2.),
                observation=NS(date='2020-01-01',time='00:00:00')))
    times, frames = jwst.get_cds(segment)
    np.testing.assert_array_equal(frames, np.ones((4,1,1)))
    # Known legacy seconds-as-days bug is frozen rather than silently corrected.
    np.testing.assert_allclose(np.diff(times), [2.,5.,2.])


def test_simple_background_mutation_is_preserved():
    frame = np.full((32,5),10.)
    spectroscopy.getSimpleSpectrum(frame, np.arange(5), np.full(5,16.),3,
                                    background_radius=10,correct_bkg=True)
    assert not np.all(frame == 10.)


@pytest.mark.parametrize('product,expected', [('ramp','RAMP'), ('ramps','RAMP'), ('uncal','UNCAL')])
def test_mast_product_aliases(monkeypatch, product, expected):
    from astropy.table import Table
    products = Table({'obs_id':['jw12345005']*2, 'description':['detector']*2,
        'productSubGroupDescription':['RAMP','UNCAL'], 'productType':['AUXILIARY','SCIENCE'],
        'size':[1024,2048]})
    selected = []
    class DownloadReached(Exception):
        pass
    def download(rows, **kwargs):
        selected.extend(rows['productSubGroupDescription'])
        raise DownloadReached  # Check selection without network/filesystem side effects.
    monkeypatch.setattr(jwst, 'Observations', NS(query_criteria=lambda **kw:None,
        get_product_list=lambda _:products, download_products=download))
    with pytest.raises(DownloadReached):
        jwst.download(12345,5,data_product=product)
    assert selected == [expected]


@pytest.mark.parametrize('instrument,grating,mode', [('MIRI',None,'miri/photometry'),
                                                    ('NIRSPEC','PRISM','nirspec/prism')])
def test_missing_filter_mode_status(instrument,grating,mode):
    steps = NS(**{name:'COMPLETE' for name in ['dq_init','saturation','refpix','linearity',
        'dark_sub','jump','ramp_fit','superbias','reset','emicorr']})
    data = NS(meta=NS(cal_step=steps,instrument=NS(name=instrument,filter=None,
                                                grating=grating,pupil=None)))
    dataset = jwst.load.__new__(jwst.load)
    dataset.check_status(data)
    assert dataset.filter == 'none' and dataset.mode == mode


@pytest.mark.parametrize('zero_nans,profile_method,parallel', [(True,None,False), (False,None,False),
    (True,'polynomial',False), (True,'gp',False), (True,'gp',True)])
def test_stage2_schema_filenames_and_cache(tmp_path, monkeypatch, zero_nans, profile_method, parallel):
    v = simulate(ncolumns=512)
    cube = np.stack([v['data'],v['data']*1.01,v['data']*.99])
    error = np.sqrt(np.broadcast_to(v['variance'],cube.shape)).copy()
    cube[0,16,100] = np.nan
    error[0,16,100] = np.nan
    meta = NS(instrument=NS(name='NIRSPEC', filter='CLEAR', grating='PRISM', detector='NRS1',fixed_slit='S1600A1'),
              subarray=NS(name='SUB512'))
    rate = NS(data=cube, err=error, dq=np.zeros_like(cube), meta=meta)
    inputs = {'rampstep':[rate], 'ints_per_segment':[3], 'times':2450000.+np.arange(3)/86400.}
    class WCS:
        def __call__(self, x, y):
            return x*0., y*0., 1.+x/100.
    monkeypatch.setattr(jwst, 'calwebb_spec2', NS(assign_wcs_step=NS(AssignWcsStep=NS(call=lambda x:x))))
    monkeypatch.setattr(jwst, 'assign_wcs', NS(nrs_wcs_set_input=lambda *args:WCS()))
    monkeypatch.setattr(jwst, 'wcstools', NS(grid_from_bounding_box=lambda bounds:np.meshgrid(np.arange(512),np.arange(32))))
    # Trace finding is tested separately; retain spline smoothing and extraction.
    def trace(image, flags, xstart, ystart, xend, **kwargs):
        x = np.arange(min(xstart,xend),max(xstart,xend)+1)
        return x, v['centers'][x]
    monkeypatch.setattr(jwst, 'trace_spectrum', trace)
    options = {} if profile_method is None else dict(optimal_extraction=True,
        extraction_backend='python', profile_method=profile_method,
        extraction_options={'polynomial_spacing':1.})
    if parallel:
        monkeypatch.setattr(jwst, 'ray', NS(init=lambda **k:None, is_initialized=lambda:True,
            shutdown=lambda:None, remote=lambda fn:NS(remote=fn), get=lambda values:values))
        options['nthreads'] = 2
    result = jwst.stage2(inputs, zero_nans=zero_nans,scale_1f=False,
                        outputfolder=str(tmp_path),suffix='test',aperture_radius=7, **options)
    assert set(result) == {'metadata','tso','tso_err','traces','spectra','whitelight','whitelight_err'}
    assert set(result['traces']) == {'times','x','y','ycorrected','ysmoothed'}
    keys = {'times','original','original_err','corrected','corrected_err','wavelength_map','wavelengths'}
    if profile_method is not None:
        keys |= {'P','Ps','extraction_settings'}
    assert set(result['spectra']) == keys
    assert result['spectra']['original'].shape == (3,441)
    assert np.all(np.isfinite(result['tso_err']))
    assert np.isnan(rate.data[0,16,100])
    folder = tmp_path/'pipeline_outputs'
    assert (folder/'traces_test.pkl').exists()
    spectra_path = folder/'spectra_test.pkl' if profile_method is None else next(folder.glob('spectra_optimal_test_*.pkl'))
    with spectra_path.open('rb') as handle:
        cached = pickle.load(handle)
    monkeypatch.setattr(jwst, 'trace_spectrum', lambda *a,**k:pytest.fail('cache should skip tracing'))
    again = jwst.stage2(inputs, zero_nans=zero_nans,scale_1f=False,
                       outputfolder=str(tmp_path),suffix='test',aperture_radius=7, **options)
    np.testing.assert_array_equal(again['spectra']['original'],cached['original'])
    if profile_method == 'gp' and not parallel:
        changed = dict(options, gp_options={'length_scale':40.})
        jwst.stage2(inputs, zero_nans=zero_nans,scale_1f=False,
                    outputfolder=str(tmp_path),suffix='test',aperture_radius=7, **changed)
        assert len(list(folder.glob('spectra_optimal_test_gp_*.pkl'))) == 2
