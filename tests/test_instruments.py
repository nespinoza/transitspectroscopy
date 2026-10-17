"""Observing-mode registry, ExtractionConfig and mode-driven Stage 2 dispatch."""
from types import SimpleNamespace as NS
import warnings

import numpy as np
import pytest

from transitspectroscopy import instruments, jwst, miri, soss
from transitspectroscopy.config import ExtractionConfig
from simulated_spectra import simulate


def test_builtin_modes_are_registered_and_identified():
    assert instruments.list_modes() == ['miri/photometry', 'niriss/soss', 'nirspec/g395h',
                                        'nirspec/g395m', 'nirspec/prism']
    cases = [(dict(instrument='NIRSPEC', dispersive_element='PRISM'), 'nirspec/prism'),
             (dict(instrument='nirspec', dispersive_element='g395h'), 'nirspec/g395h'),
             (dict(instrument='NIRSPEC', dispersive_element='G395M'), 'nirspec/g395m'),
             (dict(instrument='NIRISS', dispersive_element='GR700XD'), 'niriss/soss'),
             (dict(exp_type='NIS_SOSS'), 'niriss/soss'),
             (dict(instrument='MIRI', dispersive_element='None'), 'miri/photometry'),
             (dict(instrument='MIRI', exp_type='MIR_IMAGE'), 'miri/photometry')]
    for properties, name in cases:
        assert instruments.identify(**properties).name == name
    assert instruments.get_mode('miri/imaging') is instruments.get_mode('miri/photometry')


def test_unknown_mode_lists_registered_modes():
    with pytest.raises(instruments.UnknownModeError, match='Registered modes: miri/photometry'):
        instruments.identify(instrument='NIRCAM', dispersive_element='GRISMR')
    assert issubclass(instruments.UnknownModeError, ValueError)


def test_mode_defaults_are_the_historical_values():
    prism = instruments.get_mode('nirspec/prism')
    assert prism.detector_steps['refpix'] == 'side_pedestal'
    assert prism.detector_steps['refpix_parameters'] == dict(left_pixels=25, right_pixels=25)
    assert prism.detector_steps['tso_jump'] == dict(window=200, nsigma=10)
    assert prism.tracing_for('NRS1')['nknots'] == 8
    g395h = instruments.get_mode('nirspec/g395h')
    assert g395h.tracing_for('nrs1')['xstart'] == 2042 and g395h.tracing_for('nrs2')['xend'] == 2043
    assert g395h.extraction_defaults['aperture_radius'] == 2
    assert instruments.get_mode('nirspec/g395m').tracing_for('nrs1')['xend'] == 700
    assert instruments.get_mode('miri/photometry').detector_steps['post_linearity_steps'] == ('dark_sub', 'refpix')
    with pytest.raises(instruments.UnknownModeError, match='Detector NRS3'):
        g395h.tracing_for('NRS3')


def _patch_spectral_stage2(monkeypatch, centers, ncolumns, nrows):
    class WCS:
        def __call__(self, x, y):
            return x*0., y*0., 1.+x/1000.
    monkeypatch.setattr(jwst, 'calwebb_spec2', NS(assign_wcs_step=NS(AssignWcsStep=NS(call=lambda x: x))))
    monkeypatch.setattr(jwst, 'assign_wcs', NS(nrs_wcs_set_input=lambda *args: WCS()))
    monkeypatch.setattr(jwst, 'wcstools', NS(grid_from_bounding_box=lambda bounds: np.meshgrid(
        np.arange(ncolumns), np.arange(nrows))))

    def trace(image, flags, xstart, ystart, xend, **kwargs):
        x = np.arange(min(xstart, xend), max(xstart, xend)+1)
        return x, centers[x]
    monkeypatch.setattr(jwst, 'trace_spectrum', trace)


@pytest.mark.parametrize('grating,detector', [('PRISM', 'NRS1'), ('G395H', 'NRS1'), ('G395H', 'NRS2'),
                                              ('G395M', 'NRS1'), ('G395M', 'NRS2')])
def test_spectral_stage2_runs_for_every_spectral_mode(tmp_path, monkeypatch, grating, detector):
    nrows, ncolumns = 32, 2048
    v = simulate(nrows=nrows, ncolumns=ncolumns, seed=3)
    cube = np.stack([v['data'], v['data']*1.01, v['data']*.99])
    error = np.sqrt(np.broadcast_to(v['variance'], cube.shape)).copy()
    meta = NS(instrument=NS(name='NIRSPEC', filter='F290LP', grating=grating, detector=detector,
                            fixed_slit='S1600A1'), subarray=NS(name='SUB2048'))
    rate = NS(data=cube, err=error, dq=np.zeros_like(cube), meta=meta)
    inputs = {'rampstep': [rate], 'ints_per_segment': [3], 'times': 2450000.+np.arange(3)/86400.}
    _patch_spectral_stage2(monkeypatch, v['centers'], ncolumns, nrows)
    with warnings.catch_warnings():
        warnings.simplefilter('error', DeprecationWarning)
        result = jwst.stage2(inputs, outputfolder=str(tmp_path), suffix='t',
                             config=ExtractionConfig(aperture_radius=4, scale_1f=False))
    tracing = instruments.identify(instrument='nirspec', dispersive_element=grating).tracing_for(detector)
    assert result['traces']['x'][0] == min(tracing['xstart'], tracing['xend'])
    assert result['spectra']['original'].shape == (3, abs(tracing['xend']-tracing['xstart'])+1)
    expected = 'traces_t.pkl' if grating == 'PRISM' else f'traces_t_{detector.lower()}.pkl'
    assert (tmp_path/'pipeline_outputs'/expected).exists()


def test_soss_and_miri_dispatch_through_config_and_keywords(monkeypatch):
    calls = {}
    monkeypatch.setattr(soss, 'extract_soss', lambda data, **kw: calls.setdefault('soss', []).append(kw) or {})
    monkeypatch.setattr(miri, 'extract_photometry', lambda data, **kw: calls.setdefault('miri', []).append(kw) or {})
    soss_inputs = dict(rampstep=[NS(meta=NS(exposure=NS(type='NIS_SOSS')))])
    with pytest.warns(DeprecationWarning, match='soss_options'):
        jwst.stage2(soss_inputs, soss_options={'trace_max_shift': 3.})
    jwst.stage2(soss_inputs, config=ExtractionConfig(soss={'trace_max_shift': 3.}))
    assert calls['soss'][0] == calls['soss'][1]
    assert calls['soss'][0]['aperture_radius'] == 15.
    miri_inputs = dict(rampstep=[NS(meta=NS(instrument=NS(name='MIRI', grating=None, pupil=None),
                                            exposure=NS(type='MIR_IMAGE')))])
    jwst.stage2(miri_inputs, aperture_radius=4.)
    jwst.stage2(miri_inputs, config=ExtractionConfig(aperture_radius=4.))
    assert calls['miri'][0] == calls['miri'][1]
    assert calls['miri'][0]['preparation_options'] == {'aperture_radius': 4.}
    with pytest.raises(ValueError, match='MIRI imaging uses miri_options'):
        jwst.stage2(miri_inputs, config=ExtractionConfig(soss={}))
    with pytest.raises(ValueError, match='either in config or as keywords'):
        jwst.stage2(miri_inputs, aperture_radius=3., config=ExtractionConfig())


def test_a_new_mode_needs_only_a_registry_entry():
    def handler(input_dictionary, config, outputfolder, suffix):
        return dict(handled=True, aperture=config.aperture_radius, suffix=suffix)
    mode = instruments.Mode(name='test/plugin', observatory='TEST', instrument='plug', label='Test/Plugin',
                            kind='imaging', match=({'instrument': 'plug'},), stage2=handler)
    instruments.register(mode)
    try:
        inputs = dict(rampstep=[NS(meta=NS(instrument=NS(name='PLUG', grating=None, pupil=None)))])
        assert jwst.stage2(inputs, suffix='x', config=ExtractionConfig(aperture_radius=3.)) == dict(
            handled=True, aperture=3., suffix='x')
        with pytest.raises(ValueError, match='already registered'):
            instruments.register(mode)
    finally:
        instruments.unregister('test/plugin')
    assert 'test/plugin' not in instruments.list_modes()


def test_extraction_config_from_keywords_and_shared_profile_defaults():
    config = ExtractionConfig.from_stage2_arguments(optimal_extraction=True, extraction_backend='python',
                                                   extraction_options={'polynomial_order': 4})
    assert config.backend == 'python' and config.polynomial_order == 4
    assert config.extraction_options() == {'polynomial_order': 4}
    assert set(config._deprecated) == {'extraction_backend', 'extraction_options'}
    with pytest.raises(ValueError, match='Unknown extraction_options key'):
        ExtractionConfig.from_stage2_arguments(extraction_options={'spacing': 1.})
    spectral = instruments.get_mode('nirspec/g395h')
    with pytest.raises(ValueError, match='GP requires optimal extraction'):
        ExtractionConfig(profile_method='gp').validate(spectral)
    with pytest.raises(ValueError, match='shared_profile_options requires shared_profile'):
        ExtractionConfig(shared_profile_options={}).validate(spectral)
