"""Configuration-driven runs, standard products and provenance on synthetic JWST files."""
from types import SimpleNamespace as NS
import json
import warnings

import numpy as np
import pytest

import transitspectroscopy as ts
from transitspectroscopy import jwst
from transitspectroscopy.core import products, provenance
from simulated_spectra import simulate


def _write_rateints(folder, nrows=32, ncolumns=2048, nints=4, seed=2):
    datamodels = pytest.importorskip('stdatamodels.jwst.datamodels')
    v = simulate(nrows=nrows, ncolumns=ncolumns, seed=seed)
    data = np.stack([v['data']*(1-.01*(i in (1, 2))) for i in range(nints)]).astype('float32')
    model = datamodels.CubeModel(data=data, err=np.sqrt(np.broadcast_to(v['variance'], data.shape)).astype('float32'),
                                 dq=np.zeros(data.shape, dtype='uint32'))
    model.meta.instrument.name, model.meta.instrument.detector = 'NIRSPEC', 'NRS1'
    model.meta.instrument.grating, model.meta.instrument.filter = 'G395H', 'F290LP'
    model.meta.instrument.fixed_slit = 'S1600A1'
    model.meta.exposure.type, model.meta.exposure.nints, model.meta.exposure.ngroups = 'NRS_BRIGHTOBJ', nints, 10
    model.meta.exposure.integration_start, model.meta.exposure.integration_end = 1, nints
    model.meta.subarray.name = 'SUB2048'
    model.meta.bunit_data = 'DN/s'
    model.meta.cal_step.ramp_fit = 'COMPLETE'
    dtype = [('integration_number', 'int32')]+[(f'int_{position}_{scale}', 'float64')
             for scale in ('MJD_UTC', 'BJD_TDB') for position in ('start', 'mid', 'end')]
    table = np.zeros(nints, dtype=dtype)
    table['integration_number'] = np.arange(1, nints+1)
    table['int_mid_BJD_TDB'] = 60000.+np.arange(nints)/1440.
    model.int_times = table
    path = folder/'jw00001001001_04101_00001-seg001_nrs1_rateints.fits'
    model.save(path)
    return path, v['centers']


@pytest.fixture
def patched_stage2(monkeypatch):
    def apply(centers, ncolumns=2048, nrows=32):
        class WCS:
            def __call__(self, x, y):
                return x*0., y*0., 2.8+x/2000.
        monkeypatch.setattr(jwst, 'calwebb_spec2', NS(assign_wcs_step=NS(AssignWcsStep=NS(call=lambda x: x))))
        monkeypatch.setattr(jwst, 'assign_wcs', NS(nrs_wcs_set_input=lambda *args: WCS()))
        monkeypatch.setattr(jwst, 'wcstools', NS(grid_from_bounding_box=lambda bounds: np.meshgrid(
            np.arange(ncolumns), np.arange(nrows))))

        def trace(image, flags, xstart, ystart, xend, **kwargs):
            x = np.arange(min(xstart, xend), max(xstart, xend)+1)
            return x, centers[x]
        monkeypatch.setattr(jwst, 'trace_spectrum', trace)
    return apply


def test_run_writes_products_reuses_steps_and_recomputes_changes(tmp_path, patched_stage2):
    path, centers = _write_rateints(tmp_path)
    patched_stage2(centers)
    config = dict(run=dict(name='synthetic', output=str(tmp_path/'run')),
                  data=dict(source='local', glob=str(tmp_path/'*_rateints.fits'), product='rateints'),
                  extraction=dict(box=dict(aperture_radius=3., scale_1f=False)),
                  lightcurves=dict(resolving_power=20))
    first = ts.run(config)
    assert first.reused == [] and first.dataset.mode == 'nirspec/g395h'
    box = first.products['box']
    assert set(box) == {'nrs1_corrected', 'nrs1_original', 'nrs1_whitelight'}
    spectra = box['nrs1_corrected']
    assert spectra.flux.shape == (4, 1543) and spectra.wavelength.shape == (1543,)
    white = first.lightcurves['box']['nrs1_corrected_white']
    np.testing.assert_allclose(white.flux[0][[1, 2]]/white.flux[0][[0, 3]], .99, rtol=1e-5)  # float32 inputs
    assert len(first.lightcurves['box']['nrs1_corrected_R20'].labels) > 3
    # Products on disk round-trip, with a manifest recording inputs and configuration.
    folder = first.steps['extract']/'box'/'products'
    reread = products.read_fits(folder/'nrs1_corrected.fits')
    np.testing.assert_array_equal(reread.flux, spectra.flux)
    manifest = provenance.read_manifest(provenance.manifest_path(folder/'nrs1_corrected.fits'))
    assert manifest['input_files'][0]['name'] == path.name and len(manifest['input_files'][0]['sha256']) == 64
    assert manifest['config']['extraction']['aperture_radius'] == 3.
    # An unchanged configuration reuses every step.
    second = ts.run(config)
    assert second.reused == ['data', 'calibrate', 'extract', 'lightcurves']
    np.testing.assert_array_equal(second.products['box']['nrs1_corrected'].flux, spectra.flux)
    # Changing the extraction recomputes extraction and light curves only.
    changed = json.loads(json.dumps(config))
    changed['extraction']['box']['aperture_radius'] = 4.
    third = ts.run(changed)
    assert third.reused == ['data', 'calibrate']
    assert third.steps['extract'] != first.steps['extract']
    assert ts.run(config, from_step='lightcurves').reused == ['data', 'calibrate', 'extract']


def test_run_from_toml_and_cli(tmp_path, patched_stage2, capsys):
    path, centers = _write_rateints(tmp_path)
    patched_stage2(centers)
    toml = tmp_path/'run.toml'
    toml.write_text(f'''
[run]
name = "cli"
output = "{tmp_path/'cli'}"

[data]
source = "local"
glob = "{tmp_path/'*_rateints.fits'}"
product = "rateints"

[extraction.box]
aperture_radius = 3.0
scale_1f = false
''')
    from transitspectroscopy.cli import main
    assert main(['run', str(toml), '--no-input-hashes']) == 0
    assert 'box: nrs1_corrected, nrs1_original, nrs1_whitelight' in capsys.readouterr().out
    assert main(['modes']) == 0
    assert 'nirspec/g395h' in capsys.readouterr().out


def test_configuration_errors():
    with pytest.raises(ValueError, match='needs a \\[data\\] section'):
        ts.load_config({})
    with pytest.raises(ValueError, match='Unknown configuration sections'):
        ts.load_config(dict(data=dict(glob='x'), extract={}))
    with pytest.raises(ValueError, match='needs data.glob'):
        ts.load_config(dict(data=dict(source='local')))


def test_changed_code_warns_and_strict_raises(tmp_path):
    series = products.PhotometricTimeSeries(times=np.arange(3.), flux=np.ones(3), flux_err=np.ones(3))
    manifest = provenance.build_manifest(config=dict(a=1))
    manifest['software']['transitspectroscopy'] = '0.0.0'
    products.write_fits(series, tmp_path/'p.fits', manifest)
    with pytest.warns(UserWarning, match='produced by different code'):
        products.read_fits(tmp_path/'p.fits')
    with pytest.raises(ValueError, match='produced by different code'):
        products.read_fits(tmp_path/'p.fits', strict=True)
    current = provenance.build_manifest(config=dict(a=1))
    products.write_fits(series, tmp_path/'q.fits', current)
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        products.read_fits(tmp_path/'q.fits')
