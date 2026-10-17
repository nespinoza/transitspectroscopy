"""Chunked validation must preserve detector coordinates and integration times."""
import importlib.util
import io
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits


def test_raw_chunk_retains_absolute_integration_indices_and_arrays(tmp_path, monkeypatch):
    dm = pytest.importorskip('stdatamodels.jwst.datamodels')
    root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root/'validation'))
    spec = importlib.util.spec_from_file_location('miri_comparison', root/'validation/miri_ramp_compare.py')
    driver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(driver)
    shape = (6, 5, 8, 10)
    with dm.RampModel(data=np.arange(np.prod(shape), dtype=np.float32).reshape(shape)) as model:
        model.meta.instrument.name = 'MIRI'
        model.meta.exposure.type = 'MIR_IMAGE'
        model.meta.exposure.integration_start = 101
        model.meta.exposure.integration_end = 106
        model.meta.exposure.nints = 200
        model.meta.exposure.ngroups = 5
        model.meta.exposure.nframes = 1
        model.meta.exposure.group_time = .085
        model.meta.subarray.xstart = 1; model.meta.subarray.ystart = 10
        model.meta.subarray.xsize = 10; model.meta.subarray.ysize = 8
        model.groupdq = np.zeros(shape, dtype=np.uint8)
        model.groupdq[2, 3, 4, 5] = 4
        model.pixeldq = np.zeros(shape[-2:], dtype=np.uint32)
        model.pixeldq[1, 2] = 1
        model.zeroframe = np.arange(480, dtype=np.float32).reshape(6, 8, 10)
        dtype = [('integration_number', 'int32')]+[(f'int_{p}_{s}', 'float64')
            for s in ('MJD_UTC', 'BJD_TDB') for p in ('start', 'mid', 'end')]
        table = np.zeros(200, dtype=dtype)
        table['integration_number'] = np.arange(1, 201)
        table['int_mid_BJD_TDB'] = 60000+np.arange(200)*.001
        model.int_times = table
        path = tmp_path/'synthetic_uncal.fits'
        model.save(path)
        with fits.open(path, memmap=False) as hdus:
            chunk = driver.ramp_chunk(hdus, 2, 5, path.name)
        try:
            np.testing.assert_array_equal(chunk.data, model.data[2:5])
            np.testing.assert_array_equal(chunk.groupdq, model.groupdq[2:5])
            np.testing.assert_array_equal(chunk.pixeldq, model.pixeldq)
            np.testing.assert_array_equal(chunk.zeroframe, model.zeroframe[2:5])
            np.testing.assert_array_equal(chunk.int_times, table)
            assert chunk.meta.exposure.integration_start == 103
            assert chunk.meta.exposure.integration_end == 105
            assert chunk.meta.exposure.nints == 200
            assert chunk.meta.subarray.ystart == 10
            assert chunk.meta.exposure.group_time == .085
        finally:
            chunk.close()


def test_short_tails_use_context_without_changing_output_times(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root/'validation'))
    from miri_ramp_compare import calibration_start, trim_rateints
    dm = pytest.importorskip('stdatamodels.jwst.datamodels')
    assert calibration_start(3584, 3648, 128) == 3520
    assert calibration_start(3584, 3705, 128) == 3584
    with pytest.raises(ValueError, match='threshold'):
        calibration_start(0, 64, 64)
    with dm.CubeModel(data=np.ones((128, 2, 2), np.float32)) as model:
        model.err = np.ones_like(model.data)
        model.dq = np.zeros(model.data.shape, np.uint32)
        model.meta.exposure.integration_start = 3521
        model.meta.exposure.integration_end = 3648
        trim_rateints(model, 64)
        assert model.data.shape == model.err.shape == model.dq.shape == (64, 2, 2)
        assert model.meta.exposure.integration_start == 3585
        assert model.meta.exposure.integration_end == 3648


def test_http_range_reader_offsets_eof_and_server_guard(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root/'validation'))
    from miri_ramp_repair_tails import HTTPRangeReader
    from types import SimpleNamespace
    payload = bytes(range(256))*12
    class Response:
        def __init__(self, first, last, status):
            self.status_code = status
            self.content = payload[first:last+1]
            self.headers = {'Content-Range': f'bytes {first}-{last}/{len(payload)}', 'ETag': 'fixed'}
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def raise_for_status(self): pass
    class Session:
        def __init__(self, status=206): self.status = status
        def get(self, url, **kwargs):
            first, last = map(int, kwargs['headers']['Range'].removeprefix('bytes=').split('-'))
            return Response(first, last, self.status)
        def close(self): pass
    with HTTPRangeReader('raw.fits', session=Session()) as reader:
        with io.BufferedReader(reader, buffer_size=64) as f:
            assert f.read(13) == payload[:13]
            f.seek(147); assert f.read(97) == payload[147:244]
            f.seek(-6, 2); assert f.read(50) == payload[-6:]
            assert f.read(1) == b''
            f.seek(15); assert f.read(7) == payload[15:22]
    with pytest.raises(IOError, match='honor'):
        HTTPRangeReader('raw.fits', session=Session(status=200))
