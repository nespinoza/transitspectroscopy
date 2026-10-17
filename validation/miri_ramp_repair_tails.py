"""Repair short calibration tails of an existing paired MIRI validation.

Fetches only needed raw FITS byte ranges, preserves the original run, and
regenerates both visits' complete photometry. Unaffected cutouts are hard-linked
as immutable inputs. Use the fixed miri_ramp_compare driver for new full runs.
"""
import argparse
from copy import deepcopy
import gc
import hashlib
import io
import json
import logging
import os
from pathlib import Path
import time
import traceback

import numpy as np
from astropy.io import fits
import requests

from validate import ROOT, checkout_package, sha256
from diagnostics import write_json
from miri_validate import plots
from miri_ramp_compare import ramp_chunk, calibration_start, trim_rateints, combine_chunks, compare_products


class HTTPRangeReader(io.RawIOBase):
    """Seekable, bounded FITS reader with validated ranges and ETag pinning."""
    def __init__(self, name, session=None):
        super().__init__()
        self.name = name
        self.session = session or requests.Session()
        self.position = 0
        self.etag = None
        self.size = None
        self.ranges = []
        self._fetch(0, 0)

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        if whence not in (0, 1, 2):
            raise ValueError('Invalid seek origin')
        position = offset + (0 if whence == 0 else self.position if whence == 1 else self.size)
        if position < 0:
            raise ValueError('Negative seek position')
        self.position = position
        return position

    def _fetch(self, start, end):
        headers = {'Range': f'bytes={start}-{end}'}
        if self.etag:
            headers['If-Match'] = self.etag
        with self.session.get('https://mast.stsci.edu/api/v0.1/Download/file',
            params={'uri': 'mast:JWST/product/'+self.name}, headers=headers,
            stream=True, timeout=120) as response:
            response.raise_for_status()
            if response.status_code != 206:
                raise IOError('MAST did not honor the byte range; refusing a full-file download')
            extent = response.headers['Content-Range']
            actual_range, size = extent.removeprefix('bytes ').split('/')
            if actual_range != f'{start}-{end}':
                raise IOError('Unexpected returned byte range')
            if self.size is not None and self.size != int(size):
                raise IOError('Remote file size changed during validation')
            etag = response.headers.get('ETag')
            if self.etag is not None and etag != self.etag:
                raise IOError('Remote raw file changed during validation')
            self.size, self.etag = int(size), etag
            data = response.content
        if len(data) != end-start+1:
            raise IOError('Incomplete range download')
        self.ranges.append(dict(start=start, end=end, sha256=hashlib.sha256(data).hexdigest()))
        return data

    def read(self, size=-1):
        size = self.size-self.position if size < 0 else min(size, max(0, self.size-self.position))
        if size > 128*1024**2:
            raise IOError('Unexpected large read; use FITS sections for raw data')
        if size <= 0:
            return b''
        data = self._fetch(self.position, self.position+size-1)
        self.position += len(data)
        return data

    def readinto(self, buffer):
        data = self.read(len(buffer))
        buffer[:len(data)] = data
        return len(data)

    def close(self):
        if not self.closed:
            self.session.close()
        super().close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    source, folder = args.source.resolve(), args.output.resolve()
    folder.mkdir(parents=True, exist_ok=args.resume)
    logging.basicConfig(filename=folder/'tail_repair.log', level=logging.INFO,
                        format='%(asctime)s %(name)s %(levelname)s %(message)s')
    original = json.loads((source/'comparison_manifest.json').read_text())
    manifest = deepcopy(original)
    manifest.update(status='running', original_run=str(source),
        original_manifest_sha256=sha256(source/'comparison_manifest.json'), tail_repairs=[],
        repair_source_hashes={str(p.relative_to(ROOT)): sha256(p) for p in
            [Path(__file__), ROOT/'validation/miri_ramp_compare.py']})
    manifest['limitations'] = [
        'Shared detector calibration uses 128-integration windows; EMI and temporal jump statistics differ from full-segment calibration.',
        'Tails below 100 integrations are calibrated with 128-integration overlapping context; only their original output times are retained.',
        'SPARTA residual_correction=False; this comparison does not test the optional empirical ramp-template correction.',
        'Running-median precision is diagnostic, not an eclipse-depth error or bias measurement.']
    path = folder/'comparison_manifest.json'
    if args.resume and path.exists():
        previous = json.loads(path.read_text())
        for key in ('original_manifest_sha256', 'repair_source_hashes'):
            if previous[key] != manifest[key]:
                raise ValueError('Resume provenance mismatch: '+key)
        manifest = previous
        manifest['status'] = 'running'
    write_json(path, manifest)
    os.environ.setdefault('CRDS_PATH', str(Path.home()/'crds_cache'))
    os.environ.setdefault('CRDS_SERVER_URL', 'https://jwst-crds.stsci.edu')
    os.environ['CRDS_CONTEXT'] = original['crds_context']
    ts = checkout_package()
    from jwst.pipeline import Detector1Pipeline, Image2Pipeline
    from jwst.ramp_fitting import RampFitStep
    from jwst.gain_scale import GainScaleStep
    completed = {r['stem'] for r in manifest['tail_repairs']}
    prepared = {m: {} for m in manifest['methods']}
    started = time.monotonic()
    try:
        for index, row in enumerate(manifest['chunks']):
            stem, first, last = row['stem'], row['start'], row['stop']
            filename = f'{stem}_{first:05d}.npz'
            for method in prepared:
                out = folder/method/'prepared'/filename
                out.parent.mkdir(parents=True, exist_ok=True)
                prepared[method].setdefault(stem, []).append(out)
                if last-first >= 100 and not out.exists():
                    before = source/method/'prepared'/filename
                    if sha256(before) != row['methods'][method]['prepared_sha256']:
                        raise ValueError('Source cutout hash mismatch')
                    os.link(before, out)
            if last-first >= 100 or stem in completed:
                continue
            name = stem+'_uncal.fits'
            context = calibration_start(first, last, manifest['chunk_integrations'])
            print(f'Repairing {stem}: {last-first} outputs with {last-context} calibration integrations', flush=True)
            with HTTPRangeReader(name) as remote:
                with io.BufferedReader(remote, buffer_size=64*1024) as buffered:
                    with fits.open(buffered, memmap=False) as hdus:
                        raw = ramp_chunk(hdus, context, last, name)
                    remote_record = dict(name=name, etag=remote.etag, size=remote.size,
                                         ranges=remote.ranges, downloaded_bytes=sum(x['end']-x['start']+1 for x in remote.ranges))
            opts = deepcopy(manifest['config']['detector_options'])
            opts['ramp_fit'], opts['gain_scale'] = {'skip': True}, {'skip': True}
            tick = time.monotonic()
            calibrated = Detector1Pipeline.call(raw, steps=opts, save_results=False, save_calibrated_ramp=False)
            raw.close(); del raw; gc.collect()
            updated = deepcopy(row)
            updated['calibration_start'] = context
            updated['detector_seconds'] = time.monotonic()-tick
            updated['detector_steps'] = deepcopy(calibrated.meta.cal_step.instance)
            updated['methods'] = {}
            for method, options in manifest['methods'].items():
                tick = time.monotonic()
                if method == 'jwst':
                    image, rate = RampFitStep.call(calibrated, maximum_cores='1', save_results=False)
                    diagnostics = None
                else:
                    image, rate, diagnostics = ts.ramp_fitting.fit_model(calibrated, **options)
                image.close(); del image
                trim_rateints(rate, first-context)
                info = dict(fit_seconds=time.monotonic()-tick,
                    valid_rate_pixels=int(np.count_nonzero(np.isfinite(rate.data) & ((rate.dq & 1)==0))),
                    median_rate_error=float(np.nanmedian(rate.err)))
                if diagnostics is not None:
                    info.update(rejected_differences=int(diagnostics['n_rejected'][first-context:].sum()),
                        nonconverged_ramps=int(np.count_nonzero(~diagnostics['converged'][first-context:])),
                        provenance=diagnostics['provenance'])
                del diagnostics
                scaled = GainScaleStep.call(rate, save_results=False)
                rate.close(); del rate
                opts = deepcopy(manifest['config']['image_options'])
                opts.setdefault('resample', {})['skip'] = True
                products = Image2Pipeline.call(scaled, steps=opts, save_results=False)
                scaled.close(); del scaled
                model = products[0]
                segment = ts.miri.prepare_segment(model, **manifest['config']['preparation'])
                old = ts.miri.load_prepared(source/method/'prepared'/filename)
                np.testing.assert_array_equal(segment['times'], old['times'])
                np.testing.assert_array_equal(segment['origin'], old['origin'])
                out = folder/method/'prepared'/filename
                ts.miri.save_prepared(segment, out)
                info['prepared_sha256'] = sha256(out)
                info['fit_calibrate_prepare_seconds'] = time.monotonic()-tick
                updated['methods'][method] = info
                sample = trim_rateints(model.copy(), len(model.data)-1)
                sample.save(folder/method/f'{stem}_tail_calints.fits')
                sample.close(); model.close()
                del sample, model, products, segment, old
                gc.collect()
            calibrated.close(); del calibrated; gc.collect()
            manifest['chunks'][index] = updated
            manifest['tail_repairs'].append(dict(stem=stem, original_start=first, stop=last,
                calibration_start=context, original_chunk=row, raw_ranges=remote_record))
            write_json(path, manifest)
        results = {}
        for visit in manifest['config']['visits']:
            results[visit] = {}
            for method in prepared:
                selected = {stem: files for stem, files in prepared[method].items()
                    if any(stem.startswith(e+'-seg') for e in manifest['config']['visits'][visit]['exposures'])}
                segments = combine_chunks(ts, selected)
                result = ts.miri.extract_visit(segments, outputfolder=folder/method/f'visit{visit}', **manifest['config']['extraction'])
                write_json(folder/method/f'visit{visit}'/'metrics.json', plots(result, manifest['config'], folder/method/f'visit{visit}', visit))
                results[visit][method] = {k: result[k] for k in ('times','flux','error','segment')}
                del segments, result; gc.collect()
        manifest['comparison'] = compare_products(folder, results, manifest['config'])
        manifest['status'] = 'complete'
    except BaseException as error:
        manifest['status'] = 'failed'
        manifest['error'] = dict(type=type(error).__name__, message=str(error), traceback=traceback.format_exc())
        raise
    finally:
        manifest['repair_session_seconds'] = time.monotonic()-started
        write_json(path, manifest)
    print('Corrected comparison:', folder, flush=True)


if __name__ == '__main__':
    main()
