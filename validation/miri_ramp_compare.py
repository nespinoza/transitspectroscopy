"""Paired raw-ramp MIRI validation with bounded memory and resumable cutouts.

Both fitters see the identical calibrated ramp in each integration chunk.
The optional empirical residual-template correction is disabled to isolate
correlated ramp fitting without learning an astrophysical time-series signal.
"""
import argparse
from copy import deepcopy
import gc
import hashlib
import importlib.metadata
import json
import logging
import os
from pathlib import Path
import time
import traceback
import warnings

import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt

from validate import ROOT, checkout_package, sha256
from diagnostics import write_json, smooth_series
from miri_validate import download, inspect_input, plots, bin_series


def calibration_start(first, last, chunk_size, minimum_sigclip_groups=100):
    """Pad short tails backwards so JWST does not change jump-detection mode."""
    if chunk_size < minimum_sigclip_groups:
        raise ValueError('Chunk size must reach the integration sigma-clip threshold')
    return max(0, last-chunk_size) if last-first < minimum_sigclip_groups else first


def trim_rateints(model, offset):
    """Remove calibration context while keeping absolute INT_TIMES indexing."""
    if offset:
        for key in ('data', 'err', 'dq', 'var_poisson', 'var_rnoise', 'var_flat'):
            array = getattr(model, key, None)
            if array is not None and array.ndim == 3:
                setattr(model, key, array[offset:].copy())
        model.meta.exposure.integration_start += offset
    return model


def ramp_chunk(hdus, start, stop, filename):
    """Rebuild an uncal RampModel from only the requested integrations.

    Raw FITS metadata is retained, including the complete INT_TIMES table.
    ASDF is excluded because its external array references describe the original
    uncut file; no assigned WCS or calibrated ASDF metadata exist at this stage.
    """
    from stdatamodels.jwst import datamodels
    count = hdus['SCI'].shape[0]
    primary = hdus[0].header.copy()
    original_start = primary['INTSTART']
    primary['INTSTART'], primary['INTEND'] = original_start+start, original_start+stop-1
    output = [fits.PrimaryHDU(header=primary)]
    for hdu in hdus[1:]:
        if hdu.name == 'ASDF':
            continue
        if isinstance(hdu, (fits.ImageHDU, fits.CompImageHDU)):
            if len(hdu.shape) >= 3 and hdu.shape[0] == count:
                values = hdu.section[start:stop].copy()
            else:
                values = None if hdu.data is None else hdu.data.copy()
            output.append(fits.ImageHDU(values, header=hdu.header.copy()))
        else:
            output.append(hdu.copy())
    with fits.HDUList(output) as cut:
        model = datamodels.RampModel(cut)
        # Detach arrays from the HDUList before closing it.
        model = model.copy()
    model.meta.filename = filename
    assert model.data.shape[0] == stop-start
    return model


def combine_chunks(ts, paths):
    segments = []
    for stem, files in paths.items():
        pieces = [ts.miri.load_prepared(p) for p in files]
        first = pieces[0]
        if any(not np.array_equal(p['origin'], first['origin']) for p in pieces):
            raise ValueError(f'{stem}: stamp origin changed between chunks; align stamps before concatenation')
        for key in ('data', 'variance', 'centers', 'widths', 'sky', 'sky_variance',
                    'sky_pixels', 'flags', 'times'):
            first[key] = np.concatenate([p[key] for p in pieces])
        # Preserve the original physical-segment boundaries in filtering/masks.
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)  # permanently masked detector pixels
            first['median_image'] = np.nanmedian([p['median_image'] for p in pieces], axis=0)
        segments.append(first)
    return segments


def compare_products(folder, results, config):
    report = {}
    for visit, pair in results.items():
        a, b = pair['jwst'], pair['sparta']
        np.testing.assert_array_equal(a['times'], b['times'])
        hours = (a['times']-a['times'][0])*24
        fig, axes = plt.subplots(3, 2, figsize=(15, 10), sharex='col', layout='constrained')
        metrics = {}
        for j, method in enumerate(('sum', 'gp')):
            common = (np.isfinite(a['flux'][:, j]) & np.isfinite(b['flux'][:, j]) &
                      np.isfinite(a['error'][:, j]) & np.isfinite(b['error'][:, j]))
            residuals, normalized = {}, {}
            entry = {'common_integrations': int(common.sum())}
            for name, result in pair.items():
                scale = np.median(result['flux'][common, j])
                f, e = result['flux'][:, j]/scale, result['error'][:, j]/scale
                smooth = np.full(len(f), np.nan)
                for seg in np.unique(result['segment']):
                    select = result['segment'] == seg
                    smooth[select] = smooth_series(f[select], config['white_light_filter_points'])
                residual = (f-smooth)*1.e6
                residuals[name], normalized[name] = residual, f
                entry[name] = dict(rms_ppm=float(np.sqrt(np.mean(residual[common]**2))),
                                  median_error_ppm=float(np.median(e[common])*1.e6),
                                  median_flux=float(scale), valid_integrations=int(np.isfinite(f).sum()))
                bins = bin_series(result['times'][common], f[common], e[common], config['bin_seconds'])
                axes[0, j].errorbar((bins[:, 0]-a['times'][0])*24, bins[:, 1], yerr=bins[:, 2],
                                    fmt='.', ms=3, alpha=.6, label=name)
                axes[1, j].plot(hours[common], residual[common], '.', ms=.8, alpha=.35,
                               label=f'{name}: {entry[name]["rms_ppm"]:.1f} ppm')
            delta = (normalized['sparta']-normalized['jwst'])*1.e6
            axes[2, j].plot(hours[common], delta[common], '.', ms=1)
            axes[0, j].set(title=f'{method}: {config["bin_seconds"]:g}-s bins', ylabel='Relative flux')
            axes[1, j].set(ylabel='Running-median residual (ppm)')
            axes[2, j].set(ylabel='SPARTA − JWST (ppm)', xlabel='Hours since first integration')
            for axis in axes[:2, j]:
                axis.legend()
            entry['rms_change_percent'] = 100*(entry['sparta']['rms_ppm']/entry['jwst']['rms_ppm']-1)
            entry['paired_difference_rms_ppm'] = float(np.sqrt(np.mean(delta[common]**2)))
            np.savetxt(folder/f'visit{visit}_{method}_paired.csv', np.column_stack([
                a['times'], common, normalized['jwst'], normalized['sparta'],
                residuals['jwst'], residuals['sparta'], delta]), delimiter=',',
                header='time_bjd_tdb,common,jwst_relative_flux,sparta_relative_flux,jwst_residual_ppm,sparta_residual_ppm,difference_ppm', comments='')
            metrics[method] = entry
        fig.savefig(folder/f'visit{visit}_ramp_comparison.png', dpi=150)
        plt.close(fig)
        report[visit] = metrics
    write_json(folder/'comparison.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run', type=Path, default=ROOT/'validation/runs/miri_9234_validation_20260927_reviewed')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--chunk-integrations', type=int, default=128)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--max-chunks', type=int, help='Development smoke check only; report remains incomplete')
    args = parser.parse_args()
    folder = args.output.resolve()
    folder.mkdir(parents=True, exist_ok=args.resume)
    logging.basicConfig(filename=folder/'pipeline.log', level=logging.INFO,
                        format='%(asctime)s %(name)s %(levelname)s %(message)s')
    source = json.loads((args.source_run/'reduction.json').read_text())
    config = deepcopy(source['config'])
    minimum_groups = config['detector_options'].get('jump', {}).get('minimum_sigclip_groups', 100)
    if args.chunk_integrations < minimum_groups:
        raise ValueError('Use a chunk size at least as large as minimum_sigclip_groups')
    os.environ.setdefault('CRDS_PATH', str(Path.home()/'crds_cache'))
    os.environ.setdefault('CRDS_SERVER_URL', 'https://jwst-crds.stsci.edu')
    os.environ.setdefault('CRDS_CONTEXT', 'jwst_1584.pmap')
    ts = checkout_package()
    from jwst.pipeline import Detector1Pipeline, Image2Pipeline
    from jwst.ramp_fitting import RampFitStep
    from jwst.gain_scale import GainScaleStep
    methods = {'jwst': {}, 'sparta': {'residual_correction': False, 'sigma': 5.}}
    manifest_path = folder/'comparison_manifest.json'
    manifest = dict(status='running', config=config, methods=methods,
                    source_run=str(args.source_run.resolve()),
                    source_manifest_sha256=sha256(args.source_run/'reduction.json'),
                    chunk_integrations=args.chunk_integrations, crds_context=os.environ['CRDS_CONTEXT'],
                    software={k: importlib.metadata.version(k) for k in ('jwst','stcal','stdatamodels','numpy','scipy')},
                    source_hashes={str(p.relative_to(ROOT)): sha256(p) for p in
                        [ROOT/'src/ramp_fitting.py', ROOT/'src/miri.py', ROOT/'src/photometry.py', Path(__file__)]},
                    inputs=[], chunks=[], limitations=[
                        'Detector calibration is chunked identically for both fitters; shared EMI and jump statistics may differ from a full-segment Detector1 run. Short tails receive overlapping context to retain the same jump-detection mode.',
                        'Empirical residual-template correction is disabled to isolate covariance weighting and avoid shared temporal training.',
                        'RMS is a running-median diagnostic, not an eclipse-depth uncertainty or a bias test.'])
    if args.resume and manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        for key in ('config','methods','source_manifest_sha256','chunk_integrations','crds_context','source_hashes'):
            if previous[key] != manifest[key]:
                raise ValueError(f'Resume mismatch: {key}')
        manifest = previous
        manifest['status'] = 'running'
    write_json(manifest_path, manifest)
    start_time = time.monotonic()
    temporary = folder/'temporary_raw'; temporary.mkdir(exist_ok=True)
    completed = {(r['stem'], r['start']): r for r in manifest['chunks']}
    prepared = {method: {} for method in methods}
    new_chunks = 0
    try:
        for original in source['inputs']:
            name = Path(original['path']).name.removesuffix('.gz').replace('_rateints.fits', '_uncal.fits')
            stem = name.removesuffix('_uncal.fits')
            nints = original['shape'][0]
            for method in methods:
                prepared[method][stem] = [folder/method/'prepared'/f'{stem}_{i:05d}.npz'
                                         for i in range(0, nints, args.chunk_integrations)]
            if all((stem, i) in completed for i in range(0, nints, args.chunk_integrations)):
                for method in methods:
                    for path in prepared[method][stem]:
                        if not path.is_file():
                            raise FileNotFoundError(path)
                continue
            path, created = download(name, temporary, compress=True)
            record = inspect_input(path)
            if not any(r['sha256'] == record['sha256'] for r in manifest['inputs']):
                manifest['inputs'].append(record)
                write_json(manifest_path, manifest)
            with fits.open(path, memmap=False) as hdus:
                if hdus['SCI'].shape[0] != nints:
                    raise ValueError('Raw/archive integration count mismatch')
                for first in range(0, nints, args.chunk_integrations):
                    if (stem, first) in completed:
                        continue
                    last = min(first+args.chunk_integrations, nints)
                    print(f'{stem}: integrations {first+1}–{last}/{nints}', flush=True)
                    chunk_start = time.monotonic()
                    context_start = calibration_start(first, last, args.chunk_integrations, minimum_groups)
                    raw = ramp_chunk(hdus, context_start, last, name)
                    opts = deepcopy(config['detector_options'])
                    opts['ramp_fit'] = {'skip': True}
                    opts['gain_scale'] = {'skip': True}
                    calibrated = Detector1Pipeline.call(raw, steps=opts, save_results=False,
                        save_calibrated_ramp=False)
                    raw.close(); del raw; gc.collect()
                    row = dict(stem=stem, start=first, stop=last, calibration_start=context_start,
                               detector_seconds=time.monotonic()-chunk_start,
                               detector_steps=deepcopy(calibrated.meta.cal_step.instance), methods={})
                    for method, options in methods.items():
                        tick = time.monotonic()
                        if method == 'jwst':
                            image, rate = RampFitStep.call(calibrated, maximum_cores='1', save_results=False)
                            diagnostics = None
                        else:
                            image, rate, diagnostics = ts.ramp_fitting.fit_model(calibrated, **options)
                        image.close(); del image
                        trim_rateints(rate, first-context_start)
                        if diagnostics is not None:
                            for key in ('n_used', 'n_rejected', 'converged'):
                                diagnostics[key] = diagnostics[key][first-context_start:]
                        info = dict(fit_seconds=time.monotonic()-tick,
                                    valid_rate_pixels=int(np.count_nonzero(np.isfinite(rate.data) & ((rate.dq & 1)==0))),
                                    median_rate_error=float(np.nanmedian(rate.err)))
                        if diagnostics is not None:
                            info.update(rejected_differences=int(diagnostics['n_rejected'].sum()),
                                        nonconverged_ramps=int(np.count_nonzero(~diagnostics['converged'])),
                                        provenance=diagnostics['provenance'])
                        scaled = GainScaleStep.call(rate, save_results=False)
                        rate.close(); del rate; del diagnostics
                        image_options = deepcopy(config['image_options'])
                        image_options.setdefault('resample', {})['skip'] = True
                        products = Image2Pipeline.call(scaled, steps=image_options, save_results=False)
                        scaled.close(); del scaled
                        model = products[0]
                        s = ts.miri.prepare_segment(model, **config['preparation'])
                        out = folder/method/'prepared'/f'{stem}_{first:05d}.npz'
                        out.parent.mkdir(parents=True, exist_ok=True)
                        ts.miri.save_prepared(s, out)
                        info['prepared_sha256'] = sha256(out)
                        if first == 0:
                            # Compact real calibrated datamodel sample, retaining errors/DQ/times.
                            sample = model.copy()
                            for key in ('data','err','dq','var_poisson','var_rnoise','var_flat'):
                                arr = getattr(sample, key, None)
                                if arr is not None and arr.ndim == 3:
                                    setattr(sample, key, arr[:1].copy())
                            sample.meta.exposure.integration_end = sample.meta.exposure.integration_start
                            sample.save(out.parent.parent/f'{stem}_sample_calints.fits')
                            sample.close(); del sample
                        model.close(); del model, products, s
                        info['fit_calibrate_prepare_seconds'] = time.monotonic()-tick
                        row['methods'][method] = info
                        gc.collect()
                    calibrated.close(); del calibrated; gc.collect()
                    manifest['chunks'].append(row)
                    completed[(stem, first)] = row
                    write_json(manifest_path, manifest)
                    new_chunks += 1
                    if args.max_chunks and new_chunks >= args.max_chunks:
                        manifest['status'] = 'partial_smoke_test'
                        return
            # Only raw downloads created inside this run's private scratch folder.
            path.unlink()
        results = {}
        for visit in config['visits']:
            results[visit] = {}
            for method in methods:
                selected = {stem: paths for stem, paths in prepared[method].items()
                            if any(stem.startswith(e+'-seg') for e in config['visits'][visit]['exposures'])}
                segments = combine_chunks(ts, selected)
                result = ts.miri.extract_visit(segments, outputfolder=folder/method/f'visit{visit}', **config['extraction'])
                summary = plots(result, config, folder/method/f'visit{visit}', visit)
                write_json(folder/method/f'visit{visit}'/'metrics.json', summary)
                results[visit][method] = {k: result[k] for k in ('times','flux','error','segment')}
                del segments, result; gc.collect()
        manifest['comparison'] = compare_products(folder, results, config)
        manifest['status'] = 'complete'
    except BaseException as error:
        manifest['status'] = 'failed'
        manifest['error'] = dict(type=type(error).__name__, message=str(error), traceback=traceback.format_exc())
        raise
    finally:
        manifest['session_elapsed_seconds'] = time.monotonic()-start_time
        write_json(manifest_path, manifest)
    print('Completed comparison:', folder, flush=True)


if __name__ == '__main__':
    main()
