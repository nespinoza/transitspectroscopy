"""NRS1/G395H validation: python validation/validate.py --download --segments 1."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import gc
import glob
import hashlib
import importlib.metadata
import importlib.util
import json
import logging
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import traceback
import warnings

import numpy as np
from astropy.io import fits
from diagnostics import (detector_plots, extraction_diagnostics, ramp_residuals,
                         selected_integrations, stats, trace_diagnostics, write_json)

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(8*1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def checkout_package():
    source = ROOT/'src'
    spec = importlib.util.spec_from_file_location('transitspectroscopy', source/'__init__.py',
                                                 submodule_search_locations=[str(source)])
    package = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = package
    spec.loader.exec_module(package)
    return package


def download_segments(exposure, segments, folder):
    """Direct public downloads avoid the full MAST catalogue query."""
    import requests
    folder.mkdir(parents=True, exist_ok=True)
    paths = []
    for segment in segments:
        name = f'{exposure}-seg{segment:03d}_nrs1_uncal.fits'
        path = folder/name
        if not path.exists():
            print('Downloading', name, flush=True)
            temporary = path.with_suffix('.fits.part')
            with requests.get('https://mast.stsci.edu/api/v0.1/Download/file',
                    params={'uri': 'mast:JWST/product/'+name}, stream=True, timeout=120) as response:
                response.raise_for_status()
                with temporary.open('wb') as handle:
                    for chunk in response.iter_content(4*1024*1024):
                        handle.write(chunk)
                expected = response.headers.get('Content-Length')
                if expected is not None and temporary.stat().st_size != int(expected):
                    raise IOError(f'Incomplete download: {name}')
            with fits.open(temporary) as hdus:
                hdus.verify('exception')
            temporary.replace(path)
        paths.append(path.resolve())
    return paths


def inspect_inputs(paths):
    records = []
    for path in paths:
        with fits.open(path, memmap=True) as hdus:
            h = hdus[0].header
            shape = hdus['SCI'].shape
            if (h.get('INSTRUME'), h.get('DETECTOR'), h.get('GRATING')) != ('NIRSPEC', 'NRS1', 'G395H'):
                raise ValueError('This preset requires NIRSpec NRS1/G395H data')
            if len(shape) != 4:
                raise ValueError(f'{path}: expected 4-D uncalibrated ramps')
            records.append(dict(path=str(path), size_bytes=path.stat().st_size, sha256=sha256(path),
                shape=list(shape), integration_start=h['INTSTART'], integration_end=h['INTEND'],
                exposure_nints=h['NINTS'], exposure_start_mjd=h['EXPSTART'],
                target=h.get('TARGNAME'), group_time_s=h['TGROUP']))
    records.sort(key=lambda row: row['integration_start'])
    for previous, current in zip(records, records[1:]):
        if current['integration_start'] <= previous['integration_end']:
            raise ValueError('Input integration ranges overlap')
        if current['exposure_start_mjd'] != previous['exposure_start_mjd']:
            raise ValueError('Select segments of one exposure')
    return records


def save_sample_table(samples, folder, pixels):
    values = []
    for sample in samples:
        for p, (x, y) in enumerate(pixels):
            for g, elapsed in enumerate(sample['group_times_s']):
                values.append([sample['loaded_index'], sample['integration_number'], x, y, g+1,
                    elapsed, sample['groups'][g, p], sample['fitted'][g, p],
                    sample['residuals'][g, p], sample['valid'][g, p], sample['groupdq'][g, p],
                    sample['slopes'][p]])
    np.savetxt(folder/'ramp_samples.csv', values, delimiter=',', header=
        'loaded_index_zero_based,integration_number,column,row,group_number,time_seconds,'
        'signal_dn,fitted_dn,residual_dn,valid,groupdq,rate_dn_per_second', comments='')
    np.savez_compressed(folder/'ramp_samples.npz', **{
        f'sample_{i}_{key}': value for i, sample in enumerate(samples)
        for key, value in sample.items()})


def reduce(records, config, folder, report, ts):
    nints = sum(record['shape'][0] for record in records)
    selected = selected_integrations(nints)
    report['selection'] = dict(loaded_integrations=nints, selected_loaded_indices=selected,
        coordinate_convention='zero-based (column, row)', short_series_clamped=nints < 20)
    report['stages'] = {'detector_calibration': [], 'ramp_fitting': []}
    samples_by_index, rate_models, all_times = {}, [], []
    detector_rows, rate_rows = [], []
    start = 0
    try:
        for record in records:
            path = Path(record['path'])
            print('Calibrating', path.name, flush=True)
            tick = time.monotonic()
            dataset = ts.jwst.load([str(path)], outputfolder=str(folder))
            if len(dataset.times) != dataset.nints:
                raise ValueError('Timestamps do not match loaded integrations')
            all_times.extend(dataset.times)
            dataset.detector_calibration(parameters=config['detector_parameters'], save=False)
            ramp = dataset.ramps_per_segment[0]
            if len(getattr(ramp.meta.exposure, 'read_times', None) or []):
                raise ValueError('Uneven group timing is not supported by this preset')
            group_times = (np.arange(ramp.data.shape[1])+1)*ramp.meta.exposure.group_time
            for x, y in config['pixels_xy']:
                if not (0 <= x < ramp.data.shape[3] and 0 <= y < ramp.data.shape[2]):
                    raise ValueError(f'Pixel {(x, y)} lies outside the subarray')
            px, py = np.array(config['pixels_xy']).T
            for i in range(dataset.nints):
                image = ramp.data[i, -1]-ramp.data[i, 0]
                good = ((ramp.pixeldq & 1) == 0) & ((ramp.groupdq[i, -1] & 3) == 0) & ((ramp.groupdq[i, 0] & 3) == 0)
                detector_rows.append([record['integration_start']+i, dataset.times[i],
                                      stats(np.where(good, image, np.nan))['median']])
                if start+i in selected:
                    samples_by_index[start+i] = dict(loaded_index=start+i,
                        integration_number=record['integration_start']+i,
                        difference_image=image.copy(), group_times_s=group_times.copy(),
                        groups=ramp.data[i][:, py, px].copy(), groupdq=ramp.groupdq[i][:, py, px].copy())
            # The custom TSO-jump step already saves this model even with save=False.
            ramp_path = folder/'ts_outputs'/f'{dataset.datanames[0]}_tsojumpstep.fits'
            if not ramp_path.exists():
                ramp_path = folder/'ts_outputs'/f'{dataset.datanames[0]}_calibrated_ramp.fits'
                ramp.save(ramp_path)
            report['stages']['detector_calibration'].append(dict(
                input=path.name, elapsed_seconds=time.monotonic()-tick,
                datamodel=str(ramp_path.relative_to(folder)), status=dict(dataset.status),
                parameters=dataset.calibration_parameters.copy(), metadata=ramp.meta.instance.copy()))
            write_json(folder/'reduction.json', report)
            tick = time.monotonic()
            dataset.fit_ramps(parameters={'ramp_fit': config['ramp_fit_parameters']}, save=False)
            rate = dataset.rateints_per_segment[0]
            rate_path = folder/'ts_outputs'/f'{dataset.datanames[0]}_rateints.fits'
            rate.save(rate_path)
            for i in range(dataset.nints):
                residual, fitted, valid = ramp_residuals(ramp.data[i], rate.data[i], group_times,
                                                        ramp.groupdq[i], ramp.pixeldq)
                rate_rows.append([record['integration_start']+i, dataset.times[i],
                    stats(np.where((rate.dq[i] & 1) == 0, rate.data[i], np.nan))['median'],
                    stats(residual)['rms_about_zero'], int(valid.sum()),
                    int(np.count_nonzero(ramp.groupdq[i] & 4))])
                if start+i in samples_by_index:
                    sample = samples_by_index[start+i]
                    sample.update(rate_image=rate.data[i].copy(), slopes=rate.data[i, py, px].copy(),
                        fitted=fitted[:, py, px].copy(), residuals=residual[:, py, px].copy(), valid=valid[:, py, px].copy())
            report['stages']['ramp_fitting'].append(dict(input=path.name,
                elapsed_seconds=time.monotonic()-tick, datamodel=str(rate_path.relative_to(folder)),
                metadata=rate.meta.instance.copy(), parameters=config['ramp_fit_parameters']))
            rate_models.append(rate)
            for model in dataset.ramps_per_segment:
                model.close()
            start += dataset.nints
            del dataset, ramp
            gc.collect()
            write_json(folder/'reduction.json', report)
        samples = [dict(samples_by_index[i], selection=label) for i, label in zip(selected,
                         ['tenth', 'middle', 'tenth from last'])]
        detector_plots(samples, config, folder)
        detector_plots(samples, config, folder, fitted=True)
        save_sample_table(samples, folder, config['pixels_xy'])
        times = np.asarray(all_times)
        if np.any(np.diff(times) <= 0):
            raise ValueError('Integration times must increase strictly')
        np.savetxt(folder/'detector_timeseries.csv', detector_rows, delimiter=',',
            header='integration_number,time_bjd_tdb,last_minus_first_median_dn', comments='')
        np.savetxt(folder/'ramp_fit_timeseries.csv', rate_rows, delimiter=',',
            header='integration_number,time_bjd_tdb,median_rate_dn_per_second,residual_rms_dn,'
                   'valid_group_pixels,jump_flagged_group_pixels', comments='')
        report['metrics'] = dict(detector_image_medians_dn=stats(np.array(detector_rows)[:, 2]),
            rate_image_medians_dn_per_second=stats(np.array(rate_rows)[:, 2]),
            ramp_residual_rms_dn=stats(np.array(rate_rows)[:, 3]),
            selected_ramp_residuals_dn=[stats(sample['residuals']) for sample in samples])
        report['definitions'] = dict(
            ramp_residual='signal - pipeline_rate * group_time - median_offset; one offset per pixel; '
                'exclude group DQ bits 1/2/4 and pixel DO_NOT_USE; not segmented-fit chi-square',
            rms='sqrt(mean(x**2)); rms_about_median subtracts the finite median first',
            white_light='fixed common finite wavelength columns; normalize by time median; '
                'errors assume independent columns and are conditional on optimal profiles',
            filter='reflected padding; 100-point window offsets -50..+49; no transit-model fit',
            trace='median over columns of smoothed trace minus each column time median',
            fwhm='cubic spline half-maximum roots on extraction-preprocessed profiles; '
                'ambiguous widths are missing; changes relative to each column time median',
            differences='median over common columns of median_t(simple) - median_t(optimal)')
        inputs = dict(rampstep=rate_models, times=times,
                      ints_per_segment=[len(rate.data) for rate in rate_models])
        extractions = {}
        for method in ['simple', 'polynomial', 'gp']:
            print('Stage 2:', method, flush=True)
            tick = time.monotonic()
            result = ts.jwst.stage2(inputs, nthreads=config['nthreads'],
                outputfolder=str(folder), suffix='validation', aperture_radius=config['aperture_radius'],
                optimal_extraction=method != 'simple', extraction_backend='python',
                profile_method='gp' if method == 'gp' else 'polynomial',
                gp_options=config['gp_options'] if method == 'gp' else None,
                extraction_options=config['extraction_options'], zero_nans=True,
                mask_dq=config['mask_dq'], scale_1f=config['scale_1f'],
                single_trace_extraction=config['single_trace_extraction'])
            if method == 'simple':
                # Preserve a reopenable WCS datamodel and its reference provenance.
                # stage2 evaluates WCS internally but does not expose that model.
                wcs_model = ts.jwst.calwebb_spec2.assign_wcs_step.AssignWcsStep.call(rate_models[0])
                try:
                    wcs_model.save(folder/'wcs_first_segment_rateints.fits')
                    report['stages']['wavelength_calibration'] = dict(
                        reference_files=wcs_model.meta.ref_file.instance,
                        calibration_steps=wcs_model.meta.cal_step.instance,
                        wcs_info=wcs_model.meta.wcsinfo.instance,
                        datamodel='wcs_first_segment_rateints.fits')
                finally:
                    wcs_model.close()
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore', message='All-NaN slice encountered')
                    result['validation_median_rate'] = np.nanmedian(
                        np.concatenate([model.data for model in rate_models]), axis=0)
                report['metrics']['trace_fwhm'] = trace_diagnostics(result, times, config, folder)
                from stdatamodels.jwst import datamodels
                with datamodels.CubeModel(data=result['tso'].astype(np.float32),
                                          err=result['tso_err'].astype(np.float32)) as model:
                    model.update(rate_models[0])
                    model.dq = np.concatenate([rate.dq for rate in rate_models])
                    model.int_times = np.concatenate([rate.int_times for rate in rate_models])
                    model.meta.exposure.integration_start = records[0]['integration_start']
                    model.meta.exposure.integration_end = records[-1]['integration_end']
                    model.save(folder/'extraction_input_rateints.fits')
            spectrum = result['spectra']
            extractions[method] = {key: spectrum[key] for key in
                                  ['original', 'original_err', 'corrected', 'corrected_err', 'wavelengths']}
            extractions[method]['columns'] = result['traces']['x']
            if method != 'simple':
                np.savez_compressed(folder/f'profile_{method}.npz', profile=spectrum['P'],
                                    columns=result['traces']['x'])
            report['stages'][f'extraction_{method}'] = dict(elapsed_seconds=time.monotonic()-tick,
                settings=spectrum.get('extraction_settings', {}), metadata=result['metadata'])
            np.savez_compressed(folder/f'spectra_{method}.npz', time_bjd_tdb=times, **extractions[method])
            del result, spectrum
            gc.collect()
            write_json(folder/'reduction.json', report)
        report['metrics']['extraction'] = extraction_diagnostics(extractions, times, config, folder)
    finally:
        for model in rate_models:
            model.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-glob', help='Local raw NRS1 segment files; overrides download selection')
    parser.add_argument('--download', action='store_true', help='Download only requested public segments')
    parser.add_argument('--segments', type=int, nargs='+', default=[1])
    parser.add_argument('--exposure', default='jw01118005001_04101_00001')
    parser.add_argument('--data-dir', type=Path, default=ROOT/'validation/data')
    parser.add_argument('--output', type=Path, help='New directory; existing directories are refused')
    parser.add_argument('--config', type=Path, default=ROOT/'validation/config.json')
    args = parser.parse_args()
    if any(i < 1 for i in args.segments):
        parser.error('Segment numbers must be positive')
    if not re.fullmatch(r'jw\d{11}_\d{5}_\d{5}', args.exposure):
        parser.error('Invalid JWST exposure identifier')
    config = json.loads(args.config.read_text())
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    folder = (args.output or ROOT/'validation/runs'/stamp).resolve()
    folder.mkdir(parents=True, exist_ok=False)
    log_handler = logging.FileHandler(folder/'pipeline.log')
    log_handler.setFormatter(logging.Formatter('%(asctime)s %(name)s %(levelname)s %(message)s'))
    logging.getLogger().addHandler(log_handler)
    os.environ.setdefault('CRDS_PATH', str(Path.home()/'crds_cache'))
    os.environ.setdefault('CRDS_SERVER_URL', 'https://jwst-crds.stsci.edu')
    Path(os.environ['CRDS_PATH']).mkdir(parents=True, exist_ok=True)
    report = dict(schema_version=1, status='running', started_utc=stamp,
        command=sys.argv, config=config, source_hashes={
            str(p.relative_to(ROOT)): sha256(p) for p in
            [*sorted((ROOT/'src').glob('*.py')), Path(__file__), ROOT/'validation/diagnostics.py']},
        software={name: importlib.metadata.version(name) for name in
            ['jwst', 'stdatamodels', 'stcal', 'crds', 'numpy', 'scipy', 'astropy', 'matplotlib']},
        python=sys.version, crds={key: os.environ.get(key) for key in
            ['CRDS_PATH', 'CRDS_SERVER_URL', 'CRDS_CONTEXT']}, warnings={})
    git = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True)
    report['git_commit'] = git.stdout.strip() if git.returncode == 0 else None
    write_json(folder/'reduction.json', report)
    tick = time.monotonic()
    caught = []
    try:
        ts = checkout_package()
        if config['nthreads'] is not None:
            location = Path(importlib.metadata.distribution('transitspectroscopy').locate_file('transitspectroscopy'))
            for path in (ROOT/'src').glob('*.py'):
                if not (location/path.name).exists() or sha256(path) != sha256(location/path.name):
                    raise ValueError('Reinstall this checkout before enabling Ray (nthreads)')
        if args.input_glob:
            paths = [Path(p).resolve() for p in glob.glob(args.input_glob)]
        elif args.download:
            paths = download_segments(args.exposure, sorted(set(args.segments)), args.data_dir)
        else:
            paths = [args.data_dir/f'{args.exposure}-seg{i:03d}_nrs1_uncal.fits' for i in sorted(set(args.segments))]
        if not paths or any(not path.is_file() for path in paths):
            raise FileNotFoundError('No matching inputs. Use --download or --input-glob.')
        report['inputs'] = inspect_inputs(paths)
        write_json(folder/'reduction.json', report)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            reduce(report['inputs'], config, folder, report, ts)
        report['status'] = 'complete'
    except BaseException as error:
        report['status'] = 'failed'
        report['error'] = dict(type=type(error).__name__, message=str(error), traceback=traceback.format_exc())
        raise
    finally:
        report['elapsed_seconds'] = time.monotonic()-tick
        report['warnings'] = dict(Counter(f'{w.category.__name__}: {w.message}' for w in caught))
        report['products'] = [{'path': str(path.relative_to(folder)), 'size_bytes': path.stat().st_size}
                              for path in sorted(folder.rglob('*')) if path.is_file() and path.name != 'reduction.json']
        write_json(folder/'reduction.json', report)
        print('Validation report:', folder/'reduction.json', flush=True)
        logging.getLogger().removeHandler(log_handler)
        log_handler.close()


if __name__ == '__main__':
    main()
