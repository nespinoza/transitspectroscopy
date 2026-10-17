"""Validation of transitspectroscopy on real JWST data.

Each validation is a run configuration (``validation/configs/*.toml``): the reduction is
done by ``ts.run`` (the same as ``transitspectroscopy run <config>``), and this script adds
the diagnostics, plots and metrics of the validation reports. Run from the repository root,
with this checkout installed (``python -m pip install --no-build-isolation --no-deps .``)::

    python validation/validate.py --config validation/configs/nrs1-hatp14b.toml
    python validation/validate.py --config validation/configs/soss-hatp14b.toml
    python validation/validate.py --config validation/configs/miri-ltt1445a.toml

``--mode nrs1|soss|miri`` selects the default configuration of each validation. Rerunning
reuses completed reduction steps of the same configuration (see ``ts.run``); ``--force``
recomputes everything.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import gc
import hashlib
import importlib.metadata
import logging
import os
from pathlib import Path
import sys
import time
import traceback
import warnings

import numpy as np

from diagnostics import (detector_plots, extraction_diagnostics, ramp_residuals,
                         selected_integrations, stats, trace_diagnostics, write_json)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIGS = dict(nrs1='nrs1-hatp14b.toml', soss='soss-hatp14b.toml', miri='miri-ltt1445a.toml')


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(8*1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def installed_package():
    """The installed package; refuse a stale installation of this checkout."""
    import transitspectroscopy as ts
    location = Path(ts.__file__).parent
    if location.resolve() != (ROOT/'src').resolve():
        stale = [str(path.relative_to(ROOT/'src')) for path in (ROOT/'src').rglob('*.py')
                 if '__pycache__' not in path.parts and (not (location/path.relative_to(ROOT/'src')).exists()
                     or sha256(path) != sha256(location/path.relative_to(ROOT/'src')))]
        if stale:
            raise RuntimeError('The installed transitspectroscopy differs from this checkout ('
                               + ', '.join(stale[:5]) + '); reinstall it with '
                               '"python -m pip install --no-build-isolation --no-deps ."')
    return ts


def checkout_package():
    """The installed transitspectroscopy, checked against this checkout (used by the other scripts)."""
    return installed_package()


def read_config(path):
    """Run configuration with paths resolved against the repository root."""
    import transitspectroscopy as ts
    config = ts.load_config(Path(path))
    config['run']['output'] = str(ROOT/config['run']['output'])
    for key in ('folder', 'glob'):
        if key in config['data']:
            config['data'][key] = str(ROOT/config['data'][key])
    for settings in config['extraction'].values():
        for key in ('pastasoss_reference', 'background_reference'):
            if key in (settings.get('soss') or {}):
                settings['soss'][key] = str(ROOT/settings['soss'][key])
    return config


def new_report(config, kind, folder):
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    git = None
    try:
        import subprocess
        out = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True)
        git = out.stdout.strip() if out.returncode == 0 else None
    except OSError:
        pass
    return dict(schema_version=2, status='running', mode=kind, started_utc=stamp, command=sys.argv,
                config=config, git_commit=git,
                software={name: importlib.metadata.version(name) for name in
                          ['transitspectroscopy', 'jwst', 'stdatamodels', 'stcal', 'crds', 'numpy',
                           'scipy', 'astropy', 'matplotlib']},
                crds={key: os.environ.get(key) for key in ['CRDS_PATH', 'CRDS_SERVER_URL', 'CRDS_CONTEXT']})


def finish(report, folder, start, caught=()):
    report['elapsed_seconds'] = time.monotonic()-start
    report['warnings'] = dict(Counter(f'{w.category.__name__}: {w.message}' for w in caught))
    report['products'] = [dict(path=str(p.relative_to(folder)), size_bytes=p.stat().st_size)
                          for p in sorted(folder.rglob('*')) if p.is_file() and p.name != 'reduction.json'
                          and 'reduction' not in p.relative_to(folder).parts[:1]]
    write_json(folder/'reduction.json', report)


def run_summary(result):
    return dict(output=str(result.output), steps={k: str(v) for k, v in result.steps.items()},
                reused=result.reused, inputs=result.files)


# ---------------------------------------------------------------------------------------
# NIRSpec/G395H NRS1


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


def nrs1_settings(config):
    """Settings of the diagnostics functions (formerly validation/config.json)."""
    gp = config['extraction']['gp']
    return dict(config['validation'], aperture_radius=gp['aperture_radius'],
                extraction_options={key: gp[key] for key in ('polynomial_spacing', 'polynomial_order', 'nsigma')},
                gp_options=gp['gp_options'], mask_dq=gp['mask_dq'], scale_1f=gp['scale_1f'],
                single_trace_extraction=gp['single_trace_extraction'],
                nthreads=config['execution'].get('nthreads'))


def detector_diagnostics(dataset, settings, folder, report):
    """Last-minus-first images, rates and ramp residuals of three sample integrations."""
    ramps, rates = dataset.ramps_per_segment, dataset.rateints_per_segment
    nints = sum(len(rate.data) for rate in rates)
    selected = selected_integrations(nints)
    report['selection'] = dict(loaded_integrations=nints, selected_loaded_indices=selected,
        coordinate_convention='zero-based (column, row)', short_series_clamped=nints < 20)
    samples_by_index, detector_rows, rate_rows = {}, [], []
    start = 0
    px, py = np.array(settings['pixels_xy']).T
    for ramp, rate in zip(ramps, rates):
        if len(getattr(ramp.meta.exposure, 'read_times', None) or []):
            raise ValueError('Uneven group timing is not supported by this preset')
        times = np.asarray(rate.int_times['int_mid_BJD_TDB'])+2400000.5
        first_integration = ramp.meta.exposure.integration_start
        group_times = (np.arange(ramp.data.shape[1])+1)*ramp.meta.exposure.group_time
        for x, y in settings['pixels_xy']:
            if not (0 <= x < ramp.data.shape[3] and 0 <= y < ramp.data.shape[2]):
                raise ValueError(f'Pixel {(x, y)} lies outside the subarray')
        for i in range(len(ramp.data)):
            image = ramp.data[i, -1]-ramp.data[i, 0]
            good = ((ramp.pixeldq & 1) == 0) & ((ramp.groupdq[i, -1] & 3) == 0) & ((ramp.groupdq[i, 0] & 3) == 0)
            detector_rows.append([first_integration+i, times[i], stats(np.where(good, image, np.nan))['median']])
            residual, fitted, valid = ramp_residuals(ramp.data[i], rate.data[i], group_times,
                                                    ramp.groupdq[i], ramp.pixeldq)
            rate_rows.append([first_integration+i, times[i],
                stats(np.where((rate.dq[i] & 1) == 0, rate.data[i], np.nan))['median'],
                stats(residual)['rms_about_zero'], int(valid.sum()), int(np.count_nonzero(ramp.groupdq[i] & 4))])
            if start+i in selected:
                samples_by_index[start+i] = dict(loaded_index=start+i, integration_number=first_integration+i,
                    difference_image=image.copy(), group_times_s=group_times.copy(),
                    groups=ramp.data[i][:, py, px].copy(), groupdq=ramp.groupdq[i][:, py, px].copy(),
                    rate_image=rate.data[i].copy(), slopes=rate.data[i, py, px].copy(),
                    fitted=fitted[:, py, px].copy(), residuals=residual[:, py, px].copy(),
                    valid=valid[:, py, px].copy())
        start += len(ramp.data)
    samples = [dict(samples_by_index[i], selection=label) for i, label in zip(selected,
               ['tenth', 'middle', 'tenth from last'])]
    detector_plots(samples, settings, folder)
    detector_plots(samples, settings, folder, fitted=True)
    save_sample_table(samples, folder, settings['pixels_xy'])
    np.savetxt(folder/'detector_timeseries.csv', detector_rows, delimiter=',',
        header='integration_number,time_bjd_tdb,last_minus_first_median_dn', comments='')
    np.savetxt(folder/'ramp_fit_timeseries.csv', rate_rows, delimiter=',',
        header='integration_number,time_bjd_tdb,median_rate_dn_per_second,residual_rms_dn,'
               'valid_group_pixels,jump_flagged_group_pixels', comments='')
    return dict(detector_image_medians_dn=stats(np.array(detector_rows)[:, 2]),
                rate_image_medians_dn_per_second=stats(np.array(rate_rows)[:, 2]),
                ramp_residual_rms_dn=stats(np.array(rate_rows)[:, 3]),
                selected_ramp_residuals_dn=[stats(sample['residuals']) for sample in samples])


DEFINITIONS = dict(
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


def validate_nrs1(config, folder, report, ts, force=False, from_step=None):
    settings = nrs1_settings(config)
    report['settings'] = settings
    result = ts.run(config, force=force, from_step=from_step, keep_stage2=True)
    report['run'] = run_summary(result)
    write_json(folder/'reduction.json', report)
    dataset = result.dataset
    report['metrics'] = detector_diagnostics(dataset, settings, folder, report)
    report['definitions'] = DEFINITIONS
    times = dataset.times
    if np.any(np.diff(times) <= 0):
        raise ValueError('Integration times must increase strictly')
    rates = dataset.rateints_per_segment
    extractions = {}
    report['stages'] = {}
    for method in ['simple', 'polynomial', 'gp']:
        stage2 = result.stage2[method]
        if method == 'simple':
            # A reopenable WCS datamodel and its reference provenance.
            wcs_model = ts.jwst.calwebb_spec2.assign_wcs_step.AssignWcsStep.call(rates[0])
            try:
                wcs_model.save(folder/'wcs_first_segment_rateints.fits')
                report['stages']['wavelength_calibration'] = dict(
                    reference_files=wcs_model.meta.ref_file.instance,
                    calibration_steps=wcs_model.meta.cal_step.instance,
                    wcs_info=wcs_model.meta.wcsinfo.instance, datamodel='wcs_first_segment_rateints.fits')
            finally:
                wcs_model.close()
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', message='All-NaN slice encountered')
                stage2['validation_median_rate'] = np.nanmedian(np.concatenate([m.data for m in rates]), axis=0)
            report['metrics']['trace_fwhm'] = trace_diagnostics(stage2, times, settings, folder)
        spectrum = stage2['spectra']
        extractions[method] = {key: spectrum[key] for key in
                               ['original', 'original_err', 'corrected', 'corrected_err', 'wavelengths']}
        extractions[method]['columns'] = stage2['traces']['x']
        if method != 'simple':
            np.savez_compressed(folder/f'profile_{method}.npz', profile=spectrum['P'], columns=stage2['traces']['x'])
        report['stages'][f'extraction_{method}'] = dict(settings=spectrum.get('extraction_settings', {}),
                                                        metadata=stage2['metadata'])
        np.savez_compressed(folder/f'spectra_{method}.npz', time_bjd_tdb=times, **extractions[method])
        gc.collect()
    report['metrics']['extraction'] = extraction_diagnostics(extractions, times, settings, folder)


# ---------------------------------------------------------------------------------------


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config', type=Path, help='validation run configuration (TOML)')
    parser.add_argument('--mode', choices=sorted(DEFAULT_CONFIGS), help='use the default configuration')
    parser.add_argument('--output', type=Path, help='output folder (default: the configuration run.output)')
    parser.add_argument('--force', action='store_true', help='recompute every reduction step')
    parser.add_argument('--from-step', choices=['data', 'calibrate', 'extract', 'lightcurves'],
                        help='recompute the reduction from this step')
    parser.add_argument('--skip-transit', action='store_true', help='SOSS: skip the transit fits')
    parser.add_argument('--reference-spectrum', type=Path, help='SOSS: reference spectrum to overplot')
    parser.add_argument('--visits', nargs='+', help='MIRI: visits to reduce (default: all)')
    args = parser.parse_args(argv)
    if args.config is None:
        args.config = ROOT/'validation/configs'/DEFAULT_CONFIGS[args.mode or 'nrs1']
    config = read_config(args.config)
    kind = config['validation']['kind']
    # Diagnostics go to the output folder, the reduction steps (ts.run) to <output>/reduction.
    folder = (args.output or Path(config['run']['output'])).resolve()
    config['run']['output'] = str(folder/'reduction')
    folder.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('CRDS_PATH', str(Path.home()/'crds_cache'))
    os.environ.setdefault('CRDS_SERVER_URL', 'https://jwst-crds.stsci.edu')
    log_handler = logging.FileHandler(folder/'pipeline.log')
    log_handler.setFormatter(logging.Formatter('%(asctime)s %(name)s %(levelname)s %(message)s'))
    logging.getLogger().addHandler(log_handler)
    report = new_report(config, kind, folder)
    write_json(folder/'reduction.json', report)
    start, caught = time.monotonic(), []
    try:
        ts = installed_package()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            if kind == 'nrs1':
                validate_nrs1(config, folder, report, ts, args.force, args.from_step)
            elif kind == 'soss':
                from soss_validate import validate_soss
                validate_soss(config, folder, report, ts, args.force, args.from_step,
                              transit=not args.skip_transit, reference_spectrum=args.reference_spectrum)
            elif kind == 'miri':
                from miri_validate import validate_miri
                validate_miri(config, folder, report, ts, args.force, args.from_step, visits=args.visits)
            else:
                raise ValueError(f'Unknown validation kind {kind}')
        report['status'] = 'complete'
    except BaseException as error:
        report['status'] = 'failed'
        report['error'] = dict(type=type(error).__name__, message=str(error), traceback=traceback.format_exc())
        raise
    finally:
        finish(report, folder, start, caught)
        print('Validation report:', folder/'reduction.json', flush=True)
        logging.getLogger().removeHandler(log_handler)
        log_handler.close()


if __name__ == '__main__':
    main()
