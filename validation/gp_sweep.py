"""Repeat GP extraction from a validation run, with a fixed-band holdout check."""
import argparse
from datetime import datetime, timezone
import hashlib
import itertools
import json
from pathlib import Path
import sys
import time
import traceback
import warnings

import numpy as np
from astropy.io import fits

from diagnostics import smooth_series, write_json
from validate import ROOT, checkout_package, sha256
import matplotlib.pyplot as plt


def candidate_id(options):
    return hashlib.sha256(json.dumps(options, sort_keys=True).encode()).hexdigest()[:12]


def correct_spectra(flux, error, training):
    """Stage 2 spectral replacement, estimating its template on training rows only.

    Passing every integration reproduces the existing Stage 2 GP correction.
    Missing flux stays missing; replacement is never a temporal light-curve fit.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='All-NaN slice encountered')
        levels = np.nanmedian(flux, axis=1)
        normalized = flux[training]/levels[training, None]
        master = np.nanmedian(normalized, axis=0)
        scatter = 1.2533*1.4826*np.nanmedian(np.abs(normalized-master), axis=0)
    model = levels[:, None]*master
    sigma = levels[:, None]*scatter
    replaced = np.abs(flux-model) > 5*sigma
    return np.where(replaced, model, flux), np.where(replaced, sigma, error), replaced


def curve_statistics(flux, error, common, indices, window):
    """No candidate can improve its score by dropping wavelengths or exposures."""
    selected = np.asarray(flux)[indices][:, common]
    errors = np.asarray(error)[indices][:, common]
    if not np.all(np.isfinite(selected) & np.isfinite(errors) & (errors > 0)):
        raise ValueError('Candidate loses data in the fixed comparison band')
    integrated = selected.sum(axis=1)
    norm = np.median(integrated)
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError('Invalid white-light normalization')
    relative = integrated/norm
    filtered = smooth_series(relative, window)
    residual = (relative-filtered)*1e6
    binned = {}
    for size in (5, 10, 20):
        count = len(residual)//size
        if count:
            binned[str(size)] = float(np.sqrt(np.mean(residual[:count*size].reshape(count, size).mean(axis=1)**2)))
    metrics = dict(rms_ppm=float(np.sqrt(np.mean(residual**2))),
        first_difference_ppm=float(np.sqrt(np.mean(np.diff(relative)**2)/2)*1e6),
        binned_rms_ppm=binned, normalization_dn_per_second=float(norm), count=len(indices))
    arrays = dict(integrated_flux=integrated, relative_flux=relative,
        relative_error=np.sqrt((errors**2).sum(axis=1))/norm, median_filter=filtered,
        residual_ppm=residual)
    return metrics, arrays


def paired_bootstrap(first, second, block=20, draws=2000, seed=20260923):
    """Conditional paired moving-block uncertainty: positive means improvement."""
    first, second = np.asarray(first), np.asarray(second)
    if first.shape != second.shape or first.ndim != 1 or len(first) < block:
        raise ValueError('Paired bootstrap needs matching series longer than the block')
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, len(first)-block+1, size=(draws, int(np.ceil(len(first)/block))))
    indices = (starts[..., None]+np.arange(block)).reshape(draws, -1)[:, :len(first)]
    difference = np.sqrt(np.mean(first[indices]**2, axis=1))-np.sqrt(np.mean(second[indices]**2, axis=1))
    return dict(improvement_ppm=float(np.sqrt(np.mean(first**2))-np.sqrt(np.mean(second**2))),
                interval_95_ppm=np.percentile(difference, [2.5, 97.5]).tolist(),
                block_points=block, draws=draws, seed=seed)


class ExistingRun:
    """Reuse a saved extraction cube, traces, errors, mask and wavelengths."""
    def __init__(self, folder):
        self.folder = Path(folder).resolve()
        self.report = json.loads((self.folder/'reduction.json').read_text())
        if self.report['status'] != 'complete':
            raise ValueError('Start from a completed validation run')
        self.config = self.report['config']
        if self.config.get('spectrum_variant') != 'corrected':
            raise ValueError('This sweep currently compares the corrected validation variant')
        settings = self.report['stages']['extraction_gp']['settings']
        if settings.get('gp_mask_policy') != 'complete-column-sequential-v2':
            raise ValueError('Source run must use the current complete-column GP mask policy')
        # Reproducing a reference is meaningful only with the same extraction code.
        # A changed file does not necessarily change results (e.g. a refactor); warn, and
        # rely on the replayed source profile below to detect numerical differences.
        for name, expected in self.report['source_hashes'].items():
            current = ROOT/name
            if not current.exists() or sha256(current) != expected:
                warnings.warn(f'Source file differs from the validation run: {name}')
        self.provenance = dict(manifest=str(self.folder/'reduction.json'),
            manifest_sha256=sha256(self.folder/'reduction.json'), config=self.config,
            source_hashes=self.report['source_hashes'], software=self.report['software'], files={})
        for name in ('extraction_input_rateints.fits', 'trace_fwhm.npz', 'spectra_gp.npz', 'profile_gp.npz'):
            self.provenance['files'][name] = sha256(self.folder/name)
        with np.load(self.folder/'trace_fwhm.npz', allow_pickle=False) as trace:
            self.columns = trace['columns'].copy()
            self.traces = trace['trace_row'].copy()
            self.centers = trace['reference_trace_row'].copy()
        if not np.array_equal(self.columns, np.arange(self.columns[0], self.columns[-1]+1)):
            raise ValueError('Saved detector columns must be contiguous')
        section = slice(self.columns[0], self.columns[-1]+1)
        with fits.open(self.folder/'extraction_input_rateints.fits') as model:
            self.data = model['SCI'].data[:, :, section].astype(float)
            self.variance = model['ERR'].data[:, :, section].astype(float)**2
            if self.config['mask_dq']:
                self.variance[model['DQ'].data[:, :, section] != 0] = 0.
        with np.load(self.folder/'spectra_gp.npz', allow_pickle=False) as spectra:
            self.times = spectra['time_bjd_tdb'].copy()
            self.wavelength = spectra['wavelength_microns'].copy()
            self.common = spectra['common_white_light_columns'].copy()
            self.saved = {key: spectra[key].copy() for key in
                          ('original_flux', 'original_error', 'corrected_flux', 'corrected_error')}
        with np.load(self.folder/'profile_gp.npz', allow_pickle=False) as model:
            self.saved_profile = model['profile'].copy()
        if self.data.shape[0] != len(self.times) or self.traces.shape != (len(self.times), len(self.columns)):
            raise ValueError('Inconsistent saved cube/trace/timestamp dimensions')
        self.ts = checkout_package()
        opts = self.config['extraction_options']
        self.arguments = (self.config['aperture_radius'], 1., 1., opts['nsigma'],
                          opts['polynomial_spacing'], opts['polynomial_order'])
        self.window = self.config['white_light_filter_points']
        self.all = np.arange(len(self.times))
        self.train = self.all[:(len(self.times)+1)//2]
        self.holdout = self.all[len(self.train):]
        if min(len(self.train), len(self.holdout)) < self.window:
            raise ValueError('Each half must contain at least one full median-filter window')

    def trace(self, index):
        return self.centers if self.config['single_trace_extraction'] else self.traces[index]

    def fit(self, options, indices):
        profiles = []
        for index in indices:
            profiles.append(self.ts.spectroscopy.getP(self.data[index], self.trace(index),
                *self.arguments, data_variance=self.variance[index], backend='python',
                profile_method='gp', gp_options=options))
        profile = np.median(profiles, axis=0)
        profile /= profile.sum(axis=0)
        if not np.isfinite(profile).all() or np.any(profile < 0):
            raise ValueError('Invalid shared median profile')
        return profile

    def extract(self, profile):
        flux, error = [], []
        for index in self.all:
            result = self.ts.spectroscopy.getOptimalSpectrum(self.data[index], self.trace(index),
                *self.arguments, data_variance=self.variance[index], backend='python',
                profile_method='gp', P=profile)
            flux.append(result[1])
            with np.errstate(divide='ignore'):
                error.append(np.sqrt(1./result[2]))
        return np.asarray(flux), np.asarray(error)

    def evaluate(self, profile, template_indices, score_indices):
        flux, error = self.extract(profile)
        corrected, corrected_error, replaced = correct_spectra(flux, error, template_indices)
        metrics = {}
        for name, f, e in [('original', flux, error), ('corrected', corrected, corrected_error)]:
            metrics[name], _ = curve_statistics(f, e, self.common, score_indices, self.window)
        metrics['replacement_fraction'] = float(replaced[score_indices][:, self.common].mean())
        return metrics, dict(profile=profile, original_flux=flux, original_error=error,
            corrected_flux=corrected, corrected_error=corrected_error, replaced=replaced)


def save_curves(folder, name, run, products, indices):
    result = {}
    for variant in ('original', 'corrected'):
        metrics, arrays = curve_statistics(products[variant+'_flux'], products[variant+'_error'],
                                          run.common, indices, run.window)
        table = np.column_stack([run.times[indices], indices, *arrays.values()])
        np.savetxt(folder/f'{name}_{variant}_lightcurve.csv', table, delimiter=',',
            header='time_bjd_tdb,integration_index,'+','.join(arrays), comments='')
        result[variant] = metrics
    return result


def spectral_guard(run, reference, candidate):
    """Prevent lower scatter bought by grossly distorting the bright spectrum."""
    _, _, _, inside, *_ = run.ts.core.geometry.marsh_geometry(run.centers, run.data.shape[1],
        run.config['aperture_radius'], run.config['extraction_options']['polynomial_spacing'])
    complete = np.mean(np.all(~inside.T[None] | (run.variance[run.train] > 0), axis=1), axis=0) >= .9
    first = np.nanmedian(reference['original_flux'][run.train], axis=0)
    second = np.nanmedian(candidate['original_flux'][run.train], axis=0)
    selected = run.common & complete & (first > 100.)
    if selected.sum() < 10:
        raise ValueError('Too few complete bright columns for the spectral guard')
    return float(np.percentile(np.abs(second[selected]/first[selected]-1), 95))


def make_plots(folder, report, run, source, selected):
    rows = [row for row in report['candidates'].values() if row['status'] == 'complete']
    fig, ax = plt.subplots(figsize=(12, 5), layout='constrained')
    for kernel in sorted({row['options']['kernel'] for row in rows}):
        for amplitude in sorted({row['options']['amplitude'] for row in rows}):
            group = sorted([row for row in rows if row['options']['kernel'] == kernel
                and row['options']['amplitude'] == amplitude and row['options']['n_inducing'] == report['grid_inducing']],
                key=lambda row: row['options']['length_scale'])
            if group:
                ax.plot([row['options']['length_scale'] for row in group],
                        [row['metrics']['corrected']['rms_ppm'] for row in group], '.-',
                        label=f'{kernel}, amplitude {amplitude:g}')
    ax.set(xscale='log', xlabel='GP length scale (dispersion pixels)',
           ylabel='Development residual RMS (ppm)', title='GP sweep: fixed band, filter, aperture and rejection')
    ax.legend(fontsize=8, ncol=2)
    fig.savefig(folder/'sweep.png', dpi=150); plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(13, 7), sharex=True, layout='constrained')
    hours = (run.times-run.times[0])*24
    for column, (label, products) in enumerate([('Source GP settings', source), ('Selected GP settings', selected)]):
        metrics, arrays = curve_statistics(products['corrected_flux'], products['corrected_error'],
                                          run.common, run.all, run.window)
        axes[0, column].plot(hours, arrays['relative_flux'], '.', ms=3)
        axes[0, column].plot(hours, arrays['median_filter'], 'k-', lw=1)
        axes[0, column].set(title=label, ylabel='Relative white-light flux')
        axes[1, column].plot(hours, arrays['residual_ppm'], '.', ms=3)
        axes[1, column].set(title=f"RMS = {metrics['rms_ppm']:.2f} ppm", ylabel='Residual (ppm)', xlabel='Hours')
    for row in axes:
        limits = [ax.get_ylim() for ax in row]
        for ax in row:
            ax.set_ylim(min(v[0] for v in limits), max(v[1] for v in limits))
    fig.savefig(folder/'white_light_comparison.png', dpi=150); plt.close(fig)
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True, layout='constrained')
    a = np.nanmedian(source['corrected_flux'], axis=0)
    b = np.nanmedian(selected['corrected_flux'], axis=0)
    axes[0].plot(run.wavelength, a, label='Source GP', lw=.8)
    axes[0].plot(run.wavelength, b, label='Selected GP', lw=.8)
    axes[0].set_ylabel('Median flux (DN/s)'); axes[0].legend()
    axes[1].plot(run.wavelength, b-a, lw=.8)
    axes[1].set(xlabel='Wavelength (µm)', ylabel='Selected − source (DN/s)')
    fig.savefig(folder/'spectrum_comparison.png', dpi=150); plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--length-scales', type=float, nargs='+', default=[30., 100., 300., 1000.])
    parser.add_argument('--amplitudes', type=float, nargs='+', default=[.03, .1, 1., 3.])
    parser.add_argument('--kernels', nargs='+', choices=['matern32', 'matern52'], default=['matern32', 'matern52'])
    parser.add_argument('--train-profiles', type=int, default=32)
    parser.add_argument('--resolution-check', type=int, nargs='*', default=[12, 48])
    args = parser.parse_args()
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    folder = (args.output or ROOT/'validation/runs'/f'{stamp}_gp_sweep').resolve()
    if args.resume and not args.output:
        parser.error('--resume requires --output')
    folder.mkdir(parents=True, exist_ok=args.resume)
    run = ExistingRun(args.run)
    if not 2 <= args.train_profiles <= len(run.train):
        parser.error('--train-profiles must be between 2 and the development integration count')
    fit_indices = np.unique(np.linspace(run.train[0], run.train[-1], args.train_profiles).astype(int))
    baseline = dict(run.config['gp_options'])
    identity = dict(source=run.provenance, length_scales=args.length_scales, amplitudes=args.amplitudes,
        kernels=args.kernels, fit_indices=fit_indices.tolist(), resolution_check=args.resolution_check,
        code={str(p.relative_to(ROOT)): sha256(p) for p in [Path(__file__), ROOT/'validation/diagnostics.py']})
    manifest = folder/'sweep.json'
    if args.resume:
        report = json.loads(manifest.read_text())
        if report['identity'] != identity:
            raise ValueError('Cannot resume with changed inputs, code, settings or profile-training rows')
    else:
        report = dict(schema_version=1, status='running', started_utc=stamp, command=sys.argv,
            identity=identity, source_run=str(run.folder), baseline_options=baseline,
            grid_inducing=baseline['n_inducing'], development_indices=run.train.tolist(),
            holdout_indices=run.holdout.tolist(), common_columns=int(run.common.sum()), candidates={},
            definitions=dict(objective='development corrected white-light 100-point-filter residual RMS',
                selection='original RMS <= 1.02 * baseline and spectrum 95th-percentile change <= 0.02',
                holdout='second half; profile and spectral-replacement template fitted on first half only; '
                    'conditional on fixed full-run tracing/background/1f preprocessing, not an independent observation',
                bootstrap='paired moving blocks, conditional on selected settings and detrending; excludes preprocessing uncertainty',
                guard='complete in >=90% of development frames and baseline median flux >100 DN/s',
                full_refit='descriptive full-run result after selection; profile and template use every integration'))
    tick = time.monotonic()
    write_json(manifest, report)
    try:
        # First prove that the stored float32 input cube reproduces the run.
        metrics, source = run.evaluate(run.saved_profile, run.all, run.all)
        _, saved_curve = curve_statistics(run.saved['corrected_flux'], run.saved['corrected_error'],
                                         run.common, run.all, run.window)
        _, replay = curve_statistics(source['corrected_flux'], source['corrected_error'], run.common, run.all, run.window)
        difference = float(np.max(np.abs(replay['relative_flux']-saved_curve['relative_flux']))*1e6)
        report['source_replay'] = dict(metrics=metrics, maximum_lightcurve_difference_ppm=difference)
        if difference > .1:
            raise ValueError(f'Reused input does not reproduce the reference: {difference:g} ppm')
        print(f'Reference reproduced to {difference:.6f} ppm', flush=True)
        save_curves(folder, 'source_full', run, source, run.all)
        write_json(manifest, report)

        def evaluate(options, reference=None):
            name = candidate_id(options)
            product = folder/f'candidate_{name}.npz'
            if name in report['candidates']:
                row = report['candidates'][name]
                if row['status'] == 'complete' and product.is_file():
                    with np.load(product, allow_pickle=False) as saved:
                        return row, {k: saved[k].copy() for k in saved.files}
                if row['status'] == 'failed':
                    return row, None
            started = time.monotonic()
            row = dict(options=options, status='running')
            report['candidates'][name] = row
            write_json(manifest, report)
            print('Candidate', options, flush=True)
            try:
                profile = run.fit(options, fit_indices)
                m, products = run.evaluate(profile, run.train, run.train)
                guard = 0. if reference is None else spectral_guard(run, reference, products)
                row.update(status='complete', metrics=m, spectral_guard_fraction=guard,
                    elapsed_seconds=time.monotonic()-started, product=product.name)
                np.savez_compressed(product, **products)
                print('  development RMS', m['corrected']['rms_ppm'], 'ppm; raw', m['original']['rms_ppm'], flush=True)
            except (ValueError, np.linalg.LinAlgError) as error:
                row.update(status='failed', error=str(error), elapsed_seconds=time.monotonic()-started)
                products = None
                print('  failed:', error, flush=True)
            write_json(manifest, report)
            return row, products

        base_row, base_products = evaluate(baseline)
        if base_products is None:
            raise ValueError('Development baseline failed')
        for kernel, length, amplitude in itertools.product(args.kernels, args.length_scales, args.amplitudes):
            evaluate(dict(kernel=kernel, length_scale=length, amplitude=amplitude,
                          n_inducing=baseline['n_inducing']), base_products)

        def choose():
            eligible = []
            for name, row in report['candidates'].items():
                if row['status'] != 'complete':
                    continue
                row['eligible'] = (row['metrics']['original']['rms_ppm'] <= 1.02*base_row['metrics']['original']['rms_ppm']
                                   and row['spectral_guard_fraction'] <= .02)
                if row['eligible']:
                    eligible.append(name)
            return min(eligible, key=lambda name: report['candidates'][name]['metrics']['corrected']['rms_ppm'])

        best_grid = dict(report['candidates'][choose()]['options'])
        for number in args.resolution_check:
            evaluate(dict(best_grid, n_inducing=number), base_products)
        winner = choose()
        selected_options = report['candidates'][winner]['options']
        report['selected_id'], report['selected_options'] = winner, selected_options
        report['selection_locked_before_holdout'] = True
        write_json(manifest, report)
        print('Selected using development data only:', selected_options, flush=True)
        # Confirm with every development integration, then inspect holdout once.
        confirmation = {}
        for name, options in [('baseline', baseline), ('selected', selected_options)]:
            profile = run.fit(options, run.train)
            m, products = run.evaluate(profile, run.train, run.holdout)
            confirmation[name] = products
            report.setdefault('holdout', {})[name] = m
            save_curves(folder, name+'_holdout', run, products, run.holdout)
            np.savez_compressed(folder/f'{name}_holdout_spectra.npz', time_bjd_tdb=run.times,
                common_columns=run.common, **products)
            write_json(manifest, report)
        for variant in ('original', 'corrected'):
            residuals = [curve_statistics(p[variant+'_flux'], p[variant+'_error'], run.common,
                run.holdout, run.window)[1]['residual_ppm'] for p in confirmation.values()]
            report['holdout'][variant+'_paired_bootstrap'] = paired_bootstrap(*residuals)
        print('Holdout:', json.dumps(report['holdout']), flush=True)
        # Produce the exact full-exposure counterpart of the validation method.
        profile = run.fit(selected_options, run.all)
        final_metrics, selected = run.evaluate(profile, run.all, run.all)
        report['selected_full'] = final_metrics
        report['full_improvement_ppm'] = metrics['corrected']['rms_ppm']-final_metrics['corrected']['rms_ppm']
        report['holdout_improvement_supported'] = bool(
            report['holdout']['corrected_paired_bootstrap']['interval_95_ppm'][0] > 0
            and report['holdout']['selected']['original']['rms_ppm'] <= report['holdout']['baseline']['original']['rms_ppm'])
        save_curves(folder, 'selected_full', run, selected, run.all)
        np.savez_compressed(folder/'selected_full_spectra.npz', time_bjd_tdb=run.times,
            wavelength_microns=run.wavelength, columns=run.columns, common_columns=run.common, **selected)
        config = dict(run.config, gp_options=selected_options)
        write_json(folder/'selected_config.json', config)
        make_plots(folder, report, run, source, selected)
        rows = []
        for name, row in report['candidates'].items():
            if row['status'] == 'complete':
                o, m = row['options'], row['metrics']
                rows.append([o['length_scale'], o['amplitude'], o['n_inducing'], int(o['kernel']=='matern52'),
                    m['original']['rms_ppm'], m['corrected']['rms_ppm'], m['replacement_fraction'],
                    row['spectral_guard_fraction'], row['eligible'], name==winner])
        np.savetxt(folder/'candidates.csv', rows, delimiter=',', header='length_scale,amplitude,n_inducing,'
            'matern52,development_original_rms_ppm,development_corrected_rms_ppm,replacement_fraction,'
            'spectral_guard_fraction,eligible,selected', comments='')
        report['status'] = 'complete'
    except BaseException as error:
        report['status'] = 'failed'
        report['error'] = dict(type=type(error).__name__, message=str(error), traceback=traceback.format_exc())
        raise
    finally:
        report['elapsed_seconds_this_invocation'] = time.monotonic()-tick
        write_json(manifest, report)
        print('Sweep report:', manifest, flush=True)


if __name__ == '__main__':
    main()
