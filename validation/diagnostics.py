"""Numerical summaries and saved plots for the NRS1 validation workflow."""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import AsinhNorm


def json_value(value):
    """Strict portable JSON: unavailable/nonfinite measurements become null."""
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [json_value(v) for v in value]
    if isinstance(value, np.generic):
        return json_value(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if value is None or isinstance(value, (str, float, int, bool)):
        return value
    return str(value)


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(json_value(value), indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def stats(values):
    array = np.asarray(values, dtype=float)
    finite = array[np.isfinite(array)]
    result = dict(count=int(array.size), finite_count=int(finite.size))
    if not finite.size:
        return dict(result, median=None, rms_about_zero=None, rms_about_median=None)
    median = np.median(finite)
    return dict(result, median=float(median), minimum=float(finite.min()),
                maximum=float(finite.max()), p01=float(np.percentile(finite, 1)),
                p99=float(np.percentile(finite, 99)),
                rms_about_zero=float(np.sqrt(np.mean(finite**2))),
                rms_about_median=float(np.sqrt(np.mean((finite-median)**2))))


def selected_integrations(nints):
    if nints < 1:
        raise ValueError('No integrations loaded')
    return [min(9, nints-1), nints//2, max(0, nints-10)]


def smooth_series(values, window):
    """Exactly window samples, including even windows; reflected end padding.

    For 100 points, the central window spans offsets -50 through +49. Missing
    points do not enter the median; an entirely missing window remains NaN.
    """
    if window < 1 or int(window) != window:
        raise ValueError('Median-filter window must be a positive integer')
    import warnings
    values = np.asarray(values, float)
    if values.ndim != 1 or not values.size:
        raise ValueError('Median filtering requires a nonempty one-dimensional series')
    values = np.where(np.isfinite(values), values, np.nan)
    # Explicit symmetric padding also handles windows larger than the series.
    padded = np.pad(values, (int(window)//2, (int(window)-1)//2), mode='symmetric')
    windows = np.lib.stride_tricks.sliding_window_view(padded, int(window))
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='All-NaN slice encountered')
        return np.nanmedian(windows, axis=1)


def ramp_residuals(data, rate, group_times, groupdq, pixeldq):
    """Single-offset diagnostic using the pipeline slope, not a new ramp fit.

    Exclude DO_NOT_USE, SATURATED, and JUMP_DET groups and DO_NOT_USE pixels.
    Jump offsets in later groups remain visible; this is not JWST's segmented
    likelihood residual or chi-square. A median intercept avoids anchoring to
    a potentially bad first group. All units here are DN and seconds.
    """
    model = np.asarray(group_times)[:, None, None]*rate[None, :, :]
    valid = (np.isfinite(data) & np.isfinite(model) & ((groupdq & 7) == 0)
             & ((pixeldq[None, :, :] & 1) == 0))
    difference = np.where(valid, data-model, np.nan)
    # All-invalid pixels are expected (reference pixels and DQ masks).
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='All-NaN slice encountered')
        intercept = np.nanmedian(difference, axis=0)
    fitted = model+intercept[None, :, :]
    return np.where(valid, data-fitted, np.nan), fitted, valid


def detector_plots(samples, config, folder, fitted=False):
    plot = config['plots']
    fig, axes = plt.subplots(3 if fitted else 2, 3, figsize=(17, 9 if fitted else 6),
                             squeeze=False, layout='constrained')
    limits = plot['rate_limits_dn_per_second'] if fitted else plot['last_minus_first_limits_dn']
    width = plot['asinh_width_dn_per_second'] if fitted else plot['asinh_width_dn']
    norm = AsinhNorm(linear_width=width, vmin=limits[0], vmax=limits[1])
    for column, sample in enumerate(samples):
        image = sample['rate_image'] if fitted else sample['difference_image']
        artist = axes[0, column].imshow(image, origin='lower', aspect='auto',
                                       cmap=plot['cmap'], norm=norm)
        axes[0, column].set(title=f"Integration {sample['integration_number']} ({sample['selection']})",
                            xlabel='Detector column', ylabel='Detector row')
        fig.colorbar(artist, ax=axes[0, column], label='DN/s' if fitted else 'DN')
        for p, (x, y) in enumerate(config['pixels_xy']):
            color = f'C{p}'
            axes[0, column].plot(x, y, '+', color=color, markersize=9)
            axes[1, column].plot(sample['group_times_s'], sample['groups'][:, p],
                                 '.-', color=color, label=f'({x}, {y})')
            if fitted:
                axes[1, column].plot(sample['group_times_s'], sample['fitted'][:, p],
                                     '--', color=color)
                axes[2, column].plot(sample['group_times_s'], sample['residuals'][:, p],
                                     '.-', color=color)
                bad = ~sample['valid'][:, p]
                axes[1, column].plot(sample['group_times_s'][bad], sample['groups'][bad, p],
                                     'x', color='red')
        axes[1, column].set(xlabel='Group time since integration start (s)', ylabel='Signal (DN)',
                            ylim=plot['ramp_limits_dn'])
        axes[1, column].legend(fontsize=8)
        if fitted:
            axes[2, column].axhline(0, color='0.5', lw=.8)
            axes[2, column].set(xlabel='Group time (s)', ylabel='Data − rate × time − offset (DN)',
                                ylim=plot['ramp_residual_limits_dn'])
    fig.suptitle('Ramp fitting: dashed lines use pipeline rates + diagnostic offsets' if fitted
                 else 'Detector calibration: last minus first group')
    name = '02_ramp_fitting.png' if fitted else '01_detector_calibration.png'
    fig.savefig(Path(folder)/name, dpi=plot['dpi'])
    plt.close(fig)


def trace_diagnostics(result, times, config, folder):
    from transitspectroscopy.spectroscopy import get_fwhm
    folder = Path(folder)
    x = result['traces']['x']
    traces = result['traces']['ysmoothed']
    reference = np.nanmedian(traces, axis=0)
    movement = np.nanmedian(traces-reference, axis=1)
    # Use the same background/1-f corrected detector images used in extraction.
    frames, errors = result['tso'], result['tso_err']
    rows = np.arange(frames.shape[1])
    widths = np.full(traces.shape, np.nan)
    radius = config['fwhm_distance_from_trace']
    for t, frame in enumerate(frames):
        for j, column in enumerate(x):
            selected = ((np.abs(rows-traces[t, j]) <= radius) & np.isfinite(frame[:, column])
                        & np.isfinite(errors[t, :, column]) & (errors[t, :, column] > 0))
            if selected.sum() >= 4:
                width = get_fwhm(rows[selected], frame[selected, column])
                if np.isfinite(width) and width > 0:
                    widths[t, j] = width
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='All-NaN slice encountered')
        median_width = np.nanmedian(widths, axis=0)
        width_change = np.nanmedian(widths-median_width, axis=1)
        width_absolute = np.nanmedian(widths, axis=1)
    trace_smooth = smooth_series(movement, config['trace_filter_points'])
    width_smooth = smooth_series(width_change, config['fwhm_filter_points'])
    hours = (times-times[0])*24
    np.savetxt(folder/'trace_shape.txt', np.column_stack([x, reference, median_width]),
               header='column_zero_based median_trace_row median_fwhm_pixels')
    np.savez_compressed(folder/'trace_fwhm.npz', time_bjd_tdb=times, columns=x,
                        trace_row=traces, trace_raw=result['traces']['y'],
                        trace_corrected=result['traces']['ycorrected'],
                        reference_trace_row=reference, fwhm_pixels=widths,
                        median_fwhm_pixels=median_width)
    np.savetxt(folder/'trace_timeseries.csv', np.column_stack([
        times, hours, movement, trace_smooth, width_absolute, width_change, width_smooth]),
        delimiter=',', header='time_bjd_tdb,hours_since_first_loaded,trace_movement_pixels,'
        'trace_median_filter_pixels,median_fwhm_pixels,fwhm_change_pixels,fwhm_median_filter_pixels', comments='')
    fig, ax = plt.subplots(figsize=(14, 4), layout='constrained')
    # Median of raw rates used for tracing, supplied separately by the driver.
    image = result['validation_median_rate']
    plot = config['plots']
    artist = ax.imshow(image, origin='lower', aspect='auto', cmap=plot['cmap'],
        norm=AsinhNorm(linear_width=plot['asinh_width_dn_per_second'],
                       vmin=plot['rate_limits_dn_per_second'][0], vmax=plot['rate_limits_dn_per_second'][1]))
    ax.plot(x, reference, color='cyan', lw=1, label='Median fitted trace')
    ax.set(xlabel='Detector column', ylabel='Detector row', title='Median rate per integration and fitted trace')
    ax.legend()
    fig.colorbar(artist, ax=ax, label='DN/s')
    fig.savefig(folder/'03_trace.png', dpi=plot['dpi']); plt.close(fig)
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True, layout='constrained')
    for ax, series, filtered, label in zip(axes, [movement, width_change],
            [trace_smooth, width_smooth], ['Trace movement (pixel)', 'FWHM change (pixel)']):
        ax.plot(hours, series, '.', ms=3)
        ax.plot(hours, filtered, color='black', lw=1)
        ax.set_ylabel(label)
        ax.set_title(f"RMS about median = {stats(series)['rms_about_median']:.5g} pixel"
                     if np.any(np.isfinite(series)) else 'No valid measurements')
    axes[-1].set_xlabel('Hours since first loaded integration')
    fig.savefig(folder/'04_trace_fwhm_timeseries.png', dpi=plot['dpi']); plt.close(fig)
    return dict(trace_movement_pixels=stats(movement), fwhm_change_pixels=stats(width_change),
                median_fwhm_pixels=stats(width_absolute), fwhm_valid_fraction=float(np.isfinite(widths).mean()),
                trace_shape_pixels=stats(reference),
                trace_filter_points=config['trace_filter_points'], fwhm_filter_points=config['fwhm_filter_points'])


def extraction_diagnostics(extractions, times, config, folder):
    """Compare the same wavelength columns; never turn all-missing flux into zero."""
    folder = Path(folder)
    methods = ['simple', 'polynomial', 'gp']
    variant = config['spectrum_variant']
    if variant not in ('original', 'corrected'):
        raise ValueError('spectrum_variant must be original or corrected')
    reference_wavelength = extractions['simple']['wavelengths']
    common = np.isfinite(reference_wavelength)
    for method in methods:
        spectrum = extractions[method]
        np.testing.assert_allclose(spectrum['wavelengths'], reference_wavelength, equal_nan=True)
        flux, error = spectrum[variant], spectrum[variant+'_err']
        common &= np.all(np.isfinite(flux) & np.isfinite(error) & (error > 0), axis=0)
    if not common.any():
        raise ValueError('No common valid wavelength columns for the three extractions')
    wave = reference_wavelength[common]
    report = dict(spectrum_variant=variant, common_columns=int(common.sum()),
                  wavelength_range_microns=[float(wave.min()), float(wave.max())], methods={})
    medians = {}
    hours = (times-times[0])*24
    fig_s, ax_s = plt.subplots(figsize=(12, 4), layout='constrained')
    fig_w, axes = plt.subplots(2, 3, figsize=(17, 7), sharex=True, layout='constrained')
    for m, method in enumerate(methods):
        spectrum = extractions[method]
        flux, error = spectrum[variant], spectrum[variant+'_err']
        integrated = flux[:, common].sum(axis=1)
        norm = np.median(integrated)
        if not np.isfinite(norm) or norm <= 0:
            raise ValueError(f'{method}: nonpositive white-light normalization')
        lightcurve = integrated/norm
        light_error = np.sqrt((error[:, common]**2).sum(axis=1))/norm
        filtered = smooth_series(lightcurve, config['white_light_filter_points'])
        residual = (lightcurve-filtered)*1e6
        medians[method] = np.nanmedian(flux, axis=0)
        per_column_norm = medians[method]
        normalized = np.divide(flux, per_column_norm, out=np.full_like(flux, np.nan),
                               where=np.isfinite(per_column_norm) & (per_column_norm != 0))
        column_rms = np.sqrt(np.nanmean((normalized-1)**2, axis=0))*1e6
        np.savez_compressed(folder/f'spectra_{method}.npz', time_bjd_tdb=times,
            columns=spectrum['columns'], wavelength_microns=reference_wavelength,
            original_flux=spectrum['original'], original_error=spectrum['original_err'],
            corrected_flux=spectrum['corrected'], corrected_error=spectrum['corrected_err'],
            median_flux=medians[method], fractional_rms_ppm=column_rms, common_white_light_columns=common)
        np.savetxt(folder/f'white_light_{method}.csv', np.column_stack([
            times, hours, integrated, lightcurve, light_error, filtered, residual]), delimiter=',',
            header='time_bjd_tdb,hours_since_first_loaded,flux_dn_per_second,relative_flux,'
                   'relative_error,median_filter,residual_ppm', comments='')
        np.savetxt(folder/f'median_spectrum_{method}.csv', np.column_stack([
            spectrum['columns'], reference_wavelength, medians[method], column_rms, common]),
            delimiter=',', header='column,wavelength_microns,median_flux_dn_per_second,'
                                 'fractional_rms_ppm,common_white_light_column', comments='')
        ax_s.plot(reference_wavelength, medians[method], label=method, alpha=.8, lw=.8)
        rms = stats(residual)['rms_about_zero']
        axes[0, m].plot(hours, lightcurve, '.', ms=3, color=f'C{m}')
        axes[0, m].plot(hours, filtered, color='black', lw=1)
        axes[0, m].set(title=method, ylabel='Relative white-light flux')
        axes[1, m].plot(hours, residual, '.', ms=3, color=f'C{m}')
        axes[1, m].axhline(0, color='black', lw=.7)
        axes[1, m].set(xlabel='Hours since first loaded integration', ylabel='Residual (ppm)',
                        title=f'RMS = {rms:.1f} ppm')
        report['methods'][method] = dict(median_spectrum_dn_per_second=stats(medians[method]),
            spectrum_fractional_rms_ppm=stats(column_rms[common]), white_light=stats(lightcurve),
            white_light_residual_ppm=stats(residual), normalization_dn_per_second=float(norm),
            filter_points=config['white_light_filter_points'])
    for method in methods[1:]:
        difference = medians['simple']-medians[method]
        relative = np.divide(difference, medians['simple'], out=np.full_like(difference, np.nan),
                             where=medians['simple'] != 0)*1e6
        report[f'simple_minus_{method}'] = dict(
            difference_dn_per_second=stats(difference[common]), relative_difference_ppm=stats(relative[common]))
        np.savetxt(folder/f'simple_minus_{method}.csv', np.column_stack([
            reference_wavelength, difference, relative, common]), delimiter=',',
            header='wavelength_microns,difference_dn_per_second,relative_difference_ppm,common_column', comments='')
    ax_s.set(xlabel='Wavelength (µm)', ylabel='Median flux (DN/s)', title=f'NRS1 median spectra ({variant})')
    ax_s.legend()
    fig_s.savefig(folder/'05_median_spectra.png', dpi=config['plots']['dpi']); plt.close(fig_s)
    fig_w.suptitle(f"White-light comparison; {config['white_light_filter_points']}-point reflected median filter")
    fig_w.savefig(folder/'06_white_light.png', dpi=config['plots']['dpi']); plt.close(fig_w)
    return report
