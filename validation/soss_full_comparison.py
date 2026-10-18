"""Compare full-exposure SOSS reductions: light curves, transit fits, residuals and spectra.

Every reduction folder holds the output of ``validate.py`` for a full SOSS exposure (white
light and transit fit of the box extraction at the top level, of every optimal extraction in
``optimal/<method>/``, and the transmission spectra). For every order and extraction this
writes the detrended light curves with the transit models, their residuals, the RMS of the
residuals against bin size, the ratio of every white light to that of the first reduction,
and the transmission spectra with their differences; a table summarises the median-filter
RMS (all integrations, per segment, without the worst integration), the transit-fit residual
RMS, beta and the white-light depths. Run from the repository root:

    python validation/soss_full_comparison.py validation/runs/soss-hatp14b-full-1f \\
        group_diff=validation/runs/soss-hatp14b-full-1f/s2_group_diff_template \\
        group_1f=validation/runs/soss-hatp14b-full-1f/s2_g1f_template
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from diagnostics import smooth_series

METHODS = (('box', ''), ('polynomial', 'optimal/polynomial/'), ('gp', 'optimal/gp/'))
ORDERS = ('1', '2')
COLORS = ('tab:blue', 'tab:red', 'tab:green', 'tab:purple', 'tab:orange', 'tab:brown')


def table(path):
    """A CSV written by validate.py as a structured array."""
    return np.genfromtxt(path, delimiter=',', names=True)


def binned(values, size):
    """Means of consecutive groups of ``size`` points."""
    n = len(values)//size
    return np.nanmean(values[:n*size].reshape(n, size), axis=1)


def binned_rms(values, sizes):
    """RMS of the residuals binned in consecutive groups of every size."""
    return np.array([np.nanstd(binned(values, size)) for size in sizes])


def rms(values):
    return float(np.sqrt(np.nanmean(values**2)))


def load_reduction(folder, window, segments):
    """White light, median-filter residuals, transit fit and spectrum of every order and method."""
    folder = Path(folder)
    curves = {}
    for order in ORDERS:
        curves[order] = {}
        for method, path in METHODS:
            white = table(folder/f'{path}order{order}_white_light.csv')
            flux = white['relative_flux']/np.nanmedian(white['relative_flux'])
            filtered = (flux/smooth_series(flux, window) - 1)*1e6
            fit = table(folder/f'{path}order{order}_transit_fit.csv')
            spectrum = table(folder/f'{path}order{order}_transmission.csv')
            ok = spectrum['fit_failed'] == 0
            summary = json.loads((folder/f'{path}order{order}_transmission_fits.json').read_text())
            curves[order][method] = dict(time=white['time_bjd_tdb'], flux=flux, filtered=filtered,
                                         fit=fit, spectrum=spectrum[ok],
                                         white_fit=white_light_fit(folder, method, order))
    return curves


def white_light_fit(folder, method, order):
    """The white-light transit fit summary of one extraction, from the validation report."""
    report = json.loads((Path(folder)/'reduction.json').read_text())
    if method == 'box':
        return report['orders'][order]['transit_fit']
    return report['optimal'][method]['orders'][order]['transit_fit']


def statistics(curves, segments):
    """RMS and transit-fit summary of every order and method."""
    result = {}
    for order in curves:
        result[order] = {}
        for method, curve in curves[order].items():
            filtered = curve['filtered']
            worst = int(np.nanargmax(np.abs(filtered)))
            bounds = np.cumsum([0] + list(segments))
            fit = curve['white_fit']
            result[order][method] = dict(
                filter_rms_ppm=rms(filtered),
                filter_rms_without_worst_ppm=rms(np.delete(filtered, worst)),
                worst_integration=worst, worst_residual_ppm=float(filtered[worst]),
                segment_filter_rms_ppm=[rms(filtered[a:b])
                                        for a, b in zip(bounds[:-1], bounds[1:])],
                transit_residual_rms_ppm=rms(curve['fit']['residual_ppm']),
                transit_beta=fit['beta'], depth_ppm=fit['depth_ppm'],
                depth_error_ppm=fit['depth_error_ppm'],
                spectrum_median_error_ppm=float(np.nanmedian(curve['spectrum']['depth_error_ppm'])),
                spectrum_median_residual_rms_ppm=float(
                    np.nanmedian(curve['spectrum']['residual_rms_ppm'])),
            )
    return result


def hours(time, reference):
    return (time - reference)*24


def plot_lightcurves(reductions, order, bin_size, output):
    """Detrended light curves with the transit models, and their residuals, per extraction."""
    names = list(reductions)
    reference = reductions[names[0]][order]['box']['time'][0]
    figure, axes = plt.subplots(len(METHODS), 2, figsize=(14, 4*len(METHODS)),
                                constrained_layout=True, sharex=True)
    for row, (method, _) in enumerate(METHODS):
        for k, name in enumerate(names):
            fit = reductions[name][order][method]['fit']
            t = hours(fit['time_bjd_tdb'], reference)
            offset = -0.003*k
            axes[row, 0].plot(t, fit['normalized_flux'] + offset, '.', ms=2, alpha=0.4,
                              color=COLORS[k])
            axes[row, 0].plot(binned(t, bin_size), binned(fit['normalized_flux'], bin_size)
                              + offset, 'o', ms=3, color=COLORS[k], label=name)
            axes[row, 0].plot(t, fit['model'] + offset, '-', color='k', lw=1)
            residual = fit['residual_ppm']
            axes[row, 1].plot(t, residual - 1500*k, '.', ms=2, alpha=0.4, color=COLORS[k])
            axes[row, 1].plot(binned(t, bin_size), binned(residual, bin_size) - 1500*k, 'o',
                              ms=3, color=COLORS[k],
                              label=f'{name}: {rms(residual):.0f} ppm '
                                    f'({rms(binned(residual, bin_size)):.0f} binned)')
        axes[row, 0].set(title=f'Order {order}, {method}: detrended white light and transit model',
                         ylabel='Normalised flux (offset)')
        axes[row, 1].set(title=f'Order {order}, {method}: residuals ({bin_size}-integration bins)',
                         ylabel='Residual (ppm, offset by 1500)')
        for axis in axes[row]:
            axis.legend(fontsize=8, loc='lower left')
    for axis in axes[-1]:
        axis.set_xlabel('Hours from the first integration')
    figure.savefig(output, dpi=120)
    plt.close(figure)


def plot_binning(reductions, output):
    """RMS of the transit-fit residuals against bin size, with the white-noise expectation."""
    figure, axes = plt.subplots(len(ORDERS), len(METHODS), figsize=(15, 8.5),
                                constrained_layout=True)
    for row, order in enumerate(ORDERS):
        for column, (method, _) in enumerate(METHODS):
            axis = axes[row, column]
            for k, name in enumerate(reductions):
                residual = reductions[name][order][method]['fit']['residual_ppm']
                sizes = np.unique(np.logspace(0, np.log10(len(residual)//5), 18).astype(int))
                values = binned_rms(residual, sizes)
                axis.loglog(sizes, values, 'o-', ms=3, color=COLORS[k], label=name)
                axis.loglog(sizes, values[0]/np.sqrt(sizes), ':', color=COLORS[k])
            axis.set(title=f'Order {order}, {method}', xlabel='Bin size (integrations)',
                     ylabel='Residual RMS (ppm)')
            axis.legend(fontsize=8)
    figure.suptitle('Transit-fit residuals against bin size (dotted: white noise)')
    figure.savefig(output, dpi=120)
    plt.close(figure)


def plot_ratios(reductions, segments, output):
    """Every white light divided by that of the first reduction (same extraction)."""
    names = list(reductions)
    figure, axes = plt.subplots(len(METHODS), len(ORDERS), figsize=(14, 10),
                                constrained_layout=True, sharex=True)
    for column, order in enumerate(ORDERS):
        base = reductions[names[0]][order]
        reference = base['box']['time'][0]
        for row, (method, _) in enumerate(METHODS):
            axis = axes[row, column]
            t = hours(base[method]['time'], reference)
            limits = []
            for k, name in enumerate(names[1:], start=1):
                ratio = (reductions[name][order][method]['flux']/base[method]['flux'] - 1)*1e6
                robust = 1.4826*np.nanmedian(np.abs(ratio - np.nanmedian(ratio)))
                axis.plot(t, ratio, '.', ms=2, color=COLORS[k],
                          label=f'{name} / {names[0]}: RMS {np.nanstd(ratio):.0f} ppm '
                                f'(robust {robust:.0f}; worst #{np.nanargmax(np.abs(ratio))} '
                                f'{ratio[np.nanargmax(np.abs(ratio))]:+.0f})')
                limits.append(np.nanpercentile(np.abs(ratio), 99.5))
            for boundary in np.cumsum(segments)[:-1]:
                axis.axvline(t[boundary], color='0.6', ls='--', lw=0.8)
            limit = 1.5*max(limits)
            axis.set(title=f'Order {order}, {method} (points beyond the axes not shown)',
                     ylabel='Ratio - 1 (ppm)', ylim=(-limit, limit))
            axis.legend(fontsize=8)
    for axis in axes[-1]:
        axis.set_xlabel('Hours from the first integration (dashed: segment boundaries)')
    figure.savefig(output, dpi=120)
    plt.close(figure)


def plot_spectra(reductions, output, tolerance=2e-3):
    """Transmission spectra of every reduction and their difference from the first one.

    Only the bins with uncertainties below three times the median of their order are shown
    (this drops the order-2 bins beyond the order-1 overlap); the bins of every reduction
    are matched to those of the first by wavelength, within ``tolerance`` (microns).
    """
    names = list(reductions)
    figure, axes = plt.subplots(2, len(METHODS), figsize=(16, 8), constrained_layout=True,
                                height_ratios=(2, 1), sharex=True)
    for column, (method, _) in enumerate(METHODS):
        for order in ORDERS:
            base = reductions[names[0]][order][method]['spectrum']
            good = base['depth_error_ppm'] < 3*np.nanmedian(base['depth_error_ppm'])
            base = base[good]
            for k, name in enumerate(names):
                spectrum = reductions[name][order][method]['spectrum']
                spectrum = spectrum[spectrum['depth_error_ppm']
                                    < 3*np.nanmedian(spectrum['depth_error_ppm'])]
                shift = 0.004*(k - (len(names) - 1)/2)
                axes[0, column].errorbar(spectrum['wavelength_microns'] + shift,
                                         spectrum['depth_ppm'], spectrum['depth_error_ppm'],
                                         fmt='o' if order == '1' else 's', ms=3, lw=0.8,
                                         color=COLORS[k], alpha=0.8,
                                         label=name if order == '1' else None)
                if k == 0 or len(spectrum) == 0:
                    continue
                index = np.clip(np.searchsorted(spectrum['wavelength_microns'],
                                                base['wavelength_microns']), 1, len(spectrum) - 1)
                nearest = np.where(np.abs(spectrum['wavelength_microns'][index - 1]
                                          - base['wavelength_microns'])
                                   < np.abs(spectrum['wavelength_microns'][index]
                                            - base['wavelength_microns']), index - 1, index)
                match = np.abs(spectrum['wavelength_microns'][nearest]
                               - base['wavelength_microns']) < tolerance
                difference = spectrum['depth_ppm'][nearest[match]] - base['depth_ppm'][match]
                axes[1, column].errorbar(base['wavelength_microns'][match] + shift, difference,
                                         base['depth_error_ppm'][match], fmt='o' if order == '1'
                                         else 's', ms=3, lw=0.6, color=COLORS[k], alpha=0.8)
        axes[0, column].set(title=f'{method} (circles: order 1; squares: order 2)',
                            ylabel='Transit depth (ppm)')
        axes[0, column].legend(fontsize=8)
        axes[1, column].axhline(0, color='k', lw=0.8)
        axes[1, column].set(xlabel='Wavelength (microns)',
                            ylabel=f'Difference from {names[0]} (ppm)\n(bars: its uncertainty)')
    figure.savefig(output, dpi=120)
    plt.close(figure)


def plot_summary(stats, output):
    """Median-filter RMS, transit-fit residual RMS and median spectral uncertainty."""
    names = list(stats)
    keys = (('filter_rms_ppm', 'Median-filter RMS, all integrations'),
            ('filter_rms_without_worst_ppm', 'Median-filter RMS, without the worst integration'),
            ('transit_residual_rms_ppm', 'Transit-fit residual RMS'),
            ('spectrum_median_error_ppm', 'Median transmission uncertainty'))
    figure, axes = plt.subplots(len(ORDERS), len(keys), figsize=(18, 8), constrained_layout=True)
    width = 0.8/len(names)
    for row, order in enumerate(ORDERS):
        for column, (key, title) in enumerate(keys):
            axis = axes[row, column]
            for k, name in enumerate(names):
                values = [stats[name][order][method][key] for method, _ in METHODS]
                axis.bar(np.arange(len(METHODS)) + (k - (len(names) - 1)/2)*width, values, width,
                         color=COLORS[k], label=name)
            axis.set_xticks(np.arange(len(METHODS)), [method for method, _ in METHODS])
            axis.set(title=f'Order {order}: {title}', ylabel='ppm')
            low = min(stats[name][order][method][key] for name in names for method, _ in METHODS)
            axis.set_ylim(0.8*low, None)
            axis.legend(fontsize=7)
    figure.savefig(output, dpi=120)
    plt.close(figure)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('output', type=Path, help='folder for the figures and the table')
    parser.add_argument('reductions', nargs='+', help='name=folder of every reduction')
    parser.add_argument('--window', type=int, default=100, help='median-filter window (points)')
    parser.add_argument('--bin', type=int, default=10, help='bin size of the plotted residuals')
    parser.add_argument('--segments', type=int, nargs='+', default=[144, 144, 144, 140],
                        help='integrations of every segment')
    args = parser.parse_args(argv)
    reductions = {}
    for item in args.reductions:
        name, folder = item.split('=', 1)
        reductions[name] = load_reduction(folder, args.window, args.segments)
    stats = {name: statistics(curves, args.segments) for name, curves in reductions.items()}
    args.output.mkdir(parents=True, exist_ok=True)
    for order in ORDERS:
        plot_lightcurves(reductions, order, args.bin, args.output/f'lightcurves_order{order}.png')
    plot_binning(reductions, args.output/'residual_rms_binning.png')
    plot_ratios(reductions, args.segments, args.output/'white_light_ratios.png')
    plot_spectra(reductions, args.output/'transmission_spectra.png')
    plot_summary(stats, args.output/'precision_summary.png')
    (args.output/'comparison.json').write_text(json.dumps(stats, indent=2)+'\n')

    lines = [f'{"order/method":14s} {"reduction":16s} {"filter":>7s} {"no worst":>8s} '
             f'{"worst":>13s} {"per segment":>24s} {"transit":>7s} {"beta":>5s} '
             f'{"depth (ppm)":>15s} {"spec err":>8s}']
    for order in ORDERS:
        for method, _ in METHODS:
            for name in stats:
                s = stats[name][order][method]
                segment = '/'.join(f'{value:.0f}' for value in s['segment_filter_rms_ppm'])
                lines.append(f'{order + " " + method:14s} {name:16s} {s["filter_rms_ppm"]:7.1f} '
                             f'{s["filter_rms_without_worst_ppm"]:8.1f} '
                             f'{"#" + str(s["worst_integration"]):>5s} '
                             f'{s["worst_residual_ppm"]:+7.0f} {segment:>24s} '
                             f'{s["transit_residual_rms_ppm"]:7.1f} {s["transit_beta"]:5.2f} '
                             f'{s["depth_ppm"]:7.0f} ± {s["depth_error_ppm"]:4.0f} '
                             f'{s["spectrum_median_error_ppm"]:8.0f}')
    text = '\n'.join(lines)
    (args.output/'comparison.txt').write_text(text+'\n')
    print(text)


if __name__ == '__main__':
    main()
