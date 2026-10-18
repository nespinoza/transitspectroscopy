"""Compare the white-light scatter of one-segment SOSS reductions (e.g., Stage 1 variants).

Every reduction folder holds the output of ``validate.py`` (``order<N>_white_light.csv``
for the box extraction and ``optimal/<method>/order<N>_white_light.csv``). For each, the
white-light curve is divided by a 100-point median filter (as in the SOSS validation), and
the RMS of the residuals is tabulated and plotted, also as a function of bin size against
the white-noise expectation. Run from the repository root:

    python validation/soss_segment_comparison.py validation/runs/soss-seg1-comparison \\
        --reference archive=validation/runs/soss_prompt5_runner
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
COLORS = dict(box='tab:blue', polynomial='tab:orange', gp='tab:green')


def white_light(folder, method_path, order, integrations=None):
    """Time and relative flux of one white-light curve (optionally only its first integrations)."""
    table = np.genfromtxt(Path(folder)/f'{method_path}order{order}_white_light.csv',
                          delimiter=',', names=True)
    time, flux = table['time_bjd_tdb'], table['relative_flux']
    if integrations is not None:
        time, flux = time[:integrations], flux[:integrations]
    return time, flux/np.nanmedian(flux)


def residuals(flux, window):
    """Residuals (ppm) about the median filter, and the filter itself."""
    model = smooth_series(flux, window)
    return (flux/model - 1)*1e6, model


def binned_rms(values, sizes):
    """RMS of the residuals binned in consecutive groups of ``size`` points."""
    rms = []
    for size in sizes:
        n = len(values)//size
        rms.append(np.nanstd(values[:n*size].reshape(n, size).mean(axis=1)))
    return np.array(rms)


def plot_reduction(name, curves, window, output):
    """White light with the median filter, residuals, and RMS against bin size."""
    figure, axes = plt.subplots(3, 2, figsize=(12, 10), constrained_layout=True)
    for column, order in enumerate(sorted(curves)):
        for row, method in enumerate(curves[order]):
            time, flux, residual, model = curves[order][method]
            hours = (time - time[0])*24
            offset = -0.004*row
            axes[0, column].plot(hours, flux + offset, '.', ms=3, color=COLORS[method], label=method)
            axes[0, column].plot(hours, model + offset, '-', color='k', lw=1)
            axes[1, column].plot(hours, residual - 1500*row, '.', ms=3, color=COLORS[method],
                                 label=f'{method}: {np.sqrt(np.nanmean(residual**2)):.0f} ppm')
            sizes = np.unique(np.logspace(0, np.log10(len(residual)//4), 15).astype(int))
            rms = binned_rms(residual, sizes)
            axes[2, column].loglog(sizes, rms, 'o-', color=COLORS[method], ms=3, label=method)
            axes[2, column].loglog(sizes, rms[0]/np.sqrt(sizes), ':', color=COLORS[method])
        axes[0, column].set(title=f'{name}: order {order}', ylabel='Relative flux (offset)',
                            xlabel='Hours from first integration')
        axes[1, column].set(ylabel=f'Residual about {window}-point median (ppm, offset)',
                            xlabel='Hours from first integration')
        axes[2, column].set(xlabel='Bin size (integrations)', ylabel='RMS (ppm)',
                            title='Dotted: white-noise expectation')
        for axis in axes[:, column]:
            axis.legend(fontsize=8)
    figure.savefig(output, dpi=120)
    plt.close(figure)


def summary_plot(table, output):
    """RMS of every reduction, extraction and order."""
    names = list(table)
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True, sharey=False)
    width = 0.8/len(METHODS)
    for column, order in enumerate(('1', '2')):
        for k, (method, _) in enumerate(METHODS):
            values = [table[name][order][method]['rms_ppm'] for name in names]
            axes[column].bar(np.arange(len(names)) + (k - 1)*width, values, width,
                             color=COLORS[method], label=method)
        axes[column].set_xticks(np.arange(len(names)), names, rotation=30, ha='right', fontsize=8)
        axes[column].set(title=f'Order {order}: RMS about the median filter', ylabel='RMS (ppm)')
        axes[column].legend(fontsize=8)
    figure.savefig(output, dpi=120)
    plt.close(figure)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('folder', type=Path, help='folder with one subfolder per reduction')
    parser.add_argument('--reference', action='append', default=[],
                        help='name=folder of a full-exposure reduction; its first integrations are compared')
    parser.add_argument('--window', type=int, default=100, help='median-filter window (points)')
    args = parser.parse_args(argv)

    reductions = {path.name: (path, None) for path in sorted(args.folder.iterdir())
                  if (path/'order1_white_light.csv').exists()}
    integrations = len(white_light(next(iter(reductions.values()))[0], '', 1)[0])
    for reference in args.reference:
        name, folder = reference.split('=', 1)
        reductions[name] = (Path(folder), integrations)

    table = {}
    for name, (folder, first) in reductions.items():
        curves, table[name] = {}, {}
        for order in ('1', '2'):
            curves[order], table[name][order] = {}, {}
            for method, path in METHODS:
                time, flux = white_light(folder, path, order, first)
                residual, model = residuals(flux, args.window)
                curves[order][method] = (time, flux, residual, model)
                worst = int(np.nanargmax(np.abs(residual)))
                table[name][order][method] = dict(
                    rms_ppm=float(np.sqrt(np.nanmean(residual**2))),
                    robust_rms_ppm=float(1.4826*np.nanmedian(np.abs(residual))),
                    worst_integration=worst,
                    worst_residual_ppm=float(residual[worst]),
                )
        plot_reduction(name, curves, args.window, args.folder/f'white_light_rms_{name}.png')
    summary_plot(table, args.folder/'white_light_rms_summary.png')
    (args.folder/'white_light_rms.json').write_text(json.dumps(table, indent=2)+'\n')

    print(f'{args.window}-point median-filter RMS of the white light, ppm (robust RMS; worst integration)')
    for order in ('1', '2'):
        print(f'Order {order}:')
        for name in table:
            cells = [f"{method} {table[name][order][method]['rms_ppm']:6.1f} "
                     f"({table[name][order][method]['robust_rms_ppm']:5.1f}; "
                     f"#{table[name][order][method]['worst_integration']} "
                     f"{table[name][order][method]['worst_residual_ppm']:+.0f})"
                     for method, _ in METHODS]
            print(f'  {name:22s} ' + '  '.join(cells))


if __name__ == '__main__':
    main()
