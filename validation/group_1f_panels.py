"""Images of the group-level 1/f correction (median group-difference templates) of one segment.

Stage 1 (``load`` -> ``detector_calibration``, TSO jump step included) and the first-pass
JWST ramp fit are run once, without saving their products, and the ramps are corrected with
``group_difference_1f_correction`` (``domain='groups'``, the default of ``fit_ramps``), whose
call of ``core.detector.group_difference_1f`` is recorded. For a few integrations, a figure
shows, for every group ``k``,

- (a) the group,
- (b) the group minus its template (the median of group ``k+1`` minus group ``k``, scaled to
  group ``k`` and by the white-light factor of the integration),
- (c) the 1/f component: the column medians of (b) over the background pixels (outlined in
  white in (b)), and
- (d) the group minus (c), i.e., the corrected group,

with the colour scale set by the background of (b) (centred on the background median of every
panel; (c) uses the centre of (b)), and the column medians of the four over the background
pixels, where the 1/f pattern stands above the pixel noise. The levels recomputed from the
recorded templates are checked against the removed ones, and (a) - (c) against the corrected groups. The recorded
arrays are saved (``panels.npz``), so ``--replot`` makes the figures again without Stage 1.
Run from the repository root:

    python validation/group_1f_panels.py \\
        validation/data/jw01118005001_04101_00001-seg001_nrs1_uncal.fits \\
        --config validation/configs/nrs1-hatp14b.toml --output validation/runs/nrs1-1f-panels
"""

import argparse
import json
import os
from pathlib import Path
import shutil
import tomllib
import warnings

import numpy as np

PANELS = ('group', 'group minus template', '1/f component', 'group minus 1/f component')


def differences_of(segments, template_valid):
    """Median group differences, from which ``core.detector.group_difference_1f`` makes the
    templates (group ``k`` uses difference ``min(k, ngroups - 2)``)."""
    differences = []
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        for k in range(segments[0].shape[1] - 1):
            values = [np.where(good[:, k] & good[:, k+1], data[:, k+1] - data[:, k], np.nan)
                      .astype(np.float32) for data, good in zip(segments, template_valid)]
            differences.append(np.nanmedian(np.concatenate(values), axis=0))
    return differences


def record(integrations):
    """Wrap ``core.detector.group_difference_1f`` to keep the arrays of ``integrations``."""
    import transitspectroscopy.core.detector as detector
    original, recorded = detector.group_difference_1f, {}

    def wrapper(segments, valid, background_pixels, scale, scale_pixels, template_valid=None,
                shared_fraction=0.5):
        clean = valid if template_valid is None else template_valid
        starts = np.cumsum([0] + [len(data) for data in segments])
        where = [(np.searchsorted(starts, g, side='right') - 1, g) for g in integrations]
        where = [(s, g - starts[s]) for s, g in where]
        before = np.array([segments[s][i] for s, i in where], dtype=np.float32)
        differences = differences_of(segments, clean)
        result = original(segments, valid, background_pixels, scale, scale_pixels,
                          template_valid=template_valid, shared_fraction=shared_fraction)
        recorded.update(
            integrations=np.asarray(integrations), before=before,
            after=np.array([segments[s][i] for s, i in where], dtype=np.float32),
            valid=np.array([valid[s][i] for s, i in where]),
            clean=np.array([clean[s][i] for s, i in where]),
            background_pixels=background_pixels, scale_pixels=scale_pixels,
            scale=np.asarray(scale)[integrations], levels=result['levels'][integrations],
            shared=result['shared'][integrations], group_scale=result['group_scale'],
            templates=np.array([result['group_scale'][k] * differences[min(k, len(differences) - 1)]
                                for k in range(len(differences) + 1)], dtype=np.float32))
        return result

    detector.group_difference_1f = wrapper
    return recorded, lambda: setattr(detector, 'group_difference_1f', original)


def calibrate(ts, uncal, config, output, integrations):
    calibration = config.get('calibration', {})
    if calibration.get('crds_context'):
        os.environ['CRDS_CONTEXT'] = calibration['crds_context']
    work = output/'work'
    work.mkdir(parents=True, exist_ok=True)
    dataset = ts.jwst.load([str(uncal)], outputfolder=str(work))
    parameters = dict(calibration.get('parameters', {}))
    parameters.update(calibration.get('ramp_fit_parameters', {}))
    dataset.detector_calibration(parameters=parameters,
                                 use_tso_jump=calibration.get('use_tso_jump', True), save=False)
    dataset.run_ramp_fit(save=False, use_saved=False)
    recorded, restore = record(integrations)
    try:
        ts.jwst.group_difference_1f_correction(dataset.ramps_per_segment,
                                               dataset.rateints_per_segment,
                                               dataset.mode_definition, domain='groups')
    finally:
        restore()
    recorded['mode'] = dataset.mode
    shutil.rmtree(work, ignore_errors=True)
    return recorded


def check(recorded):
    """Largest differences of the recomputed levels and of (a) - (c) from the removed ones."""
    level_error = correction_error = 0.0
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        for j, scale in enumerate(recorded['scale']):
            for k, template in enumerate(recorded['templates']):
                residual = recorded['before'][j, k] - scale * template
                use = recorded['valid'][j, k] & recorded['background_pixels']
                if not recorded['shared'][j, k]:
                    use &= recorded['clean'][j, k]
                level = np.nanmedian(np.where(use, residual, np.nan), axis=0)
                level = np.where(np.isfinite(level), level, 0.0)
                level_error = max(level_error,
                                  np.max(np.abs(level - recorded['levels'][j, k])))
                corrected = recorded['before'][j, k] - recorded['levels'][j, k][None, :]
                correction_error = max(correction_error, np.nanmax(
                    np.abs(corrected - recorded['after'][j, k])))
    return dict(level_error_dn=float(level_error), correction_error_dn=float(correction_error))


def robust(values):
    values = values[np.isfinite(values)]
    median = np.median(values)
    return median, 1.4826 * np.median(np.abs(values - median))


def column_profile(image, use):
    """Median of every column over the pixels ``use``."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        return np.nanmedian(np.where(use, image, np.nan), axis=0)


def high_pass_rms(profile, window=51):
    """RMS of a column profile about its running median (column-to-column 1/f)."""
    from scipy.ndimage import median_filter
    good = np.isfinite(profile)
    values = profile[good]
    return float(np.std(values - median_filter(values, window, mode='nearest')))


def plot(recorded, output, nsigma=2.0):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    background = recorded['background_pixels']
    nrows, ncols = background.shape
    ngroups = recorded['before'].shape[1]
    # One column of panels for wide, short subarrays (NIRSpec), two otherwise (SOSS), and the
    # background column profiles below the images of every group:
    columns = 1 if ncols / nrows > 20 else 2
    lines = 4 // columns
    width = 16.0
    height, profile_height = max(0.45, width / columns * nrows / ncols), 1.3
    ratios = ([height] * lines + [profile_height]) * ngroups
    paths = []
    for j, integration in enumerate(recorded['integrations']):
        figure = plt.figure(figsize=(width, sum(ratios) + 0.35 * len(ratios) + 0.6))
        grid = figure.add_gridspec(len(ratios), columns, height_ratios=ratios)
        for k in range(ngroups):
            good = recorded['valid'][j, k]
            residual = recorded['before'][j, k] - recorded['scale'][j] * recorded['templates'][k]
            residual = np.where(good, residual, np.nan)
            level = np.broadcast_to(recorded['levels'][j, k][None, :], (nrows, ncols))
            images = (recorded['before'][j, k], residual, level, recorded['after'][j, k])
            centre_b, sigma = robust(residual[background])
            shared = ' (shared)' if recorded['shared'][j, k] else ''
            first = k * (lines + 1)
            for p, (title, image) in enumerate(zip(PANELS, images)):
                axis = figure.add_subplot(grid[first + p // columns, p % columns])
                centre = centre_b if p == 2 else robust(image[background & good])[0]
                axis.imshow(image, origin='lower', aspect='auto', cmap='viridis',
                            vmin=centre - nsigma * sigma, vmax=centre + nsigma * sigma,
                            interpolation='nearest')
                if p == 1:
                    axis.contour(background, levels=[0.5], colors='white', linewidths=0.4)
                axis.set_title(f'group {k + 1}{shared}: ({"abcd"[p]}) {title}, '
                               f'centre {centre:.1f} ± {nsigma:g}×{sigma:.1f} DN',
                               fontsize=7, loc='left', pad=2)
                axis.tick_params(labelsize=6)
                axis.set_xticklabels([])
            # Column medians over the background pixels, offset for clarity:
            axis = figure.add_subplot(grid[first + lines, :])
            use = background & good
            profiles = dict(a=column_profile(images[0], use), b=column_profile(residual, use),
                            c=recorded['levels'][j, k], d=column_profile(images[3], use))
            step = 4 * np.std(recorded['levels'][j, k])
            for offset, (key, profile) in enumerate(profiles.items()):
                axis.plot(profile - np.nanmedian(profile) - offset * step, lw=0.5,
                          label=f'({key}) high-pass rms {high_pass_rms(profile):.1f} DN')
            axis.set_xlim(0, ncols - 1)
            axis.set_title(f'group {k + 1}: background column medians (offset by '
                           f'{step:.0f} DN)', fontsize=7, loc='left', pad=2)
            axis.legend(fontsize=6, ncol=4, loc='upper right', frameon=False)
            axis.tick_params(labelsize=6)
            if k < ngroups - 1:
                axis.set_xticklabels([])
        top = 1 - 0.5 / figure.get_figheight()
        figure.suptitle(f'{recorded["mode"]}, integration {integration} (0-based), '
                        f'white-light factor {recorded["scale"][j]:.4f}', fontsize=10,
                        y=1 - 0.15 / figure.get_figheight(), va='top')
        figure.tight_layout(rect=(0, 0, 1, top), h_pad=0.3)
        path = output/f'group_1f_integration{integration:04d}.png'
        figure.savefig(path, dpi=110)
        plt.close(figure)
        paths.append(path)
    return paths


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('uncal', type=Path, nargs='?', help='uncal segment')
    parser.add_argument('--config', type=Path, help='validation configuration ([calibration])')
    parser.add_argument('--output', type=Path, required=True, help='output folder')
    parser.add_argument('--integrations', type=int, nargs='+', default=[10, 100, 300],
                        help='0-based integrations of the segment to show')
    parser.add_argument('--replot', action='store_true',
                        help='make the figures from the saved panels.npz')
    args = parser.parse_args(argv)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if args.replot:
        with np.load(output/'panels.npz') as saved:
            recorded = {key: saved[key] for key in saved.files}
        recorded['mode'] = str(recorded['mode'])
    else:
        os.environ.setdefault('CRDS_PATH', str(Path.home()/'crds_cache'))
        os.environ.setdefault('CRDS_SERVER_URL', 'https://jwst-crds.stsci.edu')
        from validate import installed_package

        config = tomllib.loads(args.config.read_text())
        recorded = calibrate(installed_package(), args.uncal.resolve(), config, output,
                             args.integrations)
        np.savez_compressed(output/'panels.npz', **recorded)
    checks = check(recorded)
    (output/'checks.json').write_text(json.dumps(checks, indent=2)+'\n')
    print(checks)
    for path in plot(recorded, output):
        print(path)


if __name__ == '__main__':
    main()
