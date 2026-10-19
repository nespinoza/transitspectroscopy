"""Compare the rates with and without the group-difference 1/f correction for one uncal segment.

Stage 1 (``load`` -> ``detector_calibration`` -> ramp fit) is run once, without saving its
products; the first-pass rates (no group-level 1/f correction, as before it became the
default) are saved, the ramps are corrected with ``load.group_difference_1f`` (what
``fit_ramps`` does by default) and the corrected rates are saved too. Both are then extracted
with the ``[extraction]`` variants of the configuration (``ts.run``), and the 100-point
median-filter RMS of every white light is tabulated. Run from the repository root:

    python validation/group_difference_1f_check.py \\
        validation/data/jw01118005001_04101_00001-seg001_nrs1_uncal.fits \\
        --config validation/configs/nrs1-hatp14b.toml \\
        --output validation/runs/nrs1-group-diff-1f
"""

import argparse
import gc
import json
import os
from pathlib import Path
import shutil
import time
import tomllib

import numpy as np

from diagnostics import smooth_series

ROOT = Path(__file__).resolve().parents[1]
VARIANTS = ('first_pass', 'group_difference_1f')


def calibrate(ts, uncal, config, output):
    """Save the first-pass and corrected rates of one segment; return their paths."""
    name = uncal.name.split('_uncal')[0]
    paths = {variant: output/variant/f'{name}_rateints.fits' for variant in VARIANTS}
    if all(path.exists() for path in paths.values()):
        return paths
    calibration = config.get('calibration', {})
    if calibration.get('crds_context'):
        os.environ['CRDS_CONTEXT'] = calibration['crds_context']
    work = output/'work'
    work.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    dataset = ts.jwst.load([str(uncal)], outputfolder=str(work))
    parameters = dict(calibration.get('parameters', {}))
    parameters.update(calibration.get('ramp_fit_parameters', {}))
    dataset.detector_calibration(parameters=parameters,
                                 use_tso_jump=calibration.get('use_tso_jump', True), save=False)
    if dataset.status['group_difference_1f'] is not None:
        raise ValueError(f'{dataset.mode} does not support the group-difference 1/f correction')
    timing = dict(stage1=time.monotonic() - start)
    dataset.run_ramp_fit(save=False, use_saved=False)
    paths['first_pass'].parent.mkdir(parents=True, exist_ok=True)
    dataset.rateints_per_segment[0].save(str(paths['first_pass']))
    timing['first_ramp_fit'] = time.monotonic() - start - sum(timing.values())
    dataset.group_difference_1f(save=False)
    paths['group_difference_1f'].parent.mkdir(parents=True, exist_ok=True)
    dataset.rateints_per_segment[0].save(str(paths['group_difference_1f']))
    timing['correction_and_ramp_fit'] = time.monotonic() - start - sum(timing.values())
    diagnostics = dataset.group_difference_1f_diagnostics
    np.savez_compressed(output/f'{name}_group_difference_1f.npz', **diagnostics)
    record = dict(input=str(uncal), mode=dataset.mode, status=dataset.status,
                  timing_seconds=timing, group_scale=diagnostics['group_scale'].tolist())
    (output/f'{name}_calibration.json').write_text(json.dumps(record, indent=2, default=str)+'\n')
    del dataset
    gc.collect()
    shutil.rmtree(work, ignore_errors=True)
    return paths


def white_lights(stage2):
    """Times and white light of every extraction of a Stage 2 result (SOSS: orders 1 and 2)."""
    if 'orders' not in stage2:
        return {'': (stage2['traces']['times'], stage2['whitelight'])}
    curves = {}
    for order in (1, 2):
        curves[f'order{order}_box'] = (stage2['times'], stage2['orders'][order]['white_light'])
        for method, products in stage2.get('optimal', {}).items():
            curves[f'order{order}_{method}'] = (stage2['times'],
                                                products['orders'][order]['white_light'])
    return curves


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('uncal', type=Path, help='uncal segment')
    parser.add_argument('--config', type=Path, required=True,
                        help='validation configuration ([calibration] and [extraction])')
    parser.add_argument('--output', type=Path, required=True, help='output folder')
    parser.add_argument('--window', type=int, default=100, help='median-filter window (points)')
    args = parser.parse_args(argv)
    os.environ.setdefault('CRDS_PATH', str(Path.home()/'crds_cache'))
    os.environ.setdefault('CRDS_SERVER_URL', 'https://jwst-crds.stsci.edu')
    from validate import installed_package

    ts = installed_package()
    config = tomllib.loads(args.config.read_text())
    paths = calibrate(ts, args.uncal.resolve(), config, args.output.resolve())

    table = {}
    for variant, path in paths.items():
        run = dict(run=dict(name=variant, output=str(args.output.resolve()/f'stage2_{variant}')),
                   data=dict(source='local', glob=str(path), product='rateints'),
                   mode=config.get('mode', dict(name='auto')), extraction=config['extraction'])
        result = ts.run(run, keep_stage2=True)
        table[variant] = {}
        for extraction, stage2 in result.stage2.items():
            for label, (times, white) in white_lights(stage2).items():
                white = np.asarray(white, float)
                residual = (white/smooth_series(white, args.window) - 1)*1e6
                name = f'{extraction}_{label}' if label else extraction
                np.savetxt(args.output/f'white_light_{variant}_{name}.csv',
                           np.column_stack([times, white, residual]),
                           delimiter=',', header='time_bjd_tdb,relative_flux,residual_ppm')
                table[variant][name] = dict(
                    rms_ppm=float(np.sqrt(np.nanmean(residual**2))),
                    robust_rms_ppm=float(1.4826*np.nanmedian(np.abs(residual))))
        del result
        gc.collect()
    (args.output/'white_light_rms.json').write_text(json.dumps(table, indent=2)+'\n')
    print(f'{args.window}-point median-filter RMS of the white light, ppm (robust RMS)')
    for variant in table:
        print(f'  {variant:20s} ' + '  '.join(
            f"{extraction} {values['rms_ppm']:6.1f} ({values['robust_rms_ppm']:5.1f})"
            for extraction, values in table[variant].items()))


if __name__ == '__main__':
    main()
