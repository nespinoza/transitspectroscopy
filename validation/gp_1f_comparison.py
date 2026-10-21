"""Compare the GP (readout-time) 1/f correction of the group differences with the median ones.

One variant per call (a fresh process each, to keep the memory of one Stage 1): Stage 1
(``load`` -> ``detector_calibration``, TSO jump step included) is run without saving its
products, the ramps are fitted once with the JWST ramp fit (first pass: traces and white-light
factors), corrected with ``group_difference_1f_correction`` and fitted again:

- ``groups_jwst``: 1/f levels as column medians of the groups, JWST ramp fit (the default);
- ``differences_sparta``: column medians of the group differences, SPARTA fit;
- ``gp_sparta``: the 1/f of every group difference predicted at every pixel by a GP in readout
  time (`gp_1f.gp_difference_1f`, which replaces ``core.detector.difference_1f``), SPARTA fit
  (``sigma=None`` after the TSO jump step).

The ramps are freed before the rates, marked as group-corrected, are extracted with the
``[extraction]`` variants of the configuration; the rates and Stage 2 products are then deleted
(unless ``--keep-rates``). ``--summarize`` tabulates the variants in the output folder against
``groups_jwst``: white-light RMS with the block-bootstrap error of its paired difference, and
the per-channel RMS ratios of the spectroscopic light curves. Run from the repository root:

    python validation/gp_1f_comparison.py \\
        validation/data/jw01118005001_04101_00001-seg001_nrs1_uncal.fits \\
        --config validation/configs/nrs1-hatp14b.toml --output validation/runs/nrs1-gp-1f \\
        --variant gp_sparta
    python validation/gp_1f_comparison.py --output validation/runs/nrs1-gp-1f --summarize
"""

import argparse
from functools import partial
import gc
import json
import os
from pathlib import Path
import shutil
import time
import tomllib

import numpy as np

from group_difference_1f_check import white_light_rms

VARIANTS = dict(groups_jwst=('groups', 'jwst'), differences_sparta=('differences', 'sparta'),
                gp_sparta=('gp', 'sparta'))
REFERENCE = 'groups_jwst'


def peak_memory_mb():
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20


def calibrate(ts, uncal, config, output, variant):
    """Stage 1, correction and ramp fit of one variant; returns the rate file and a record."""
    import transitspectroscopy.core.detector as detector
    from transitspectroscopy.jwst_models import mark_group_1f
    from gp_1f import gp_difference_1f, readout_ticks

    domain, method = VARIANTS[variant]
    name = uncal.name.split('_uncal')[0]
    calibration = config.get('calibration', {})
    if calibration.get('crds_context'):
        os.environ['CRDS_CONTEXT'] = calibration['crds_context']
    work = output/'work'
    work.mkdir(parents=True, exist_ok=True)
    timing, start = {}, time.monotonic()

    def lap(label):
        timing[label] = time.monotonic() - start - sum(timing.values())

    dataset = ts.jwst.load([str(uncal)], outputfolder=str(work))
    parameters = dict(calibration.get('parameters', {}))
    parameters.update(calibration.get('ramp_fit_parameters', {}))
    dataset.detector_calibration(parameters=parameters,
                                 use_tso_jump=calibration.get('use_tso_jump', True), save=False)
    lap('stage1')
    dataset.run_ramp_fit(save=False, use_saved=False)
    first_pass = dataset.rateints_per_segment
    lap('first_ramp_fit')

    record = dict(input=str(uncal), mode=dataset.mode, variant=variant)
    original = detector.difference_1f
    if domain == 'gp':
        subarray = dataset.ramps_per_segment[0].meta.subarray
        ticks = readout_ticks(dataset.ramps_per_segment[0].data.shape[2:], subarray.fastaxis,
                              subarray.slowaxis)
        record.update(fastaxis=subarray.fastaxis, slowaxis=subarray.slowaxis, gp={})
        detector.difference_1f = partial(gp_difference_1f, ticks=ticks, record=record['gp'])
    try:
        diagnostics = ts.jwst.group_difference_1f_correction(
            dataset.ramps_per_segment, first_pass, dataset.mode_definition,
            domain='groups' if domain == 'groups' else 'differences')
    finally:
        detector.difference_1f = original
    record['shared'] = np.argwhere(diagnostics['shared']).tolist()
    for rateints in first_pass:
        rateints.close()
    del diagnostics, first_pass
    dataset.rateints_per_segment = dataset.rateints = None
    gc.collect()
    lap('correction')

    dataset.ramp_fit_method, dataset.ramp_fit_options = method, {}
    dataset.run_ramp_fit(save=False, use_saved=False)
    if method == 'sparta':
        record['sparta_options'] = dataset.ramp_fit_diagnostics[0]['provenance']['options']
    path = output/variant/f'{name}_rateints.fits'
    path.parent.mkdir(parents=True, exist_ok=True)
    for rateints in dataset.rateints_per_segment:
        mark_group_1f(rateints)
        rateints.save(str(path))
        rateints.close()
    lap('ramp_fit')
    record.update(timing_seconds=timing, stage1_peak_memory_mb=peak_memory_mb())
    del dataset
    gc.collect()
    shutil.rmtree(work, ignore_errors=True)
    return path, record


def run_variant(uncal, config, output, variant, window, keep_rates):
    from validate import installed_package

    ts = installed_package()
    path, record = calibrate(ts, uncal, config, output, variant)
    folder = output/variant
    start = time.monotonic()
    table = white_light_rms(ts, {variant: path}, config, folder, window, spectroscopic=True)
    record['timing_seconds']['stage2'] = time.monotonic() - start
    record['peak_memory_mb'] = peak_memory_mb()
    (folder/'record.json').write_text(json.dumps(record, indent=2, default=str)+'\n')
    if not keep_rates:
        path.unlink()
        shutil.rmtree(folder/f'stage2_{variant}', ignore_errors=True)
    return table


def paired(reference, other, block=20, draws=2000, seed=0):
    """RMS difference (other - reference) of two residual series, with its block-bootstrap
    standard deviation."""
    good = np.isfinite(reference) & np.isfinite(other)
    reference, other = reference[good], other[good]
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, len(reference) - block, (draws, len(reference)//block))
    index = (starts[..., None] + np.arange(block)).reshape(draws, -1)
    rms = lambda values: np.sqrt(np.mean(values**2, axis=-1))
    samples = rms(other[index]) - rms(reference[index])
    return float(rms(other) - rms(reference)), float(np.std(samples))


def summarize(output):
    lines = []
    variants = [v for v in VARIANTS if (output/v/'white_light_rms.json').exists()]
    if REFERENCE not in variants:
        raise FileNotFoundError(f'No {REFERENCE} results in {output}')
    tables = {v: json.loads((output/v/'white_light_rms.json').read_text())[v] for v in variants}
    lines.append('White light, 100-point median-filter RMS (ppm); difference from '
                 f'{REFERENCE} +- block-bootstrap sd')
    for variant in variants:
        row = []
        for extraction, values in tables[variant].items():
            if extraction == 'spectroscopic':
                continue
            curve = lambda v: np.loadtxt(output/v/f'white_light_{v}_{extraction}.csv',
                                         delimiter=',')[:, 2]
            text = f"{extraction} {values['rms_ppm']:6.1f}"
            if variant != REFERENCE:
                difference, sd = paired(curve(REFERENCE), curve(variant))
                text += f' ({difference:+.1f} +- {sd:.1f})'
            row.append(text)
        lines.append(f'  {variant:20s} ' + '  '.join(row))
    lines.append(f'Spectroscopic light curves: median channel RMS (ppm); median and 16-84% of the '
                 f'per-channel RMS ratio to {REFERENCE}')
    for variant in variants:
        for extraction, products in tables[variant].get('spectroscopic', {}).items():
            for product in products:
                if 'original' in product:
                    continue
                row = []
                for power in products[product]:
                    def load(v):
                        file = output/v/f'spectroscopic_{v}_{extraction}_{product}.npz'
                        with np.load(file) as saved:
                            return saved[f'{power}_rms_ppm']
                    text = f"{power} {products[product][power]['median_rms_ppm']:.0f}"
                    if variant != REFERENCE:
                        ratio = load(variant)/load(REFERENCE)
                        low, mid, high = np.percentile(ratio, [16, 50, 84])
                        text += f' ({mid:.3f}, {low:.3f}-{high:.3f})'
                    row.append(text)
                lines.append(f'  {variant:20s} {extraction}/{product}: ' + '  '.join(row))
    for variant in variants:
        record = json.loads((output/variant/'record.json').read_text())
        if 'gp' in record:
            holdout = record['gp'].get('holdout', {})
            kernel = record['gp']['kernel']
            lines.append(f'  {variant}: kernel white {kernel["white"]:.1f} DN^2, correlated '
                         f'{sum(kernel["amplitudes"]):.1f} DN^2 (fit rms {kernel["fit_rms"]:.2f});'
                         ' hold-out band-mean rms (DN) ' + ', '.join(
                             f'{key} {value:.2f}' for key, value in holdout.items()
                             if key != 'column_differences'))
        lines.append(f'  {variant}: peak memory {record.get("peak_memory_mb", 0):.0f} MB, '
                     'seconds ' + ', '.join(f'{k} {v:.0f}'
                                            for k, v in record['timing_seconds'].items()))
    text = '\n'.join(lines)
    (output/'summary.txt').write_text(text+'\n')
    print(text)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('uncal', type=Path, nargs='?', help='uncal segment')
    parser.add_argument('--config', type=Path, help='validation configuration')
    parser.add_argument('--output', type=Path, required=True, help='output folder')
    parser.add_argument('--variant', choices=list(VARIANTS), help='variant to run')
    parser.add_argument('--window', type=int, default=100, help='median-filter window (points)')
    parser.add_argument('--keep-rates', action='store_true',
                        help='keep the rates and Stage 2 products of the variant')
    parser.add_argument('--summarize', action='store_true',
                        help='tabulate the variants in the output folder')
    args = parser.parse_args(argv)
    output = args.output.resolve()
    if args.summarize:
        summarize(output)
        return
    if args.uncal is None or args.config is None or args.variant is None:
        parser.error('uncal, --config and --variant are needed to run a variant')
    os.environ.setdefault('CRDS_PATH', str(Path.home()/'crds_cache'))
    os.environ.setdefault('CRDS_SERVER_URL', 'https://jwst-crds.stsci.edu')
    output.mkdir(parents=True, exist_ok=True)
    config = tomllib.loads(args.config.read_text())
    run_variant(args.uncal.resolve(), config, output, args.variant, args.window, args.keep_rates)


if __name__ == '__main__':
    main()
