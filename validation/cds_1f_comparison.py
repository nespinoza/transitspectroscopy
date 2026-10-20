"""Compare 1/f corrections on the groups and on the group differences (CDS frames).

Stage 1 (``load`` -> ``detector_calibration``, TSO jump step included) is run once on one uncal
segment, without saving its products, and the ramps are fitted once with the JWST ramp fit (the
first pass, which gives the traces and white-light factors of both corrections). Then the ramps
are corrected with ``group_difference_1f_correction``

- ``domain='groups'``: 1/f levels measured on the groups (the default of ``fit_ramps``), and
- ``domain='differences'``: 1/f levels measured on the group differences,

and each corrected set of ramps is fitted with the JWST ramp fit and with the correlated
(SPARTA, arXiv:2305.06240) fit, which uses the group differences directly. The standard pipeline
is ``groups_jwst``; the CDS one is ``differences_sparta``. The rates of every variant are marked
as group-corrected (Stage 2 then removes no 1/f noise), extracted with the ``[extraction]``
variants of the configuration (``ts.run``) and deleted (unless ``--keep-rates``), and the
100-point median-filter RMS of every white light and spectroscopic light curve (columns
and R = 30, 100, 300 bins) is tabulated. Run from the repository root:

    python validation/cds_1f_comparison.py \\
        validation/data/jw01118005001_04101_00001-seg001_nrs1_uncal.fits \\
        --config validation/configs/nrs1-hatp14b.toml --output validation/runs/nrs1-cds-1f
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

from group_difference_1f_check import white_light_rms

DOMAINS = ('groups', 'differences')
METHODS = ('jwst', 'sparta')


def undo(ramps, domain, levels):
    """Add back the levels removed by a 1/f correction in ``domain``."""
    start = 0
    for ramp in ramps:
        data, removed = ramp.data, levels[start:start + len(ramp.data)]
        if domain == 'groups':
            data += removed[:, :, None, :].astype(data.dtype)
        else:
            data[:, 1:] += np.cumsum(removed, axis=1)[:, :, None, :].astype(data.dtype)
        start += len(ramp.data)


def checksum(ramps):
    """Hash of the data and group DQ of the ramps (the ramp fits must not change them)."""
    import hashlib
    digest = hashlib.sha1()
    for ramp in ramps:
        digest.update(np.ascontiguousarray(ramp.data).tobytes())
        digest.update(np.ascontiguousarray(ramp.groupdq).tobytes())
    return digest.hexdigest()


def calibrate_and_extract(ts, uncal, config, output, window, keep_rates):
    from transitspectroscopy.jwst_models import mark_group_1f

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

    table, record = {}, dict(input=str(uncal), mode=dataset.mode, status=dataset.status)
    for domain in DOMAINS:
        diagnostics = ts.jwst.group_difference_1f_correction(
            dataset.ramps_per_segment, first_pass, dataset.mode_definition, domain=domain)
        np.savez_compressed(output/f'{name}_{domain}_1f.npz', **diagnostics)
        record[f'{domain}_shared'] = np.argwhere(diagnostics['shared']).tolist()
        lap(f'{domain}_correction')
        for method in METHODS:
            variant = f'{domain}_{method}'
            dataset.ramp_fit_method, dataset.ramp_fit_options = method, {}
            before = checksum(dataset.ramps_per_segment)
            dataset.run_ramp_fit(save=False, use_saved=False)
            if checksum(dataset.ramps_per_segment) != before:
                raise RuntimeError(f'The {method} ramp fit changed the ramps')
            path = output/variant/f'{name}_rateints.fits'
            path.parent.mkdir(parents=True, exist_ok=True)
            for rateints in dataset.rateints_per_segment:
                mark_group_1f(rateints)
                rateints.save(str(path))
                rateints.close()
            dataset.rateints_per_segment = dataset.rateints = None
            gc.collect()
            lap(f'{variant}_ramp_fit')
            table.update(white_light_rms(ts, {variant: path}, config, output, window,
                                         spectroscopic=True))
            lap(f'{variant}_stage2')
            if not keep_rates:
                path.unlink()
                shutil.rmtree(output/f'stage2_{variant}', ignore_errors=True)
        undo(dataset.ramps_per_segment, domain, diagnostics['levels'])

    record['timing_seconds'] = timing
    (output/f'{name}_calibration.json').write_text(json.dumps(record, indent=2, default=str)+'\n')
    del dataset, first_pass
    gc.collect()
    shutil.rmtree(work, ignore_errors=True)
    return table


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('uncal', type=Path, help='uncal segment')
    parser.add_argument('--config', type=Path, required=True,
                        help='validation configuration ([calibration] and [extraction])')
    parser.add_argument('--output', type=Path, required=True, help='output folder')
    parser.add_argument('--window', type=int, default=100, help='median-filter window (points)')
    parser.add_argument('--keep-rates', action='store_true',
                        help='keep the rates and Stage 2 products of every variant')
    args = parser.parse_args(argv)
    os.environ.setdefault('CRDS_PATH', str(Path.home()/'crds_cache'))
    os.environ.setdefault('CRDS_SERVER_URL', 'https://jwst-crds.stsci.edu')
    from validate import installed_package

    ts = installed_package()
    config = tomllib.loads(args.config.read_text())
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    table = calibrate_and_extract(ts, args.uncal.resolve(), config, output, args.window,
                                  args.keep_rates)
    (output/'white_light_rms.json').write_text(json.dumps(table, indent=2)+'\n')
    print(f'{args.window}-point median-filter RMS of the white light, ppm (robust RMS)')
    for variant, rms in table.items():
        print(f'  {variant:20s} ' + '  '.join(
            f"{extraction} {values['rms_ppm']:6.1f} ({values['robust_rms_ppm']:5.1f})"
            for extraction, values in rms.items() if extraction != 'spectroscopic'))
    for variant, rms in table.items():
        for extraction, products in rms.get('spectroscopic', {}).items():
            for name, powers in products.items():
                print(f'  {variant:20s} {extraction}/{name}: median channel RMS ppm (excess) '
                      + '  '.join(f"{power} {values['median_rms_ppm']:.0f} "
                                  f"({values['median_excess']:.3f})"
                                  for power, values in powers.items()))


if __name__ == '__main__':
    main()
