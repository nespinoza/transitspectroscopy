"""Group-level 1/f correction of one SOSS segment with median group-difference templates.

Stage 1 without any group-level 1/f correction (no clean_flicker_noise, no group 1/f; TSO
jump step) gives ramps and first-pass rates. A box-only Stage 2 of those rates gives the
traces (orders 1, 2 and 3 masked), background, median rate and white-light factor of every
integration, with which ``soss.correct_ramps_1f`` corrects every group of the ramps; the
ramps are then fitted again. Both rateints are saved, for full Stage 2 runs with
``validate.py --input-glob``. Run from the repository root:

    python validation/soss_group_1f_experiment.py \\
        validation/data/soss_uncal/jw01541001001_04101_00001-seg001_nis_uncal.fits \\
        --config validation/configs/soss-hatp14b-uncal.toml \\
        --output validation/runs/soss-seg1-group-diff-1f
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

ROOT = Path(__file__).resolve().parents[1]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('uncal', type=Path, help='uncal segment')
    parser.add_argument('--config', type=Path, required=True,
                        help='SOSS validation configuration (for the Stage 2 references)')
    parser.add_argument('--output', type=Path, required=True, help='output folder')
    parser.add_argument('--drop-first-pass', action='store_true',
                        help='delete the first-pass rateints at the end (disk)')
    args = parser.parse_args(argv)
    os.environ.setdefault('CRDS_PATH', str(Path.home()/'crds_cache'))
    os.environ.setdefault('CRDS_SERVER_URL', 'https://jwst-crds.stsci.edu')
    from validate import installed_package

    ts = installed_package()
    from transitspectroscopy import soss
    config = tomllib.loads(args.config.read_text())
    extraction = config['extraction']['soss']
    options = dict(extraction['soss'])
    for key in ('pastasoss_reference', 'background_reference'):
        options[key] = str(ROOT/options[key])
    stage2 = dict(pastasoss_reference=options['pastasoss_reference'],
                  background_reference=options['background_reference'],
                  background_options=options['background_options'],
                  trace_max_shift=options['trace_max_shift'],
                  sample_integrations=options['sample_integrations'],
                  aperture_radius=extraction['aperture_radius'], mask_dq=extraction['mask_dq'])

    name = args.uncal.name.split('_uncal')[0]
    work = args.output/'work'
    first = args.output/'no_group_1f'/'stage1'/f'{name}_rateints.fits'
    corrected = args.output/'group_diff_1f'/'stage1'/f'{name}_rateints.fits'
    for path in (work, first.parent, corrected.parent):
        path.mkdir(parents=True, exist_ok=True)
    timing = {}
    start = time.monotonic()
    dataset = ts.jwst.load([str(args.uncal)], outputfolder=str(work))
    dataset.detector_calibration(parameters={'skip': ['clean_flicker_noise']}, use_tso_jump=True,
                                 group_1f=False, save=False)
    timing['stage1'] = time.monotonic() - start
    dataset.fit_ramps(save=False)
    dataset.rateints_per_segment[0].save(str(first))
    dataset.rateints_per_segment = []
    gc.collect()
    timing['first_ramp_fit'] = time.monotonic() - start - timing['stage1']

    print('Box-only Stage 2 of the first-pass rates...', flush=True)
    result = soss.extract_soss([str(first)], **stage2)
    diagnostics = soss.correct_ramps_1f(dataset.ramps_per_segment, [str(first)], result)
    del result
    gc.collect()
    timing['group_correction'] = time.monotonic() - start - sum(timing.values())
    print('Group template factors:', np.round(diagnostics['group_scale'], 4), flush=True)

    dataset.fit_ramps(save=False)
    dataset.rateints_per_segment[0].save(str(corrected))
    timing['second_ramp_fit'] = time.monotonic() - start - sum(timing.values())
    np.savez_compressed(args.output/'group_diff_1f'/f'{name}_diagnostics.npz', **diagnostics)
    record = dict(input=str(args.uncal), config=str(args.config), stage2_first_pass=stage2,
                  stage1_status=dataset.status, timing_seconds=timing,
                  group_scale=diagnostics['group_scale'].tolist(),
                  white_light_scale_range=[float(diagnostics['white_light_scale'].min()),
                                           float(diagnostics['white_light_scale'].max())])
    (args.output/'group_diff_1f'/f'{name}_run.json').write_text(
        json.dumps(record, indent=2, default=str)+'\n')
    shutil.rmtree(work, ignore_errors=True)
    if args.drop_first_pass:
        first.unlink()
    print(json.dumps(timing, indent=2), flush=True)


if __name__ == '__main__':
    main()
