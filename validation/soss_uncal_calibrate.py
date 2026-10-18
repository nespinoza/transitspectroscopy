"""Stage 1 of a NIRISS/SOSS TSO from uncal files, one segment at a time.

``ts.run`` calibrates all segments of an exposure together, which for a full SOSS TSO
(e.g. PID 1541, 572 integrations of 6 groups) needs more memory than a laptop has. This
script runs the same steps (``load`` -> ``detector_calibration`` -> ``fit_ramps``, with the
mode defaults, including ``clean_flicker_noise`` after the jump step) on each segment
separately and keeps only the rates per integration. The TSO jump step then sees one
segment at a time; with its 10-integration median filter this only changes the first and
last few integrations of each segment. ``--jump jwst`` uses the STScI jump step instead,
with the rejection threshold of the CRDS calwebb_detector1 parameters for SOSS (6).
Run from the repository root:

    python validation/soss_uncal_calibrate.py validation/data/soss_uncal/*_uncal.fits \
        --output validation/runs/soss-hatp14b-uncal/stage1-tsojump
"""

import argparse
import gc
import json
import os
from pathlib import Path
import shutil
import time


def calibrate_segment(ts, filename, output, jump='tso', rejection_threshold=6.0, group_1f=False,
                      clean_flicker_noise=True):
    """Calibrate one uncal segment; return the path of its rateints."""
    name = Path(filename).name.split('_uncal')[0]
    folder = output/name
    final = output/f'{name}_rateints.fits'
    if final.exists():
        print('Found', final, flush=True)
        return final
    folder.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    dataset = ts.jwst.load([str(filename)], outputfolder=str(folder))
    parameters = {} if clean_flicker_noise else {'skip': ['clean_flicker_noise']}
    if jump == 'jwst':
        parameters['jump'] = {'rejection_threshold': rejection_threshold}
    dataset.detector_calibration(
        parameters=parameters, use_tso_jump=jump == 'tso', group_1f=group_1f, save=False
    )
    dataset.fit_ramps(save=False)
    rateints = dataset.rateints_per_segment[0]
    rateints.save(str(final))
    record = dict(
        input=str(filename),
        jump=jump,
        group_1f=group_1f,
        clean_flicker_noise=clean_flicker_noise,
        calibration_parameters=dataset.calibration_parameters,
        status=dataset.status,
        elapsed_seconds=time.monotonic() - start,
    )
    (folder/'calibration.json').write_text(json.dumps(record, indent=2, default=str)+'\n')
    del dataset, rateints
    gc.collect()
    # Nothing is saved there with save=False, but remove the folder load creates:
    shutil.rmtree(folder/'ts_outputs', ignore_errors=True)
    print(f'{filename}: {record["elapsed_seconds"]:.0f} s', flush=True)
    return final


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('uncal', nargs='+', type=Path, help='uncal segment files')
    parser.add_argument('--output', type=Path, required=True, help='output folder')
    parser.add_argument('--jump', choices=['tso', 'jwst'], default='tso',
                        help='TSO jump step (default) or the STScI jump step')
    parser.add_argument('--rejection-threshold', type=float, default=6.0,
                        help='STScI jump step rejection threshold (with --jump jwst)')
    parser.add_argument('--group-1f', action='store_true',
                        help='group-level 1/f correction (as for NIRSpec), before linearity')
    parser.add_argument('--skip-clean-flicker-noise', action='store_true',
                        help='do not run clean_flicker_noise')
    args = parser.parse_args(argv)
    os.environ.setdefault('CRDS_PATH', str(Path.home()/'crds_cache'))
    os.environ.setdefault('CRDS_SERVER_URL', 'https://jwst-crds.stsci.edu')
    from validate import installed_package

    ts = installed_package()
    for filename in sorted(args.uncal):
        calibrate_segment(ts, filename, args.output, args.jump, args.rejection_threshold,
                          args.group_1f, not args.skip_clean_flicker_noise)


if __name__ == '__main__':
    main()
