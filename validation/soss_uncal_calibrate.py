"""Stage 1 of a NIRISS/SOSS TSO from uncal files, one segment at a time.

``ts.run`` calibrates all segments of an exposure together, which for a full SOSS TSO
(e.g. PID 1541, 572 integrations of 6 groups) needs more memory than a laptop has. This
script runs the same steps (``load`` -> ``detector_calibration`` -> ``fit_ramps``, with the
mode defaults) on each segment separately, keeps the rates per integration and deletes the
large intermediate ramps. The TSO jump step then sees one segment at a time; with its
10-integration median filter this only changes the first and last few integrations of
each segment. Run from the repository root:

    python validation/soss_uncal_calibrate.py validation/data/soss_uncal/*_uncal.fits \
        --output validation/runs/soss-hatp14b-uncal/stage1
"""

import argparse
import gc
import json
import os
from pathlib import Path
import shutil
import time


def calibrate_segment(ts, filename, output):
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
    dataset.detector_calibration()
    dataset.fit_ramps()
    record = dict(
        input=str(filename),
        calibration_parameters=dataset.calibration_parameters,
        status=dataset.status,
        elapsed_seconds=time.monotonic() - start,
    )
    (folder/'calibration.json').write_text(json.dumps(record, indent=2, default=str)+'\n')
    rateints = sorted((folder/'ts_outputs').glob('*ramp_fitstep.fits'))
    rateints = [path for path in rateints if '_1_ramp_fitstep' in path.name]
    if len(rateints) != 1:
        raise RuntimeError(f'Expected one rateints product in {folder}, found {rateints}')
    os.replace(rateints[0], final)
    del dataset
    gc.collect()
    # The saved linearity and jump ramps are several GB per segment:
    shutil.rmtree(folder/'ts_outputs')
    print(f'{filename}: {record["elapsed_seconds"]:.0f} s', flush=True)
    return final


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('uncal', nargs='+', type=Path, help='uncal segment files')
    parser.add_argument('--output', type=Path, required=True, help='output folder')
    args = parser.parse_args(argv)
    os.environ.setdefault('CRDS_PATH', str(Path.home()/'crds_cache'))
    os.environ.setdefault('CRDS_SERVER_URL', 'https://jwst-crds.stsci.edu')
    import transitspectroscopy as ts

    for filename in sorted(args.uncal):
        calibrate_segment(ts, filename, args.output)


if __name__ == '__main__':
    main()
