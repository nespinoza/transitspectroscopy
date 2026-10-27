"""Package stage 2 (box extraction, the G395H aperture radius of 2 pixels, DQ masked) of the
rates per integration of one variant of one segment, its chunks given as segments.

NRS2: the G395H trace of this target leaves the 32-row subarray near column 1968, so the
tracing stops at column 1900 (the package default, 2043, fails).
  python nl_groups_stage2.py <folder> <variant, e.g. crds_n3> <out.npz>
"""
import os, sys, glob, time, shutil, dataclasses
from pathlib import Path
import numpy as np
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO/'tests'))
import _support  # noqa: F401
os.environ.setdefault('CRDS_PATH', str(Path.home()/'crds_cache'))
os.environ.setdefault('CRDS_SERVER_URL', 'https://jwst-crds.stsci.edu')
import transitspectroscopy as ts
from transitspectroscopy import instruments
from stdatamodels.jwst import datamodels

folder, variant, out = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
start = time.monotonic()
files = sorted(glob.glob(str(folder/f'{variant}_[0-9]*.fits')))
rates = [datamodels.open(f) for f in files]
times = np.concatenate([np.asarray(r.int_times['int_mid_BJD_TDB']) + 2400000.5 for r in rates])
inputs = {'rampstep': rates, 'times': times, 'ints_per_segment': [len(r.data) for r in rates]}
mode = instruments.get_mode('nirspec/g395h')
if rates[0].meta.instrument.detector.lower() == 'nrs2':
    tracing = dict(mode.tracing)
    tracing['nrs2'] = dict(tracing['nrs2'], xend=1900)
    mode = dataclasses.replace(mode, tracing=tracing)
work = folder/f'stage2_{variant}'
work.mkdir(exist_ok=True)
outputs = ts.jwst.stage2(inputs, aperture_radius=2, mask_dq=True, mode=mode,
                         outputfolder=str(work) + '/', suffix=variant)
spectra = outputs['spectra']
np.savez(out, times=times, wavelengths=spectra['wavelengths'], spectra=spectra['corrected'],
         spectra_err=spectra['corrected_err'], original=spectra['original'],
         whitelight=outputs['whitelight'], files=np.array([Path(f).name for f in files]))
for r in rates:
    r.close()
shutil.rmtree(work, ignore_errors=True)
print(f'done: {time.monotonic() - start:.0f} s')
