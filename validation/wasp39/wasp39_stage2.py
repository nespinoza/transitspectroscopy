"""Package stage 2 (box extraction, the mode's default aperture, DQ-masked but for SATURATED)
of one segment's chunks.

All of an exposure's rates do not fit in memory in stage 2, so it runs per segment; with
<reference work folder>, the traces of that earlier run are reused (one trace for all
segments, so that the aperture does not move between them).
  python wasp39_stage2.py <folder> <variant> <first,first,...> <out.npz> [<reference work>]
"""
import os, sys, time, shutil, glob
from pathlib import Path
import numpy as np
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO/'tests'))
import _support  # noqa: F401
os.environ.setdefault('CRDS_PATH', str(Path.home()/'crds_cache'))
os.environ.setdefault('CRDS_SERVER_URL', 'https://jwst-crds.stsci.edu')
import transitspectroscopy as ts
from stdatamodels.jwst import datamodels
from stdatamodels.jwst.transforms import models as transform_models

# stdatamodels 6.0 with astropy 7.2: Rotation3DToGWA.evaluate takes angles[0], expecting a 2-D
# parameter array, but the single-angle rotation of the NIRSpec PRISM model arrives 1-D
# (angles[0] is a float and AssignWcsStep fails). Same rotation, any of those shapes:
_rotation_evaluate = transform_models.Rotation3DToGWA.evaluate


def rotation_evaluate(self, x, y, z, angles):
    angles = np.asarray(angles, dtype=float)
    angles = np.atleast_1d(angles[0] if angles.ndim == 2 else angles)
    return _rotation_evaluate(self, x, y, z, angles[None, :])


transform_models.Rotation3DToGWA.evaluate = rotation_evaluate

folder, variant, out = Path(sys.argv[1]), sys.argv[2], sys.argv[4]
firsts = [int(f) for f in sys.argv[3].split(',')]
reference = Path(sys.argv[5]) if len(sys.argv) > 5 else None
start = time.monotonic()
rates = [datamodels.open(str(folder/f'{variant}_{f:05d}.fits')) for f in firsts]
# A rate fitted from the unsaturated groups is valid, but keeps the SATURATED flag, which
# mask_dq would mask (in the integrations where the pixel saturates, not in the others):
SATURATED = datamodels.dqflags.pixel['SATURATED']
for r in rates:
    r.dq = np.asarray(r.dq) & ~np.uint32(SATURATED)
times = np.concatenate([np.asarray(r.int_times['int_mid_BJD_TDB']) + 2400000.5 for r in rates])
inputs = {'rampstep': rates, 'times': times, 'ints_per_segment': [len(r.data) for r in rates]}
work = Path(out).with_suffix('')
(work/'pipeline_outputs').mkdir(parents=True, exist_ok=True)
if reference is not None:
    traces = glob.glob(str(reference/'pipeline_outputs'/'traces*.pkl'))
    assert len(traces) == 1, traces
    shutil.copy(traces[0], work/'pipeline_outputs'/f'traces_{variant}.pkl')
outputs = ts.jwst.stage2(inputs, mask_dq=True, outputfolder=str(work) + '/', suffix=variant)
spectra = outputs['spectra']
np.savez(out, times=times, wavelengths=spectra['wavelengths'], spectra=spectra['corrected'],
         spectra_err=spectra['corrected_err'], original=spectra['original'],
         x=np.asarray(outputs['traces']['x']),
         trace=np.nanmedian(outputs['traces']['ysmoothed'], axis=0))
for r in rates:
    r.close()
for name in glob.glob(str(work/'pipeline_outputs'/'spectra*.pkl')):
    os.remove(name)  # keep only the traces (reused by the other segments)
print(f'{variant} {firsts}: {len(times)} integrations, {time.monotonic() - start:.0f} s',
      flush=True)
