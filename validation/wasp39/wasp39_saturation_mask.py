"""Static mask of the PRISM pixels that saturate (wasp39_stage2.py, mask argument).

The SATURATED flags of the rates of all chunks of all segments of one variant are counted; a
pixel flagged in more than ``min_fraction`` of the integrations (the trace core; the others
are sporadic, cosmic-ray saturations) is masked in all of them, so that every column keeps the
same pixels throughout and loses only its saturated core. Optionally the mask grows by
``grow`` pixels along the rows (cross-dispersion).
  python wasp39_saturation_mask.py <prism folder> <variant> <out.npy> [grow] [min_fraction]
"""
import sys, glob
from pathlib import Path
import numpy as np
from astropy.io import fits

folder, variant, out = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
grow = int(sys.argv[4]) if len(sys.argv) > 4 else 0
min_fraction = float(sys.argv[5]) if len(sys.argv) > 5 else 1e-3
SATURATED = 2
files = sorted(glob.glob(str(folder/'seg*'/f'{variant}_?????.fits')))
flagged, n = 0, 0
for f in files:
    dq = fits.getdata(f, 'DQ')
    flagged = flagged + ((dq & SATURATED) != 0).sum(0)
    n += len(dq)
core = flagged > min_fraction*n
mask = core.copy()
for shift in range(1, grow + 1):
    mask[shift:] |= core[:-shift]
    mask[:-shift] |= core[shift:]
np.save(out, mask)
columns = np.nonzero(core.any(0))[0]
print(f'{len(files)} files, {n} integrations: {(flagged > 0).sum()} pixels saturate in some '
      f'integration, {core.sum()} in more than {min_fraction:g} of them (columns '
      f'{columns.min()}-{columns.max()}, at most {core.sum(0).max()} per column); mask (grow '
      f'{grow}) {mask.sum()} pixels')
