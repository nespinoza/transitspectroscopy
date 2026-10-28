"""One self non-linearity correction from the per-chunk fits of wasp39_stage1.py ('fit'):
the mean ramps of all chunks are combined (count-weighted) into the exposure's mean ramp and
fitted once with transitspectroscopy.core.linearity.fit_self_linearity.
  python wasp39_combine_fit.py <folder> <out.npz> [read noise DN] [gain e/DN]
"""
import sys, glob
from pathlib import Path
import numpy as np
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO/'tests'))
import _support  # noqa: F401
from transitspectroscopy.core.linearity import SelfLinearity, fit_self_linearity

folder, out = Path(sys.argv[1]), sys.argv[2]
read_noise = float(sys.argv[3]) if len(sys.argv) > 3 else 7.0
gain = float(sys.argv[4]) if len(sys.argv) > 4 else 1.4
fits = [SelfLinearity.load(f) for f in sorted(glob.glob(str(folder/'seg*'/'selffit_*.npz')))]
sums = sum(f.mean_ramps[0]*f.counts[0] for f in fits)
counts = sum(f.counts[0] for f in fits)
metadata = {k: v for k, v in fits[0].metadata.items()
            if k in ('detector', 'subarray', 'xstart', 'ystart', 'xsize', 'ysize')}
metadata['chunks'] = len(fits)
correction = fit_self_linearity(sums/np.maximum(counts, 1), counts, read_noise=read_noise,
                                gain=gain, metadata=metadata)
correction.save(out)
used = (counts >= correction.metadata['min_ramps']).sum(0)
bright = np.max(np.where(counts > 0, sums/np.maximum(counts, 1), 0), axis=0) > 500
print(f'{len(fits)} chunks, {int(counts[0].max())} integrations; fitted '
      f'{correction.metadata["reliable"]}/{correction.metadata["fitted"]} pixels; bright pixels '
      f'by groups used: ' + ', '.join(f'{n}: {int((bright & (used == n)).sum())}'
                                       for n in range(counts.shape[0], -1, -1)))
