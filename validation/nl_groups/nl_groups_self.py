"""Self non-linearity correction of one segment and its rates fitted with the first N groups.

From the chunks of nl_groups_stage1.py (rawdark, groupdq, template): the mean ramp of every
pixel over the integrations where none of its groups is flagged, fitted (GLS, read noise
READ_NOISE DN plus Poisson noise at GAIN e/DN) with a cubic Chebyshev correction,
  g(x) = x + S (a2 T2(u) + a3 T3(u)) = z + b t,   u = 2x/top - 1, S = top/2, top = 1.02 x_last,
for pixels whose mean ramp ends above MIN_COUNTS DN. The correction is
  h(x) = c1 (g(x) - g(0))/g'(0)   (h(0) = 0, h'(0) = c1, the CRDS linear coefficient),
continued outside [0, top] by straight lines with h's value and slope at the ends; other
pixels (faint, flagged, or g' <= 0 on [0, top]) get the CRDS polynomial. The corrected
ramps are fitted (JWST ramp fit, the package's parameters) with the first N groups:
  self_n<N>_<first>.fits, and self_coefficients.npz.
  python nl_groups_self.py <folder> <N1,N2,...> [minimum unflagged ramps, 50]
"""
import os, sys, json, glob, time, resource
from pathlib import Path
import numpy as np
os.environ.setdefault('CRDS_PATH', str(Path.home()/'crds_cache'))
os.environ.setdefault('CRDS_SERVER_URL', 'https://jwst-crds.stsci.edu')
from stdatamodels.jwst import datamodels
from jwst.pipeline import calwebb_detector1
from jwst.stpipe.core import JwstStep
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO/'tests'))
import _support  # noqa: F401
from transitspectroscopy.jwst_models import mark_group_1f

_prepare_output = JwstStep.prepare_output


def prepare_output(self, init, make_copy=None, **kwargs):
    return _prepare_output(self, init, make_copy=False, **kwargs)


JwstStep.prepare_output = prepare_output

folder = Path(sys.argv[1])
groups = [int(n) for n in sys.argv[2].split(',')]
READ_NOISE, GAIN, MIN_COUNTS, DO_NOT_USE = 7.08, 1.44, 500., 1
MIN_RAMPS = int(sys.argv[3]) if len(sys.argv) > 3 else 50
start = time.monotonic()
tags = sorted(Path(p).stem.split('_')[1] for p in glob.glob(str(folder/'rawdark_*.npy')))
template = datamodels.RampModel(folder/f'template_{tags[0]}.fits')
ngroups, nrows, ncols = template.data.shape[1:]

# CRDS linearity coefficients of the subarray:
name = open(folder/f'linearity_{tags[0]}.txt').read().strip().replace('crds://', '')
path = Path(os.environ['CRDS_PATH'])/'references'/'jwst'/'nirspec'/name
with datamodels.LinearityModel(str(path)) as model:
    x0 = template.meta.subarray.xstart - (model.meta.subarray.xstart or 1)
    y0 = template.meta.subarray.ystart - (model.meta.subarray.ystart or 1)
    crds_coeffs = np.array(model.coeffs[:, y0:y0 + nrows, x0:x0 + ncols], float)
assert crds_coeffs.shape[1:] == (nrows, ncols)


def crds_polynomial(x):
    out = np.zeros_like(x)
    for c in crds_coeffs[::-1]:
        out = out*x + c
    return out


# Mean ramps over the unflagged integrations:
total = np.zeros((ngroups, nrows, ncols))
count = np.zeros((nrows, ncols))
for t in tags:
    x = np.load(folder/f'rawdark_{t}.npy', mmap_mode='r')
    dq = np.load(folder/f'groupdq_{t}.npy', mmap_mode='r')
    for i in range(len(x)):
        good = (np.asarray(dq[i]) == 0).all(0) & np.isfinite(x[i]).all(0)
        total += np.where(good, x[i], 0.)
        count += good
mean = total/np.maximum(count, 1)
print(f'mean ramps: {time.monotonic() - start:.0f} s', flush=True)

# GLS fits of the pixels bright enough:
fit = (count >= MIN_RAMPS) & (mean[-1] > MIN_COUNTS)
rows, cols = np.nonzero(fit)
xm = mean[:, rows, cols].T                                  # (pixel, group)
top = 1.02*xm[:, -1]
u = 2*xm/top[:, None] - 1
S = top/2
t = np.arange(ngroups, dtype=float)
design = np.stack([np.ones_like(xm), np.broadcast_to(t, xm.shape),
                   -S[:, None]*(2*u**2 - 1), -S[:, None]*(4*u**3 - 3*u)], axis=-1)
charge = np.maximum(xm, 0.)
covariance = (READ_NOISE**2*np.eye(ngroups)[None]
              + np.minimum(charge[:, :, None], charge[:, None, :])/GAIN)
weighted = np.linalg.solve(covariance, np.concatenate([design, xm[..., None]], -1))
normal = np.einsum('pgi,pgj->pij', design, weighted[..., :4])
rhs = np.einsum('pgi,pg->pi', design, weighted[..., 4])[..., None]
z, b, a2, a3 = np.linalg.solve(normal, rhs)[..., 0].T
# g'(u)/1 = 1 + 4 a2 u + a3 (12 u^2 - 3) > 0 on [-1, 1] (ends and the vertex):
uv = np.clip(np.where(a3 != 0, -a2/(6*np.where(a3 != 0, a3, 1)), -1), -1, 1)
slope_min = np.minimum.reduce([1 - 4*a2 + 9*a3, 1 + 4*a2 + 9*a3,
                               1 + 4*a2*uv + a3*(12*uv**2 - 3)])
reliable = np.isfinite(slope_min) & (slope_min > 0)
coefficients = {}
for key, value in dict(a2=a2, a3=a3, top=top, z=z, b=b).items():
    coefficients[key] = np.full((nrows, ncols), np.nan)
    coefficients[key][rows, cols] = np.where(reliable, value, np.nan)
np.savez(folder/'self_coefficients.npz', mean=mean, count=count, crds_coeffs=crds_coeffs,
         **coefficients)
self_pixels = np.isfinite(coefficients['a2'])
print(f'fits: {fit.sum()} pixels above {MIN_COUNTS:g} DN, {self_pixels.sum()} reliable; '
      f'{time.monotonic() - start:.0f} s', flush=True)
A2, A3, TOP = (np.where(self_pixels, coefficients[k], 0.) for k in ('a2', 'a3', 'top'))
TOP = np.where(self_pixels, TOP, 1.)
C1 = crds_coeffs[1]
G0 = (TOP/2)*(A2 - A3)
D0 = 1 - 4*A2 + 9*A3
D1 = 1 + 4*A2 + 9*A3


def g(x):
    u = 2*x/TOP - 1
    return x + (TOP/2)*(A2*(2*u**2 - 1) + A3*(4*u**3 - 3*u))


H_TOP = C1*(g(TOP) - G0)/D0


def correct(x):
    """The self correction of ramps x (..., row, column) in DN."""
    inside = C1*(g(np.clip(x, 0, TOP)) - G0)/D0
    out = np.where(x > TOP, H_TOP + C1*D1/D0*(x - TOP), np.where(x < 0, C1*x, inside))
    return np.where(self_pixels, out, crds_polynomial(x)).astype(np.float32)


parameters = json.load(open(folder/f'ramp_fit_{tags[0]}.json'))
for t in tags:
    chunk_template = datamodels.RampModel(folder/f'template_{t}.fits')
    raw = np.load(folder/f'rawdark_{t}.npy', mmap_mode='r')
    data = np.empty(raw.shape, np.float32)
    for i in range(len(raw)):
        data[i] = correct(np.asarray(raw[i], float))
    groupdq = np.load(folder/f'groupdq_{t}.npy')
    ramp = datamodels.RampModel(data=data, groupdq=groupdq.copy(),
                                pixeldq=np.asarray(chunk_template.pixeldq))
    ramp.update(chunk_template, only=None)
    ramp.int_times = chunk_template.int_times
    ramp.meta.cal_step.linearity = 'COMPLETE'
    for n in groups:
        ramp.groupdq = groupdq.copy()
        ramp.groupdq[:, n:] |= DO_NOT_USE
        rateints = calwebb_detector1.ramp_fit_step.RampFitStep.call(ramp, **parameters)[1]
        mark_group_1f(rateints)
        for array in ('var_poisson', 'var_rnoise', 'var_flat'):
            if rateints.hasattr(array):
                delattr(rateints, array)
        rateints.save(folder/f'self_n{n}_{t}.fits')
        rateints.close()
    ramp.close()
    del data
    print(f'chunk {t}: {time.monotonic() - start:.0f} s', flush=True)
print(f'done: {time.monotonic() - start:.0f} s, peak '
      f'{resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**30:.1f} GB')
