"""Summary of onef_sweep.py outputs (all segments of one exposure).

Per 1/f pixel band (inner, width): pixels per column; white-light rms (ppm, about a
51-integration running median, 5-sigma clipped; per aperture); median spectroscopic-bin rms;
the injected chromatic signal kept, sum(dL dL_true)/sum(dL_true^2) (1 = none removed); and
the first-difference variance of the white light (and median over bins) over the sum of
those of its pixels (below 1: pixel noise anti-correlated, i.e. removed with the levels).
  python onef_sweep_summary.py <seg1.npz> ... [--csv out.csv]
"""
import sys
import numpy as np
from scipy.ndimage import median_filter

args = [a for a in sys.argv[1:] if not a.startswith('--')]
csv = sys.argv[sys.argv.index('--csv') + 1] if '--csv' in sys.argv else None
args = [a for a in args if a != csv]
files = [np.load(f) for f in args]
configs = files[0]['configs']
apertures = files[0]['apertures']
ncfg = len(configs)
W = np.concatenate([f['W'] for f in files], axis=1).astype(float)
L = np.concatenate([f['L'] for f in files], axis=1).astype(float)
dL = np.concatenate([f['dL'] for f in files], axis=1).astype(float)
dL_true = np.concatenate([f['dL_true'] for f in files]).astype(float)
ncols = len(files[0]['factors']) and None
fd_w = sum(np.sum(np.diff(f['W'].astype(float), axis=1)**2, axis=1) for f in files)
fd_l = sum(np.sum(np.diff(f['L'].astype(float), axis=1)**2, axis=1) for f in files)
fd_p = sum(f['fd_pixels'] for f in files)
fd_pb = sum(f['fd_pixels_bins'] for f in files)


def detrended(x):
    n = x/np.nanmedian(x, axis=-1, keepdims=True)
    return n - median_filter(n, size=[1]*(n.ndim - 1) + [51], mode='nearest')


raw = W[-1]
keep = np.ones(W.shape[1], bool)
for a in range(len(apertures)):
    d = detrended(raw[:, a])
    keep &= np.abs(d - np.median(d)) < 5*1.4826*np.median(np.abs(d - np.median(d)))
rows = []
for c in range(ncfg + 1):
    rms = [1e6*np.std(detrended(W[c, :, a])[keep]) for a in range(len(apertures))]
    spec = 1e6*np.nanmedian(np.std(detrended(L[c].T)[:, keep], axis=1))
    finite = np.isfinite(dL[c]) & np.isfinite(dL_true)
    kept = np.sum((dL[c]*dL_true)[finite])/np.sum(dL_true[finite]**2)
    ratio_w = (fd_w[c]/fd_p[c])[0]  # first aperture
    ratio_b = np.nanmedian(fd_l[c]/fd_pb[c])
    label = ('raw', '') if c == ncfg else (f'{configs[c][0]:g}', f'{configs[c][1]:g}')
    npix = files[0]['npix'][c] if c < ncfg else 0
    rows.append((label, npix, rms, spec, kept, ratio_w, ratio_b))

ncolumns = len(files[0]['factors'])
print(f"{'inner':>5} {'width':>5} {'pix':>7} " + ' '.join(f'rms(ap {a:g})' for a in apertures)
      + '  spec rms   kept   floor(white) floor(bins)')
for label, npix, rms, spec, kept, rw, rb in rows:
    print(f'{label[0]:>5} {label[1]:>5} {npix:7d} ' + ' '.join(f'{r:11.1f}' for r in rms)
          + f'  {spec:8.1f}  {kept:6.4f}  {rw:8.3f}  {rb:8.3f}')
if csv:
    with open(csv, 'w') as f:
        f.write('inner,width,pixels,' + ','.join(f'rms_ap{a:g}' for a in apertures)
                + ',spec_rms,kept,floor_white,floor_bins\n')
        for label, npix, rms, spec, kept, rw, rb in rows:
            f.write(f'{label[0]},{label[1]},{npix},' + ','.join(f'{r:.2f}' for r in rms)
                    + f',{spec:.2f},{kept:.5f},{rw:.4f},{rb:.4f}\n')
