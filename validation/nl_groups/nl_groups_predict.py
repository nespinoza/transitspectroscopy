"""Noise-free transit-depth factors of the CRDS and self corrections for N-group ramp fits.

The self correction of each segment (self_coefficients.npz of nl_groups_self.py) is taken as
the truth: every reliable pixel's mean ramp x(t) holds the charge h(x(t)); a transit of depth
DEPTH scales it, x'(t) = h^-1((1 - DEPTH) h(x(t))). Each correction C (CRDS polynomial,
self h) is applied to x and x', and the slopes over the first N groups are fitted with the
JWST ramp-fit weights (optimal weighting: |t - t_mid|^P, P from the signal-to-noise ratio
of the ramp); the apparent depth of a pixel is 1 - slope(C(x'))/slope(C(x)). Spectroscopic
bins (the columns of nl_groups_spectrum.py, from the first traced column) sum the box aperture (rows within RADIUS of the
flux-weighted centre of the column); the depth factor is the apparent over the true depth.
  python nl_groups_predict.py <run folder> <spectrum.npz> <detector> [...] --groups 3,10,20
"""
import sys, glob
from pathlib import Path
import numpy as np

DEPTH, RADIUS, READ_NOISE, GAIN = 0.0069, 2, 7.08, 1.44
FIRST_COLUMN = {'nrs1': 500, 'nrs2': 5}
SNR_LIMITS, POWERS = [5, 10, 20, 50, 100], [0., 0.4, 1., 3., 6., 10.]
args = [a for a in sys.argv[1:] if not a.startswith('--')]
groups = [int(n) for n in sys.argv[sys.argv.index('--groups') + 1].split(',')]
args.remove(sys.argv[sys.argv.index('--groups') + 1])
if '--out' in sys.argv:
    args.remove(sys.argv[sys.argv.index('--out') + 1])
run, spectrum, detectors = Path(args[0]), np.load(args[1]), args[2:]


def jwst_slope(y, n):
    """Slopes (per group) of ramps y (pixel, group) over the first n groups, JWST weights."""
    y = y[:, :n]
    t = np.arange(n, dtype=float)
    signal = np.maximum(y[:, -1] - y[:, 0], 0)*GAIN
    snr = signal/np.sqrt((READ_NOISE*GAIN)**2 + signal)
    power = np.array(POWERS)[np.searchsorted(SNR_LIMITS, snr)]
    mid = (n - 1)/2
    w = np.abs((t[None] - mid)/max(mid, 1))**power[:, None]
    tw = np.sum(w*t, 1)/np.sum(w, 1)
    yw = np.sum(w*y, 1)/np.sum(w, 1)
    return np.sum(w*(t - tw[:, None])*(y - yw[:, None]), 1)/np.sum(w*(t - tw[:, None])**2, 1)


results = {}
for detector in detectors:
    # Spectra are indexed along the traced columns (G395H: NRS1 500-2042, NRS2 5-1900 here):
    columns_of_bins = spectrum[f'{detector}__bin_columns'] + FIRST_COLUMN[detector]
    factors = {f'{c}_n{n}': [] for c in ('crds', 'self') for n in groups}
    for segment in sorted(glob.glob(str(run/detector/'seg*'))):
        c = np.load(Path(segment)/'self_coefficients.npz')
        mean, crds_coeffs = c['mean'], c['crds_coeffs']
        ngroups, nrows, ncols = mean.shape

        reliable = np.isfinite(c['a2'])
        rows, cols = np.nonzero(reliable)
        x = mean[:, rows, cols].T
        a2, a3, top = c['a2'][rows, cols], c['a3'][rows, cols], c['top'][rows, cols]
        c1 = crds_coeffs[1][rows, cols]
        d0 = 1 - 4*a2 + 9*a3
        d1 = 1 + 4*a2 + 9*a3
        g0 = top/2*(a2 - a3)

        def h(x):
            xc = np.clip(x, 0, top[:, None])
            u = 2*xc/top[:, None] - 1
            inside = c1[:, None]*(xc + top[:, None]/2*(a2[:, None]*(2*u**2 - 1)
                                                     + a3[:, None]*(4*u**3 - 3*u))
                                  - g0[:, None])/d0[:, None]
            above = inside + (c1*d1/d0)[:, None]*(x - top[:, None])
            return np.where(x > top[:, None], above, np.where(x < 0, c1[:, None]*x, inside))

        def h_prime(x):
            u = 2*np.clip(x, 0, top[:, None])/top[:, None] - 1
            inside = c1[:, None]*(1 + 4*a2[:, None]*u + a3[:, None]*(12*u**2 - 3))/d0[:, None]
            return np.where(x > top[:, None], (c1*d1/d0)[:, None],
                            np.where(x < 0, c1[:, None], inside))

        def crds(x):
            coefficients = crds_coeffs[:, rows, cols]
            out = np.zeros_like(x)
            for k in coefficients[::-1]:
                out = out*x + k[:, None]
            return out

        target = (1 - DEPTH)*h(x)
        xt = x*(1 - DEPTH)
        for _ in range(20):
            xt = xt - (h(xt) - target)/h_prime(xt)
        assert np.max(np.abs(h(xt) - target)) < 1e-6*np.max(np.abs(target)) + 1e-6

        # Box aperture around the flux-weighted centre of every column:
        last = np.where(np.isfinite(mean[-1]), mean[-1], 0)
        peak = np.argmax(last, 0)
        r = np.arange(nrows)[:, None]
        window = np.abs(r - peak[None]) <= 4
        centre = np.sum(np.where(window, last*r, 0), 0)/np.sum(np.where(window, last, 0), 0)
        aperture = (np.abs(r - centre[None]) <= RADIUS)[rows, cols]
        for corr, function in (('crds', crds), ('self', h)):
            for n in groups:
                slope0 = jwst_slope(function(x), n)
                slope1 = jwst_slope(function(xt), n)
                per_bin = []
                for c0, c1_ in columns_of_bins:
                    use = aperture & (cols >= c0) & (cols <= c1_)
                    apparent = 1 - np.sum(slope1[use])/np.sum(slope0[use])
                    per_bin.append(apparent/DEPTH)
                use = aperture & (cols >= columns_of_bins.min()) & (cols <= columns_of_bins.max())
                white = (1 - np.sum(slope1[use])/np.sum(slope0[use]))/DEPTH
                factors[f'{corr}_n{n}'].append(per_bin + [white])
    for k, v in factors.items():
        v = np.array(v)
        results[f'{detector}__{k}'] = v
        print(f'{detector} {k}: white depth factor per segment '
              + ' '.join(f'{f:.5f}' for f in v[:, -1])
              + f'; bins (mean of segments) {np.median(v[:, :-1].mean(0)):.5f} median, '
              f'{v[:, :-1].mean(0).min():.5f}-{v[:, :-1].mean(0).max():.5f}', flush=True)
np.savez(sys.argv[sys.argv.index('--out') + 1] if '--out' in sys.argv
         else str(run/'predicted_factors.npz'), **results)
