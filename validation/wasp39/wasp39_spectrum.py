"""Transmission spectrum of WASP-39 b from stage 2 spectra (wasp39_stage2.py outputs).

Orbital parameters fixed to Carter, May et al. 2024 (arXiv:2407.13893, Table 1): P =
4.0552842 d, a/R* = 11.390, i = 87.7369 deg, e = 0, and the mid-transit time of the mode.
Every light curve (white, and each wavelength bin) is fitted for Rp/R*, quadratic limb
darkening (q1, q2 free in [0, 1], Kipping 2013; the paper fixed them to ExoTiC-LD values,
not available here) and a linear baseline in time, by least squares; errors are linearised
and scaled by the residual scatter and the time-averaging beta factor. Two variants of the
same mode can also be differenced on their light-curve ratio (paired; see --reference).
  python wasp39_spectrum.py <mode: prism|soss> <out.npz> <spec.npz> [...] --edges e0,e1,...
      [--skip N] [--reference <spec.npz> ...]
Columns whose spectra have non-finite values in more than 1% of the integrations are left
out of the bins.
"""
import sys
import numpy as np
import batman
from scipy.optimize import minimize

PERIOD, A_RS, INC = 4.0552842, 11.390, 87.7369
T0 = {'prism': 2459771.335647, 'soss': 2459787.5567843}


def arguments():
    args, options, key = [], {}, None
    for a in sys.argv[1:]:
        if a.startswith('--'):
            key = a[2:]
            options[key] = []
        elif key is None:
            args.append(a)
        else:
            options[key].append(a)
    return args, options


def load(files, skip):
    data = [np.load(f) for f in files]
    times = np.concatenate([d['times'] for d in data])
    spectra = np.concatenate([d['spectra'] for d in data])
    return times[skip:], spectra[skip:], data[0]['wavelengths']


def transit(times, mode, rp, q1, q2):
    p = batman.TransitParams()
    p.t0, p.per, p.rp, p.a, p.inc, p.ecc, p.w = T0[mode], PERIOD, rp, A_RS, INC, 0., 90.
    p.limb_dark = 'quadratic'
    p.u = [2*np.sqrt(q1)*q2, np.sqrt(q1)*(1 - 2*q2)]
    return batman.TransitModel(p, times).light_curve(p)


def beta(residuals, sizes=(10, 30, 100, 300)):
    sigma, values = np.std(residuals), []
    for n in sizes:
        m = len(residuals)//n
        if m < 5:
            continue
        binned = residuals[:m*n].reshape(m, n).mean(1)
        values.append(np.std(binned)/(sigma/np.sqrt(n)*np.sqrt(m/(m - 1))))
    return max(1., np.mean(values)) if values else 1.


def fit(flux, times, mode, x, start=(0.145, 0.3, 0.3)):
    """Rp/R*, q1, q2 and a linear baseline; returns rp, its error, q1, q2, rms, residuals."""
    design = np.stack([np.ones_like(x), x], 1)

    def residuals(p):
        rp, q1, q2 = p
        if not (0 < q1 < 1 and 0 < q2 < 1 and 0 < rp < 0.3):
            return None
        shape = transit(times, mode, rp, q1, q2)
        coefficients = np.linalg.lstsq(design*shape[:, None], flux, rcond=None)[0]
        return flux - (design @ coefficients)*shape

    def cost(p):
        r = residuals(p)
        return 1e10 if r is None else np.sum(r**2)
    best = min((minimize(cost, s, method='Nelder-Mead',
                         options=dict(maxiter=4000, xatol=1e-7, fatol=1e-14))
                for s in (start, (start[0], 0.1, 0.4), (start[0], 0.6, 0.2))),
               key=lambda r: r.fun)
    rp, q1, q2 = best.x
    r = residuals(best.x)
    # Linearised errors (rp, q1, q2 and the baseline):
    shape = transit(times, mode, rp, q1, q2)
    baseline = flux - r
    columns = [baseline]
    for j, step in enumerate((1e-5, 1e-4, 1e-4)):
        p = np.array(best.x, float)
        p[j] = min(p[j] + step, 0.9999) if j else p[j] + step
        columns.append((transit(times, mode, *p) - shape)/step*baseline/shape)
    jac = np.column_stack(columns[1:] + [design*shape[:, None]])
    with np.errstate(all='ignore'):
        covariance = np.linalg.pinv(jac.T @ jac)*np.var(r)
    return rp, np.sqrt(covariance[0, 0])*beta(r), q1, q2, np.std(r), r


def difference(flux, reference, times, mode, x, rp, q1, q2):
    """Depth difference (flux minus reference) from their ratio, with the reference's shape."""
    ratio = flux/reference
    ratio = ratio/np.median(ratio)
    shape = transit(times, mode, rp, q1, q2)
    d_shape = (transit(times, mode, np.sqrt(rp**2 + 1e-5), q1, q2) - shape)/1e-5/shape
    design = np.column_stack([np.ones_like(x), x, d_shape])
    coefficients = np.linalg.lstsq(design, ratio, rcond=None)[0]
    r = ratio - design @ coefficients
    covariance = np.linalg.inv(design.T @ design)*np.var(r)
    return coefficients[-1], np.sqrt(covariance[-1, -1])*beta(r)


def bin_fluxes(spectra, wavelengths, edges):
    good = np.isfinite(wavelengths) & (np.mean(~np.isfinite(spectra), 0) < 0.01)
    fluxes, centres, columns = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        use = good & (wavelengths >= lo) & (wavelengths < hi)
        if use.sum() == 0:
            fluxes.append(None)
            centres.append(np.nan)
            columns.append(0)
            continue
        f = np.nansum(spectra[:, use], 1)
        fluxes.append(f/np.median(f))
        centres.append(np.mean(wavelengths[use]))
        columns.append(int(use.sum()))
    return fluxes, np.array(centres), np.array(columns), good


args, options = arguments()
mode, out, files = args[0], args[1], args[2:]
edges = np.array([float(e) for e in options['edges'][0].split(',')])
skip = int(options.get('skip', ['0'])[0])
times, spectra, wavelengths = load(files, skip)
x = (times - times.mean())/(times.max() - times.min())
fluxes, centres, ncolumns, good = bin_fluxes(spectra, wavelengths, edges)
white = np.nansum(spectra[:, good & (wavelengths >= edges[0]) & (wavelengths < edges[-1])], 1)
white = white/np.median(white)
rp_w, e_w, q1_w, q2_w, rms_w, res_w = fit(white, times, mode, x)
print(f'{mode} {files[0]}...: {len(times)} integrations; white Rp/R* {rp_w:.5f} +- {e_w:.5f}, '
      f'depth {1e6*rp_w**2:.0f} +- {2e6*rp_w*e_w:.0f} ppm, q1 {q1_w:.3f} q2 {q2_w:.3f}, '
      f'rms {1e6*rms_w:.0f} ppm', flush=True)
results = dict(edges=edges, centres=centres, ncolumns=ncolumns, white=np.array(
    [rp_w**2, 2*rp_w*e_w, q1_w, q2_w, rms_w]), white_residuals=res_w, times=times)
depth, error, q1s, q2s, rms = [], [], [], [], []
for f in fluxes:
    if f is None:
        depth.append(np.nan); error.append(np.nan); q1s.append(np.nan); q2s.append(np.nan)
        rms.append(np.nan)
        continue
    rp, e, q1, q2, s, _ = fit(f, times, mode, x, (rp_w, q1_w, q2_w))
    depth.append(rp**2); error.append(2*rp*e); q1s.append(q1); q2s.append(q2); rms.append(s)
results.update(depth=np.array(depth), error=np.array(error), q1=np.array(q1s),
               q2=np.array(q2s), rms=np.array(rms))
print(f'  bins: {np.isfinite(depth).sum()}; median depth error '
      f'{1e6*np.nanmedian(error):.0f} ppm', flush=True)
if 'reference' in options:
    t_r, s_r, w_r = load(options['reference'], skip)
    assert np.allclose(t_r, times) and np.allclose(w_r, wavelengths, equal_nan=True)
    f_r, _, _, _ = bin_fluxes(s_r, wavelengths, edges)
    diff, differr = [], []
    for f, fr, rp, q1, q2 in zip(fluxes, f_r, np.sqrt(results['depth']), q1s, q2s):
        if f is None or fr is None:
            diff.append(np.nan); differr.append(np.nan)
            continue
        d, e = difference(f, fr, times, mode, x, rp, q1, q2)
        diff.append(d); differr.append(e)
    w_ref = np.nansum(s_r[:, good & (wavelengths >= edges[0]) & (wavelengths < edges[-1])], 1)
    wd = difference(white, w_ref/np.median(w_ref), times, mode, x, rp_w, q1_w, q2_w)
    results.update(diff=np.array(diff), differr=np.array(differr), whitediff=np.array(wd))
    print(f'  minus reference: white {1e6*wd[0]:+.1f} +- {1e6*wd[1]:.1f} ppm', flush=True)
np.savez(out, **results)
