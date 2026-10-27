"""Transmission spectra of the group-number study (outputs of nl_groups_driver.sh).

Per detector: the white light of the reference variant (CRDS, all groups) fixes the transit
shape (t0, a/R*, inclination; P fixed) and the quadratic limb darkening of the white light;
every spectroscopic bin (BIN columns) of the reference is then fitted for Rp/R* and its limb
darkening (q1, q2), and every variant for Rp/R* with that bin's limb darkening; baselines
are quadratic in time. Depth differences between two variants come from the ratio of their
light curves, fitted with a quadratic baseline and the derivative of the transit model with
respect to the depth (paired: the noise both share cancels), with errors from the residual
scatter times the time-averaging beta factor.
Integrations excluded: the first SKIP of the exposure, and 25-65 of segment 1 (a moving
row-offset band crosses the trace; see Claude_progress.md).
  python nl_groups_spectrum.py <run folder> <detector> [<detector> ...] --out <out.npz>
"""
import sys, glob
from pathlib import Path
import numpy as np
import batman
from scipy.optimize import minimize

PERIOD, BIN, SKIP = 4.627669, 64, 10
BAND = (25, 66)
args = [a for a in sys.argv[1:] if not a.startswith('--')]
out = sys.argv[sys.argv.index('--out') + 1]
args.remove(out)
run, detectors = Path(args[0]), args[1:]


def load(detector, variant):
    segments = sorted(glob.glob(str(run/detector/'seg*'/f'spec_{variant}.npz')))
    data = [np.load(s) for s in segments]
    times = np.concatenate([d['times'] for d in data])
    spectra = np.concatenate([d['spectra'] for d in data])
    keep = np.ones(len(times), bool)
    keep[:SKIP] = False
    keep[BAND[0]:BAND[1]] = False   # segment 1 comes first
    return times[keep], spectra[keep], data[0]['wavelengths']


def model(params, times, rp, u):
    t0, a, inc = params
    p = batman.TransitParams()
    p.t0, p.per, p.rp, p.a, p.inc, p.ecc, p.w = t0, PERIOD, rp, a, inc, 0., 90.
    p.limb_dark, p.u = 'quadratic', list(u)
    return batman.TransitModel(p, times).light_curve(p)


def quadratic_u(q1, q2):
    return 2*np.sqrt(q1)*q2, np.sqrt(q1)*(1 - 2*q2)


def baseline_fit(flux, shape, x):
    """Least-squares quadratic baseline of flux/shape; returns residuals and the baseline."""
    design = np.stack([np.ones_like(x), x, x**2], 1)*shape[:, None]
    coefficients = np.linalg.lstsq(design, flux, rcond=None)[0]
    fitted = design @ coefficients
    return flux - fitted, fitted


def beta(residuals, sizes=(5, 10, 20, 40)):
    """Time-averaging factor: binned rms over the white-noise expectation (at least 1)."""
    sigma = np.std(residuals)
    values = []
    for n in sizes:
        m = len(residuals)//n
        binned = residuals[:m*n].reshape(m, n).mean(1)
        values.append(np.std(binned)/(sigma/np.sqrt(n)*np.sqrt(m/(m - 1))))
    return max(1., np.mean(values))


def depth_fit(flux, shape_params, times, x, u, rp0):
    """Rp/R* (limb darkening fixed), its error (linearised, beta-scaled), residual rms."""
    def cost(rp):
        r, _ = baseline_fit(flux, model(shape_params, times, rp[0], u), x)
        return np.sum(r**2)
    rp = minimize(cost, [rp0], method='Nelder-Mead', options=dict(xatol=1e-7, fatol=1e-12)).x[0]
    shape = model(shape_params, times, rp, u)
    residuals, fitted = baseline_fit(flux, shape, x)
    derivative = (model(shape_params, times, rp + 1e-4, u) - shape)/1e-4*fitted/shape
    design = np.stack([np.ones_like(x), x, x**2], 1)*shape[:, None]
    design = np.column_stack([design, derivative])
    covariance = np.linalg.inv(design.T @ design)*np.var(residuals)
    error = np.sqrt(covariance[-1, -1])*beta(residuals)
    return rp, error, np.std(residuals)


def difference_fit(flux, reference, shape_params, times, x, u, rp):
    """Depth difference (flux minus reference) from their ratio; error (beta-scaled)."""
    ratio = flux/reference
    ratio = ratio/np.median(ratio)
    shape = model(shape_params, times, rp, u)
    d_shape = (model(shape_params, times, np.sqrt(rp**2 + 1e-5), u) - shape)/1e-5/shape
    design = np.column_stack([np.ones_like(x), x, x**2, d_shape])
    coefficients = np.linalg.lstsq(design, ratio, rcond=None)[0]
    residuals = ratio - design @ coefficients
    covariance = np.linalg.inv(design.T @ design)*np.var(residuals)
    return coefficients[-1], np.sqrt(covariance[-1, -1])*beta(residuals)


results = {}
for detector in detectors:
    variants = sorted({Path(p).stem[5:] for p in glob.glob(str(run/detector/'seg*'/'spec_*.npz'))})
    times, reference, wavelengths = load(detector, 'crds_n20')
    x = (times - times.mean())/(times.max() - times.min())
    good = np.isfinite(wavelengths) & np.isfinite(reference).all(0)
    columns = np.nonzero(good)[0]
    bins = [b for b in np.array_split(columns, max(1, len(columns)//BIN))]
    white = np.nansum(reference[:, columns], 1)
    white = white/np.median(white)

    # White light: t0, a, inc, rp, q1, q2:
    def white_cost(p):
        t0, a, inc, rp, q1, q2 = p
        if not (0 < q1 < 1 and 0 < q2 < 1 and 80 < inc < 90 and 3 < a < 20 and 0 < rp < 0.2):
            return 1e10
        r, _ = baseline_fit(white, model((t0, a, inc), times, rp, quadratic_u(q1, q2)), x)
        return np.sum(r**2)
    mid = times[np.argmin(np.convolve(white, np.ones(21)/21, 'same')[10:-10]) + 10]
    best = None
    for start in ([mid, 9.2, 84.3, 0.083, 0.3, 0.3], [mid, 8.5, 83.5, 0.08, 0.1, 0.4],
                  [mid, 10., 85., 0.085, 0.5, 0.2]):
        fit = minimize(white_cost, start, method='Nelder-Mead',
                       options=dict(maxiter=20000, maxfev=20000, xatol=1e-8, fatol=1e-14))
        if best is None or fit.fun < best.fun:
            best = fit
    t0, a, inc, rp_white, q1w, q2w = best.x
    shape_params = (t0, a, inc)
    print(f'{detector}: white light t0 {t0:.5f}, a {a:.3f}, inc {inc:.3f}, rp {rp_white:.5f}, '
          f'q1 {q1w:.3f}, q2 {q2w:.3f}; {len(times)} integrations, {len(bins)} bins', flush=True)

    # Reference bins: rp and limb darkening:
    lds, rp_reference = [], []
    for cols in bins:
        flux = reference[:, cols].sum(1)
        flux = flux/np.median(flux)

        def cost(p):
            rp, q1, q2 = p
            if not (0 < q1 < 1 and 0 < q2 < 1 and 0 < rp < 0.2):
                return 1e10
            r, _ = baseline_fit(flux, model(shape_params, times, rp, quadratic_u(q1, q2)), x)
            return np.sum(r**2)
        fit = minimize(cost, [rp_white, q1w, q2w], method='Nelder-Mead',
                       options=dict(maxiter=4000, xatol=1e-7, fatol=1e-14))
        lds.append(quadratic_u(*fit.x[1:]))
        rp_reference.append(fit.x[0])

    # Every variant, with every bin's limb darkening; white light with its own:
    out_detector = dict(wavelength=np.array([np.nanmean(wavelengths[c]) for c in bins]),
                        bin_columns=np.array([[c[0], c[-1]] for c in bins]),
                        white_params=best.x, variants=np.array(variants))
    fluxes = {}
    for variant in variants:
        t_v, spectra, _ = load(detector, variant)
        assert np.allclose(t_v, times)
        f = np.array([spectra[:, c].sum(1) for c in bins]).T
        fluxes[variant] = f/np.median(f, 0)
        w = np.nansum(spectra[:, columns], 1)
        fluxes[variant + '_white'] = w/np.median(w)
        depth, error, rms = [], [], []
        for b in range(len(bins)):
            rp, e, s = depth_fit(fluxes[variant][:, b], shape_params, times, x, lds[b],
                                 rp_reference[b])
            depth.append(rp**2)
            error.append(2*rp*e)
            rms.append(s)
        rp, e, s = depth_fit(fluxes[variant + '_white'], shape_params, times, x,
                             quadratic_u(q1w, q2w), rp_white)
        out_detector[f'depth_{variant}'] = np.array(depth)
        out_detector[f'error_{variant}'] = np.array(error)
        out_detector[f'rms_{variant}'] = np.array(rms)
        out_detector[f'white_{variant}'] = np.array([rp**2, 2*rp*e, s])
        print(f'  {variant}: white depth {1e6*rp**2:.1f} +- {2e6*rp*e:.1f} ppm, rms '
              f'{1e6*s:.0f} ppm; median bin error {1e6*np.median(error):.0f} ppm', flush=True)

    # Paired differences: every variant minus the same correction with all groups, and self
    # minus CRDS with the same groups:
    groups = sorted({int(v.split('_n')[1]) for v in variants})
    pairs = [(f'{c}_n{n}', f'{c}_n{groups[-1]}') for c in ('crds', 'self') for n in groups[:-1]]
    pairs += [(f'self_n{n}', f'crds_n{n}') for n in groups]
    for v, r in pairs:
        diff = [difference_fit(fluxes[v][:, b], fluxes[r][:, b], shape_params, times, x, lds[b],
                               rp_reference[b]) for b in range(len(bins))]
        d, e = np.array(diff).T
        wd = difference_fit(fluxes[v + '_white'], fluxes[r + '_white'], shape_params, times, x,
                            quadratic_u(q1w, q2w), rp_white)
        out_detector[f'diff_{v}-{r}'] = d
        out_detector[f'differr_{v}-{r}'] = e
        out_detector[f'whitediff_{v}-{r}'] = np.array(wd)
        weights = 1/e**2
        mean = np.sum(weights*d)/np.sum(weights)
        chi2 = np.sum(((d - mean)/e)**2)
        chi0 = np.sum((d/e)**2)
        print(f'  {v} - {r}: white {1e6*wd[0]:+.1f} +- {1e6*wd[1]:.1f} ppm; bins mean '
              f'{1e6*mean:+.1f} +- {1e6/np.sqrt(np.sum(weights)):.1f} ppm, chi2 vs 0 '
              f'{chi0:.1f}, vs mean {chi2:.1f} ({len(d)} bins)', flush=True)
    results[detector] = out_detector

np.savez(out, **{f'{det}__{k}': v for det, o in results.items() for k, v in o.items()})
