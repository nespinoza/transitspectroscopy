"""Explicitly conditional least-squares transit diagnostics for SOSS validation.

These fits are not a replacement for a posterior analysis. Geometry is fixed,
quadratic limb darkening is fitted per band, and reported local errors include
a residual time-binning inflation factor but not geometry/calibration uncertainty.
"""
import numpy as np
from scipy.optimize import least_squares


def fit_transit(times, flux, error, config, *, fit_time=False, q_start=(.3, .1)):
    import batman
    times, flux, error = map(lambda a: np.asarray(a, float), (times, flux, error))
    if not (times.shape == flux.shape == error.shape) or len(times) < 30:
        raise ValueError('Transit fitting requires matching time series with at least 30 points')
    if not np.all(np.isfinite(times) & np.isfinite(flux) & np.isfinite(error) & (error > 0)):
        raise ValueError('Transit fitting requires finite flux and positive errors')
    p = batman.TransitParams()
    p.t0, p.per, p.a = config['t0_bjd_tdb'], config['period_days'], config['a_rs']
    # b_circular is a*cos(i), not the eccentric conjunction impact parameter.
    p.inc = np.degrees(np.arccos(config['b_circular']/p.a))
    p.ecc, p.w, p.rp = config['eccentricity'], config['omega_degrees'], .082
    p.limb_dark, p.u = 'quadratic', [.2, .3]
    cadence = np.median(np.diff(times))
    model = batman.TransitModel(p, times, supersample_factor=5, exp_time=cadence)
    trial = model.light_curve(p)
    inside, outside = trial < .9999, trial > .999999
    if inside.sum() < 10 or (outside & (times < p.t0)).sum() < 10 or (outside & (times > p.t0)).sum() < 10:
        raise ValueError('Transit validation needs transit and both out-of-transit baselines')
    norm = np.median(flux[outside])
    if norm <= 0:
        raise ValueError('Nonpositive baseline flux')
    y, sigma = flux/norm, error/norm
    t = (times-config['t0_bjd_tdb'])/np.ptp(times)
    x0 = [.082, *q_start, 1., 0., 0.]
    low, high = [.03, 0., 0., .95, -.05, -.05], [.2, 1., 1., 1.05, .05, .05]
    if fit_time:
        x0.append(0.)
        low.append(-.02)
        high.append(.02)
    def evaluate(theta):
        p.rp = theta[0]
        q1, q2 = theta[1:3]
        p.u = [2*np.sqrt(q1)*q2, np.sqrt(q1)*(1-2*q2)]
        p.t0 = config['t0_bjd_tdb'] + (theta[6] if fit_time else 0.)
        baseline = theta[3]+theta[4]*t+theta[5]*t*t
        return model.light_curve(p)*baseline
    residual = lambda theta: (evaluate(theta)-y)/sigma
    fit = least_squares(residual, x0, bounds=(low, high), x_scale='jac', max_nfev=1500)
    if not fit.success:
        raise RuntimeError('Transit optimizer failed: '+fit.message)
    prediction = evaluate(fit.x)
    r = y-prediction
    dof = len(y)-len(fit.x)
    reduced_chi2 = float(np.sum((r/sigma)**2)/dof)
    beta = 1.
    white = np.std(r, ddof=1)
    for n in (4, 8, 16):
        count = len(r)//n
        if count >= 3 and white > 0:
            measured = np.std(r[:count*n].reshape(count,n).mean(axis=1),ddof=1)
            beta = max(beta, measured/(white/np.sqrt(n)*np.sqrt(count/(count-1))))
    information = fit.jac.T@fit.jac
    rank = np.linalg.matrix_rank(information)
    covariance = np.linalg.pinv(information)*max(1., reduced_chi2)*beta**2
    depth = fit.x[0]**2*1e6
    depth_error = 2*fit.x[0]*np.sqrt(max(0., covariance[0,0]))*1e6 if rank == len(fit.x) else np.nan
    names = ['rp_rs','q1','q2','baseline','linear','quadratic']+(['delta_t0_days'] if fit_time else [])
    boundary = [name for name,v,l,h in zip(names,fit.x,low,high) if min(v-l,h-v) < 1e-4*(h-l)]
    metrics = dict(depth_ppm=float(depth), depth_error_ppm=float(depth_error),
        t0_bjd_tdb=float(p.t0), parameters=dict(zip(names,map(float,fit.x))),
        residual_rms_ppm=float(np.sqrt(np.mean(r*r))*1e6), beta=float(beta),
        reduced_chi2=reduced_chi2, points=len(y), covariance_rank=int(rank),
        boundary_parameters=boundary, normalization=float(norm),
        uncertainty='local covariance, residual variance and time-binning inflation; conditional on fixed geometry')
    return metrics, dict(time_bjd_tdb=times, normalized_flux=y, normalized_error=sigma,
                         model=prediction, residual_ppm=r*1e6)


def transmission_spectrum(times, trace, config, *, t0=None):
    """Fit log-spaced bins using a fixed, complete band in every integration."""
    config = dict(config)
    if t0 is not None:
        config['t0_bjd_tdb'] = t0
    first = config.get('trim_first_integrations', 80)
    times = np.asarray(times)[first:]
    wave, common = trace['wavelength'], trace['white_light_columns']
    resolution = config.get('resolving_power', 30.)
    if resolution <= 0 or common.sum() < 5:
        raise ValueError('Invalid transmission binning')
    edges = [float(wave[common].min())]
    while edges[-1] < wave[common].max():
        edges.append(edges[-1]*(1+1/resolution))
    rows, diagnostics = [], []
    for low, high in zip(edges[:-1],edges[1:]):
        use = common & (wave >= low) & (wave < high)
        if use.sum() < 5:
            continue
        flux = trace['flux'][first:,use].sum(axis=1)
        error = np.sqrt((trace['error'][first:,use]**2).sum(axis=1))
        try:
            m, _ = fit_transit(times,flux,error,config)
        except (ValueError, RuntimeError) as error:
            rows.append([np.mean(wave[use]), low, high, use.sum(), np.nan, np.nan,
                         np.nan, np.nan, 0, 1])
            diagnostics.append(dict(status='failed', lower_microns=low, upper_microns=high,
                                    reason=str(error)))
            continue
        m['status'] = 'complete'
        rows.append([np.mean(wave[use]), low, high, use.sum(),m['depth_ppm'],m['depth_error_ppm'],
                     m['residual_rms_ppm'],m['beta'],len(m['boundary_parameters']),0])
        diagnostics.append(m)
    if not rows:
        raise ValueError('No transmission bins with sufficient complete columns')
    return np.asarray(rows), diagnostics
