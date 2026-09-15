"""NumPy/SciPy implementation of the repository's Marsh C algorithm.

The polynomial path deliberately retains the legacy bounds, integer aperture,
edge scaling, empirical-variance formula and iteration order. It is not a
rewrite of those scientific choices. Arrays remain on the detector pixel grid.
The optional GP is a jointly fitted inducing-point approximation with fixed
hyperparameters; returned extraction errors remain conditional on fitted P.
"""
import numpy as np
from scipy.linalg import cho_factor, cho_solve, cholesky, qr, solve_triangular


def validate(data, centroids, variance=None):
    data = np.asarray(data, dtype=np.float64)
    centers = np.ascontiguousarray(centroids, dtype=np.float64)
    if data.ndim != 2 or min(data.shape) == 0:
        raise ValueError('data must be a nonempty (row, column) image')
    if centers.shape != (data.shape[1],) or not np.all(np.isfinite(centers)):
        raise ValueError('centroids must contain one finite value per column')
    if not np.all(np.isfinite(data)):
        raise ValueError('data must be finite; use -9999 for legacy masked pixels')
    if variance is not None:
        variance = np.asarray(variance, dtype=np.float64)
        if variance.shape != data.shape:
            raise ValueError('data_variance must match data.shape')
    return data, centers, variance


def check_aperture(centers, nrows, spacing, aperture):
    if not np.isfinite(aperture) or aperture < 1:
        raise ValueError('aperture_radius must be finite and at least one pixel')
    if not np.isfinite(spacing) or spacing <= 0:
        raise ValueError('polynomial_spacing must be finite and positive')
    initial = radius = int(aperture)  # C's CheckAperture accepts an int.
    k = int(radius / spacing + 0.5)
    i = 0
    while i < len(centers):
        upper = int(centers[i] + spacing * (k + 1.) + .5)
        lower = int(centers[i] - spacing * (k + 1.) - .5)
        if upper >= nrows:
            radius -= 1
            k = int(radius / spacing + .5)
            lower = int(centers[i] - spacing * (k + 1.) - .5)
            i -= 1
        if lower < 0 and upper >= nrows:
            radius -= 1
        if lower < 0 and upper < nrows:
            radius -= 1
            k = int(radius / spacing + .5)
            i -= 1
        if radius < 1:
            radius = initial
            break
        i += 1
    return radius


def _report_aperture(requested, radius):
    # The original C backend prints rather than raising a Python warning.
    if radius != int(requested):
        print(f'MARSH CODE: Changed central aperture to {radius}')


def geometry(centers, nrows, aperture, spacing, min_column=None, max_column=None,
             simple=False):
    radius = check_aperture(centers, nrows, spacing, aperture)
    count = 2 * int(radius / spacing + .5) + 1
    offset = -(spacing * (1. + (count - 1.) / 2.))
    if simple:
        low, high = centers - radius + .5, centers + radius + .5
    else:
        low = (centers + offset) + spacing + .5
        high = (centers + offset) + count * spacing + .5
    ncol = len(centers)
    start, stop, found = -1, ncol, False
    # Reproduce RangeDetector's directional/end-point decisions.
    upper_test = np.trunc(high).astype(int)
    lower_test = np.trunc(low).astype(int)
    if simple:
        upper_test = np.trunc(centers + radius) + .5
        lower_test = np.trunc(centers - radius) + .5
    if centers[0] > centers[ncol // 2] and np.any(upper_test[[0, -1]] >= nrows):
        start = 0
        for j, value in enumerate(np.trunc(high).astype(int)):
            if value < nrows and not found:
                start, found = j, True
            elif value == nrows and found:
                stop = j - 1
                break
    elif centers[0] < centers[ncol // 2] and np.any(lower_test[[0, -1]] < 0):
        start = 0
        for j, value in enumerate(np.trunc(low).astype(int)):
            if value == 0 and not found:
                start, found = j, True
            elif value < 0 and found:
                stop = j - 1
                break
    cmin = 0 if min_column is None else min_column
    cmax = ncol if max_column is None else max_column
    if int(cmin) != cmin or int(cmax) != cmax:
        raise ValueError('column bounds must be integers')
    cmin = int(cmin) + (cmin != 0)
    cmax = int(cmax) - (cmax != 0)
    if start == -1:
        start, stop = 0, ncol - 1
    start = max(start, cmin)
    if cmax != 0:
        stop = min(stop, cmax)
    if not 0 <= start <= stop < ncol:
        raise ValueError('legacy column bounds select an invalid detector range')
    columns = np.arange(start, stop + 1)
    low, high = low[columns], high[columns]
    lo, hi = np.trunc(low).astype(int), np.trunc(high).astype(int)
    if np.any(lo < 0) or np.any(hi >= nrows):
        raise ValueError('trace/aperture extends outside the detector')
    rows = np.arange(nrows)[None, :]
    inside = (rows >= lo[:, None]) & (rows <= hi[:, None])
    return columns, low, high, inside, radius, count, offset


def resample(data, low, high, sentinel=True):
    result = data.copy()
    for j, (lower, upper) in enumerate(zip(low, high)):
        lo, hi = int(lower), int(upper)
        if not sentinel or result[j, lo] != -9999:
            result[j, lo] *= 1. - (lower - lo)
            result[j, hi] *= upper - hi
    return result


def overlaps(centers, rows, spacing, count, offset):
    """C CalculateQ: integrated linear spatial basis on detector pixels."""
    d = np.abs(centers[:, None, None] +
               spacing * np.arange(1, count + 1)[None, None, :] + offset -
               np.arange(rows)[None, :, None])
    q = np.zeros_like(d)
    full = (d < .5 + spacing) & (d + spacing <= .5)
    inner = (d < .5 + spacing) & ~full & (d <= .5)
    outer = (d < .5 + spacing) & ~full & (d > .5)
    q[full] = spacing
    q[inner] = spacing / 2. + (.5 - d[inner]) - (.5 - d[inner])**2 / (2. * spacing)
    q[outer] = (.5 - d[outer]) + (.5 - d[outer])**2 / (2. * spacing) + spacing / 2.
    return q


def _gp_basis(ncolumns, options):
    options = dict(options or {})
    unknown = set(options) - {'length_scale', 'amplitude', 'n_inducing', 'kernel'}
    if unknown:
        raise ValueError(f'Unknown GP options: {sorted(unknown)}')
    length = float(options.get('length_scale', max(10., ncolumns / 5.)))
    amplitude = float(options.get('amplitude', 1.))
    number = options.get('n_inducing', min(16, ncolumns))
    kernel = options.get('kernel', 'matern32')
    if not np.isfinite(length) or length <= 0 or not np.isfinite(amplitude) or amplitude <= 0:
        raise ValueError('GP length_scale and amplitude must be finite and positive')
    if int(number) != number or not 2 <= number <= ncolumns:
        raise ValueError('GP n_inducing must be between 2 and the selected column count')
    if kernel not in ('matern32', 'matern52'):
        raise ValueError('GP kernel must be matern32 or matern52')
    def covariance(x, y):
        distance = np.abs(x[:, None] - y[None, :]) / length
        if kernel == 'matern32':
            r = np.sqrt(3.) * distance
            return amplitude**2 * (1. + r) * np.exp(-r)
        r = np.sqrt(5.) * distance
        return amplitude**2 * (1. + r + r*r / 3.) * np.exp(-r)
    x = np.arange(ncolumns, dtype=float)
    z = np.linspace(0., ncolumns - 1., int(number))
    kzz = covariance(z, z) + np.eye(len(z)) * amplitude**2 * 1e-10
    # g = Kxz Lzz^-T u, u ~ N(0,I): explicit low-rank GP prior.
    return solve_triangular(cholesky(kzz, lower=True), covariance(z, x), lower=True).T


def fit_profile(data, centers, aperture, ron, gain, nsigma, spacing, order,
                min_column=None, max_column=None, variance=None,
                profile_method='polynomial', gp_options=None):
    data, centers, variance = validate(data, centers, variance)
    if not np.isfinite(nsigma) or nsigma <= 0:
        raise ValueError('profile nsigma must be finite and positive')
    if int(order) != order or order < 1:
        raise ValueError('polynomial_order is a positive number of terms')
    columns, low, high, inside, radius, count, offset = geometry(
        centers, data.shape[0], aperture, spacing, min_column, max_column)
    _report_aperture(aperture, radius)
    a = resample(data[:, columns].T, low, high)
    good = inside & (a != -9999)
    rs = np.where(good, a, 0.).sum(axis=1)
    if np.any(rs == 0):
        raise ValueError('cannot estimate a profile from a zero-sum aperture')
    e = np.where(good, a, 0.) / rs[:, None]
    q = overlaps(centers[columns], data.shape[0], spacing, count, offset)
    if profile_method == 'polynomial':
        basis = np.arange(len(columns), dtype=float)[:, None] ** np.arange(int(order))
        regularized = False
    elif profile_method == 'gp':
        basis = _gp_basis(len(columns), gp_options)
        regularized = True
    else:
        raise ValueError('profile_method must be polynomial or gp')
    # Shared pixels couple spatial components in BOTH profile models.
    design = (q[..., None] * basis[:, None, None, :]).reshape(*a.shape, -1)
    v = np.ones_like(a) if variance is None else variance[:, columns].T.copy()
    if np.any(v[good] <= 0) or not np.all(np.isfinite(v[good])):
        raise ValueError('data_variance must be finite and positive in the valid aperture')
    def empirical_variance():
        varrs = np.where(good, v, 0.).sum(axis=1)
        return ((1. / rs[:, None]**2 - 2. * a / rs[:, None]**3) * v +
                (e / rs[:, None])**2 * varrs[:, None])
    vare = empirical_variance()
    for iteration in range(a.size + 2):
        if np.any(vare[good] <= 0) or not np.all(np.isfinite(vare[good])):
            raise ValueError('nonpositive empirical profile variance')
        matrix = design[good]
        weight = 1. / vare[good]
        normal = matrix.T @ (matrix * weight[:, None])
        rhs = matrix.T @ (e[good] * weight)
        if regularized:
            normal.flat[::normal.shape[0] + 1] += 1.
            coefficients = cho_solve(cho_factor(normal), rhs)
        else:
            # Same normal equations/unpivoted QR as C, with LAPACK rounding.
            orthogonal, triangular = qr(normal, mode='economic')
            coefficients = solve_triangular(triangular, orthogonal.T @ rhs)
        p = np.maximum(design @ coefficients, 0.) * inside
        sums = p.sum(axis=1)
        if np.any(sums <= 0) or not np.all(np.isfinite(p)):
            raise ValueError('profile fit has no positive, finite support')
        p /= sums[:, None]
        if variance is None:
            v = (ron / gain)**2 + np.abs(rs[:, None] * p) / gain
        vare = empirical_variance()  # C updates before, rather than after, rejection.
        # C always refits once before beginning rejection; E and RS stay fixed.
        if iteration == 0:
            continue
        reject = good & ((a - rs[:, None] * p)**2 >= nsigma**2 * v)
        if not np.any(reject):
            break
        good[reject] = False
        if np.any(good.sum(axis=1) == 0):
            raise ValueError('profile rejection removed an entire column')
    output = np.zeros_like(data)
    output[:, columns] = p.T
    output[output > 1.] = 0.
    return output


def extract(data, centers, aperture, ron, gain, nsigma, spacing, profile,
            min_column=None, max_column=None, variance=None, background=None):
    data, centers, variance = validate(data, centers, variance)
    columns, low, high, inside, radius, _, _ = geometry(
        centers, data.shape[0], aperture, spacing, min_column, max_column)
    _report_aperture(aperture, radius)
    a = resample(data[:, columns].T, low, high)
    p = np.asarray(profile, dtype=float)[:, columns].T
    good = inside & (a != -9999)
    flux = np.where(good, a, 0.).sum(axis=1)
    supplied = None if variance is None else variance[:, columns].T
    for iteration in range(a.size + 1):
        v = (ron / gain)**2 + np.abs(flux[:, None] * p) / gain if supplied is None else supplied
        if np.any(v[good] <= 0) or not np.all(np.isfinite(v[good])):
            raise ValueError('extraction requires positive finite pixel variances')
        inverse = np.zeros_like(a)
        inverse[good] = 1. / v[good]
        denominator = (p*p*inverse).sum(axis=1)
        if np.any(denominator <= 0):
            raise ValueError('no weighted profile support in an extracted column')
        weights = p * inverse / denominator[:, None]
        flux = (weights * np.where(good, a, 0.)).sum(axis=1)
        varflux = (weights**2 * np.where(good, v, 0.)).sum(axis=1)
        if nsigma == 0:
            break
        score = np.abs(a - flux[:, None] * p) - nsigma * (
            np.sqrt(np.maximum(v, 0.)) + np.sqrt(varflux)[:, None] * p)
        score[~good] = -np.inf
        worst = np.argmax(score)  # column-major scan; first tie, as in C.
        if score.flat[worst] <= 0:
            break
        good.flat[worst] = False
    output = np.zeros((3 if background is None else 4, data.shape[1]))
    output[0] = np.arange(data.shape[1])
    output[1, columns], output[2, columns] = flux, 1. / varflux
    if background is not None:
        b = resample(np.asarray(background)[:, columns].T, low, high, sentinel=False)
        output[3, columns] = (weights * np.where(good & (b != -9999), b, 0.)).sum(axis=1)
    return output


def simple_extract(data, centers, aperture, min_column=None, max_column=None):
    data, centers, _ = validate(data, centers)
    columns, low, high, inside, radius, _, _ = geometry(
        centers, data.shape[0], aperture, 1., min_column, max_column, simple=True)
    _report_aperture(aperture, radius)
    a = resample(data[:, columns].T, low, high, sentinel=False)
    # SimpleExtraction loops i < floating pmax, unlike optimal extraction.
    inside &= np.arange(data.shape[0])[None, :] < high[:, None]
    output = np.zeros(data.shape[1])
    output[columns] = np.where(inside, a, 0.).sum(axis=1)
    return output, float(radius)
