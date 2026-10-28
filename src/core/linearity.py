"""Self non-linearity correction: a per-pixel correction fitted on the science data themselves.

The mean ramp of a pixel over many integrations is measured very precisely, so its curvature
can be fitted directly. For every pixel the correction is

    g(x) = x + (top/2) [a2 T2(u) + a3 T3(u)],    u = 2x/top - 1,

where ``x`` are the raw counts (DN), ``T2`` and ``T3`` the Chebyshev polynomials of degree 2
and 3 and ``top`` the end of the fitted range; ``a2`` and ``a3`` are fitted so that ``g`` of
the mean ramp is a straight line in time (generalized least squares, with read and Poisson
noise). The corrected counts are

    h(x) = gain (g(x) - g(0)) / g'(0),

so that ``h(0) = 0`` and ``h'(0) = gain`` (e.g., the linear coefficient of the CRDS
correction, so that corrected pixels keep its relative gains). Outside ``[0, top]``, ``h``
continues as straight lines with its value and slope at the ends. The Chebyshev basis keeps
the coefficients nearly uncorrelated and comparable from pixel to pixel.

Several mean ramps (e.g., the exposures of several visits of the same star, with the same
subarray) can be fitted together: they share ``a2`` and ``a3`` and each has its own zero
point and slope (`fit_self_linearity`).
"""

import warnings

import numpy as np

__all__ = ['SelfLinearity', 'mean_ramp_sums', 'fit_self_linearity']


def mean_ramp_sums(data, groupdq=None, nsigma=5.0, rows_per_block=4):
    """Sums and numbers of the groups of the unflagged, non-outlier integrations of every pixel.

    For every pixel, the groups of an integration from the first flagged one on (any
    ``groupdq`` flag, e.g. saturation) are excluded, and so is the whole integration if any of
    its remaining groups deviates from the median over the integrations by more than
    ``nsigma`` times their robust scatter (e.g., a cosmic ray). Sums of several exposures can
    be added before dividing.

    Parameters
    ----------
    data : numpy.ndarray
        ``(integration, group, row, column)`` raw counts (DN), before any linearity correction.
    groupdq : numpy.ndarray, optional
        Group data-quality flags like ``data``; non-zero flags exclude a group and the later
        ones of its integration.
    nsigma : float, optional
        Outlier threshold of an integration's groups, in robust standard deviations (1.4826
        times the median absolute deviation over the integrations). Default is 5.
    rows_per_block : int, optional
        Rows processed at a time (memory). Default is 4.

    Returns
    -------
    sums : numpy.ndarray
        ``(group, row, column)`` sums of the counts used (float64).
    counts : numpy.ndarray
        ``(group, row, column)`` numbers of integrations used.
    """
    nints, ngroups, nrows, ncols = data.shape
    sums = np.zeros((ngroups, nrows, ncols))
    counts = np.zeros((ngroups, nrows, ncols), dtype=np.int64)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        for r0 in range(0, nrows, rows_per_block):
            r1 = min(nrows, r0 + rows_per_block)
            x = np.asarray(data[:, :, r0:r1], dtype=np.float64)
            good = np.isfinite(x)
            if groupdq is not None:
                good &= np.asarray(groupdq[:, :, r0:r1]) == 0
            good = np.cumprod(good, axis=1).astype(bool)  # up to the first bad group
            values = np.where(good, x, np.nan)
            median = np.nanmedian(values, axis=0)
            scatter = 1.4826*np.nanmedian(np.abs(values - median), axis=0)
            deviation = np.abs(values - median)/np.where(scatter > 0, scatter, np.inf)
            outlier = np.nanmax(np.where(good, deviation, 0.), axis=1) > nsigma
            good &= ~outlier[:, None]
            sums[:, r0:r1] = np.where(good, x, 0.).sum(axis=0)
            counts[:, r0:r1] = good.sum(axis=0)
    return sums, counts


class SelfLinearity:
    """Coefficients of a self non-linearity correction (see the module description).

    Parameters
    ----------
    a2, a3 : numpy.ndarray
        ``(row, column)`` Chebyshev coefficients; NaN for pixels without a correction.
    top : numpy.ndarray
        ``(row, column)`` end of the fitted range of every pixel (DN).
    metadata : dict, optional
        Free-form information (e.g., detector, subarray corner and size, fit options).
    mean_ramps, counts : list of numpy.ndarray, optional
        The ``(group, row, column)`` mean ramps and numbers of integrations the coefficients
        were fitted to, kept so that fits can be combined later.
    """

    def __init__(self, a2, a3, top, metadata=None, mean_ramps=None, counts=None):
        self.a2 = np.asarray(a2, dtype=float)
        self.a3 = np.asarray(a3, dtype=float)
        self.top = np.asarray(top, dtype=float)
        if not self.a2.shape == self.a3.shape == self.top.shape or self.a2.ndim != 2:
            raise ValueError('a2, a3 and top must be (row, column) arrays of the same shape')
        self.metadata = dict(metadata or {})
        self.mean_ramps = list(mean_ramps or [])
        self.counts = list(counts or [])

    @property
    def pixels(self):
        """``(row, column)`` True where the correction is defined."""
        return (np.isfinite(self.a2) & np.isfinite(self.a3) & np.isfinite(self.top)
                & (self.top > 0))

    @property
    def shape(self):
        return self.a2.shape

    def correct(self, x, gain=None, rows=None, columns=None):
        """Corrected counts ``h(x)`` of raw counts ``x``.

        Parameters
        ----------
        x : numpy.ndarray
            Raw counts (DN) ``(..., row, column)`` of the whole map, or ``(..., pixel)`` of the
            pixels ``rows``, ``columns``.
        gain : numpy.ndarray or float, optional
            ``h'(0)`` of every pixel, like the last axes of ``x`` (e.g., the CRDS linear
            coefficient). Default is 1.
        rows, columns : numpy.ndarray, optional
            Pixel indices, if ``x`` holds only some pixels.

        Returns
        -------
        numpy.ndarray
            ``h(x)`` (float64); ``x`` where the correction is not defined.
        """
        index = (slice(None), slice(None)) if rows is None else (rows, columns)
        a2, a3, top = self.a2[index], self.a3[index], self.top[index]
        defined = self.pixels[index]
        a2, a3 = np.where(defined, a2, 0.), np.where(defined, a3, 0.)
        top = np.where(defined, top, 1.)
        gain = 1. if gain is None else np.asarray(gain, dtype=float)
        x = np.asarray(x, dtype=float)
        inside = np.clip(x, 0., top)
        u = 2*inside/top - 1
        g = inside + top/2*(a2*(2*u**2 - 1) + a3*(4*u**3 - 3*u))
        g0 = top/2*(a2 - a3)
        d0 = 1 - 4*a2 + 9*a3
        d1 = 1 + 4*a2 + 9*a3
        h = gain*(g - g0)/d0
        h = np.where(x > top, h + gain*d1/d0*(x - top), np.where(x < 0, gain*x, h))
        return np.where(defined, h, x)

    def save(self, path):
        """Save to a NumPy ``.npz`` file (`load` reads it back)."""
        arrays = dict(a2=self.a2, a3=self.a3, top=self.top,
                      metadata_keys=np.array(list(self.metadata), dtype=str),
                      metadata_values=np.array([repr(v) for v in self.metadata.values()],
                                               dtype=str))
        for j, (mean, count) in enumerate(zip(self.mean_ramps, self.counts)):
            arrays[f'mean_ramp_{j}'] = mean
            arrays[f'counts_{j}'] = count
        np.savez(path, **arrays)

    @classmethod
    def load(cls, path):
        """Read a file written by `save`, or any ``.npz`` file with ``a2``, ``a3`` and ``top``."""
        import ast
        with np.load(path) as f:
            metadata = {}
            if 'metadata_keys' in f.files:
                for key, value in zip(f['metadata_keys'], f['metadata_values']):
                    try:
                        metadata[str(key)] = ast.literal_eval(str(value))
                    except (ValueError, SyntaxError):
                        metadata[str(key)] = str(value)
            n = len([k for k in f.files if k.startswith('mean_ramp_')])
            return cls(f['a2'], f['a3'], f['top'], metadata,
                       [f[f'mean_ramp_{j}'] for j in range(n)],
                       [f[f'counts_{j}'] for j in range(n)])


def _fit_block(mean_ramps, counts, used, rows, cols, top, read_variance, gain):
    """Least-squares (a2, a3, z_1, b_1, ...) of the pixels ``rows``, ``cols``."""
    nsets = len(mean_ramps)
    nparameters = 2 + 2*nsets
    normal = np.zeros((len(rows), nparameters, nparameters))
    vector = np.zeros((len(rows), nparameters))
    for s, (mean, count, use) in enumerate(zip(mean_ramps, counts, used)):
        x = mean[:, rows, cols].T                               # (pixel, group)
        n = count[:, rows, cols].T.astype(float)
        u_ok = use[:, rows, cols].T
        ngroups = x.shape[1]
        x = np.where(u_ok, x, 0.)
        u = 2*x/top[:, None] - 1
        design = np.zeros((len(rows), ngroups, nparameters))
        design[:, :, 0] = -top[:, None]/2*(2*u**2 - 1)
        design[:, :, 1] = -top[:, None]/2*(4*u**3 - 3*u)
        design[:, :, 2 + 2*s] = 1.
        design[:, :, 3 + 2*s] = np.arange(ngroups)[None, :]
        design *= u_ok[:, :, None]
        charge = np.maximum(x, 0.)
        covariance = (read_variance[:, None, None]*np.eye(ngroups)[None]
                      + np.minimum(charge[:, :, None], charge[:, None, :])/gain[:, None, None])
        covariance /= np.maximum(np.maximum(n[:, :, None], n[:, None, :]), 1.)
        # Unused groups: no correlations and a negligible weight.
        covariance = np.where(u_ok[:, :, None] & u_ok[:, None, :], covariance, 0.)
        diagonal = np.einsum('pgg->pg', covariance)
        diagonal[...] = np.where(u_ok, diagonal, 1e30)
        weighted = np.linalg.solve(covariance, np.concatenate([design, x[..., None]], -1))
        normal += np.einsum('pgi,pgj->pij', design, weighted[..., :nparameters])
        vector += np.einsum('pgi,pg->pi', design, weighted[..., nparameters])
    # Sets without groups in a pixel: fix their parameters at zero.
    stuck = np.einsum('pii->pi', normal) == 0
    np.einsum('pii->pi', normal)[...] += stuck
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        return np.linalg.solve(normal, vector[..., None])[..., 0]


def fit_self_linearity(mean_ramps, counts, read_noise=7.0, gain=1.4, min_counts=500.,
                       min_ramps=20, min_groups=5, top_factor=1.02, metadata=None,
                       pixels_per_block=20000):
    """Fit a self non-linearity correction to one or several mean ramps.

    Every pixel whose highest mean ramp ends above ``min_counts`` DN is fitted: the mean
    ramps ``x_s`` of the sets ``s`` (exposures, or visits) satisfy

        x_s(t_k) + (top/2) [a2 T2(u_sk) + a3 T3(u_sk)] = z_s + b_s t_k,
        u_sk = 2 x_s(t_k)/top - 1,

    with ``t_k`` the group number, common ``a2``, ``a3`` and ``top = top_factor`` times the
    highest mean count, and a zero point ``z_s`` and slope ``b_s`` per set; the fit is linear
    and weighted by the covariance of each mean ramp (read noise in every group, Poisson noise
    accumulated along the ramp, divided by the number of integrations). Groups with fewer than
    ``min_ramps`` integrations are not used. A pixel's correction is kept if it has at least
    ``min_groups`` groups in total and ``g`` is increasing on ``[0, top]``.

    Parameters
    ----------
    mean_ramps : numpy.ndarray or list of numpy.ndarray
        ``(group, row, column)`` mean raw ramps (DN), one per set; sets may have different
        numbers of groups.
    counts : numpy.ndarray or list of numpy.ndarray
        Numbers of integrations behind every mean, like ``mean_ramps`` (e.g., from
        `mean_ramp_sums`).
    read_noise : float or numpy.ndarray, optional
        Read noise of a group (DN); only weights the fit. Default is 7.
    gain : float or numpy.ndarray, optional
        Detector gain (electrons/DN); only weights the fit. Default is 1.4.
    min_counts : float, optional
        Minimum highest mean count (DN) of a fitted pixel. Default is 500.
    min_ramps : int, optional
        Minimum number of integrations of a group to use it. Default is 20.
    min_groups : int, optional
        Minimum number of groups of a pixel (all sets together). Default is 5.
    top_factor : float, optional
        ``top`` over the highest mean count. Default is 1.02.
    metadata : dict, optional
        Stored with the result.
    pixels_per_block : int, optional
        Pixels fitted at a time (memory). Default is 20000.

    Returns
    -------
    SelfLinearity
        The coefficients (NaN where not fitted), with the mean ramps and counts.
    """
    if isinstance(mean_ramps, np.ndarray) and mean_ramps.ndim == 3:
        mean_ramps, counts = [mean_ramps], [counts]
    mean_ramps = [np.asarray(m, dtype=float) for m in mean_ramps]
    counts = [np.asarray(c) for c in counts]
    shape = mean_ramps[0].shape[1:]
    if any(m.shape[1:] != shape or c.shape != m.shape for m, c in zip(mean_ramps, counts)):
        raise ValueError('mean_ramps and counts must be (group, row, column) arrays of the '
                         'same rows and columns, with counts like their mean ramps')
    nsets = len(mean_ramps)
    used = [(c >= min_ramps) & np.isfinite(m) for m, c in zip(mean_ramps, counts)]
    highest = np.max([np.where(u, m, -np.inf).max(axis=0) for m, u in zip(mean_ramps, used)],
                     axis=0)
    ngroups_used = np.sum([u.sum(axis=0) for u in used], axis=0)
    fit = (highest > min_counts) & (ngroups_used >= min_groups)
    rows, cols = np.nonzero(fit)
    top = top_factor*highest[rows, cols]
    read_variance = np.broadcast_to(np.asarray(read_noise, float)**2, shape)[rows, cols]
    gain_map = np.broadcast_to(np.asarray(gain, float), shape)[rows, cols]
    nparameters = 2 + 2*nsets
    solution = np.zeros((len(rows), nparameters))
    for p0 in range(0, len(rows), pixels_per_block):
        block = slice(p0, min(len(rows), p0 + pixels_per_block))
        solution[block] = _fit_block(mean_ramps, counts, used, rows[block], cols[block],
                                     top[block], read_variance[block], gain_map[block])
    a2, a3 = solution[:, 0], solution[:, 1]
    # g'/1 = 1 + 4 a2 u + a3 (12 u^2 - 3) > 0 on [-1, 1] (ends and the vertex):
    with np.errstate(divide='ignore', invalid='ignore'):
        vertex = np.clip(np.where(a3 != 0, -a2/(6*a3), -1.), -1., 1.)
    slope = np.minimum.reduce([1 - 4*a2 + 9*a3, 1 + 4*a2 + 9*a3,
                               1 + 4*a2*vertex + a3*(12*vertex**2 - 3)])
    reliable = np.isfinite(slope) & (slope > 0)
    maps = {}
    for key, value in dict(a2=a2, a3=a3, top=top).items():
        maps[key] = np.full(shape, np.nan)
        maps[key][rows, cols] = np.where(reliable, value, np.nan)
    info = dict(read_noise=float(np.mean(read_noise)), gain=float(np.mean(gain)),
                min_counts=min_counts, min_ramps=min_ramps, min_groups=min_groups,
                top_factor=top_factor, sets=nsets, fitted=int(fit.sum()),
                reliable=int(reliable.sum()))
    info.update(metadata or {})
    return SelfLinearity(maps['a2'], maps['a3'], maps['top'], info, mean_ramps, counts)
