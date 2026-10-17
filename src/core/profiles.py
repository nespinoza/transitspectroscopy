"""Spatial-profile models for optimal extraction.

Marsh (1989, PASP 101, 1032) writes the normalized spatial profile as a sum of
spatial components, each multiplied by a smooth function of the dispersion
coordinate: ``P[row, col] = sum_k Q[k, row, col] G_k(col)``. This module provides
the smooth functions ("bases") and the legacy Marsh profile fit:

- ``power_basis``: powers of the column index, as in the original C code
  (``polynomial_order`` counts terms, so the maximum degree is ``order - 1``);
- ``legendre_basis``: scaled Legendre polynomials, used by shared profiles;
- ``gp_basis``: a whitened inducing-point Gaussian process (``core.kernels``),
  used by both the legacy GP option and shared profiles.
"""

import numpy as np
from scipy.linalg import cho_factor, cho_solve, qr, solve_triangular

from . import geometry
from .kernels import KERNELS, whitened_basis


def power_basis(ncolumns, nterms):
    """``column**n`` for ``n = 0 .. nterms - 1``; shape ``(ncolumns, nterms)``."""
    return np.arange(ncolumns, dtype=float)[:, None] ** np.arange(int(nterms))


def legendre_basis(x, nterms, amplitude):
    """Legendre polynomials of degree < ``nterms`` on ``x`` (mapped to [-1, 1]), scaled."""
    return amplitude * np.polynomial.legendre.legvander(
        2 * (x - x[0]) / (x[-1] - x[0]) - 1, nterms - 1
    )


def gp_basis(x, n_inducing, kernel, amplitude, length_scale):
    """Whitened Matérn GP basis on coordinates ``x`` with evenly spaced inducing points."""
    if kernel not in KERNELS[:2]:
        raise ValueError('GP kernel must be matern32 or matern52')
    inducing = np.linspace(x[0], x[-1], n_inducing)
    return whitened_basis(x, inducing, kernel, amplitude, length_scale)


def marsh_gp_options(ncolumns, options):
    """Validate legacy GP-profile options and fill in their defaults.

    Defaults: ``kernel='matern32'``, ``length_scale=max(10, ncolumns/5)`` columns,
    ``amplitude=1`` and ``n_inducing=min(16, ncolumns)``.
    """
    options = dict(options or {})
    unknown = set(options) - {'length_scale', 'amplitude', 'n_inducing', 'kernel'}
    if unknown:
        raise ValueError(f'Unknown GP options: {sorted(unknown)}')
    length = float(options.get('length_scale', max(10.0, ncolumns / 5.0)))
    amplitude = float(options.get('amplitude', 1.0))
    number = options.get('n_inducing', min(16, ncolumns))
    kernel = options.get('kernel', 'matern32')
    if not np.isfinite(length) or length <= 0 or not np.isfinite(amplitude) or amplitude <= 0:
        raise ValueError('GP length_scale and amplitude must be finite and positive')
    if int(number) != number or not 2 <= number <= ncolumns:
        raise ValueError('GP n_inducing must be between 2 and the selected column count')
    if kernel not in ('matern32', 'matern52'):
        raise ValueError('GP kernel must be matern32 or matern52')
    return dict(kernel=kernel, amplitude=amplitude, length_scale=length, n_inducing=int(number))


def fit_marsh_profile(
    data,
    centers,
    aperture,
    ron,
    gain,
    nsigma,
    spacing,
    order,
    min_column=None,
    max_column=None,
    variance=None,
    profile_method='polynomial',
    gp_options=None,
):
    """Fit the Marsh (1989) spatial profile ``P`` of one 2-D spectrum.

    The polynomial path reproduces the original C implementation, including its
    aperture/column-range rules, empirical-variance formula and rejection order.
    The GP path (experimental) replaces the polynomials along the dispersion
    direction with a jointly fitted inducing-point GP with fixed hyperparameters;
    only complete aperture columns train it.

    Returns
    -------
    numpy.ndarray
        Normalized profile on the full ``(row, column)`` detector grid.
    """
    if profile_method == 'gp':
        data = geometry.mask_invalid_pixels(data, variance)
    data, centers, variance = geometry.validate_image(data, centers, variance)
    if not np.isfinite(nsigma) or nsigma <= 0:
        raise ValueError('profile nsigma must be finite and positive')
    if int(order) != order or order < 1:
        raise ValueError('polynomial_order is a positive number of terms')
    columns, low, high, inside, radius, count, offset = geometry.marsh_geometry(
        centers, data.shape[0], aperture, spacing, min_column, max_column
    )
    geometry.report_aperture(aperture, radius)
    a = geometry.scale_edge_pixels(data[:, columns].T, low, high)
    good = inside & (a != geometry.MASKED)
    rs = np.where(good, a, 0.0).sum(axis=1)
    if profile_method == 'gp':
        # An incomplete aperture cannot supply an unbiased normalized profile.
        # This mask controls training only; extraction uses its own pixel mask.
        complete = np.all(~inside | good, axis=1) & np.isfinite(rs) & (rs > 0)
        good &= complete[:, None]
        if not np.any(good):
            raise ValueError('no complete valid columns for GP profile training')
        rs = np.where(complete, rs, 1.0)
        a = np.where(good, a, 0.0)
    elif np.any(rs == 0):
        raise ValueError('cannot estimate a profile from a zero-sum aperture')
    e = np.where(good, a, 0.0) / rs[:, None]
    q = geometry.pixel_overlaps(centers[columns], data.shape[0], spacing, count, offset)
    if profile_method == 'polynomial':
        basis = power_basis(len(columns), order)
        regularized = False
    elif profile_method == 'gp':
        options = marsh_gp_options(len(columns), gp_options)
        basis = gp_basis(
            np.arange(len(columns), dtype=float),
            options['n_inducing'],
            options['kernel'],
            options['amplitude'],
            options['length_scale'],
        )
        regularized = True
    else:
        raise ValueError('profile_method must be polynomial or gp')
    # Shared pixels couple the spatial components in both profile models.
    design = (q[..., None] * basis[:, None, None, :]).reshape(*a.shape, -1)
    v = np.ones_like(a) if variance is None else variance[:, columns].T.copy()
    if regularized:
        v = np.where(good, v, 1.0)
    if np.any(v[good] <= 0) or not np.all(np.isfinite(v[good])):
        raise ValueError('data_variance must be finite and positive in the valid aperture')

    def empirical_variance():
        varrs = np.where(good, v, 0.0).sum(axis=1)
        return (1.0 / rs[:, None] ** 2 - 2.0 * a / rs[:, None] ** 3) * v + (
            e / rs[:, None]
        ) ** 2 * varrs[:, None]

    vare = empirical_variance()
    for iteration in range(a.size + 2):
        if np.any(vare[good] <= 0) or not np.all(np.isfinite(vare[good])):
            raise ValueError('nonpositive empirical profile variance')
        matrix = design[good]
        weight = 1.0 / vare[good]
        normal = matrix.T @ (matrix * weight[:, None])
        rhs = matrix.T @ (e[good] * weight)
        if regularized:
            normal.flat[:: normal.shape[0] + 1] += 1.0
            coefficients = cho_solve(cho_factor(normal), rhs)
        else:
            # Same normal equations and unpivoted QR as the C code (LAPACK rounding).
            orthogonal, triangular = qr(normal, mode='economic')
            coefficients = solve_triangular(triangular, orthogonal.T @ rhs)
        p = np.maximum(design @ coefficients, 0.0) * inside
        sums = p.sum(axis=1)
        if np.any(sums <= 0) or not np.all(np.isfinite(p)):
            raise ValueError('profile fit has no positive, finite support')
        p /= sums[:, None]
        if variance is None:
            v = (ron / gain) ** 2 + np.abs(rs[:, None] * p) / gain
        vare = empirical_variance()  # C updates before, rather than after, rejection.
        # C always refits once before beginning rejection; E and RS stay fixed.
        if iteration == 0:
            continue
        reject = good & ((a - rs[:, None] * p) ** 2 >= nsigma**2 * v)
        if not np.any(reject):
            break
        if regularized:
            # Never turn a rejected column into a renormalized partial profile:
            # predict it from the remaining columns, leaving extraction intact.
            # Refit after the worst column only, since a pathological pixel can
            # initially pull the GP across neighbouring good wavelengths.
            score = np.where(reject, (a - rs[:, None] * p) ** 2 / v, -np.inf)
            worst_column = np.unravel_index(np.argmax(score), score.shape)[0]
            good[worst_column] = False
            if not np.any(good):
                raise ValueError('GP rejection removed all profile-training columns')
        else:
            good[reject] = False
            if np.any(good.sum(axis=1) == 0):
                raise ValueError('profile rejection removed an entire column')
    output = np.zeros_like(data)
    output[:, columns] = p.T
    output[output > 1.0] = 0.0
    return output
