"""Gaussian-process kernels and whitened inducing-point bases.

A latent function g evaluated at points x is approximated through m inducing
points z as g(x) = K(x, z) L^-T u, where L is the Cholesky factor of K(z, z) and
u ~ N(0, I). The matrix B = K(x, z) L^-T is the "whitened basis": a GP prior on g
becomes an isotropic Gaussian prior on the coefficients u. The same construction
is used for 1-D spatial profiles along the dispersion direction and for 2-D PSFs.
"""

import numpy as np
from scipy.linalg import cholesky, solve_triangular

KERNELS = ('matern32', 'matern52', 'squared_exponential')


def covariance(a, b, kernel, amplitude, length_scale):
    """Covariance matrix between two sets of points.

    Parameters
    ----------
    a, b : numpy.ndarray
        Points. Matérn kernels take one-dimensional coordinates, shape ``(n,)``;
        the squared-exponential kernel takes points of shape ``(..., d)``.
    kernel : {'matern32', 'matern52', 'squared_exponential'}
        Covariance function.
    amplitude : float
        Standard deviation of the process.
    length_scale : float
        Correlation length, in the units of the points.

    Returns
    -------
    numpy.ndarray
        ``K[i, j] = k(a[i], b[j])``.
    """
    if kernel == 'squared_exponential':
        squared_distance = np.sum((a[..., None, :] - b) ** 2, axis=-1)
        return amplitude**2 * np.exp(-0.5 * squared_distance / length_scale**2)
    r = np.abs(a[:, None] - b[None, :]) / length_scale
    if kernel == 'matern32':
        r *= np.sqrt(3.0)
        return amplitude**2 * (1 + r) * np.exp(-r)
    if kernel == 'matern52':
        r *= np.sqrt(5.0)
        return amplitude**2 * (1 + r + r * r / 3) * np.exp(-r)
    raise ValueError(f'kernel must be one of {KERNELS}')


def inducing_cholesky(inducing, kernel, amplitude, length_scale, jitter=1e-10):
    """Lower Cholesky factor of K(z, z) plus a small relative jitter."""
    k = covariance(inducing, inducing, kernel, amplitude, length_scale)
    k = k + np.eye(len(inducing)) * amplitude**2 * jitter
    return cholesky(k, lower=True)


def whitened_basis(points, inducing, kernel, amplitude, length_scale, jitter=1e-10):
    """Whitened inducing-point basis ``B = K(x, z) L^-T`` of shape ``(n_points, n_inducing)``.

    ``points`` may have leading dimensions; for the squared-exponential kernel they
    are flattened and the basis is returned with shape ``(n_points_total, n_inducing)``.
    """
    chol = inducing_cholesky(inducing, kernel, amplitude, length_scale, jitter)
    if kernel == 'squared_exponential':
        cross = covariance(points, inducing, kernel, amplitude, length_scale)
        cross = cross.reshape(-1, len(inducing)).T
    else:
        cross = covariance(inducing, points, kernel, amplitude, length_scale)
    return solve_triangular(chol, cross, lower=True).T
