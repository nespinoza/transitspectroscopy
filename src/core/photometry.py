"""Aperture photometry and shared, pixel-integrated 2-D Gaussian-process PSF photometry.

Images are ``(y, x)`` stamps and centres are ``(x, y)`` pixel coordinates. The GP is
spatial, never temporal: independent integration fluxes are profiled out while one
positive log-PSF is fitted to several integrations; missing pixels carry no likelihood.
"""

from dataclasses import dataclass
import json
import time

import numpy as np
from scipy.linalg import solve_triangular
from scipy.optimize import least_squares

from .kernels import covariance, inducing_cholesky, whitened_basis
from .shared_profile import fit_coefficients

# The PSF model uses a squared-exponential kernel with a larger relative jitter.
_KERNEL = 'squared_exponential'
_JITTER = 1e-7


def valid_pixels(data, variance, dq=None):
    good = np.isfinite(data) & (data != -9999) & np.isfinite(variance) & (variance > 0)
    if dq is not None:
        good &= (np.asarray(dq, dtype=np.uint32) & 3) == 0  # DO_NOT_USE, SATURATED
    return good


def circular_weights(shape, center, radius, subpixels=12):
    """Fractional circular aperture, deterministic subpixel area integration."""
    if radius <= 0 or subpixels < 1:
        raise ValueError('Positive radius and subpixel count required')
    yy, xx = np.indices(shape, dtype=float)
    result = np.zeros(shape)
    offsets = (np.arange(subpixels) + 0.5) / subpixels - 0.5
    for dy in offsets:
        for dx in offsets:
            result += (xx + dx - center[0]) ** 2 + (yy + dy - center[1]) ** 2 < radius**2
    return result / subpixels**2


def background(image, variance, center, annulus=(12.0, 20.0), valid=None, exclude=None):
    """Clipped inverse-variance annular mean and its sampling variance.

    Clip only spatial sky pixels, never the source's time series. Scatter inflates
    the formal mean variance when the annulus is more variable than pixel errors.
    """
    y, x = np.indices(image.shape)
    r = np.hypot(x - center[0], y - center[1])
    good = valid_pixels(image, variance) & (r >= annulus[0]) & (r <= annulus[1])
    if valid is not None:
        good &= valid
    if exclude is not None:
        good &= ~exclude
    for _ in range(3):
        if good.sum() < 20:
            raise ValueError('Insufficient valid sky annulus pixels')
        v = image[good]
        med = np.median(v)
        scale = max(1.4826 * np.median(abs(v - med)), np.sqrt(np.median(variance[good])))
        good &= abs(image - med) < 5 * scale
    w = 1 / variance[good]
    level = np.sum(w * image[good]) / w.sum()
    inflation = max(1.0, np.sum(w * (image[good] - level) ** 2) / (good.sum() - 1))
    return float(level), float(inflation / w.sum()), int(good.sum())


def centroid(image, variance, guess, radius=5.0, max_shift=3.0):
    """Bounded Gaussian core centroid; its flux is not used as photometry."""
    y, x = np.indices(image.shape)
    good = (
        valid_pixels(image, variance)
        & (abs(x - guess[0]) <= radius)
        & (abs(y - guess[1]) <= radius)
    )
    if good.sum() < 15:
        raise ValueError('Insufficient centroid pixels')
    d, v, x, y = image[good], variance[good], x[good], y[good]
    b = np.median(d)
    amplitude = max(np.max(d) - b, np.finfo(float).tiny)
    core = np.maximum(d - b - 0.3 * amplitude, 0.0)
    start = np.array([np.sum(x * core), np.sum(y * core)]) / core.sum()
    start = np.clip(start, np.asarray(guess) - 0.9 * max_shift, np.asarray(guess) + 0.9 * max_shift)
    initial = [amplitude, *start, 1.8, 1.8, b]

    def residual(p):
        a, cx, cy, sx, sy, sky = p
        return (
            a * np.exp(-0.5 * ((x - cx) / sx) ** 2 - 0.5 * ((y - cy) / sy) ** 2) + sky - d
        ) / np.sqrt(v)

    fit = least_squares(
        residual,
        initial,
        bounds=(
            [0, guess[0] - max_shift, guess[1] - max_shift, 0.4, 0.4, -np.inf],
            [np.inf, guess[0] + max_shift, guess[1] + max_shift, radius, radius, np.inf],
        ),
        loss='soft_l1',
        f_scale=5.0,
        x_scale=[amplitude, 1, 1, 1, 1, amplitude],
        max_nfev=200,
    )
    if not fit.success or np.any(abs(fit.x[1:3] - guess) > 0.99 * max_shift):
        raise ValueError('Centroid failed or reached search boundary')
    return fit.x[1:3], fit.x[3:5]


def sky_precision(vector, variance, good, sky_variance):
    """Apply ``(diag(variance) + sky_variance * 1 1^T)^-1`` to ``vector`` on observed pixels.

    The second term is the variance of a common background estimate, which
    correlates all pixels of a stamp.
    """
    iv = np.divide(1.0, variance, out=np.zeros_like(variance, dtype=float), where=good)
    z = np.where(good, vector, 0.0) * iv
    return z - iv * (sky_variance * z.sum() / (1 + sky_variance * iv.sum()))


def optimal_flux(data, variance, profile, valid=None, sky_variance=0.0, minimum_fraction=0.8):
    good = valid_pixels(data, variance)
    if valid is not None:
        good &= valid
    observed = profile[good].sum()
    if observed < minimum_fraction:
        return np.nan, np.nan, observed, np.zeros_like(profile)
    cp = sky_precision(profile, variance, good, sky_variance)
    norm = np.sum(profile * cp)
    if norm <= 0:
        return np.nan, np.nan, observed, np.zeros_like(profile)
    weights = cp / norm
    return (
        float(np.sum(weights * np.where(good, data, 0))),
        float(np.sqrt(1 / norm)),
        float(observed),
        weights,
    )


@dataclass
class GPPSF:
    """Finite inducing-point RBF GP for the logarithm of a 2-D PSF.

    Whitened inducing coefficients have N(0,I) priors. Pixel integrals use
    quadrature before normalization over the supplied detector stamp. The
    returned flux therefore refers to that stamp, not infinite encircled energy.
    """

    inducing: np.ndarray
    coefficients: np.ndarray
    length_scale: float = 1.5
    amplitude: float = 2.0
    width: float = 2.0
    quadrature: int = 2
    diagnostics: dict = None

    def design(self, shape, center):
        y, x = np.indices(shape, dtype=float)
        offsets = (np.arange(self.quadrature) + 0.5) / self.quadrature - 0.5
        xy = np.stack(
            [
                np.stack((x + dx - center[0], y + dy - center[1]), axis=-1)
                for dy in offsets
                for dx in offsets
            ],
            axis=2,
        )
        basis = whitened_basis(
            xy, self.inducing, _KERNEL, self.amplitude, self.length_scale, jitter=_JITTER
        )
        mean = -0.5 * np.sum(xy**2, axis=-1) / self.width**2
        return basis.reshape(*mean.shape, -1), mean

    @staticmethod
    def render(coefficients, design, derivative=False):
        basis, mean = design
        z = mean + basis @ coefficients
        e = np.exp(z - z.max())
        p = e.sum(axis=2)
        total = p.sum()
        p /= total
        if not derivative:
            return p
        jac = np.sum(e[..., None] * basis, axis=2) / total
        jac -= p[..., None] * jac.sum(axis=(0, 1))
        return p, jac

    def evaluate(self, shape, center):
        return self.render(self.coefficients, self.design(shape, center))

    def evaluator(self, shape, centers, step=0.08):
        """Cache a spatial log-PSF grid; cubic interpolation only of the model.

        Science pixels are never interpolated. Validate this approximation
        against ``evaluate`` at the actual centers before using it in a run.
        """
        from scipy.ndimage import map_coordinates, spline_filter

        centers = np.asarray(centers)
        lo = -centers.max(axis=0) - 2
        hi = np.array(shape[::-1]) - centers.min(axis=0) + 2
        xs, ys = [np.arange(a, b + step, step) for a, b in zip(lo, hi)]
        xx, yy = np.meshgrid(xs, ys)
        points = np.column_stack((xx.ravel(), yy.ravel()))
        chol = inducing_cholesky(
            self.inducing, _KERNEL, self.amplitude, self.length_scale, jitter=_JITTER
        )
        alpha = solve_triangular(chol.T, self.coefficients, lower=False)
        logs = np.empty(len(points))
        for start in range(0, len(points), 2048):
            p = points[start : start + 2048]
            cross = covariance(p, self.inducing, _KERNEL, self.amplitude, self.length_scale)
            logs[start : start + len(p)] = (
                -0.5 * np.sum(p * p, axis=-1) / self.width**2 + cross @ alpha
            )
        grid = spline_filter(logs.reshape(xx.shape), order=3)
        y, x = np.indices(shape, dtype=float)
        offsets = (np.arange(self.quadrature) + 0.5) / self.quadrature - 0.5

        def evaluate(center):
            values = np.array(
                [
                    map_coordinates(
                        grid,
                        [(y + dy - center[1] - ys[0]) / step, (x + dx - center[0] - xs[0]) / step],
                        order=3,
                        mode='mirror',
                        prefilter=False,
                    )
                    for dy in offsets
                    for dx in offsets
                ]
            )
            p = np.exp(values - values.max()).sum(axis=0)
            return p / p.sum()

        return evaluate

    def save(self, path):
        np.savez_compressed(
            path,
            inducing=self.inducing,
            coefficients=self.coefficients,
            length_scale=self.length_scale,
            amplitude=self.amplitude,
            width=self.width,
            quadrature=self.quadrature,
            diagnostics=json.dumps(self.diagnostics or {}),
        )

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as a:
            return cls(
                a['inducing'],
                a['coefficients'],
                float(a['length_scale']),
                float(a['amplitude']),
                float(a['width']),
                int(a['quadrature']),
                json.loads(str(a['diagnostics'])),
            )


def fit_gp_psf(
    images,
    variances,
    centers,
    *,
    sky_variances=None,
    n_inducing=11,
    extent=10.0,
    length_scale=1.5,
    amplitude=2.0,
    width=2.0,
    quadrature=2,
    maxiter=400,
):
    """Joint native-pixel likelihood with an independent flux per training image."""
    started = time.perf_counter()
    images, variances = np.asarray(images), np.asarray(variances)
    if images.ndim != 3 or images.shape != variances.shape or len(centers) != len(images):
        raise ValueError('Expected matching integration,y,x arrays and centers')
    if (
        not np.all(np.isfinite([extent, length_scale, amplitude, width]))
        or min(extent, length_scale, amplitude, width) <= 0
        or n_inducing < 3
        or quadrature < 1
        or int(n_inducing) != n_inducing
        or int(quadrature) != quadrature
    ):
        raise ValueError('Invalid GP parameters')
    if images.size * quadrature**2 * n_inducing**2 * 8 > 512 * 1024**2:
        raise ValueError('GP design exceeds 512 MiB; reduce training images/grid/quadrature')
    z = np.linspace(-extent, extent, n_inducing)
    xx, yy = np.meshgrid(z, z)
    model = GPPSF(
        np.column_stack((xx.ravel(), yy.ravel())),
        np.zeros(n_inducing**2),
        length_scale,
        amplitude,
        width,
        quadrature,
    )
    sky_variances = np.zeros(len(images)) if sky_variances is None else np.asarray(sky_variances)
    if (
        sky_variances.shape != (len(images),)
        or not np.isfinite(sky_variances).all()
        or np.any(sky_variances < 0)
    ):
        raise ValueError('Sky variances must be finite, nonnegative, one per integration')
    designs = [model.design(images.shape[1:], c) for c in centers]
    masks = valid_pixels(images, variances)
    if np.any(masks.sum(axis=(1, 2)) < n_inducing):
        raise ValueError('Insufficient PSF training data')

    class Executor:
        def evaluate(self, a, basis=None, curvature=False):
            value, gradient = 0.0, np.zeros_like(a)
            hessian = np.zeros((len(a), len(a))) if curvature else None
            for data, var, good, sv, design in zip(
                images, variances, masks, sky_variances, designs
            ):
                p, jac = model.render(a, design, True)
                f, e, _, _ = optimal_flux(data, var, p, good, sv, minimum_fraction=0.0)
                if not np.isfinite(f) or f <= 0:
                    if curvature:
                        raise ValueError('Nonpositive training source flux at current profile')
                    # A line-search trial may put the PSF on sky/negative pixels.
                    # Reject that trial, allowing the existing optimizer to shorten its step.
                    return np.inf, np.zeros_like(a), None, []
                residual = np.where(good, f * p - data, 0.0)
                cr = sky_precision(residual, var, good, sv)
                value += 0.5 * np.sum(residual * cr)
                gradient += f * np.einsum('yxk,yx->k', jac, cr)
                if curvature:
                    iv = np.divide(1.0, var, out=np.zeros_like(var, dtype=float), where=good)
                    j = jac.reshape(-1, len(a)) * f
                    cj = j * iv.ravel()[:, None]
                    cj -= iv.ravel()[:, None] * (sv * cj.sum(axis=0) / (1 + sv * iv.sum()))
                    cross = j.T @ sky_precision(p, var, good, sv).ravel()
                    hessian += j.T @ cj - np.outer(cross, cross) * e**2
            return value, gradient, hessian, []

    # Reuse the shared-profile optimizer and its Gaussian coefficient prior.
    fit = fit_coefficients(
        Executor(), None, model.coefficients, maxiter, {'coefficient_objective_seconds': 0.0}
    )
    if not fit.success:
        raise RuntimeError('2-D GP PSF fit failed: ' + fit.message)
    model.coefficients = fit.x
    model.diagnostics = dict(
        success=True,
        iterations=int(fit.nit),
        objective=float(fit.fun),
        seconds=time.perf_counter() - started,
        training_integrations=len(images),
        hyperparameters_optimized=False,
        likelihood='native pixels, independent profiled flux, correlated annular sky error',
    )
    return model
