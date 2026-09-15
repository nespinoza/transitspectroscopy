"""Joint, reusable intrinsic profiles on native detector pixels.

This opt-in model is distinct from the legacy Marsh coefficient fit. Polynomial
or inducing-point GP functions describe *log* spatial intensities relative to a
fixed Gaussian mean. A spatial contrast removes their common normalization mode.
The likelihood uses supplied Gaussian pixel variances and a fixed mask. Fluxes
are profiled independently for each integration/column. Hyperparameter fitting
uses a Laplace approximation over shared coefficients, CONDITIONAL on those
fitted fluxes; it is not a fully marginalized spectrum likelihood.
"""
from collections import OrderedDict
from dataclasses import dataclass, field
import hashlib
import json
import time
import warnings

import numpy as np
from scipy.linalg import cho_factor, cho_solve, helmert
from scipy.optimize import minimize, OptimizeResult

from . import _optimal_extraction as _legacy

__all__ = ['SharedProfile', 'fit_shared_profile']
_VERSION = 1


def _integer(value, name, minimum=1):
    if not np.isscalar(value) or not np.isfinite(value) or int(value) != value or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')
    return int(value)


def _coordinates(values, count):
    x = np.arange(count, dtype=float) if values is None else np.asarray(values, dtype=float)
    if x.shape != (count,) or not np.all(np.isfinite(x)) or np.any(np.diff(x) <= 0):
        raise ValueError('dispersion_coordinates must be finite, increasing, one per column')
    return x


def _component_count(radius, spacing):
    # Whole boundary pixels can extend one pixel beyond the nominal radius.
    # Model that illumination too; truncating the intrinsic basis there biases
    # narrow-aperture fits as the trace moves across pixel boundaries.
    return 2*int(radius/spacing+.5)+1+2*int(np.ceil(1./spacing))


def _basis(x, method, order, options):
    if method == 'polynomial':
        return options['amplitude'] * np.polynomial.legendre.legvander(
            2 * (x-x[0])/(x[-1]-x[0])-1, order-1)
    # The existing GP helper assumes consecutive pixels; use physical supplied
    # coordinates here, preserving the same Matern kernel and whitening.
    from scipy.linalg import cholesky, solve_triangular
    z = np.linspace(x[0], x[-1], options['n_inducing'])
    def cov(a, b):
        r = np.abs(a[:, None]-b[None, :])/options['length_scale']
        if options['kernel'] == 'matern32':
            r *= np.sqrt(3.)
            return options['amplitude']**2*(1+r)*np.exp(-r)
        r *= np.sqrt(5.)
        return options['amplitude']**2*(1+r+r*r/3)*np.exp(-r)
    k = cov(z, z) + np.eye(len(z))*options['amplitude']**2*1e-10
    return solve_triangular(cholesky(k, lower=True), cov(z, x), lower=True).T


def _geometry(centers, nrows, radius, spacing):
    centers = np.asarray(centers, dtype=float)
    if centers.ndim != 1 or not np.all(np.isfinite(centers)):
        raise ValueError('each trace must be a finite one-dimensional array')
    # Fix support across integrations. Never silently shrink a trained model.
    count = _component_count(radius,spacing)
    extent = spacing*int(radius/spacing+.5)
    low = np.trunc(centers-extent+.5).astype(int)
    high = np.trunc(centers+extent+.5).astype(int)
    if np.any(centers-extent+.5 < 0) or np.any(high >= nrows):
        raise ValueError('shared profile aperture extends outside detector; select a smaller aperture')
    rows = low[:, None]+np.arange(int(np.max(high-low))+1)
    inside = rows <= high[:, None]
    rows = np.minimum(rows, nrows-1)
    return rows, inside, count


def _q(centers, rows, inside, spacing, count):
    offset = -spacing*(1+(count-1)/2)
    # All row windows are contiguous, so the existing pixel-overlap calculation
    # can run on cropped coordinates without allocating full-height images.
    return _legacy.overlaps(centers-rows[:, 0], rows.shape[1], spacing, count, offset)*inside[..., None]


def _pixel_profile(q, z):
    exponential = np.exp(z-z.max(axis=1, keepdims=True))
    weighted = q*exponential[:, None, :]
    total = weighted.sum(axis=(1, 2))
    if np.any(total <= 0) or not np.all(np.isfinite(total)):
        raise FloatingPointError('profile has no finite detector support')
    weighted /= total[:, None, None]
    return weighted.sum(axis=2), weighted


def _linear_gaussian_evidence(y, design, variance):
    """Reference primitive: w~N(0,I); no nonlinear normalization or flux fit."""
    x = design/np.sqrt(variance)[:, None]
    yw = y/np.sqrt(variance)
    normal = np.eye(x.shape[1])+x.T@x
    factor = cho_factor(normal)
    mean = cho_solve(factor, x.T@yw)
    residual = yw-x@mean
    nll = .5*(residual@residual+mean@mean+np.log(variance).sum()
              +2*np.log(np.diag(factor[0])).sum()+len(y)*np.log(2*np.pi))
    return nll, mean, cho_solve(factor, np.eye(len(mean)))


def _helmert_diagonal(values):
    """C.T @ diag(values[j]) @ C in O(J K^2), for C=helmert(K).T.

    The analytic prefix sums avoid a cubic dense spatial transform per column.
    Values can be signed (the nonlinear Hessian correction is not a covariance).
    """
    count = values.shape[1]-1
    a = np.arange(count)
    scale = np.sqrt((a+1.)*(a+2.))
    lower = np.minimum(a[:,None],a[None,:])
    prefix = np.cumsum(values,axis=1)[:,:count]
    off = prefix-(a+1.)*values[:,1:]
    result = off[:,lower]/(scale[:,None]*scale[None,:])
    result[:,a,a] = (prefix+(a+1.)**2*values[:,1:])/scale**2
    return result


class _BatchWorker:
    """Resident cropped data; bounded Q cache; no worker-local profile fit."""
    def __init__(self, frames, spacing, count, contrast, mean, column_batch_size, cache_mb):
        self.frames = frames
        self.spacing, self.count = spacing, count
        self.contrast, self.mean = contrast, mean
        self.column_batch_size = column_batch_size
        self.cache_limit = int(cache_mb*1e6)
        self.cache, self.cache_bytes = OrderedDict(), 0

    def geometry(self, index, frame):
        if index in self.cache:
            self.cache.move_to_end(index)
            return self.cache[index]
        q = _q(frame['centers'], frame['rows'], frame['inside'], self.spacing, self.count)
        if self.cache_bytes+q.nbytes <= self.cache_limit:
            # Frames are revisited in a fixed sweep. Evicting on every miss would
            # make an oversized sweep thrash an LRU cache with zero reuse.
            self.cache[index] = q
            self.cache_bytes += q.nbytes
        return q

    def evaluate(self, coefficients, basis, curvature=False, details=False):
        c = self.contrast
        z = basis@coefficients.T@c.T+self.mean
        value = 0.
        gradient = np.zeros_like(coefficients)
        hessian = np.zeros((coefficients.size, coefficients.size)) if curvature else None
        reports = []
        for index, frame in enumerate(self.frames):
            q = self.geometry(index, frame)
            chi2, npixels = 0., 0
            fluxes = []
            for start in range(0, len(basis), self.column_batch_size):
                sl = slice(start, start+self.column_batch_size)
                b = basis[sl]
                p, weighted = _pixel_profile(q[sl], z[sl])
                y, iv = frame['data'][sl], frame['inverse'][sl]
                den = np.sum(p*p*iv, axis=1)
                flux = np.divide(np.sum(p*y*iv, axis=1), den,
                                 out=np.zeros(len(p)), where=den > 0)
                residual = flux[:, None]*p-y
                value += .5*np.sum(residual**2*iv)
                if details:
                    chi2 += float(np.sum(residual**2*iv))
                    npixels += int(np.count_nonzero(iv))
                    fluxes.extend(flux)
                gp = flux[:, None]*residual*iv
                s = weighted.sum(axis=1)
                w = np.einsum('jrk,jr->jk', weighted, gp)
                cp = np.sum(p*gp, axis=1)
                gz = w-cp[:, None]*s
                gradient += (gz@c).T@b
                if curvature:
                    # Exact coefficient Hessian CONDITIONAL on the fitted flux.
                    sc, wc = s@c, w@c
                    jac = weighted@c-p[..., None]*sc[:, None, :]
                    hc = np.einsum('jrk,jrl,jr->jkl', jac, jac, flux[:, None]**2*iv,
                                   optimize=True)
                    if curvature == 'gauss_newton':
                        # Eliminate the independent flux increments from the
                        # Gauss-Newton system (variable projection). The prior
                        # makes the resulting coefficient system positive definite.
                        cross = np.einsum('jrk,jr->jk',jac,flux[:,None]*p*iv)
                        hc -= np.divide(cross[:,:,None]*cross[:,None,:],den[:,None,None],
                                        out=np.zeros_like(hc),where=den[:,None,None]>0)
                    else:
                        hc -= wc[:, :, None]*sc[:, None, :]+sc[:, :, None]*wc[:, None, :]
                        hc += 2*cp[:, None, None]*sc[:, :, None]*sc[:, None, :]
                        hc += _helmert_diagonal(gz)
                    hessian += np.einsum('jab,jm,jn->ambn', hc, b, b,
                                         optimize=True).reshape(hessian.shape)
            value += frame['log_normalization']
            if details:
                reports.append(dict(index=frame['index'], chi2=chi2, valid_pixels=npixels,
                                    flux_min=float(np.min(fluxes)), flux_max=float(np.max(fluxes)),
                                    flux_median=float(np.median(fluxes))))
        return value, gradient, hessian, reports


class _Executor:
    def __init__(self, frames, config, execution, workers, batch_size):
        self.execution, self.owns_ray = execution, False
        self.workers = []
        self.last_basis, self.basis_ref = None, None
        if execution == 'serial':
            # One cache spans frames; inputs were already streamed/cropped.
            self.workers = [_BatchWorker(frames, **config)]
        elif execution == 'ray':
            try:
                import ray
            except ImportError as exc:
                raise ImportError('execution="ray" requires the optional ray package') from exc
            self.ray = ray
            if not ray.is_initialized():
                ray.init(num_cpus=workers, include_dashboard=False)
                self.owns_ray = True
            if min(workers,len(frames)) > ray.cluster_resources().get('CPU',0):
                if self.owns_ray:
                    ray.shutdown()
                raise ValueError('n_workers exceeds Ray cluster CPU capacity')
            actor = ray.remote(num_cpus=1, runtime_env={'env_vars': {
                'OPENBLAS_NUM_THREADS':'1', 'OMP_NUM_THREADS':'1', 'MKL_NUM_THREADS':'1'}})(_BatchWorker)
            # Contiguous batches assigned round-robin; each actor retains its
            # cropped measurements throughout all optimizer evaluations.
            shards = [[] for _ in range(min(workers, len(frames)))]
            effective_batch = min(batch_size, (len(frames)+len(shards)-1)//len(shards))
            for start in range(0, len(frames), effective_batch):
                shards[(start//effective_batch) % len(shards)].extend(frames[start:start+effective_batch])
            self.workers = [actor.remote(shard, **config) for shard in shards if shard]
        else:
            raise ValueError('execution must be serial or ray')

    def evaluate(self, coefficients, basis, curvature=False, details=False):
        if self.execution == 'serial':
            return self.workers[0].evaluate(coefficients, basis, curvature, details)
        if basis is not self.last_basis:
            self.basis_ref = self.ray.put(basis)
            self.last_basis = basis
        pending = [w.evaluate.remote(coefficients, self.basis_ref, curvature, details) for w in self.workers]
        value, gradient, hessian, reports = 0., np.zeros_like(coefficients), None, []
        if curvature:
            hessian = np.zeros((coefficients.size, coefficients.size))
        # Deterministic bounded reduction. Do not ray.get the list of matrices.
        for ref in pending:
            v, g, h, report = self.ray.get(ref)
            value += v
            gradient += g
            if curvature:
                hessian += h
            reports.extend(report)
        return value, gradient, hessian, sorted(reports, key=lambda r: r['index'])

    def close(self):
        if self.execution == 'ray':
            for worker in self.workers:
                self.ray.kill(worker)
            if self.owns_ray:
                self.ray.shutdown()


def _fit_coefficients(executor, basis, initial, maxiter, timing):
    """Damped variable-projection Gauss-Newton with an explicit Gaussian prior.

    Scaled-gradient convergence avoids raw coefficient conditioning dominating
    stopping criteria. Line search always checks the actual nonlinear objective.
    """
    shape = initial.shape
    flat = initial.ravel().copy()
    calls = 0
    def objective(values, curvature=False):
        nonlocal calls
        tick = time.perf_counter()
        value, gradient, hessian, _ = executor.evaluate(values.reshape(shape),basis,curvature)
        timing['coefficient_objective_seconds'] += time.perf_counter()-tick
        calls += 1
        if curvature:
            hessian.flat[::len(flat)+1] += 1.
        return value+.5*values@values, gradient.ravel()+values, hessian
    success, message = False, 'maximum coefficient iterations reached'
    for iteration in range(maxiter):
        value, gradient, hessian = objective(flat,'gauss_newton')
        scale = np.sqrt(np.maximum(np.diag(hessian),1.))
        if np.max(abs(gradient)/scale) < 1e-5:
            success, message = True, 'scaled coefficient gradient converged'
            break
        direction = -cho_solve(cho_factor((hessian+hessian.T)/2),gradient)
        slope = gradient@direction
        # Preserve tight inner solves for hyperparameter finite differences.
        # Large likelihood sums need a rounding floor for their reductions:
        # at most 1e-7 in squared local posterior step (<0.00032 in that metric).
        # A uniformly looser tolerance makes small-data evidence gradients noisy.
        decrement_tolerance = max(1e-8, min(1e-7, 32*np.finfo(float).eps*abs(value)))
        if -slope < decrement_tolerance:
            success = True
            message = f'coefficient Newton decrement converged (tolerance={decrement_tolerance:.6g})'
            break
        step = 1.
        for _ in range(25):
            candidate = flat+step*direction
            new_value, new_gradient, _ = objective(candidate)
            if new_value <= value+1e-4*step*slope:
                flat = candidate
                break
            step *= .5
        else:
            # At machine precision an Armijo decrease can become unresolvable.
            if np.max(abs(direction)*scale) < 1e-4:
                success, message = True, 'coefficient step at numerical precision'
            else:
                message = ('coefficient line search failed at iteration '
                           f'{iteration+1}: scaled_gradient={np.max(abs(gradient)/scale):.6g}, '
                           f'Newton_decrement={-slope:.6g}, '
                           f'scaled_step={np.max(abs(direction)*scale):.6g}')
            break
    value, gradient, _ = objective(flat)
    return OptimizeResult(x=flat, fun=value, jac=gradient, success=success,
                          message=message, nit=iteration+1, nfev=calls)


@dataclass(frozen=True)
class SharedProfile:
    """Frozen fitted log-intensity model; evaluate/extract never retrain it.

    Arrays returned by evaluate use native, whole detector pixels within the
    aperture. Use extract() or getOptimalSpectrum(P=model) for consistent native
    pixel weighting. Passing evaluate()'s array into legacy extraction retains
    that API's historical fractional-edge treatment instead.
    """
    coefficients: np.ndarray
    basis: np.ndarray
    contrast: np.ndarray
    mean: np.ndarray
    coordinates: np.ndarray
    detector_rows: int
    aperture_radius: int
    spacing: float
    profile_method: str
    options: dict
    diagnostics: dict
    covariance: object = None
    _log_intensity: np.ndarray = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        for name in ('coefficients', 'basis', 'contrast', 'mean', 'coordinates', 'covariance'):
            array = getattr(self, name)
            if array is not None:
                array = np.array(array, dtype=float, copy=True)
                if not np.all(np.isfinite(array)):
                    raise ValueError(f'nonfinite {name} in shared profile')
                array.flags.writeable = False
                object.__setattr__(self, name, array)
        count = _component_count(self.aperture_radius,self.spacing)
        if (self.coefficients.ndim != 2 or self.basis.ndim != 2 or
                self.coefficients.shape != (count-1, self.basis.shape[1]) or
                self.basis.shape[0] != len(self.coordinates) or
                self.contrast.shape != (count,count-1) or self.mean.shape != (count,)):
            raise ValueError('inconsistent shared-profile array shapes')
        if self.covariance is not None and self.covariance.shape != (self.coefficients.size,)*2:
            raise ValueError('inconsistent coefficient covariance shape')
        z = self.basis@self.coefficients.T@self.contrast.T+self.mean
        z.flags.writeable = False
        object.__setattr__(self, '_log_intensity', z)

    def _check_trace(self, trace, coordinates=None):
        trace = np.asarray(trace, dtype=float)
        if trace.shape != self.coordinates.shape:
            raise ValueError('trace must match the trained dispersion grid; no extrapolation')
        if coordinates is not None and not np.array_equal(coordinates, self.coordinates):
            raise ValueError('dispersion coordinates differ from the trained grid')
        return trace

    def _cropped(self, trace, coordinates=None):
        trace = self._check_trace(trace, coordinates)
        rows, inside, count = _geometry(trace, self.detector_rows, self.aperture_radius, self.spacing)
        q = _q(trace, rows, inside, self.spacing, count)
        p, _ = _pixel_profile(q, self._log_intensity)
        return rows, inside, p

    def evaluate(self, trace, *, dispersion_coordinates=None):
        """Render a normalized P_t on the original detector grid, without fitting."""
        rows, inside, p = self._cropped(trace, dispersion_coordinates)
        output = np.zeros((self.detector_rows, len(self.coordinates)))
        columns = np.broadcast_to(np.arange(len(p))[:, None], rows.shape)
        output[rows[inside], columns[inside]] = p[inside]
        return output

    def extract(self, data, trace, data_variance, *, mask=None, background=None,
                nsigma=0., return_P=False, dispersion_coordinates=None):
        """Native-pixel weighted extraction; row 2 is inverse conditional variance.

        mask=True excludes a pixel. Optional rejection removes the single largest
        standardized residual in each column per pass. Invalid columns return
        NaN flux and zero inverse variance. Profile uncertainty is not included.
        """
        rows, inside, p = self._cropped(trace, dispersion_coordinates)
        shape = (self.detector_rows, len(self.coordinates))
        data, variance = np.asarray(data, dtype=float), np.asarray(data_variance, dtype=float)
        if data.shape != shape or variance.shape != shape:
            raise ValueError('data and data_variance must match the trained detector grid')
        if not np.isfinite(nsigma) or nsigma < 0:
            raise ValueError('nsigma must be nonnegative')
        columns = np.arange(shape[1])[:, None]
        y, v = data[rows, columns].copy(), variance[rows, columns]
        good = inside & np.isfinite(y) & (y != -9999) & np.isfinite(v) & (v > 0)
        if mask is not None:
            if np.shape(mask) != shape:
                raise ValueError('mask must match data.shape')
            good &= ~np.asarray(mask, dtype=bool)[rows, columns]
        if background is not None:
            if np.shape(background) != shape:
                raise ValueError('background must match data.shape')
            b = np.asarray(background)[rows, columns]
            good &= np.isfinite(b)
            y -= b
        y = np.where(good, y, 0.)
        for _ in range(rows.shape[1]+1):
            iv = np.divide(1., v, out=np.zeros_like(v), where=good)
            den = np.sum(p*p*iv, axis=1)
            flux = np.divide(np.sum(p*y*iv, axis=1), den,
                             out=np.full(len(p), np.nan), where=den > 0)
            if nsigma == 0:
                break
            score = np.where(good, np.abs(y-flux[:, None]*p)*np.sqrt(iv), -np.inf)
            worst = score.argmax(axis=1)
            reject = score[np.arange(len(p)), worst] > nsigma
            if not np.any(reject):
                break
            good[np.flatnonzero(reject), worst[reject]] = False
        output = np.vstack((np.arange(len(p)), flux, den))
        return (output, self.evaluate(trace)) if return_P else output

    def fingerprint(self):
        digest = hashlib.sha256()
        for array in (self.coefficients, self.basis, self.contrast, self.mean, self.coordinates):
            digest.update(np.ascontiguousarray(array).tobytes())
        digest.update(json.dumps(dict(version=_VERSION, rows=self.detector_rows,
                                      radius=self.aperture_radius, spacing=self.spacing,
                                      method=self.profile_method, options=self.options),
                                 sort_keys=True, allow_nan=False).encode())
        return digest.hexdigest()

    def diagnose(self, data, trace, data_variance, *, mask=None, background=None):
        """Held-out residual diagnostics with the profile fixed and flux re-extracted.

        chi2_per_column / degrees_of_freedom is conditional on the trained model;
        structured excess across time/columns can reveal profile instability.
        """
        result, profile = self.extract(data,trace,data_variance,mask=mask,
                                       background=background,return_P=True)
        data, var = np.asarray(data), np.asarray(data_variance)
        good = (profile > 0) & np.isfinite(data) & (data != -9999) & np.isfinite(var) & (var > 0)
        if mask is not None:
            good &= ~np.asarray(mask,dtype=bool)
        if background is not None:
            good &= np.isfinite(background)
            data = data-background
        residual = np.where(good, data-profile*result[1], 0.)
        chi2 = np.sum(np.divide(residual**2,var,out=np.zeros_like(profile),where=good),axis=0)
        dof = np.maximum(good.sum(axis=0)-1,0)
        return dict(spectrum=result, chi2_per_column=chi2, degrees_of_freedom=dof,
                    reduced_chi2=float(chi2.sum()/dof.sum()) if dof.sum() else None)

    def save(self, path):
        """Save portable arrays/JSON without pickle or an executable payload."""
        metadata = dict(version=_VERSION, detector_rows=self.detector_rows,
                        aperture_radius=self.aperture_radius, spacing=self.spacing,
                        profile_method=self.profile_method, options=self.options,
                        diagnostics=self.diagnostics, fingerprint=self.fingerprint())
        with open(path, 'wb') as handle:
            np.savez_compressed(handle, coefficients=self.coefficients, basis=self.basis,
                                contrast=self.contrast, mean=self.mean, coordinates=self.coordinates,
                                covariance=np.empty((0, 0)) if self.covariance is None else self.covariance,
                                metadata=json.dumps(metadata, allow_nan=False))

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as arrays:
            meta = json.loads(str(arrays['metadata']))
            if meta.pop('version') != _VERSION:
                raise ValueError('unsupported shared-profile version')
            expected = meta.pop('fingerprint')
            covariance = arrays['covariance']
            result = cls(**meta, **{name: arrays[name] for name in
                         ('coefficients', 'basis', 'contrast', 'mean', 'coordinates')},
                         covariance=None if covariance.size == 0 else covariance)
        if result.fingerprint() != expected:
            raise ValueError('shared-profile fingerprint mismatch')
        return result


def fit_shared_profile(data, data_variance, centroids, *, aperture_radius=7, spacing=1.,
                       profile_method='gp', polynomial_order=3, gp_options=None,
                       polynomial_amplitude=1., optimize_hyperparameters=True,
                       training_indices=None, mask=None, background=None,
                       dispersion_coordinates=None, integration_ids=None,
                       batch_size=8, execution='serial', n_workers=1,
                       maxiter=500, hyper_maxiter=30, n_starts=2,
                       initial_width=1.5, column_batch_size=32,
                       geometry_cache_mb=64., memory_limit_mb=1024.,
                       store_covariance=True, hyperparameter_bounds=None):
    """Train one intrinsic profile from independent integrations and known variances.

    data/variance can be cubes, memory maps, or indexable sequences of 2-D arrays.
    Only selected integrations are read and only aperture pixels are retained.
    centroids is (integration,column), or one (column,) trace shared by all.
    mask=True excludes pixels; nonfinite data/variance, nonpositive variances and
    -9999 are also excluded. The mask is frozen throughout fitting. Supply a
    background estimate explicitly; its uncertainty must already be in variance.

    GP amplitude is in log intrinsic-intensity units. Polynomial coefficients use
    a scaled Legendre basis with an isotropic Gaussian prior. These are NEW models,
    not the legacy GP amplitude or unregularized Marsh polynomials. Bounds are a
    dict of positive (low, high) pairs for amplitude and (GP only) length_scale.

    All integrations must be conditionally independent: accumulated groups from
    the same ramp are not supported. Flux errors remain conditional on the fitted
    profile; covariance stores only the local coefficient covariance conditional
    on training fluxes and fitted hyperparameters. Failure to converge raises.
    """
    started = time.perf_counter()
    radius = _integer(aperture_radius, 'aperture_radius')
    order = _integer(polynomial_order, 'polynomial_order')
    batch_size = _integer(batch_size,'batch_size')
    n_workers = _integer(n_workers,'n_workers')
    maxiter = _integer(maxiter,'maxiter')
    hyper_maxiter = _integer(hyper_maxiter,'hyper_maxiter')
    n_starts = _integer(n_starts,'n_starts')
    column_batch_size = _integer(column_batch_size,'column_batch_size')
    if profile_method not in ('gp', 'polynomial') or execution not in ('serial','ray'):
        raise ValueError('invalid profile_method or execution')
    for value, name in [(spacing,'spacing'), (initial_width,'initial_width'),
                        (memory_limit_mb,'memory_limit_mb')]:
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f'{name} must be finite and positive')
    if not np.isfinite(geometry_cache_mb) or geometry_cache_mb < 0:
        raise ValueError('geometry_cache_mb must be finite and nonnegative')
    if len(data) == 0 or len(data_variance) != len(data):
        raise ValueError('data and variance must contain the same nonzero number of integrations')
    indices = np.arange(len(data)) if training_indices is None else np.asarray(training_indices)
    if indices.ndim != 1 or len(indices) == 0 or not np.issubdtype(indices.dtype, np.integer):
        raise ValueError('training_indices must be a nonempty integer sequence')
    if np.any(indices < 0) or np.any(indices >= len(data)) or len(np.unique(indices)) != len(indices):
        raise ValueError('training_indices must be unique and in range')
    shape = np.shape(data[int(indices[0])])
    if len(shape) != 2 or shape[1] < 2:
        raise ValueError('each integration must be a 2-D image with at least two columns')
    nrows, ncolumns = shape
    x = _coordinates(dispersion_coordinates, ncolumns)
    traces = np.asarray(centroids, dtype=float)
    if traces.shape == (ncolumns,):
        traces = np.broadcast_to(traces, (len(data), ncolumns))
    if traces.shape != (len(data), ncolumns):
        raise ValueError('centroids must have shape (integration,column) or (column,)')
    if integration_ids is not None and len(integration_ids) != len(data):
        raise ValueError('integration_ids must contain one ID per integration')
    options = dict(gp_options or {})
    if profile_method == 'polynomial':
        if gp_options is not None:
            raise ValueError('gp_options requires profile_method="gp"')
        options = {'amplitude': float(polynomial_amplitude)}
    else:
        if set(options)-{'amplitude','length_scale','kernel','n_inducing'}:
            raise ValueError('unknown GP option')
        options = dict(amplitude=float(options.get('amplitude', 1.)),
                       length_scale=float(options.get('length_scale', max(10., (x[-1]-x[0])/5))),
                       kernel=options.get('kernel','matern32'),
                       n_inducing=_integer(options.get('n_inducing',min(16,ncolumns)), 'n_inducing',2))
        if options['kernel'] not in ('matern32','matern52') or options['n_inducing'] > ncolumns:
            raise ValueError('invalid kernel or n_inducing')
    if any(not np.isfinite(v) or v <= 0 for k,v in options.items() if k in ('amplitude','length_scale')):
        raise ValueError('amplitude and length_scale must be finite and positive')
    count = _component_count(radius,spacing)
    if count < 3:
        raise ValueError('spacing must provide at least three spatial components')
    terms = order if profile_method == 'polynomial' else options['n_inducing']
    dimension = (count-1)*terms
    # Curvature, factorization, covariance and worker reductions need multiple
    # d-by-d arrays. Fail before a large allocation; never silently coarsen Q.
    matrix_mb = 8*dimension**2/1e6
    if matrix_mb*(5+(n_workers if execution == 'ray' else 1)) > memory_limit_mb:
        raise ValueError('coefficient curvature exceeds memory_limit_mb; reduce spacing resolution, '
                         'basis size, workers, or explicitly increase the budget')
    estimated_workers = min(n_workers,len(indices)) if execution == 'ray' else 1
    max_window = int(2*spacing*int(radius/spacing+.5))+2
    projected_crops = len(indices)*ncolumns*(25*max_window+8)/1e6
    projected_q = 8*ncolumns*max_window*count/1e6
    projected_total = (projected_crops*(2 if execution == 'ray' else 1)
                       +matrix_mb*(5+estimated_workers)
                       +estimated_workers*(geometry_cache_mb+projected_q))
    if projected_total > memory_limit_mb:
        raise ValueError('selected aperture crops and curvature exceed memory_limit_mb; '
                         'select fewer integrations or explicitly increase the budget')
    contrast = helmert(count).T
    offsets = spacing*(np.arange(count)-(count-1)/2)
    mean = -.5*(offsets/initial_width)**2
    frames, digest = [], hashlib.sha256()
    for index in indices:
        index = int(index)
        y, v = np.asarray(data[index], dtype=float), np.asarray(data_variance[index], dtype=float)
        if y.shape != shape or v.shape != shape:
            raise ValueError('all training images and variances must have matching shapes')
        rows, inside, _ = _geometry(traces[index], nrows, radius, spacing)
        columns = np.arange(ncolumns)[:, None]
        a, var = y[rows, columns].copy(), v[rows, columns].copy()
        good = inside & np.isfinite(a) & (a != -9999) & np.isfinite(var) & (var > 0)
        if mask is not None:
            if np.shape(mask[index]) != shape:
                raise ValueError('mask images must match data')
            good &= ~np.asarray(mask[index], dtype=bool)[rows, columns]
        if background is not None:
            if np.shape(background[index]) != shape:
                raise ValueError('background images must match data')
            b = np.asarray(background[index])[rows, columns]
            good &= np.isfinite(b)
            a -= b
        # One pixel and one free flux contain no profile information.
        good &= (good.sum(axis=1) >= 2)[:, None]
        iv = np.divide(1., var, out=np.zeros_like(var), where=good)
        a = np.where(good, a, 0.)
        frame = dict(index=index, centers=traces[index].copy(), rows=rows, inside=inside,
                     data=a, inverse=iv,
                     log_normalization=float(.5*np.log(2*np.pi*var[good]).sum()))
        for array in (frame['centers'], a, iv):
            digest.update(np.ascontiguousarray(array).tobytes())
        frames.append(frame)
    if not np.all(np.any([f['inverse'].sum(axis=1)>0 for f in frames], axis=0)):
        raise ValueError('every column needs at least one training integration with two valid pixels')
    cropped_bytes = sum(sum(a.nbytes for a in f.values() if isinstance(a, np.ndarray)) for f in frames)
    active_workers = min(n_workers,len(frames)) if execution == 'ray' else 1
    largest_q_mb = max(f['data'].size for f in frames)*count*8/1e6
    estimated_memory = (cropped_bytes/1e6*(2 if execution == 'ray' else 1)
                        +matrix_mb*(5+active_workers)
                        +active_workers*(geometry_cache_mb+largest_q_mb))
    if estimated_memory > memory_limit_mb:
        raise ValueError('cropped training data and curvature exceed memory_limit_mb; select fewer integrations')
    timing = dict(preparation_seconds=time.perf_counter()-started,
                  coefficient_objective_seconds=0., curvature_seconds=0.,
                  coefficient_solver_seconds=0.)
    setup_start = time.perf_counter()
    executor = _Executor(frames, dict(spacing=spacing, count=count, contrast=contrast, mean=mean,
                                     column_batch_size=column_batch_size,
                                     cache_mb=geometry_cache_mb), execution, n_workers, batch_size)
    timing['executor_setup_seconds'] = time.perf_counter()-setup_start
    evaluations, warm = [], np.zeros((count-1, terms))
    warm_basis = None
    best, last_payload, cache = None, None, {}
    def solve(log_values, names, force=False):
        nonlocal warm, warm_basis, best, last_payload
        key = tuple(log_values)
        if key in cache and not force:
            return cache[key]
        current = dict(options)
        current.update(zip(names, np.exp(log_values)))
        basis = _basis(x, profile_method, order, current)
        initial_coefficients = (warm if warm_basis is None else
                                np.linalg.lstsq(basis, warm_basis@warm.T, rcond=None)[0].T)
        tick = time.perf_counter()
        result = _fit_coefficients(executor,basis,initial_coefficients,maxiter,timing)
        timing['coefficient_solver_seconds'] += time.perf_counter()-tick
        if not result.success:
            raise RuntimeError('shared coefficient fit did not converge: '+str(result.message))
        coefficients = result.x.reshape(warm.shape)
        tick = time.perf_counter()
        nll, _, curvature, _ = executor.evaluate(coefficients, basis, curvature=True)
        curvature.flat[::dimension+1] += 1.
        try:
            factor = cho_factor((curvature+curvature.T)/2)
        except np.linalg.LinAlgError as exc:
            raise RuntimeError('conditional profile Hessian is not positive definite; Laplace fit invalid') from exc
        evidence = nll+.5*result.x@result.x+np.log(np.diag(factor[0])).sum()
        timing['curvature_seconds'] += time.perf_counter()-tick
        record = dict(objective=float(evidence), options=current,
                      inner_iterations=int(result.nit), gradient_max=float(np.max(abs(result.jac))))
        record['inner_message'] = str(result.message)
        record['coefficient_objective_calls'] = int(result.nfev)
        evaluations.append(record)
        payload = (float(evidence), coefficients.copy(), basis, factor, record)
        last_payload = payload
        # Cache only objective numbers: retaining every factor exhausts memory.
        cache[key] = float(evidence)
        warm = coefficients.copy()
        warm_basis = basis
        if best is None or evidence < best[0]:
            best = payload
        return float(evidence)
    try:
        names = ['amplitude']+(['length_scale'] if profile_method == 'gp' else [])
        bounds = {'amplitude':(.03, 10.)}
        if profile_method == 'gp':
            bounds['length_scale'] = ((x[-1]-x[0])/(options['n_inducing']-1),
                                      10*(x[-1]-x[0]))
        if hyperparameter_bounds:
            if set(hyperparameter_bounds)-set(names):
                raise ValueError('unknown hyperparameter bound')
            bounds.update(hyperparameter_bounds)
        for name, limits in bounds.items():
            if len(limits) != 2 or not 0 < limits[0] < limits[1] or not np.all(np.isfinite(limits)):
                raise ValueError('hyperparameter bounds must be positive finite ordered pairs')
        starts, outer_results, converged_solution = [], [], None
        if optimize_hyperparameters:
            log_bounds = np.log([bounds[name] for name in names])
            initial = np.clip(np.log([options[name] for name in names]), log_bounds[:,0], log_bounds[:,1])
            for start in range(n_starts):
                proposal = initial.copy()
                if start:
                    # Deterministic interior alternatives; no random state.
                    fraction = start/(n_starts+1)
                    proposal = log_bounds[:,0]+fraction*(log_bounds[:,1]-log_bounds[:,0])
                starts.append(np.exp(proposal).tolist())
                cache.clear()
                warm = np.zeros_like(warm)
                warm_basis = None
                def hyper_objective(values):
                    try:
                        return solve(values, names)
                    except RuntimeError as exc:
                        # A failed numerical trial is not a valid evidence value.
                        # Finite penalty lets line search retreat without NaNs.
                        evaluations.append(dict(options=dict(zip(names,np.exp(values))),
                                                failed=True, message=str(exc)))
                        return 1e30
                def hyper_value_gradient(values):
                    value = hyper_objective(values)
                    gradient = np.empty(len(values))
                    for i in range(len(values)):
                        plus, minus = values.copy(), values.copy()
                        plus[i] = min(values[i]+1e-3, log_bounds[i,1])
                        minus[i] = max(values[i]-1e-3, log_bounds[i,0])
                        gradient[i] = (hyper_objective(plus)-hyper_objective(minus))/(plus[i]-minus[i])
                    return value, gradient
                result = minimize(hyper_value_gradient, proposal, jac=True,
                                  method='L-BFGS-B', bounds=log_bounds,
                                  options={'maxiter':hyper_maxiter, 'ftol':1e-7,
                                           'gtol':5e-3, 'maxls':20})
                outer_results.append(dict(success=bool(result.success), message=str(result.message),
                                          objective=float(result.fun), iterations=int(result.nit)))
                if result.success and result.fun < 1e30:
                    # Select an optimizer endpoint, not a finite-difference trial
                    # which happened to have a slightly lower objective.
                    solve(result.x, names, force=True)
                    if converged_solution is None or last_payload[0] < converged_solution[0]:
                        converged_solution = last_payload
            if converged_solution is None:
                raise RuntimeError('hyperparameter optimization did not converge: '+outer_results[-1]['message'])
            best = converged_solution
        else:
            solve(np.log([options[name] for name in names]), names)
        value, coefficients, basis, factor, record = best
        _, _, _, reports = executor.evaluate(coefficients, basis, details=True)
        covariance = cho_solve(factor, np.eye(dimension)) if store_covariance else None
        fitted = record['options']
        boundary = [name for name in names if optimize_hyperparameters and
                    min(abs(np.log(fitted[name]/bounds[name][0])),
                        abs(np.log(fitted[name]/bounds[name][1]))) < .01]
        diagnostics = dict(version=_VERSION, approximation='Laplace conditional on optimized training fluxes',
                           objective=value, hyperparameters_optimized=bool(optimize_hyperparameters),
                           hyperparameter_bounds=bounds, boundary_parameters=boundary,
                           starts=starts, outer_results=outer_results, evaluations=evaluations,
                           training_indices=indices.tolist(),
                           integration_ids=[str(integration_ids[i]) if integration_ids is not None else str(i)
                                            for i in indices], training_fingerprint=digest.hexdigest(),
                           training=reports, execution=execution, n_workers=n_workers,
                           active_workers=len(executor.workers),
                           batch_size=batch_size, initial_width=initial_width,
                           polynomial_order=order, coefficient_dimension=dimension,
                           cropped_data_mb=cropped_bytes/1e6, curvature_mb=matrix_mb,
                           estimated_working_memory_mb=estimated_memory,
                           seconds=time.perf_counter()-started,
                           timing=timing,
                           uncertainty='coefficient covariance conditional on fluxes and hyperparameters; '
                                       'extracted inverse variance conditional on profile')
        if boundary:
            warnings.warn('Shared-profile hyperparameters near bounds: '+', '.join(boundary), UserWarning)
        return SharedProfile(coefficients, basis, contrast, mean, x, nrows, radius, spacing,
                             profile_method, fitted, diagnostics, covariance)
    finally:
        executor.close()
