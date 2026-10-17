"""Correlated adjacent-read ramp fitting (arXiv:2305.06240, Methods).

Up-the-ramp group differences of a non-destructively read detector are correlated:
each read's noise enters two adjacent differences. This module fits the slope of
each ramp with the full covariance of its differences (photon noise ``s`` on the
diagonal, read noise ``R`` coupling neighbours), rejecting jumps one difference at a
time. It is an independent NumPy implementation of the paper's equations; SPARTA's
``calibrate.py`` was a scientific reference, not a dependency.

Arrays enter in DN and leave in DN/s. ``read_noise`` is the RMS in electrons of a
single read.
"""

import warnings

import numpy as np


def validate_selection(method, options=None):
    """Validate backend selection before any calibration or file access."""
    if method not in ('jwst', 'sparta'):
        raise ValueError("ramp_fit_method must be 'jwst' or 'sparta'")
    if options is not None and not isinstance(options, dict):
        raise TypeError('ramp_fit_options must be a dictionary')
    if method == 'jwst' and options:
        raise ValueError('Use the existing ramp_fit step parameters for the JWST backend')
    allowed = {
        'gain',
        'read_noise',
        'override_gain',
        'override_readnoise',
        'residual_correction',
        'residual_template',
        'template_integrations',
        'sigma',
        'max_iterations',
        'firstgroup',
        'lastgroup',
        'chunk_pixels',
    }
    unknown = set(options or {}) - allowed
    if unknown:
        raise ValueError(f'Unknown SPARTA ramp options: {sorted(unknown)}')


def _median(array, axis):
    # Empty masked ramps/pixels are expected and explicitly flagged downstream.
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        return np.nanmedian(array, axis=axis)


def difference_weights(signal, read_variance, valid):
    """Return ``C^-1 1`` for the covariance ``C`` of adjacent group differences.

    ``C`` has ``signal + 2 R^2`` on the diagonal and ``-R^2`` next to it; it is
    solved with a batched tridiagonal (Thomas) algorithm, with groups last.

    Removing a difference deletes its row/column in C. Zero off-diagonals at
    gaps implement that restricted covariance, not zeroed full-ramp weights.
    The s=0 read-noise limit is solved directly without hyperbolic overflow.
    """
    r2 = np.broadcast_to(read_variance, valid.shape[:-1])
    diagonal = np.where(valid, (signal + 2 * r2)[..., None], 1.0).astype(float)
    rhs = valid.astype(float)
    off = -r2[..., None] * (valid[..., :-1] & valid[..., 1:])
    for j in range(1, valid.shape[-1]):
        factor = off[..., j - 1] / diagonal[..., j - 1]
        diagonal[..., j] -= factor * off[..., j - 1]
        rhs[..., j] -= factor * rhs[..., j - 1]
    rhs[..., -1] /= diagonal[..., -1]
    for j in range(valid.shape[-1] - 2, -1, -1):
        rhs[..., j] = (rhs[..., j] - off[..., j] * rhs[..., j + 1]) / diagonal[..., j]
    return rhs


def _fit_differences(differences, valid, read_variance, sigma, max_iterations):
    valid = valid.copy()
    initial_valid = valid.copy()
    estimate = np.nan_to_num(_median(np.where(valid, differences, np.nan), -1))
    converged = np.ones(valid.shape[:-1], dtype=bool)
    for iteration in range(max_iterations):
        signal = np.maximum(estimate, 0.0)
        # A zero-R, non-positive-s covariance has no defined likelihood.
        valid &= ((signal + 2 * read_variance) > 0)[..., None]
        rejected = np.zeros(valid.shape[:-1], dtype=bool)
        if sigma is not None:
            noise = np.sqrt(signal + 2 * read_variance)
            z = np.divide(
                np.abs(differences - estimate[..., None]),
                noise[..., None],
                out=np.zeros_like(differences),
                where=noise[..., None] > 0,
            )
            z = np.where(valid, z, -np.inf)
            worst = np.argmax(z, axis=-1)
            rejected = np.max(z, axis=-1) > sigma
            # One difference at a time avoids rejecting neighbouring good reads
            # when a large jump initially perturbs the slope estimate.
            np.put_along_axis(
                valid,
                worst[..., None],
                np.take_along_axis(valid, worst[..., None], -1) & ~rejected[..., None],
                axis=-1,
            )
        weights = difference_weights(signal, read_variance, valid)
        norm = weights.sum(axis=-1)
        alpha = np.divide(
            weights, norm[..., None], out=np.zeros_like(weights), where=norm[..., None] > 0
        )
        estimate = np.sum(alpha * np.where(valid, differences, 0.0), axis=-1)
        if not np.any(rejected):
            break
        if iteration == max_iterations - 1:
            converged = ~rejected
    square = np.sum(alpha**2, axis=-1)
    photon = signal * square
    read = read_variance * (2 * square - 2 * np.sum(alpha[..., :-1] * alpha[..., 1:], axis=-1))
    good = valid.any(axis=-1) & converged
    return (
        np.where(good, estimate, np.nan),
        np.where(good, photon, np.nan),
        np.where(good, read, np.nan),
        valid.sum(axis=-1),
        (initial_valid & ~valid).sum(axis=-1),
        converged,
    )


def fit_ramps(
    data,
    *,
    group_time,
    gain,
    read_noise,
    groupdq=None,
    pixeldq=None,
    nframes=1,
    residual_correction=False,
    residual_template=None,
    template_integrations=None,
    sigma=5.0,
    max_iterations=50,
    firstgroup=None,
    lastgroup=None,
    chunk_pixels=256,
):
    """Fit calibrated (integration, group, row, column) DN ramps.

    ``gain`` is electrons/DN; ``read_noise`` is RMS electrons per single read
    (NOT the JWST READNOISE reference convention). Both accept scalars or maps.
    Uniform group timing and NFRAMES=1 are required. ``firstgroup``/``lastgroup``
    are inclusive zero-based limits. DNU/SATURATION groups and the difference
    ending at a JUMP_DET group are excluded. No one-group slope is invented.

    Residual correction is off by default. When enabled, it estimates a median
    template per group/pixel from this segment, or the specified integration
    indices, which should exclude the transit/eclipse. Use homogeneous flux blocks
    or disable it for variable scenes. A supplied template is in DN. Errors
    condition on that template and estimated covariance; uncertainty shared
    between integrations through template training is not included.

    Return a dictionary of rate/error/variance/DQ cubes, counts and the applied
    template. Input arrays are never modified. Extra working memory is chunked
    over pixels with a target of approximately 64 MiB (excluding input/output).
    """
    data = np.asarray(data)
    if data.ndim != 4 or min(data.shape) < 1 or data.shape[1] < 2:
        raise ValueError('Expected a nonempty 4-D ramp with at least two groups')
    if nframes != 1:
        raise ValueError('SPARTA covariance requires NFRAMES=1; use JWST for averaged groups')
    if not np.isfinite(group_time) or group_time <= 0:
        raise ValueError('group_time must be finite and positive')
    if sigma is not None and (not np.isfinite(sigma) or sigma <= 0):
        raise ValueError('sigma must be positive or None to disable clipping')
    for value, name in ((max_iterations, 'max_iterations'), (chunk_pixels, 'chunk_pixels')):
        if not isinstance(value, (int, np.integer)) or value < 1:
            raise ValueError(f'{name} must be a positive integer')
    ni, ng, ny, nx = data.shape
    first = 0 if firstgroup is None else firstgroup
    last = ng - 1 if lastgroup is None else lastgroup
    if (
        not all(isinstance(v, (int, np.integer)) for v in (first, last))
        or not 0 <= first < last < ng
    ):
        raise ValueError('firstgroup/lastgroup must retain at least two groups')
    gain = np.broadcast_to(np.asarray(gain, float), (ny, nx)).reshape(-1)
    noise = np.broadcast_to(np.asarray(read_noise, float), (ny, nx)).reshape(-1)
    good_reference = np.isfinite(gain) & (gain > 0) & np.isfinite(noise) & (noise >= 0)
    safe_gain = np.where(good_reference, gain, 1.0)
    noise = np.where(good_reference, noise, 1.0)
    dq = np.zeros(data.shape, np.uint32) if groupdq is None else np.asarray(groupdq)
    pdq = np.zeros((ny, nx), np.uint32) if pixeldq is None else np.asarray(pixeldq)
    if dq.shape != data.shape or pdq.shape != (ny, nx):
        raise ValueError('DQ arrays must match the input ramp and image shapes')
    if residual_template is not None and not residual_correction:
        raise ValueError('residual_template requires residual_correction=True')
    if template_integrations is not None and (
        not residual_correction or residual_template is not None
    ):
        raise ValueError('template_integrations requires training a residual template')
    training = np.arange(ni) if template_integrations is None else np.asarray(template_integrations)
    if (
        training.ndim != 1
        or not len(training)
        or training.dtype.kind not in 'iu'
        or np.any((training < 0) | (training >= ni))
        or len(np.unique(training)) != len(training)
    ):
        raise ValueError(
            'template_integrations must contain unique valid local integration indices'
        )
    template = np.zeros((ng, ny * nx), float)
    if residual_template is not None:
        supplied = np.asarray(residual_template, float)
        if supplied.shape != (ng, ny, nx) or not np.isfinite(supplied).all():
            raise ValueError('residual_template must be finite with shape (group, row, column)')
        template[:] = supplied.reshape(ng, -1)
    out = {k: np.empty((ni, ny * nx), float) for k in ('data', 'var_poisson', 'var_rnoise')}
    out.update({k: np.empty((ni, ny * nx), np.int32) for k in ('n_used', 'n_rejected')})
    out['converged'] = np.empty((ni, ny * nx), bool)
    out['dq'] = np.broadcast_to(pdq.reshape(1, -1), (ni, ny * nx)).astype(np.uint32).copy()
    out['dq'][:, ~good_reference] |= 1
    # Standard JWST DQ bit values: DO_NOT_USE=1, SATURATED=2, JUMP_DET=4.
    flat_data, flat_dq = data.reshape(ni, ng, -1), dq.reshape(ni, ng, -1)
    size = max(1, min(chunk_pixels, (64 * 1024**2) // (ni * ng * 8 * 20)))
    for start in range(0, ny * nx, size):
        sl = slice(start, min(start + size, ny * nx))
        values = np.asarray(flat_data[..., sl], float) * safe_gain[sl]
        flags = flat_dq[..., sl]
        saturated = np.maximum.accumulate((flags & 2) != 0, axis=1)
        valid_read = np.isfinite(values) & ((flags & 1) == 0) & ~saturated
        valid_read[:, :first] = False
        valid_read[:, last + 1 :] = False
        valid_read &= (good_reference[sl] & ((pdq.reshape(-1)[sl] & 1) == 0))[None, None, :]
        valid_diff = valid_read[:, 1:] & valid_read[:, :-1] & ((flags[:, 1:] & 4) == 0)
        valid_diff = valid_diff.transpose(0, 2, 1)
        r2 = noise[sl][None, :] ** 2
        if residual_correction and residual_template is None:
            initial = _fit_differences(
                np.diff(values, axis=1).transpose(0, 2, 1), valid_diff, r2, None, 1
            )[0]
            residual = values - initial[:, None, :] * np.arange(ng)[None, :, None]
            residual = np.where(valid_read, residual, np.nan)
            # Remove each ramp's arbitrary pedestal before pooling integrations.
            residual -= _median(residual, 1)[:, None, :]
            # Jump-flagged ramps do not train the empirical correction template.
            clean = ~np.any((flags & 4) != 0, axis=1)
            residual = np.where(clean[:, None, :], residual, np.nan)
            correction = _median(residual[training], 0)
            correction -= _median(correction, 0)[None, :]
            template[:, sl] = np.nan_to_num(correction) / safe_gain[sl]
        corrected = values - template[None, :, sl] * safe_gain[sl]
        fitted = _fit_differences(
            np.diff(corrected, axis=1).transpose(0, 2, 1), valid_diff, r2, sigma, max_iterations
        )
        scale = safe_gain[sl] * group_time
        for key, value, divisor in zip(
            ('data', 'var_poisson', 'var_rnoise'), fitted[:3], (scale, scale**2, scale**2)
        ):
            out[key][:, sl] = value / divisor
        out['n_used'][:, sl], out['n_rejected'][:, sl], out['converged'][:, sl] = fitted[3:]
        out['dq'][:, sl] |= np.where(np.isfinite(out['data'][:, sl]), 0, 1).astype(np.uint32)
        out['dq'][:, sl] |= np.where(
            (fitted[4] > 0) | np.any((flags & 4) != 0, axis=1), 4, 0
        ).astype(np.uint32)
    out = {k: v.reshape(ni, ny, nx) for k, v in out.items()}
    out['err'] = np.sqrt(out['var_poisson'] + out['var_rnoise'])
    out['residual_template'] = template.reshape(ng, ny, nx)
    return out
