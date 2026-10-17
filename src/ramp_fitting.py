"""Correlated adjacent-read ramp fitting (arXiv:2305.06240, Methods).

Independent implementation of the paper's covariance equations; SPARTA's
``calibrate.py`` is a scientific reference, not an imported dependency.
Arrays enter in DN and leave in DN/s. Explicit read noise is per-read electrons.
"""
from copy import deepcopy
import hashlib
import json
import os
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
    allowed = {'gain', 'read_noise', 'override_gain', 'override_readnoise',
               'residual_correction', 'residual_template', 'template_integrations',
               'sigma', 'max_iterations', 'firstgroup', 'lastgroup', 'chunk_pixels'}
    unknown = set(options or {}) - allowed
    if unknown:
        raise ValueError(f'Unknown SPARTA ramp options: {sorted(unknown)}')


def _median(array, axis):
    # Empty masked ramps/pixels are expected and explicitly flagged downstream.
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        return np.nanmedian(array, axis=axis)


def _weights(signal, read_variance, valid):
    """Return C^-1 1 using a batched tridiagonal solve; groups are last.

    Removing a difference deletes its row/column in C. Zero off-diagonals at
    gaps implement that restricted covariance, not zeroed full-ramp weights.
    The s=0 read-noise limit is solved directly without hyperbolic overflow.
    """
    r2 = np.broadcast_to(read_variance, valid.shape[:-1])
    diagonal = np.where(valid, (signal + 2*r2)[..., None], 1.).astype(float)
    rhs = valid.astype(float)
    off = -r2[..., None] * (valid[..., :-1] & valid[..., 1:])
    for j in range(1, valid.shape[-1]):
        factor = off[..., j-1] / diagonal[..., j-1]
        diagonal[..., j] -= factor * off[..., j-1]
        rhs[..., j] -= factor * rhs[..., j-1]
    rhs[..., -1] /= diagonal[..., -1]
    for j in range(valid.shape[-1]-2, -1, -1):
        rhs[..., j] = (rhs[..., j] - off[..., j]*rhs[..., j+1]) / diagonal[..., j]
    return rhs


def _fit_differences(differences, valid, read_variance, sigma, max_iterations):
    valid = valid.copy()
    initial_valid = valid.copy()
    estimate = np.nan_to_num(_median(np.where(valid, differences, np.nan), -1))
    converged = np.ones(valid.shape[:-1], dtype=bool)
    for iteration in range(max_iterations):
        signal = np.maximum(estimate, 0.)
        # A zero-R, non-positive-s covariance has no defined likelihood.
        valid &= ((signal + 2*read_variance) > 0)[..., None]
        rejected = np.zeros(valid.shape[:-1], dtype=bool)
        if sigma is not None:
            noise = np.sqrt(signal + 2*read_variance)
            z = np.divide(np.abs(differences-estimate[..., None]), noise[..., None],
                          out=np.zeros_like(differences), where=noise[..., None] > 0)
            z = np.where(valid, z, -np.inf)
            worst = np.argmax(z, axis=-1)
            rejected = np.max(z, axis=-1) > sigma
            # One difference at a time avoids rejecting neighbouring good reads
            # when a large jump initially perturbs the slope estimate.
            np.put_along_axis(valid, worst[..., None],
                              np.take_along_axis(valid, worst[..., None], -1)
                              & ~rejected[..., None], axis=-1)
        weights = _weights(signal, read_variance, valid)
        norm = weights.sum(axis=-1)
        alpha = np.divide(weights, norm[..., None], out=np.zeros_like(weights),
                          where=norm[..., None] > 0)
        estimate = np.sum(alpha*np.where(valid, differences, 0.), axis=-1)
        if not np.any(rejected):
            break
        if iteration == max_iterations-1:
            converged = ~rejected
    square = np.sum(alpha**2, axis=-1)
    photon = signal*square
    read = read_variance*(2*square - 2*np.sum(alpha[..., :-1]*alpha[..., 1:], axis=-1))
    good = valid.any(axis=-1) & converged
    return (np.where(good, estimate, np.nan), np.where(good, photon, np.nan),
            np.where(good, read, np.nan), valid.sum(axis=-1),
            (initial_valid & ~valid).sum(axis=-1), converged)


def fit_ramps(data, *, group_time, gain, read_noise, groupdq=None, pixeldq=None,
              nframes=1, residual_correction=False, residual_template=None,
              template_integrations=None, sigma=5., max_iterations=50,
              firstgroup=None, lastgroup=None, chunk_pixels=256):
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
    last = ng-1 if lastgroup is None else lastgroup
    if not all(isinstance(v, (int, np.integer)) for v in (first, last)) or not 0 <= first < last < ng:
        raise ValueError('firstgroup/lastgroup must retain at least two groups')
    gain = np.broadcast_to(np.asarray(gain, float), (ny, nx)).reshape(-1)
    noise = np.broadcast_to(np.asarray(read_noise, float), (ny, nx)).reshape(-1)
    good_reference = np.isfinite(gain) & (gain > 0) & np.isfinite(noise) & (noise >= 0)
    safe_gain = np.where(good_reference, gain, 1.)
    noise = np.where(good_reference, noise, 1.)
    dq = np.zeros(data.shape, np.uint32) if groupdq is None else np.asarray(groupdq)
    pdq = np.zeros((ny, nx), np.uint32) if pixeldq is None else np.asarray(pixeldq)
    if dq.shape != data.shape or pdq.shape != (ny, nx):
        raise ValueError('DQ arrays must match the input ramp and image shapes')
    if residual_template is not None and not residual_correction:
        raise ValueError('residual_template requires residual_correction=True')
    if template_integrations is not None and (not residual_correction or residual_template is not None):
        raise ValueError('template_integrations requires training a residual template')
    training = np.arange(ni) if template_integrations is None else np.asarray(template_integrations)
    if training.ndim != 1 or not len(training) or training.dtype.kind not in 'iu' or np.any((training < 0) | (training >= ni)) or len(np.unique(training)) != len(training):
        raise ValueError('template_integrations must contain unique valid local integration indices')
    template = np.zeros((ng, ny*nx), float)
    if residual_template is not None:
        supplied = np.asarray(residual_template, float)
        if supplied.shape != (ng, ny, nx) or not np.isfinite(supplied).all():
            raise ValueError('residual_template must be finite with shape (group, row, column)')
        template[:] = supplied.reshape(ng, -1)
    out = {k: np.empty((ni, ny*nx), float) for k in ('data', 'var_poisson', 'var_rnoise')}
    out.update({k: np.empty((ni, ny*nx), np.int32) for k in ('n_used', 'n_rejected')})
    out['converged'] = np.empty((ni, ny*nx), bool)
    out['dq'] = np.broadcast_to(pdq.reshape(1, -1), (ni, ny*nx)).astype(np.uint32).copy()
    out['dq'][:, ~good_reference] |= 1
    # Standard JWST DQ bit values: DO_NOT_USE=1, SATURATED=2, JUMP_DET=4.
    flat_data, flat_dq = data.reshape(ni, ng, -1), dq.reshape(ni, ng, -1)
    size = max(1, min(chunk_pixels, (64*1024**2)//(ni*ng*8*20)))
    for start in range(0, ny*nx, size):
        sl = slice(start, min(start+size, ny*nx))
        values = np.asarray(flat_data[..., sl], float)*safe_gain[sl]
        flags = flat_dq[..., sl]
        saturated = np.maximum.accumulate((flags & 2) != 0, axis=1)
        valid_read = np.isfinite(values) & ((flags & 1) == 0) & ~saturated
        valid_read[:, :first] = False
        valid_read[:, last+1:] = False
        valid_read &= (good_reference[sl] & ((pdq.reshape(-1)[sl] & 1) == 0))[None, None, :]
        valid_diff = valid_read[:, 1:] & valid_read[:, :-1] & ((flags[:, 1:] & 4) == 0)
        valid_diff = valid_diff.transpose(0, 2, 1)
        r2 = noise[sl][None, :]**2
        if residual_correction and residual_template is None:
            initial = _fit_differences(np.diff(values, axis=1).transpose(0, 2, 1),
                                       valid_diff, r2, None, 1)[0]
            residual = values - initial[:, None, :]*np.arange(ng)[None, :, None]
            residual = np.where(valid_read, residual, np.nan)
            # Remove each ramp's arbitrary pedestal before pooling integrations.
            residual -= _median(residual, 1)[:, None, :]
            # Jump-flagged ramps do not train the empirical correction template.
            clean = ~np.any((flags & 4) != 0, axis=1)
            residual = np.where(clean[:, None, :], residual, np.nan)
            correction = _median(residual[training], 0)
            correction -= _median(correction, 0)[None, :]
            template[:, sl] = np.nan_to_num(correction)/safe_gain[sl]
        corrected = values - template[None, :, sl]*safe_gain[sl]
        fitted = _fit_differences(np.diff(corrected, axis=1).transpose(0, 2, 1),
                                  valid_diff, r2, sigma, max_iterations)
        scale = safe_gain[sl]*group_time
        for key, value, divisor in zip(('data', 'var_poisson', 'var_rnoise'), fitted[:3],
                                       (scale, scale**2, scale**2)):
            out[key][:, sl] = value/divisor
        out['n_used'][:, sl], out['n_rejected'][:, sl], out['converged'][:, sl] = fitted[3:]
        out['dq'][:, sl] |= np.where(np.isfinite(out['data'][:, sl]), 0, 1).astype(np.uint32)
        out['dq'][:, sl] |= np.where((fitted[4] > 0) | np.any((flags & 4) != 0, axis=1), 4, 0).astype(np.uint32)
    out = {k: v.reshape(ni, ny, nx) for k, v in out.items()}
    out['err'] = np.sqrt(out['var_poisson'] + out['var_rnoise'])
    out['residual_template'] = template.reshape(ng, ny, nx)
    return out


def _option_record(options):
    """Compact reproducible provenance without embedding large arrays in headers."""
    record = {}
    for key, value in options.items():
        if isinstance(value, np.ndarray):
            arr = np.ascontiguousarray(value)
            value = dict(shape=list(arr.shape), dtype=str(arr.dtype),
                         sha256=hashlib.sha256(arr.tobytes()).hexdigest())
        elif isinstance(value, np.generic):
            value = value.item()
        elif isinstance(value, os.PathLike):
            value = os.fspath(value)
        record[key] = value
    return record


def fit_model(model, *, reference_step=None, **options):
    """Return (ImageModel, CubeModel, diagnostics), resolving JWST references.

    Missing gain/read noise are read from CRDS (or override_gain/readnoise).
    JWST READNOISE is converted to per-read electrons as gain*reference/sqrt(2),
    matching stcal for NFRAMES=1. Explicit read_noise already has those units.
    """
    from stdatamodels.jwst import datamodels
    from jwst.ramp_fitting import RampFitStep
    from jwst.lib import reffile_utils
    validate_selection('sparta', options)
    read_times = getattr(model.meta.exposure, 'read_times', None)
    if read_times is not None and len(read_times):
        raise ValueError('SPARTA requires uniform group timing; use JWST for read_times')
    if model.meta.exposure.nframes != 1:
        raise ValueError('SPARTA covariance requires NFRAMES=1')
    original_options = deepcopy(options)
    overrides = {k: options.pop(k) for k in ('override_gain', 'override_readnoise') if k in options}
    step = reference_step if reference_step is not None else RampFitStep(**overrides)
    references = {}
    gain_factor = model.meta.exposure.gain_factor
    for kind, key, cls in (('gain', 'gain', datamodels.GainModel),
                           ('readnoise', 'read_noise', datamodels.ReadnoiseModel)):
        if key in options:
            if 'override_'+kind in overrides:
                raise ValueError(f'Supply either {key} or override_{kind}, not both')
            continue
        filename = overrides.get('override_'+kind) or step.get_reference_file(model, kind)
        if not filename or filename == 'N/A':
            raise ValueError(f'No {kind} reference available; supply an explicit {key}')
        filename = os.fspath(filename)
        references[kind] = filename
        with cls(filename) as ref:
            if reffile_utils.ref_matches_sci(model, ref):
                array = ref.data.copy()
            else:
                with reffile_utils.get_subarray_model(model, ref) as sub:
                    array = sub.data.copy()
            if kind == 'gain' and ref.meta.exposure.gain_factor is not None:
                gain_factor = ref.meta.exposure.gain_factor
        options[key] = array if kind == 'gain' else array*np.asarray(options['gain'])/np.sqrt(2.)
    result = fit_ramps(model.data, group_time=model.meta.exposure.group_time,
                       nframes=model.meta.exposure.nframes, groupdq=model.groupdq,
                       pixeldq=model.pixeldq, **options)
    cube = datamodels.CubeModel()
    cube.update(model)
    for key in ('data', 'err', 'var_poisson', 'var_rnoise', 'dq'):
        setattr(cube, key, result[key])
    if model.int_times is not None:
        cube.int_times = model.int_times.copy()
    stripe = getattr(model, 'int_times_stripe', None)
    if stripe is not None and len(stripe):
        cube.int_times_stripe = stripe.copy()
    image = datamodels.ImageModel()
    image.update(model)
    variance = result['err']**2
    valid = np.isfinite(variance) & (variance > 0) & ((result['dq'] & 1) == 0)
    precision = np.divide(1., variance, out=np.zeros_like(variance), where=valid)
    total = precision.sum(axis=0)
    weight = np.divide(precision, total, out=np.zeros_like(precision), where=total > 0)
    image.data = np.where(total > 0, np.sum(weight*np.where(valid, result['data'], 0.), axis=0), np.nan)
    for key in ('var_poisson', 'var_rnoise'):
        setattr(image, key, np.where(total > 0, np.sum(weight**2*np.where(valid, result[key], 0.), axis=0), np.nan))
    image.err = np.sqrt(image.var_poisson + image.var_rnoise)
    image.dq = np.bitwise_or.reduce(result['dq'] & np.uint32(0xfffffffe), axis=0)
    image.dq |= np.where(total > 0, 0, 1).astype(np.uint32)
    provenance = dict(method='sparta', implementation=1, paper='arXiv:2305.06240v1',
                      options=_option_record(original_options), references=references)
    for product in (image, cube):
        product.meta.cal_step.ramp_fit = 'COMPLETE'
        product.meta.bunit_data = product.meta.bunit_err = 'DN/s'
        product.meta.exposure.gain_factor = gain_factor
        for kind, filename in references.items():
            getattr(product.meta.ref_file, kind).name = filename
        product.add_history_entry('transitspectroscopy ramp fit: '+json.dumps(provenance, sort_keys=True))
    diagnostics = {k: result[k] for k in ('n_used', 'n_rejected', 'converged', 'residual_template')}
    diagnostics['provenance'] = provenance
    return image, cube, diagnostics


def detector1_class(options=None):
    """Official Detector1 with only its ramp-fitting step replaced.

    This keeps MIRI reset/firstframe/lastframe corrections and gain_scale in
    their official order. Options belong to this invocation, never global state.
    """
    from jwst.pipeline import Detector1Pipeline
    from jwst.ramp_fitting import RampFitStep
    configured = deepcopy(options or {})
    validate_selection('sparta', configured)

    class CorrelatedRampFitStep(RampFitStep):
        # Explicit arrays/files need no additional CRDS reference prefetch.
        reference_file_types = [kind for kind, key in (('gain', 'gain'), ('readnoise', 'read_noise'))
                                if key not in configured and 'override_'+kind not in configured]

        def process(self, step_input):
            from stdatamodels.jwst import datamodels
            with datamodels.open(step_input) as model:
                local = deepcopy(configured)
                for key in ('firstgroup', 'lastgroup'):
                    value = getattr(self, key)
                    if value is not None:
                        local.setdefault(key, value)
                image, cube, self.ramp_diagnostics = fit_model(model, reference_step=self, **local)
            return image, cube

    class CorrelatedDetector1Pipeline(Detector1Pipeline):
        step_defs = dict(Detector1Pipeline.step_defs, ramp_fit=CorrelatedRampFitStep)

    return CorrelatedDetector1Pipeline
