"""Correlated adjacent-read ramp fitting of JWST datamodels.

The array-level fitter is :func:`transitspectroscopy.core.ramps.fit_ramps`
(re-exported here). This module adds the JWST layer: reading gain and read noise
from CRDS reference files, writing rate datamodels, and a ``calwebb_detector1``
pipeline whose ramp-fitting step uses the correlated fitter.
"""
from copy import deepcopy
import json
import os

import numpy as np

from .core.provenance import option_record
from .core.ramps import difference_weights, fit_ramps, validate_selection  # noqa: F401


def jump_step_options(options, use_tso_jump):
    """SPARTA options after a jump step.

    With the TSO jump step, its JUMP_DET flags alone define the excluded differences: SPARTA's
    own rejection is off (``sigma=None``) unless ``sigma`` is given in ``options``.
    """
    options = dict(options or {})
    if use_tso_jump:
        options.setdefault('sigma', None)
    return options


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
                      options=option_record(original_options), references=references)
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
