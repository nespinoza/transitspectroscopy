"""Small helpers for JWST datamodels shared by the instrument modules."""

from contextlib import contextmanager
from pathlib import Path

import numpy as np


@contextmanager
def open_model(value):
    """Open a datamodel from a path, or pass an already-open model through.

    Only models opened here are closed on exit.
    """
    if isinstance(value, (str, Path)):
        from stdatamodels.jwst import datamodels

        with datamodels.open(str(value)) as model:
            yield model
    else:
        yield value


def integration_times(model):
    """Mid-integration times (BJD_TDB) of the integrations loaded in ``model``.

    JWST ``INT_TIMES`` stores barycentric *modified* Julian dates despite its column
    name ``int_mid_BJD_TDB``; 2400000.5 is added here.
    """
    table = model.int_times
    start = model.meta.exposure.integration_start
    end = model.meta.exposure.integration_end
    selection = (table['integration_number'] >= start) & (table['integration_number'] <= end)
    time = np.asarray(table['int_mid_BJD_TDB'][selection], float)
    if len(time) != model.data.shape[0] or not np.isfinite(time).all():
        raise ValueError('INT_TIMES does not match the loaded integrations')
    return time + 2400000.5


def ramp_1f_validity(ramps):
    """Groups usable for the group-level 1/f levels and templates of every segment.

    Returns two lists of boolean ``(integration, group, row, column)`` arrays: ``valid``
    (not DO_NOT_USE or SATURATED, not a DO_NOT_USE or reference pixel) and ``template_valid``
    (also not at or after a jump). ``transitspectroscopy.core.detector.group_difference_1f``
    builds its templates from ``template_valid`` groups, and its 1/f levels too unless most
    background pixels of a group are flagged (an offset shared by the whole group).
    """
    from stdatamodels.jwst.datamodels import dqflags

    unusable = dqflags.group['DO_NOT_USE'] | dqflags.group['SATURATED']
    reference = dqflags.pixel['DO_NOT_USE'] | dqflags.pixel['REFERENCE_PIXEL']
    valid, template_valid = [], []
    for ramp in ramps:
        groupdq = ramp.groupdq
        good = ((groupdq & unusable) == 0) & ((ramp.pixeldq & reference) == 0)[None, None]
        after_jump = np.logical_or.accumulate((groupdq & dqflags.group['JUMP_DET']) > 0, axis=1)
        valid.append(good)
        template_valid.append(good & ~after_jump)
    return valid, template_valid


def linearity_coefficients(ramp):
    """Linearity polynomial ``(coefficient, row, column)`` used for ``ramp``, or None.

    The reference file recorded by the linearity step (``meta.ref_file.linearity``) is read
    from the CRDS cache and cut to the subarray of ``ramp``; pixels without a correction
    (``NO_LIN_CORR`` or non-finite coefficients) get the identity. None if the ramp records no
    linearity reference.
    """
    import os

    name = getattr(getattr(ramp.meta.ref_file, 'linearity', None), 'name', None)
    if not name or name.upper() == 'N/A':
        return None
    import crds
    from jwst.lib.reffile_utils import get_subarray_model
    from stdatamodels.jwst.datamodels import LinearityModel, dqflags

    path = crds.locate_file(os.path.basename(name), observatory='jwst')
    with LinearityModel(path) as reference:
        subarray = get_subarray_model(ramp, reference)
        coefficients = np.array(subarray.coeffs, dtype=np.float64)
        dq = np.asarray(subarray.dq)
    identity = (((dq & dqflags.pixel['NO_LIN_CORR']) != 0)
                | ~np.all(np.isfinite(coefficients), axis=0))
    coefficients[:, identity] = 0.
    coefficients[1, identity] = 1.
    return coefficients


def ramp_jumps(ramps):
    """Boolean ``(integration, group, row, column)`` arrays, True where a jump was detected."""
    from stdatamodels.jwst.datamodels import dqflags

    return [(ramp.groupdq & dqflags.group['JUMP_DET']) > 0 for ramp in ramps]


def dq_masked(dq):
    """Pixels that ``mask_dq`` excludes: any DQ flag but JUMP_DET.

    A JUMP_DET pixel of a rate has its slope fitted around the jump, so its rate is valid.
    """
    from stdatamodels.jwst.datamodels import dqflags

    # Stage 2 keeps DQ as floats; the flags are integers stored exactly.
    return (np.asarray(dq).astype(np.uint32) & ~np.uint32(dqflags.pixel['JUMP_DET'])) != 0


# Shorter than a FITS HISTORY card (72 characters), so that it is read back as one entry:
GROUP_1F_HISTORY = 'transitspectroscopy: group-difference 1/f correction (fit_ramps)'


def mark_group_1f(model):
    """Record in ``model`` (rates per integration) the group-difference 1/f correction."""
    model.history.append(GROUP_1F_HISTORY)


def group_1f_corrected(value):
    """True if the rates (a model or a path) were corrected with ``mark_group_1f``."""
    with open_model(value) as model:
        return any(GROUP_1F_HISTORY in str(entry) for entry in getattr(model, 'history', ()))
