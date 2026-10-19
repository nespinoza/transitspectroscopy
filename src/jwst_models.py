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
    (not DO_NOT_USE or SATURATED, not a DO_NOT_USE or reference pixel), for the 1/f levels,
    and ``template_valid`` (also not at or after a jump), for the templates. Jump-flagged
    groups are excluded from the templates but not from the levels: an offset shared by a
    whole integration (PID 1541 observation 1, integration 60) gets most background pixels
    flagged, and would otherwise remain on the (unflagged) trace pixels.
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
