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
