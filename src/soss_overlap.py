"""Joint extraction of overlapping NIRISS/SOSS orders (compatibility module).

The implementation lives in :mod:`transitspectroscopy.core.overlap` and
:func:`transitspectroscopy.core.extraction.solve_blended`.
"""
from .core.extraction import solve_blended as solve_flux  # noqa: F401
from .core.overlap import (  # noqa: F401
    JointProfileFit, extract_blended as extract, train_joint, unmodeled_pixels,
)
