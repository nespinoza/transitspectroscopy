"""Time-series outlier detection (compatibility module).

The implementation lives in :mod:`transitspectroscopy.core.timeseries`.
"""
from .utils import *  # noqa: F401,F403  (historical star-exported names)
from .core.timeseries import outlier_detector  # noqa: F401
