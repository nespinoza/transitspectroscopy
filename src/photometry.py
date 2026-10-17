"""Aperture and 2-D GP PSF photometry (compatibility module).

The implementation lives in :mod:`transitspectroscopy.core.photometry`.
"""
from .core.photometry import (  # noqa: F401
    GPPSF, background, centroid, circular_weights, fit_gp_psf, optimal_flux, sky_precision,
    valid_pixels,
)
