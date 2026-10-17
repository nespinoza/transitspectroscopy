"""Observatory-agnostic algorithms of transitspectroscopy.

Everything in ``transitspectroscopy.core`` works on plain NumPy arrays (and small
dataclasses), so the same algorithms can be fed by any observatory adapter and
tested without the JWST pipeline. Modules in this subpackage never import
``jwst``, ``stdatamodels`` or ``gwcs``; instrument-specific code lives outside it.

Modules
-------
kernels         Gaussian-process covariance functions and whitened inducing-point bases.
geometry        Pixel/aperture geometry shared by profile fitting and extraction.
profiles        Marsh (1989) spatial-profile fitting, with polynomial or GP bases.
extraction      Box, Marsh optimal and joint multi-trace flux extraction.
shared_profile  One intrinsic profile trained on several integrations (SharedProfile).
overlap         Joint profile training for overlapping traces (e.g. SOSS orders).
tracing         Trace finding, cross-correlation and FWHM measurements.
photometry      Aperture, background, centroid and 2-D GP PSF photometry.
ramps           Correlated adjacent-read ramp fitting.
detector        Detector-level corrections: pedestals, 1/f, jumps, LOOM/ROEBA, masks.
timeseries      Outlier detection in time series.
binning         Binning of spectra at a given resolving power.
provenance      File hashes and other provenance helpers.
"""
