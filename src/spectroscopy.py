"""Spectral extraction: the public, backward-compatible interface.

The algorithms live in ``transitspectroscopy.core`` (``core.extraction``,
``core.profiles``, ``core.tracing``); this module keeps the original function names
and signatures, and dispatches the Marsh routines to the optional C extension when
it is installed.
"""
import numpy as np

# These imports are kept so the names historically star-exported from this module
# (``from transitspectroscopy.spectroscopy import *``) remain available.
from astropy.convolution import convolve_fft  # noqa: F401
from astropy.modeling.models import Gaussian1D  # noqa: F401
from scipy.interpolate import splev, splrep, sproot, interp1d  # noqa: F401
from scipy.ndimage import gaussian_filter1d, median_filter  # noqa: F401
from scipy.optimize import fminbound  # noqa: F401

from ._optional import OptionalModule as _OptionalModule
from .core import extraction as _extraction
from .core import geometry as _geometry
from .core import profiles as _profiles
from .core.backends import select_backend as _select_backend
from .core.tracing import (  # noqa: F401  (public re-exports)
    double_gaussian, gaussian, get_ccf, get_ccf_convolve, get_fwhm, get_mad_sigma,
    get_pm_ccf, trace_fwhm, trace_spectrum,
)

CCF = _OptionalModule('CCF')
Marsh = _OptionalModule('Marsh')


def _extraction_inputs(data, centroids, variance, aperture, spacing, ron, gain,
                       nsigma, min_column, max_column, mask_invalid=False):
    if mask_invalid:
        data = _geometry.mask_invalid_pixels(data, variance)
    data, centroids, variance = _geometry.validate_image(data, centroids, variance)
    if variance is None and (not np.isfinite(ron) or ron < 0 or not np.isfinite(gain) or gain <= 0):
        raise ValueError('ron must be nonnegative and gain positive and finite')
    if not np.isfinite(nsigma) or nsigma < 0:
        raise ValueError('nsigma must be finite and nonnegative')
    # Validate before passing any raw pointers to the historical C backend.
    columns, _, _, inside, _, _, _ = _geometry.marsh_geometry(
        centroids, data.shape[0], aperture, spacing, min_column, max_column)
    if variance is not None:
        good = inside & (data[:, columns].T != -9999)
        v = variance[:, columns].T[good]
        if np.any(v <= 0) or not np.all(np.isfinite(v)):
            raise ValueError('data_variance must be finite and positive in the valid aperture')
    return data, centroids, variance


def getP(data, centroids, aperture_radius, ron, gain, nsigma, polynomial_spacing, polynomial_order, min_column = None, max_column = None, return_flat = False, data_variance = None, backend='auto', profile_method='polynomial', gp_options=None):
    """
    Given a 2D-spectrum, centroids over it and various properties of the noise, this function returns the light 
    fractions of a spectrum using the algorithm described in detail in Marsh (1989, PASP 101, 1032). 
   
    Parameters
    ----------
    data : numpy.array
        Array containing the 2D-spectrum. 

    centroids : numpy.array
        Array containing the centroids at each column of the spectra

    aperture_radius : double
        Aperture radius (measured from the center).

    ron : double 
        Read-out-noise of the detector, in electrons.

    gain : double
        Gain of the detector, in electrons/ADU.

    nsigma : double
        Number-of-sigmas to reject outliers.

    polynomial_spacing : double
        Pixel spacing between polynomials of Marsh's algorithm.

    polynomial_order : int
        Order of the polynomials that will be fitted to the surface in Marsh's algorithm (N in the paper). 

    min_column : int 
        (Optional) Minimum column to consider the calculation from.

    max_column : int
        (Optional) Maximum column to consider the calculation from.

    return_flat : bool
        (Optional) If `True`, returns the flattened version of the light fractions. Default is `False`.

    data_variance : numpy.array
        (Optional) Array containing the variances of each of the points in the 2-D `data` array. If defined, the `ron` and `gain` will be ignored.

    backend : {'auto', 'c', 'python'}
        Auto uses Marsh if installed, otherwise the NumPy/SciPy implementation.
        Existing polynomial calls retain their numerical model and return layout.
    profile_method : {'polynomial', 'gp'}
        GP is an experimental joint inducing-point profile fit along dispersion.
        It requires Python and never becomes active without explicit selection.
        Missing science/nonpositive or nonfinite variance is masked in the GP
        path. Only complete aperture columns train its normalized profile;
        rejection removes entire training columns, which are still predicted
        and remain extractable using their valid pixels.
    gp_options : dict or None
        Fixed kernel ('matern32' or 'matern52'), length_scale in dispersion pixels,
        amplitude in unnormalized profile units, and n_inducing. Defaults are
        matern32, max(10, selected_columns/5), 1, min(16, selected_columns).
        The model clips negative predictions and normalizes each column; it does
        not propagate profile uncertainty into extracted-spectrum errors.

    Returns
    -------

    P : numpy.array
        Light frations (weights) of the optimal extraction.

    """

    data, centroids, data_variance = _extraction_inputs(
        data, centroids, data_variance, aperture_radius, polynomial_spacing,
        ron, gain, nsigma, min_column, max_column, mask_invalid=profile_method == 'gp')
    if profile_method not in ('polynomial', 'gp'):
        raise ValueError('profile_method must be polynomial or gp')
    if int(polynomial_order) != polynomial_order or polynomial_order < 1 or nsigma == 0:
        raise ValueError('profile fitting requires positive polynomial_order and nsigma')
    if profile_method == 'gp' and backend == 'c':
        raise ValueError('GP profile fitting requires the Python backend')
    if gp_options is not None and profile_method != 'gp':
        raise ValueError('gp_options requires profile_method="gp"')
    Marsh = _select_backend('Marsh', 'python' if profile_method == 'gp' and backend == 'auto' else backend)
    if Marsh is None:
        P = _profiles.fit_marsh_profile(data, centroids, aperture_radius, ron, gain, nsigma,
            polynomial_spacing, polynomial_order, min_column, max_column,
            data_variance, profile_method, gp_options)
        return P.ravel() if return_flat else P

    # Prepare inputs to the algorithm:
    flattened_data = data.flatten().astype('double')
    nrows, ncolumns = data.shape
    ncentroids = len(centroids)

    if min_column is None:

        min_column = 0

    if max_column is None:
        
        max_column = ncolumns

    # Calculate the light fractions (P's):
    if data_variance is None:

        flattened_P = Marsh.ObtainP(flattened_data,
                                    centroids,
                                    nrows,
                                    ncolumns,
                                    ncentroids,
                                    aperture_radius,
                                    ron,
                                    gain,
                                    nsigma,
                                    polynomial_spacing,
                                    polynomial_order,
                                    0,
                                    min_column,
                                    max_column
                                   )

    else:

        flat_ones_array = np.ones(nrows * ncolumns).astype('double')
        flattened_variance = data_variance.flatten().astype('double')
        flattened_P = Marsh.SObtainP(flattened_data,
                                    flat_ones_array,
                                    flattened_variance,
                                    centroids,
                                    nrows,
                                    ncolumns,
                                    ncentroids,
                                    aperture_radius,
                                    ron,
                                    gain,
                                    nsigma,
                                    polynomial_spacing,
                                    polynomial_order,
                                    0,  
                                    min_column,
                                    max_column
                                   )  

    # Obtain the P's back:
    P = np.asarray(flattened_P).astype('double')

    if not return_flat:

        P.resize(nrows, ncolumns)

    # Return light fractions:
    return P

def getOptimalSpectrum(data, centroids, aperture_radius, ron, gain, nsigma, polynomial_spacing, polynomial_order, min_column = None, max_column = None, P = None, return_P = False, data_variance = None, backend='auto', profile_method='polynomial', gp_options=None):

    """
    Given a 2D-spectrum, this function returns the optimal extracted spectrum using the algorithm detailed in Marsh (1989, PASP 101, 1032). 
    By default, this function calculates the light fractions individually for each spectrum; if you have a pre-computed one (e.g., 
    obtained with the `getP` function), you can ingest that instead which will significantly speed-up the algorithm when ran on 
    several spectra. 
 
    Parameters
    ----------
    data : numpy.array
        Array containing the 2D-spectrum. 

    centroids : numpy.array
        Array containing the centroids at each column of the spectra

    aperture_radius : double
        Aperture radius (measured from the center).

    ron : double 
        Read-out-noise of the detector, in electrons.

    gain : double
        Gain of the detector, in electrons/ADU.

    nsigma : double
        Number-of-sigmas to reject outliers.

    polynomial_spacing : double
        Pixel spacing between polynomials of Marsh's algorithm.

    polynomial_order : int
        Order of the polynomials that will be fitted to the surface in Marsh's algorithm (N in the paper). 

    min_column : int 
        (Optional) Minimum column to consider the calculation from.

    max_column : int
        (Optional) Maximum column to consider the calculation from.

    P : numpy.array
        (Optional) Array containing the 2-D light fractions (the P's --- if not ingested, will be obtained using the `getP` function).

    return_P : bool
        (Optional) If `True`, function also returns the light fractions (P's).

    data_variance : numpy.array
        (Optional) Array containing the variances of each of the points in the 2-D `data` array. If defined, the `ron` and `gain` will be ignored.

    Returns
    -------

    spectrum : numpy.array
        A 3-dimensional cube with spectrum[0,:] indicating the columns, spectrum[1,:] the optimally extracted spectra at those columns and 
        spectrum[2,:] having the *inverse* of the variance of the spectra.

    Notes
    -----
    backend, profile_method and gp_options follow getP. Supplying P bypasses
    profile fitting. Returned errors condition on P, including with a GP profile;
    they do not include GP posterior uncertainty or cross-column covariance.
    Polynomial order counts terms (maximum degree is polynomial_order - 1).
    A SharedProfile instance supplied as P uses its native-pixel extraction
    (supplied variance required), evaluating the trace without retraining.

    """

    from .shared_profile import SharedProfile
    if isinstance(P, SharedProfile):
        if backend not in ('auto', 'python'):
            raise ValueError('SharedProfile requires the Python backend')
        if data_variance is None:
            raise ValueError('SharedProfile requires supplied data_variance')
        if aperture_radius != P.aperture_radius or polynomial_spacing != P.spacing:
            raise ValueError('aperture/spacing must match the trained SharedProfile')
        if min_column is not None or max_column is not None or gp_options is not None:
            raise ValueError('SharedProfile uses its trained grid/options; crop before training')
        return P.extract(data, centroids, data_variance, nsigma=nsigma, return_P=return_P)

    data, centroids, data_variance = _extraction_inputs(
        data, centroids, data_variance, aperture_radius, polynomial_spacing,
        ron, gain, nsigma, min_column, max_column, mask_invalid=profile_method == 'gp')
    if profile_method not in ('polynomial', 'gp'):
        raise ValueError('profile_method must be polynomial or gp')
    if profile_method == 'gp' and backend == 'c':
        raise ValueError('GP profile fitting requires the Python backend')
    if gp_options is not None and profile_method != 'gp':
        raise ValueError('gp_options requires profile_method="gp"')
    Marsh = _select_backend('Marsh', 'python' if profile_method == 'gp' and backend == 'auto' else backend)
    if P is not None:
        P = np.array(P, dtype=float, copy=True)
        if P.shape != data.shape or not np.all(np.isfinite(P)) or np.any(P < 0):
            raise ValueError('P must be a finite nonnegative image matching data.shape')
    else:
        P = getP(data, centroids, aperture_radius, ron, gain, nsigma,
                 polynomial_spacing, polynomial_order, min_column, max_column,
                 data_variance=data_variance, backend=backend,
                 profile_method=profile_method, gp_options=gp_options)
    if Marsh is None:
        spectrum = _extraction.marsh_optimal(data, centroids, aperture_radius, ron, gain,
            nsigma, polynomial_spacing, P, min_column, max_column, data_variance,
            allow_missing=profile_method == 'gp')
        return (spectrum, P) if return_P else spectrum

    flattened_P = P.flatten().astype('double')

    # Prepare inputs:
    flattened_data = data.flatten().astype('double')
    nrows, ncolumns = data.shape
    ncentroids = len(centroids)

    if min_column is None:

        min_column = 0

    if max_column is None:

        max_column = ncolumns

    # Obtain extracted spectrum:
    if data_variance is None:

        flattened_spectrum, size = Marsh.ObtainSpectrum(flattened_data,
                                                        centroids,
                                                        flattened_P,
                                                        nrows,
                                                        ncolumns,
                                                        ncentroids,
                                                        aperture_radius,
                                                        ron,
                                                        gain,
                                                        polynomial_spacing,
                                                        nsigma,
                                                        min_column,
                                                        max_column
                                                       )

    else:

        flat_ones_array = np.ones(nrows * ncolumns).astype('double')
        flattened_variance = data_variance.flatten().astype('double')
        flattened_spectrum, size = Marsh.SObtainSpectrum(flattened_data,
                                                        flat_ones_array,
                                                        flattened_variance,
                                                        centroids,
                                                        flattened_P,
                                                        nrows,
                                                        ncolumns,
                                                        ncentroids,
                                                        aperture_radius,
                                                        ron,
                                                        gain,
                                                        polynomial_spacing,
                                                        nsigma,
                                                        min_column,
                                                        max_column
                                                       )
                          
    spectrum = np.asarray(flattened_spectrum) 
    spectrum.resize(3, size) 

    # Return results depending on user-input:
    if not return_P:

        return spectrum                  

    else:

        P = np.asarray(flattened_P).astype('double')
        P.resize(nrows, ncolumns)

        return spectrum, P

def getFastSimpleSpectrum(data, centroids, aperture_radius, min_column = None, max_column = None, return_aperture = False, backend='auto'):

    """
    Given a 2D-spectrum, this function returns a simple-extracted spectrum. This function is fast to 
    compute, but it doesn't calculate errors on the spectra.
 
    Parameters
    ----------
    data : numpy.array
        Array containing the 2D-spectrum. 

    centroids : numpy.array
        Array containing the centroids at each column of the spectra

    aperture_radius : double
        Aperture radius (measured from the center).

    min_column : int 
        (Optional) Minimum column to consider the calculation from.

    max_column : int
        (Optional) Maximum column to consider the calculation from.

    return_aperture : bool
        (Optional) If the spectral trace (centroids) hits edges with the aperture, algorithm will select
        a smaller aperture for the extraction. If `True`, this function returns that selected aperture.

    backend : {'auto', 'c', 'python'}
        Auto retains C when available and otherwise uses Python. The legacy
        integer aperture and fractional-edge conventions are preserved.

    """

    data, centroids, _ = _geometry.validate_image(data, centroids)
    _geometry.marsh_geometry(centroids, data.shape[0], aperture_radius, 1., min_column, max_column, simple=True)
    Marsh = _select_backend('Marsh', backend)
    if Marsh is None:
        spectrum, aperture = _extraction.marsh_box(data, centroids, aperture_radius, min_column, max_column)
        return (spectrum, aperture) if return_aperture else spectrum

    # Prepare inputs:
    flattened_data = data.flatten().astype('double')
    nrows, ncolumns = data.shape
    ncentroids = len(centroids)

    if min_column is None:

        min_column = 0

    if max_column is None:

        max_column = ncolumns

    # Generate spectrum:
    flattened_spectrum, aperture = Marsh.SimpleExtraction(flattened_data,
                                                          centroids,
                                                          nrows,
                                                          ncolumns,
                                                          ncentroids,
                                                          aperture_radius,
                                                          min_column,
                                                          max_column
                                   )      

    # Prepare outputs:
    spectrum = np.asarray(flattened_spectrum)

    if not return_aperture:

        return spectrum

    else:

        return spectrum, aperture

def getSimpleSpectrum(data, x, y, aperture_radius, background_radius=50, error_data=None, correct_bkg=False, method = 'sum', bkg_method = 'all'):
    """
    This function takes as inputs two arrays (x,y) that follow the trace,
    and returns the added flux over the defined aperture radius (and its error, if an error matrix
    is given as well), substracting in the way any background between the aperture radius and the
    background radius. The background is calculated by taking the median of the points between the
    aperture_radius and the background_radius.

    Parameters
    ----------
    data: ndarray
        Image from which the spectrum wants to be extracted
    x: ndarray
        Array with the x-axis of the trace (i.e., the columns, wavelength direction)
    y: ndarray
        Array with the y-axis of the trace (i.e., rows, spatial direction)
    aperture_radius: float
        Distance from the center of the trace at which you want to add fluxes.
    background_radius: float
        Distance from the center of the trace from which you want to calculate the background. The
        background region will be between this radius and the aperture_radius.
    error_data: ndarray
        Image with the errors of each pixel value on the data ndarray above
    correct_bkg: boolean
        If True, apply background correction. If false, ommit this.
    method : string
        Method used to perform the extraction. Default is `sum`; `average` takes the average of the non-fractional pixels 
        used to extract the spectrum. This latter one is useful if the input is a wavelength map.
    bkg_method : string
        Method for the background substraction. Currently accepts 'all' to use pixels at both sides, 'up' to use pixels "above" the spectrum and 
        'down' to use pixels "below" the spectrum.
    """

    return _extraction.box_spectrum(data, x, y, aperture_radius, background_radius=background_radius,
                                    error_data=error_data, correct_bkg=correct_bkg, method=method,
                                    bkg_method=bkg_method)
