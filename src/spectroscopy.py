import numpy as np

import Marsh

def getP(data, centroids, aperture_radius, ron, gain, nsigma, polynomial_spacing, polynomial_order, min_column = None, max_column = None, return_flat = False):
    """
    Given a 2D-spectrum, centroids over it and various properties of the noise, this function returns the light 
    fractions of a spectrum using the algorithm described in detail in Marsh (1989, PASP 101, 1032). 
   
    Attributes
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

    Returns
    -------

    P : numpy.array
        Light frations (weights) of the optimal extraction.

    """

    # Prepare inputs to the algorithm:
    flattened_data = data.flatten().astype('double')
    nrows, ncolumns = data.shape
    ncentroids = len(centroids)

    if min_column is None:

        min_column = 0

    if max_column is None:
        
        max_column = ncolumns

    # Calculate the light fractions (P's):
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

    # Obtain the P's back:
    P = np.asarray(flattened_P).astype('double')

    if not return_flat:

        P.resize(nrows, ncolumns)

    # Return light fractions:
    return P

def getOptimalSpectrum(data, centroids, aperture_radius, ron, gain, nsigma, polynomial_spacing, polynomial_order, min_column = None, max_column = None, P = None, return_P = False):

    """
    Given a 2D-spectrum, this function returns the optimal extracted spectrum using the algorithm detailed in Marsh (1989, PASP 101, 1032). 
    By default, this function calculates the light fractions individually for each spectrum; if you have a pre-computed one (e.g., 
    obtained with the `getP` function), you can ingest that instead which will significantly speed-up the algorithm when ran on 
    several spectra. 
 
    Attributes
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

    Returns
    -------

    spectrum : numpy.array
        A 3-dimensional cube with spectrum[0,:] indicating the columns, spectrum[1,:] the optimally extracted spectra at those columns and 
        spectrum[2,:] having the *inverse* of the variance of the spectra.

    """

    if P is not None:

        flattened_P = P.flatten().astype('double')

    else:

        flattened_P = getP(data, centroids, aperture_radius, ron, gain, nsigma, polynomial_spacing, polynomial_order, 
                            min_column = min_column, max_column = max_column, return_flat = True)

    # Prepare inputs:
    flattened_data = data.flatten().astype('double')
    nrows, ncolumns = data.shape
    ncentroids = len(centroids)

    if min_column is None:

        min_column = 0

    if max_column is None:

        max_column = ncolumns

    # Obtain extracted spectrum:
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
                          
    spectrum = np.asarray(flattened_spectrum) 
    spectrum.resize(3, size) 

    # Return results depending on user-input:
    if not return_P:

        return spectrum                  

    else:

        P = np.asarray(flattened_P).astype('double')
        P.resize(nrows, ncolumns)

        return spectrum, P

def getFastSimpleSpectrum(data, centroids, aperture_radius, min_column = None, max_column = None, return_aperture = False):

    """
    Given a 2D-spectrum, this function returns a simple-extracted spectrum. This function is fast to 
    compute, but it doesn't calculate errors on the spectra.
 
    Attributes
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

    """

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

def getSimpleExtraction(data, x, y, aperture_radius, background_radius=50, error_data=None, correct_bkg=True, method = 'sum'):
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
    """

    method = method.lower()

    # If average method being used, remove background correction:
    if method == 'average':
        correct_bkg = False

    # Create array that will save our fluxes:
    flux = np.zeros(len(x))
    if error_data is not None:
        flux_error = np.zeros(len(x))
    max_column = data.shape[0] - 1

    for i in range(len(x)):

        # Cut the column with which we'll be working with:
        column = data[:,int(x[i])]
        if error_data is not None:
            variance_column = error_data[:,int(x[i])]**2

        # Define limits given by the aperture_radius and background_radius variables:
        if correct_bkg:
            left_side_bkg = np.max([y[i] - background_radius, 0])
            right_side_bkg = np.min([max_column, y[i] + background_radius])
        left_side_ap = np.max([y[i] - aperture_radius, 0])
        right_side_ap = np.min([max_column, y[i] + aperture_radius])

        # Extract background, being careful with edges:
        if correct_bkg:
            bkg_left = column[np.max([0, int(left_side_bkg)]) : np.max([0, int(left_side_ap)])]
            bkg_right = column[np.min([int(right_side_ap), max_column]) : np.max([int(right_side_bkg), max_column])]
            bkg = np.median(np.append(bkg_left, bkg_right))
        else:
            bkg = 0.

        # Substract it from the column:
        column -= bkg

        # Perform aperture extraction of the background-substracted column, being careful with pixelization
        # at the edges. First, deal with left (up) side:
        l_decimal, l_integer = np.modf(left_side_ap)
        l_integer = int(l_integer)
        if l_decimal < 0.5:
            l_fraction = (0.5 - l_decimal) * column[np.min([l_integer, max_column])]
            l_limit = l_integer + 1
            if error_data is not None:
                l_fraction_variance = ((0.5 - l_decimal)**2) * variance_column[np.min([l_integer, max_column])]
        else:
            l_fraction = (1. - (l_decimal - 0.5)) * column[np.min([l_integer + 1, max_column])]
            l_limit = l_integer + 2
            if error_data is not None:
                l_fraction_variance = ((1. - (l_decimal - 0.5))**2) * variance_column[np.min([l_integer + 1, max_column])]

        # Now right (down) side:
        r_decimal, r_integer = np.modf(right_side_ap)
        r_integer = int(r_integer)
        if r_decimal < 0.5:
            r_fraction = (1. - (0.5 - r_decimal)) * column[np.min([max_column, r_integer])]
            r_limit = r_integer
            if error_data is not None:
                r_fraction_variance = ((1. - (0.5 - r_decimal))**2) * variance_column[np.min([max_column, r_integer])]
        else:
            r_fraction = (r_decimal - 0.5) * column[np.min([max_column, r_integer + 1])]
            r_limit = r_integer + 1
            if error_data is not None:
                r_fraction_variance = ((r_decimal - 0.5)**2) * variance_column[np.min([max_column, r_integer + 1])]

        # Save total flux in current column:
        if method == 'sum':
            flux[i] = l_fraction + r_fraction + np.sum(column[l_limit:r_limit])

        elif method == 'average':
            flux[i] = np.mean(column[l_limit:r_limit])

        else:
            raise Exception('Method "'+method+'" currently not supported for aperture extraction. Select either "sum" or "average".')

        if error_data is not None:
            # Total error is the sum of the variances:
            flux_error[i] = np.sqrt(np.sum(variance_column[l_limit:r_limit]) + l_fraction_variance + \
                                    r_fraction_variance)
    if error_data is not None:
        return flux, flux_error
    else:
        return flux
