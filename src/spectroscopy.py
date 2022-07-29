import numpy as np
from scipy.ndimage import median_filter
from scipy.ndimage import gaussian_filter1d

import Marsh
import CCF

def getP(data, centroids, aperture_radius, ron, gain, nsigma, polynomial_spacing, polynomial_order, min_column = None, max_column = None, return_flat = False, data_variance = None):
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

    data_variance : numpy.array
        (Optional) Array containing the variances of each of the points in the 2-D `data` array. If defined, the `ron` and `gain` will be ignored.

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

def getOptimalSpectrum(data, centroids, aperture_radius, ron, gain, nsigma, polynomial_spacing, polynomial_order, min_column = None, max_column = None, P = None, return_P = False, data_variance = None):

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

    data_variance : numpy.array
        (Optional) Array containing the variances of each of the points in the 2-D `data` array. If defined, the `ron` and `gain` will be ignored.

    Returns
    -------

    spectrum : numpy.array
        A 3-dimensional cube with spectrum[0,:] indicating the columns, spectrum[1,:] the optimally extracted spectra at those columns and 
        spectrum[2,:] having the *inverse* of the variance of the spectra.

    """

    if P is not None:

        flattened_P = P.flatten().astype('double')

    else:

        if data_variance is None:

            flattened_P = getP(data, centroids, aperture_radius, ron, gain, nsigma, polynomial_spacing, polynomial_order, 
                               min_column = min_column, max_column = max_column, return_flat = True)

        else:

            flattened_P = getP(data, centroids, aperture_radius, ron, gain, nsigma, polynomial_spacing, polynomial_order,
                               min_column = min_column, max_column = max_column, return_flat = True, data_variance = data_variance)

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

            if bkg_method == 'all':

                bkg = np.median(np.append(bkg_left, bkg_right))

            elif bkg_method == 'up':

                bkg = np.median(bkg_right)

            elif bkg_method == 'down':

                bkg = np.median(bkg_left)

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

def gaussian(x, mean = 0., sigma = 1.):
    """
    This function returns a gaussian evaluated at x

    Parameters
    ----------

    x : numpy.array
        Array containing where the gaussian will be evaluated
    mean : double
        Mean of the gaussian.
    sigma : double
        Standard-deviation of the gaussian

    Returns
    -------
    Gaussian evaluated at `x`.

    """

    norm = 1. / ( np.sqrt(2. * np.pi) * sigma )

    return norm * np.exp( - ( (x - mean)**2 ) / (2 * sigma**2) )

def double_gaussian(x, mean1 = -7.9, mean2 = 7.9, sigma1 = 1., sigma2 = 1.):
    """
    This function returns the sum of two gaussians evaluated at x. This function reproduces the expected separation of 
    the "horns" in a NIRISS/SOSS profile.

    Parameters
    ----------

    x : numpy.array
        Array containing where the gaussian will be evaluated
    mean1 : double
        Mean of the first gaussian.
    mean2 : double  
        Mean of the second gaussian.
    sigma1 : double
        Standard-deviation of the first gaussian.
    sigma2 : double
        Standard-deviation of the second gaussian.

    Returns
    -------
    Double gaussian evaluated at `x`.
    """

    return gaussian(x, mean1, sigma1) + gaussian(x, mean2, sigma2)

def get_ccf(x, y, function = 'gaussian', parameters = None, pixelation = False, lag_step = 0.001):
    """
    Function that obtains the CCF between input data defined between x and y and a pre-defined function.

    Parameters
    ----------

    x : numpy.array
        Array containing the values at which each of the y-values are defined.
    y : numpy.array
        Array containing the input values.
    function : string or function
        String containing the function against which the CCF wants to be computed. Default is 'gaussian'; can also be 'double gaussian'. Alternatively it can be 
        a function of choice that needs to be able to be evaluated at `x`.
    parameters : list
        Parameters for the input function. For the gaussian, the first item identifies the mean, the second the standard deviation. For the double gaussian, the first two 
        are the mean and standard deviation of the first gaussian, the last two are the mean and standard deviation of the second gaussian. Default is None, in which case 
        the 'gaussian' is set to mean 0 and standard deviaton of 1, and for the 'double gaussian', the standard deviations are also 1, but the mean of the first and second 
        gaussians are set to -7.9 and +7.9 --- these are consistent with the distance between SOSS' horns.
    pixelation : bool
        Boolean deciding whether to apply pixelation effects (i.e., integrating function over a pixel)
    lag_step : double
        Steps used to lag the `function` along all the input x-values. Default is 0.001.
        

    Returns
    -------
    ccf : numpy.array
        Array containing the cross-correlation function between y and the selected function

    """

    # Create array of lags:
    lags = np.arange(np.min(x), np.max(x), lag_step) 

    # Define which functions to use. All are coded in C (see c-codes/Utilities/CCF.c)
    if type(function) is str:

        if function == 'gaussian':

            if parameters is None:
            
                mean, sigma = 0., 1.

            else:

                mean, sigma = parameters

            ccf = CCF.Gaussian(x.astype('double'), y.astype('double'), lags.astype('double'), len(x), len(lags), mean, sigma)

        elif function == 'double gaussian':

            if parameters is None:
        
                mean1, sigma1, mean2, sigma2 = -7.9, 1., 7.9, 1.

            else:

                mean1, sigma1, mean2, sigma2 = parameters

            ccf = CCF.DoubleGaussian(x.astype('double'), y.astype('double'), lags.astype('double'), len(x), len(lags), mean1, sigma1, mean2, sigma2)

        else:
        
            raise Exception('Function '+function+' not available for CCF. Try "gaussian", "double gaussian" or define your own function as input.')

    else:

        # Create matrix of dimensions [len(lags), len(x)]; each row contains x - lags[i]:
        all_lags = np.tile( x.astype('double'), (len(lags), 1) )
        all_lags = (all_lags.transpose() - lags).transpose()

        # Evaluate input function at those lags:
        evaluated_function = function(all_lags)

        # Compute CCF in C:
        ccf = CCF.AnyFunction(y.astype('double'), evaluated_function.flatten(), len(x), len(lags))

    return lags, ccf 

def trace_spectrum(image, dqflags, xstart, ystart, profile_radius=20, correct_outliers = False, nsigma = 100, median_filter_radius = 5, method = 'ccf', ccf_function = 'gaussian', ccf_parameters = None, ccf_step = 0.001, gaussian_filter = False, gauss_filter_width=10, xend=None, y_tolerance = 2, verbose = False):
    """
    Function that non-parametrically traces spectra. There are various methods to trace the spectra. The default method is `ccf`, which performs cross-correlation 
    to find the trace positions given a user-specified function (default is 'gaussian'; can also be 'double gaussian' or a user-specified function). Tracing happens from columns 
    `xstart` until `xend` --- default for `xend` is `0`.
 
    Parameters
    ----------

    image: numpy.array
        The image that wants to be traced.
    dqflags: ndarray
        The data quality flags for each pixel in the image. Only pixels with DQ flags of zero will be used 
        in the tracing.
    xstart: float
        The x-position (column) on which the tracing algorithm will be started
    ystart: float
        The estimated y-position (row) of the center of the trace. An estimate within a few pixels is enough (defined by y_tolerance).
    profile_radius: float
        Expected radius of the profile measured from its center. Only this region will be used to estimate 
        the trace position of the spectrum.
    correct_outliers : bool
        Decide if to correct outliers or not on each column. If True, outliers are detected via a median filter.
    nsigma : float
        Median filters are applied to each column in search of outliers if `correct_outliers` is `True`. `nsigma` defines 
        how many n-sigma above the noise level the residuals of the median filter and the image should be considered outliers. 
    median_filter_radius : int
        Radius of the median filter in case `correct_outliers` is `True`. Needs to be an odd number. Default is `5`.
    method : string
        Method by which the tracing is expected to happen. Default is `ccf`; can also be `centroid`, which will use the centroid of each column 
        to estimate the center of the trace.
    ccf_function : string or function
        Function to cross-correlate cross-dispersion profiles against. Default is `gaussian` (useful for most instruments) --- can also be `double gaussian` (useful for 
        e.g., NIRISS/SOSS --- double gaussian separation tailored to that instrument). Alternatively, a function can be passed directly --- this function needs to be 
        evaluated at a set of arrays `x`, and be centered at `x=0`.
    ccf_parameters : list
        Parameters of the function against which data will be CCF'ed. For details, see the get_ccf function; default is None, which defaults to the get_ccf defaults.
    ccf_step : double
        Step at which the CCF will run. The smallest, the most accurate, but also the slower the CCF method is. Default is `0.001`.
    gaussian_filter : bool
        Flag that defines if each column will be convolved with a gaussian filter (good to smooth profile to match a gaussian better). Default is `False`.
    gauss_filter_width : float
        Width of the gaussian filter used to perform the centroiding of the first column, if `gaussian_filter` is `True`.
    xend: int
        x-position at which tracing ends. If none, trace all the columns left to xstart.
    y_tolerance: float
        When tracing, if the difference between the two difference traces at two contiguous columns is larger than this, 
        then assume tracing failed (e.g., cosmic ray).
    verbose: boolean
        If True, print error messages.

    Returns
    -------

    x : numpy.array
        Columns at which the trace position is being calculated.
    y : numpy.array
        Estimated trace position.
    """
    
    # Define x-axis:
    if xend is not None:

        if xend < xstart:

            x = np.arange(xend, xstart + 1)
            indexes = range(len(x))[::-1]
            direction = 'left'

        else:

            x = np.arange(xstart, xend + 1)
            indexes = range(len(x))
            direction = 'right'

    else:

        x = np.arange(0, xstart + 1)
        
    # Define y-axis:
    y = np.arange(image.shape[0])
    
    # Define status of good/bad for each trace position:
    status = np.full(len(x), True, dtype=bool)
   
    # Define array that will save trace at each x:
    ytraces = np.zeros(len(x))
   
    first_time = True 
    for i in indexes:

        xcurrent = x[i]

        # Perform median filter to identify nasty (i.e., cosmic rays) outliers in the column:
        mf = median_filter(image[:,xcurrent], size = median_filter_radius)

        if correct_outliers:
      
            residuals = mf - image[:,xcurrent]
            mad_sigma = get_mad_sigma(residuals)
            column_nsigma = np.abs(residuals) / mad_sigma

        else:

            column_nsigma = np.zeros(image.shape[0]) * nsigma
        
        # Extract data-quality flags for current column; index good pixels --- mask nans as well:
        idx_good = np.where((dqflags[:, xcurrent] == 0) & (~np.isnan(image[:, xcurrent]) & (column_nsigma < nsigma)))[0]        
        idx_bad = np.where(~((dqflags[:, xcurrent] == 0) & (~np.isnan(image[:, xcurrent]) & (column_nsigma < nsigma))))[0]
       
        if len(idx_good) > 0:

            # Replace bad values with the ones in the median filter:
            column_data = np.copy(image[:, xcurrent])
            column_data[idx_bad] = mf[idx_bad]

            if gaussian_filter:

                # Convolve column with a gaussian filter; remove median before convolving:
                filtered_column = gaussian_filter1d(column_data - \
                                                    np.median(column_data), gauss_filter_width)

            else:

                filtered_column = column_data - np.median(column_data)
    
            # Find trace depending on the method, only within pixels close to profile_radius:
            idx = np.where(np.abs(y - ystart) < profile_radius)[0]
            if method == 'ccf':

                # Run CCF search using only the pixels within profile_radius:
                lags, ccf = get_ccf(y[idx], filtered_column[idx], function = ccf_function, parameters = ccf_parameters, lag_step = ccf_step)
                idx_max = np.where(ccf == np.max(ccf))[0]
                
                ytraces[i] = lags[idx_max]

            elif method == 'centroid':
 
                # Find pixel centroid within profile_radius pixels of the initial guess:
                ytraces[i] = np.sum(y[idx] * filtered_column[idx]) / np.sum(filtered_column[idx])

            else:
        
                raise Exception('Cannot trace spectra with method "'+method+'": method not recognized. Available methods are "ccf" and "centroid"')

            # Get the difference of the current trace position with the previous one (if any):
            if not first_time:
    
                if direction == 'left':

                    previous_trace = ytraces[i + 1]

                else:

                    previous_trace = ytraces[i - 1]

            else:
                
                previous_trace = ystart
                first_time = False


            difference = np.abs(previous_trace - ytraces[i])

            if (difference > y_tolerance):

                if verbose:
                    print('Tracing failed at column',xcurrent,'; estimated trace position:',ytraces[i],', previous one:',previous_trace,'> than tolerance: ',y_tolerance,\
                          '. Replacing with closest good trace position.')

                ytraces[i] = previous_trace

            ystart = ytraces[i]

        else:

            print(xcurrent,'is a bad column. Setting to previous trace position:')
            ytraces[i] = previous_trace
            status[i] = True
    
    # Return all trace positions:
    return x, ytraces

def get_mad_sigma(x):

    x_median = np.nanmedian(x)

    return 1.4826 * np.nanmedian( np.abs(x - x_median) )
