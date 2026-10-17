"""Tracing of spectra and the cross-correlation tools it relies on.

Traces give the cross-dispersion centre of a spectrum at every detector column.
``trace_spectrum`` finds them column by column by cross-correlating each column
with a model profile (or by a flux centroid); ``refine_trace`` refines a reference
trace (e.g. from an instrument model) with a bounded double-Gaussian correlation;
``trace_fwhm`` and ``trace_motion`` measure profile widths and motion over time.
"""

import warnings

import numpy as np
from astropy.convolution import convolve_fft
from astropy.modeling.models import Gaussian1D
from scipy.interpolate import UnivariateSpline, interp1d, splrep, sproot
from scipy.ndimage import gaussian_filter1d, map_coordinates, median_filter
from scipy.optimize import fminbound, minimize_scalar

from .backends import select_backend
from .ccf import correlate


def gaussian(x, mean=0.0, sigma=1.0):
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

    norm = 1.0 / (np.sqrt(2.0 * np.pi) * sigma)

    return norm * np.exp(-((x - mean) ** 2) / (2 * sigma**2))


def double_gaussian(x, mean1=-7.9, mean2=7.9, sigma1=1.0, sigma2=1.0):
    """
    This function returns the sum of two gaussians evaluated at x. This function reproduces the
    expected separation of
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


def get_ccf_convolve(signal, function='gaussian', ccf_parameters=None):
    """
    Function that obtains the CCF by convolution using the signal, represented as the column
    of pixel from the detector, and a double gaussian kernel. This method is an improvement
    over the ` get_ccf ` below.

    Parameters
    ----------

    signal : numpy.array
        Array contain the column signal of the detector

    function : str
        Define kernel to be used; default is gaussian.

    ccf_parameters : list
        List of floats related the gaussian [mu1, sig1] or double gaussian kernel parameters [mu1,
        sig1, mu2, sig2]

    Returns
    -------
    ccf : numpy.array
        Array containing the cross-correlation function between y and the selected function

    """

    npix = len(signal)
    x = np.linspace(0, npix, npix + 1)  # must be odd

    if function == 'gaussian':
        if ccf_parameters is None:
            mu1, sig1 = [0.0, 1.0]

        else:
            mu1, sig1 = ccf_parameters

        g1 = Gaussian1D(amplitude=1, mean=mu1 + npix / 2, stddev=sig1)

        kernel = g1

    elif function == 'double gaussian':
        if ccf_parameters is None:
            mu1, sig1, mu2, sig2 = [-7.5, 3.0, 7.5, 3.0]

        else:
            mu1, sig1, mu2, sig2 = ccf_parameters

        g1 = Gaussian1D(amplitude=1, mean=mu1 + npix / 2, stddev=sig1)
        g2 = Gaussian1D(amplitude=1, mean=mu2 + npix / 2, stddev=sig2)

        kernel = g1 + g2

    # convolve kernel and signal
    ccf = convolve_fft(signal, kernel(x))

    return ccf


def get_pm_ccf(x1, y1, x2, y2, max_shift=5, delta=0.1, method='abs'):
    """
    This function gets you the "poor man's CCF" of a pair of arrays. x1 and x2 might be time or
    wavelength,
    y1 and y2 might be an observable (e.g., flux). The function returns the lags and the "CCF",
    which is really
    the absolute value of the sum of the substraction between the y1's minus the shifted y2's, with
    both interpolated
    to a common grid. This can also return the standard CCF if method = 'standard'.

    Parameters
    ----------

    x1 : numpy.array
        Input array containing the values at which each of the y1-values are defined.
    y1 : numpy.array
        Array containing the input values for x1.
    x2 : numpy.array
        Input array containing the values at which each of the y2-values are defined.
    y2 : numpy.array
        Array containing the input values for x2.
    max_shift : double
        Maximum value by which x2 will be shifted to match (x1, y1)
    delta : double
        Steps at which max_shift will be tried.
    method : string
        Set to 'abs' by default, which calculates the absolute difference between the shifted
        arrays.
        To return the standard CCF (assuming data has been normalized), set to 'standard'.
        Boolean deciding whether to apply pixelation effects (i.e., integrating function over a
        pixel)

    Returns
    -------

    shifts : Values at which y2 has been shifted.

    ccf : numpy.array
        Array containing the (abs or standard) cross-correlation function between y1 and y2.

    """
    shifts = np.arange(-max_shift, max_shift, delta)
    residuals = np.zeros(len(shifts))

    # Pad edges with zeroes:
    new_x1 = np.append(np.min(np.append(x1, x2)) - max_shift * 2, x1)
    new_x1 = np.append(new_x1, np.max(np.append(x1, x2)) + max_shift * 2)
    new_x2 = np.append(np.min(np.append(x1, x2)) - max_shift * 2, x2)
    new_x2 = np.append(new_x2, np.max(np.append(x1, x2)) + max_shift * 2)

    new_y1 = np.append(y1, 0.0)
    new_y1 = np.append(0.0, new_y1)

    new_y2 = np.append(y2, 0.0)
    new_y2 = np.append(0.0, new_y2)

    # Interpoalte both series:
    f1 = interp1d(new_x1, new_y1)
    f2 = interp1d(new_x2, new_y2)

    # Evaluate absolute difference on a common grid:
    for i in range(len(shifts)):
        if method == 'abs':
            residuals[i] = np.sum(np.abs(f1(x2) - f2(x2 - shifts[i])))

        elif method == 'standard':
            residuals[i] = np.sum(f1(x2) * f2(x2 - shifts[i]))

    return shifts, residuals


def get_ccf(
    x, y, function='gaussian', parameters=None, pixelation=False, lag_step=0.001, backend='auto'
):
    """
    Function that obtains the CCF between input data defined between x and y and a pre-defined
    function.

    Parameters
    ----------

    x : numpy.array
        Array containing the values at which each of the y-values are defined.
    y : numpy.array
        Array containing the input values.
    function : string or function
        String containing the function against which the CCF wants to be computed. Default is
        'gaussian'; can also be 'double gaussian'. Alternatively it can be
        a function of choice that needs to be able to be evaluated at `x`.
    parameters : list
        Parameters for the input function. For the gaussian, the first item identifies the mean, the
        second the standard deviation. For the double gaussian, the first two
        are the mean and standard deviation of the first gaussian, the last two are the mean and
        standard deviation of the second gaussian. Default is None, in which case
        the 'gaussian' is set to mean 0 and standard deviaton of 1, and for the 'double gaussian',
        the standard deviations are also 1, but the mean of the first and second
        gaussians are set to -7.9 and +7.9 --- these are consistent with the distance between SOSS'
        horns.
    pixelation : bool
        Boolean deciding whether to apply pixelation effects (i.e., integrating function over a
        pixel)
    lag_step : double
        Steps used to lag the `function` along all the input x-values. Default is 0.001.
    backend : {'auto', 'c', 'python'}
        Auto retains C when available and otherwise uses NumPy. Built-in Python
        profiles use the historical C pi approximation and lag convention.


    Returns
    -------
    ccf : numpy.array
        Array containing the cross-correlation function between y and the selected function

    """

    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if (
        x.ndim != 1
        or y.shape != x.shape
        or x.size == 0
        or not np.all(np.isfinite(x))
        or not np.all(np.isfinite(y))
    ):
        raise ValueError('x and y must be matching nonempty finite vectors')
    if not np.isfinite(lag_step) or lag_step <= 0:
        raise ValueError('lag_step must be finite and positive')
    CCF = select_backend('CCF', backend)
    if CCF is None:
        return correlate(x, y, function, parameters, lag_step)

    # Create array of lags:
    lags = np.arange(np.min(x), np.max(x), lag_step)

    # Define which functions to use. All are coded in C (see c-codes/Utilities/CCF.c)
    if type(function) is str:
        if function == 'gaussian':
            if parameters is None:
                mean, sigma = 0.0, 1.0

            else:
                mean, sigma = parameters

            ccf = CCF.Gaussian(
                x.astype('double'),
                y.astype('double'),
                lags.astype('double'),
                len(x),
                len(lags),
                mean,
                sigma,
            )

        elif function == 'double gaussian':
            if parameters is None:
                mean1, sigma1, mean2, sigma2 = -7.9, 1.0, 7.9, 1.0

            else:
                mean1, sigma1, mean2, sigma2 = parameters

            ccf = CCF.DoubleGaussian(
                x.astype('double'),
                y.astype('double'),
                lags.astype('double'),
                len(x),
                len(lags),
                mean1,
                sigma1,
                mean2,
                sigma2,
            )

        else:
            raise Exception(
                'Function '
                + function
                + ' not available for CCF. Try "gaussian", "double gaussian" or define your '
                'own function as input.'
            )

    else:
        # Create matrix of dimensions [len(lags), len(x)]; each row contains x - lags[i]:
        all_lags = np.tile(x.astype('double'), (len(lags), 1))
        all_lags = (all_lags.transpose() - lags).transpose()

        # Evaluate input function at those lags:
        evaluated_function = function(all_lags)

        # Compute CCF in C:
        evaluated_function = np.asarray(evaluated_function, dtype='double')
        if evaluated_function.shape != all_lags.shape:
            raise ValueError('CCF callable must preserve the input array shape')
        ccf = CCF.AnyFunction(y.astype('double'), evaluated_function.flatten(), len(x), len(lags))

    return lags, ccf


def trace_spectrum(
    image,
    dqflags,
    xstart,
    ystart,
    profile_radius=20,
    correct_outliers=False,
    nsigma=100,
    median_filter_radius=5,
    method='ccf',
    ccf_function='gaussian',
    ccf_parameters=None,
    ccf_step=0.001,
    gaussian_filter=False,
    gauss_filter_width=10,
    xend=None,
    y_tolerance=2,
    verbose=False,
):
    """
    Function that non-parametrically traces spectra. There are various methods to trace the spectra.
    The default method is `ccf`, which performs cross-correlation
    to find the trace positions given a user-specified function (default is 'gaussian'; can also be
    'double gaussian' or a user-specified function). Tracing happens from columns
    `xstart` until `xend` --- default for `xend` is `0`.

    Parameters
    ----------

    image: numpy.array
        The image that wants to be traced.
    dqflags: ndarray
        The data quality flags for each pixel in the image. Only pixels with DQ flags of zero will
        be used
        in the tracing.
    xstart: float
        The x-position (column) on which the tracing algorithm will be started
    ystart: float
        The estimated y-position (row) of the center of the trace. An estimate within a few pixels
        is enough (defined by y_tolerance).
    profile_radius: float
        Expected radius of the profile measured from its center. Only this region will be used to
        estimate
        the trace position of the spectrum.
    correct_outliers : bool
        Decide if to correct outliers or not on each column. If True, outliers are detected via a
        median filter.
    nsigma : float
        Median filters are applied to each column in search of outliers if `correct_outliers` is
        `True`. `nsigma` defines
        how many n-sigma above the noise level the residuals of the median filter and the image
        should be considered outliers.
    median_filter_radius : int
        Radius of the median filter in case `correct_outliers` is `True`. Needs to be an odd number.
        Default is `5`.
    method : string
        Method by which the tracing is expected to happen. Default is `ccf`; can also be `centroid`,
        which will use the centroid of each column
        to estimate the center of the trace.
    ccf_function : string or function
        Function to cross-correlate cross-dispersion profiles against. Default is `gaussian` (useful
        for most instruments) --- can also be `double gaussian` (useful for
        e.g., NIRISS/SOSS --- double gaussian separation tailored to that instrument).
        Alternatively, a function can be passed directly --- this function needs to be
        evaluated at a set of arrays `x`, and be centered at `x=0`.
    ccf_parameters : list
        Parameters of the function against which data will be CCF'ed. For details, see the get_ccf
        function; default is None, which defaults to the get_ccf defaults.
    ccf_step : double
        Step at which the CCF will run. The smallest, the most accurate, but also the slower the CCF
        method is. Default is `0.001`.
    gaussian_filter : bool
        Flag that defines if each column will be convolved with a gaussian filter (good to smooth
        profile to match a gaussian better). Default is `False`.
    gauss_filter_width : float
        Width of the gaussian filter used to perform the centroiding of the first column, if
        `gaussian_filter` is `True`.
    xend: int
        x-position at which tracing ends. If none, trace all the columns left to xstart.
    y_tolerance: float
        When tracing, if the difference between the two difference traces at two contiguous columns
        is larger than this,
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
        indexes = range(len(x))[::-1]
        direction = 'left'

    # Define y-axis:
    y = np.arange(image.shape[0])

    # Define status of good/bad for each trace position:
    status = np.full(len(x), True, dtype=bool)

    # Define array that will save trace at each x:
    ytraces = np.zeros(len(x))

    first_time = True
    previous_trace = ystart
    for i in indexes:
        xcurrent = x[i]

        # Perform median filter to identify nasty (i.e., cosmic rays) outliers in the column:
        mf = median_filter(image[:, xcurrent], size=median_filter_radius)

        if correct_outliers:
            residuals = mf - image[:, xcurrent]
            mad_sigma = get_mad_sigma(residuals)
            column_nsigma = np.abs(residuals) / mad_sigma

        else:
            column_nsigma = np.zeros(image.shape[0]) * nsigma

        # Extract data-quality flags for current column; index good pixels --- mask nans as well:
        idx_good = np.where(
            (dqflags[:, xcurrent] == 0) & (~np.isnan(image[:, xcurrent]) & (column_nsigma < nsigma))
        )[0]
        idx_bad = np.where(
            ~(
                (dqflags[:, xcurrent] == 0)
                & (~np.isnan(image[:, xcurrent]) & (column_nsigma < nsigma))
            )
        )[0]

        if len(idx_good) > 0:
            # Replace bad values with the ones in the median filter:
            column_data = np.copy(image[:, xcurrent])
            column_data[idx_bad] = mf[idx_bad]

            if gaussian_filter:
                # Convolve column with a gaussian filter; remove median before convolving:
                filtered_column = gaussian_filter1d(
                    column_data - np.median(column_data), gauss_filter_width
                )

            else:
                filtered_column = column_data - np.median(column_data)

            # Find trace depending on the method, only within pixels close to profile_radius:
            idx = np.where(np.abs(y - ystart) < profile_radius)[0]
            if method == 'ccf':
                # Run CCF search using only the pixels within profile_radius:
                lags, ccf = get_ccf(
                    y[idx],
                    filtered_column[idx],
                    function=ccf_function,
                    parameters=ccf_parameters,
                    lag_step=ccf_step,
                )
                idx_max = np.where(ccf == np.max(ccf))[0]

                ytraces[i] = lags[idx_max]

            elif method == 'convolve':
                # Run CCF using using convolution method applied to entire column
                ccf = get_ccf_convolve(
                    filtered_column, function=ccf_function, ccf_parameters=ccf_parameters
                )

                # Find where the minimum occurs within the profile_radius
                # by finding the peak. A cubic spline interpolations is
                # used to smooth the ccf. The interpolated ccf has been
                # compared to numerically computed ccf with fine sampling
                # and yield similar results. By applying a miminization
                # scheme we can idenitfy the optimal position where the peak
                # occurs in the region of interest
                #
                y1, y2 = y[idx][[0, -1]]
                ytraces[i] = fminbound(interp1d(y, -1 * ccf, kind='cubic'), y1, y2)

            elif method == 'centroid':
                # Find pixel centroid within profile_radius pixels of the initial guess:
                ytraces[i] = np.sum(y[idx] * filtered_column[idx]) / np.sum(filtered_column[idx])

            else:
                raise Exception(
                    'Cannot trace spectra with method "'
                    + method
                    + '": method not recognized. Available methods are "ccf" and "centroid"'
                )

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

            if difference > y_tolerance:
                if verbose:
                    print(
                        'Tracing failed at column',
                        xcurrent,
                        '; estimated trace position:',
                        ytraces[i],
                        ', previous one:',
                        previous_trace,
                        '> than tolerance: ',
                        y_tolerance,
                        '. Replacing with closest good trace position.',
                    )

                ytraces[i] = previous_trace

            ystart = ytraces[i]

        else:
            print(xcurrent, 'is a bad column. Setting to previous trace position:')
            ytraces[i] = ystart
            status[i] = True

    # Return all trace positions:
    return x, ytraces


def get_fwhm(x, y, k=3):
    """
    Given a profile (x,y), this function calculates the FWHM of the profile via spline
    interpolation. The idea is to interpolate the profile, and find where the profile
    minus half the maximum crosses zero via root-finding.

    Parameters
    ----------

    x : numpy.array
        Array containing the x-axis of the profile (e.g., pixels).
    y : numpy.array
        Array containing the y-axis of the profile (e.g., counts).
    k : int
        Interpolation degree

    Returns
    -------

    The FWHM (a double) if there is a single root; 0 if it doesn't exist/there are more than one
    root.
    """

    half_max = np.max(y) / 2.0
    s = splrep(x, y - half_max, k=k)
    roots = sproot(s)

    if len(roots) > 2:
        return 0.0

    elif len(roots) < 2:
        return 0.0
    else:
        return abs(roots[1] - roots[0])


def trace_fwhm(tso, x, y, distance_from_trace=15):
    """
    This function calculates the FWHM time-series accross a set of frames (the `tso` array), as a
    function of position/wavelength (the trace position
    `x` and `y`).

    Parameters
    ----------

    tso : numpy.array
          Array of shape `(time, rows, columns)` of a set of frames on which one wants to calculate
          the FWHM as a function of `time` and `columns`.
    x : numpy.array
          Trace column position.
    y : int
          Trace row position.
    distance_from_trace : int
          Distance from the trace at which the FWHM will be calculated.

    Returns
    -------

    fwhms : numpy.array
          Array of shape `(time, columns)` having the FWHM at each of the `x` columns and `times`.

    superfwhm : numpy.array
          The median FWHM accross all `x` columns
    """

    fwhms = np.zeros([tso.shape[0], tso.shape[2]])

    row_index = np.arange(tso.shape[1])
    min_row = 0
    max_row = tso.shape[1] - 1

    for i in range(tso.shape[0]):
        for j in range(tso.shape[2]):
            if j in x:
                idx = np.where(j == x)[0]
                integer_row = int(y[idx])

                lower_row = np.max([min_row, integer_row - distance_from_trace])
                upper_row = np.min([max_row, integer_row + distance_from_trace])

                fwhms[i, j] = get_fwhm(
                    row_index[lower_row:upper_row], tso[i, lower_row:upper_row, j]
                )

    normalized_fwhms = np.zeros(fwhms.shape)

    # Nan the zeroes in fwhms:
    idx = np.where(fwhms == 0.0)
    fwhms[idx] = np.nan

    for i in range(tso.shape[2]):
        # Normalize:
        normalized_fwhms[:, i] = fwhms[:, i] - np.nanmedian(fwhms[:, i])

    super_fwhm = np.nanmedian(normalized_fwhms, axis=1)

    return fwhms, super_fwhm


def get_mad_sigma(x):

    x_median = np.nanmedian(x)

    return 1.4826 * np.nanmedian(np.abs(x - x_median))


def refine_trace(image, columns, reference, *, valid=None, max_shift=4.0, step=8):
    """Bounded, normalized double-Gaussian correlation and a smooth offset spline.

    Uses the notebook's peaks at +/-7.5 pixels, sigma=3. Missing pixels do not
    contribute to the correlation. The bounded search cannot jump to another
    spectral order. Low-quality/boundary solutions retain the reference locally.
    """
    if max_shift <= 0 or step < 1:
        raise ValueError('Trace search width and sampling step must be positive')
    valid = np.isfinite(image) if valid is None else valid & np.isfinite(image)
    offsets = np.linspace(-max_shift, max_shift, 81)
    dx, dy = [], []
    rows = np.arange(image.shape[0])
    for j in range(0, len(columns), step):
        x, center = columns[j], reference[j]
        keep = valid[:, x] & (np.abs(rows - center) < 22 + max_shift)
        if keep.sum() < 25:
            continue
        signal = image[keep, x].astype(float)
        signal -= np.mean(signal)
        norm = np.linalg.norm(signal)
        if norm == 0:
            continue
        pos = rows[keep][None, :] - center - offsets[:, None]
        template = np.exp(-0.5 * ((pos - 7.5) / 3.0) ** 2) + np.exp(-0.5 * ((pos + 7.5) / 3.0) ** 2)
        template -= template.mean(axis=1)[:, None]
        score = template @ signal / (np.linalg.norm(template, axis=1) * norm)
        k = np.argmax(score)
        if k in (0, len(offsets) - 1) or score[k] < 0.5:
            continue
        curvature = score[k - 1] - 2 * score[k] + score[k + 1]
        correction = 0.5 * (score[k - 1] - score[k + 1]) / curvature if curvature < 0 else 0.0
        dx.append(x)
        dy.append(offsets[k] + np.clip(correction, -0.5, 0.5) * (offsets[1] - offsets[0]))
    if len(dx) < 5:
        raise ValueError('Too few reliable double-Gaussian trace measurements')
    dx, dy = np.asarray(dx), np.asarray(dy)
    smooth = median_filter(dy, size=7, mode='nearest')
    good = np.abs(dy - smooth) < max(0.25, 5 * 1.4826 * np.median(np.abs(dy - smooth)))
    if good.sum() < 5:
        raise ValueError('Trace refinement rejected too many measurements')
    spline = UnivariateSpline(dx[good], smooth[good], s=0.01 * good.sum(), ext=3)
    shift = np.clip(spline(columns), -max_shift, max_shift)
    return reference + shift, dict(
        sample_columns=dx, sample_offsets=dy, accepted=good, fitted_offset=shift
    )


def trace_motion(image, valid, columns, centers, template=None):
    """Measure cross-dispersion motion and profile second-moment width.

    The width is a moment diagnostic, explicitly not Gaussian FWHM for the
    double-peaked SOSS PSF. It is never used as a wavelength calibration.
    """
    indices = np.linspace(0, len(columns) - 1, min(64, len(columns))).astype(int)
    offset = np.arange(-23.0, 24.0)
    yy = centers[indices][None, :] + offset[:, None]
    xx = np.broadcast_to(columns[indices], yy.shape)
    values = map_coordinates(
        np.where(valid, image, np.nan), [yy, xx], order=1, mode='constant', cval=np.nan
    )
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        values /= np.nanmedian(values, axis=0)[None, :]
        profile = np.nanmedian(values, axis=1)
    if np.sum(np.isfinite(profile)) < 30:
        return np.nan, np.nan, profile
    profile = np.interp(offset, offset[np.isfinite(profile)], profile[np.isfinite(profile)])
    profile -= np.median(np.r_[profile[:3], profile[-3:]])
    profile /= max(np.linalg.norm(profile), 1e-20)
    shift = 0.0
    if template is not None:

        def objective(d):
            return np.sum((profile - np.interp(offset - d, offset, template)) ** 2)

        fit = minimize_scalar(objective, bounds=(-3.0, 3.0), method='bounded')
        shift = fit.x if fit.success and abs(fit.x) < 2.95 else np.nan
    positive = np.maximum(profile, 0.0)
    center = np.sum(offset * positive) / positive.sum()
    width = np.sqrt(np.sum((offset - center) ** 2 * positive) / positive.sum())
    return float(shift), float(width), profile
