"""Detector-level algorithms for time-series observations.

Pedestal and 1/f (column striping) corrections, odd/even models (ROEBA, LOOM),
masks of unilluminated pixels and a time-series jump detector. Arrays follow the
detector orientation: frames are ``(row, column)`` with columns along the slow-read
(1/f striping) direction, and ramps are ``(integration, group, row, column)``.
"""

from copy import deepcopy

import numpy as np
from scipy.ndimage import median_filter
from scipy.sparse.linalg import lsmr

from .timeseries import outlier_detector


def subtract_side_pedestal(data, left_pixels=25, right_pixels=25):
    """Subtract, in place, a pedestal estimated from the detector edge columns.

    For every integration and group of ``data`` (``(integration, group, row, column)``),
    the median of the leftmost ``left_pixels`` and rightmost ``right_pixels`` columns is
    removed from the whole group. Useful for NIRSpec/PRISM, whose reference pixels do
    not track the pedestal well.
    """
    for integration in range(data.shape[0]):
        for group in range(data.shape[1]):
            background = np.hstack(
                (
                    data[integration, group, :, :left_pixels],
                    data[integration, group, :, -right_pixels:],
                )
            )

            # Subtract pedestal from data:
            data[integration, group, :, :] -= np.nanmedian(background)

    return data


def group_1f_correct(segments, npixel=4, nsigma=10, column_window=3, row_window=20, mask_radius=10):
    """Group-level 1/f correction of the segments of a time series.

    (a) Estimate the trace centroid of every column from the median last-minus-first
    frame (2-D median-filtered to remove outliers); (b) mask ``mask_radius`` pixels
    around it; (c) subtract the median of the remaining pixels, column by column, from
    every group of every integration.

    Parameters
    ----------
    segments : list of numpy.ndarray
        Chronologically ordered ``(integration, group, row, column)`` arrays.
    npixel, nsigma : int
        Unused; kept for compatibility.
    column_window, row_window : int
        Size of the 2-D median filter applied to the median last-minus-first frame.
    mask_radius : int
        Half-width of the masked region around the trace.

    Returns
    -------
    list of numpy.ndarray
        Corrected copies of the segments.
    """
    ncolumns = segments[0].shape[3]
    corrected = [deepcopy(data) for data in segments]

    lmf = np.array([])
    for i in range(len(segments)):
        # Store the last-minus-first frames of all segments in one array:
        if i == 0:
            lmf = get_last_minus_first(segments[i], return_median=False)

        else:
            lmf = np.vstack((lmf, get_last_minus_first(segments[i], return_median=False)))

    # Median of the LMF, median-filtered to get a clean spectral shape:
    median_lmf = np.nanmedian(lmf, axis=0)
    lmf_mf = median_filter(median_lmf, [column_window, row_window])

    # Centroids, and a mask of NaNs within mask_radius pixels of them:
    nan_mask = np.ones(lmf_mf.shape)
    centroids = np.zeros(lmf_mf.shape[1])
    x = np.arange(lmf_mf.shape[0])
    for i in range(lmf_mf.shape[1]):
        centroids[i] = np.sum((x * np.abs(lmf_mf[:, i]))) / np.sum(np.abs(lmf_mf[:, i]))
        idx = np.where(np.abs(x - centroids[i]) < mask_radius)[0]
        nan_mask[idx, i] = np.nan

    # Go group by group removing the median of the unmasked pixels of each column:
    for i in range(len(segments)):
        for j in range(segments[i].shape[0]):
            for k in range(segments[i].shape[1]):
                for l in range(ncolumns):
                    corrected[i][j, k, :, l] -= np.nanmedian(
                        corrected[i][j, k, :, l] * nan_mask[:, l]
                    )

    return corrected


def tso_jump_flags(segments, window, nsigma=10):
    """Detect jumps using the time series of each pixel's group differences.

    For every pixel, the difference between groups ``k+1`` and ``k`` of every
    integration forms a time series across all segments. A median filter of width
    ``window`` is subtracted, and outliers beyond ``nsigma`` (MAD-based) are jumps,
    flagged at group ``k+1``. Assumes the same target in all segments, more than one
    integration in total and more than one group per integration.

    Parameters
    ----------
    segments : list of numpy.ndarray
        Chronologically ordered ``(integration, group, row, column)`` arrays.
    window : int
        Median-filter window; the number of integrations should be much larger.
    nsigma : float
        Outlier threshold.

    Returns
    -------
    list of numpy.ndarray
        Boolean arrays, one per segment, ``True`` where a jump was detected.
    """
    flags = [np.zeros(data.shape, dtype=bool) for data in segments]
    ips = [data.shape[0] for data in segments]
    input_nints = sum(ips)
    input_ngroups, nrows, ncolumns = segments[0].shape[1:]

    # Difference images of all segments:
    di = np.zeros([input_nints, input_ngroups - 1, nrows, ncolumns])
    cis = []  # Cumulative number of integrations up to each segment
    ci = 0
    for i in range(len(segments)):
        for j in range(input_ngroups - 1):
            di[ci : ci + ips[i], j, :, :] = segments[i][:, j + 1, :, :] - segments[i][:, j, :, :]

        ci += ips[i]
        cis.append(ci)

    # Outlier detection on each pixel's group-difference time series:
    for i in range(nrows):
        for j in range(ncolumns):
            for k in range(input_ngroups - 1):
                idx = outlier_detector(di[:, k, i, j], nsigma=nsigma, window=window)

                for outlier_index in idx:
                    # Map the outlier to its segment, and its index in that segment:
                    for ii in range(len(segments)):
                        if outlier_index - cis[ii] < 0:
                            if ii == 0:
                                flags[ii][outlier_index, k + 1, i, j] = True

                            else:
                                flags[ii][outlier_index - cis[ii - 1], k + 1, i, j] = True

                            break

    return flags


def cc_uniluminated_outliers(data, mask, nsigma=5):
    """
    Column-to-column background outlier detection

    This function goes column-by-column and detects outliers on a given frame (`data`) wherever
    there are no sources. The user needs to provide a mask where values of 1 are the
    uniluminated pixels, and 0's are the illuminated pixels. Main difference with
    `get_uniluminated_mask` is that this function gets you hot pixels, cosmic rays and bad pixels.

    Parameters
    ----------

    data : numpy.array
        Numpy array of dimensions (npixel, npixel). It is assumed columns go in the slow-direction
        (i.e., 1/f striping direction) and rows go
        in the fast-read direction (i.e., odd-even effect direction).

    mask : numpy.array
        Numpy array of the same length as `data`; pixels that should be included in the calculation
        (expected to be non-iluminated by the main source)
        should be set to 1 --- the rest should be zeros

    Returns
    -------

    updated_mask : numpy.array
        Combination of the input mask with the outliers in the background, which are identified with
        zeroes.

    """

    # Turn all zeroes in the mask to nans:
    nan_data = np.copy(data)
    nan_data[mask == 0] = np.nan

    # Compute column medians:
    column_medians = np.nanmedian(nan_data, axis=0)

    # Compute column median-absolute deviation:
    column_mads = np.nanmedian(np.abs(nan_data - column_medians), axis=0)

    # Detect outliers:
    idx = np.where(np.abs(data - column_medians) > nsigma * column_mads * 1.4826)

    # Create new mask:
    new_mask = np.ones(data.shape)
    new_mask[idx] = 0

    # Return combined mask:
    return mask * new_mask


def get_roeba(data, mask):
    """
    ROEBA algorithm for even/odd and one-over-f --- algorithm is Everett Schlawlin's idea (so cite
    tshirt when using this: https://tshirt.readthedocs.io/en/latest/specific_modules/ROEBA.html)

    Parameters
    ----------

    data : numpy.array
        Numpy array of dimensions (npixel, npixel). It is assumed columns go in the slow-direction
        (i.e., 1/f striping direction) and rows go
        in the fast-read direction (i.e., odd-even effect direction).

    mask : numpy.array
        Numpy array of the same length as `data`; pixels that should be included in the calculation
        (expected to be non-iluminated by the main source)
        should be set to 1 --- the rest should be zeros

    Returns
    -------

    roeba : numpy.array
        Odd-even, one-over-f correction model
    """

    # Nan-ed data so we do nanmedians to mask:
    idx = np.where(mask == 0.0)
    nan_data = np.copy(data)
    nan_data[idx] = np.nan

    # Create output model:
    roeba = np.zeros(data.shape)

    # First compute odd-even model:
    roeba[::2, :] = np.nanmedian(nan_data[::2, :])
    roeba[1::2, :] = np.nanmedian(nan_data[1::2, :])

    slowread_model = np.copy(roeba)
    odd_even_corr = nan_data - slowread_model

    # Now do one-over-f:
    roeba += np.nanmedian(odd_even_corr, axis=0)

    # Return model:
    return roeba


def get_loom(data, mask, background=None, return_parameters=False):
    """
    Least-squares Odd-even and One-over-f correction Model (LOOM)

    This function returns the best-fit LOOM to a given frame/group. Note given the least-squares
    nature of LOOM,
    this is quite sensitive to outliers --- be sure to mask those out as well when using this
    function.

    Parameters
    ----------

    data : numpy.array
        Numpy array of dimensions (npixel, npixel). It is assumed columns go in the slow-direction
        (i.e., 1/f striping direction) and rows go
        in the fast-read direction (i.e., odd-even effect direction).

    mask : numpy.array
        Numpy array of the same length as `data`; pixels that should be included in the calculation
        (expected to be non-iluminated by the main source)
        should be set to 1 --- the rest should be zeros

    background : numpy.array
        Numpy array of the same length as `data` containing a background model to fit simultaneously
        to odd/even and 1/f.

    return_parameters : bool
        (Optional) If True, parameters of the LOOM are returned as well. Default is False.

    Returns
    -------

    loom : numpy.array
        Best-fit LOOM that considers a frame-wise offset, odd-even effect and 1/f striping along the
        columns. Has same dimensions as input `data`.

    parameters : numpy.array
        (Optional) Parameters of the LOOM --- [O, E, a_0, a_1, a_2, ..., a_(ncolumns-1)]. E are the
        even rows, O the odd rows,
        and the a_i the mean 1/f pattern of each column. Note E and O also account for overall
        offsets in the image.

    """

    # Extract some basic information from the data:
    nrows, ncolumns = data.shape

    if background is None:
        # Now, initialize the A matrix and b vector:
        A = np.zeros([ncolumns + 2, ncolumns + 2])
        b = np.zeros(ncolumns + 2)

        # Compute various numbers we will need to fill this matrix:
        nrows_j = np.sum(mask, axis=0)  # number of pixels on each column j
        neven_j = np.sum(mask[::2], axis=0)  # number of even pixels on each column j
        nodd_j = np.sum(mask[1::2], axis=0)  # number of odd pixels on each column j
        ncols_i = np.sum(mask, axis=1)  # number of pixels on each row i
        nE = np.sum(ncols_i[::2])  # number of total pixels on even rows
        nO = np.sum(ncols_i[1::2])  # number of total pixels on odd rows

        # Start filling the A matrix and b vector. First column of A matrix are coefficients
        # for mu, second for odd, third for even, and the rest are the coefficients for each
        # column a_j. Start with results from equation for the mu partial derivative:

        # A[0,0], A[0,1], A[0,2], A[0,3:] = npix, nO, nE, nrows_j

        # b[0] = np.sum(mask * data)

        # Now equation for O partial derivative:

        A[0, 0], A[0, 1], A[0, 2:] = nO, 0.0, nodd_j

        b[0] = np.sum(mask[1::2, :] * data[1::2, :])

        # Same for E partial derivative:

        A[1, 0], A[1, 1], A[1, 2:] = 0.0, nE, neven_j

        b[1] = np.sum(mask[::2, :] * data[::2, :])

        # And, finally, for the a_j partial derivatives:

        A[2:, 0], A[2:, 1] = nodd_j, neven_j

        for j in range(ncolumns):
            A[j + 2, j + 2] = nrows_j[j]

            b[j + 2] = np.sum(mask[:, j] * data[:, j])

        # Solve system:
        x = lsmr(A, b)[0]

        # Create LOOM:
        # loom = np.ones(data.shape) * x[0] # Set mean-level
        loom = np.zeros(data.shape)
        loom[1::2, :] += x[0]  # Add odd level
        loom[::2, :] += x[1]  # Add even level

        # Add 1/f column pattern:
        for j in range(ncolumns):
            loom[:, j] += x[j + 2]

    else:
        S_squared = np.sum((background * mask) ** 2)
        odd_S = np.sum(background[1::2, :] * mask[1::2, :])
        even_S = np.sum(background[::2, :] * mask[::2, :])

        # Now, initialize the A matrix and b vector:
        A = np.zeros([ncolumns + 3, ncolumns + 3])
        b = np.zeros(ncolumns + 3)

        # Compute various numbers we will need to fill this matrix:
        nrows_j = np.sum(mask, axis=0)  # number of pixels on each column j
        neven_j = np.sum(mask[::2], axis=0)  # number of even pixels on each column j
        nodd_j = np.sum(mask[1::2], axis=0)  # number of odd pixels on each column j
        ncols_i = np.sum(mask, axis=1)  # number of pixels on each row i
        nE = np.sum(ncols_i[::2])  # number of total pixels on even rows
        nO = np.sum(ncols_i[1::2])  # number of total pixels on odd rows

        # Start filling the A matrix and b vector. First column of A matrix are coefficients
        # for mu, second for odd, third for even, and the rest are the coefficients for each
        # column a_j. Start with results from equation for the mu partial derivative:

        A[0, 0], A[0, 1], A[0, 2] = S_squared, odd_S, even_S

        for i in range(ncolumns):
            A[0, 3 + i] = np.sum(background[:, i] * mask[:, i])

        b[0] = np.sum(data * background * mask)

        # Now equation for O partial derivative:

        A[1, 0], A[1, 1], A[1, 2], A[1, 3:] = odd_S, nO, 0.0, nodd_j

        b[1] = np.sum(mask[1::2, :] * data[1::2, :])

        # Same for E partial derivative:

        A[2, 0], A[2, 1], A[2, 2], A[2, 3:] = even_S, 0.0, nE, neven_j

        b[2] = np.sum(mask[::2, :] * data[::2, :])

        # And, finally, for the a_j partial derivatives:

        A[3:, 1], A[3:, 2] = nodd_j, neven_j

        for j in range(ncolumns):
            A[j + 3, 0] = np.sum(background[:, j] * mask[:, j])

            A[j + 3, j + 3] = nrows_j[j]

            b[j + 3] = np.sum(mask[:, j] * data[:, j])

        # Solve system:
        x = lsmr(A, b)[0]

        # Create LOOM:
        loom = background * x[0]
        loom[1::2, :] += x[1]  # Add odd level
        loom[::2, :] += x[2]  # Add even level

        # Add 1/f column pattern:
        for j in range(ncolumns):
            loom[:, j] += x[j + 3]

    # Return model (and parameters, if wanted):
    if not return_parameters:
        return loom

    else:
        return loom, x


def get_last_minus_first(data, min_group=None, max_group=None, return_median=True):
    """
    This function returns a last-minus-first slope estimate. This is typically very useful for
    various reasons --- from a quick-data-reduction standpoint
    to a true analysis alternative with Poisson-dominated last-groups.

    Parameters
    ---------

    data : numpy.array
        Numpy array of dimension [nintegrations, ngroups, npixels, npixels], i.e., group-level data.
    min_group : int
        (Optional) Minimum group to use in the last-minus-first (i.e., group that will be the
        "first" group). Number is expected to be in python indexing (i.e., first group
        is index zero). If not define, min_group will be set to 0.
    max_group : int
        (Optional) Maximum group to use in the last-minus-first (i.e., group that will be the "last"
        group). Number is expected to be in python indexing (i.e., last group of
        a 9-group in tegration is expected to be 8). If not, define max_group as data.shape[1] - 1.
    return_median : bool
        (Optional) if True, returns median of LMF. Default is True.


    Returns
    -------

    lmf : numpy.array
        Last-minus-first slope in units of the input data (i.e., divide by the integration-time to
        get the rate).
    median_lmf : numpy.array
        (Optional) Median of the last-minus-first slope estimate.

    """

    # First, extract dimensions:
    nintegrations, ngroups = data.shape[0], data.shape[1]
    # Create array that will save the LMF array:
    lmf = np.zeros([nintegrations, data.shape[2], data.shape[3]])

    # Check if user ingested number of groups:
    if max_group is None:
        max_group = ngroups - 1

    if min_group is None:
        min_group = 0

    # Now iterate through group-level data to get LMF:
    for i in range(nintegrations):
        # Remove median to account for group-to-group median differences:
        last_group = data[i, max_group, :, :] - np.nanmedian(data[i, max_group, :, :])
        first_group = data[i, min_group, :, :] - np.nanmedian(data[i, min_group, :, :])

        lmf[i, :, :] = last_group - first_group

    if return_median:
        # Get median LMF:
        median_lmf = np.nanmedian(lmf, axis=0)

        # Return products:
        return lmf, median_lmf

    else:
        return lmf


def spill_filter(mask, spill_length=10, box_length=20, fraction=0.5):
    """
    Filter that "spills" a value of 0 in an image, around all pixels that have values of 0 ---
    except in the corner of the images. Only pixels on which a box around it of
    length box_length have more than `fraction` pixels with zeros are able to spill zeroes.

    Parameters
    ----------

    mask : numpy.array
        Numpy array with values of 0 and 1's.

    spill_length : int
        For each pixel with a value of 0, this sets the radius around which pixels will be also set
        as zero.

    box_length : int
        Box for checking fractions of zero-pixels

    fraction : float
        Minimum fraction of zero pixels to activate the algorithm

    """

    new_mask = np.copy(mask)

    rows, columns = mask.shape

    idx = np.where(mask == 0)

    for i in range(len(idx[0])):
        row, column = idx[0][i], idx[1][i]

        if row > 5 and row < rows - 5 and column > 5 and column < columns - 5:
            box = mask[
                row - int(box_length * 0.5) : row + int(box_length * 0.5),
                column - int(box_length * 0.5) : column + int(box_length * 0.5),
            ]

            if box.size == 0:
                continue  # The legacy NaN comparison also skipped empty boxes.

            idx_box = np.where(box == 0.0)[0]
            current_fraction = np.double(len(idx_box)) / np.double(box.shape[0] * box.shape[1])

            if current_fraction > fraction:
                new_mask[
                    row - spill_length : row + spill_length,
                    column - spill_length : column + spill_length,
                ] = 0

    return new_mask


def get_uniluminated_mask(data, pixeldq=None, nsigma=3, first_time=True, spill_length=10):
    """
    Given a frame (or group, or average of integrations) --- this function masks all pixels that are
    uniluminated. The function
    returns 1's on all uniluminated pixels, and 0's on all illuminated ones.

    Parameters
    ---------

    data : numpy.array
        Numpy array of dimension [npixels, npixels], i.e., a frame, group, average of integrations,
        etc.

    pixeldq : numpy.array
        Numpy array of same dimensions as data, containing pixel data-qualities. Only pixels with
        values of 0 will be used to
        perform calculations.

    nsigma : double
        (Optional) n-sigma to define above which, at each column, a pixel is illuminated.

    first_time : bool
        (Optional) If True, this is the first time this function is called (useful for recursions).

    spill_length : int
        (Optional) Number of pixels on which the mask will "spill" around.

    Returns
    ---------

    mask : numpy.array
        Numpy array with masked pixels. 1's are uniluminated pixels; 0's are illuminated ones

    """
    # Create output mask:
    mask = np.ones(data.shape)

    # Create bad-pixel mask, if pixeldq is available:
    data_quality = np.ones(data.shape)
    idx_bad_pixels = np.where(np.zeros(data.shape, dtype=bool))

    if pixeldq is not None:
        idx_bad_pixels = np.where(pixeldq != 0)
        data_quality[idx_bad_pixels] = np.nan
        mask[idx_bad_pixels] = 0  # Mask bad pixels out

    # Get column-to-column level (to account for 1/f):
    cc = np.nanmedian(data * data_quality, axis=0)

    # Iterate throughout columns to find uniluminated pixels:
    for i in range(len(cc)):
        # Get sigma:
        column_residuals = (data[:, i] - cc[i]) * data_quality[:, i]
        mad = np.nanmedian(np.abs(column_residuals - np.nanmedian(column_residuals)))
        sigma = mad * 1.4826

        # Identify iluminated pixels:
        idx = np.where(data[:, i] > cc[i] + nsigma * sigma)[0]

        # Mask them:
        mask[idx, i] = 0

    # Mask all bad pixels:
    mask[idx_bad_pixels] = 0

    # Run same algorithm one more time:
    if first_time:
        mask = get_uniluminated_mask(data, pixeldq=mask - 1, nsigma=nsigma, first_time=False)

    # Return mask after spill-filter:
    return spill_filter(mask, spill_length=spill_length)


def correct_1f(
    input_frame,
    template_frame,
    x_trace,
    y_trace,
    scale_factor=1.0,
    inner_radius=3,
    outer_radius=10,
    return_detector=False,
):

    # Get the detector frame by substracting the template, accounting for the relative flux:
    detector = input_frame - (template_frame * scale_factor)

    # Iterate through the trace, remove median of non-ommited pixels:
    for i in range(len(x_trace)):
        signal = detector[:, x_trace[i]]

        signal[int(y_trace[i]) - inner_radius : int(y_trace[i]) + inner_radius] = np.nan
        signal[: int(y_trace[i] - outer_radius)] = np.nan
        signal[int(y_trace[i] + outer_radius) :] = np.nan

    one_f = np.nanmedian(detector, axis=0)

    if not return_detector:
        return input_frame - one_f

    else:
        return input_frame - one_f, detector
