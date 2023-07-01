import os
import numpy as np
from scipy.sparse.linalg import lsmr
from copy import copy, deepcopy
from scipy.ndimage import median_filter


from astropy.utils.data import download_file
from astropy import units as u
from astropy.time import Time
from astropy.timeseries import TimeSeries

from jwst.pipeline import calwebb_detector1, calwebb_spec2
from jwst import datamodels

from .utils import *
from .spectroscopy import *
from .timeseries import *

def tso_jumpstep(tso_list, window, nsigma = 10):
    """ 
    This function performs the same functionality as the jump step on a *set* of JWST datamodels (from, e.g., segmented data), repurposed to 
    take advantage of the uniqueeness of TSOs; it however works with any multi-integration exposure. The assumptions are:

    1. You are locked on the same target in all the sets of exposures given to this function.
    2. Adding all exposures together, the number of integrations is > 1.
    2. The number of groups per integration is > 1.
    
    For every pixel, a group difference is calculated for every integration in the TSO between groups i+1 and i; this defines the time-series of this group-difference. 
    Then, a median filter with an input `window` is substracted from this time-series. This is expected to leave only noise on the time-series. 
    Outliers are detected on this group difference time-series, and those are identified as jumps. Those are then marked for group i+1 in the `groupdq` extension 
    with a value of 4 --- indicating a jump has been detected. This is repeated from group 1 to the second-to-last group (Ngroups - 1).

    Parameters
    ----------

    tso_list : list
        List of JWST datamodels for a given exposure (e.g., `tso_list = [datamodels.open(file1), datamodels.open(file2)]`). 
        Typically, these would be the segments of a TSO. The algorithm expects those to be ordered chronologically.

    window : int
        Window for the median filter. It is expected that the number of integrations >> window (a factor of window/nints ~ 1/20 or 1/50 works well for transits/eclipses/phase curves).

    nsigma (optional) : float
        Number of sigmas used to identify outliers.

    Returns
    -------

    output_tso_list : numpy.array
        Same as the `tso_list`, but with the `groupdq` updated to reflect the jump detection. Note this is a shallow copy of `tso_list`, so changing attributes of `tso_list` will also 
        change those of the output and viceversa.

    """

    # List that will save the shallow copies of the input TSO with the updated groupdq's:
    output_tso_list = []
    # Create list to save the number of integrations per segment:
    ips = []
    # Various bookeeping variables:
    input_nints = 0
    input_ngroups = tso_list[0].data.shape[1]
    nrows = tso_list[0].data.shape[2]
    ncolumns = tso_list[0].data.shape[3]

    for i in range( len(tso_list) ):

        # Shallow copy of input list:
        output_tso_list.append( copy(tso_list[i]) )
        # Modify groupdq attribute of the output_tso_list elements so its a deepcopy of the original --- so we can 
        # change it later if needed by our algorithm:
        output_tso_list[-1].groupdq = deepcopy(output_tso_list[-1].groupdq)

        input_nints += tso_list[i].data.shape[0]
        ips.append(tso_list[i].data.shape[0])

    # All right. Now, generate the difference image array:
    di = np.zeros([input_nints,
                   input_ngroups - 1,
                   nrows,
                   ncolumns
                  ])

    # We generate it by performing difference imaging on each element of the input list. For free we get the cummulative number 
    # of integrations to a given segment (will be useful in the next for loop):
    cis = []  # Cummulative number of integrations up to a given segment
    ci = 0    # Current integration counter
    for i in range( len(tso_list) ):

        for j in range(input_ngroups - 1):

            di[ci:ci+ips[i], j, :, :] = tso_list[i].data[:, j+1, :, :] - tso_list[i].data[:, j, :, :]

        ci += ips[i]
        cis.append(ci)
                   
    # All right, now that we have the difference image, we perform outlier detection. This could be easily parallelized BTW:
    for i in range(nrows):
        
        for j in range(ncolumns):
            
            for k in range(input_ngroups - 1): 
               
                # Find outliers for pixel (i,j): 
                idx = outlier_detector(di[:, k, i, j], nsigma = nsigma, window = window)
                
                # If outliers are detected, iterate through them and add them to the corresponding groupdq in the output list:
                if len(idx) > 0:
                    
                    for outlier_index in idx:
                        
                        # For each detected outlier in the difference image, we need to map its location to a segment (and index 
                        # in that segment). We do that by searching segment by segment until the outlier_index surpasses the number 
                        # of cummulative integrations --- when it surpasses it, it means we found the segment:
                        for ii in range( len(output_tso_list) ):

                            index_difference = outlier_index - cis[ii]
                            #print('outlier_index:',outlier_index)
                            #print('cis:',cis)
                            #print('ii:',ii)


                            # If difference is less than zero, ii is the index of the segment for the outlier:
                            if index_difference <= 0:

                                if ii == 0:

                                    output_tso_list[ii].groupdq[outlier_index, k+1, i, j] = 4
                                    break

                                else:

                                    output_tso_list[ii].groupdq[outlier_index - cis[ii-1] - 1, k+1, i, j] = 4
                                    break

    return output_tso_list

def correct_1f(median_frame, frame, x, y, min_bkg_row = 20, max_bkg_row = 35, mask = None):
   
    edge_row = median_frame.shape[0]
 
    new_frame = np.copy(frame)
    
    ms = frame - median_frame
    
    # Go column-by-column substracting values around the trace:
    for i in range(len(x)):
        
        column = x[i]
        row = int(y[i])
        
        min_row = np.max([0, row - max_bkg_row])
        max_row = np.min([edge_row, row + max_bkg_row])
        
        bkg = np.append(ms[min_row:row - min_bkg_row, column], ms[row + min_bkg_row:max_row, column])
        idx = np.argsort(bkg)
        npoints = len(idx)
        new_frame[:, column] = new_frame[:, column] - np.nanmedian(bkg[idx][int(npoints*0.25):int(npoints*0.5)])
        
    return new_frame

def cc_uniluminated_outliers(data, mask, nsigma = 5):
    """
    Column-to-column background outlier detection

    This function goes column-by-column and detects outliers on a given frame (`data`) wherever there are no sources. The user needs to provide a mask where values of 1 are the 
    uniluminated pixels, and 0's are the illuminated pixels. Main difference with `get_uniluminated_mask` is that this function gets you hot pixels, cosmic rays and bad pixels.

    Parameters
    ----------

    data : numpy.array
        Numpy array of dimensions (npixel, npixel). It is assumed columns go in the slow-direction (i.e., 1/f striping direction) and rows go 
        in the fast-read direction (i.e., odd-even effect direction).

    mask : numpy.array
        Numpy array of the same length as `data`; pixels that should be included in the calculation (expected to be non-iluminated by the main source) 
        should be set to 1 --- the rest should be zeros

    Returns
    -------

    updated_mask : numpy.array
        Combination of the input mask with the outliers in the background, which are identified with zeroes.

    """

    # Turn all zeroes in the mask to nans:
    nan_data = np.copy(data)
    nan_data[mask == 0] = np.nan

    # Compute column medians:
    column_medians = np.nanmedian(nan_data, axis = 0)
    
    # Compute column median-absolute deviation:
    column_mads = np.nanmedian(np.abs(nan_data - column_medians), axis = 0)

    # Detect outliers:
    idx = np.where(np.abs(data - column_medians) > nsigma * column_mads * 1.4826)

    # Create new mask:
    new_mask = np.ones(data.shape)
    new_mask[idx] = 0

    # Return combined mask:
    return mask * new_mask

def get_roeba(data, mask):
    """
    ROEBA algorithm for even/odd and one-over-f --- algorithm is Everett Schlawlin's idea (so cite tshirt when using this: https://tshirt.readthedocs.io/en/latest/specific_modules/ROEBA.html)

    Parameters
    ----------

    data : numpy.array
        Numpy array of dimensions (npixel, npixel). It is assumed columns go in the slow-direction (i.e., 1/f striping direction) and rows go 
        in the fast-read direction (i.e., odd-even effect direction).

    mask : numpy.array
        Numpy array of the same length as `data`; pixels that should be included in the calculation (expected to be non-iluminated by the main source) 
        should be set to 1 --- the rest should be zeros

    Returns
    -------

    roeba : numpy.array
        Odd-even, one-over-f correction model
    """

    # Nan-ed data so we do nanmedians to mask:
    idx = np.where(mask == 0.)
    nan_data = np.copy(data)
    nan_data[idx] = np.nan

    # Create output model:
    roeba = np.zeros(data.shape)

    # First compute odd-even model:
    roeba[::2,:] = np.nanmedian(nan_data[::2,:])
    roeba[1::2,:] = np.nanmedian(nan_data[1::2,:])
    
    slowread_model = np.copy(roeba)
    odd_even_corr = nan_data - slowread_model
    
    # Now do one-over-f:
    roeba += np.nanmedian(odd_even_corr, axis = 0)

    # Return model:
    return roeba
    

def get_loom(data, mask, background = None, return_parameters = False):
    """
    Least-squares Odd-even and One-over-f correction Model (LOOM)

    This function returns the best-fit LOOM to a given frame/group. Note given the least-squares nature of LOOM, 
    this is quite sensitive to outliers --- be sure to mask those out as well when using this function.

    Parameters
    ----------
    
    data : numpy.array
        Numpy array of dimensions (npixel, npixel). It is assumed columns go in the slow-direction (i.e., 1/f striping direction) and rows go 
        in the fast-read direction (i.e., odd-even effect direction).

    mask : numpy.array
        Numpy array of the same length as `data`; pixels that should be included in the calculation (expected to be non-iluminated by the main source) 
        should be set to 1 --- the rest should be zeros

    background : numpy.array
        Numpy array of the same length as `data` containing a background model to fit simultaneously to odd/even and 1/f.

    return_parameters : bool
        (Optional) If True, parameters of the LOOM are returned as well. Default is False.

    Returns
    -------

    loom : numpy.array
        Best-fit LOOM that considers a frame-wise offset, odd-even effect and 1/f striping along the columns. Has same dimensions as input `data`.

    parameters : numpy.array
        (Optional) Parameters of the LOOM --- [O, E, a_0, a_1, a_2, ..., a_(ncolumns-1)]. E are the even rows, O the odd rows, 
        and the a_i the mean 1/f pattern of each column. Note E and O also account for overall offsets in the image.
    
    """

    # Extract some basic information from the data:
    nrows, ncolumns = data.shape

    if background is None:

        # Now, initialize the A matrix and b vector:
        A = np.zeros([ncolumns + 2, ncolumns + 2])
        b = np.zeros(ncolumns + 2)

        # Compute various numbers we will need to fill this matrix:
        npix = np.sum(mask)                     # number of pixels used to compute model
        nrows_j = np.sum(mask, axis = 0)        # number of pixels on each column j
        neven_j = np.sum(mask[::2], axis = 0)   # number of even pixels on each column j
        nodd_j = np.sum(mask[1::2], axis = 0)   # number of odd pixels on each column j
        ncols_i = np.sum(mask, axis = 1)        # number of pixels on each row i
        nE = np.sum(ncols_i[::2])               # number of total pixels on even rows
        nO = np.sum(ncols_i[1::2])              # number of total pixels on odd rows

        # Start filling the A matrix and b vector. First column of A matrix are coefficients for mu, second for odd, third for even, and the rest are the coefficients for 
        # each column a_j. Start with results from equation for the mu partial derivative:

        #A[0,0], A[0,1], A[0,2], A[0,3:] = npix, nO, nE, nrows_j

        #b[0] = np.sum(mask * data)

        # Now equation for O partial derivative:

        A[0,0], A[0,1], A[0,2:] = nO, 0., nodd_j

        b[0] = np.sum(mask[1::2, :] * data[1::2, :])
         
        # Same for E partial derivative:

        A[1,0], A[1,1], A[1,2:] = 0., nE, neven_j

        b[1] = np.sum(mask[::2, :] * data[::2, :])

        # And, finally, for the a_j partial derivatives:

        A[2:, 0], A[2:, 1] = nodd_j, neven_j
        
        for j in range(ncolumns):

            A[j + 2, j + 2] = nrows_j[j]

            b[j + 2] = np.sum(mask[:, j] * data[:, j])

        # Solve system:
        x = lsmr(A, b)[0]

        # Create LOOM:
        #loom = np.ones(data.shape) * x[0] # Set mean-level
        loom = np.zeros(data.shape)
        loom[1::2, :] += x[0]             # Add odd level
        loom[::2, :] += x[1]              # Add even level
       
        # Add 1/f column pattern: 
        for j in range(ncolumns):

            loom[:, j] += x[j + 2]

    else:

        S_squared = np.sum( ( background * mask )**2 )
        odd_S = np.sum( background[1::2, :] * mask[1::2, :] )
        even_S = np.sum( background[::2, :] * mask[::2, :] )

        # Now, initialize the A matrix and b vector:
        A = np.zeros([ncolumns + 3, ncolumns + 3]) 
        b = np.zeros(ncolumns + 3)

        # Compute various numbers we will need to fill this matrix:
        nrows_j = np.sum(mask, axis = 0)        # number of pixels on each column j
        neven_j = np.sum(mask[::2], axis = 0)   # number of even pixels on each column j
        nodd_j = np.sum(mask[1::2], axis = 0)   # number of odd pixels on each column j
        ncols_i = np.sum(mask, axis = 1)        # number of pixels on each row i
        nE = np.sum(ncols_i[::2])               # number of total pixels on even rows
        nO = np.sum(ncols_i[1::2])              # number of total pixels on odd rows

        # Start filling the A matrix and b vector. First column of A matrix are coefficients for mu, second for odd, third for even, and the rest are the coefficients for 
        # each column a_j. Start with results from equation for the mu partial derivative:

        A[0,0], A[0,1], A[0,2] = S_squared, odd_S, even_S

        for i in range(ncolumns):

            A[0, 3 + i] = np.sum( background[:, i] * mask[:, i] )

        b[0] = np.sum( data * background * mask )

        # Now equation for O partial derivative:

        A[1,0], A[1,1], A[1,2], A[1,3:] = odd_S, nO, 0., nodd_j

        b[1] = np.sum(mask[1::2, :] * data[1::2, :])

        # Same for E partial derivative:

        A[2,0], A[2,1], A[2,2], A[2,3:] = even_S, 0., nE, neven_j

        b[2] = np.sum(mask[::2, :] * data[::2, :])

        # And, finally, for the a_j partial derivatives:

        A[3:, 1], A[3:, 2] = nodd_j, neven_j

        for j in range(ncolumns):

            A[j + 3, 0] = np.sum( background[:, j] * mask[:, j] )

            A[j + 3, j + 3] = nrows_j[j]

            b[j + 3] = np.sum(mask[:, j] * data[:, j])

        # Solve system:
        x = lsmr(A, b)[0]

        # Create LOOM:
        loom = background * x[0]
        loom[1::2, :] += x[1]             # Add odd level
        loom[::2, :] += x[2]              # Add even level

        # Add 1/f column pattern: 
        for j in range(ncolumns):

            loom[:, j] += x[j + 3]

    # Return model (and parameters, if wanted):
    if not return_parameters:
        
        return loom
    
    else:

        return loom, x

def download_reference_file(filename):
    """
    This function downloads a reference file from CRDS given a reference file filename. File gets downloaded to the current working folder.
    """

    print('\n\t >> Downloading {} reference file from CRDS...'.format(filename))
    download_filename = download_file('https://jwst-crds.stsci.edu/unchecked_get/references/jwst/' + filename, cache=True)

    # Rename file:
    os.rename(download_filename, filename)

def get_last_minus_first(data, min_group = None, max_group = None):
    """
    This function returns a last-minus-first slope estimate. This is typically very useful for various reasons --- from a quick-data-reduction standpoint 
    to a true analysis alternative with Poisson-dominated last-groups.

    Parameters
    ---------

    data : numpy.array
        Numpy array of dimension [nintegrations, ngroups, npixels, npixels], i.e., group-level data.
    min_group : int
        (Optional) Minimum group to use in the last-minus-first (i.e., group that will be the "first" group). Number is expected to be in python indexing (i.e., first group 
        is index zero). If not define, min_group will be set to 0.
    max_group : int
        (Optional) Maximum group to use in the last-minus-first (i.e., group that will be the "last" group). Number is expected to be in python indexing (i.e., last group of 
        a 9-group in tegration is expected to be 8). If not, define max_group as data.shape[1] - 1.

    Returns
    -------

    lmf : numpy.array
        Last-minus-first slope in units of the input data (i.e., divide by the integration-time to get the rate).
    median_lmf : numpy.array
        Median of the last-minus-first slope estimate.

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

    # Get median LMF:
    median_lmf = np.nanmedian(lmf, axis = 0)

    # Return products:
    return lmf, median_lmf

def spill_filter(mask, spill_length = 10, box_length = 20, fraction = 0.5):
    """
    Filter that "spills" a value of 0 in an image, around all pixels that have values of 0 --- except in the corner of the images. Only pixels on which a box around it of 
    length box_length have more than `fraction` pixels with zeros are able to spill zeroes.

    Parameters
    ----------

    mask : numpy.array
        Numpy array with values of 0 and 1's.

    spill_length : int
        For each pixel with a value of 0, this sets the radius around which pixels will be also set as zero.

    box_length : int
        Box for checking fractions of zero-pixels
    
    fraction : float
        Minimum fraction of zero pixels to activate the algorithm

    """

    new_mask = np.copy(mask)

    rows, columns = mask.shape

    idx = np.where( mask == 0 )

    for i in range( len(idx[0]) ):

        row, column = idx[0][i], idx[1][i]

        if row > 5 and row < rows-5 and column > 5 and column < columns - 5:

            box = mask[row - int(box_length*0.5) : row + int(box_length*0.5), column - int(box_length*0.5) : column + int(box_length*0.5)]

            idx_box = np.where(box == 0.)[0]
            current_fraction = np.double( len(idx_box) ) / np.double(box.shape[0] * box.shape[1])

            if current_fraction > fraction:

                new_mask[row-spill_length:row+spill_length, column-spill_length:column+spill_length] = 0

    return new_mask 

def get_uniluminated_mask(data, pixeldq = None, nsigma = 3, first_time = True, spill_length = 10):
    """
    Given a frame (or group, or average of integrations) --- this function masks all pixels that are uniluminated. The function 
    returns 1's on all uniluminated pixels, and 0's on all illuminated ones.

    Parameters
    ---------

    data : numpy.array
        Numpy array of dimension [npixels, npixels], i.e., a frame, group, average of integrations, etc.

    pixeldq : numpy.array
        Numpy array of same dimensions as data, containing pixel data-qualities. Only pixels with values of 0 will be used to 
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

    if pixeldq is not None:

        idx_bad_pixels = np.where( pixeldq != 0 )
        data_quality[idx_bad_pixels] = np.nan
        mask[idx_bad_pixels] = 0 # Mask bad pixels out

    # Get column-to-column level (to account for 1/f):
    cc = np.nanmedian(data * data_quality, axis=0)

    # Iterate throughout columns to find uniluminated pixels:
    for i in range(len(cc)):

        # Get sigma:
        column_residuals = ( data[:, i] - cc[i] ) * data_quality[:, i]
        mad = np.nanmedian(np.abs(column_residuals - np.nanmedian(column_residuals)))
        sigma = mad * 1.4826

        # Identify iluminated pixels:
        idx = np.where( data[:,i] > cc[i] + nsigma * sigma )[0]

        # Mask them:
        mask[idx, i] = 0

    # Mask all bad pixels:
    mask[idx_bad_pixels] = 0

    # Run same algorithm one more time:
    if first_time:
    
        mask = get_uniluminated_mask(data, pixeldq = mask - 1, nsigma = nsigma, first_time = False)
    
    # Return mask after spill-filter:
    return spill_filter(mask, spill_length = spill_length)

def get_cds(data):
    """
    This function gets performs Correlated Double Sampling on an image or set of images, and return the CDS product. The function simply substracts group 2 minus 1, 
    3 minus 2, etc. for all groups in every integration or exposure. Useful for quicklooks at data.

    Parameters
    ----------

    data : `jwst.RampModel` object or list
        Input `jwst.RampModel` object for which CDS will be performed. Can also be a list of objects, in which case it will be assumed this is segmented data that 
        wants to be reduced. If a list, it is assumed the objects are chronologically ordered.

    Returns
    -------

    times : `np.array`
        Time-stamp for each of the `nint * (ngroups - 1)` CDS frames (spacecraft clock).

    cds_frames : `np.array`
        Numpy array containing the CDS frames of dimensions `(nint * (ngroups - 1), nx, ny)`, where the first dimension are all the possible CDS frames obtainable 
        from the data, and `nx` and `ny` are the frame dimensions.

    """

    if type(data) is list:

        nintegrations = 0
        nsegments = len(data)

        for i in range( nsegments ):

            nintegration, ngroups = data[i].data.shape[0], data[i].meta.exposure.ngroups
            nintegrations += nintegration

        times = np.zeros( ( ngroups - 1 ) * nintegrations )

    else:

        nintegrations, ngroups = data.data.shape[0], data.meta.exposure.ngroups
        data = [data]

        times = np.zeros( ( ngroups - 1 ) * nintegrations )

    # Set array that will save CDS frames:
    cds_frames = np.zeros( [ (ngroups - 1) * nintegrations, \
                             data[0].data.shape[2],\
                             data[0].data.shape[3]  ] )

    # Initialize some variables:
    counter = 0
    first_time = True

    # Run through all data:
    for dataset in data:

        if first_time:

            # For the first time, set time of exposure time:
            second_to_day = 1. / (24. * 3600.)
            frametime = dataset.meta.exposure.frame_time # seconds
            grouptime = dataset.meta.exposure.group_time # seconds

            time_start = dataset.meta.observation.date + 'T' + dataset.meta.observation.time
            ts = Time(time_start)

            #      v-orig-v   v-----skip reset------v
            tstart = ts.jd + frametime * second_to_day 
            first_time = False



        for integration in range(dataset.shape[0]):

            for current_group in range( ngroups - 1 ):
            
                cds_frames[counter, :, :] = dataset.data[integration, current_group + 1, :, :] - \
                                            dataset.data[integration, current_group, :, :] 

                # tstart here is the time just after the reset of the current integration.
                times[counter] = tstart + (current_group + 1) * grouptime
                counter += 1

            # When we swap integrations, we add a group-time (to finish the last group time) and a frame time (to jump from the reset):
            tstart = tstart + (current_group + 1) * grouptime + grouptime + frametime

    return times, cds_frames

def correct_local_1f(input_frame, template_frame, relative_flux, x_trace, y_trace, ommited_radius = 3, outer_radius = 10):
    
    # Get the detector frame by substracting the template, accounting for the relative flux:
    detector = input_frame - (template_frame * relative_flux)
    
    # Iterate through the trace, remove median of non-ommited pixels:
    for i in range( len(x_trace) ):
        
        signal = detector[:, x_trace[i]]
        
        signal[int(y_trace[i])-ommited_radius:int(y_trace[i])+ommited_radius] = np.nan
        signal[:int(y_trace[i]-outer_radius)] = np.nan
        signal[int(y_trace[i]+outer_radius):] = np.nan
        
    one_f = np.nanmedian(detector, axis = 0)
        
    return input_frame - one_f, detector

def old_correct_1f(input_frame, x_trace, y_trace, outer_radius = 10):
    
    # Create the frame we'll play with:
    detector = np.copy(input_frame)
    
    # Iterate through the trace, mark as nans pixels we won't be using:
    for i in range( len(x_trace) ):
        
        signal = detector[:, x_trace[i]]
        
        signal[int(y_trace[i])-outer_radius:int(y_trace[i])+outer_radius] = np.nan
        
    one_f = np.nanmedian(detector, axis = 0)
        
    return input_frame - one_f, detector

def cds_stage1(datafiles, nintegrations, ngroups, trace_radius = 10, ommited_trace_radius = 3, instrument = 'nirspec/g395h', background_model = None, background_mask = None):
    """
    Initial version of a CDS-based Stage 1 pipeline. Inputs are `*ramp*` files. This assumes the `datafiles` is a list with ordered segments (e.g., first element is first segment, etc.).

    Parameters
    ----------

    datafiles : `list` of strings
        List containing the data filenames to ingest. These should be ramps, and they are assumed to be ordered.

    nintegrations : int
        Number of integrations in the exposure.

    ngroups : int
        Number of groups per integration.

    trace_radius : int
        Radius of the trace's PSF.

    ommited_trace_radius : int
        Radius from the center of the trace for which pixels will be ommited when doing the local 1/f correction (typically pixels too close to the center 
        of the trace where the PSF removal is not too good).

    instrumnet : string
        Currently supports 'nirspec/g395h' and `niriss/soss/substrip256`.

    background_model : `np.array`
        Background model of the same size as the data. This will be scaled against the data using non-iluminated pixels in the `background_mask`.
    
    background_mask : `np.array`
        Array containing pixels that will be used to scale the background. Values of 0 are assumed to be illuminated pixels, values of 1 are background pixels.

    Returns
    -------

    times : `np.array`
        Time-stamp for each of the integrations.

    cds_data : `np.array`
        Numpy array containing the CDS frames of dimensions `(nint * (ngroups - 1), nx, ny)`, where the first dimension are all the possible CDS frames obtainable 
        from the data, and `nx` and `ny` are the frame dimensions.

    initial_whitelight: `np.array`
        Initial white-light lightcurve obtained from the average CDSs per integration.

    smooth_wl : `np.array`
        Smoothed version of the `initial_whitelight`, obtained to find the weights that went into the local 1/f corrections.

    """

    if instrument.lower() == 'nirspec/g395h':

        rows = 32
        columns = 2048
        initial_1f = True

    elif instrument.lower == 'niriss/soss/substrip256':

        rows = 256
        columns = 2048
        initial_1f = True

    else:

        print('Need instrument name. Ending.')
        sys.exit()

    # First, extract data and time-stamps from the datamodel:
    data, err = np.zeros([nintegrations, ngroups, rows, columns]), np.zeros([nintegrations, ngroups, rows, columns])
    times = np.zeros(nintegrations)

    past_nints = 0
    for i in range( len(datafiles) ):

        dm = datamodels.RampModel(datafiles[i])

        current_nints = dm.data.shape[0]

        times[past_nints:past_nints+current_nints] = np.copy(dm.int_times['int_mid_BJD_TDB'])
        data[past_nints:past_nints+current_nints, :, :, :] = np.copy(dm.data)
        err[past_nints:past_nints+current_nints, :, :, :] = np.copy(dm.err)

        past_nints = past_nints + current_nints
 
    # Now, get CDSs:
    cds_data = np.zeros([data.shape[0], data.shape[1]-1, data.shape[2], data.shape[3]])

    for i in range(cds_data.shape[0]):
        
        for j in range(cds_data.shape[1]):
            
            cds_data[i, j, :, :] = data[i, j+1, :, :] - data[i, j, :, :]

    # Get median and trace:
    median_cds = np.nanmedian(cds_data, axis = (0,1))

    # Get initial centroid of right-most part of the spectrum:
    if instrument.lower() == 'nirspec/g395h':

        if 'nrs1' in datafiles[0]:

            xstart, xend, nknots = 2043, 500, 60
            median_edge_psf = np.nanmedian(median_cds[:, xstart-200:xstart], axis = 1)

        else:

            xstart, xend, nknots = 5, 2043, 60
            median_edge_psf = np.nanmedian(median_cds[:, xstart:xstart+200], axis = 1)

    elif 'niriss/soss' in instrument.lower():

        xstart, xend, nknots = 2043, 5, 30
        median_edge_psf = np.nanmedian(median_cds[:100, xstart-200:xstart], axis = 1)

        xstart2, xend2, nknots2 = 700, 1755, 30
        median_edge_psf2 = np.nanmedian(median_cds[75:110, xstart:xstart+200], axis = 1)

        

    centroid = np.nansum ( median_edge_psf * np.arange(len(median_edge_psf))  ) / np.nansum( median_edge_psf )

    x1, y1 = trace_spectrum(median_cds, np.zeros(median_cds.shape), 
                            xstart = xstart, ystart = centroid, xend = xend)

    # Handle outliers. First, get standard deviation:
    mf = median_filter(y1, 3)
    sigma = np.nanmedian( np.abs(mf - y1) ) * 1.4826

    # Identify values above 5-sigma from that, replace them by the median filter:
    idx = np.where( np.abs(mf - y1) > 5*sigma )[0]
    new_y1 = np.copy(y1)
    new_y1[idx] = mf[idx] 

    # Smooth trace:
    _, ysmooth = fit_spline(x1, new_y1, nknots = nknots)

    # If substrip256, get Order 2 as well:
    if 'substrip256' in instrument.lower():

        xstart2, xend2, nknots2 = 700, 1755, 30
        median_edge_psf2 = np.nanmedian(median_cds[75:110, xstart2:xstart2+200], axis = 1)

        centroid2 = np.nansum ( median_edge_psf2 * np.arange(75,110,1)  ) / np.nansum( median_edge_psf2 )

        x2, y2 = trace_spectrum(median_cds, np.zeros(median_cds.shape),
                                xstart = xstart2, ystart = centroid2, xend = xend2)

        # Handle outliers. First, get standard deviation:
        mf2 = median_filter(y2, 3)
        sigma2 = np.nanmedian( np.abs(mf2 - y2) ) * 1.4826

        # Identify values above 5-sigma from that, replace them by the median filter:
        idx2 = np.where( np.abs(mf - y2) > 5*sigma2 )[0]
        new_y2 = np.copy(y2)
        new_y2[idx] = mf2[idx] 

        # Smooth trace:
        _, ysmooth2 = fit_spline(x2, new_y2, nknots = nknots2)

    # All right. Now, let's do some background substraction using this median CDS frame. If using NIRISS/SOSS, we use the 
    # input background_model and background_mask:
    if instrument.lower() == 'nirspec/g395h':

        trace_radius = 10

    # To estimate it, let's use the background counts measured by the median CDS frame outside from around 
    # trace_radius from the trace:
    in_trace_pixels = np.zeros(median_cds.shape)
    in_trace_pixels[:] = np.nan
    out_of_trace_pixels = np.ones(median_cds.shape)
    rows = np.arange(median_cds.shape[0])

    for i in range(len(x1)):
        
        idx_in = np.where( np.abs( rows - ysmooth[i] ) <= trace_radius)[0]
        out_of_trace_pixels[idx_in, x1[i]] = np.nan
        in_trace_pixels[idx_in, x1[i]] = 1.

    # Substract from the data --- and also get an initial version of the 1/f noise-corrected data (using only outside pixels), 
    # and a corresponding white-light lightcurve out of that:
    cds_bkg = np.nanmedian(median_cds * out_of_trace_pixels, axis = 0)
    cds_initial1fcorrected = np.zeros(cds_data.shape)
    initial_whitelight = np.zeros(cds_data.shape[0])

    if not initial_1f:

        cds_initial1fcorrected = np.copy(cds_data) 

    for integration in range(cds_data.shape[0]):

        for group in range(cds_data.shape[1]):

            # Background corrected data:
            cds_data[integration, group, :, :] = cds_data[integration, group, :, :] - cds_bkg 

            if initial_1f:

                # 1/f corrected data using out-of-trace pixels only:
                onef_2D = out_of_trace_pixels * cds_data[integration, group, :, :]
                cds_initial1fcorrected[integration, group, :, :] = cds_data[integration, group, :, :] - np.nanmedian( onef_2D, axis = 0 )

        # Get initial white-light lightcurve. First, get median frame for the current integration accross groups:
        median_integration = np.nanmedian( cds_initial1fcorrected[integration, :, :, :], axis = 0 )
        # Now sum all the in-trace pixels in this median integration:
        initial_whitelight[integration] = np.nansum( median_integration * in_trace_pixels )

    # Create relative flux white-light:
    initial_whitelight = initial_whitelight / np.nanmedian(initial_whitelight)

    # Get smoothed version via median-filter to remove any outliers:
    smooth_wl = median_filter(initial_whitelight, 5)

    # Create new, bkg-substracted median:
    new_median_cds = np.nanmedian(cds_data, axis = (0,1))

    # Now, 1/f noise. We do a "local" 1/f noise removal --- we include the white-light lightcurve to scale the template (hence why we need 
    # background-removed lightcurves):
    for integration in range(cds_data.shape[0]):

        for group in range(cds_data.shape[1]):

            cds_data[integration, group, :, :], _ = correct_local_1f(cds_data[integration, group, :, :], \
                                                                     new_median_cds, smooth_wl[integration], \
                                                                     x1, ysmooth, \
                                                                     ommited_radius = ommited_trace_radius, outer_radius = trace_radius)

    # For now, return CDS, backgroud and 1/f-noise-corrected data and time-stamps:
    return times, cds_data, initial_whitelight, smooth_wl

def stage1(uncal_filenames, maximum_cores = 'all', background_model = None, outputfolder = '', use_tso_jump = True, **kwargs):
    """
    This function calibrates a set of *uncal.fits files through a "special" version of the JWST TSO CalWebb Stage 1. It has been tested thoroughly for NIRISS/SOSS, NIRSpec/G395H and NIRSpec/Prism.

    Parameters
    ----------

    uncal_filenames : list
        A list of filenames having `*uncal.fits` files. It is expected the inputs are different segments, and that they are given in chronological order --- e.g.,
        `uncal_filenames = ['jw01331003001_04101_00001-seg001_nrs1_uncal.fits', 'jw01331003001_04101_00001-seg002_nrs1_uncal.fits']`.
    maximum_cores : string
        If 'all', multiprocessing will be used for the `ramp_fit` step using all available cores.
    background_model : numpy.array
        (Optional) Array of the same size as the data groups and/or frames containing a model background (if any) to scale.
    outputfolder : string
        (Optional) String indicating the folder where the outputs want to be saved. Default is current working directory.
    use_tso_jump : bool
        (Optional) If `True`, use the TSO jump detection algorithm implemented here. If `False`, will use the standard STScI one. The TSO Jump is controlled by 
        parameters `jump_nsigma` and `jump_window`.

    Returns
    -------

    output_dictionary : dict
        Dictionary containing by default the `times`, `rateints`, `errors` and `dataquality` (data-quality flags). 
    
    """

    # Define output folder if empty:
    if outputfolder != '':
        if outputfolder[-1] != '/':
            outputfolder += '/'

    # Create output dictionary:
    output_dictionary = {}

    #####################################################
    #                       STAGE 1                     #
    #####################################################

    # Create folder that will store pipeline outputs:
    if not os.path.exists(outputfolder+'pipeline_outputs'):
        os.mkdir(outputfolder+'pipeline_outputs')

    # Open the uncal files through a datamodel; save those in a list. Take the chance to extract the int times:
    uncal_data = []
    datanames = []
    times = np.array([])

    for i in range( len(uncal_filenames) ):

        uncal_data.append( datamodels.RampModel(uncal_filenames[i]) )
        datanames.append( uncal_filenames[i].split('/')[-1].split('uncal.fits')[0][:-1] )
        times = np.append(times, uncal_data[-1].int_times['int_mid_BJD_TDB'] + 2400000.5)

    # Extract some useful meta-data:
    instrument_name = uncal_data[-1].meta.instrument.name
    instrument_filter = uncal_data[-1].meta.instrument.filter
    instrument_grating = uncal_data[-1].meta.instrument.grating

    # Print some information to the user:
    print('\t >> Processing '+str(len(uncal_data))+' *uncal files.\n')
    print('\t    - TSO total duration: {0:.1f} hours'.format((np.max(times)-np.min(times))*24.))

    if instrument_name == 'NIRSPEC' and instrument_grating == 'PRISM':

        print('\t    - Instrument/Mode: NIRSpec/PRISM\n')

        mode = 'nirspec/prism'

    else:

        raise Exception('\t Error: Instrument/Grating/Filter: '+instrument_name+'/'+instrument_grating+'/'+instrument_filter+' not yet supported!')

    if not os.path.exists(outputfolder+'pipeline_outputs/'+datanames[-1]+'_linearitystep.fits'):

        # First, perform standard processing in the first few steps of the JWST pipeline. First, the DQ step:
        dq_data = []
        for i in range( len(uncal_data) ):

            dq_data.append( calwebb_detector1.dq_init_step.DQInitStep.call(uncal_data[i]) )

        # Saturation step:
        saturation_data = []
        for i in range( len(dq_data) ):

            if 'override_saturation' in kwargs.keys():
            
                saturation_data.append( calwebb_detector1.saturation_step.SaturationStep.call(dq_data[i], 
                                                                                              override_saturation = kwargs['override_saturation']) 
                                      )

            else:

                saturation_data.append( calwebb_detector1.saturation_step.SaturationStep.call(dq_data[i]) 
                                      )

        # Superbias step:
        superbias_data = []
        for i in range( len(saturation_data) ):

            if 'override_superbias' in kwargs.keys():
         
                superbias_data.append( calwebb_detector1.superbias_step.SuperBiasStep.call(saturation_data[i], 
                                                                                           override_superbias = kwargs['override_superbias']) 
                                     )    

            else:

                superbias_data.append( calwebb_detector1.superbias_step.SuperBiasStep.call(saturation_data[i]) 
                                     )

        
        # Depending on the instrument/mode, we perform our own reference pixel step:
        refpix_data = []
        if mode == 'nirspec/prism':

            for i in range( len(superbias_data) ): 

                refpix_data.append( copy(superbias_data[i]) )
                refpix_data[-1].data = deepcopy(superbias_data[i].data)

                # Go integration per integration, group by group:
                for integration in range(refpix_data[-1].data.shape[0]):

                    for group in range(refpix_data[-1].data.shape[1]):

                        # Background will be pixels to the left of 25 and the last 25 columns:
                        background = np.hstack( ( refpix_data[-1].data[integration, group, :, :25],
                                                  refpix_data[-1].data[integration, group, :, -25:]
                                                )
                                              )

                        # Substract that from the data to remove pedestal changes:
                        refpix_data[-1].data[integration, group, :, :] -= np.nanmedian( background )

                        # TODO: Get fancy and remove odd/even, too.

        else:

            for i in range( len(superbias_data) ):

                refpix_data.append( calwebb_detector1.refpix_step.RefPixStep.call(superbias_data[i])
                                  )

                
        # Linearity step:
        linearity_data = []
        for i in range( len(refpix_data) ):

            if 'override_linearity' in kwargs.keys():

                linearity_data.append( calwebb_detector1.linearity_step.LinearityStep.call(refpix_data[i],
                                                                                           override_linearity = kwargs['override_linerity'], 
                                                                                           output_dir=outputfolder+'pipeline_outputs', \
                                                                                           save_results = True 
                                                                                          )
                                     )

            else:

                linearity_data.append( calwebb_detector1.linearity_step.LinearityStep.call(refpix_data[i],
                                                                                           output_dir=outputfolder+'pipeline_outputs', \
                                                                                           save_results = True 
                                                                                          )
                                     )


    else:

        print('\t >> Linearity files found. Loading them...\n')

        linearity_data = []
        for i in range( len(uncal_filenames) ):

            linearity_data.append( datamodels.RampModel(outputfolder+'pipeline_outputs/'+datanames[i]+'_linearitystep.fits') )

    # Now, instead of passing this through the normal jump step, we pass it through our own jump step detection:
    if mode == 'nirspec/prism':

        if 'jump_window' not in kwargs.keys():

            jump_window = 200

        else:

            jump_window = kwargs['jump_window']

        if 'jump_nsigma' not in kwargs.keys():

            jump_nsigma = 10

        else:

            jump_nsigma = kwargs['jump_nsigma']


    if use_tso_jump:

        jump_data = tso_jumpstep(linearity_data, window = jump_window, nsigma = jump_nsigma)
        
        # Save results:
        for i in range( len(jump_data) ):

            jump_data[i].save(datanames[i]+'_tsojump', dir_path = outputfolder+'pipeline_outputs')

    else:

        if 'jump_threshold' not in kwargs.keys():

            jump_threshold = 15

        else:

            jump_threshold = kwargs['jump_threshold']

        jump_data = []
        for i in range( len(linearity_data) ):

            jump_data.append( calwebb_detector1.jump_step.JumpStep.call(linearity_data[i], 
                                                                        output_dir=outputfolder+'pipeline_outputs', 
                                                                        save_results = True,
                                                                        rejection_threshold = jump_threshold,
                                                                        maximum_cores = maximum_cores)
                            )

    # Finally, do ramp-fitting:
    ramp_data = []

    if use_tso_jump:
        
        suffix = 'tso_jump'

    else:

        suffix = ''

    output_dictionary['rateints'] = np.array([])
    output_dictionary['errors'] = np.array([])
    output_dictionary['dataquality'] = np.array([])

    for i in range( len(jump_data) ):

        ramp_data.append( calwebb_detector1.ramp_fit_step.RampFitStep.call(jump_data[i], 
                                                                           output_dir=outputfolder+'pipeline_outputs',
                                                                           save_results = True,
                                                                           suffix = suffix )
                        )

        if i == 0:

            output_dictionary['rateints'] = ramp_data[-1][1].data
            output_dictionary['errors'] = ramp_data[-1][1].err
            output_dictionary['dataquality'] = ramp_data[-1][1].dq

        else:

            output_dictionary['rateints'] = np.vstack(( output_dictionary['rateints'], ramp_data[-1][1].data ))
            output_dictionary['errors'] = np.vstack(( output_dictionary['errors'], ramp_data[-1][1].err ))
            output_dictionary['dataquality'] = np.vstack(( output_dictionary['dataquality'], ramp_data[-1][1].dq ))

    # Having finished, pack outputs into the dictionary:
    output_dictionary['times'] = times

    return output_dictionary
