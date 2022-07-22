import os
import numpy as np
from scipy.sparse.linalg import lsmr
from copy import deepcopy

from astropy.utils.data import download_file
from astropy import units as u
from astropy.time import Time
from astropy.timeseries import TimeSeries

from jwst.pipeline import calwebb_detector1, calwebb_spec2
from jwst import datamodels

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
 

def stage1(datafile, jump_threshold = 15, get_times = True, get_wavelength_map = True, maximum_cores = 'all', preamp_correction = 'stsci', skip_steps = [], outputfolder = '', quicklook = False, uniluminated_mask = None, background_model = None, manual_bias = False, instrument = 'niriss', **kwargs):
    """
    This function calibrates an *uncal.fits file through a "special" version of the JWST TSO CalWebb Stage 1, also passing the data through the assign WCS step to 
    get the wavelength map from Stage 2. With all this, this function by default returns the rates per integrations, errors on those rates, data-quality flags, 
    times, and the wavelength map arising from this calibration. The latter two outputs can be optionally skipped from being calculated/outputted 
    via the `get_times` and `get_wavelength_map` flags.

    In addition to the default flags defined above, "override" flags can be passed as well to pass your own reference files. To pass the bias reference file, for instance, 
    one would do `stage1(datafile, ..., override_superbias = bias_filename)` where `bias_filename` is the location of the superbias reference file that wants to be used.
    
    Note by default, LOOM is used instead of reference pixel correction. This algorithm uses non-iluminated pixels to estimate the odd-even and 1/f simple corrections. To 
    use the pipeline's reference pixel correction, set `loom = False`.

    Parameters
    ----------

    datafile : string
        Input filename; it is expected to be of the form '/your/data/folder/dataname_uncal.fits' or '/your/data/folder/dataname_rateints.fits'. If `rateints`, will ommit Stage 1 processing and will 
        return wavelenght maps and time-stamps only.
    jump_threshold : int
        Number of sigmas used to detect jumps in the `jump` step of the CalWebb pipeline. Default is 15.
    get_times : bool
        If True, mid-times of the integrations are returned in BJD-TDB. Default is True. 
    get_wavelength_map : bool
        If True, a wavelength map will be saved to a file called `wavelength_map_soss.npy`, and also returned by the function. Default is True.
    maximum_cores : string
        If 'all', multiprocessing will be used for the `jump` and `ramp_fit` steps using all available cores.
    skip_steps : list
        List of all the names (strings) of steps we should skip.
    preamp_correction : string
        String defining the method to use to correct for pre-amp reset corrections (i.e., odd/even and one-over-f). Can be 'roeba' for Evertt Schlawlin's ROEBA, 'loom' for the 
        Least-squares Odd-even, One-over-f Model (LOOM) or 'stsci' to let the STScI pipeline handle it through the refpix correction (if not skipped). 
    reference_files : list
        List of all the reference files (strings) we will be using for the reduction. These will supercede the default ones used by the pipeline. 
    quicklook : bool
        If True, pipeline processing stops at the linearity step and returns last-minus-first frames. Default is `False`.
    uniluminated_mask : numpy.array
        (Optional) Array of the same size as the data groups and/or frames. Values of 1 indicate uniluminated pixels, while 0 indicate iluminated pixels. Uniluminated refers to 
        "not iluminated by the main sources in the group/frame". 
    background_model : numpy.array
        (Optional) Array of the same size as the data groups and/or frames containing a model background (if any) to scale.
    outputfolder : string
        (Optional) String indicating the folder where the outputs want to be saved. Default is current working directory.

    Returns
    -------

    output_dictionary : dict
        Dictionary containing by default the `rateints`, `rateints_err` and `rateints_dq` (data-quality flags). For each step name, it also 
        extract the object (which contain the reduced data for each). In addition, the keys `times` and `wavelength_maps` might be returned if 
        flag was set by the user with the time in BJD and wavelength map array (data cube of length (2, 256, 2048) with the wavelength map for 
        order 1 and 2)
    
    """

    # Define output folder if empty:
    if outputfolder != '':
        if outputfolder[-1] != '/':
            outputfolder += '/'

    # Download reference files if not present in the system:
    for kwarg in kwargs.keys():

        if 'override_' in kwarg:

            if not os.path.exists(kwargs[kwarg]):

                rfile = kwargs[kwarg].split('/')[-1]
                download_reference_file(rfile)
                os.rename(rfile, kwargs[kwarg])

    # Lower-case all steps-to-be-skipped:
    for i in range(len(skip_steps)):
    
        skip_steps[i] = skip_steps[i].lower()

    # Lower-case pre-amp reset correction:
    preamp_correction = preamp_correction.lower()
    if preamp_correction not in ['roeba', 'loom', 'stsci']:

        raise Exception('The preamp_correction flag has to be either "roeba", "loom" or "stsci" --- "'+str(preamp_correction)+'" is not a valid flag.')

    # Create output dictionary:
    output_dictionary = {}

    #####################################################
    #                       STAGE 1                     #
    #####################################################

    # Create folder that will store pipeline outputs:
    if not os.path.exists(outputfolder+'pipeline_outputs'):
        os.mkdir(outputfolder+'pipeline_outputs')

    # Open the uncal files through a datamodel:
    uncal_flag = True
    if 'rateints' in datafile:

        uncal_data = datamodels.open(datafile)
        uncal_flag = False

    else:

        uncal_data = datamodels.RampModel(datafile)

    # This fixes a bug in some simulated datasets:
    try:

        uncal_data.meta.dither.dither_points = int(uncal_data.meta.dither.dither_points)

    except:

        print('\n\t \t >> Warning: model.meta.dither.dither_points gave ', uncal_data.meta.dither.dither_points)
        print('\n\t \t >> Setting manually to 1.')
        uncal_data.meta.dither.dither_points = 1

    # Extract times from uncal file:
    if get_times:

        try:

            times = uncal_data.int_times['int_mid_BJD_TDB']

        except:

            # If no time-stamps, generate our own. Note they will be in UTC; need to conver to BJD later.
            # First, we extract the frame-time. We assume NISRAPID sampling here, where t_group = t_frame:
            frametime = uncal_data.meta.exposure.group_time # seconds

            time_start = uncal_data.meta.observation.date + 'T' + uncal_data.meta.observation.time

            print('\n\t \t >> WARNING: time-stamps not found on uncal file. '+\
                  'Filling time-stamps starting from the DATE-OBS and TIME-OBS on PRIMARY header: '+time_start+', assuming a '+str(frametime)+'s group-time.')
            print('\n\t \t >> NOTE THIS WILL SAVE TIME-STAMPS IN UTC JD!')

            nintegrations, ngroups = uncal_data.meta.exposure.nints, uncal_data.meta.exposure.ngroups
    
            # Generate time-series stamps; delta between integrations is (Frametime) x (Ngroups + 1) --- this accounts for resets.
            ts = TimeSeries(time_start=time_start, time_delta = frametime * (ngroups + 1)  * u.s, data = {'flux': np.ones(nintegrations)})

            # Generate time-stamps in JD. Add factor to JD-timestamps so the stamps are mid-integration:
            second_to_day = 1. / (24. * 3600.)
            #      v-- orig---v      v--skip reset--v              v---- mid-up-the-ramp  ----v
            times = ts.time.jd + frametime * second_to_day + (frametime * ngroups) * 0.5 * second_to_day

            times = times - 2400000.5

            print('\n \t \t >> First time-stamp (- 2400000.5):' + str(times[0]) + '; last one: ' + str(times[-1]))


    # Extract filename before *uncal:
    if uncal_flag:

        dataname = datafile.split('/')[-1].split('uncal.fits')[0][:-1]

    else:

        dataname = datafile.split('/')[-1].split('rateints.fits')[0][:-1]

    full_datapath = outputfolder+'pipeline_outputs' + '/' + dataname

    if uncal_flag:

        # Run steps sequentially. First, the DQInitStep:
        if 'dqinit' not in skip_steps:

            output_filename = full_datapath + '_dqinitstep.fits'
            if not os.path.exists(output_filename):

                dqinit = calwebb_detector1.dq_init_step.DQInitStep.call(uncal_data, output_dir=outputfolder+'pipeline_outputs', save_results = True)
                output_dictionary['dqinit'] = dqinit

            else:

                print('\t >> dqinit step products found, loading them...')
                output_dictionary['dqinit'] = datamodels.RampModel(output_filename)

        else:

            output_dictionary['dqinit'] = uncal_data

        # Next, saturation step:
        if 'saturation' not in skip_steps:

            output_filename = full_datapath + '_saturationstep.fits'
            if not os.path.exists(output_filename):

                if 'override_saturation' in kwargs.keys():

                    saturation = calwebb_detector1.saturation_step.SaturationStep.call(output_dictionary['dqinit'], output_dir=outputfolder+'pipeline_outputs', save_results = True, \
                                                                                       override_saturation = kwargs['override_saturation'])

                else:

                    saturation = calwebb_detector1.saturation_step.SaturationStep.call(output_dictionary['dqinit'], output_dir=outputfolder+'pipeline_outputs', save_results = True)

                output_dictionary['saturation'] = saturation

            else:

                print('\t >> saturation step products found, loading them...')
                output_dictionary['saturation'] = datamodels.RampModel(output_filename)

        else:

            output_dictionary['saturation'] = output_dictionary['dqinit']

        # Next up, superbias step:
        if 'superbias' not in skip_steps:

            if not manual_bias: 

                output_filename = full_datapath + '_superbiasstep.fits'
                if not os.path.exists(output_filename):

                    if 'override_superbias' in kwargs.keys():

                        superbias = calwebb_detector1.superbias_step.SuperBiasStep.call(output_dictionary['saturation'], output_dir=outputfolder+'pipeline_outputs', save_results = True, \
                                                                                        override_superbias = kwargs['override_superbias'])

                    else:

                        superbias = calwebb_detector1.superbias_step.SuperBiasStep.call(output_dictionary['saturation'], output_dir=outputfolder+'pipeline_outputs', save_results = True)

                    output_dictionary['superbias'] = superbias

                else:

                    print('\t >> superbias step products found, loading them...')
                    output_dictionary['superbias'] = datamodels.RampModel(output_filename)

            else:

                output_dictionary['superbias'] = output_dictionary['saturation']
                sbias = datamodels.open(kwargs['override_superbias'])
                output_dictionary['superbias'].data -= sbias.data

        else:

            output_dictionary['superbias'] = output_dictionary['saturation']

        # Now reference pixel correction:
        if 'refpix' not in skip_steps:

            if preamp_correction == 'stsci':

                output_filename = full_datapath + '_refpixstep.fits'
                if not os.path.exists(output_filename):

                    refpix = calwebb_detector1.refpix_step.RefPixStep.call(output_dictionary['superbias'], output_dir=outputfolder+'pipeline_outputs', save_results = True)
                    output_dictionary['refpix'] = refpix

                else:

                    print('\t >> refpix step products found, loading them...')
                    output_dictionary['refpix'] = datamodels.RampModel(output_filename)

        else:

            output_dictionary['refpix'] = output_dictionary['superbias']

        # Get some important data out of the current data:
        nintegrations, ngroups, nrows, ncolumns = output_dictionary['superbias'].data.shape

        min_group, max_group = 0, ngroups - 1 

        if 'min_group' in kwargs.keys():

            min_group = kwargs['min_group']

        if 'max_group' in kwargs.keys():

            max_group = kwargs['max_group'] 

        if (preamp_correction == 'loom') or (preamp_correction == 'roeba'):

            lmf, median_lmf = get_last_minus_first(output_dictionary['superbias'].data, min_group = min_group, max_group = max_group)

            # Generate mask of uniluminated pixels from the median last-minus-first frame, if not available:
            if uniluminated_mask is None:

                mask = get_uniluminated_mask(median_lmf, pixeldq = output_dictionary['superbias'].pixeldq)

            else:

                mask = uniluminated_mask

            # Now go through each group, and correct 1/f and odd-even with the LOOM: cc_uniluminated_outliers(data, mask, nsigma = 5)
            refpix = deepcopy(output_dictionary['superbias'])
            lmf_after = np.zeros(lmf.shape)

            if preamp_correction == 'loom':

                output_filename = full_datapath + '_refpixstep_loom.fits'
                if not os.path.exists(output_filename):
        
                    looms = np.zeros([nintegrations, ngroups, nrows, ncolumns])
                    for i in range(nintegrations):
                
                        for j in range(ngroups):
                
                            # Get mask with group-dependant outliers:
                            group_mask = cc_uniluminated_outliers(refpix.data[i, j, :, :], mask)

                            # Get LOOM estimate:
                            looms[i, j, :, :] = get_loom(refpix.data[i, j, :, :], group_mask, background = background_model)

                            # Substract it from the data:
                            refpix.data[i, j, :, :] -= looms[i, j, :, :]

                        lmf_after[i, :, :] = refpix.data[i, max_group, :, :] - refpix.data[i, min_group, :, :]

                    refpix.save(output_filename)

                else:

                    print('\t >> refpix LOOM step products found, loading them...')
                    refpix = datamodels.RampModel(output_filename)
                    lmf_after, _ = get_last_minus_first(refpix.data, min_group = min_group, max_group = max_group)

                full_datapath += '_refpixstep_loom'

            if preamp_correction == 'roeba':

                output_filename = full_datapath + '_refpixstep_roeba.fits'
                if not os.path.exists(output_filename):

                    roebas = np.zeros([nintegrations, ngroups, nrows, ncolumns])

                    # Do some special treatment for 512s, which doesn't have many background pixels:
                    if instrument == 'nirspec-512s':

                        # First, remove possible bias jumps from each group using the background regions to the left and right of the detector:
                        for i in range(nintegrations):

                            for j in range(ngroups):

                                background = np.hstack(( refpix.data[i, j, :, 0:30], refpix.data[i, j, :, 500:] ))
                                refpix.data[i, j, :, :] -= np.nanmedian( background )

                    else:

                        for i in range(nintegrations):

                            for j in range(ngroups):

                                # Create group-mask to scale and remove the background model if available:
                                if background_model is not None:

                                    group_mask = cc_uniluminated_outliers(refpix.data[i, j, :, :], mask)

                                # ROEBA is outlier-resistant, so don't bother with group-masks:
                                roebas[i, j, :, :] = get_roeba(refpix.data[i, j, :, :], mask)

                                # Substract from the data:
                                refpix.data[i, j, :, :] -= roebas[i, j, :, :]

                            lmf_after[i, :, :] = refpix.data[i, max_group, :, :] - refpix.data[i, min_group, :, :]

                    refpix.save(output_filename)

                else:

                    print('\t >> refpix ROEBA step products found, loading them...')
                    refpix = datamodels.RampModel(output_filename)
                    lmf_after, _ = get_last_minus_first(refpix.data, min_group = min_group, max_group = max_group)

                full_datapath += '_refpixstep_roeba'

            # Save results:
            output_dictionary['mask'] = mask
            output_dictionary['refpix'] = refpix
            output_dictionary['lmf_before'] = lmf 
            output_dictionary['lmf_after'] = lmf_after

        # Linearity step:
        if 'linearity' not in skip_steps:

            output_filename = full_datapath + '_linearitystep.fits'
            if not os.path.exists(output_filename):

                if 'override_linearity' in kwargs.keys():

                    linearity = calwebb_detector1.linearity_step.LinearityStep.call(output_dictionary['refpix'], output_dir=outputfolder+'pipeline_outputs', save_results = True, \
                                                                                    override_linearity = kwargs['override_linearity'])

                else:

                   linearity = calwebb_detector1.linearity_step.LinearityStep.call(output_dictionary['refpix'], output_dir=outputfolder+'pipeline_outputs', save_results = True)

                output_dictionary['linearity'] = linearity

            else:

                print('\t >> linearity step products found, loading them...')
                output_dictionary['linearity'] = datamodels.RampModel(output_filename)

        else:

            output_dictionary['linearity'] = output_dictionary['refpix']

        # Generate LMF after linearity correction:
        output_dictionary['linearity_lmf'], _ = get_last_minus_first(output_dictionary['linearity'].data, min_group = min_group, max_group = max_group) 

        if quicklook:

            return output_dictionary

        # DarkCurrent step:
        if 'darkcurrent' not in skip_steps:

            output_filename = full_datapath + '_darkcurrentstep.fits'
            if not os.path.exists(output_filename):

                if 'override_darkcurrent' in kwargs.keys():

                    darkcurrent = calwebb_detector1.dark_current_step.DarkCurrentStep.call(output_dictionary['linearity'], output_dir=outputfolder+'pipeline_outputs', save_results = True, \
                                                                                           override_dark = kwargs['override_dark'])

                else:

                    darkcurrent = calwebb_detector1.dark_current_step.DarkCurrentStep.call(output_dictionary['linearity'], output_dir=outputfolder+'pipeline_outputs', save_results = True)

                output_dictionary['darkcurrent'] = darkcurrent

            else:
                
                print('\t >> darkcurrent step products found, loading them...')
                output_dictionary['darkcurrent'] = datamodels.RampModel(output_filename)

        else:

            output_dictionary['darkcurrent'] = output_dictionary['linearity']

        # JumpStep:
        if 'jumpstep' not in skip_steps:

            output_filename = full_datapath + '_jumpstep.fits'
            if not os.path.exists(output_filename):

                if ('override_readnoise' in kwargs.keys()) and ('override_gain' in kwargs.keys()):

                    jumpstep = calwebb_detector1.jump_step.JumpStep.call(output_dictionary['darkcurrent'], output_dir=outputfolder+'pipeline_outputs', save_results = True,
                                                                         rejection_threshold=jump_threshold,
                                                                         maximum_cores = maximum_cores, 
                                                                         override_readnoise = kwargs['override_readnoise'],
                                                                         override_gain = kwargs['override_gain'])

                elif 'override_readnoise' in kwargs.keys():

                    jumpstep = calwebb_detector1.jump_step.JumpStep.call(output_dictionary['darkcurrent'], output_dir=outputfolder+'pipeline_outputs', save_results = True,
                                                                         rejection_threshold=jump_threshold,
                                                                         maximum_cores = maximum_cores, 
                                                                         override_readnoise = kwargs['override_readnoise'])

                elif 'override_gain' in kwargs.keys():

                    jumpstep = calwebb_detector1.jump_step.JumpStep.call(output_dictionary['darkcurrent'], output_dir=outputfolder+'pipeline_outputs', save_results = True,
                                                                         rejection_threshold=jump_threshold,
                                                                         maximum_cores = maximum_cores, 
                                                                         override_gain = kwargs['override_gain'])

                else:

                    jumpstep = calwebb_detector1.jump_step.JumpStep.call(output_dictionary['darkcurrent'], output_dir=outputfolder+'pipeline_outputs', save_results = True, 
                                                                         rejection_threshold=jump_threshold,
                                                                         maximum_cores = maximum_cores)

                output_dictionary['jumpstep'] = jumpstep

            else:

                print('\t >> jump step products found, loading them...')
                output_dictionary['jumpstep'] = datamodels.RampModel(output_filename)

        else:

            output_dictionary['jumpstep'] = output_dictionary['darkcurrent']

        # And finally, the (unskippable) ramp-step:

        output_filename0 = full_datapath + '_0_rampfitstep.fits'
        output_filename1 = full_datapath + '_1_rampfitstep.fits'

        if not os.path.exists(output_filename0):

            if ('override_readnoise' in kwargs.keys()) and ('override_gain' in kwargs.keys()):

                rampstep = calwebb_detector1.ramp_fit_step.RampFitStep.call(output_dictionary['jumpstep'], output_dir=outputfolder+'pipeline_outputs', save_results = True,
                                                                           maximum_cores = maximum_cores,
                                                                           override_readnoise = kwargs['override_readnoise'],
                                                                           override_gain = kwargs['override_gain'])

            elif 'override_readnoise' in kwargs.keys():

                rampstep = calwebb_detector1.ramp_fit_step.RampFitStep.call(output_dictionary['jumpstep'], output_dir=outputfolder+'pipeline_outputs', save_results = True,
                                                                           maximum_cores = maximum_cores,
                                                                           override_readnoise = kwargs['override_readnoise'])

            elif 'override_gain' in kwargs.keys():

                rampstep = calwebb_detector1.ramp_fit_step.RampFitStep.call(output_dictionary['jumpstep'], output_dir=outputfolder+'pipeline_outputs', save_results = True,
                                                                           maximum_cores = maximum_cores,
                                                                           override_gain = kwargs['override_gain'])

            else:

                rampstep = calwebb_detector1.ramp_fit_step.RampFitStep.call(output_dictionary['jumpstep'], output_dir=outputfolder+'pipeline_outputs', save_results = True,
                                                                           maximum_cores = maximum_cores)

        else:

            print('\t >> rampfit step products found, loading them...')
            rampstep = [datamodels.open(output_filename0), datamodels.open(output_filename1)]

        output_dictionary['rampstep'] = rampstep

    else:

        rampstep = [[], uncal_data]

    # This concludes our passage through Stage 1 (yay!):
    print('\n\t \t \t >> Finished Stage 1!\n')

    #####################################################
    #                       STAGE 2                     #
    #####################################################

    # Alright; now we perform the assign_wcs step to the rates per integration (the so-called "rateint" products):
    output_filename = full_datapath + '_1_assignwcsstep.fits'

    if not os.path.exists(output_filename):

        assign_wcs = calwebb_spec2.assign_wcs_step.AssignWcsStep.call(rampstep[1], \
                                                                      output_dir=outputfolder+'pipeline_outputs',save_results=True)

    else:

        assign_wcs = datamodels.open(output_filename)

    # And get the wavelength map:
    if get_wavelength_map:

        print('\t \t [A.2] Obtaining wavelength maps...')
        wmap_fname = 'wavelength_map'

        if not os.path.exists(outputfolder+'pipeline_outputs/'+wmap_fname+'.npy'):

            rows, columns = assign_wcs.data[0,:,:].shape

            if instrument == 'niriss':

                wavelength_maps = np.zeros([2,rows,columns])
                for order in [1,2]:
                    for row in range(rows):
                        for column in range(columns):
                            wavelength_maps[order-1,row,column] = assign_wcs.meta.wcs(column, row, order)[-1]

                # Save it so we do this only once:
                np.save(outputfolder+'pipeline_outputs/'+wmap_fname, wavelength_maps)

        else:

            print('\t \t \t >> Detected wavelength map; loading it...')
            wavelength_maps = np.load(outputfolder+'pipeline_outputs/'+wmap_fname+'.npy')

    print('\n\t \t [A] Successfully finished JWST calibration. \n')

    # Clean output dictionary before returning results:
    for skipped in skip_steps:

        if skipped in list(output_dictionary.keys()):
        
            output_dictionary.pop(skipped)

    # Now we return outputs based on user inputs:

    output_dictionary['rateints'] = assign_wcs.data
    output_dictionary['rateints_err'] = assign_wcs.err
    output_dictionary['rateints_dq'] = assign_wcs.dq

    if get_times:

        output_dictionary['times'] = times + 2400000.5

    if get_wavelength_map:

        output_dictionary['wavelength_maps'] = wavelength_maps

    return output_dictionary
