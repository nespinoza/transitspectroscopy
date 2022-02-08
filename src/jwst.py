import os
import numpy as np

from astropy.utils.data import download_file
from astropy import units as u
from astropy.timeseries import TimeSeries

from jwst.pipeline import calwebb_detector1, calwebb_spec2
from jwst import datamodels

def download_reference_file(filename):
    """
    This function downloads a reference file from CRDS given a reference file filename. File gets downloaded to the current working folder.
    """

    print('\n\t >> Downloading {} reference file from CRDS...'.format(filename))
    download_filename = download_file('https://jwst-crds.stsci.edu/unchecked_get/references/jwst/' + filename, cache=True)

    # Rename file:
    os.rename(download_filename, filename)

def stage1(datafile, jump_threshold = 15, get_times = True, get_wavelength_map = True, maximum_cores = 'all', skip_steps = [], outputfolder = '', **kwargs):
    """
    This function calibrates an *uncal.fits file through a "special" version of the JWST TSO CalWebb Stage 1, also passing the data through the assign WCS step to 
    get the wavelength map from Stage 2. With all this, this function by default returns the rates per integrations, errors on those rates, data-quality flags, 
    times, and the wavelength map arising from this calibration. The latter two outputs can be optionally skipped from being calculated/outputted 
    via the `get_times` and `get_wavelength_map` flags.

    In addition to the default flags defined above, "override" flags can be passed as well to pass your own reference files. To pass the bias reference file, for instance, 
    one would do `stage1(datafile, ..., override_superbias = bias_filename)` where `bias_filename` is the location of the superbias reference file that wants to be used.
    

    Parameters
    ----------

    datafile : string
        Input filename; it is expected to be of the form '/your/data/folder/dataname_uncal.fits'
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
    reference_files : list
        List of all the reference files (strings) we will be using for the reduction. These will supercede the default ones used by the pipeline. 
    outputfolder : string
        String indicating the folder where the outputs want to be saved. Default is current working directory.

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

            if not os.path.exists(kwarg):

                rfile = kwargs[kwarg].split('/')[-1]
                download_reference_file(rfile)
                os.rename(rfile, kwargs[kwarg])

    # Lower-case all steps-to-be-skipped:
    for i in range(len(skip_steps)):
    
        skip_steps[i] = skip_steps.lower()

    # Create output dictionary:
    output_dictionary = {}

    #####################################################
    #                       STAGE 1                     #
    #####################################################

    # Create folder that will store pipeline outputs:
    if not os.path.exists(outputfolder+'pipeline_outputs'):
        os.mkdir(outputfolder+'pipeline_outputs')

    # Open the uncal files through a datamodel:
    uncal_data = datamodels.open(datafile)

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
    dataname = datafile.split('/')[-1].split('uncal.fits')[0][:-1]

    # Run steps sequentially. First, the DQInitStep:
    if 'dqinit' not in skip_steps:

        dqinit = calwebb_detector1.dq_init_step.DQInitStep.call(uncal_data, output_dir=outputfolder+'pipeline_outputs', save_results = True)
        output_dictionary['dqinit'] = dqinit

    else:

        output_dictionary['dqinit'] = uncal_data

    # Next, saturation step:
    if 'saturation' not in skip_steps:

        if 'override_saturation' in kwargs.keys():

            saturation = calwebb_detector1.saturation_step.SaturationStep.call(output_dictionary['dqinit'], output_dir=outputfolder+'pipeline_outputs', save_results = True, \
                                                                               override_saturation = kwargs['override_saturation'])

        else:

            saturation = calwebb_detector1.saturation_step.SaturationStep.call(output_dictionary['dqinit'], output_dir=outputfolder+'pipeline_outputs', save_results = True)

        output_dictionary['saturation'] = saturation

    else:

        output_dictionary['saturation'] = output_dictionary['dqinit']

    # Next up, superbias step:
    if 'superbias' not in skip_steps:

        if 'override_superbias' in kwargs.keys():
    
            superbias = calwebb_detector1.superbias_step.SuperBiasStep.call(output_dictionary['saturation'], output_dir=outputfolder+'pipeline_outputs', save_results = True, \
                                                                            override_superbias = kwargs['override_superbias'])

        else:

            superbias = calwebb_detector1.superbias_step.SuperBiasStep.call(output_dictionary['saturation'], output_dir=outputfolder+'pipeline_outputs', save_results = True)

        output_dictionary['superbias'] = superbias

    else:

        output_dictionary['superbias'] = output_dictionary['saturation']

    # Now reference pixel correction:
    if 'refpix' not in skip_steps:

        refpix = calwebb_detector1.refpix_step.RefPixStep.call(output_dictionary['superbias'], output_dir=outputfolder+'pipeline_outputs', save_results = True)
        output_dictionary['refpix'] = refpix

    else:

        output_dictionary['refpix'] = output_dictionary['superbias']

    # Linearity step:
    if 'linearity' not in skip_steps:

        if 'override_linearity' in kwargs.keys():

            linearity = calwebb_detector1.linearity_step.LinearityStep.call(output_dictionary['refpix'], output_dir=outputfolder+'pipeline_outputs', save_results = True, \
                                                                            override_linearity = kwargs['override_linearity'])

        else:

           linearity = calwebb_detector1.linearity_step.LinearityStep.call(output_dictionary['refpix'], output_dir=outputfolder+'pipeline_outputs', save_results = True)

        output_dictionary['linearity'] = linearity

    else:

        output_dictionary['linearity'] = output_dictionary['refpix']

    # DarkCurrent step:
    if 'darkcurrent' not in skip_steps:

        if 'override_darkcurrent' in kwargs.keys():

            darkcurrent = calwebb_detector1.dark_current_step.DarkCurrentStep.call(output_dictionary['linearity'], output_dir=outputfolder+'pipeline_outputs', save_results = True, \
                                                                                   override_dark = kwargs['override_dark'])

        else:

            darkcurrent = calwebb_detector1.dark_current_step.DarkCurrentStep.call(output_dictionary['linearity'], output_dir=outputfolder+'pipeline_outputs', save_results = True)

        output_dictionary['darkcurrent'] = darkcurrent

    else:

        output_dictionary['darkcurrent'] = output_dictionary['linearity']

    # JumpStep:
    if 'jumpstep' not in skip_steps:


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

        output_dictionary['jumpstep'] = output_dictionary['darkcurrent']

    # And finally, the (unskippable) ramp-step:

    if ('override_readnoise' in kwargs.keys()) and ('override_gain' in kwargs.keys()):

        rampstep = alwebb_detector1.ramp_fit_step.RampFitStep.call(output_dictionary['jumpstep'], output_dir=outputfolder+'pipeline_outputs', save_results = True,
                                                                   maximum_cores = maximum_cores,
                                                                   override_readnoise = kwargs['override_readnoise'],
                                                                   override_gain = kwargs['override_gain'])

    elif 'override_readnoise' in kwargs.keys():

        rampstep = alwebb_detector1.ramp_fit_step.RampFitStep.call(output_dictionary['jumpstep'], output_dir=outputfolder+'pipeline_outputs', save_results = True,
                                                                   maximum_cores = maximum_cores,
                                                                   override_readnoise = kwargs['override_readnoise'])

    elif 'override_gain':

        rampstep = alwebb_detector1.ramp_fit_step.RampFitStep.call(output_dictionary['jumpstep'], output_dir=outputfolder+'pipeline_outputs', save_results = True,
                                                                   maximum_cores = maximum_cores,
                                                                   override_gain = kwargs['override_gain'])

    else:

        rampstep = alwebb_detector1.ramp_fit_step.RampFitStep.call(output_dictionary['jumpstep'], output_dir=outputfolder+'pipeline_outputs', save_results = True,
                                                                   maximum_cores = maximum_cores)

    output_dictionary['rampstep'] = rampstep

    # This concludes our passage through Stage 1 (yay!):
    print('\n\t \t \t >> Finished Stage 1!\n')

    #####################################################
    #                       STAGE 2                     #
    #####################################################

    # Alright; now we perform the assign_wcs step to the rates per integration (the so-called "rateint" products):
    assign_wcs = calwebb_spec2.assign_wcs_step.AssignWcsStep.call(rampstep[1], \
                                                                  output_dir=outputfolder+'pipeline_outputs',save_results=True)

    # And get the wavelength map:
    if get_wavelength_map:

        print('\t \t [A.2] Obtaining wavelength maps...')
        wmap_fname = 'wavelength_map'

        if not os.path.exists(outputfolder+'pipeline_outputs/'+wmap_fname+'.npy'):

            rows, columns = assign_wcs.data[0,:,:].shape
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

    # Now we return oututs based on user inputs:

    output_dictionary['rateints'] = assign_wcs.data
    output_dictionary['rateints_err'] = assign_wcs.err
    output_dictionary['rateints_dq'] = assign_wcs.dq

    if get_times:

        output_dictionary['times'] = times + 2400000.5

    if get_wavelength_map:

        output_dictionary['wavelength_maps'] = wavelength_maps

    return output_dictionary
