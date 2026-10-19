import os
import shutil
import glob
import time
import pickle
import numpy as np
from scipy.sparse.linalg import lsmr
from copy import copy, deepcopy
from scipy.ndimage import median_filter

from ._optional import OptionalModule as _OptionalModule
pd = _OptionalModule('pandas')
from tqdm import tqdm
from astropy.utils.data import download_file
from astropy import units as u
from astropy.time import Time
from astropy.timeseries import TimeSeries

calwebb_detector1 = _OptionalModule('jwst.pipeline.calwebb_detector1')
calwebb_spec2 = _OptionalModule('jwst.pipeline.calwebb_spec2')
assign_wcs = _OptionalModule('jwst.assign_wcs')
datamodels = _OptionalModule('stdatamodels.jwst.datamodels')
wcstools = _OptionalModule('gwcs.wcstools')

def __getattr__(name):
    if name == 'jwstversion':
        return _OptionalModule('jwst').__version__
    raise AttributeError(name)

from .utils import *
from .spectroscopy import *
from .timeseries import *
from . import instruments
from .config import DEPRECATED_STAGE2_KEYWORDS, STAGE2_ARGUMENTS, ExtractionConfig
from .core.detector import (cc_uniluminated_outliers, correct_1f, get_last_minus_first,  # noqa: F401
                            get_loom, get_roeba, get_uniluminated_mask, group_1f_correct,
                            spill_filter, subtract_side_pedestal, tso_jump_flags)

from importlib.util import find_spec as _find_spec
try:
    ray_is_installed = _find_spec('ray') is not None
except (ImportError, ValueError):
    ray_is_installed = False
ray = _OptionalModule('ray')

# A class-like lazy alias preserves Observations.query_criteria, etc.
class _ObservationsProxy:
    def __getattr__(self, name):
        return getattr(_OptionalModule('astroquery.mast').Observations, name)

Observations = _ObservationsProxy()

def download(pid, obs_num, mast_api_token = None, outputfolder = None, data_product = 'uncal', exposure = None, segments = None, detectors = None, skip_existing = True):
    """
    Given a program ID and an observation number, this function downloads all the data associated with that PID and program.

    Example usage:

        >>> ts.jwst.download(pid = 1118, obs_num = '5')

    If data is propietary, one needs to ingest a MAST API token as follows:

        >>> ts.jwst.download(pid = 1118, obs_num = '5', mast_api_token = 'asdfg')

    Parameters
    ----------

    pid : int or str
        Number (`int` or `str`) indicating the Program ID.
   
    obs_num : int or str
        Number (`int` or `str`) indicating the observation number.

    mast_api_token : str
        (Optional) MAST API token; can be obtained from https://auth.mast.stsci.edu/info.

    outputfolder : str
        (Optional) Folder where the data will be stored. If not defined, data will be stored in a local folder called `JWSTdata`

    data_product : str
        (Optional) String representing the kind of data products the user wants; `uncal` will download the uncalibrated products, 
        `ramp` will download the calibrated ramps, `rateints` will download the rates per integration. Default is to download `uncal` products.
    exposure : str or list of str
        (Optional) Only download files of this exposure (or these exposures), e.g. `'jw01118005001_04101_00001'`.
    segments : list of int
        (Optional) Only download these segments (1 is `-seg001`).
    detectors : list of str
        (Optional) Only download these detectors, e.g. `['nrs1']`.
    skip_existing : bool
        (Optional) Do not download again files already in `outputfolder` (also as `.fits.gz`). Default is True.

    Returns
    -------
    filenames : list
        Paths of the selected files in `outputfolder` (downloaded now or already present).

    """

    pid = int(pid)
    obs_num = int(obs_num)

    if outputfolder is None:

        outputfolder = 'JWSTdata'

    if mast_api_token is not None:

        Observations.login(token=mast_api_token)

    data_product = data_product.upper()
    if data_product == 'RAMPS':
        data_product = 'RAMP'

    if data_product not in ['UNCAL', 'RAMP', 'RATEINTS', 'DARK']:

        raise Exception("Error: data product "+data_product+" not recognized. It has to be 'uncal', 'ramps' or 'rateints'")

    # Perform a query for the observations:
    observation = Observations.query_criteria(proposal_id = str(pid), obs_collection = 'JWST')

    # Cut products by the observation number:
    data_products = Observations.get_product_list(observation)

    indexes = []
    for i in range( len(data_products) ):

        # Check location of observation number
        if data_products[i]['obs_id'][7:10].isdigit(): 
            
            # Extract observation number:
            data_obs_num = int(data_products[i]['obs_id'][7:10])
    
            # Cut datasets by observation number, not FGS data::
            if ( data_obs_num == obs_num ) and ( 'FGS' not in data_products[i]['description'] ):
    
                # Cut further based on user's choice for data product; this will select both 2D spectra and/or imaging, plus TA frames:
                if data_product == 'UNCAL':
    
                    if data_products[i]['productSubGroupDescription'] == 'UNCAL' and data_products[i]['productType'] == 'SCIENCE':
                            
                        indexes.append(i)
    
                elif data_product == 'RAMP':
    
                    if data_products[i]['productSubGroupDescription'] == 'RAMP' and data_products[i]['productType'] == 'AUXILIARY':
    
                        indexes.append(i)
    
                elif data_product == 'RATEINTS':
    
                    if data_products[i]['productSubGroupDescription'] == 'RATEINTS' and data_products[i]['productType'] == 'SCIENCE':
    
                        indexes.append(i)
    
                elif data_product == 'DARK':
    
                    if data_products[i]['productSubGroupDescription'] == 'DARK' and data_products[i]['productType'] == 'AUXILIARY':
    
                        indexes.append(i)

    # Apply mask:
    data_products = data_products[indexes]

    def local_file(name):
        # An already downloaded file, possibly compressed afterwards:
        for candidate in (outputfolder+'/'+name, outputfolder+'/'+name+'.gz'):
            if os.path.exists(candidate):
                return candidate
        return None

    # Optional selection of exposure, segments and detectors from the file names
    # (e.g., jw01118005001_04101_00001-seg001_nrs1_uncal.fits):
    exposures = [exposure] if isinstance(exposure, str) else exposure

    def selected(filename):
        name = filename.lower()
        if exposures is not None and not any(name.startswith(e.lower()) for e in exposures):
            return False
        if segments is not None and not any('-seg{0:03d}'.format(int(segment)) in name for segment in segments):
            return False
        if detectors is not None and not any('_'+detector.lower()+'_' in name for detector in detectors):
            return False
        return True

    if 'productFilename' in data_products.colnames:

        data_products = data_products[[selected(str(name)) for name in data_products['productFilename']]]
        wanted = sorted(str(name) for name in data_products['productFilename'])

        if skip_existing:

            data_products = data_products[[local_file(str(name)) is None for name in data_products['productFilename']]]

    elif exposure is not None or segments is not None or detectors is not None:

        raise ValueError('MAST product list has no productFilename column; cannot select files')

    else:

        wanted = None

    if len(data_products) == 0 and wanted is not None:

        print('\n\t >> All {0:} selected files are already in {1:}.'.format(len(wanted), outputfolder))
        return [local_file(name) for name in wanted]

    # Report final size of the package with the user:
    print( '\n\t >> {0:.2f} GB of data will be downloaded in total considering the following files:\n'.format( data_products['size'].sum()/1e9 ) )
    print(data_products)
    print( '\n\t >> Downloading...\n' )

    # Download:
    Observations.download_products(data_products, download_dir = outputfolder)

    # Get those files out of hiding:
    folders = glob.glob(outputfolder+'/mastDownload/JWST/*')
    fnames = []

    for folder in folders:

        files = glob.glob(folder+'/*.fits')

        for file in files:

            fname = file.split('/')[-1]
            shutil.move(file, outputfolder+'/'+fname)
            fnames.append(fname)

    shutil.rmtree(outputfolder+'/mastDownload')
    fnames.sort()

    # Print some handy info about the downloaded files to the user:
    print('\n\t >> ...done! Downloaded and stored the following files on the '+outputfolder+' folder:\n')

    # Pack outputs in lists so we can then ingest them into a dataframe:

    filenames = []
    description = []
    detector_names = []
    subarrays = []
    filters = []
    exptypes = []
    gratings = []

    for fname in fnames:

        dm = datamodels.open(outputfolder+'/'+fname)

        filenames.append(fname)
        description.append(dm.meta.observation.observation_label)
        detector_names.append(dm.meta.instrument.detector)
        subarrays.append(dm.meta.subarray.name)
        filters.append(dm.meta.instrument.filter)
        exptypes.append(dm.meta.exposure.type)

        if dm.meta.instrument.grating is not None:

            gratings.append(dm.meta.instrument.grating)

        elif dm.meta.instrument.pupil is not None:

            gratings.append(dm.meta.instrument.pupil)

        else:

            gratings.append('N/A')

    downloaded_files = { 'Filename': filenames,
                         'Description': description,
                         'Detector': detector_names,
                         'Grating/Pupil' : gratings,
                         'Subarray': subarrays,
                         'Filter': filters,
                         'Exposure type': exptypes
                       }

    # Convert to pandas for neater output to user:
    pd.options.display.max_colwidth = 100
    downloaded_files = pd.DataFrame(downloaded_files)
    print(downloaded_files)

    return [local_file(name) or outputfolder+'/'+name for name in (wanted if wanted is not None else fnames)]

    print('\n\t >> Data gathered on: ', dm.meta.observation.date,' | PI: ', dm.meta.program.pi_name)

class load(object):
    """
    Given a list of JWST filenames, understood to be Time Series Observations (TSOs), this class loads JWST data into an object which can then be 
    manipulated to perform data calibration and reduction. 

    Example usage:

            >>> dataset = ts.jwst.load(filenames)

    Parameters
    ----------

    filenames : list
        List of strings containing the paths to the different segments of the TSO (e.g., `filenames = ['jw01331104001_04101_00001-seg001_nrs1_uncal.fits', 
        'jw01331104001_04101_00001-seg002_nrs1_uncal.fits']`). Input files can come from any of the JWST pipeline products in Stage 1.
   
    outputfolder : str
        (Optional) folder where the outputs will be saved to in case this was note defined before. By default, a folder `ts_outputs` is created in that folder. 
        Default is the current working directory.
 
    """

    def check_dataset(self, fnames = None, single_input = False):
        """
        This function checks if the input data (a) is either a ramp (i.e., we have integrations and groups) or a rateint (i.e., only integrations) and (b) has a common model 
        (i.e. they are all ramps or rateints).
        """

        if fnames is None:

            fnames = self.filenames

        # Check first element first:
        datatype = fnames[0].split('_')[-1].split('.fits')[0]
        if datatype == 'rampfitstep' or datatype == 'rateints' or datatype == 'fitstep':

            first_datatype = 'rates per integration'

        else:

            first_datatype = 'ramps'

        # Check you get the same result for all the filenames. If not, raise exception to user:
        if not single_input:

            for i in range(1, len(fnames)):

                current_datatype = self.check_dataset(fnames = [fnames[i]], single_input = True)
                if current_datatype != first_datatype:

                    raise Exception("Filename "+fnames[0]+" seem to be "+first_datatype+", whereas filename "+fnames[i]+" seem to be "+current_datatype+". \n"+\
                                    "Files ending in *rampfitstep.fits or *rateints.fits are detected by transitspectroscopy as rates per integration; all the rest as ramps. "+\
                                    "Be sure to set all input files to ramps or rates per integration and try loading again.")
       
        # If all goes well, set datatype attribute:
        return first_datatype 

    def sort_filenames(self):

        # Sort by exposure (the file name before '-seg'), then by segment number. Within
        # one exposure this is the segment order:
        keys = []
        for i in range( len(self.filenames) ):

            name = os.path.basename(self.filenames[i])
            keys.append( (name.split('-seg')[0], int( name.split('-seg')[1][:3] )) )

        idx = sorted(range(len(keys)), key = lambda i: keys[i])
        self.filenames = np.array(self.filenames)[idx]

    def check_status(self, data):

        self.status = {}

        # Common checks for all TSOs (https://jwst-pipeline.readthedocs.io/en/latest/jwst/pipeline/calwebb_detector1.html):
        self.status['dq_init'] = data.meta.cal_step.dq_init
        self.status['saturation'] = data.meta.cal_step.saturation
        self.status['refpix'] = data.meta.cal_step.refpix
        self.status['linearity'] = data.meta.cal_step.linearity
        self.status['dark_sub'] = data.meta.cal_step.dark_sub
        self.status['jump'] = data.meta.cal_step.jump
        self.status['clean_flicker_noise'] = getattr(data.meta.cal_step, 'clean_flicker_noise', None)
        self.status['ramp_fit'] = data.meta.cal_step.ramp_fit
        self.status['tracing'] = False
        self.status['centroiding'] = False

        # Extract instrument/mode:
        self.instrument = data.meta.instrument.name.lower()
        self.filter = str(data.meta.instrument.filter).lower()

        if data.meta.instrument.grating is not None:
        
            self.dispersive_element = data.meta.instrument.grating.lower()

        elif data.meta.instrument.pupil is not None:
        
            self.dispersive_element = data.meta.instrument.pupil.lower()

        else:

            self.dispersive_element = 'None'

        # Different checks for NIR and MIRI:
        if self.instrument in ['nirspec', 'nircam', 'niriss']:

            # Following https://jwst-pipeline.readthedocs.io/en/latest/jwst/pipeline/calwebb_detector1.html:
            self.status['superbias'] = data.meta.cal_step.superbias

        else:

            # Following https://jwst-pipeline.readthedocs.io/en/latest/jwst/pipeline/calwebb_detector1.html:
            self.status['reset'] = data.meta.cal_step.reset

            # Cover in case user is using an older version of the JWST pipeline:
            steplist = dir(data.meta.cal_step)
            if 'emicorr' not in steplist:

                print('Warning! Using version ', __getattr__('jwstversion'), ' of the JWST Calibration pipeline, which does NOT have the emicorr step. Upgrade to run the step.')
                self.status['emicorr'] = None

            else:

                self.status['emicorr'] = data.meta.cal_step.emicorr

        # Take the chance to populate the mode (see transitspectroscopy.instruments):
        self.mode_definition = instruments.identify(instrument = self.instrument,
                                                    dispersive_element = self.dispersive_element,
                                                    filter = self.filter,
                                                    exp_type = getattr(getattr(data.meta, 'exposure', None), 'type', None))
        self.mode = self.mode_definition.name
        print('\t    - Instrument/Mode: {0:}\n'.format(self.mode_definition.describe(self.filter)))

    def fill_calibration_parameters(self, parameters = {}, use_tso_jump = True, group_1f = False, save = True,
                                    group_difference_1f = False):
        """
        This function fills in the `self.calibration_parameters` dictionary which holds all the calibration parameters.

        Parameters
        ----------

        parameters : dict
            Input dictionary defining the parameters for each step in the pipeline, e.g., `parameters['refpix']['left_pixels'] = 10`.

        use_tso_jump: bool
            (Optional) Boolean deciding if to apply or not the TSO jump step. Default is `True`; if `False` the JWST pipeline jump step is used.

        group_1f : bool
            (Optional) Decide whether to use or not the group-level 1/f correction algorithm. Default is `False`.

        save : bool
            (Optional) Decide if save by-products of the pipeline to disk. Default is `True`.

        group_difference_1f : bool
            (Optional) Decide whether `fit_ramps` applies the group-difference 1/f correction
            (see `group_difference_1f_correction`), for modes that support it. If applied, it
            replaces the `clean_flicker_noise` step of NIRISS/SOSS. Default is `False` here;
            `detector_calibration` defaults to `True`.

        """
        if not group_1f:

            self.status['group_1f'] = 'SKIPPED'

        else:

            self.status['group_1f'] = None

        mode_steps = self.mode_definition.detector_steps
        if group_difference_1f and mode_steps.get('group_difference_1f', False):

            self.status['group_difference_1f'] = None

            # It replaces clean_flicker_noise (NIRISS/SOSS):
            if ('clean_flicker_noise' in mode_steps.get('post_jump_steps', ())
                    and self.status.get('clean_flicker_noise') is None):

                self.status['clean_flicker_noise'] = 'SKIPPED'

        else:

            self.status['group_difference_1f'] = 'SKIPPED'

        # Fill empty parameter sets for steps for which no input parameters were given:
        for step in list( self.status.keys() ):

            if step in list( parameters.keys() ):

                self.calibration_parameters[step] = parameters[step]

            else:

                self.calibration_parameters[step] = {}

        # Check which parameters want to be skipped, if any:
        if 'skip' in parameters.keys():

            self.calibration_parameters['skip'] = parameters['skip']

        else:

            self.calibration_parameters['skip'] = {}

        # Set saving parameters if save is True:
        if save:

            for step in ['linearity', 'jump', 'ramp_fit']:

                self.calibration_parameters[step]['output_dir'] = self.outputfolder+'ts_outputs'
                self.calibration_parameters[step]['save_results'] = True
                self.calibration_parameters[step]['suffix'] = self.actual_suffix+step+'step'

        # Set common parameters for various steps for different instrument/modes:
        steps = self.mode_definition.detector_steps

        # Reference pixel parameters:
        for key, value in steps.get('refpix_parameters', {}).items():

            if key not in self.calibration_parameters['refpix'].keys():

                self.calibration_parameters['refpix'][key] = value

        # Tracing parameters (window of the median filter that fills NaNs):
        if steps.get('nan_fill_window') is not None:

            if 'row_window' not in self.calibration_parameters['tracing'].keys():

                self.calibration_parameters['tracing']['row_window'] = steps['nan_fill_window']['row_window']
                self.calibration_parameters['tracing']['column_window'] = steps['nan_fill_window']['column_window']

        # If TSO jump is going to be used, remove some dict keys and set parameters:
        if use_tso_jump:

            if save:

                # Pop elements defined above:
                [self.calibration_parameters['jump'].pop(key) for key in ['output_dir', 'save_results', 'suffix']]

            # Define parameters:
            for key in ['window', 'nsigma']:

                if key not in self.calibration_parameters['jump'].keys():

                    self.calibration_parameters['jump'][key] = steps['tso_jump'][key]

        else:

            # `jump_threshold` is this package's name for the JumpStep `rejection_threshold`
            # (as in `stage1`); an explicit `rejection_threshold` takes precedence:
            jump_threshold = self.calibration_parameters['jump'].pop('jump_threshold', 15)
            self.calibration_parameters['jump'].setdefault('rejection_threshold', jump_threshold)

    def set_outputfolder(self, outputfolder):

        self.outputfolder = outputfolder

        # Define output folder if empty:
        if self.outputfolder != '':
            if self.outputfolder[-1] != '/':
                self.outputfolder += '/'

        # Create folder that will store outputs:
        if not os.path.exists(self.outputfolder+'ts_outputs'):
            os.mkdir(self.outputfolder+'ts_outputs')

    def initialize_step(self, stepname, parameters, save, suffix, outputfolder):
        """
        This function initializes a given step: (1) checks calibration parameters and adds it to the self.calibration_parameters, 
        checks/sets output folder and checks/sets suffixes
        """

        if suffix is None and not hasattr(self, 'suffix'):
            suffix = ''

        # First of all, double check if self.calibration_parameters exists (this function can be run without
        # running detector_calibration). If it hasn't, initialize the calibration_parameters:
        if not hasattr(self, 'calibration_parameters'):

            if not hasattr(self, 'actual_suffix'):
                   
                 self.actual_suffix = '' 

            if not hasattr(self, 'suffix'):

                 self.suffix = suffix

            self.calibration_parameters = {} 
            self.fill_calibration_parameters(parameters = parameters, save = save)

        else:

            # Update any new keys in the current step:
            if stepname in parameters.keys():

                for k in list(parameters[stepname].keys()):

                    self.calibration_parameters[stepname][k] = parameters[stepname][k]

        # Check output folder details:
        if (outputfolder is None) and (self.outputfolder is None):

            self.set_outputfolder('') # If no outputfolder, define the standard one

        elif (outputfolder is not None) and (self.outputfolder is not None):

            self.set_outputfolder(outputfolder) # If new outputfolder for detector data, re-initialize

        elif (outputfolder is not None) and (self.outputfolder is None):

            self.set_outputfolder(outputfolder) # If none defined prior, set it to input

        # Check suffix if new or if not already defined:
        if (suffix is not None) or (suffix is None and not hasattr(self, 'suffix')):

            suffix = '' if suffix is None else suffix
            self.suffix = suffix

            if suffix != '':

                self.actual_suffix = suffix+'_'

            else:

                self.actual_suffix = ''

            # If a new suffix, change it for the saved files:
            if 'suffix' in list(self.calibration_parameters[stepname].keys()):

                self.calibration_parameters[stepname]['suffix'] = self.actual_suffix+stepname+'step'

    def check_nans(self, array):

        self.has_nan = False
        # Get indexes of nan:
        self.nan_locations = np.where( np.isnan(array) )

        # Count them and compare against all pixels:
        number_of_nans = len(self.nan_locations[0])
        number_of_pixels = np.prod(array.shape)
        self.nan_fraction = ( float(number_of_nans) / float(number_of_pixels) ) * 100.

        # Report to user:
        if number_of_nans > 0:

            self.has_nan = True

    def interpolate_nans(self, frame, nan_location):

        # Copy input frame so we save corrected pixels in there:
        corrected_frame = deepcopy(frame)
        corrected_frame[nan_location] = 0.

        # Compute median filter:
        mf = median_filter(corrected_frame, [self.calibration_parameters['tracing']['row_window'],
                                         self.calibration_parameters['tracing']['column_window']
                                        ]
                          )

        # Fill nans with median filters:
        corrected_frame[nan_location] = mf[nan_location]
    
        # Return corrected frame:
        return corrected_frame


    def trace_spectra(self, parameters ={}, save = True, suffix = None, outputfolder = None, **kwargs):
        """
        This function performs spectral tracing to all the rates per integration in self.rateints.
        """

        self.initialize_step('tracing', parameters, save, suffix, outputfolder)

        print('\t >> Performing (Spectral) Tracing step...\n')

        # First, check if there are any nans; if there are, tell to the user what is going to happen:
        self.check_nans(self.rateints)

        if self.has_nan:

           print('\t Warning: rateints array has {0:.2f}% of pixels marked as nan. These will be replaced by local values via spatial interpolation for tracing.'.format( self.nan_fraction ))

        # To prepare the interpolator, first obtain the median integration:
        self.median_rateints = np.nanmedian(self.rateints, axis = 0)

        # Useful for the next steps is to correct any reminder nans in the median frame first; do that:
        self.median_rateints_nan_locations = np.where( np.isnan( self.median_rateints ) )
        self.nan_corrected_median_rateints = self.interpolate_nans( self.median_rateints, 
                                                                    self.median_rateints_nan_locations
                                                                  )

        # Next-up, create nan-corrected rateints using this median rateints. To do this, simply scale the 
        # median frame to local pixels for each nan:
        #for i in range
               
 

    def fit_ramps(self, parameters = {}, save = True, suffix = None, outputfolder = None, **kwargs):
        """ 
        This function gets an estimate of the rates per integration via the self.ramps_per_segment. By default, this uses 
        the STScI JWST Pipeline to do its magic.

        Select ``ramp_fit_method='sparta'`` and ``ramp_fit_options`` in ``load``
        before detector calibration to use correlated adjacent-read fitting.
        That backend always recomputes, writes distinct output names, and keeps
        counts/template/provenance in ``ramp_fit_diagnostics`` per segment.

        If `detector_calibration` enabled the group-difference 1/f correction (the default for
        NIRSpec and NIRISS/SOSS), the ramps are then corrected using these rates (see
        `group_difference_1f`) and fitted again; with ``save = True`` the corrected rates are
        saved as ``<name>_<suffix>groupdiff1f_rateints.fits``.
        """

        self.initialize_step('ramp_fit', parameters, save, suffix, outputfolder)
        self.run_ramp_fit(save)

        if self.status.get('group_difference_1f', 'SKIPPED') is None:

            self.group_difference_1f(save = save)

    def group_difference_1f(self, save = True):
        """
        This function performs the group-level 1/f correction with median group-difference
        templates on self.ramps_per_segment, using the current rates per integration (see
        `group_difference_1f_correction`; options from ``parameters['group_difference_1f']`` of
        `detector_calibration`), and fits the ramps again. Diagnostics are kept in
        ``self.group_difference_1f_diagnostics``. Called by `fit_ramps` when the correction is
        enabled.
        """

        print('\t >> Performing group-difference 1/f correction...\n')
        options = self.calibration_parameters.get('group_difference_1f', {})
        diagnostics = group_difference_1f_correction(self.ramps_per_segment,
                                                     self.rateints_per_segment,
                                                     self.mode_definition, **options)
        self.group_difference_1f_diagnostics = diagnostics

        # Fit the corrected ramps (without overwriting or reusing the saved first-pass rates),
        # and mark the rates so that stage2 does not remove 1/f noise again:
        self.run_ramp_fit(save = False, use_saved = False)
        from .jwst_models import mark_group_1f
        for rateints in self.rateints_per_segment:

            mark_group_1f(rateints)

        if save:

            for i in range( len(self.rateints_per_segment) ):

                name = self.datanames[i]+'_'+self.actual_suffix+'groupdiff1f_rateints.fits'
                self.rateints_per_segment[i].save( name, dir_path = self.outputfolder+'ts_outputs' )

        if 'group_scale' in diagnostics:

            factors = diagnostics['group_scale'].tolist()
            self.calibration_parameters['Group-difference 1/f template factors'] = factors
        self.status['group_difference_1f'] = 'COMPLETE'

    def run_ramp_fit(self, save = True, use_saved = True):
        """
        Fit self.ramps_per_segment with the selected ramp-fitting method (see `fit_ramps`). With
        ``use_saved``, rates saved by an earlier run of the STScI ramp fit are loaded instead;
        without it, the STScI ramp fit neither loads nor saves its products (``save`` only
        applies to the sparta backend).
        """

        self.rateints_per_segment = []

        if getattr(self, 'ramp_fit_method', 'jwst') == 'sparta':
            from .ramp_fitting import fit_model
            self.ramp_fit_diagnostics = []
            for name, ramp in zip(self.datanames, self.ramps_per_segment):
                rate, rateints, diagnostics = fit_model(ramp, **self.ramp_fit_options)
                try:
                    if save:
                        from pathlib import Path
                        import json
                        base = Path(self.outputfolder)/'ts_outputs'/f'{name}_{self.actual_suffix}sparta'
                        rate.save(str(base)+'_rate.fits')
                        rateints.save(str(base)+'_rateints.fits')
                        np.savez_compressed(str(base)+'_ramp_diagnostics.npz',
                            **{k: v for k, v in diagnostics.items() if k != 'provenance'})
                        Path(str(base)+'_ramp_fit.json').write_text(json.dumps(diagnostics['provenance'], indent=2)+'\n')
                finally:
                    rate.close()
                self.rateints_per_segment.append(rateints)
                self.ramp_fit_diagnostics.append(diagnostics)
            self.calibration_parameters['Ramp-fitting method'] = 'sparta'
            self.calibration_parameters['Ramp-fitting provenance'] = [d['provenance'] for d in self.ramp_fit_diagnostics]
            self.datatype = 'ramps and rates per integration'
            self.merge_rateints_segments()
            self.status['ramp_fit'] = 'COMPLETE'
            return

        print('\t >> Performing Rampfit step...\n')

        ramp_fit_parameters = dict(self.calibration_parameters['ramp_fit'])
        if not use_saved:

            for key in ['output_dir', 'save_results', 'suffix']:

                ramp_fit_parameters.pop(key, None)

        # Run if not already in:
        for i in range( len(self.ramps_per_segment) ):

            rateint_filename = self.outputfolder+'ts_outputs/'+self.datanames[i]+'_'+self.actual_suffix+'1_ramp_fitstep.fits'
            if not (use_saved and os.path.exists(rateint_filename)):

                self.rateints_per_segment.append(
                    calwebb_detector1.ramp_fit_step.RampFitStep.call(self.ramps_per_segment[i],
                                                                     **ramp_fit_parameters)[1]
                )

            else:

                print('\t >> Rampfit files found for {0:}. Loading them...\n'.format(self.datanames[i]))
                self.rateints_per_segment.append( datamodels.open(rateint_filename)
                                    )

        # Save current version of the pipeline and CRDS context being used for the dataproducts generated/read above:
        self.calibration_parameters['Ramp-fitting STScI Pipeline Version'] = self.rateints_per_segment[-1].meta.calibration_software_version
        self.calibration_parameters['Ramp-fitting CRDS context'] = self.rateints_per_segment[-1].meta.ref_file.crds.context_used
        self.datatype = 'ramps and rates per integration'

        # Merge with self.rateints:
        self.merge_rateints_segments()

        # Modify status of the step:
        self.status['ramp_fit'] = 'COMPLETE'

    def detector_calibration(self, parameters = {}, use_tso_jump = True, group_1f = False, save = True, suffix = '', outputfolder = None,
                             group_difference_1f = True, group_difference_1f_domain = None,
                             ramp_fit_method = None, ramp_fit_options = None, **kwargs):
        """
        This function performs detector-level calibration --- i.e., most of what is Stage1 in the STScI JWST pipeline,
        but not performing ramp-fitting nor CDS. In other words, this stops at the jump step.

        Parameters
        ----------

        parameters : dict
            (Optional) dictionary hosting different parameters for different steps in the JWST pipeline. Default is empy dict. Each key should be a pipeline step 
            matching the self.status keywords. Each one of those, in turn, should have keys specifying different pipeline parameters, e.g.::

                parameters['superbias']['override_superbias'] = 'my_jwst_superbias_reference.fits'

        use_tso_jump : bool
            (Optional) flag defining if to run the TSO-based jump step implemented in transitspectroscopy. Default is True. If False, the jump 
            step in the JWST pipeline will be used.

        group_1f : bool
            (Optional) flag defining if to perform 1/f noise correction using non-iluminated pixels at the group-level. Only available for NIRSpec for now. Default is False.

        save : bool
            (Optional) flag to save results into fits files. Default is True.

        suffix : str
            (Optional) string defining suffix for output files if ``save = True``. Default is ``''`` (no suffix).

        outputfolder : str
            (Optional) folder where the outputs will be saved to in case this was note defined before. By default, a folder `ts_outputs` is created in that folder. 
            Default is the current working directory.

        group_difference_1f : bool
            (Optional) flag defining if `fit_ramps` performs the group-level 1/f correction with
            median group-difference templates (see `group_difference_1f_correction`): the ramps
            are fitted, corrected using the traces and white-light factors of those rates, and
            fitted again. Applied to the modes that support it (NIRSpec and NIRISS/SOSS); for
            NIRISS/SOSS it replaces `clean_flicker_noise`. Options go in
            ``parameters['group_difference_1f']``. Default is True.

        group_difference_1f_domain : str
            (Optional) where that correction measures the 1/f levels: ``'groups'`` (the groups)
            or ``'differences'`` (the group differences, or CDS frames, where offsets that
            change from integration to integration cancel; their cumulative sums are removed
            from the groups). Default is None: ``parameters['group_difference_1f']['domain']``
            if given, else ``'groups'``.

        ramp_fit_method : str
            (Optional) ramp fit of `fit_ramps`: ``'jwst'`` (the JWST ramp-fitting step) or
            ``'sparta'`` (the correlated fit of the group differences, arXiv:2305.06240; see
            `transitspectroscopy.ramp_fitting`), replacing the one selected in `load`. Default
            is None (keep it).

        ramp_fit_options : dict
            (Optional) options of the ``'sparta'`` ramp fit, with ``ramp_fit_method``. Default
            is None (none).

        """

        if ramp_fit_method is not None:

            from .ramp_fitting import validate_selection
            validate_selection(ramp_fit_method, ramp_fit_options)
            self.ramp_fit_method = ramp_fit_method
            self.ramp_fit_options = deepcopy(ramp_fit_options or {})

        elif ramp_fit_options is not None:

            raise ValueError('ramp_fit_options requires ramp_fit_method')

        self.suffix = suffix

        # Add _ if suffix is given to the actual_suffix:
        if suffix != '':

            self.actual_suffix = suffix+'_'

        else:

            self.actual_suffix = '' 

        # Decide if to update output folder details or not:
        if (outputfolder is None) and (self.outputfolder is None):

            self.set_outputfolder('') # If no outputfolder, define the standard one

        elif (outputfolder is not None) and (self.outputfolder is not None):

            self.set_outputfolder(outputfolder) # If new outputfolder for detector data, re-initialize

        elif (outputfolder is not None) and (self.outputfolder is None):

            self.set_outputfolder(outputfolder) # If none defined prior, set it to input


        # Check and fill detector calibration input parameters per step:
        self.calibration_parameters = {} 
        self.fill_calibration_parameters(parameters, use_tso_jump, group_1f, save,
                                         group_difference_1f)
        options = dict(self.calibration_parameters.get('group_difference_1f', {}))
        if group_difference_1f_domain is not None:

            if options.get('domain', group_difference_1f_domain) != group_difference_1f_domain:

                raise ValueError('group_difference_1f_domain and '
                                 "parameters['group_difference_1f']['domain'] differ")

            options['domain'] = group_difference_1f_domain

        if options.get('domain', 'groups') not in ('groups', 'differences'):

            raise ValueError("group_difference_1f_domain must be 'groups' or 'differences'")

        self.calibration_parameters['group_difference_1f'] = options

        # Print some information to the user:
        print('\t [START] Detector-level Calibration\n\n')
        print('\t >> Processing '+str(len(self.ramps_per_segment))+' files.\n')
        print('\t    - TSO total duration: {0:.1f} hours'.format((np.max(self.times)-np.min(self.times))*24.))
        print('\t    - Calibration parameters:')
        print(self.calibration_parameters)

        # First, check steps the user wants skipped:
        for steptoskip in self.calibration_parameters['skip']:

            self.status[steptoskip] = 'SKIPPED'

        # Set definition for function calls:
        self.step_calls = {} 
        self.step_calls['dq_init'] = calwebb_detector1.dq_init_step.DQInitStep.call
        self.step_calls['emicorr'] = calwebb_detector1.emicorr_step.EmiCorrStep.call
        self.step_calls['saturation'] = calwebb_detector1.saturation_step.SaturationStep.call
        self.step_calls['superbias'] = calwebb_detector1.superbias_step.SuperBiasStep.call
        self.step_calls['dark_sub'] = calwebb_detector1.dark_current_step.DarkCurrentStep.call

        # The reference pixel step is typically skipped for NIRSpec/PRISM, so the mode can select our own pedestal correction:
        mode_steps = self.mode_definition.detector_steps
        if mode_steps['refpix'] == 'side_pedestal':

            self.step_calls['refpix'] = side_refpix_correction

        else:

            self.step_calls['refpix'] = calwebb_detector1.refpix_step.RefPixStep.call

        self.step_calls['group_1f'] = group_1f_correction
        self.step_calls['linearity'] = calwebb_detector1.linearity_step.LinearityStep.call
        self.step_calls['clean_flicker_noise'] = calwebb_detector1.clean_flicker_noise_step.CleanFlickerNoiseStep.call

        # Now go step-by-step checking which steps were done. If linearity was done and saved, read it:
        if not os.path.exists(self.outputfolder+'ts_outputs/'+self.datanames[-1]+'_'+self.actual_suffix+'linearitystep.fits'):

            steps = list(mode_steps['steps'])

            for step in steps:

                # Make a difference with group_1f which handles all ramps together:
                if step in ['group_1f']:

                    if self.status[step] is None:

                        print('\t >> Running Group 1/f correction...')
                        self.ramps_per_segment = self.step_calls[step]( self.ramps_per_segment, **self.calibration_parameters[step] )
                        print('\t >> Done!')

                else:

                    for i in range( len(self.ramps_per_segment) ):
                
                        if self.status[step] is None:

                            self.ramps_per_segment[i] = self.step_calls[step]( self.ramps_per_segment[i], **self.calibration_parameters[step] )

                if self.status[step] is None:

                    self.status[step] = 'COMPLETE'

        else:

            print('\t >> Linearity files found. Loading them...\n')
            for i in range( len(self.ramps_per_segment) ):

                self.ramps_per_segment[i] = datamodels.RampModel(self.outputfolder+'ts_outputs/'+self.datanames[i]+'_'+self.actual_suffix+'linearitystep.fits')

            # Check status of the loaded files:
            self.check_status(self.ramps_per_segment[-1])
            
        # Run rest of the steps prior to the jump but after linearity:
        steps = list(mode_steps['post_linearity_steps'])

        for step in steps:

            for i in range( len(self.ramps_per_segment) ):

                if self.status[step] is None:

                    self.ramps_per_segment[i] = self.step_calls[step]( self.ramps_per_segment[i], **self.calibration_parameters[step] )

            if self.status[step] is None:

                self.status[step] = 'COMPLETE'

        # Now for the jump step; depends on which jump step user wants:
        if self.status['jump'] is None:

            if use_tso_jump:

                if not os.path.exists(self.outputfolder+'ts_outputs/'+self.datanames[-1]+'_'+self.actual_suffix+'tsojumpstep.fits'):

                    print('\t >> Performing TSO-jump...\n')
                    self.ramps_per_segment = tso_jumpstep(self.ramps_per_segment, **self.calibration_parameters['jump'])
                    print('\t >> ...done!\n')

                    # Save results:
                    if save:

                        for i in range( len(self.ramps_per_segment) ):

                            self.ramps_per_segment[i].save( self.datanames[i]+'_'+self.actual_suffix+'tsojumpstep.fits', dir_path = self.outputfolder+'ts_outputs' )

                    self.status['jump'] = 'COMPLETE'

                else:

                    print('\t >> TSO-jump files found. Loading them...\n')

                    
                    for i in range( len(self.ramps_per_segment) ):

                        self.ramps_per_segment[i] = datamodels.RampModel(self.outputfolder+'ts_outputs/'+self.datanames[i]+'_'+self.actual_suffix+'tsojumpstep.fits')

                    # Check status of the loaded files:
                    self.check_status(self.ramps_per_segment[-1])
                    self.status['jump'] = 'COMPLETE'

                    if self.suffix == '':

                        self.suffix = 'tsojumpstep'
                        self.actual_suffix = 'tsojumpstep_'

                    else:

                        self.suffix += '_tsojumpstep'
                        self.actual_suffix = self.suffix+'_'

            else:

                if not os.path.exists(self.outputfolder+'ts_outputs/'+self.datanames[-1]+'_'+self.actual_suffix+'jumpstep.fits'):

                    for i in range( len(self.ramps_per_segment) ):

                        self.ramps_per_segment[i] = calwebb_detector1.jump_step.JumpStep.call(self.ramps_per_segment[i], **self.calibration_parameters['jump'])

                    self.status['jump'] = 'COMPLETE'

                else:

                    print('\t >> Jump files found. Loading them...\n')

                    for i in range( len(self.ramps_per_segment) ):

                        self.ramps_per_segment[i] = datamodels.RampModel(self.outputfolder+'ts_outputs/'+self.datanames[i]+'_'+self.actual_suffix+'jumpstep.fits')

                    # Check status of the loaded files:
                    self.check_status(self.ramps_per_segment[-1])

        # Steps between the jump step and ramp fitting (e.g., clean_flicker_noise for NIRISS/SOSS):
        for step in mode_steps.get('post_jump_steps', ()):

            if self.status[step] is None:

                print('\t >> Running '+step+'...\n')
                for i in range( len(self.ramps_per_segment) ):

                    self.ramps_per_segment[i] = self.step_calls[step]( self.ramps_per_segment[i], **self.calibration_parameters[step] )

                self.status[step] = 'COMPLETE'

        # Save current version of the pipeline and CRDS context being used for the dataproducts generated/read above:
        self.calibration_parameters['STScI Pipeline Version'] = self.ramps_per_segment[-1].meta.calibration_software_version
        self.calibration_parameters['CRDS context'] = self.ramps_per_segment[-1].meta.ref_file.crds.context_used

        # Merge new results with full self.ramps:
        self.merge_ramps_segments()

        print('\t [END] Detector-level Calibration\n\n')

    def merge_ramps_segments(self):
        """Merge ramps and link segment views, accepting models without ERR.

        Modern RampModel has no ERR field. For compatibility, ``ramps_err``
        remains an array with zero placeholders where errors are absent;
        ``ramps_err_available`` identifies integrations with supplied errors.
        Placeholders are not measured uncertainties and must not be used as
        noise estimates. Rate-product uncertainties are handled separately.
        """
        errors = []
        for segment in self.ramps_per_segment:
            error = getattr(segment, 'err', None)
            if error is not None and error.size == 0:
                error = None
            if error is not None and error.shape != segment.data.shape:
                raise ValueError('Ramp ERR shape must match the segment data shape')
            errors.append(error)

        self.ramps = np.zeros([self.nints, self.ngroups, self.nrows, self.ncols], dtype = self.ramps_per_segment[0].data.dtype)
        error_dtypes = [error.dtype for error in errors if error is not None]
        error_dtype = np.result_type(*error_dtypes) if error_dtypes else self.ramps.dtype
        self.ramps_err = np.zeros(self.ramps.shape, dtype=error_dtype)
        self.ramps_err_available = np.zeros(self.nints, dtype=bool)

        self.groupdq = np.zeros([self.nints, self.ngroups, self.nrows, self.ncols] , dtype = self.ramps_per_segment[0].groupdq.dtype)
        self.pixeldq = self.ramps_per_segment[0].pixeldq

        # First, fill the ramp and groupdq arrays:
        current_nintegrations = 0
        for i in range( len(self.ramps_per_segment) ):

            end_nintegrations = current_nintegrations + self.ints_per_segment[i]
            self.ramps[current_nintegrations : end_nintegrations, :, :, :] = self.ramps_per_segment[i].data

            if errors[i] is not None:

                self.ramps_err[current_nintegrations : end_nintegrations, :, :, :] = errors[i]
                self.ramps_err_available[current_nintegrations : end_nintegrations] = True

            self.groupdq[current_nintegrations : end_nintegrations, :, :, :] = self.ramps_per_segment[i].groupdq 

            current_nintegrations = current_nintegrations + self.ints_per_segment[i]

        # Now, to link this new full array and the segmented one, re-copy to re-point arrays --- link all pixeldq:
        current_nintegrations = 0
        for i in range( len(self.ramps_per_segment) ):

            end_nintegrations = current_nintegrations + self.ints_per_segment[i]
            self.ramps_per_segment[i].data = self.ramps[current_nintegrations : end_nintegrations, :, :, :]
            
            if errors[i] is not None:
            
                self.ramps_per_segment[i].err = self.ramps_err[current_nintegrations : end_nintegrations, :, :, :]
            
            self.ramps_per_segment[i].pixeldq = self.pixeldq
            self.ramps_per_segment[i].groupdq = self.groupdq[current_nintegrations : end_nintegrations, :, :, :]

            current_nintegrations = current_nintegrations + self.ints_per_segment[i]

    def merge_rateints_segments(self):

        self.rateints = np.zeros([self.nints, self.nrows, self.ncols], dtype = self.rateints_per_segment[0].data.dtype)
        self.rateints_err = np.zeros([self.nints, self.nrows, self.ncols], dtype = self.rateints_per_segment[0].err.dtype)
        self.rateints_dq = np.zeros([self.nints, self.nrows, self.ncols], dtype = self.rateints_per_segment[0].dq.dtype)

        # First, fill the rateint and groupdq arrays:
        current_nintegrations = 0
        for i in range( len(self.rateints_per_segment) ):

            end_nintegrations = current_nintegrations + self.ints_per_segment[i]
            self.rateints[current_nintegrations : end_nintegrations, :, :] = self.rateints_per_segment[i].data
            self.rateints_err[current_nintegrations : end_nintegrations, :, :] = self.rateints_per_segment[i].err
            self.rateints_dq[current_nintegrations : end_nintegrations, :, :] = self.rateints_per_segment[i].dq

            current_nintegrations = current_nintegrations + self.ints_per_segment[i]

        # Now, to link this new full array and the segmented one, re-copy to re-point arrays --- link all pixeldq:
        current_nintegrations = 0
        for i in range( len(self.rateints_per_segment) ):

            end_nintegrations = current_nintegrations + self.ints_per_segment[i]
            self.rateints_per_segment[i].data = self.rateints[current_nintegrations : end_nintegrations, :, :]
            self.rateints_per_segment[i].err = self.rateints_err[current_nintegrations : end_nintegrations, :, :]
            self.rateints_per_segment[i].dq = self.rateints_dq[current_nintegrations : end_nintegrations, :, :]

            current_nintegrations = current_nintegrations + self.ints_per_segment[i]
            
    def __init__(self, input_filenames, outputfolder = None, *,
                 ramp_fit_method='jwst', ramp_fit_options=None):

        from .ramp_fitting import validate_selection
        validate_selection(ramp_fit_method, ramp_fit_options)
        self.ramp_fit_method = ramp_fit_method
        self.ramp_fit_options = deepcopy(ramp_fit_options or {})

        if outputfolder is not None:

            self.set_outputfolder(outputfolder)

        else:

            self.set_outputfolder('')

        self.filenames = input_filenames

        # Read data in for the dataset; detect what kind of product these are first:
        self.datatype = self.check_dataset()
        if ramp_fit_method == 'sparta' and self.datatype != 'ramps':
            raise ValueError('Selecting an alternative ramp fitter requires ramp inputs, not rateints')

        # Open the data, extract timestamps and useful (to transitspectroscopy) information about the instrument/mode:
        self.datanames = []
        self.ramps_per_segment = []
        self.rateints_per_segment = []
        self.times = np.array([])

        # Before actually reading the data, sort the filenames from the smallest to the largest segment:
        self.sort_filenames()       

        # Read data, including time-stamps: 
        self.ints_per_segment = []
        for i in range( len(self.filenames) ):

            if self.datatype == 'ramps':

                self.ramps_per_segment.append( datamodels.RampModel( self.filenames[i] ) )
                self.times = np.append(self.times, self.ramps_per_segment[-1].int_times['int_mid_BJD_TDB'] + 2400000.5)
                self.ints_per_segment.append( self.ramps_per_segment[-1].data.shape[0] )

            elif self.datatype == 'rates per integration':

                self.rateints_per_segment.append( datamodels.open( self.filenames[i] ) )
                self.times = np.append(self.times, self.rateints_per_segment[-1].int_times['int_mid_BJD_TDB'] + 2400000.5)
                self.ints_per_segment.append( self.rateints_per_segment[-1].data.shape[0] )         

            else:

                raise Exception('Error: JWST data type '+self.datatype+' not recognized.')

            self.datanames.append( get_dataname(self.filenames[i]) )

        # Create a merged array with the data and also save important metadata; first, current status of the reduction (i.e., which steps were run, 
        # which are left to run, set in self.status) and check instrument (saved to self.instrument, self.filter and self.dispersive_element):
        if self.datatype == 'ramps':

            self.check_status(self.ramps_per_segment[-1])
            # NINTS describes the full exposure, even when only a subset of
            # its segments was supplied. Allocate only integrations loaded.
            self.nints = sum(self.ints_per_segment)
            self.ngroups = self.ramps_per_segment[-1].meta.exposure.ngroups
            self.nrows = self.ramps_per_segment[-1].data.shape[2]
            self.ncols = self.ramps_per_segment[-1].data.shape[3]

            # Create self.ramps, self.pixeldq and self.groupdq; and link it with the per segment;
            self.merge_ramps_segments()
        
        else:

            self.check_status(self.rateints_per_segment[-1])
            self.nints = sum(self.ints_per_segment)
            self.ngroups = self.rateints_per_segment[-1].meta.exposure.ngroups
            self.nrows = self.rateints_per_segment[-1].data.shape[1]
            self.ncols = self.rateints_per_segment[-1].data.shape[2]
            
            # Create self.rateints,  self.pixeldq and self.groupdq; and link it with the per segment;
            self.merge_rateints_segments()

def get_dataname(filename):

    return '_'.join( filename.split('/')[-1].split('_')[:-1] )

def side_refpix_correction(ramp, left_pixels = 25, right_pixels = 25):
    """
    Given a ramp, this function will use on every group and integration `left_pixels` to the left of the array and `right_pixels` to the 
    right to calculate a background (detector or physical) and remove it from each group via the median. Useful for NIRSpec/PRISM.
    The algorithm is `transitspectroscopy.core.detector.subtract_side_pedestal`.
    """

    subtract_side_pedestal(ramp.data, left_pixels, right_pixels)

    ramp.meta.cal_step.refpix = 'COMPLETE'
    return ramp

def group_1f_correction(tso_list, npixel = 4, nsigma = 10, column_window = 3, row_window = 20, mask_radius = 10):

    """ 
    This function performs group-level 1/f corrections by (a) estimating centroids of every column via last-minus first frame, 
    frame which is 2D median-filtered to remove outliers, (b) using those to mask `mask_radius` pixels around the centroids, 
    (c) taking the median of the unmasked pixels, column-by-column, and subtracting these values from each group on each integration.

    Parameters
    ----------

    tso_list : list
        List of JWST datamodels for a given exposure (e.g., `tso_list = [datamodels.open(file1), datamodels.open(file2)]`). 
        Typically, these would be the segments of a TSO. The algorithm expects those to be ordered chronologically.

    column_window : int
        Algorithm uses a 2D median filter; this is the column-wise window of this filter.

    row_window : int
        Row-wise window of the 2D median filter.        

    mask_radius : int
        Radius within the estimated centroids which will be masked to perform the 1/f reduction.

    Returns
    -------

    output_tso_list : numpy.array
        Same as the `tso_list`, but with the group-level 1/f correction performed as described above.
    """

    # The algorithm is `transitspectroscopy.core.detector.group_1f_correct`. Outputs are
    # shallow copies of the input models holding corrected copies of their data:
    corrected = group_1f_correct([model.data for model in tso_list], npixel = npixel, nsigma = nsigma,
                                 column_window = column_window, row_window = row_window,
                                 mask_radius = mask_radius)

    output_tso_list = []
    for model, data in zip(tso_list, corrected):

        output_tso_list.append( copy(model) )
        output_tso_list[-1].data = data

    return output_tso_list

def group_difference_1f_correction(ramps, rateints, mode, **options):
    """
    This function performs the group-level 1/f correction with median group-difference
    templates, the default for the near-infrared spectroscopic modes. The rates of a first ramp
    fit give the traces and the white-light factor of every integration; for every group ``k``,
    the median over all integrations of group ``k+1`` minus group ``k`` (the last group uses the
    last difference), scaled to group ``k`` and by the white-light factor, is subtracted, and the
    median of the remaining background pixels of each column (the 1/f level) is removed from that
    group (see `transitspectroscopy.core.detector.group_difference_1f`). Ramps have to be fitted
    again afterwards.

    Parameters
    ----------

    ramps : list
        List of calibrated ramps (RampModels, after the jump step) of the segments of an
        exposure, in chronological order. Corrected in place.

    rateints : list
        List of the rates per integration (CubeModels) of those ramps.

    mode : transitspectroscopy.instruments.Mode
        Mode of the data. For NIRISS/SOSS, a box-only `transitspectroscopy.soss.extract_soss` of
        the rates gives the traces (orders 1, 2 and 3 masked), background, median rate and
        white-light factors (as ``onef_method = 'template'``), and ``options`` are passed to it
        (e.g., ``pastasoss_reference``). For the other spectroscopic modes, the median rate is
        traced, the white-light factors are those of the Stage 2 1/f correction
        (``scale_1f_rows``, ``scale_1f_columns`` and ``scale_1f_window`` of the mode) and the 1/f
        levels are the medians between ``inner_radius`` and ``outer_radius`` pixels from the
        trace, as in `transitspectroscopy.core.detector.correct_1f` (but also in the untraced
        columns, around the trace extended at its edge values); ``options`` may set
        ``aperture_radius`` (pixels used to scale each template to its group), ``inner_radius``
        and ``outer_radius`` (default: the mode's). For all modes, ``domain = 'differences'``
        measures and removes the 1/f levels on the group differences (CDS frames) instead of
        the groups (see `transitspectroscopy.core.detector.difference_1f`); the default is
        ``'groups'``.

    Returns
    -------

    diagnostics : dict
        Template factor of every group (``group_scale``, groups only), removed levels
        (``levels``, per integration, group or difference, and column) and white-light factors
        (``white_light_scale``).
    """

    if mode.stage2 == 'soss':

        from .soss import correct_ramps_1f, extract_soss
        settings = dict(options)
        domain = settings.pop('domain', 'groups')
        settings.setdefault('aperture_radius', mode.extraction_defaults['aperture_radius'])
        result = extract_soss(rateints, **settings)
        return correct_ramps_1f(ramps, rateints, result, domain = domain)

    if mode.stage2 != 'spectral':

        raise ValueError('The group-difference 1/f correction is only defined for '
                         'spectroscopic modes')

    return _spectral_group_difference_1f(ramps, rateints, mode, **options)

def _spectral_group_difference_1f(ramps, rateints, mode, aperture_radius = None,
                                  inner_radius = None, outer_radius = None, domain = 'groups'):

    from .core.detector import difference_1f, group_difference_1f
    from .jwst_models import ramp_1f_validity, ramp_jumps

    if domain not in ('groups', 'differences'):

        raise ValueError("domain must be 'groups' or 'differences'")

    tracing = mode.tracing_for(ramps[0].meta.instrument.detector)
    defaults = mode.extraction_defaults
    aperture_radius = defaults['aperture_radius'] if aperture_radius is None else aperture_radius
    inner_radius = defaults['onef_inner_radius'] if inner_radius is None else inner_radius
    outer_radius = defaults['onef_outer_radius'] if outer_radius is None else outer_radius

    # Rates and median rate, NaNs zeroed as in stage2:
    rates = np.concatenate([np.asarray(model.data, dtype = np.float32) for model in rateints])
    median_rate = np.nanmedian(rates, axis = 0)
    median_rate_nan = deepcopy(median_rate)
    median_rate[np.isnan(median_rate)] = 0.
    rates[np.isnan(rates)] = 0.

    # Trace of the median rate, from the mode's starting columns, with outliers replaced and
    # smoothed as in stage2:
    first_column, last_column = tracing['start_columns']
    lags, ccf = get_ccf(np.arange(median_rate.shape[0]),
                        np.nanmedian( median_rate[:, first_column:last_column], axis = 1) )
    center_pixel = lags[np.argmax(ccf)]
    x, y = trace_spectrum(median_rate, np.zeros(median_rate.shape),
                          xstart = tracing['xstart'], ystart = center_pixel, xend = tracing['xend'],
                          y_tolerance = 5,
                          method = tracing['ccf_method'],
                          ccf_function = tracing['ccf_function'],
                          ccf_parameters = tracing['ccf_parameters']
                         )
    x, y = np.asarray(x, dtype = int), np.array(y, dtype = float)
    idx_outliers, mfilter = outlier_detector(y, nsigma = tracing['outlier_nsigma'],
                                             window = tracing['outlier_window'],
                                             return_filter = True)
    y[idx_outliers] = mfilter[idx_outliers]
    _, y = fit_spline(x, y, nknots = tracing['nknots'])

    # Pixels: the 1/f levels use the rows between inner_radius and outer_radius from the trace,
    # as correct_1f but without its wrap-around near the edges, and also in the untraced columns
    # (where correct_1f uses all rows), with the trace extended at its edge values; the template
    # factors use the aperture:
    nrows, ncols = median_rate.shape
    rows = np.arange(nrows)
    order = np.argsort(x)
    centers = np.interp(np.arange(ncols), x[order], y[order])
    background_pixels = np.ones(median_rate.shape, dtype = bool)
    for column, center in enumerate(centers):

        excluded = np.zeros(nrows, dtype = bool)
        excluded[max(int(center) - inner_radius, 0) : max(int(center) + inner_radius, 0)] = True
        excluded[: max(int(center - outer_radius), 0)] = True
        excluded[max(int(center + outer_radius), 0) :] = True
        background_pixels[:, column] = ~excluded

    scale_pixels = np.zeros(median_rate.shape, dtype = bool)
    scale_pixels[:, x] = np.abs(rows[:, None] - y[None, :]) <= aperture_radius

    # White-light factors as in the stage2 1/f correction (after its background subtraction,
    # if any):
    if defaults['background_subtraction']:

        out_of_trace = np.ones(median_rate.shape)
        for column, center in zip(x, y):

            distance = np.abs(rows - center)
            out_of_trace[distance <= defaults['background_inner_radius'], column] = np.nan
            if defaults['background_outer_radius'] is not None:

                out_of_trace[distance >= defaults['background_outer_radius'], column] = np.nan

        background = np.nanmedian(median_rate_nan * out_of_trace, axis = 0)
        background[np.isnan(background)] = 0.
        rates -= background.astype(np.float32)

    scale_1f_rows, scale_1f_columns = defaults['scale_1f_rows'], defaults['scale_1f_columns']
    timeseries = np.nansum(rates[:, scale_1f_rows[0]:scale_1f_rows[1],
                                 scale_1f_columns[0]:scale_1f_columns[1]],
                           axis = (1, 2), dtype = float)
    scale = median_filter(timeseries / np.nanmedian(timeseries), defaults['scale_1f_window'])
    del rates

    valid, template_valid = ramp_1f_validity(ramps)
    data = [ramp.data for ramp in ramps]
    if domain == 'differences':

        diagnostics = difference_1f(data, valid, background_pixels, scale,
                                    jumps = ramp_jumps(ramps))

    else:

        diagnostics = group_difference_1f(data, valid, background_pixels, scale, scale_pixels,
                                          template_valid = template_valid)
    for ramp, corrected in zip(ramps, data):

        ramp.data = corrected

    diagnostics.update(white_light_scale = scale, trace_columns = x, trace = y)

    return diagnostics

def tso_jumpstep(tso_list, window, nsigma = 10):
    """ 
    This function performs the same functionality as the jump step on a *set* of JWST datamodels (from, e.g., segmented data), repurposed to 
    take advantage of the uniqueeness of TSOs; it however works with any multi-integration exposure. The assumptions are:

    1. You are locked on the same target in all the sets of exposures given to this function.
    2. Adding all exposures together, the number of integrations is > 1.
    3. The number of groups per integration is > 1.

    
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

    # The algorithm is `transitspectroscopy.core.detector.tso_jump_flags`. Outputs are shallow
    # copies of the inputs with deep-copied groupdq's, where jumps are marked with a value of 4:
    flags = tso_jump_flags([model.data for model in tso_list], window, nsigma = nsigma)

    output_tso_list = []
    for model, jumps in zip(tso_list, flags):

        output_tso_list.append( copy(model) )
        output_tso_list[-1].groupdq = deepcopy(output_tso_list[-1].groupdq)
        output_tso_list[-1].groupdq[jumps] = 4

    return output_tso_list

def download_reference_file(filename):
    """
    This function downloads a reference file from CRDS given a reference file filename. File gets downloaded to the current working folder.
    """

    print('\n\t >> Downloading {} reference file from CRDS...'.format(filename))
    download_filename = download_file('https://jwst-crds.stsci.edu/unchecked_get/references/jwst/' + filename, cache=True)

    # Rename file:
    os.rename(download_filename, filename)

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
    # CDS uses only successive signal differences. Ramp ERR was unused here
    # and is no longer present in modern JWST ramp models.
    data = np.zeros([nintegrations, ngroups, rows, columns])
    times = np.zeros(nintegrations)

    past_nints = 0
    for i in range( len(datafiles) ):

        with datamodels.RampModel(datafiles[i]) as dm:
            current_nints = dm.data.shape[0]
            times[past_nints:past_nints+current_nints] = dm.int_times['int_mid_BJD_TDB']
            data[past_nints:past_nints+current_nints, :, :, :] = dm.data

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

            cds_data[integration, group, :, :] = correct_1f(cds_data[integration, group, :, :], 
                                                            new_median_cds, 
                                                            x1, ysmooth, 
                                                            scale_factor = smooth_wl[integration],
                                                            inner_radius = ommited_trace_radius, outer_radius = trace_radius)

    # For now, return CDS, backgroud and 1/f-noise-corrected data and time-stamps:
    return times, cds_data, initial_whitelight, smooth_wl

def stage1(uncal_filenames, maximum_cores = 'all', background_model = None, outputfolder = '', use_tso_jump = True, suffix = '', ommit_pixeldq = False, *, ramp_fit_method='jwst', ramp_fit_options=None, **kwargs):
    """
    This function calibrates a set of ``*uncal.fits`` files through a "special" version of the JWST TSO CalWebb Stage 1. It has been tested thoroughly for NIRISS/SOSS, NIRSpec/G395H and NIRSpec/Prism.

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
    suffix  : string
        (Optional) Suffix to add to each out the outputs.
    ommit_pixeldq : bool
        (Optional) If True, ommit the pixel data quality flags and fit ramps to every pixel in the array

    Returns
    -------

    output_dictionary : dict
        Dictionary containing datamodels for the jump step, rampfit step and associated meta-data, including the times of each integration in BJD.
    
    """

    from .ramp_fitting import validate_selection, fit_model
    validate_selection(ramp_fit_method, ramp_fit_options)

    # Add _ if suffix is given to the actual_suffix:
    if suffix != '':

        actual_suffix = suffix+'_'

    else:

        actual_suffix = ''

    # Define output folder if empty:
    if outputfolder != '':
        if outputfolder[-1] != '/':
            outputfolder += '/'

    # Create output dictionary:
    output_dictionary = {}
    output_dictionary['metadata'] = {}
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

    output_dictionary['metadata']['instrument_name'] = instrument_name
    output_dictionary['metadata']['instrument_filter'] = instrument_filter
    output_dictionary['metadata']['instrument_grating'] = instrument_grating
    output_dictionary['metadata']['instrument_detector'] = uncal_data[-1].meta.instrument.detector
    output_dictionary['metadata']['instrument_subarray'] = uncal_data[-1].meta.subarray.name
    output_dictionary['metadata']['date_beg'] = uncal_data[-1].meta.observation.date_beg
    output_dictionary['metadata']['date_end'] = uncal_data[-1].meta.observation.date_end

    # Print some information to the user:
    print('\t >> Processing '+str(len(uncal_data))+' *uncal files.\n')
    print('\t    - TSO total duration: {0:.1f} hours'.format((np.max(times)-np.min(times))*24.))

    # Identify the mode (see transitspectroscopy.instruments). This procedural Stage 1 supports NIRSpec:
    mode_definition = instruments.identify(instrument = instrument_name, dispersive_element = instrument_grating,
                                           filter = instrument_filter)
    if mode_definition.instrument != 'nirspec':

        raise Exception('\t Error: Instrument/Grating/Filter: '+instrument_name+'/'+str(instrument_grating)+'/'+str(instrument_filter)+' not yet supported by stage1; use ts.jwst.load instead.')

    mode = mode_definition.name
    print('\t    - Instrument/Mode: {0:}\n'.format(mode_definition.describe()))

    if not os.path.exists(outputfolder+'pipeline_outputs/'+datanames[-1]+'_'+actual_suffix+'linearitystep.fits'):

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
        if mode_definition.detector_steps['refpix'] == 'side_pedestal':

            for i in range( len(superbias_data) ): 

                refpix_data.append( copy(superbias_data[i]) )
                refpix_data[-1].data = deepcopy(superbias_data[i].data)

                # Remove pedestal changes using the leftmost and rightmost columns of each group.
                # TODO: Get fancy and remove odd/even, too.
                subtract_side_pedestal(refpix_data[-1].data,
                                       **mode_definition.detector_steps['refpix_parameters'])

        else:

            for i in range( len(superbias_data) ):

                refpix_data.append( calwebb_detector1.refpix_step.RefPixStep.call(superbias_data[i])
                                  )

                
        # Linearity step:
        linearity_data = []

        if ommit_pixeldq:

            for i in range( len(superbias_data) ):

                refpix_data[i].pixeldq = np.zeros( refpix_data[i].pixeldq.shape )

        for i in range( len(refpix_data) ):

            if 'override_linearity' in kwargs.keys():

                linearity_data.append( calwebb_detector1.linearity_step.LinearityStep.call(refpix_data[i],
                                                                                           override_linearity = kwargs['override_linearity'], 
                                                                                           output_dir=outputfolder+'pipeline_outputs', \
                                                                                           save_results = True, \
                                                                                           suffix = actual_suffix+'linearitystep', 
                                                                                          )
                                     )

            else:

                linearity_data.append( calwebb_detector1.linearity_step.LinearityStep.call(refpix_data[i],
                                                                                           output_dir=outputfolder+'pipeline_outputs', \
                                                                                           save_results = True, \
                                                                                           suffix = actual_suffix+'linearitystep', 
                                                                                          )
                                     )


    else:

        print('\t >> Linearity files found. Loading them...\n')

        linearity_data = []
        for i in range( len(uncal_filenames) ):

            linearity_data.append( datamodels.RampModel(outputfolder+'pipeline_outputs/'+datanames[i]+'_'+actual_suffix+'linearitystep.fits') )

    # Now, instead of passing this through the normal jump step, we pass it through our own jump step detection:
    jump_window = kwargs.get('jump_window', mode_definition.detector_steps['tso_jump']['window'])
    jump_nsigma = kwargs.get('jump_nsigma', mode_definition.detector_steps['tso_jump']['nsigma'])

    if use_tso_jump:

        if not os.path.exists(outputfolder+'pipeline_outputs/'+datanames[-1]+'_'+actual_suffix+'tsojumpstep.fits'):

            jump_data = tso_jumpstep(linearity_data, window = jump_window, nsigma = jump_nsigma)
            
            # Save results:
            for i in range( len(jump_data) ):

                jump_data[i].save(datanames[i]+'_'+actual_suffix+'tsojumpstep.fits', dir_path = outputfolder+'pipeline_outputs')

        else:

            print('\t >> TSO-jump files found. Loading them...\n')

            jump_data = []

            for i in range( len(linearity_data) ):

                jump_data.append( datamodels.RampModel(outputfolder+'pipeline_outputs/'+datanames[i]+'_'+actual_suffix+'tsojumpstep.fits') )

        prefix = 'tsojumpstep_'

    else:

        if 'jump_threshold' not in kwargs.keys():

            jump_threshold = 15

        else:

            jump_threshold = kwargs['jump_threshold']


        if not os.path.exists(outputfolder+'pipeline_outputs/'+datanames[-1]+'_'+actual_suffix+'jumpstep.fits'):

            jump_data = []
            for i in range( len(linearity_data) ):

                jump_data.append( calwebb_detector1.jump_step.JumpStep.call(linearity_data[i], 
                                                                            output_dir=outputfolder+'pipeline_outputs', 
                                                                            save_results = True,
                                                                            rejection_threshold = jump_threshold,
                                                                            maximum_cores = maximum_cores,
                                                                            suffix = actual_suffix+'jumpstep')
                                )

        else:

            print('\t >> Jump files found. Loading them...\n')

            jump_data = [] 

            for i in range( len(linearity_data) ):

                jump_data.append( datamodels.RampModel(outputfolder+'pipeline_outputs/'+datanames[i]+'_'+actual_suffix+'jumpstep.fits') ) 

        prefix = ''

    # Finally, do (or load products of the) ramp-fitting step --- return those, the jump-step products, the times and other metadata 
    # of interest:
    ramp_data = []
    ints_per_segment = []
    for i in range( len(jump_data) ):

        if ramp_fit_method == 'sparta':
            rate, rateints, diagnostics = fit_model(jump_data[i], **(ramp_fit_options or {}))
            base = outputfolder+'pipeline_outputs/'+datanames[i]+'_'+actual_suffix+'sparta'
            try:
                rate.save(base+'_rate.fits')
                rateints.save(base+'_rateints.fits')
            finally:
                rate.close()
            ramp_data.append(rateints)
            output_dictionary.setdefault('ramp_fit_diagnostics', []).append(diagnostics)
            output_dictionary['metadata']['ramp_fit_method'] = 'sparta'

        elif not os.path.exists(outputfolder+'pipeline_outputs/'+datanames[i]+'_'+actual_suffix+prefix+'1_'+actual_suffix+'rampfitstep.fits'):

            ramp_data.append( calwebb_detector1.ramp_fit_step.RampFitStep.call(jump_data[i], 
                                                                               output_dir=outputfolder+'pipeline_outputs',
                                                                               save_results = True,
                                                                               maximum_cores = maximum_cores,
                                                                               suffix = actual_suffix+'rampfitstep'
                                                                              )[1]
                            )

        else:

            print('\t >> Rampfit files found for {0:}. Loading them...\n'.format(datanames[i]))
            ramp_data.append( datamodels.open(outputfolder+'pipeline_outputs/'+datanames[i]+'_'+actual_suffix+prefix+'1_'+actual_suffix+'rampfitstep.fits') )

        ints_per_segment.append(ramp_data[-1].data.shape[0])

    output_dictionary['times'] = times
    output_dictionary['ints_per_segment'] = np.array(ints_per_segment)
    output_dictionary['nints'] = ramp_data[-1].meta.exposure.nints
    output_dictionary['ngroups'] = ramp_data[-1].meta.exposure.ngroups  
    output_dictionary['rampstep'] = ramp_data
    output_dictionary['jumpstep'] = jump_data

    output_dictionary['metadata']['calwebb_version'] = ramp_data[-1].meta.calibration_software_version
    output_dictionary['metadata']['param_context'] = ramp_data[-1].meta.ref_file.crds.context_used

    return output_dictionary

def _optimal_frame(data, errors, invalid_mask=None):
    """Prepare one integration without duplicating an entire visit's arrays."""
    frame = data.copy()
    variance = errors**2
    invalid = ~np.isfinite(frame) | ~np.isfinite(variance) | (variance <= 0)
    if invalid_mask is not None:
        invalid |= invalid_mask
    frame[invalid] = -9999
    return frame, variance


class _SharedVarianceSequence:
    """Square one integration's error image at a time during profile training."""
    def __init__(self, errors):
        self.errors = errors

    def __len__(self):
        return len(self.errors)

    def __getitem__(self, index):
        return self.errors[index]**2


class _Stage2SharedMask:
    """Recover original invalid/DQ pixels after legacy Stage 2 image filling."""
    def __init__(self, inputs, mask_dq, columns):
        self.products = inputs['rampstep']
        self.stops = np.cumsum(inputs['ints_per_segment'])
        self.mask_dq, self.columns = mask_dq, columns

    def __len__(self):
        return int(self.stops[-1])

    def __getitem__(self, index):
        if index < 0 or index >= len(self):
            raise IndexError(index)
        segment = int(np.searchsorted(self.stops,index,side='right'))
        local = index-(0 if segment == 0 else self.stops[segment-1])
        product = self.products[segment]
        image, error = product.data[local,:,self.columns], product.err[local,:,self.columns]
        invalid = ~np.isfinite(image) | (image == -9999) | ~np.isfinite(error) | (error <= 0)
        if self.mask_dq:
            dq = product.dq[local] if product.dq.ndim == 3 else product.dq
            invalid |= dq[:,self.columns] != 0
        return invalid


class _CombinedSharedMask:
    def __init__(self, primary, secondary, count):
        self.primary, self.secondary, self.count = primary, secondary, count

    def __len__(self):
        return self.count

    def __getitem__(self,index):
        if index < 0 or index >= self.count:
            raise IndexError(index)
        if self.primary is None:
            return self.secondary[index]
        if self.secondary is None:
            return self.primary[index]
        return np.asarray(self.primary[index],dtype=bool) | np.asarray(self.secondary[index],dtype=bool)


def _shared_cache_json(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        import hashlib
        digest = hashlib.sha256()
        for part in np.atleast_1d(value):
            digest.update(np.ascontiguousarray(part).tobytes())
        return dict(shape=list(value.shape), dtype=str(value.dtype), sha256=digest.hexdigest())
    raise TypeError('shared profile cache settings must be JSON values or NumPy arrays')


def _extract_shared_sequence(data, errors, traces, reference_trace, specification,
                             options, radius, spacing, method, order, gp_options,
                             nsigma, nthreads, coordinates, validity_mask=None):
    from .shared_profile import SharedProfile, fit_shared_profile
    options = dict(options or {})
    store_profiles = options.pop('store_profiles', False)
    supplied_mask, supplied_background = options.pop('mask',None), options.pop('background',None)
    combined_mask = (None if validity_mask is None and supplied_mask is None else
                     _CombinedSharedMask(validity_mask,supplied_mask,len(data)))
    if isinstance(specification, str) and specification == 'train':
        reserved = {'aperture_radius', 'spacing', 'profile_method', 'polynomial_order',
                    'gp_options', 'dispersion_coordinates'}
        if set(options) & reserved:
            raise ValueError('shared_profile_options duplicates Stage 2 geometry/profile options')
        options.setdefault('execution', 'serial' if nthreads is None else 'ray')
        options.setdefault('n_workers', 1 if nthreads is None else nthreads)
        model = fit_shared_profile(data, _SharedVarianceSequence(errors), traces,
                                   aperture_radius=radius, spacing=spacing,
                                   profile_method=method, polynomial_order=order,
                                   gp_options=gp_options, dispersion_coordinates=coordinates,
                                   mask=combined_mask, background=supplied_background,
                                   **options)
    else:
        model = specification
        if options:
            raise ValueError('a supplied SharedProfile cannot have training options')
        if not isinstance(model, SharedProfile):
            raise ValueError('shared_profile must be "train" or a SharedProfile instance')
        if model.aperture_radius != radius or model.spacing != spacing:
            raise ValueError('supplied shared profile has different aperture/spacing')
    spectra, errors_out = [], []
    profiles = [] if store_profiles else None
    # The expensive training is distributed when requested. Evaluation/extraction
    # streams integrations and never builds a full P_t cube unless requested.
    for index,(image, error, trace) in enumerate(zip(data, errors, traces)):
        result = model.extract(image, trace, error**2, nsigma=nsigma,
                               mask=None if combined_mask is None else combined_mask[index],
                               background=None if supplied_background is None else supplied_background[index],
                               dispersion_coordinates=coordinates)
        spectra.append(result[1])
        errors_out.append(np.sqrt(np.divide(1., result[2], out=np.full(len(trace), np.inf),
                                            where=result[2] > 0)))
        if store_profiles:
            profiles.append(model.evaluate(trace))
    return (np.asarray(spectra), np.asarray(errors_out), model.evaluate(reference_trace),
            None if profiles is None else np.asarray(profiles), model)


def _rate_level_1f(config, input_dictionary):
    """Whether stage2 removes 1/f noise from the rates: ``scale_1f``, unless all the rates were
    corrected at the group level already (``load.group_difference_1f``)."""
    from .jwst_models import group_1f_corrected
    if not config.scale_1f:

        return False

    if all(group_1f_corrected(model) for model in input_dictionary['rampstep']):

        print('\t    - Rates corrected for 1/f noise at the group level: '
              'no rate-level 1/f correction.')
        return False

    return True

def _stage2_imaging(input_dictionary, config, mode, outputfolder, suffix):
    """Stage 2 of imaging modes (MIRI): calibrated cutouts, aperture and GP PSF photometry."""
    from .miri import extract_photometry
    options = dict(config.miri or {})
    if config.gp_options is not None:
        options.setdefault('gp_options', config.gp_options)
    if config.aperture_radius is not None:
        options.setdefault('aperture_radius', config.aperture_radius)
        prep = dict(options.get('preparation_options', {}))
        prep.setdefault('aperture_radius', config.aperture_radius)
        options['preparation_options'] = prep
    return extract_photometry(input_dictionary,
        outputfolder=os.path.join(outputfolder, 'pipeline_outputs', 'miri'+('_'+suffix if suffix else '')),
        **options)


def _stage2_soss(input_dictionary, config, mode, outputfolder, suffix):
    """Stage 2 of NIRISS/SOSS: JWST background step, then tracing and extraction of three orders."""
    from .soss import extract_soss
    options = dict(config.soss or {})
    if config.optimal_extraction:
        options.setdefault('optimal_methods', (config.profile_method,))
        settings = dict(options.get('optimal_options', {}))
        if config.gp_options is not None:
            settings['gp_options'] = config.gp_options
        for name, value in config.extraction_options().items():
            settings[{'polynomial_spacing': 'spacing'}.get(name, name)] = value
        options['optimal_options'] = settings
    aperture_radius = config.aperture_radius
    if aperture_radius is None:
        aperture_radius = mode.extraction_defaults['aperture_radius']
    return extract_soss(input_dictionary, aperture_radius=aperture_radius,
                        mask_dq=config.mask_dq, scale_1f=_rate_level_1f(config, input_dictionary),
                        single_trace_extraction=config.single_trace_extraction,
                        outputfolder=os.path.join(outputfolder, 'pipeline_outputs', 'soss'+('_'+suffix if suffix else '')),
                        **options)


def stage2(input_dictionary, nthreads = None, zero_nans = True, scale_1f = True, single_trace_extraction = True, optimal_extraction = False, outputfolder = '', suffix = '', aperture_radius = None, mask_dq = False, extraction_backend='auto', profile_method='polynomial', gp_options=None, extraction_options=None, shared_profile=None, shared_profile_options=None, soss_options=None, miri_options=None, *, config=None, mode=None, write_products=True, **kwargs):
    """
    This function takes an `input_dictionary` having as keys the `rampstep` products on a (chronologically-ordered) list, `times` having the times at 
    each integration in BJD and the integrations per segment `ints_per_segment`. Using those, it performs wavelength calibration, spectral tracing and 
    extraction --- returning the products in another dictionary.

    Parameters
    ----------

    input_dictionary : list
        A dictionary with at least three keys: `rampstep` products on a (chronologically-ordered) list, `times` having the times at each integration in BJD 
        and the integrations per segment `ints_per_segment`.
    nthreads : int
        (Optional) Number of threads to use to parallellize the scripts.
    zero_nans : bool
        (Optional) If True, all nans are converted to zeroes in the ramps. If False, the median rate is used to fill those nans. The latter could dilute the signal, 
        but the resulting stellar spectrum will look better if simple extraction is used.
    scale_1f : bool
        (Optional) If True, the "scale 1/f" noise technique will be used to remove 1/f noise at the ramp level. This removes the scaled median frame from each ramp, 
        and estimates (and removes) the 1/f noise from the resultant frame on the original frame. Rates already corrected at the group level
        (`load.group_difference_1f`, the default of `fit_ramps` for NIRSpec and NIRISS/SOSS) are not corrected again.
    single_trace_extraction : bool
        (Optional) If True, a single trace is used to extract the spectra --- the median of all, integration-level traces. If False, the trace for each 
        individual integration is used to extract the spectra for that integration.
    optimal_extraction : bool
        (Optional) If True, perform spectral extraction via Optimal Extraction (Marsh, 1989PASP..101.1032M). If False, simple spectral extraction will 
        be performed.
    aperture_radius : int
        (Optional) Aperture radius to be used for spectral extraction.
    outputfolder : string
        (Optional) String indicating the folder where the outputs want to be saved. Default is current working directory.
    suffix  : string
        (Optional) Suffix to add to each out the outputs.
    mask_dq : string
        (Optional) Mask pixels that don't have dq of 0.
    extraction_backend : {'auto', 'c', 'python'}
        Optimal-extraction backend. Auto preserves C when installed, otherwise Python.
    profile_method : {'polynomial', 'gp'}
        GP is experimental, opt-in joint spatial-profile fitting; errors remain
        conditional on the fitted profile. Requires optimal_extraction=True.
    gp_options : dict or None
        Fixed GP kernel, length_scale (dispersion pixels), amplitude, n_inducing.
    extraction_options : dict or None
        Override polynomial_spacing, polynomial_order (number of terms), nsigma.
        Explicit extraction settings get distinct spectrum cache filenames.
    shared_profile : None, 'train', or SharedProfile
        Opt-in native-pixel extraction with one intrinsic profile evaluated at
        each supplied trace. Use single_trace_extraction=False for trace motion.
        'train' fits on selected integrations; an existing model is never refit.
    shared_profile_options : dict or None
        Additional fit_shared_profile options, including training_indices and
        optimize_hyperparameters. store_profiles=True additionally stores all
        evaluated P_t arrays; otherwise Ps is None and P is a reference rendering.
        The fitted model is stored in spectra['shared_profile'] and as an NPZ.
    miri_options : dict or None
        MIRI MIR_IMAGE dispatches to miri.extract_photometry, returning sum and 2-D GP
        photometry. Use this dictionary for preparation_options, image_options,
        training_integrations and spatial gp_options. Spectral extraction
        options do not define the MIRI PSF model.
    config : transitspectroscopy.config.ExtractionConfig or None
        All extraction settings in one object (recommended). The keywords
        extraction_backend, profile_method, gp_options, extraction_options,
        shared_profile, shared_profile_options, soss_options and miri_options are
        deprecated in favour of it. If given, the other extraction keywords must keep
        their defaults. How Stage 2 runs is decided by the observing mode of the data
        (see transitspectroscopy.instruments).
    mode : str, Mode or None
        Observing mode to use instead of identifying it from the data's metadata.
    write_products : bool
        Write the standard products (``output['products']``: spectroscopic time series and
        white-light curves, see transitspectroscopy.core.products) with a provenance
        manifest to ``pipeline_outputs/products<suffix>``. NIRISS/SOSS and MIRI write theirs
        with their own products.

    Returns
    -------

    output_dictionary : dict
        Dictionary containing the traces, FWHM, extracted spectra as well as lightcurves at the resolution-level of the instrument     
 
    """

    # All extraction settings live in one ExtractionConfig. The keywords added during the 0.5
    # development still work (with a DeprecationWarning) and are mapped onto it:
    legacy = dict(zero_nans = zero_nans, scale_1f = scale_1f, single_trace_extraction = single_trace_extraction,
                  optimal_extraction = optimal_extraction, aperture_radius = aperture_radius, mask_dq = mask_dq,
                  extraction_backend = extraction_backend, profile_method = profile_method,
                  gp_options = gp_options, extraction_options = extraction_options,
                  shared_profile = shared_profile, shared_profile_options = shared_profile_options,
                  soss_options = soss_options, miri_options = miri_options)
    if config is None:

        config = ExtractionConfig.from_stage2_arguments(**legacy)
        config.warn_deprecated()

    else:

        changed = [name for name, value in legacy.items()
                   if value is not {**STAGE2_ARGUMENTS, **DEPRECATED_STAGE2_KEYWORDS}[name]
                   and value != {**STAGE2_ARGUMENTS, **DEPRECATED_STAGE2_KEYWORDS}[name]]
        if changed:

            raise ValueError('Pass extraction settings either in config or as keywords, not both: '+', '.join(changed))

    # The observing mode decides how Stage 2 runs (see transitspectroscopy.instruments). NIRISS/SOSS has
    # several overlapping orders and its own post-ramp background step, MIRI imaging is photometry,
    # and plug-in modes can provide their own Stage 2:
    from .jwst_models import open_model
    if mode is None:

        with open_model(input_dictionary['rampstep'][0]) as first_model:
            stage2_mode = instruments.identify_model(first_model)

    else:

        stage2_mode = instruments.get_mode(mode) if isinstance(mode, str) else mode

    config = config.validate(stage2_mode)
    handler = stage2_mode.stage2
    if callable(handler):

        return handler(input_dictionary, config, outputfolder, suffix)

    elif handler == 'imaging':

        return _stage2_imaging(input_dictionary, config, stage2_mode, outputfolder, suffix)

    elif handler == 'soss':

        return _stage2_soss(input_dictionary, config, stage2_mode, outputfolder, suffix)

    elif handler != 'spectral':

        raise ValueError('Unknown Stage 2 handler {0:} of mode {1:}'.format(handler, stage2_mode.name))

    # Spectral (single-trace) Stage 2. Unpack the validated configuration:
    zero_nans, scale_1f = config.zero_nans, _rate_level_1f(config, input_dictionary)
    single_trace_extraction, optimal_extraction = config.single_trace_extraction, config.optimal_extraction
    aperture_radius, mask_dq = config.aperture_radius, config.mask_dq
    extraction_backend, profile_method, gp_options = config.backend, config.profile_method, config.gp_options
    extraction_options = config.extraction_options()
    shared_profile, shared_profile_options = config.shared_profile, config.shared_profile_options
    custom_extraction = (extraction_backend != 'auto' or profile_method != 'polynomial'
                         or bool(extraction_options) or shared_profile is not None)
    extraction_settings = dict(backend=extraction_backend, profile_method=profile_method,
                               gp_options=gp_options, options=extraction_options,
                               aperture_radius=aperture_radius)
    if profile_method == 'gp' and shared_profile is None:
        extraction_settings['gp_mask_policy'] = 'complete-column-sequential-v2'
    if shared_profile is not None:
        from .core.shared_profile import MODEL_VERSION as shared_profile_version
        extraction_settings['shared_profile_version'] = shared_profile_version
        extraction_settings['shared_profile'] = (shared_profile if isinstance(shared_profile, str)
                                                  else shared_profile.fingerprint())
        extraction_settings['shared_profile_options'] = shared_profile_options
        extraction_settings['single_trace_extraction'] = single_trace_extraction
        extraction_settings['preprocessing'] = dict(zero_nans=zero_nans, scale_1f=scale_1f, mask_dq=mask_dq)

    # Add _ if suffix is given to the actual_suffix:
    if suffix != '':

        actual_suffix = '_'+suffix

    else:

        actual_suffix = ''

    # Define output folder if empty:
    if outputfolder != '':
        if outputfolder[-1] != '/':
            outputfolder += '/'

    # Create output dictionary:
    output_dictionary = {}
    output_dictionary['metadata'] = {}
    output_dictionary['metadata']['rate_level_1f'] = scale_1f

    #####################################################
    #                       STAGE 2                     #
    #####################################################

    # Create folder that will store pipeline outputs:
    if not os.path.exists(outputfolder+'pipeline_outputs'):
        os.mkdir(outputfolder+'pipeline_outputs')

    # Set the mode:
    print('\t >> Processing through spectral tracing and extraction:\n')
    print('\t    - TSO total duration: {0:.1f} hours'.format((np.max(input_dictionary['times'])-np.min(input_dictionary['times']))*24.))

    # Extract some useful meta-data:
    instrument_name = input_dictionary['rampstep'][0].meta.instrument.name
    instrument_filter = input_dictionary['rampstep'][0].meta.instrument.filter
    instrument_grating = input_dictionary['rampstep'][0].meta.instrument.grating
    instrument_detector = input_dictionary['rampstep'][0].meta.instrument.detector
    instrument_subarray = input_dictionary['rampstep'][0].meta.subarray.name

    output_dictionary['metadata']['instrument_name'] = instrument_name
    output_dictionary['metadata']['instrument_filter'] = instrument_filter
    output_dictionary['metadata']['instrument_grating'] = instrument_grating
    output_dictionary['metadata']['instrument_detector'] = instrument_detector
    output_dictionary['metadata']['instrument_subarray'] = instrument_subarray

    # Mode-dependent parameters come from the mode registry (see transitspectroscopy.instruments):
    mode_definition = stage2_mode
    print('\t    - Instrument/Mode: {0:}\n'.format(mode_definition.describe(instrument_filter)))
    tracing = mode_definition.tracing_for(instrument_detector)
    defaults = mode_definition.extraction_defaults

    # Parameters if zero_nans is False:
    row_window = tracing['nan_fill_window']['row_window']
    column_window = tracing['nan_fill_window']['column_window']

    # Set suffix for each detector (e.g., nrs1 and nrs2) of multi-detector modes:
    if len(mode_definition.detectors) > 1:

        if suffix != '':

            suffix = suffix + '_' + instrument_detector.lower()

        else:

            suffix = instrument_detector.lower()

        actual_suffix = '_'+suffix

    print('\t    - Detector/Subarray: {0:}/{1:}\n'.format(instrument_detector, instrument_subarray))
    # First things first, save the results from the rates in a single array for the rates and the errors:
    nints = np.sum( input_dictionary['ints_per_segment'] )
    tso = np.zeros([nints, input_dictionary['rampstep'][0].data.shape[1], input_dictionary['rampstep'][0].data.shape[2]])
    tso_dq = np.zeros([nints, input_dictionary['rampstep'][0].dq.shape[1], input_dictionary['rampstep'][0].dq.shape[2]])
    tso_err = np.zeros([nints, input_dictionary['rampstep'][0].data.shape[1], input_dictionary['rampstep'][0].data.shape[2]])

    current_integration = 0
    for i in range( len(input_dictionary['rampstep']) ):

        tso[current_integration:current_integration + input_dictionary['ints_per_segment'][i], :, :] = input_dictionary['rampstep'][i].data
        tso_dq[current_integration:current_integration + input_dictionary['ints_per_segment'][i], :, :] = input_dictionary['rampstep'][i].dq
        tso_err[current_integration:current_integration + input_dictionary['ints_per_segment'][i], :, :] = input_dictionary['rampstep'][i].err
        current_integration += input_dictionary['ints_per_segment'][i]

    # Great. Now that we have all the rates, perform spectral tracing. We do the same for all instruments: we choose a set of pixels in one edge of 
    # the detector, take the median of the spectrum in that edge, and then trace the spectra left or the right of that calculated position. For this 
    # initial position check, performing the analysis on the median rates is very useful. First, get a NaN-free median frame:
    median_rate_nan = np.nanmedian(tso, axis = 0)
    idx = np.isnan(median_rate_nan)
    median_rate = deepcopy(median_rate_nan)
    median_rate[idx] = 0.

    # Fill NaNs with zeroes or the median rate, depending on user-input. Default is zeroes, median_rate could dilute transits:
    if not zero_nans:

        mf_median_rate = median_filter(median_rate, [row_window, column_window])
        median_rate[idx] = mf_median_rate[idx]

        # Same for errors:
        median_rate_err_nan = np.nanmedian(tso_err, axis = 0)
        median_rate_err = deepcopy(median_rate_err_nan)
        idx_err = np.isnan(median_rate_err)
        median_rate_err[idx_err] = 0.
        mf_median_err_rate = median_filter(median_rate_err, [row_window, column_window])
        median_rate_err[idx_err] = mf_median_err_rate[idx_err]

    # Same for the entire TSO:
    for i in range(tso.shape[0]):

        if mask_dq:

            idx = np.where( np.isnan( tso[i,:,:] ) | ( tso_dq[i, :, :] != 0. ) )

        else:

            idx = np.where( np.isnan( tso[i,:,:] ) )

        if len(idx[0])!=0:

            if zero_nans:

                tso[i, :, :][idx] = 0.

                # If optimal extraction, set errors to a large number because weights will turn to zero
                # automatically this way. If simple extraction, leave error to zero so it doesn't add to the 
                # total variance:
                if optimal_extraction:

                    tso_err[i, :, :][idx] = 1e9

                else:

                    tso_err[i, :, :][idx] = 0.

            else:

                tso[i, :, :][idx] = mf_median_rate[idx]
                tso_err[i, :, :][idx] = median_rate_err[idx]

    output_dictionary['tso'] = tso
    output_dictionary['tso_err'] = tso_err

    if os.path.exists( outputfolder+'pipeline_outputs/traces'+actual_suffix+'.pkl' ):

        print('\t >> Traces found in file {0:}. Loading...'.format(outputfolder+'pipeline_outputs/traces'+actual_suffix+'.pkl'))
        output_dictionary['traces'] = pickle.load( open(outputfolder+'pipeline_outputs/traces'+actual_suffix+'.pkl', 'rb') )

    else:

        xstart = tracing['xstart']
        xend = tracing['xend']
        trace_outlier_nsigma = tracing['outlier_nsigma']
        trace_outlier_window = tracing['outlier_window']
        nknots = tracing['nknots']

        trace_ccf_method = tracing['ccf_method']
        trace_ccf_function = tracing['ccf_function']
        trace_ccf_parameters = tracing['ccf_parameters']

        # Take the starting point from the average spectral shape in the mode's starting columns (e.g., the
        # edges of NRS1 or NRS2):
        first_column, last_column = tracing['start_columns']
        lags, ccf = get_ccf(np.arange(median_rate.shape[0]), np.nanmedian( median_rate[:, first_column:last_column], axis = 1) )

        # Find maximum of the initial CCF:
        idx = np.where(ccf == np.max(ccf))[0]
        center_pixel = lags[idx][0]

        # Prepare output dictionaries. First, trace the median spectrum as a test:
        tic = time.time()

        x1, y1 = trace_spectrum(median_rate, np.zeros(median_rate.shape), 
                                xstart = xstart, ystart = center_pixel, xend = xend, 
                                y_tolerance = 5, 
                                method = trace_ccf_method, 
                                ccf_function = trace_ccf_function,
                                ccf_parameters = trace_ccf_parameters
                               )

        toc = time.time()

        total_time = (toc-tic)/3600.

        # Now, create output dicts:
        output_dictionary['traces'] = {}
        output_dictionary['traces']['times'] = input_dictionary['times'] 
        output_dictionary['traces']['x'] = x1 
        output_dictionary['traces']['y'] = np.zeros([ tso.shape[0], len(y1) ])
        output_dictionary['traces']['ycorrected'] = np.zeros([ tso.shape[0], len(y1) ])
        output_dictionary['traces']['ysmoothed'] = np.zeros([ tso.shape[0], len(y1) ])

        # Then use this to trace the entire spectrum. Be smart about tracing and do it via ray, i.e., using multi-processing if `nthreads` is set:
        if nthreads is None:
        
            print('\t >> Warning: tracing will be done WITHOUT parallelization:')
            print('\t    - It should take about {0:.2f} hours to trace all {1:} integrations.'.format(total_time * tso.shape[0], str(tso.shape[0])))

            # First, perform normal tracing:
            tic = time.time()
            for i in tqdm(range(tso.shape[0])):

                _, output_dictionary['traces']['y'][i, :] = trace_spectrum(tso[i, :, :], np.zeros(median_rate.shape),
                                                                           xstart = xstart, ystart = center_pixel, xend = xend,
                                                                           y_tolerance = 5, 
                                                                           method = trace_ccf_method,
                                                                           ccf_function = trace_ccf_function,
                                                                           ccf_parameters = trace_ccf_parameters
                                                                          )
            toc = time.time()
            total_time = (toc-tic)/3600.

        else:

            if not ray.is_initialized():

                ray.init(address='local', num_cpus = nthreads) 

            print('\t >> Tracing will be done via the ray library:')
            print('\t    - It should take about {0:.2f} hours to trace all {1:} integrations.'.format( (total_time * tso.shape[0]) / nthreads, str(tso.shape[0])))

            tic = time.time()
            # First, decorate the tracing function to make it ray-amenable:
            ray_trace_spectrum = ray.remote(trace_spectrum)

            # Prepare the handle for all traces:
            all_traces = []
            for i in range( tso.shape[0] ):

                all_traces.append( ray_trace_spectrum.remote(tso[i, :, :], np.zeros(median_rate.shape), 
                                                             xstart = xstart, ystart = center_pixel, xend = xend,
                                                             y_tolerance = 5, 
                                                             method = trace_ccf_method,
                                                             ccf_function = trace_ccf_function,
                                                             ccf_parameters = trace_ccf_parameters
                                                            ) 
                                 )

            # Run the process with ray:
            trace_results = ray.get(all_traces)
            # Save traces in the dictionary:
            for i in range(tso.shape[0]):

                _, output_dictionary['traces']['y'][i, :] = trace_results[i]

            toc = time.time()
            total_time = (toc-tic)/3600.

        # Next, go trace by trace correcting outliers and smoothing traces via a spline. We don't parallellize this as this seems to be pretty quick even for NIRSpec/PRISM:
        print('\t    - Done! Took {0:.2f} hours. Correcting outliers and smoothing traces...'.format(total_time))

        tic = time.time()
        for i in tqdm(range(tso.shape[0])):

            # Find outliers:
            idx_outliers, mfilter = outlier_detector(output_dictionary['traces']['y'][i, :],  
                                                     nsigma = trace_outlier_nsigma,
                                                     window = trace_outlier_window, 
                                                     return_filter = True)

            # Fix them if present:
            if len(idx_outliers) > 0: 

                output_dictionary['traces']['ycorrected'][i, :] = deepcopy(output_dictionary['traces']['y'][i, :])
                output_dictionary['traces']['ycorrected'][i, idx_outliers] = mfilter[idx_outliers]

            else:

                output_dictionary['traces']['ycorrected'][i, :] = output_dictionary['traces']['y'][i, :]

            # Smooth trace:
            _, output_dictionary['traces']['ysmoothed'][i, :] = fit_spline(x1, output_dictionary['traces']['ycorrected'][i, :],  
                                                                           nknots = nknots)

        toc = time.time()
        total_time = (toc-tic)

        print('\t    - Done! Took {0:.2f} seconds. Saving...'.format(total_time)) 
        pickle.dump( output_dictionary['traces'], open(outputfolder+'pipeline_outputs/traces'+actual_suffix+'.pkl', 'wb') )

    # Now that traces have been obtained, perform spectral extraction. For this, it is useful to get a "master" trace from our data:
    y1 = np.nanmedian( output_dictionary['traces']['ysmoothed'], axis = 0 )

    # If running optimal extraction, add suffix automatically:
    if optimal_extraction:

        suffix = 'optimal_' + suffix 
        actual_suffix = '_' + suffix
        if custom_extraction:
            import hashlib
            import json
            if shared_profile is not None:
                # New-model cache identity includes actual input/trace content.
                # Hash per integration, avoiding a second full-cube allocation.
                content = hashlib.sha256()
                shared_columns = output_dictionary['traces']['x']
                shared_mask = _Stage2SharedMask(input_dictionary,mask_dq,
                                                slice(shared_columns[0],shared_columns[-1]+1))
                for array in (tso, tso_err, output_dictionary['traces']['ysmoothed']):
                    for integration in array:
                        content.update(np.ascontiguousarray(integration).tobytes())
                for integration in shared_mask:
                    content.update(np.ascontiguousarray(integration).tobytes())
                extraction_settings['input_fingerprint'] = content.hexdigest()
            identity = json.dumps(extraction_settings, sort_keys=True, allow_nan=False,
                                  default=_shared_cache_json)
            actual_suffix += '_' + profile_method + '_' + hashlib.sha256(identity.encode()).hexdigest()[:12]

    # Extract spectra:
    if os.path.exists( outputfolder+'pipeline_outputs/spectra'+actual_suffix+'.pkl' ):

        print('\t >> Spectra found in file {0:}. Loading...'.format(outputfolder+'pipeline_outputs/spectra'+actual_suffix+'.pkl'))
        output_dictionary['spectra'] = pickle.load( open(outputfolder+'pipeline_outputs/spectra'+actual_suffix+'.pkl', 'rb') )

    else:

        print('\t >> Performing spectral extraction...') 
        tic = time.time()
        # Define spectral extraction parameters:
        # Initial time-series parameters:
        scale_1f_rows = defaults['scale_1f_rows']
        scale_1f_columns = defaults['scale_1f_columns']
        scale_1f_window = defaults['scale_1f_window']

        # Optimal extraction parameters:
        spectra_oe_polynomial_spacing = defaults['polynomial_spacing']
        spectra_oe_nsigma = defaults['nsigma']
        spectra_oe_polynomial_order = defaults['polynomial_order']

        # Spectral extraction, 1/f and background correction/removal parameters:
        if aperture_radius is None:

            spectra_aperture_radius = defaults['aperture_radius']

        else:

            spectra_aperture_radius = aperture_radius

        spectra_1f_inner_radius = defaults['onef_inner_radius']
        spectra_1f_outer_radius = defaults['onef_outer_radius']

        spectra_bkg_substraction = defaults['background_subtraction']
        spectra_bkg_inner_radius = defaults['background_inner_radius']
        spectra_bkg_outer_radius = defaults['background_outer_radius']

        spectra_oe_polynomial_spacing = extraction_options.get('polynomial_spacing', spectra_oe_polynomial_spacing)
        spectra_oe_polynomial_order = extraction_options.get('polynomial_order', spectra_oe_polynomial_order)
        spectra_oe_nsigma = extraction_options.get('nsigma', spectra_oe_nsigma)

        if spectra_bkg_substraction:

            # First, estimate the background using the median rate frame:
            in_trace_pixels = np.zeros(median_rate_nan.shape)
            in_trace_pixels[:] = np.nan
            out_of_trace_pixels = np.ones(median_rate_nan.shape)
            rows = np.arange(median_rate_nan.shape[0])

            for i in range( len(output_dictionary['traces']['x']) ):

                idx_in = np.where( np.abs( rows - y1[i] ) <= spectra_bkg_inner_radius)[0]
                out_of_trace_pixels[idx_in, output_dictionary['traces']['x'][i]] = np.nan  
                in_trace_pixels[idx_in, output_dictionary['traces']['x'][i]] = 1.

                if spectra_bkg_outer_radius is not None:

                    idx_out = np.where( np.abs( rows - y1[i] ) >= spectra_bkg_outer_radius)[0]  
                    out_of_trace_pixels[idx_out, output_dictionary['traces']['x'][i]] = np.nan

            bkg = np.nanmedian(median_rate_nan * out_of_trace_pixels, axis = 0)

            # Zero nans on background:
            idx_nan_bkg = np.where(np.isnan(bkg))
            bkg[idx_nan_bkg] = 0.

            # Now, remove this background signal from every integration:
            for i in range( tso.shape[0] ):

                tso[i, :, :] -= bkg

        # Next, perform spectral extraction. To this end, first generate a fast white-light lightcurve that we will 
        # use to perform the 1/f scaling (if the user wants to):
        if scale_1f:
    
            timeseries = np.nansum(tso[:, 
                                    scale_1f_rows[0]:scale_1f_rows[1], 
                                    scale_1f_columns[0]:scale_1f_columns[1]], 
                                    axis = (1,2)
                                  )

            mf = median_filter( timeseries / np.nanmedian(timeseries), scale_1f_window )


        spectra = np.zeros([tso.shape[0], len(y1)])
        spectra_err = np.zeros([tso.shape[0], len(y1)])
        
        # If 1/f noise removal/supression is activated, run that first:
        if scale_1f:

            print('\t    - Performing 1/f supression via median-scaling...')

            for i in tqdm(range(tso.shape[0])):

                if not single_trace_extraction:

                    x, y = output_dictionary['traces']['x'], output_dictionary['traces']['ysmoothed'][i, :]

                else:

                    x, y = output_dictionary['traces']['x'], y1

                tso[i, :, :] = correct_1f(tso[i, :, :],
                                          median_rate, 
                                          x, y, 
                                          scale_factor = mf[i], 
                                          inner_radius = spectra_1f_inner_radius, 
                                          outer_radius = spectra_1f_outer_radius
                                          )
        
        if optimal_extraction and shared_profile is not None:
            print('\t    - Training/evaluating a shared intrinsic profile.')
            x = output_dictionary['traces']['x']
            shared_traces = (np.broadcast_to(y1, (len(tso), len(y1))) if single_trace_extraction
                             else output_dictionary['traces']['ysmoothed'])
            spectra, spectra_err, P, Ps, shared_model = _extract_shared_sequence(
                tso[:, :, x[0]:x[-1]+1], tso_err[:, :, x[0]:x[-1]+1], shared_traces, y1,
                shared_profile, shared_profile_options, spectra_aperture_radius,
                spectra_oe_polynomial_spacing, profile_method, spectra_oe_polynomial_order,
                gp_options, spectra_oe_nsigma, nthreads, x, validity_mask=shared_mask)
            shared_model.save(outputfolder+'pipeline_outputs/shared_profile'+actual_suffix+'.npz')

        elif optimal_extraction:

            print('\t    - Performing spectral extraction via Optimal Extraction.')

            # First calculate light profiles for every integration:
            x = output_dictionary['traces']['x']
            extracted_columns = (x[-1]+1) - x[0]
            gp_mask = (_Stage2SharedMask(input_dictionary, mask_dq, slice(x[0], x[-1]+1))
                       if profile_method == 'gp' else None)
            Ps = np.zeros([tso.shape[0], tso.shape[1], extracted_columns])

            if nthreads is None:

                print('\t    - Calculating light profiles (without parallelization)...')
                for i in tqdm(range(tso.shape[0])):

                    if not single_trace_extraction:

                        x, y = output_dictionary['traces']['x'], output_dictionary['traces']['ysmoothed'][i, :]

                    else:

                        x, y = output_dictionary['traces']['x'], y1

                    optimal_data, optimal_variance = _optimal_frame(
                        tso[i, :, x[0]:x[-1]+1], tso_err[i, :, x[0]:x[-1]+1],
                        invalid_mask=None if gp_mask is None else gp_mask[i])
                    Ps[i, :, :] = getP(optimal_data, y, spectra_aperture_radius, 1., 1.,
                                       spectra_oe_nsigma, spectra_oe_polynomial_spacing, spectra_oe_polynomial_order,
                                       data_variance=optimal_variance,
                                       backend=extraction_backend, profile_method=profile_method, gp_options=gp_options)

            else:

                print('\t    - Calculating light profiles via parallelization...')

                if not ray.is_initialized():

                    ray.init(address='local', num_cpus = nthreads)

                ray_getP = ray.remote(getP)

                # Prepare the handle for all P's:
                all_Ps = [] 
                for i in range( tso.shape[0] ):

                    if not single_trace_extraction:

                        x, y = output_dictionary['traces']['x'], output_dictionary['traces']['ysmoothed'][i, :]

                    else:

                        x, y = output_dictionary['traces']['x'], y1

                    optimal_data, optimal_variance = _optimal_frame(
                        tso[i, :, x[0]:x[-1]+1], tso_err[i, :, x[0]:x[-1]+1],
                        invalid_mask=None if gp_mask is None else gp_mask[i])
                    all_Ps.append( ray_getP.remote(optimal_data, y, spectra_aperture_radius, 1., 1.,
                                                    spectra_oe_nsigma, spectra_oe_polynomial_spacing, spectra_oe_polynomial_order,
                                                    data_variance=optimal_variance,
                                                    backend=extraction_backend, profile_method=profile_method, gp_options=gp_options
                                                  ) 
                                 )

                # Run the process with ray:
                P_results = ray.get(all_Ps)

                # Save Ps in the target array:
                for i in range(tso.shape[0]):

                    Ps[i, :, :] = P_results[i]

            # Get the median light fraction and renormalize it:
            P = np.nanmedian(Ps, axis = 0)
            for i in range(P.shape[1]):

                P[:, i] = P[:, i] / np.nansum( P[:, i] )

            # Now use this to extract the spectra:
            if nthreads is None:

                print('\t    - Done! Extracting optimal spectra (without parallelization)...')
                for i in tqdm(range(tso.shape[0])):

                    if not single_trace_extraction:

                        x, y = output_dictionary['traces']['x'], output_dictionary['traces']['ysmoothed'][i, :]

                    else:

                        x, y = output_dictionary['traces']['x'], y1

                    optimal_data, optimal_variance = _optimal_frame(
                        tso[i, :, x[0]:x[-1]+1], tso_err[i, :, x[0]:x[-1]+1],
                        invalid_mask=None if gp_mask is None else gp_mask[i])
                    opt_spec = getOptimalSpectrum(optimal_data, y, spectra_aperture_radius, 1., 1.,
                                                  spectra_oe_nsigma, spectra_oe_polynomial_spacing, spectra_oe_polynomial_order,
                                                  data_variance=optimal_variance,
                                                  backend=extraction_backend, profile_method=profile_method,
                                                  P = P
                                                 )

                    spectra[i, :] = deepcopy(opt_spec[1, :])
                    spectra_err[i, :] = np.sqrt( 1. / opt_spec[2, :] )

            else:

                print('\t    - Done! Extracting optimal spectra via parallelization...')
                
                if not ray.is_initialized():

                    ray.init(address='local', num_cpus = nthreads)

                ray_getOS = ray.remote(getOptimalSpectrum)

                # Prepare the handle for all P's:
                all_spectra = []
                for i in range( tso.shape[0] ):

                    if not single_trace_extraction:

                        x, y = output_dictionary['traces']['x'], output_dictionary['traces']['ysmoothed'][i, :]

                    else:

                        x, y = output_dictionary['traces']['x'], y1

                    optimal_data, optimal_variance = _optimal_frame(
                        tso[i, :, x[0]:x[-1]+1], tso_err[i, :, x[0]:x[-1]+1],
                        invalid_mask=None if gp_mask is None else gp_mask[i])
                    all_spectra.append( ray_getOS.remote(optimal_data, y, spectra_aperture_radius, 1., 1.,
                                                         spectra_oe_nsigma, spectra_oe_polynomial_spacing, spectra_oe_polynomial_order,
                                                         data_variance=optimal_variance,
                                                         backend=extraction_backend, profile_method=profile_method,
                                                         P = P
                                                        )
                                      )

                # Run the process with ray:
                OS_results = ray.get(all_spectra)

                # Save spectra in the target arrays:
                for i in range(tso.shape[0]):

                    spectra[i, :], spectra_err[i, :] = OS_results[i][1,:], np.sqrt( 1. / OS_results[i][2,:] )

        else:
    
            print('\t    - Performing spectral extraction via Simple Extraction...')
            
            for i in tqdm(range(tso.shape[0])):

                if not single_trace_extraction:

                    x, y = output_dictionary['traces']['x'], output_dictionary['traces']['ysmoothed'][i, :]

                else:

                    x, y = output_dictionary['traces']['x'], y1
        
                spectra[i, :], spectra_err[i, :] = getSimpleSpectrum(tso[i, :, :], 
                                                                     x,
                                                                     y, 
                                                                     spectra_aperture_radius, 
                                                                     error_data=tso_err[i, :, :], 
                                                                     correct_bkg=False
                                                                    )

        output_dictionary['spectra'] = {}
        output_dictionary['spectra']['times'] = input_dictionary['times']
        output_dictionary['spectra']['original'] = spectra
        output_dictionary['spectra']['original_err'] = spectra_err

        if optimal_extraction:

            output_dictionary['spectra']['Ps'] = Ps
            output_dictionary['spectra']['P'] = P
            if shared_profile is not None:
                output_dictionary['spectra']['shared_profile'] = shared_model
            if custom_extraction:
                output_dictionary['spectra']['extraction_settings'] = extraction_settings

        # Now correct for outliers not accounted for in previous steps:
        master_spectra = np.zeros(spectra.shape)

        for i in range(spectra.shape[0]):
            
            master_spectra[i, :] = spectra[i,:] / np.nanmedian(spectra[i,:])
            
        master_spectrum = np.zeros(spectra.shape[1])
        sigma_master_spectrum = np.zeros(spectra.shape[1])

        for i in range(spectra.shape[1]):
            
            median = np.nanmedian(master_spectra[:, i])
            master_spectrum[i], sigma_master_spectrum[i] = median, \
                                                           1.2533*get_mad_sigma(median, master_spectra[:, i])

        corrected_spectra = deepcopy(spectra)

        corrected_spectra_err = deepcopy(spectra_err)

        for i in range(spectra.shape[0]):
            
            # First, get median to scale:
            median = (np.nanmedian(spectra[i, :]) if profile_method == 'gp'
                      else np.median(spectra[i, :]))

            # Scale master spectrum and sigma:
            model = master_spectrum * median
            sigma = sigma_master_spectrum * median
            
            # Identify bad pixels/columns:
            residuals = np.abs(spectra[i, :] - model)
            idx_bad = np.where(residuals > 5 * sigma)[0]
            
            # Replace:
            if len(idx_bad) != 0:
                
                corrected_spectra[i, idx_bad] = model[idx_bad]
                corrected_spectra_err[i, idx_bad] = sigma[idx_bad]

        # Save to output dictionary:
        output_dictionary['spectra']['corrected'] = corrected_spectra
        output_dictionary['spectra']['corrected_err'] = corrected_spectra_err 

        print('\t    - Done! Extracting wavelength map...')
        # Extract wavelength solution --- a bit different depending on the instrument, but all require the assign_wcs step to be ran:
        results = calwebb_spec2.assign_wcs_step.AssignWcsStep.call(input_dictionary['rampstep'][0])
        wcs_out = assign_wcs.nrs_wcs_set_input(results, results.meta.instrument.fixed_slit)

        if instrument_name == 'NIRSPEC':

            # Get the output wavelength map from the bounding box along with bounding box coordinates. Define bounding box depending on the 
            # subarray size (by default, wcs_out maps the slit in the detector for NIRSpec):
            wcs_out.bounding_box = ( (-0.5, input_dictionary['rampstep'][0].data.shape[2]-0.5), 
                                     (-0.5, input_dictionary['rampstep'][0].data.shape[1]-0.5) 
                                   )
            bb_columns, bb_rows = wcstools.grid_from_bounding_box( wcs_out.bounding_box )
            _, _, bb_wavelength_map = wcs_out(bb_columns, bb_rows)
    
            # Prepare and fill wavelength map given this bounding box (fill with nans because later nans are ommited):
            wavelength_map = np.full([ tso.shape[1], tso.shape[2] ], np.nan)

            for i in range( bb_wavelength_map.shape[0] ):

                for j in range( bb_wavelength_map.shape[1] ):

                        wavelength_map[int(bb_rows[i, j]), int(bb_columns[i, j])] = bb_wavelength_map[i, j]

            output_dictionary['spectra']['wavelength_map'] = wavelength_map

            # Now, using the wavelength map, get average wavelength per column (nans will be ommited):
            wavelengths = getSimpleSpectrum(wavelength_map,
                                            x,
                                            y1,
                                            spectra_aperture_radius,
                                            method = 'average'
                                           )

            output_dictionary['spectra']['wavelengths'] = wavelengths

        # Save output:
        toc = time.time()
        total_time = (toc-tic)

        print('\t    - Done! Took {0:.2f} seconds. Saving...'.format(total_time)) 
        pickle.dump( output_dictionary['spectra'], open(outputfolder+'pipeline_outputs/spectra'+actual_suffix+'.pkl', 'wb') )

    # Generate quicklook white-light lightcurve:
    output_dictionary['whitelight'] = np.nansum(output_dictionary['spectra']['corrected'], axis = 1)
    norm_factor = np.nanmedian( output_dictionary['whitelight'] )
    output_dictionary['whitelight'] = output_dictionary['whitelight'] / norm_factor

    # Same, errors:
    output_dictionary['whitelight_err'] = np.sqrt( np.nansum(output_dictionary['spectra']['corrected_err']**2, axis = 1) ) / norm_factor

    # Standard products (see transitspectroscopy.core.products):
    output_dictionary['products'] = _spectral_products(output_dictionary, mode_definition, config,
                                                       instrument_detector)
    if write_products:

        from .core import products as standard_products, provenance
        manifest = provenance.build_manifest(config = config.as_dict(), step = 'stage2', mode = mode_definition.name,
                                             input_models = [str(getattr(model.meta, 'filename', '')) for model in input_dictionary['rampstep']])
        standard_products.write_products(output_dictionary['products'], outputfolder+'pipeline_outputs/products'+actual_suffix, manifest)

    if nthreads is not None:

        if ray.is_initialized():

            ray.shutdown()

    return output_dictionary


def _spectral_products(output_dictionary, mode, config, detector):
    """Standard products of the spectral Stage 2: corrected and original spectra, white light."""
    from .core.products import LightCurveSet, SpectroscopicTimeSeries
    spectra, traces = output_dictionary['spectra'], output_dictionary['traces']
    base = str(detector).lower() if detector else 'spectra'
    metadata = dict(output_dictionary['metadata'], mode = mode.name, extraction = config.as_dict())
    products = {}
    for variant in ('corrected', 'original'):

        name = base+'_'+variant
        products[name] = SpectroscopicTimeSeries(times = spectra['times'], wavelength = spectra['wavelengths'],
                                                 flux = spectra[variant], flux_err = spectra[variant+'_err'],
                                                 columns = traces['x'], trace = traces['ysmoothed'], name = name,
                                                 flux_unit = 'DN/s', metadata = dict(metadata, variant = variant))

    products[base+'_whitelight'] = LightCurveSet(times = spectra['times'], flux = output_dictionary['whitelight'][None],
                                                 flux_err = output_dictionary['whitelight_err'][None],
                                                 labels = ['white'], name = base+'_whitelight',
                                                 metadata = dict(metadata, variant = 'corrected',
                                                                 band = 'sum over all extracted columns'))
    return products
