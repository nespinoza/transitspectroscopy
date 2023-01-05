import os
import numpy as np

class load(object):
    """
    Given a dictionary with priors (or a filename pointing to a prior file) and data either 
    given through arrays or through files containing the data, this class loads data into an 
    object which holds all the information about the dataset. Example usage:

               >>> dataset = ts.retrievals.load(priors=priors, wavelengths=w, data=data, errors=errors)

    Or, also,

               >>> dataset = ts.retrievals.load(input_folder = folder)  

    :param priors: (optional, dict or string)
        This can be either a python ``string`` or a python ``dict``. See the docs on how to set either 
        (if familiar with `juliet` priors, this is the very same format).

    :param wavelengths: (dictionary)
        Dictionary whose keys are instrument names. Each key should contain either a simple, 1-dimensional 
        array with wavelengths or be a `[N, 2]` array, in which the first column is the lower wavelength 
        bound and the second is the upper wavelength bound of each of the `N` datapoints. Wavelenghts are 
        assumed to be in microns (um).

    :param depths: (dictionary)
        Dictionary whose keys are instrument names. Each key should contain a simple, 1-dimensional array 
        containing the depths in parts-per-million (ppm). If fitting the two limbs individually, this should 
        be a `[N, 2]` array; first column being the hot limb, second the cold limb.

    :param errors: (dictionary)
        Same as `data`, but for the errors. If fitting limbs individually, this should be a `[N, 3]` array, 
        first column being the standard-deviation of the hot limb, second the one for the cold limb and third 
        the covariance between the two.

    :param resolution: (optional, int or dict)
        This defines the resolution (lambda / delta_lambda) of the data. It can be common to all the data, 
        or it can be a dictionary whose keys are instrument names, and thus a different resolution can be 
        selected for different datasets.

    :param input_folder: (optional, string)
        Python ``string`` containing the path to a folder containing all the input data. This input folder has 
        to contain a ``priors.dat`` file with the priors and  a ``data.dat`` file containing: 

        - the wavelengths (first two columns defining lower and upper wavelengths for each bin), 
        - the depths (3rd column if fitting a simple transit spectrum; 3rd & 4th if fitting the limbs --- all in ppm), 
        - the errors (4th column if fitting a simple transit spectrum; 5th, 6th and 7th for the hot, cold and 
          covariance between the limbs --- all in ppm),
        - the instrument names (5th column if fitting a simple transit spectrum, 8th if fitting the limbs). 

    :param output_folder: (optional, string)
        Python ``string`` defining the name of an optional output folder. If defined, this is where all the output data 
        will be saved.
    """

    def read_priors(self, priorname):
        """
        This function takes a string and spits out the prior in dictionary format. 
        """

        fin = open(priorname)
        priors = {}
        starting_point = {}

        while True:

            line = fin.readline()

            if line != '':

                if line[0] != '#':

                    # Read in line containing the prior info:
                    prior_vector = line.split()

                    # Extract name, prior distribution and hyperparameters. Also check 
                    # if user is giving a starting point or not --- and save it:
                    if len(prior_vector) == 3:

                        parameter, prior_name, vals = prior_vector

                    else:

                        parameter, prior_name, vals, sp = prior_vector
                        starting_point[parameter] = np.double(sp)

                    # Reutilize some variable names; free prior_name and vals of any 
                    # loose spaces; create key for the parameter: 
                    prior_name = prior_name.split()[0]
                    vals = vals.split()[0]
                    priors[parameter] = {}

                    # Stick priors to dictionary depending on prior type. If fixed, easy; 
                    # the play with the other prior types as needed:
                    if prior_name.lower() == 'fixed':

                        priors[parameter]['distribution'] = prior_name.lower()
                        priors[parameter]['hyperparameters'] = np.double(vals)
                        priors[parameter]['cvalue'] = np.double(vals)

                    else:

                        priors[parameter]['distribution'] = prior_name.lower()

                        if priors[parameter]['distribution'] != 'truncatednormal':

                            if priors[parameter]['distribution'] == 'exponential':

                                priors[parameter]['hyperparameters'] = [ np.double(vals) ]

                            else:

                                v1, v2 = vals.split(',')

                                priors[parameter]['hyperparameters'] = [
                                    np.double(v1), np.double(v2)
                                ]

                        else:

                            v1, v2, v3, v4 = vals.split(',')
                            priors[parameter]['hyperparameters'] = [
                                np.double(v1),
                                np.double(v2),
                                np.double(v3),
                                np.double(v4)

                            ]
                        priors[parameter]['cvalue'] = 0.

        return priors

    def set_parameters(self, priors):
        """
        Function that saves the number of parameters and instruments onto the `load` object.
        """

        self.nparameters = 0
        self.instruments = []        

        parameters = list(priors.keys())

        for k in parameters:

            if priors[k]['distribution'] != 'fixed':

                self.nparameters += 1

            if 'offset_' in k:

                instrument = k.split('_')[1]
                if instrument not in self.instruments:

                    self.instruments.append(instrument)

    def set_wavelengths(self, wavelengths):
        """
        Function that saves the wavelengths in "proper formatting" to the `load` object. Also checks wavelengths against 
        user input.
        """

        # Scream if input is not right:
        if type(wavelengths) is not dict:

            wtype = str(type(wavelengths)).split("'")[1]
            raise Exception('INPUT ERROR: Wavelengths are type '+wtype+'; but they should be dictionaries.')

        self.wavelengths = {}

        for instrument in self.instruments:

            w = wavelengths[instrument]

            # If user ingested a one-dimensional array, create a new 2-D array with upper and lower limits on the 
            # wavelengths assuming these are ordered for the instrument:
            if len(w.shape) == 1:

                self.wavelengths[instrument] = np.zeros( [len(w), 2] )

                for i in range( len(wavelengths[instrument]) ):

                    if i == 0:

                        delta_w = ( wavelengths[instrument][1] - wavelengths[instrument][0] ) * 0.5

                    else:

                        delta_w = ( wavelengths[instrument][i] - wavelengths[instrument][i-1] ) * 0.5

                    self.wavelengths[instrument][0] = wavelengths[instrument][i] - delta_w
                    self.wavelengths[instrument][1] = wavelengths[instrument][i] + delta_w

            elif len(w.shape) > 2:

                raise Exception('INPUT ERROR: Wavelengths for instrument '+instrument+' need to be either uni-dimensional or of shape [N, 2].')

            else:

                self.wavelengths[instrument] = w
               
    def read_input_data(self, input_folder):

        fin = open(input_folder + 'data.dat', 'r')
        wavelengths = {}
        depths = {}
        errors = {}

        while True:

            line = fin.readline()
    
            if line != '':

                vector = line.split()

                instrument = vector[-1].split()[0]

                wlow, wup = np.double( vector[0] ), np.double( vector[1] )

                if instrument not in list( wavelengths.keys() ):

                    wavelengths[instrument] = np.array([wlow, wup])
                    depths[instrument] = np.array([ np.double( vector[2] ) ])
                    errors[instrument] = np.array([ np.double( vector[3] ) ])

                else:

                    wavelengths[instrument] = np.vstack(( wavelengths[instrument], np.array([wlow, wup]) ))
                    depths[instrument] = np.append( depths[instrument], np.double( vector[2] ) )
                    errors[instrument] = np.append( errors[instrument], np.double( vector[3] ) )

            else:
        
                fin.close()
                break

        return wavelengths, depths, errors

    def save(self):

        # Save everything to an output folder if it not already exists:
        if not os.path.exists(self.output_folder):

            instruments = list( self.depths.keys() )
            fpriors = open(self.output_folder + 'priors.dat', 'w')
            fdata = open(self.output_folder + 'data.dat', 'w')

            # Save data first:
            for instrument in instruments:

                for i in range( len(self.depths[instrument]) ):

                    wl, wu = self.wavelengths[instrument][i][0], self.wavelengths[instrument][i][1]
                    depth, error = self.depths[instrument][i], self.errors[instrument][i]

                    fdata.write('{0:.5f} {1:.5f} {2:.1f} {3:.1f} {4:}\n'.format(wl, wu, depth, error))

            # Save priors:
            parameters = list( self.priors.keys() )

            for parameter in parameters.keys():

                distribution = priors[parameter]['distribution']
                hyperparameters = ','.join( list( priors[parameter]['hyperparameters'].astype('str') ) )

                fpriors.write('{0:}\t{1:}\t{2:}\n'.format(parameter, distribution, hyperparameters))

            fdata.close()
            fpriors.close()

    def fit(self, **kwargs):

        return fit(self, **kwargs)

    def __init__(self, priors = None, wavelengths = None, depths = None, errors = None, resolution = 100, \
                 input_folder = None, output_folder = None):

        # If input_folder is given, read all inputs from the data file:
        if input_folder is not None:

            if input_folder[-1] != '/':
                
                input_folder += '/'

            priors = input_folder + 'priors.dat'

            wavelengths, depths, errors = self.read_input_data(input_folder)

            # Inhert input_folder as the output_folder if not given:
            if output_folder is None:

                self.output_folder = input_folder

        # Read priors if a file is given; read it if it's already a dictionary:
        if type(priors) == str:

            self.prior_fname = priors
            self.priors = self.read_priors(priors)

        else:

            self.priors = priors

        # Get general checks of the to-be-fitted parameters. Count number of free parameters user wants to fit, get 
        # instruments out of it too:
        self.set_parameters(priors)

        # Deal with the wavelengths. We want them such that they are `[N, 2]` arrays, in which the first column is the lower 
        # wavelength range and the second the upper. Check instrument by instrument on whether the user gave one or the other, 
        # and save accordingly:
        self.set_wavelengths(wavelengths)

        # Set depths and errors:
        self.depths = depths
        self.errors = errors

        # Scream to the user if something is missing or the input is incorrect:
        if self.depths is None:

            raise Exception('INPUT ERROR: Depths were not given; these need to be ingested via retrievals.load(..., depths = depths).')

        if self.errors is None:

            raise Exception('INPUT ERROR: Errors were not given; these need to be ingested via retrievals.load(..., errors = errors).')

        if type(depths) is not dict:

            dtype = str(type(depths)).split("'")[1]
            raise Exception('INPUT ERROR: Depths are type '+dtype+'; but they should be dictionaries.')

        if type(errors) is not dict:

            etype = str(type(errors)).split("'")[1]
            raise Exception('INPUT ERROR: Errors are type '+etype+'; but they should be dictionaries.')

        # If output_folder is given (and no input_folder), save all data to this folder:
        if ( input_folder is None ) and ( output_folder is not None ):

            self.output_folder = output_folder
            self.save()

class fit(object):

    def __init__(self, data, sampler = 'dynesty', n_live_points = 500, \
                 nwalkers = 100, nsteps = 300, nburnin = 500, emcee_factor = 1e-4, \
                 nthreads = None):

        self.sampler = sampler

        self.data = data
