import numpy as np

import juliet

try:

    import ray

except:

    print('Could not import the "ray" library. If you want to parallelize lightcurve fitting, please install by doing "pip install ray".')

def fit_lightcurves(data_dictionaries, priors, sampler = 'multinest', starting_points = None, nthreads = None, **kwargs):
    """
    This is a master function that will fit lightcurves simultaneously to several 
    independant wavelength bins with multi-processing.

    Parameters
    ----------

    data_dictionaries : dict
        Dictionaries containing the data. Each key of this dictionary holds a different 
        dataset that wants to be fitted in parallel (e.g., `data_dictionary.keys()` can 
        give back ['wav1', 'wav2', 'wav3'...], etc.). Each of those keys, in turn, is 
        a dictionary that must have three keys: `times`, `flux` and 'error'. Each should 
        be numpy arrays of the same length --- corresponding to the times, fluxes and errors.
 
        Additional extra keys that can be added: 

           - `GP_external_parameters`: Array having all external parameters that want to 
                                       be added as GP regressors. Must have dimensions of 
                                       (len(times), number-of-regressors).

           - `linear_external_parameters`: Same as `GP_external_parameters`, but for linear 
                                           model components.

    priors : dict
        Prior dictionary a-la-juliet for each dataset.

    sampler : string
        (optional) Sampler to be used on the fits. Default is `multinest`.

    starting_points : dict
        (optional) If sampler is `emcee`, this dictionary must contain the starting points for 
        each dataset fit. Keys should be the same as the `priors` keys; inside, starting points 
        have to be defined for each dataset, e.g., `starting_points[1.31]['p_p1'] = 0.1` and so 
        on. This is ignored if Nested Samplers are being used.

    nthreads : int
        (optional) Number of threads to run the lightcurve fits on. By default there is no parallel 
        processing (so this is `None`). If an int, `ray` is activated to run fits in parallel.

    Returns
    -------

    results : dict
        Dictionary with all results --- has same keys as input dictionary; results are 
        stored on each dict key.
    """

    if starting_points is None:
        starting_points = dict.fromkeys(data_dictionaries.keys(), {})

    # Create results output dictionary right away:
    results = dict.fromkeys(data_dictionaries.keys(), [])

    # Get key names (useful later, to fill results):
    keynames = list(data_dictionaries.keys())

    # Two ways to go depending if multithreading is initiated or not:
    if nthreads is not None:

        # Initialize ray:
        ray.init(num_cpus = nthreads)

        # Set juliet fits as remotes:
        all_fits = []
        for i in range(len(keynames)):
        
            # Add fits in Ray-remotes:
            keyname = keynames[i]
            all_fits.append( fit_data.remote(data_dictionaries[keyname], \
                                             priors[keyname], output_folder = keyname, \
                                             starting_point = starting_points[keyname], 
                                             sampler = sampler, **kwargs) )

        # Run the thing!
        ray_results = ray.get(all_fits)

        # Return results ordered by keyname on the results dictionary:
        for i in range(len(keynames)):

            keyname = keynames[i]
            results[keyname] = ray_results[i]

    else:

        results = {}

        for i in range(len(keynames)):

            keyname = keynames[i]

            results[keyname] = notremote_fit_data( data_dictionaries[keyname], \
                                                   priors[keyname], output_folder = keyname, \
                                                   starting_point = starting_points[keyname], \
                                                   sampler = sampler, **kwargs) 

    return results

@ray.remote
def fit_data(data_dictionary, priors, ld_laws = 'quadratic', output_folder = None, starting_point = {}, sampler = 'multinest', **kwargs):

    """
    The un-decorated version of this function simply fits a single input dataset.

    Parameters
    ----------

    data_dictionary : dict
        Dictionary containing the data. It's expected this dictionary contains the following 
        keys:

           - `times`, containing the time-stamps of the time-series.

           - `flux`, containing the relative flux of the time-series.

           - `error`, containing the errors on those relative fluxes.
 
        Additionally, optinal extra keys that can be added: 

           - `GP_external_parameters`: Array having all external parameters that want to 
                                       be added as GP regressors. Must have dimensions of 
                                       (len(times), number-of-regressors).

           - `linear_external_parameters`: Same as `GP_external_parameters`, but for linear 
                                           model components.

    priors : dict
        Prior dictionary a-la-juliet for the dataset. See, e.g., the juliet docs on how to 
        set those: https://juliet.readthedocs.io/en/latest/tutorials/transitfits.html

    ld_laws : string
        Limb-darkening laws to use. Default is quadratic.

    output_folder : string
        (optional) String containing the output folder. If not given, a random number is 
        assigned to the output.

    starting_point : dict
        (optional) Parameter starting point in case `emcee` sampler is being used. Ignored if 
        Nested Samplers are used. Must have starting point values for each non-fixed key in 
        the `priors` dict.

    sampler : string
        (optional) Sampler to be used to run the fits. Default is `multinest`.

    Returns
    -------

    results : juliet.dataset
        A juliet object containing all the information about the fit, including posterior distributions.

    """

    all_keys = list(data_dictionary.keys())

    if output_folder is None:

        output_folder = 'out_' + str(np.random.randint(0, 9999999))

    # Set dataset depending on whether external parameters are given for a GP or linear 
    # regressor. First, set dicts on juliet-friendly formats in prep for fits:

    tjuliet, fjuliet, ferrjuliet = {}, {}, {}
    tjuliet['SOSS'], fjuliet['SOSS'], ferrjuliet['SOSS'] = data_dictionary['times'], data_dictionary['flux'], data_dictionary['error']

    if 'GP_external_parameters' not in all_keys and 'linear_external_parameters' not in all_keys:

        dataset = juliet.load(priors = priors, t_lc = tjuliet, \
                              y_lc = fjuliet, \
                              yerr_lc = ferrjuliet, \
                              out_folder = output_folder, \
                              starting_point = starting_point, \
                              ld_laws = ld_laws)

    elif 'GP_external_parameters' in all_keys and 'linear_external_parameters' not in all_keys:

        GP_regressors = {}
        GP_regressors['SOSS'] = data_dictionary['GP_external_parameters']
        dataset = juliet.load(priors = priors, t_lc = tjuliet, \
                              y_lc = fjuliet, \
                              yerr_lc = ferrjuliet, \
                              out_folder = output_folder, \
                              GP_regressors_lc = GP_regressors, \
                              starting_point = starting_point, \
			      ld_laws = ld_laws)

    elif 'GP_external_parameters' not in all_keys and 'linear_external_parameters' in all_keys:

        linear_regressors = {}
        print('linear regressor detected')
        linear_regressors['SOSS'] = data_dictionary['linear_external_parameters']
        dataset = juliet.load(priors = priors, t_lc = tjuliet, \
                              y_lc = fjuliet, \
                              yerr_lc = ferrjuliet, \
                              out_folder = output_folder, \
                              linear_regressors_lc = linear_regressors, \
                              starting_point = starting_point, \
			      ld_laws = ld_laws)

    else:

        GP_regressors, linear_regressors = {}, {}
        GP_regressors['SOSS'] = data_dictionary['GP_external_parameters']
        linear_regressors['SOSS'] = data_dictionary['linear_external_parameters']

        dataset = juliet.load(priors = priors, t_lc = tjuliet, \
                              y_lc = fjuliet, \
                              yerr_lc = ferrjuliet, \
                              out_folder = output_folder, \
                              GP_regressors_lc = GP_regressors,\
                              linear_regressors_lc = linear_regressors, \
                              starting_point = starting_point, \
                              ld_laws = ld_laws)

    # With the dataset set, run the fit:
    dataset.fit(sampler = sampler, **kwargs)

def notremote_fit_data(data_dictionary, priors, ld_laws = 'quadratic', output_folder = None, starting_point = {}, sampler = 'multinest', **kwargs):

    """
    The un-decorated version of fit_data(). This function simply fits a single input dataset --- useful when running fits in serial mode.

    Parameters
    ----------

    data_dictionary : dict
        Dictionary containing the data. It's expected this dictionary contains the following 
        keys:

           - `times`, containing the time-stamps of the time-series.

           - `flux`, containing the relative flux of the time-series.

           - `error`, containing the errors on those relative fluxes.
 
        Additionally, optinal extra keys that can be added: 

           - `GP_external_parameters`: Array having all external parameters that want to 
                                       be added as GP regressors. Must have dimensions of 
                                       (len(times), number-of-regressors).

           - `linear_external_parameters`: Same as `GP_external_parameters`, but for linear 
                                           model components.

    priors : dict
        Prior dictionary a-la-juliet for the dataset. See, e.g., the juliet docs on how to 
        set those: https://juliet.readthedocs.io/en/latest/tutorials/transitfits.html

    output_folder : string
        (optional) String containing the output folder. If not given, a random number is 
        assigned to the output.

    starting_point : dict
        (optional) Parameter starting point in case `emcee` sampler is being used. Ignored if 
        Nested Samplers are used. Must have starting point values for each non-fixed key in 
        the `priors` dict.

    sampler : string
        (optional) Sampler to be used to run the fits. Default is `multinest`.

    Returns
    -------

    results : juliet.dataset
        A juliet object containing all the information about the fit, including posterior distributions.

    """

    all_keys = list(data_dictionary.keys())

    if output_folder is None:

        output_folder = 'out_' + str(np.random.randint(0, 9999999))

    # Set dataset depending on whether external parameters are given for a GP or linear 
    # regressor. First, set dicts on juliet-friendly formats in prep for fits:

    tjuliet, fjuliet, ferrjuliet = {}, {}, {}
    tjuliet['SOSS'], fjuliet['SOSS'], ferrjuliet['SOSS'] = data_dictionary['times'], data_dictionary['flux'], data_dictionary['error']

    if 'GP_external_parameters' not in all_keys and 'linear_external_parameters' not in all_keys:

        dataset = juliet.load(priors = priors, t_lc = tjuliet, \
                              y_lc = fjuliet, \
                              yerr_lc = ferrjuliet, \
                              out_folder = output_folder, \
                              starting_point = starting_point, \
		              ld_laws = ld_laws)

    elif 'GP_external_parameters' in all_keys and 'linear_external_parameters' not in all_keys:

        GP_regressors = {}
        GP_regressors['SOSS'] = data_dictionary['GP_external_parameters']
        dataset = juliet.load(priors = priors, t_lc = tjuliet, \
                              y_lc = fjuliet, \
                              yerr_lc = ferrjuliet, \
                              out_folder = output_folder, \
                              GP_regressors_lc = GP_regressors, \
                              starting_point = starting_point, \
			      ld_laws = ld_laws)

    elif 'GP_external_parameters' not in all_keys and 'linear_external_parameters' in all_keys:

        linear_regressors = {}
        linear_regressors['SOSS'] = data_dictionary['linear_external_parameters']
        dataset = juliet.load(priors = priors, t_lc = tjuliet, \
                              y_lc = fjuliet, \
                              yerr_lc = ferrjuliet, \
                              out_folder = output_folder, \
                              linear_regressors_lc = linear_regressors, \
                              starting_point = starting_point, \
			      ld_laws = ld_laws)

    else:

        GP_regressors, linear_regressors = {}, {}
        GP_regressors['SOSS'] = data_dictionary['GP_external_parameters']
        linear_regressors['SOSS'] = data_dictionary['linear_external_parameters']

        dataset = juliet.load(priors = priors, t_lc = tjuliet, \
                              y_lc = fjuliet, \
                              yerr_lc = ferrjuliet, \
                              out_folder = output_folder, \
                              GP_regressors_lc = GP_regressors,\
                              linear_regressors_lc = linear_regressors, \
                              starting_point = starting_point, \
			      ld_laws = ld_laws)

    # With the dataset set, run the fit:
    return dataset.fit(sampler = sampler, **kwargs)
