import numpy as np

from scipy.interpolate import splrep, splev

def get_phases(t,P,t0):
    """
    Given input times, a period (or posterior dist of periods)
    and time of transit center (or posterior), returns the 
    phase at each time t.
    """
    if type(t) is not float:
        phase = ((t - np.median(t0))/np.median(P)) % 1
        ii = np.where(phase>=0.5)[0]
        phase[ii] = phase[ii]-1.0
    else:
        phase = ((t - np.median(t0))/np.median(P)) % 1
        if phase>=0.5:
            phase = phase - 1.0
    return phase

def function_quantiles(X, alpha = 0.68, method = 'median'):
    """
    If `X` is a matrix of length N x M, where there are N evaluations of a model at M index-points, this function returns the 
    credibility band of the model given these samples.

    Parameters
    ----------

    X : numpy.array
        Array containing N evaluations of a model in the rows at M index (e.g., time-) points.
    alpha : float
        Credibility band percentage.
    method : string
        Method to use to generate the bands; `median` is default (and only supported mode for now).


    Returns
    -------
    median_model : numpy.array
        Array of length M denoting the median model
    upper_band : numpy.array
        Array of length M denoting the upper `alpha`*100 credibility band.
    lower_band : numpy.array
        Array of length M denoting the lower `alpha`*100 credibility band.

    """

    median_model, lower_band, upper_band = np.zeros(X.shape[1]), np.zeros(X.shape[1]), np.zeros(X.shape[1])

    for i in range(X.shape[1]):
        median_model[i], upper_band[i], lower_band[i] = get_quantiles(X[:,i], alpha = alpha)

    return median_model, upper_band, lower_band

def get_quantiles(dist,alpha = 0.68, method = 'median'):
    """ 
    get_quantiles function
    DESCRIPTION
        This function returns, in the default case, the parameter median and the error% 
        credibility around it. This assumes you give a non-ordered 
        distribution of parameters.
    OUTPUTS
        Median of the parameter,upper credibility bound, lower credibility bound
    """
    ordered_dist = dist[np.argsort(dist)]
    param = 0.0 
    # Define the number of samples from posterior
    nsamples = len(dist)
    nsamples_at_each_side = int(nsamples*(alpha/2.)+1)
    if(method == 'median'):
       med_idx = 0 
       if(nsamples%2 == 0.0): # Number of points is even
          med_idx_up = int(nsamples/2.)+1
          med_idx_down = med_idx_up-1
          param = (ordered_dist[med_idx_up]+ordered_dist[med_idx_down])/2.
          return param,ordered_dist[med_idx_up+nsamples_at_each_side],\
                 ordered_dist[med_idx_down-nsamples_at_each_side]
       else:
          med_idx = int(nsamples/2.)
          param = ordered_dist[med_idx]
          return param,ordered_dist[med_idx+nsamples_at_each_side],\
                 ordered_dist[med_idx-nsamples_at_each_side]

def fit_spline(x, y, nknots = None, x_knots = None):
    """
    This function fits a spline to data `x` and `y`. The code can be use in three ways: 

    1.  Passing a value to `nknots`; in that case, `nknots` equally spaced knots will be placed along `x` 
        to fit the data.

    2.  Passing an array to `x_knots`. In this case, knots will be placed at `x_knots`.

    3.  Passing a list of `nknots` and `x_knots`. In this case, each element of `x_knots` is assumed to be the lower and 
        upper limits of a region; the corresponding element of `nknots` will be used to put equally spaced knots in 
        that range.

    Parameters
    ----------

    x : numpy.array
        x-values for input data.
        
    y : numpy.array
        y-values for input data.

    nknots : int or list
        Number of knots to be used.

    x_knots : numpy.array or list
        Position of the knots or regions of knots (see description)

    Returns
    -------

    function : spline object
        Function over which the spline can be evaluated at.
    prediction : numpy.array
        Array of same dimensions of `x` and `y` with the spline evaluated at the input `x`.

    """

    xmin, xmax = np.min(x), np.max(x)

    if (nknots is not None) and (x_knots is not None): 

        knots = np.array([])
        for i in range( len(x_knots) ):

            knots = np.append( knots, np.linspace(x_knots[i][0], x_knots[i][1], nknots[i])  )

    elif x_knots is not None:

        knots = x_knots

    elif nknots is not None:

        idx = np.argsort(x)

        knots = np.linspace(x[idx][1], x[idx][-2], nknots)

    # Check knots are well-positioned:
    if np.min(knots) <= xmin:

        raise Exception('Lower knot cannot be equal or smaller than the smallest x input value.')

    if np.max(knots) >= xmax:

        raise Exception('Higher knot cannot be equal or larger than the largest x input value.')

    # Obtain spline representation:
    tck = splrep(x, y, t = knots)
    function = lambda x: splev(x, tck)

    # Return it:
    return function, function(x)

