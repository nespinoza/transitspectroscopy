import numpy as np

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
