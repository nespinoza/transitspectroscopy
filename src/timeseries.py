import numpy as np

from scipy.ndimage import median_filter

from .utils import *

def outlier_detector(x, nsigma = 10, window = 200):
    """
    Given a uniformly sampled time-series `x`, this function substracts a median filter to it with a window `window`, 
    estimates the standard deviation of the residuals via an outlier-robust metric (the median-absolute-deviation-based 
    standard deviation) and identifies outliers in it, returning the indexes at which they are located


    Parameters
    ----------

    x : numpy.array
        Numpy array of a uniformly-spaced time-series.

    nsigma : float
        Number of sigmas used to identify outliers.

    window : int
        Window for the median filter

    Returns
    -------

    outlier_indexes : numpy.array
        Indexes in `x` at which outliers are located.

    """

    # First, get a median filter to get rid of the transit (or any astrophysical) event:
    mf = median_filter(x, window)
    residuals = (x - mf)
    
    # Now, with those zero-median residuals, estimate the median absolute deviation:
    sigma = get_mad_sigma( residuals, np.nanmedian(residuals) )
    
    # Now, find any datapoint 10-sigma away:
    idx = np.where( np.abs(residuals) > nsigma * sigma )[0]
    
    # Return location of the outliers:
    return idx
