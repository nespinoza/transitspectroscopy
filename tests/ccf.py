import time

import numpy as np
import matplotlib.pyplot as plt

import transitspectroscopy as ts

def gaussian(x, mean = 0., sigma = 1.):
    """
    This function returns a gaussian evaluated at x

    Parameters
    ----------

    x : numpy.array
        Array containing where the gaussian will be evaluated
    mean : double
        Mean of the gaussian.
    sigma : double
        Standard-deviation of the gaussian

    Returns
    -------
    Gaussian evaluated at `x`.

    """

    norm = 1. / ( np.sqrt(2. * np.pi) * sigma )

    return norm * np.exp( - ( (x - mean)**2 ) / (2 * sigma**2) )

def double_gaussian(x, mean1 = -7.9, mean2 = 7.9, sigma1 = 1., sigma2 = 1.):
    """
    This function returns the sum of two gaussians evaluated at x. This function reproduces the expected separation of 
    the "horns" in a NIRISS/SOSS profile.

    Parameters
    ----------

    x : numpy.array
        Array containing where the gaussian will be evaluated
    mean1 : double
        Mean of the first gaussian.
    mean2 : double  
        Mean of the second gaussian.
    sigma1 : double
        Standard-deviation of the first gaussian.
    sigma2 : double
        Standard-deviation of the second gaussian.

    Returns
    -------
    Double gaussian evaluated at `x`.
    """

    return gaussian(x, mean1, sigma1) + gaussian(x, mean2, sigma2)

def evaluate_lags(x, y, function = 'gaussian', lag_step = 0.001):

    lags = np.arange(np.min(x), np.max(x), lag_step)

    # Repeate x len(lags) times in a matrix:
    X = np.tile( x, (len(lags), 1) )

    # Substract lags to each row:
    X = (X.transpose() - lags).transpose()

    # Evaluate matrix:
    return gaussian(X)

def get_ccf(x, y, function = 'gaussian', pixelation = False, lag_step = 0.001):

    if not pixelation:

        if type(function) is str:

            if function == 'gaussian':

                f = gaussian

            elif function == 'double gaussian':

                f = double_gaussian

        else:

                f = function

    # Create array of lags:
    lags = np.arange(np.min(x), np.max(x), lag_step)
    ccf = np.zeros(len(lags))

    # Compute CCF for all lags:
    for i in range(len(lags)):

        ccf[i] = np.correlate( y, f(x - lags[i]) )[0]

    return lags, ccf

def get_sum_ccf(x, y, function = 'gaussian', pixelation = False, lag_step = 0.001):

    if not pixelation:

        if type(function) is str:

            if function == 'gaussian':

                f = gaussian

            elif function == 'double gaussian':

                f = double_gaussian

        else:

                f = function

    # Create array of lags:
    lags = np.arange(np.min(x), np.max(x), lag_step)
    ccf = np.zeros(len(lags))

    # Compute CCF for all lags:
    for i in range(len(lags)):

        ccf[i] = np.sum( y * f(x - lags[i])) #np.correlate( y, f(x - lags[i]) )[0]

    return lags, ccf

x = np.arange(0,20,1)
y = gaussian(x,mean=10) + np.random.normal(0., 1., len(x))

print('Python version:')
tic = time.time()
lags, ccf = get_ccf(x, y)
toc = time.time()
print(toc-tic)
plt.plot(lags, ccf/np.max(ccf), label = 'np.correlate')

print('Python version, but sum ccf:')
tic = time.time()
lags, ccf = get_sum_ccf(x, y)
toc = time.time()
print(toc-tic)
plt.plot(lags, ccf/np.max(ccf), label = 'np.sum')

print('C-version, sum ccf --- gaussian:')
tic = time.time()
lags, ccf = ts.spectroscopy.get_ccf(x, y)
toc = time.time()
print(toc-tic)

plt.plot(lags, ccf/np.max(ccf), label = 'C-sum')

print('C-version, sum ccf --- double gaussian:')
tic = time.time()
lags, ccf = ts.spectroscopy.get_ccf(x, y, function = 'double gaussian')
toc = time.time()
print(toc-tic)

print('C-version, sum ccf --- gaussian, but python evaluation:')
tic = time.time()
lags, ccf = ts.spectroscopy.get_ccf(x, y, function = gaussian)
toc = time.time()
print(toc-tic)
plt.plot(lags, ccf/np.max(ccf), label = 'C-sum, python eval')

plt.xlabel('Lags')
plt.ylabel('Normalized CCF')
plt.legend()
plt.show()
