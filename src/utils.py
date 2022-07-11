import numpy as np

import jdcal
from calendar import monthrange

from math import modf

from scipy.interpolate import splrep, splev

from astropy.time import Time
from astropy import units as q
from astropy.constants import G, k_B, u, au, M_jup, R_jup, R_sun, M_sun

def get_transpec_signal(Rp, Rstar, Mp, aR, Teff, mu = 2.3, albedo = 0., emissivity = 1.):
    """
    Given the planetary radius, mass, semi-major axis in stellar units and stellar effective temperature, 
    this function calculates the atmospheric signal produced by an atmosphere in ppm for one scale-height. 
    This has to be multiplied by a factor between 1-3 to estimate the real signal (see, e.g., 
    Wakeford+2019; https://ui.adsabs.harvard.edu/abs/2019RNAAS...3....7W/abstract).

    Parameters
    -----

    Rp : float
        Radius of the planet in Jupiter units.

    Mp : float
        Mass of the planet in Jupiter units.

    aR : float
        Semi-major axis in stellar units (a/R*)

    Teff : float
        Stellar effective temperature

    mu : float
        Mean molecular weight of the atmosphere.

    albedo : float
        (optional) Planetary albedo.
    
    emissivity : float
        (optional) Emmissivity of the planet.

    Returns
    ------

    Atmospheric signal in transmission for one-scale height in ppm.

    """    
    H = get_scaleheight(Rp, Mp, aR, Teff, mu, albedo, emissivity)

    atmospheric_signal = (2. * ( Rp * R_jup ) * ( H ) / ( Rstar * R_sun )**2 )*1e6

    return atmospheric_signal.value

def get_scaleheight(Rp, Mp, aR, Teff, mu = 2.3, albedo = 0., emissivity = 1.):
    """
    Given the planetary radius, mass, semi-major axis in stellar units and stellar effective temperature, 
    this function calculates the atmospheric scale-height of an atmosphere:

    Parameters
    -----

    Rp : float
        Radius of the planet in Jupiter units.

    Mp : float
        Mass of the planet in Jupiter units.

    aR : float
        Semi-major axis in stellar units (a/R*)

    Teff : float
        Stellar effective temperature

    mu : float
        Mean molecular weight of the atmosphere.

    albedo : float
        (optional) Planetary albedo.
    
    emissivity : float
        (optional) Emmissivity of the planet.

    Returns
    ------

    H : astropy.unit
        Planetary scale-height in km.

    """

    # Planetary gravity:
    g = G * ( Mp * M_jup ) / ( Rp * R_jup )**2

    # Calculate equilibrium temperature of planet (assuming zero-albedo)
    Teq = (Teff * q.K) * ( (1. - albedo) / emissivity )**(1./4.) * np.sqrt( 0.5 / aR )

    # Get scale-height:
    H = k_B * Teq / ( mu * u * g )

    return H.to(q.km)


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

def get_mad_sigma(x, median):
    """
    This function returns the MAD-based standard-deviation.
    """

    mad = np.nanmedian( np.abs ( x - median ) )

    return 1.4826*mad

def RA_to_deg(coords):
    """
    Given a RA string in hours (e.g., '11:12:12.11'), returns the corresponding 
    coordinate in degrees.
    """

    hh,mm,ss = coords.split(':')

    hours = np.double(hh) + np.double(mm)/60. + np.double(ss)/3600.

    return hours * 360./24.

def DEC_to_deg(coords):
    """
    Given a DEC string in degrees (e.g., '-30:12:12.11'), returns the corresponding 
    coordinate in degrees.
    """

    dd,mm,ss = coords.split(':')

    if dd[0] == '-':

        return np.double(dd) - np.double(mm)/60. - np.double(ss)/3600.

    else:

        return np.double(dd) + np.double(mm)/60. + np.double(ss)/3600.

def mag_to_flux(m, merr, nsim = 1000):
    """
    Convert magnitude to relative fluxes via Monte Carlo sampling 

    Parameters
    ----------

    m : list or np.array
        List or array of magnitudes

    merr : list or np.array
        List or array of errors on the magnitudes (same dimension as `m`).

    Returns
    -------

    fluxes : np.array
           Relative fluxes 

    fluxes_err : np.array
           Error on relative fluxes.Radius of the planet in Jupiter units.
    """

    fluxes = np.zeros(len(m))

    fluxes_err = np.zeros(len(m))

    for i in range(len(m)):

        dist = 10**(- np.random.normal( m[i], merr[i], nsim) / 2.51)

        fluxes[i] = np.mean( dist )

        fluxes_err[i] = np.sqrt( np.var(dist) )

    return fluxes,fluxes_err

def getCalDay(JD):

    year, month, day, hour= jdcal.jd2gcal(JD,0.0)
    hour = hour*24
    minutes = modf(hour)[0]*60.0
    seconds = modf(minutes)[0]*60.0

    hh = int(modf(hour)[1])
    mm = int(modf(minutes)[1])
    ss = seconds

    if(hh<10):

       hh = '0'+str(hh)

    else:

       hh = str(hh)

    if(mm<10):

       mm = '0'+str(mm)

    else:

       mm = str(mm)

    if(ss<10):

       ss = '0'+str(np.round(ss,1))

    else:

       ss = str(np.round(ss,1))

    return year,month,day,hh,mm,ss

def transit_predictor(year, month, P, t0, tduration, day=None):
    """
    This function predicts transits of a planet given input period `P`, duration `tduration` and time-of-transit 
    center `t0` on a given month of a year. The script can also receive a day to predict transits of an 
    exoplanet on a given day/month/year. Months run from 1 (January) to 12 (December). 

    Parameters
    ----------

    year : float
        Year on which you want to predict transits.
    
    month : float
        Month on which you want to predict transits.

    P : float
        Period of the exoplanet in days.
    
    t0 : float
        Time-of-transit center of the exoplanet in JD.

    tduration : float
        Transit duration in hours.

    day : float (optional)
        Day on which transits want to be predicted.

    """
   
    # If no input day, get all days in the month:
    if day is None:

        first_w_day,max_d = monthrange(year, month)
        days = range(1,max_d+1)

    else:
   
        days = [day]

    transits_t0 = np.array([]) 

    for cday in days:

        # Check closest transit to given day:

        t = Time(str(int(year))+'-'+str(int(month))+'-'+str(int(cday))+' 00:00:00', \
                 format='iso', scale='utc')

        ntimes = int(np.ceil(1./P))

        for n in range(ntimes):

            c_t0 = float(t.jd) - P * get_phases(float(t.jd), P, t0) + n * P

            # Check if mid-transit, egress or ingress happens whithin the 
            # desired day. If it does, and we have not saved it, save the 
            # JD of the transit event:

            tyear,tmonth,tday,thh,tmm,tss = getCalDay(c_t0) 

            if tday == cday and tmonth == month and tyear == year:

                if c_t0 not in transits_t0:

                    transits_t0 = np.append(transits_t0,c_t0)

            else:

                tyear, tmonth, tday, thh, tmm, tss = getCalDay( c_t0 + (tduration / 2.) )

                if tday == cday and tmonth == month and tyear == year:

                    if c_t0 not in transits_t0:

                        transits_t0 = np.append(transits_t0, c_t0)

                else:

                    tyear, tmonth, tday, thh, tmm, tss = getCalDay( c_t0 - (tduration / 2.) )

                    if tday == cday and tmonth == month and tyear == year:

                        if c_t0 not in transits_t0:

                            transits_t0 = np.append(transits_t0,c_t0)

    # Now print the transits we found:
    counter = 0
    if len(transits_t0)>0:

        print('Transits for this exoplanet')
        print('--------------------------\n')

    else:

        print('No transits found for this planet in the input period')

    for ct0 in transits_t0:

        print('\t Transit number '+str(counter+1)+':')
        print('\t ----------------------------')

        tyear, tmonth, tday, thh, tmm, tss = getCalDay( ct0 - (tduration / (2. * 24.)) ) 

        print('\t Ingress     : '+str(tyear)+'-'+str(tmonth)+'-'+str(tday)+' at '+str(thh)+\
              ':'+str(tmm)+':'+str(tss)+' ('+str( ct0 - (tduration / (2. * 24.) ) )+' JD)')

        tyear,tmonth,tday,thh,tmm,tss = getCalDay(ct0)

        print('\t Mid-transit : '+str(tyear)+'-'+str(tmonth)+'-'+str(tday)+' at '+str(thh)+\
              ':'+str(tmm)+':'+str(tss)+' ('+str(ct0)+' JD)')

        tyear,tmonth,tday,thh,tmm,tss = getCalDay( ct0 + ( tduration/(2. * 24.) ) )

        print('\t Egress      : '+str(tyear)+'-'+str(tmonth)+'-'+str(tday)+' at '+str(thh)+\
              ':'+str(tmm)+':'+str(tss)+' ('+str( ct0 + (tduration / (2. * 24.) ) )+' JD)')

        counter = counter + 1

def bin_at_resolution(wavelengths, depths, R = 100, method = 'median'):
    """
    Function that bins input wavelengths and transit depths (or any other observable, like flux) to a given 
    resolution `R`. Useful for binning transit depths down to a target resolution on a transit spectrum.

    Parameters
    ----------

    wavelengths : np.array
        Array of wavelengths
    
    depths : np.array
        Array of depths at each wavelength.

    R : int
        Target resolution at which to bin (default is 100)

    method : string
        'mean' will calculate resolution via the mean --- 'median' via the median resolution of all points 
        in a bin.

    Returns
    -------

    wout : np.array
        Wavelength of the given bin at resolution R.

    dout : np.array
        Depth of the bin.

    derrout : np.array
        Error on depth of the bin.
    

    """

    # Sort wavelengths from lowest to highest:
    idx = np.argsort(wavelengths)

    ww = wavelengths[idx]
    dd = depths[idx]

    # Prepare output arrays:
    wout, dout, derrout = np.array([]), np.array([]), np.array([])

    oncall = False

    # Loop over all (ordered) wavelengths:
    for i in range(len(ww)):

        if not oncall:

            # If we are in a given bin, initialize it:
            current_wavs = np.array([ww[i]])
            current_depths = np.array(dd[i])
            oncall = True

        else:

            # On a given bin, append next wavelength/depth:
            current_wavs = np.append(current_wavs, ww[i])
            current_depths = np.append(current_depths, dd[i])

            # Calculate current mean R:
            current_R = np.mean(current_wavs) / np.abs(current_wavs[0] - current_wavs[-1])

            # If the current set of wavs/depths is below or at the target resolution, stop and move to next bin:
            if current_R <= R:

                wout = np.append(wout, np.mean(current_wavs))
                dout = np.append(dout, np.mean(current_depths))
                derrout = np.append(derrout, np.sqrt(np.var(current_depths)) / np.sqrt(len(current_depths)))

                oncall = False

    return wout, dout, derrout

def vacuum_to_air(wavelength):
    """
    Given wavelengths (in microns) in vacuum, convert them to 
    air wavelengths following Morton (2000, ApJ, Suppl, 130, 403) 
    as explained: https://www.astro.uu.se/valdwiki/Air-to-vacuum%20conversion#:~:text=The%20conversion%20is%20then%3A%20%CE%BB,0.045%25%20CO2%20by%20volume.
    """

    # Convert input in microns to angstroms:
    w_ang = wavelength * 1e4

    # Generate the s variable:
    s = 1e4 / w_ang

    # Compute refraction index:
    n = 1 + 0.0000834254 + 0.02406147 / (130. - s**2) + 0.00015998 / (38.9 - s**2)

    # Return converted wavelengths in microns:
    return ( w_ang / n ) * 1e-4

def air_to_vacuum(wavelength):
    """
    Given wavelengths (in microns) in air, convert them to vacuum following the 
    explaination in https://www.astro.uu.se/valdwiki/Air-to-vacuum%20conversion#:~:text=The%20conversion%20is%20then%3A%20%CE%BB,0.045%25%20CO2%20by%20volume..
    This is the not-as-trivial reversion of the vacuum_to_air function.
    """

    # Convert input in microns to angstroms:
    w_ang = wavelength * 1e4 

    # Generate the s variable:
    s = 1e4 / w_ang

    # Compute refraction index:
    n = 1 + 0.00008336624212083 + 0.02408926869968 / (130.1065924522 - s**2) + 0.0001599740894897 / (38.92568793293 - s**2)

    # Return converted wavelengths in microns:
    return ( w_ang * n ) * 1e-4 
