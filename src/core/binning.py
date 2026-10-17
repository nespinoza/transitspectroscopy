"""Binning of spectra (e.g. transit depths) at a target resolving power."""

import numpy as np


def bin_at_resolution(wavelengths, depths, depths_errors=None, R=100, method='mean'):
    """
    Function that bins input wavelengths and transit depths (or any other observable, like flux) to
    a given
    resolution `R`. Useful for binning transit depths down to a target resolution on a transit
    spectrum.

    Parameters
    ----------

    wavelengths : np.array
        Array of wavelengths

    depths : np.array
        Array of depths at each wavelength.

    R : int
        Target resolution at which to bin (default is 100)

    method : string
        'mean' will calculate resolution via the mean --- 'median' via the median resolution of all
        points
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

    if depths_errors is not None:
        dderr = depths_errors[idx]

    # Prepare output arrays:
    wout, dout, derrout = np.array([]), np.array([]), np.array([])

    oncall = False

    # Loop over all (ordered) wavelengths:
    for i in range(len(ww)):
        if not oncall:
            # If we are in a given bin, initialize it:
            current_wavs = np.array([ww[i]])
            current_depths = np.array(dd[i])

            if depths_errors is not None:
                current_depth_errors = np.array(dderr[i])

            oncall = True

        else:
            # On a given bin, append next wavelength/depth:
            current_wavs = np.append(current_wavs, ww[i])
            current_depths = np.append(current_depths, dd[i])

            if depths_errors is not None:
                current_depth_errors = np.append(current_depth_errors, dderr[i])

            # Calculate current mean R:
            if method == 'mean':
                representative_wavelength = np.mean(current_wavs)

            elif method == 'median':
                representative_wavelength = np.median(current_wavs)

            else:
                raise Exception('Method ' + method + ' not supported. Try "mean" or "median".')

            delta_wavelength = np.abs(current_wavs[-1] - current_wavs[0])

            delta_wavelength += 0.5 * np.abs(current_wavs[1] - current_wavs[0])

            delta_wavelength += 0.5 * np.abs(current_wavs[-1] - current_wavs[-2])

            current_R = representative_wavelength / delta_wavelength

            # If the current set of wavs/depths is below or at the target resolution, stop, save
            # and move to next bin:
            if current_R <= R:
                wout = np.append(wout, np.mean(current_wavs))

                if depths_errors is None:
                    dout = np.append(dout, np.mean(current_depths))
                    derrout = np.append(
                        derrout, np.sqrt(np.var(current_depths)) / np.sqrt(len(current_depths))
                    )

                else:
                    weights = 1.0 / current_depth_errors**2

                    dout = np.append(dout, np.sum(weights * current_depths) / np.sum(weights))

                    derrout = np.append(derrout, 1.0 / np.sqrt(np.sum(weights)))

                oncall = False

    return wout, dout, derrout
