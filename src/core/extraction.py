"""Spectral extraction: box apertures, Marsh optimal extraction and blended traces.

All functions work on ``(row, column)`` images with dispersion along columns and
return per-column fluxes. Optimal-extraction errors are conditional on the
supplied (or fitted) spatial profile: they do not include its uncertainty.
"""

import numpy as np

from . import geometry


def box_spectrum(
    data,
    x,
    y,
    aperture_radius,
    background_radius=50,
    error_data=None,
    correct_bkg=False,
    method='sum',
    bkg_method='all',
):
    """Sum (or average) the flux within ``aperture_radius`` of a trace, column by column.

    Fractional pixels at the aperture edges are weighted by their coverage. If
    ``correct_bkg`` is True, the median of the pixels between ``aperture_radius`` and
    ``background_radius`` is subtracted first; note this subtraction is done in place
    on ``data``.

    Parameters
    ----------
    data : numpy.ndarray
        Image from which the spectrum is extracted.
    x : numpy.ndarray
        Columns of the trace (the wavelength direction).
    y : numpy.ndarray
        Rows of the trace centre (the spatial direction).
    aperture_radius : float
        Distance from the trace centre within which fluxes are added.
    background_radius : float
        Outer radius of the background region.
    error_data : numpy.ndarray, optional
        Per-pixel errors of ``data``; if given, the flux errors are also returned.
    correct_bkg : bool
        Subtract a median background.
    method : {'sum', 'average'}
        ``'average'`` returns the mean of the whole pixels in the aperture (useful for
        wavelength maps) and disables background subtraction.
    bkg_method : {'all', 'up', 'down'}
        Use background pixels on both sides, above or below the trace.

    Returns
    -------
    flux : numpy.ndarray
    flux_error : numpy.ndarray
        Only if ``error_data`` is given.
    """
    method = method.lower()

    # If the average method is used, remove the background correction:
    if method == 'average':
        correct_bkg = False

    flux = np.zeros(len(x))
    if error_data is not None:
        flux_error = np.zeros(len(x))
    max_column = data.shape[0] - 1

    for i in range(len(x)):
        # Cut the column we work with:
        column = data[:, int(x[i])]
        if error_data is not None:
            variance_column = error_data[:, int(x[i])] ** 2

        # Limits given by the aperture_radius and background_radius:
        if correct_bkg:
            left_side_bkg = np.max([y[i] - background_radius, 0])
            right_side_bkg = np.min([max_column, y[i] + background_radius])
        left_side_ap = np.max([y[i] - aperture_radius, 0])
        right_side_ap = np.min([max_column, y[i] + aperture_radius])

        # Extract the background, being careful with edges:
        if correct_bkg:
            bkg_left = column[np.max([0, int(left_side_bkg)]) : np.max([0, int(left_side_ap)])]
            bkg_right = column[
                np.min([int(right_side_ap), max_column]) : np.max([int(right_side_bkg), max_column])
            ]

            if bkg_method == 'all':
                bkg = np.median(np.append(bkg_left, bkg_right))
            elif bkg_method == 'up':
                bkg = np.median(bkg_right)
            elif bkg_method == 'down':
                bkg = np.median(bkg_left)

        else:
            bkg = 0.0

        # Subtract it from the column:
        column -= bkg

        # Aperture extraction of the background-subtracted column, being careful with
        # pixelization at the edges. First, the left (up) side:
        l_decimal, l_integer = np.modf(left_side_ap)
        l_integer = int(l_integer)
        if l_decimal < 0.5:
            l_fraction = (0.5 - l_decimal) * column[np.min([l_integer, max_column])]
            l_limit = l_integer + 1
            if error_data is not None:
                l_fraction_variance = ((0.5 - l_decimal) ** 2) * variance_column[
                    np.min([l_integer, max_column])
                ]
        else:
            l_fraction = (1.0 - (l_decimal - 0.5)) * column[np.min([l_integer + 1, max_column])]
            l_limit = l_integer + 2
            if error_data is not None:
                l_fraction_variance = ((1.0 - (l_decimal - 0.5)) ** 2) * variance_column[
                    np.min([l_integer + 1, max_column])
                ]

        # Now the right (down) side:
        r_decimal, r_integer = np.modf(right_side_ap)
        r_integer = int(r_integer)
        if r_decimal < 0.5:
            r_fraction = (1.0 - (0.5 - r_decimal)) * column[np.min([max_column, r_integer])]
            r_limit = r_integer
            if error_data is not None:
                r_fraction_variance = ((1.0 - (0.5 - r_decimal)) ** 2) * variance_column[
                    np.min([max_column, r_integer])
                ]
        else:
            r_fraction = (r_decimal - 0.5) * column[np.min([max_column, r_integer + 1])]
            r_limit = r_integer + 1
            if error_data is not None:
                r_fraction_variance = ((r_decimal - 0.5) ** 2) * variance_column[
                    np.min([max_column, r_integer + 1])
                ]

        # Total flux in the current column:
        if method == 'sum':
            flux[i] = l_fraction + r_fraction + np.sum(column[l_limit:r_limit])
        elif method == 'average':
            flux[i] = np.nanmean(column[l_limit:r_limit])
        else:
            raise Exception(
                'Method "' + method + '" currently not supported for aperture extraction. '
                'Select either "sum" or "average".'
            )

        if error_data is not None:
            # The total error is the sum of the variances:
            flux_error[i] = np.sqrt(
                np.sum(variance_column[l_limit:r_limit]) + l_fraction_variance + r_fraction_variance
            )
    if error_data is not None:
        return flux, flux_error
    else:
        return flux


def fractional_box(data, variance, valid, columns, centers, radius):
    """Box-aperture flux and error with fractional edge pixels and explicit masks.

    Columns whose aperture is incomplete, off the detector or contains a masked pixel
    are returned as NaN rather than being filled or renormalized.

    Parameters
    ----------
    data, variance : numpy.ndarray
        ``(row, column)`` image and its per-pixel variance.
    valid : numpy.ndarray
        Boolean ``(row, column)`` mask of usable pixels.
    columns : numpy.ndarray
        Detector columns to extract.
    centers : numpy.ndarray
        Trace centre (row) at each of those columns.
    radius : float
        Aperture half-width in pixels.
    """
    weights = geometry.aperture_weights(data.shape[0], centers, radius)
    active = weights > 0
    good = (
        valid[:, columns]
        & np.isfinite(data[:, columns])
        & np.isfinite(variance[:, columns])
        & (variance[:, columns] > 0)
    )
    complete = np.all(~active | good, axis=0) & np.isclose(weights.sum(axis=0), 2 * radius)
    flux = np.sum(weights * np.where(good, data[:, columns], 0.0), axis=0)
    error = np.sqrt(np.sum(weights**2 * np.where(good, variance[:, columns], 0.0), axis=0))
    return np.where(complete, flux, np.nan), np.where(complete, error, np.nan)


def marsh_box(data, centers, aperture, min_column=None, max_column=None):
    """Fast box extraction with the legacy Marsh C conventions (``SimpleExtraction``).

    Returns the flux of every column and the aperture radius actually used.
    """
    data, centers, _ = geometry.validate_image(data, centers)
    columns, low, high, inside, radius, _, _ = geometry.marsh_geometry(
        centers, data.shape[0], aperture, 1.0, min_column, max_column, simple=True
    )
    geometry.report_aperture(aperture, radius)
    a = geometry.scale_edge_pixels(data[:, columns].T, low, high, sentinel=False)
    # SimpleExtraction loops while row < (floating) upper edge, unlike optimal extraction.
    inside &= np.arange(data.shape[0])[None, :] < high[:, None]
    output = np.zeros(data.shape[1])
    output[columns] = np.where(inside, a, 0.0).sum(axis=1)
    return output, float(radius)


def marsh_optimal(
    data,
    centers,
    aperture,
    ron,
    gain,
    nsigma,
    spacing,
    profile,
    min_column=None,
    max_column=None,
    variance=None,
    background=None,
    allow_missing=False,
):
    """Marsh (1989) optimal extraction with a given spatial profile.

    Outliers are rejected one pixel at a time (the worst first) until none exceeds
    ``nsigma``. Without ``variance``, pixel variances are modelled from ``ron`` and
    ``gain``.

    Returns
    -------
    numpy.ndarray
        ``(3, column)`` rows: column index, flux and **inverse** variance; a fourth row
        holds the extracted ``background`` if one is given.
    """
    data, centers, variance = geometry.validate_image(data, centers, variance)
    columns, low, high, inside, radius, _, _ = geometry.marsh_geometry(
        centers, data.shape[0], aperture, spacing, min_column, max_column
    )
    geometry.report_aperture(aperture, radius)
    a = geometry.scale_edge_pixels(data[:, columns].T, low, high)
    p = np.asarray(profile, dtype=float)[:, columns].T
    good = inside & (a != geometry.MASKED)
    flux = np.where(good, a, 0.0).sum(axis=1)
    supplied = None if variance is None else variance[:, columns].T
    for iteration in range(a.size + 1):
        v = (ron / gain) ** 2 + np.abs(flux[:, None] * p) / gain if supplied is None else supplied
        if np.any(v[good] <= 0) or not np.all(np.isfinite(v[good])):
            raise ValueError('extraction requires positive finite pixel variances')
        inverse = np.zeros_like(a)
        inverse[good] = 1.0 / v[good]
        denominator = (p * p * inverse).sum(axis=1)
        supported = denominator > 0
        if not allow_missing and np.any(~supported):
            raise ValueError('no weighted profile support in an extracted column')
        weights = np.divide(
            p * inverse, denominator[:, None], out=np.zeros_like(p), where=supported[:, None]
        )
        flux = (weights * np.where(good, a, 0.0)).sum(axis=1)
        varflux = (weights**2 * np.where(good, v, 0.0)).sum(axis=1)
        if allow_missing:
            good[~supported] = False
            flux[~supported], varflux[~supported] = np.nan, np.inf
        if nsigma == 0:
            break
        with np.errstate(invalid='ignore'):
            score = np.abs(a - flux[:, None] * p) - nsigma * (
                np.sqrt(np.maximum(v, 0.0)) + np.sqrt(varflux)[:, None] * p
            )
        score[~good] = -np.inf
        worst = np.argmax(score)  # Column-major scan, first tie, as in C.
        if score.flat[worst] <= 0:
            break
        good.flat[worst] = False
    output = np.zeros((3 if background is None else 4, data.shape[1]))
    output[0] = np.arange(data.shape[1])
    output[1, columns], output[2, columns] = flux, 1.0 / varflux
    if background is not None:
        b = geometry.scale_edge_pixels(
            np.asarray(background)[:, columns].T, low, high, sentinel=False
        )
        output[3, columns] = (weights * np.where(good & (b != geometry.MASKED), b, 0.0)).sum(axis=1)
    return output


def frozen_profile(
    profile, frame, variance, valid, aperture_fraction, nsigma=0.0, minimum_profile_fraction=0.5
):
    """Optimal extraction with a fixed native-pixel profile, on the box-aperture flux scale.

    Each pixel's full measured flux and variance enter the weighted fit. The fitted
    total is then multiplied by ``aperture_fraction`` (the profile-weighted fraction of
    light inside a reference box aperture), which matches a box extraction's flux
    convention without rescaling the pixel errors. Columns where less than
    ``minimum_profile_fraction`` of the profile was observed are returned as NaN rather
    than extrapolated from little remaining light.

    Parameters
    ----------
    profile : numpy.ndarray
        ``(row, column)`` normalized profile.
    frame, variance : numpy.ndarray
        Data and variance for the same ``(row, column)`` pixels.
    valid : numpy.ndarray
        Boolean mask of usable pixels.
    aperture_fraction : numpy.ndarray
        Per-column flux-scale factor.
    nsigma : float
        If positive, iteratively reject the worst pixel of each column beyond ``nsigma``.
    minimum_profile_fraction : float
        Minimum observed profile fraction for a column to be reported.

    Returns
    -------
    flux, error, observed_fraction : numpy.ndarray
    """
    y, v = frame, variance
    ncolumns = profile.shape[1]
    good = (
        valid & np.isfinite(y) & (y != geometry.MASKED) & np.isfinite(v) & (v > 0) & (profile > 0)
    )
    y = np.where(good, y, 0.0)
    for _ in range(int(np.max(np.sum(profile > 0, axis=0))) + 1):
        inverse = np.divide(1.0, v, out=np.zeros_like(v), where=good)
        denominator = np.sum(profile**2 * inverse, axis=0)
        flux = np.divide(
            np.sum(profile * y * inverse, axis=0),
            denominator,
            out=np.full(ncolumns, np.nan),
            where=denominator > 0,
        )
        if nsigma == 0:
            break
        score = np.where(good, np.abs(y - profile * flux) * np.sqrt(inverse), -np.inf)
        worst = np.argmax(score, axis=0)
        reject = score[worst, np.arange(ncolumns)] > nsigma
        if not np.any(reject):
            break
        good[worst[reject], np.flatnonzero(reject)] = False
    fraction = np.sum(np.where(good, profile, 0.0), axis=0)
    supported = (fraction >= minimum_profile_fraction) & (denominator > 0)
    flux = np.where(supported, flux * aperture_fraction, np.nan)
    error = np.full(ncolumns, np.nan)
    np.divide(aperture_fraction, np.sqrt(denominator), out=error, where=supported)
    return flux, error, fraction


def solve_blended(profiles, data, variance, valid, condition_limit=1e6):
    """Weighted least-squares fluxes of several overlapping traces, column by column.

    At each column ``D = sum_o F_o P_o + noise``. The normal equations are solved for
    all traces at once and the full conditional covariance is returned. Traces absent
    from a column get zero precision; ill-conditioned columns (judged on the
    diagonally normalized normal matrix) return no flux rather than a pseudo-inverse.

    Parameters
    ----------
    profiles : numpy.ndarray
        ``(column, row, trace)`` normalized profiles.
    data, variance : numpy.ndarray
        ``(column, row)`` data and variance.
    valid : numpy.ndarray
        Boolean ``(column, row)`` mask of usable pixels.
    condition_limit : float
        Maximum condition number of the normalized normal matrix.

    Returns
    -------
    flux : numpy.ndarray
        ``(column, trace)``.
    covariance : numpy.ndarray
        ``(column, trace, trace)``.
    supported : numpy.ndarray
        ``(column, trace)`` flags of fluxes that were measured.
    condition : numpy.ndarray
        ``(column,)`` condition numbers.
    inverse_variance : numpy.ndarray
        ``(column, row)`` pixel weights used.
    """
    good = (
        valid
        & np.isfinite(data)
        & (data != geometry.MASKED)
        & np.isfinite(variance)
        & (variance > 0)
    )
    good &= np.any(profiles > 0, axis=2)
    iv = np.divide(1.0, variance, out=np.zeros_like(variance, dtype=float), where=good)
    normal = np.einsum('jri,jrk,jr->jik', profiles, profiles, iv, optimize=True)
    rhs = np.einsum('jri,jr->ji', profiles, np.where(good, data, 0.0) * iv, optimize=True)
    diagonal = np.diagonal(normal, axis1=1, axis2=2)
    observed = diagonal > 0
    scale = np.sqrt(np.where(observed, diagonal, 1.0))
    correlation = normal / scale[:, :, None] / scale[:, None, :]
    o = np.arange(profiles.shape[2])
    correlation[:, o, o] = 1.0
    eigenvalues = np.linalg.eigvalsh(correlation)
    condition = np.divide(
        eigenvalues[:, -1],
        eigenvalues[:, 0],
        out=np.full(len(profiles), np.inf),
        where=eigenvalues[:, 0] > 0,
    )
    supported = (condition < condition_limit) & np.any(observed, axis=1)
    covariance = np.zeros_like(normal)
    covariance[supported] = (
        np.linalg.inv(correlation[supported])
        / scale[supported, :, None]
        / scale[supported, None, :]
    )
    covariance *= observed[:, :, None] & observed[:, None, :]
    flux = np.einsum('jik,jk->ji', covariance, rhs)
    return flux, covariance, observed & supported[:, None], condition, iv
