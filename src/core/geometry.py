"""Pixel and aperture geometry shared by profile fitting and extraction.

Conventions: images are ``(row, column)`` with the dispersion direction along
columns, and a trace gives one cross-dispersion centre per column. Pixel ``i``
covers rows ``i - 0.5`` to ``i + 0.5``.

Two families of geometry live here:

- the *legacy Marsh* geometry (``marsh_geometry``, ``pixel_overlaps``,
  ``scale_edge_pixels``), which reproduces the original C implementation of
  Marsh (1989), including its aperture shrinking and column-range rules;
- the *native-pixel* geometry used by shared intrinsic profiles
  (``component_count``, ``native_window``, ``cropped_overlaps``), which keeps a
  fixed, whole-pixel support so a trained profile can be evaluated through any
  trace without changing shape.
"""

import numpy as np

# Legacy sentinel for masked pixels in the Marsh routines.
MASKED = -9999


def validate_image(data, centroids, variance=None):
    """Check an image, its trace and (optionally) its variance; return float arrays."""
    data = np.asarray(data, dtype=np.float64)
    centers = np.ascontiguousarray(centroids, dtype=np.float64)
    if data.ndim != 2 or min(data.shape) == 0:
        raise ValueError('data must be a nonempty (row, column) image')
    if centers.shape != (data.shape[1],) or not np.all(np.isfinite(centers)):
        raise ValueError('centroids must contain one finite value per column')
    if not np.all(np.isfinite(data)):
        raise ValueError('data must be finite; use -9999 for legacy masked pixels')
    if variance is not None:
        variance = np.asarray(variance, dtype=np.float64)
        if variance.shape != data.shape:
            raise ValueError('data_variance must match data.shape')
    return data, centers, variance


def mask_invalid_pixels(data, variance=None):
    """Replace missing science values (or invalid variances) by the -9999 sentinel.

    The input is not modified.
    """
    data = np.asarray(data, dtype=float)
    invalid = ~np.isfinite(data)
    if variance is not None:
        variance = np.asarray(variance, dtype=float)
        if variance.shape != data.shape:
            raise ValueError('data_variance must match data.shape')
        invalid |= ~np.isfinite(variance) | (variance <= 0)
    return np.where(invalid, float(MASKED), data)


def check_aperture(centers, nrows, spacing, aperture):
    """Shrink the aperture radius until the profile support fits on the detector.

    Reproduces the C routine ``CheckAperture``, which works with an integer radius.
    """
    if not np.isfinite(aperture) or aperture < 1:
        raise ValueError('aperture_radius must be finite and at least one pixel')
    if not np.isfinite(spacing) or spacing <= 0:
        raise ValueError('polynomial_spacing must be finite and positive')
    initial = radius = int(aperture)
    k = int(radius / spacing + 0.5)
    i = 0
    while i < len(centers):
        upper = int(centers[i] + spacing * (k + 1.0) + 0.5)
        lower = int(centers[i] - spacing * (k + 1.0) - 0.5)
        if upper >= nrows:
            radius -= 1
            k = int(radius / spacing + 0.5)
            lower = int(centers[i] - spacing * (k + 1.0) - 0.5)
            i -= 1
        if lower < 0 and upper >= nrows:
            radius -= 1
        if lower < 0 and upper < nrows:
            radius -= 1
            k = int(radius / spacing + 0.5)
            i -= 1
        if radius < 1:
            radius = initial
            break
        i += 1
    return radius


def report_aperture(requested, radius):
    """Print a message when the aperture was shrunk, as the C backend does."""
    if radius != int(requested):
        print(f'MARSH CODE: Changed central aperture to {radius}')


def marsh_geometry(
    centers, nrows, aperture, spacing, min_column=None, max_column=None, simple=False
):
    """Columns, aperture edges and pixel support of the legacy Marsh extraction.

    Returns
    -------
    columns : numpy.ndarray
        Detector columns that are extracted.
    low, high : numpy.ndarray
        Lower and upper aperture edges (in pixel-edge coordinates) for those columns.
    inside : numpy.ndarray
        Boolean ``(column, row)`` mask of pixels within the aperture.
    radius : int
        Aperture radius after ``check_aperture``.
    count : int
        Number of spatial components (polynomials) in the profile model.
    offset : float
        Offset of the first spatial component relative to the trace.
    """
    radius = check_aperture(centers, nrows, spacing, aperture)
    count = 2 * int(radius / spacing + 0.5) + 1
    offset = -(spacing * (1.0 + (count - 1.0) / 2.0))
    if simple:
        low, high = centers - radius + 0.5, centers + radius + 0.5
    else:
        low = (centers + offset) + spacing + 0.5
        high = (centers + offset) + count * spacing + 0.5
    ncol = len(centers)
    start, stop, found = -1, ncol, False
    # Reproduce the directional and end-point decisions of the C RangeDetector.
    upper_test = np.trunc(high).astype(int)
    lower_test = np.trunc(low).astype(int)
    if simple:
        upper_test = np.trunc(centers + radius) + 0.5
        lower_test = np.trunc(centers - radius) + 0.5
    if centers[0] > centers[ncol // 2] and np.any(upper_test[[0, -1]] >= nrows):
        start = 0
        for j, value in enumerate(np.trunc(high).astype(int)):
            if value < nrows and not found:
                start, found = j, True
            elif value == nrows and found:
                stop = j - 1
                break
    elif centers[0] < centers[ncol // 2] and np.any(lower_test[[0, -1]] < 0):
        start = 0
        for j, value in enumerate(np.trunc(low).astype(int)):
            if value == 0 and not found:
                start, found = j, True
            elif value < 0 and found:
                stop = j - 1
                break
    cmin = 0 if min_column is None else min_column
    cmax = ncol if max_column is None else max_column
    if int(cmin) != cmin or int(cmax) != cmax:
        raise ValueError('column bounds must be integers')
    cmin = int(cmin) + (cmin != 0)
    cmax = int(cmax) - (cmax != 0)
    if start == -1:
        start, stop = 0, ncol - 1
    start = max(start, cmin)
    if cmax != 0:
        stop = min(stop, cmax)
    if not 0 <= start <= stop < ncol:
        raise ValueError('legacy column bounds select an invalid detector range')
    columns = np.arange(start, stop + 1)
    low, high = low[columns], high[columns]
    lo, hi = np.trunc(low).astype(int), np.trunc(high).astype(int)
    if np.any(lo < 0) or np.any(hi >= nrows):
        raise ValueError('trace/aperture extends outside the detector')
    rows = np.arange(nrows)[None, :]
    inside = (rows >= lo[:, None]) & (rows <= hi[:, None])
    return columns, low, high, inside, radius, count, offset


def scale_edge_pixels(data, low, high, sentinel=True):
    """Weight the two partially covered edge pixels of each column by their coverage.

    ``data`` is ``(column, row)``. With ``sentinel=True`` masked (-9999) pixels are
    left untouched.
    """
    result = data.copy()
    for j, (lower, upper) in enumerate(zip(low, high)):
        lo, hi = int(lower), int(upper)
        if not sentinel or result[j, lo] != MASKED:
            result[j, lo] *= 1.0 - (lower - lo)
        if not sentinel or result[j, hi] != MASKED:
            result[j, hi] *= upper - hi
    return result


def pixel_overlaps(centers, rows, spacing, count, offset):
    """Integrated overlap ``Q`` of linear spatial components with detector pixels.

    This is the C ``CalculateQ`` of Marsh (1989): component ``k`` is a triangle of
    half-width ``spacing`` centred at ``center + offset + spacing*(k + 1)``.
    Returns an array of shape ``(column, row, count)``.
    """
    d = np.abs(
        centers[:, None, None]
        + spacing * np.arange(1, count + 1)[None, None, :]
        + offset
        - np.arange(rows)[None, :, None]
    )
    q = np.zeros_like(d)
    full = (d < 0.5 + spacing) & (d + spacing <= 0.5)
    inner = (d < 0.5 + spacing) & ~full & (d <= 0.5)
    outer = (d < 0.5 + spacing) & ~full & (d > 0.5)
    q[full] = spacing
    q[inner] = spacing / 2.0 + (0.5 - d[inner]) - (0.5 - d[inner]) ** 2 / (2.0 * spacing)
    q[outer] = (0.5 - d[outer]) + (0.5 - d[outer]) ** 2 / (2.0 * spacing) + spacing / 2.0
    return q


def component_count(radius, spacing):
    """Number of spatial components of a native-pixel (shared) profile.

    Whole boundary pixels can extend one pixel beyond the nominal radius; the
    basis covers them too, because truncating it there biases narrow-aperture
    fits as the trace moves across pixel boundaries.
    """
    return 2 * int(radius / spacing + 0.5) + 1 + 2 * int(np.ceil(1.0 / spacing))


def native_window(centers, nrows, radius, spacing):
    """Fixed whole-pixel row window around a trace.

    Returns
    -------
    rows : numpy.ndarray
        ``(column, window)`` detector row of each window pixel.
    inside : numpy.ndarray
        Boolean mask of window pixels within the aperture.
    count : int
        Number of spatial components (see ``component_count``).
    """
    centers = np.asarray(centers, dtype=float)
    if centers.ndim != 1 or not np.all(np.isfinite(centers)):
        raise ValueError('each trace must be a finite one-dimensional array')
    # The support is fixed across integrations: a trained model is never shrunk.
    count = component_count(radius, spacing)
    extent = spacing * int(radius / spacing + 0.5)
    low = np.trunc(centers - extent + 0.5).astype(int)
    high = np.trunc(centers + extent + 0.5).astype(int)
    if np.any(centers - extent + 0.5 < 0) or np.any(high >= nrows):
        raise ValueError(
            'shared profile aperture extends outside detector; select a smaller aperture'
        )
    rows = low[:, None] + np.arange(int(np.max(high - low)) + 1)
    inside = rows <= high[:, None]
    rows = np.minimum(rows, nrows - 1)
    return rows, inside, count


def cropped_overlaps(centers, rows, inside, spacing, count):
    """``pixel_overlaps`` evaluated only on a ``native_window`` (column, window, component)."""
    offset = -spacing * (1 + (count - 1) / 2)
    # Row windows are contiguous, so the overlap calculation runs on cropped
    # coordinates without allocating full-height images.
    return (
        pixel_overlaps(centers - rows[:, 0], rows.shape[1], spacing, count, offset)
        * inside[..., None]
    )


def aperture_weights(nrows, centers, radius):
    """Fractional overlap ``(row, column)`` of detector pixels with a box aperture.

    The aperture spans ``center - radius`` to ``center + radius`` in each column.
    """
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError('Aperture radius must be positive')
    rows = np.arange(nrows)[:, None]
    return np.maximum(
        0.0, np.minimum(rows + 0.5, centers + radius) - np.maximum(rows - 0.5, centers - radius)
    )
