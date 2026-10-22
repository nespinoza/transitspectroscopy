"""Experimental 1/f correction of the group differences with a Gaussian process in readout time.

The pixels of a frame are read one after the other (10 us each, plus a new-line overhead of 12
pixel times), so a frame is one time series. The 1/f noise of every group difference (CDS
frame) is modelled as a Gaussian process in that time: white noise plus a sum of exponential
(Ornstein-Uhlenbeck; celerite "real") terms, whose amplitudes are fitted (non-negative least
squares) to the autocorrelation of the background residuals. Conditioned on the valid
background pixels of a difference, its predictive mean is the 1/f estimate at every pixel,
trace included, and is removed from the difference. ``gp_difference_1f`` has the signature of
``transitspectroscopy.core.detector.difference_1f``, which it replaces in
``gp_1f_comparison.py``; it is not part of the pipeline.
"""

import warnings

import numpy as np

LINE_OVERHEAD = 12


def readout_ticks(shape, fastaxis, slowaxis, overhead=LINE_OVERHEAD):
    """Readout time of every pixel of a ``(row, column)`` frame, in pixel times.

    Only fast reads along the rows (``|fastaxis| = 2``, NIRSpec and NIRISS) are supported; the
    sign of ``fastaxis`` (``slowaxis``) gives the direction of the reads within a column (of the
    columns).
    """
    if abs(fastaxis) != 2 or abs(slowaxis) != 1:
        raise ValueError('Only FASTAXIS = +-2 (reads along the rows) is supported')
    nrows, ncols = shape
    rows, columns = np.mgrid[:nrows, :ncols]
    if fastaxis < 0:
        rows = nrows - 1 - rows
    if slowaxis < 0:
        columns = ncols - 1 - columns
    return columns*(nrows + overhead) + rows


def masked_acf(images, masks, ticks, maximum_lag):
    """Autocorrelation (lags 0 to ``maximum_lag - 1`` pixel times) of masked frames, and the
    number of pixel pairs at every lag."""
    length = int(ticks.max()) + 1
    size = 1 << int(np.ceil(np.log2(length + maximum_lag)))
    products, pairs = np.zeros(maximum_lag), np.zeros(maximum_lag)
    for image, mask in zip(images, masks):
        series, weight = np.zeros(length), np.zeros(length)
        series[ticks[mask]] = image[mask]
        weight[ticks[mask]] = 1.
        products += np.fft.irfft(np.abs(np.fft.rfft(series, size))**2, size)[:maximum_lag]
        pairs += np.fft.irfft(np.abs(np.fft.rfft(weight, size))**2, size)[:maximum_lag]
    return products/np.maximum(np.round(pairs), 1), np.round(pairs)


def fit_kernel(images, masks, ticks, lines=200, nterms=12, frequencies=(0.,), periodic=()):
    """White variance and exponential terms fitted to the autocorrelation of ``images``.

    The timescales are log-spaced from one pixel time to ``lines`` columns; the terms are
    ``exp(-lag/timescale) cos(frequency lag)`` for every timescale and angular ``frequency``
    (radians per pixel time; ``0`` for the plain exponentials, ``pi`` for an alternating
    pattern of consecutive pixels), and their amplitudes are the non-negative least-squares
    fit of the autocorrelation at lags 1 or more (weighted by the square root of the number of
    pixel pairs), with the white variance a non-negative term at lag 0 only. Every angular
    frequency of ``periodic`` adds one term whose timescale is ten frames (a pattern constant
    over a frame; e.g., the harmonics of the line period for offsets of the rows that are
    common to all columns).
    """
    from scipy.optimize import nnls
    line = int(np.max(ticks[:, 1] - ticks[:, 0])) if ticks.shape[1] > 1 else ticks.shape[0]
    maximum_lag = min(abs(line)*lines, int(ticks.max())//2)
    acf, pairs = masked_acf(images, masks, ticks, maximum_lag)
    timescales = np.logspace(0, np.log10(maximum_lag), nterms)
    timescales, frequency = (np.tile(timescales, len(frequencies)),
                             np.repeat(np.asarray(frequencies, float), nterms))
    timescales = np.concatenate([timescales, np.full(len(periodic), 10.*ticks.max())])
    frequency = np.concatenate([frequency, np.asarray(periodic, float)])
    lags = np.unique(np.concatenate([np.arange(0, min(4*abs(line), maximum_lag)),
                                     np.logspace(0, np.log10(maximum_lag - 1), 400).astype(int)]))
    weight = np.sqrt(pairs[lags])
    design = np.exp(-lags[:, None]/timescales[None])*np.cos(frequency[None]*lags[:, None])
    design = np.column_stack([lags == 0, design])
    solution, _ = nnls(design*weight[:, None], acf[lags]*weight)
    white, amplitudes = solution[0], solution[1:]
    if white <= 0:
        raise ValueError('No white variance left by the fitted 1/f terms')
    keep = amplitudes > 0
    model = design @ solution
    return dict(white=float(white), amplitudes=amplitudes[keep], timescales=timescales[keep],
                frequencies=frequency[keep], variance=float(acf[0]),
                fit_rms=float(np.sqrt(np.average((model - acf[lags])**2, weights=pairs[lags]))))


def kernel_terms(kernel):
    from celerite2 import terms
    result = None
    frequencies = kernel.get('frequencies', np.zeros(len(kernel['amplitudes'])))
    for amplitude, timescale, frequency in zip(kernel['amplitudes'], kernel['timescales'],
                                               frequencies):
        if frequency:
            term = terms.ComplexTerm(a=float(amplitude), b=0., c=1./float(timescale),
                                     d=float(frequency))
        else:
            term = terms.RealTerm(a=float(amplitude), c=1./float(timescale))
        result = term if result is None else result + term
    return result


class TimePredictor:
    """Predictive mean, at every pixel of a frame, of the GP conditioned on some of them."""

    def __init__(self, kernel, ticks):
        import celerite2
        self.kernel, self.ticks = kernel, ticks
        self.terms = kernel_terms(kernel)
        self.order = np.argsort(ticks, axis=None)
        self.times = ticks.ravel()[self.order].astype(float)
        self.process = celerite2.GaussianProcess(self.terms)

    def __call__(self, image, use):
        flat_use = use.ravel()[self.order]
        if flat_use.sum() < 2:
            return np.zeros(image.shape)
        values = image.ravel()[self.order][flat_use]
        self.process.compute(self.times[flat_use], diag=np.full(flat_use.sum(),
                                                                self.kernel['white']))
        prediction = np.empty(image.size)
        prediction[self.order] = self.process.predict(values, t=self.times)
        return prediction.reshape(image.shape)


def clipped(residual, use, nsigma):
    """``use`` without the pixels (and their 4 neighbours) more than ``nsigma`` robust standard
    deviations away from the median of their column."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        level = np.nanmedian(np.where(use, residual, np.nan), axis=0)
    deviation = residual - np.where(np.isfinite(level), level, 0.)[None]
    sigma = 1.4826*np.median(np.abs(deviation[use]))
    bad = use & (np.abs(deviation) > nsigma*sigma)
    bad[1:] |= bad[:-1].copy()
    bad[:-1] |= bad[1:].copy()
    bad[:, 1:] |= bad[:, :-1].copy()
    bad[:, :-1] |= bad[:, 1:].copy()
    return use & ~bad


def holdout_band(background_pixels):
    """Background pixels hidden as a fake trace: below the median background row of every
    column for short subarrays (the band under the NIRSpec trace), else rows 200 to 215 of the
    columns where all of them are background (a SOSS region free of orders 1 to 3)."""
    nrows = background_pixels.shape[0]
    rows = np.arange(nrows)[:, None]
    if nrows <= 64:
        middle = np.array([np.median(np.flatnonzero(column)) if column.any() else -1
                           for column in background_pixels.T])
        return background_pixels & (rows < middle[None])
    band = background_pixels & (rows >= 200) & (rows < 216)
    return band & (band.sum(axis=0) == 16)[None]


class Holdout:
    """Errors at hidden background pixels of no correction, the column median of the remaining
    background, and the GP prediction from it: of the band mean of every column, and of the
    band sum over all columns (per pixel; what a white light is sensitive to)."""

    def __init__(self, background_pixels):
        self.band = holdout_band(background_pixels)
        self.squares = dict(none=0., median=0., gp=0., floor=0.)
        self.band_sums = dict(none=0., median=0., gp=0., floor=0.)
        self.count = self.frames = self.pixels = 0

    def add(self, residual, use, predictor, white):
        test = use & self.band
        train = use & ~self.band
        columns = (test.sum(axis=0) >= 3) & (train.sum(axis=0) >= 3)
        if not columns.any():
            return
        prediction = predictor(residual, train)[:, columns]
        test, train = test[:, columns], train[:, columns]
        sub = residual[:, columns]
        ntest = test.sum(axis=0)
        truth = np.sum(np.where(test, sub, 0.), axis=0)/ntest
        median = np.nanmedian(np.where(train, sub, np.nan), axis=0)
        gp = np.sum(np.where(test, prediction, 0.), axis=0)/ntest
        for name, estimate in (('none', 0.), ('median', median), ('gp', gp)):
            self.squares[name] += float(np.sum((truth - estimate)**2))
            # Error of the band sum over all columns (as a white light), per pixel:
            self.band_sums[name] += float(np.sum(ntest*(truth - estimate))/ntest.sum())**2
        self.squares['floor'] += float(np.sum(white/ntest))
        self.band_sums['floor'] += float(white/ntest.sum())
        self.count += int(columns.sum())
        self.frames += 1
        self.pixels += int(ntest.sum())

    def summary(self):
        if not self.count:
            return {}
        columns = {name: float(np.sqrt(total/self.count)) for name, total in self.squares.items()}
        sums = {f'band_sum_{name}': float(np.sqrt(total/self.frames))
                for name, total in self.band_sums.items()}
        return columns | sums | dict(column_differences=self.count, differences=self.frames,
                                     pixels_per_difference=self.pixels/self.frames)


def gp_difference_1f(segments, valid, background_pixels, scale, jumps=None, shared_fraction=0.5,
                     *, ticks, kernel_integrations=16, kernel_lines=200, nsigma=5.0,
                     holdout_every=20, holdout_only=False, record=None):
    """1/f of the group differences predicted at every pixel by a GP in readout time.

    The templates and the pixels used are those of
    `transitspectroscopy.core.detector.difference_1f` (template: median of the difference over
    the integrations, scaled by the white-light factor; valid background pixels, without the
    jumped ones unless more than ``shared_fraction`` of them are), and pixels more than
    ``nsigma`` robust standard deviations from their column median, and their neighbours, are
    also excluded. The kernel is fitted (`fit_kernel`) to the residuals of
    ``kernel_integrations`` integrations spread over the segments, and the predictive mean of
    every difference, plus the median of its used residuals, is removed from it (the
    cumulative sums from the groups after the first). Every ``holdout_every`` integrations,
    a band of background pixels is also hidden and predicted (`Holdout`).

    Parameters
    ----------
    segments, valid, background_pixels, scale, jumps, shared_fraction
        As in `transitspectroscopy.core.detector.difference_1f`.
    ticks : numpy.ndarray
        ``(row, column)`` readout times in pixel times (`readout_ticks`).
    kernel_integrations : int, optional
        Number of integrations whose residuals give the kernel.
    kernel_lines : int, optional
        Longest timescale of the kernel, in columns (`fit_kernel`).
    nsigma : float, optional
        Clipping threshold of the residuals.
    holdout_every : int or None, optional
        Integration step of the hold-out test; None for no test.
    holdout_only : bool, optional
        Only run the hold-out test (the ramps are not corrected).
    record : dict, optional
        Updated with the kernel, hold-out summary and timings.

    Returns
    -------
    dict
        ``levels``, the column means of the removed ``(integration, difference, column)`` 1/f,
        and ``shared`` as in `difference_1f`.
    """
    import time
    ngroups = segments[0].shape[1]
    if ngroups < 2:
        raise ValueError('The 1/f correction of group differences needs at least two groups')
    if len(scale) != sum(len(data) for data in segments):
        raise ValueError('One white-light factor per integration is needed')
    if jumps is None:
        jumps = [np.zeros(good.shape, bool) for good in valid]
    start = time.monotonic()

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        templates = []
        for k in range(ngroups - 1):
            values = [np.where(good[:, k] & good[:, k+1] & ~jump[:, k+1],
                               data[:, k+1] - data[:, k], np.nan).astype(np.float32)
                      for data, good, jump in zip(segments, valid, jumps)]
            templates.append(np.nanmedian(np.concatenate(values), axis=0))
            del values
    templates = np.nan_to_num(np.array(templates))

    nints = len(scale)
    locations = [(s, i) for s, data in enumerate(segments) for i in range(len(data))]

    def residuals(integration):
        s, i = locations[integration]
        data, good, jump = segments[s], valid[s], jumps[s]
        difference = np.diff(data[i].astype(float), axis=0)
        result = []
        for k in range(ngroups - 1):
            use = good[i, k] & good[i, k+1] & background_pixels
            unflagged = use & ~jump[i, k+1]
            shared = (use.sum() - unflagged.sum()) > shared_fraction*max(use.sum(), 1)
            if not shared:
                use = unflagged
            residual = difference[k] - scale[integration]*templates[k]
            use = use & np.isfinite(residual)
            use = clipped(residual, use, nsigma)
            offset = np.median(residual[use]) if use.any() else 0.
            result.append((residual - offset, use, offset, shared))
        return result

    sample = np.unique(np.linspace(0, nints - 1, kernel_integrations).astype(int))
    images, masks = [], []
    for integration in sample:
        for residual, use, _, _ in residuals(integration):
            images.append(np.where(use, residual, 0.))
            masks.append(use)
    kernel = fit_kernel(images, masks, ticks, lines=kernel_lines)
    del images, masks
    predictor = TimePredictor(kernel, ticks)
    holdout = Holdout(background_pixels)
    fitted = time.monotonic()

    levels = np.zeros((nints, ngroups - 1, segments[0].shape[3]))
    shared = np.zeros((nints, ngroups - 1), bool)
    for integration, (s, i) in enumerate(locations):
        if holdout_only and integration % holdout_every:
            continue
        data = segments[s]
        removed = np.zeros(data.shape[2:])
        for k, (residual, use, offset, flag) in enumerate(residuals(integration)):
            if holdout_every and integration % holdout_every == 0:
                holdout.add(residual, use, predictor, kernel['white'])
            if holdout_only:
                continue
            onef = predictor(residual, use) + offset
            levels[integration, k] = onef.mean(axis=0)
            shared[integration, k] = flag
            removed += onef
            data[i, k+1] -= removed.astype(data.dtype)

    if record is not None:
        record.update(kernel={key: np.asarray(value).tolist() for key, value in kernel.items()},
                      holdout=holdout.summary(), kernel_integrations=sample.tolist(),
                      seconds=dict(kernel=fitted - start, correction=time.monotonic() - fitted))
    return dict(levels=levels, shared=shared)
