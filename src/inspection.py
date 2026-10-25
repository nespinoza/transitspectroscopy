"""
Interactive inspection of ramps: the detector image of one integration and group, against
statistics of those images over the integrations or over the groups of the ramps.

The viewer is instrument independent: it takes the ramps (integration x group x row x column),
optionally with their group DQ, rates per integration and light curves, e.g.::

    >>> viewer = ts.utils.inspect_ramps(ramps, groupdq, rateints=rateints,
    ...                                 curves={'white light residuals (ppm)': residuals})

or, for a dataset calibrated with `transitspectroscopy.jwst.load.detector_calibration`, the
ramps of any calibration stage::

    >>> viewer = dataset.inspect_ramps(stage='linearity', segments=0)

Keep the returned viewer referenced: its sliders and buttons stop responding when it is
garbage collected. matplotlib is needed (``pip install transitspectroscopy[plots]``); in a
notebook, use an interactive backend (e.g., ``%matplotlib widget``).
"""
import warnings

import numpy as np

# JUMP_DET bit of the JWST group DQ:
JUMP_DET = 4

COUNT_MODES = ('counts', 'counts - median', 'counts - median (sigma)', 'increment',
               'increment - median', 'increment - median (sigma)')
RATE_MODES = ('rate / median - 1', 'rate ratio / median - 1')
AXES = ('integrations', 'groups')
SCALES = (3, 5, 10, 30)


def _quiet(function, *args, **kwargs):
    with warnings.catch_warnings(), np.errstate(all='ignore'):
        warnings.simplefilter('ignore', RuntimeWarning)
        return function(*args, **kwargs)


def _median(values):
    """Median over the first axis, ignoring NaNs (numpy.median, much faster, if none)."""
    if np.isnan(values).any():
        return _quiet(np.nanmedian, values, axis=0)
    return np.median(values, axis=0)


def _robust(values):
    """Median and robust (MAD-based) standard deviation of the finite values."""
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.nan, np.nan
    median = np.median(finite)
    return median, 1.4826*np.median(np.abs(finite - median))


def estimate_trace(signal, half_width=4, order=3):
    """
    Trace centre of every column of a dispersed image, for the aperture of `RampViewer`.

    The centroid of the rows within ``half_width`` of the brightest row of every column,
    smoothed by a polynomial of the column (weighted by the column flux; columns with less
    than 10% of the 95th percentile of the flux, and centroids more than 1 row off the first
    fit, are left out of the fit). For traces that run along the columns; for other
    geometries, give the trace to `RampViewer`.

    Parameters
    ----------

    signal : numpy.ndarray
        Image (row x column), e.g. the median counts of the last group minus those of the
        first group.

    half_width : int
        (Optional) rows on each side of the brightest row used for the centroid. Default 4.

    order : int
        (Optional) order of the polynomial. Default 3.

    Returns
    -------

    numpy.ndarray
        Trace centre (row) of every column.
    """

    nrows, ncols = signal.shape
    rows = np.arange(nrows)[:, None]
    finite = np.isfinite(signal)
    peak = np.argmax(np.where(finite, signal, -np.inf), axis=0)
    weights = np.where((np.abs(rows - peak[None]) <= half_width) & finite,
                       np.clip(signal, 0, None), 0.)
    flux = weights.sum(0)
    centroid = (weights*rows).sum(0)/np.maximum(flux, 1e-30)
    columns = np.arange(ncols)
    use = flux > 0.1*np.percentile(flux, 95)
    if use.sum() <= order + 2:
        return centroid
    for _ in range(2):
        coefficients = np.polyfit(columns[use], centroid[use], order, w=np.sqrt(flux[use]))
        trace = np.polyval(coefficients, columns)
        use &= np.abs(centroid - trace) < 1
        if use.sum() <= order + 2:
            break
    return trace


class RampViewer:
    """
    Interactive viewer of ramps. The top panel shows the image of one integration and group
    in one of several modes; the middle panel, the mean of that image within the aperture
    (around the trace) and its median in the background (far from it), over the integrations (at the
    group shown) or over the groups (of the integration shown), with the number of pixels of
    the aperture with a new jump; the bottom panel (if ``curves`` are given), user curves
    against the integration, e.g. light curves or their residuals.

    Image modes (``mode``; radio buttons):

    ``'counts'``
        The counts of the group (DN).
    ``'counts - median'``
        Those, minus their median over the integrations, for the same group (DN): changes
        from integration to integration, e.g. offsets, persistence or moving features.
    ``'counts - median (sigma)'``
        That, in units of its noise: the robust standard deviation over the integrations of
        every pixel and group, or, with ``read_noise`` and ``gain``,
        ``sqrt(read_noise**2 + counts/gain)``; bright and faint pixels on one scale.
    ``'increment'``
        The group minus the previous group (DN; the counts for group 0).
    ``'increment - median'``, ``'increment - median (sigma)'``
        Like the counts, for the increments: a jump, or a frame-wide change, shows in the
        group where it happens.
    ``'rate / median - 1'``
        With ``rateints``: the rate of the integration over the median rate (%).
    ``'rate ratio / median - 1'``
        With ``rateints`` and ``comparison_rateints``: their ratio, relative to its median
        over the integrations (%), e.g. to compare two calibrations of the same data.

    Overlays: the aperture (dashed lines), and, with ``groupdq``, the pixels flagged
    JUMP_DET in the group shown (red) and in an earlier group of the integration (orange).

    Controls: integration and group sliders, and keys left/right (integration), up/down
    (group), ``'a'`` (swap the middle panel between integrations and groups) and ``'c'``
    (cycle the colour range: 3, 5, 10 or 30 robust standard deviations of the aperture).

    Parameters
    ----------

    ramps : array_like
        Ramps (integration x group x row x column), e.g. a numpy array, a memory map or the
        ``data`` of a JWST ``RampModel``. Read one group or integration at a time.

    groupdq : array_like
        (Optional) group DQ of the ramps (same shape), for the jumps. Default is None.

    rateints : array_like
        (Optional) rates per integration (integration x row x column). Default is None.

    comparison_rateints : array_like
        (Optional) other rates per integration of the same ramps, for the
        ``'rate ratio / median - 1'`` mode (``rateints / comparison_rateints``). Default is
        None.

    curves : dict
        (Optional) curves for the bottom panel: label -> one value per integration (e.g.
        transit-fit residuals); plotted as given. Default is None (no bottom panel).

    trace : numpy.ndarray or bool
        (Optional) trace centre (row) of every column, for the aperture; None (default) to
        estimate it (see `estimate_trace`) from the median counts of the last group minus
        those of the first group; False for no trace (the aperture is then the whole image,
        and there is no background).

    aperture_radius : float
        (Optional) half width of the aperture, in rows. Default is 2.5.

    background_distance : float
        (Optional) rows farther than this from the trace are the background. Default is None
        (three times ``aperture_radius``).

    read_noise, gain : float
        (Optional) read noise (DN) and gain (e/DN) for the noise of the sigma modes, for
        ramps with bias and dark subtracted. Default is None (both): the noise is estimated
        from the scatter over the integrations.

    reference_integrations : int
        (Optional) maximum number of integrations (evenly spaced) for the medians and the
        scatter over the integrations. Default is 100.

    integration, group : int
        (Optional) integration and group shown first. Defaults are 0 and -1 (the last group).

    mode : str
        (Optional) image mode shown first. Default is ``'counts - median (sigma)'``.

    axis : str
        (Optional) axis of the middle panel shown first: ``'integrations'`` (default) or
        ``'groups'``.

    first_integration : int
        (Optional) number of the first integration of ``ramps`` shown on the sliders and the
        axes, e.g. for the ramps of a segment. Default is 0.

    title : str
        (Optional) title of the figure. Default is None.

    Attributes
    ----------

    figure : matplotlib.figure.Figure
        The figure.

    trace, aperture, background : numpy.ndarray
        The trace centre of every column, and the aperture and background pixel masks.
    """

    def __init__(self, ramps, groupdq = None, *, rateints = None, comparison_rateints = None,
                 curves = None, trace = None, aperture_radius = 2.5, background_distance = None,
                 read_noise = None, gain = None, reference_integrations = 100,
                 integration = 0, group = -1, mode = 'counts - median (sigma)',
                 axis = 'integrations', first_integration = 0, title = None):

        try:
            import matplotlib.pyplot as plt
            from matplotlib.widgets import CheckButtons, RadioButtons, Slider
            from matplotlib.ticker import MaxNLocator
        except ImportError as exc:
            raise ImportError('The ramp viewer requires matplotlib; install '
                              'transitspectroscopy[plots].') from exc

        if len(np.shape(ramps)) != 4:
            raise ValueError('ramps must be integration x group x row x column')
        self.ramps, self.groupdq = ramps, groupdq
        self.nints, self.ngroups, self.nrows, self.ncols = np.shape(ramps)
        if groupdq is not None and np.shape(groupdq) != np.shape(ramps):
            raise ValueError('groupdq must have the shape of the ramps')
        self.rateints = None if rateints is None else np.asarray(rateints, np.float32)
        self.modes = COUNT_MODES
        if self.rateints is not None:
            if self.rateints.shape != (self.nints, self.nrows, self.ncols):
                raise ValueError('rateints must be integration x row x column, as the ramps')
            self.median_rate = _quiet(np.nanmedian, self.rateints, axis=0)
            self.modes = self.modes + RATE_MODES[:1]
            if comparison_rateints is not None:
                comparison = np.asarray(comparison_rateints, np.float32)
                if comparison.shape != self.rateints.shape:
                    raise ValueError('comparison_rateints must have the shape of rateints')
                ratio = _quiet(np.divide, self.rateints, comparison)
                self.rate_ratio = _quiet(np.divide, ratio, _quiet(np.nanmedian, ratio, axis=0))
                self.modes = self.modes + RATE_MODES[1:]
        elif comparison_rateints is not None:
            raise ValueError('comparison_rateints requires rateints')
        if mode not in self.modes:
            raise ValueError(f'mode must be one of {self.modes}')
        if axis not in AXES:
            raise ValueError(f'axis must be one of {AXES}')
        self.curves = dict(curves or {})
        for label, values in self.curves.items():
            if len(values) != self.nints:
                raise ValueError(f'curve {label!r} must have one value per integration')
        if (read_noise is None) != (gain is None):
            raise ValueError('give both read_noise and gain, or neither')
        self.read_noise, self.gain = read_noise, gain
        count = min(self.nints, int(reference_integrations))
        self.reference = np.unique(np.linspace(0, self.nints - 1, count).round().astype(int))
        self.first_integration = int(first_integration)
        self._statistics_cache, self._profile_cache = {}, {}
        self.state = dict(mode=mode, axis=axis, scale=1, jumps=groupdq is not None)

        # Aperture and background:
        rows = np.arange(self.nrows)[:, None]
        if trace is None:
            signal = (self._statistics(self.ngroups - 1)['counts'][0]
                      - self._statistics(0)['counts'][0])
            trace = estimate_trace(signal)
        if trace is False:
            self.trace = None
            self.aperture = np.ones((self.nrows, self.ncols), bool)
            self.background = np.zeros((self.nrows, self.ncols), bool)
        else:
            self.trace = np.asarray(trace, float)
            if self.trace.shape != (self.ncols,):
                raise ValueError('trace must have one value per column')
            distance = np.abs(rows - self.trace[None])
            if background_distance is None:
                background_distance = 3*aperture_radius
            self.aperture = distance <= aperture_radius
            self.background = distance > background_distance
        self.aperture_radius = aperture_radius
        self.new_jumps = None if groupdq is None else self._new_jumps()

        # Figure:
        self.figure = figure = plt.figure(figsize=(16, 10 if self.curves else 9))
        top = 0.94
        if self.curves:
            image_box, profile_box = [0.06, 0.62, 0.74, 0.30], [0.06, 0.37, 0.74, 0.19]
            curve_box = [0.06, 0.16, 0.74, 0.16]
        else:
            image_box, profile_box = [0.06, 0.53, 0.74, 0.39], [0.06, 0.18, 0.74, 0.28]
        self.ax_image = figure.add_axes(image_box)
        self.ax_bar = figure.add_axes([0.81, image_box[1], 0.01, image_box[3]])
        self.ax_profile = figure.add_axes(profile_box)
        self.ax_profile.xaxis.set_major_locator(MaxNLocator(integer=True))
        self.ax_jumps = self.ax_profile.twinx()
        if title is not None:
            figure.suptitle(title, y=top + 0.045, fontsize=11)

        self.shown = self.ax_image.imshow(np.zeros((self.nrows, self.ncols)), origin='lower',
                                          aspect='auto', interpolation='nearest')
        self.bar = figure.colorbar(self.shown, cax=self.ax_bar)
        if self.trace is not None:
            columns = np.arange(self.ncols)
            for sign in (-1, 1):
                self.ax_image.plot(columns, self.trace + sign*aperture_radius, 'w--', lw=0.6)
        self.current_jumps = self.ax_image.scatter([], [], s=6, c='red', marker='s')
        self.earlier_jumps = self.ax_image.scatter([], [], s=6, facecolors='none',
                                                   edgecolors='orange', marker='s')
        self.ax_image.set_xlim(-0.5, self.ncols - 0.5)
        self.ax_image.set_ylim(-0.5, self.nrows - 0.5)
        self.ax_image.set_ylabel('row')

        self.aperture_line, = self.ax_profile.plot([], [], '.-', ms=3, lw=0.6,
                                                   color='tab:blue', label='aperture')
        self.background_line, = self.ax_profile.plot([], [], '.-', ms=3, lw=0.6,
                                                     color='tab:orange', label='background')
        self.jump_line, = self.ax_jumps.plot([], [], '-', color='tab:green', lw=0.8, alpha=0.7)
        self.ax_jumps.set_ylabel('new jumps in the aperture', color='tab:green')
        self.profile_marker = self.ax_profile.axvline(0, color='0.3', lw=1)
        self.ax_profile.legend(fontsize=8, loc='upper right')

        self.curve_marker = None
        if self.curves:
            self.ax_curves = figure.add_axes(curve_box, sharex=None)
            x = self.first_integration + np.arange(self.nints)
            for label, values in self.curves.items():
                self.ax_curves.plot(x, values, '.-', ms=3, lw=0.5, label=label)
            self.ax_curves.set_xlim(x[0] - 0.5, x[-1] + 0.5)
            self.ax_curves.set_xlabel('integration')
            self.ax_curves.legend(fontsize=8, loc='lower right', ncol=3)
            self.curve_marker = self.ax_curves.axvline(x[0], color='0.3', lw=1)

        # Widgets:
        group = group % self.ngroups
        integration = integration % self.nints
        first = self.first_integration
        self.integration_slider = Slider(figure.add_axes([0.10, 0.075, 0.70, 0.025]),
                                         'integration', first, first + self.nints - 1,
                                         valinit=first + integration, valstep=1)
        self.group_slider = Slider(figure.add_axes([0.10, 0.03, 0.70, 0.025]), 'group', 0,
                                   self.ngroups - 1, valinit=group, valstep=1)
        self.mode_buttons = RadioButtons(figure.add_axes([0.865, 0.42, 0.13, 0.30]),
                                         self.modes, active=self.modes.index(mode))
        self.axis_buttons = RadioButtons(figure.add_axes([0.865, 0.30, 0.13, 0.09]), AXES,
                                         active=AXES.index(axis))
        for buttons in (self.mode_buttons, self.axis_buttons):
            for label in buttons.labels:
                label.set_fontsize(8)
        self.jump_buttons = CheckButtons(figure.add_axes([0.865, 0.23, 0.13, 0.05]), ['jumps'],
                                         [self.state['jumps']])
        self.integration_slider.on_changed(self.update)
        self.group_slider.on_changed(self.update)
        self.mode_buttons.on_clicked(lambda label: self.set(mode=label))
        self.axis_buttons.on_clicked(lambda label: self.set(axis=label))
        self.jump_buttons.on_clicked(self._toggle_jumps)
        figure.canvas.mpl_connect('key_press_event', self._key)
        self.update()

    # Data:

    def _frame(self, i, g):
        return np.asarray(self.ramps[i, g], np.float32)

    def _statistics(self, g):
        """Median and noise over the reference integrations of the counts and increments of
        group ``g`` (computed once per group)."""

        if g not in self._statistics_cache:
            counts = np.asarray(self.ramps[self.reference, g], np.float32)
            increment = counts - np.asarray(self.ramps[self.reference, g - 1], np.float32) \
                if g else counts
            statistics = {}
            for name, values in (('counts', counts), ('increment', increment)):
                median = _median(values)
                if self.read_noise is None:
                    noise = 1.4826*_median(np.abs(values - median[None]))
                else:
                    reads = 1 if (name == 'counts' or g == 0) else 2
                    noise = np.sqrt(reads*self.read_noise**2
                                    + np.clip(median, 0, None)/self.gain)
                statistics[name] = (median, np.where(noise > 0, noise, np.nan))
            self._statistics_cache[g] = statistics
        return self._statistics_cache[g]

    def image(self, integration, group, mode = None):
        """
        Image of an integration (index within the ramps) and group in a mode (default: the
        current mode). Returns the image and its unit.
        """

        mode = self.state['mode'] if mode is None else mode
        i, g = integration, group
        if mode == 'rate / median - 1':
            return 100*_quiet(lambda: self.rateints[i]/self.median_rate - 1), '%'
        if mode == 'rate ratio / median - 1':
            return 100*(self.rate_ratio[i] - 1), '%'
        kind = 'counts' if mode.startswith('counts') else 'increment'
        data = self._frame(i, g)
        if kind == 'increment' and g:
            data = data - self._frame(i, g - 1)
        if mode in ('counts', 'increment'):
            return data, 'DN'
        median, noise = self._statistics(g)[kind]
        if mode.endswith('(sigma)'):
            return _quiet(lambda: (data - median)/noise), 'sigma'
        return data - median, 'DN'

    def _new_jumps(self):
        """Number of pixels of the aperture with a new jump, per integration and group."""

        counts = np.zeros((self.nints, self.ngroups), int)
        for i in range(self.nints):
            flagged = (np.asarray(self.groupdq[i]) & JUMP_DET) != 0
            new = flagged.copy()
            new[1:] &= ~flagged[:-1]
            counts[i] = (new & self.aperture[None]).sum(axis=(1, 2))
        return counts

    def profile(self, axis = None, integration = None, group = None, mode = None):
        """
        Mean of the image within the aperture (as an extraction would see it) and median in
        the background (robust to the trace wings and to outliers), over the integrations
        (at ``group``) or over the groups (of ``integration``; index within the ramps), in a
        mode (defaults: the current axis, integration, group and mode). Returns the two
        arrays.
        """

        axis = self.state['axis'] if axis is None else axis
        mode = self.state['mode'] if mode is None else mode
        i = self._integration() if integration is None else integration
        g = int(self.group_slider.val) if group is None else group
        if mode in RATE_MODES and axis == 'groups':
            i, g = i, 0
        key = (axis, mode, g if axis == 'integrations' else i)
        if key not in self._profile_cache:
            n = self.nints if axis == 'integrations' else self.ngroups
            values = np.full((2, n), np.nan)
            for k in range(n):
                data = self.image(k, g, mode)[0] if axis == 'integrations' else \
                    self.image(i, k, mode)[0]
                values[0, k] = _quiet(np.nanmean, data[self.aperture])
                if self.background.any():
                    values[1, k] = _robust(data[self.background])[0]
            self._profile_cache[key] = values
        return self._profile_cache[key]

    # Display:

    def _integration(self):
        return int(self.integration_slider.val) - self.first_integration

    def _limits(self, data):
        if self.state['mode'] in ('counts', 'increment'):
            finite = data[self.aperture & np.isfinite(data)]
            if finite.size == 0:
                return 0, 1
            return np.percentile(finite, 0.5), np.percentile(finite, 99.5)
        median, sd = _robust(data[self.aperture])
        width = SCALES[self.state['scale']]*sd
        if not (np.isfinite(median) and np.isfinite(width) and width > 0):
            return -1, 1
        return median - width, median + width

    def update(self, _ = None):
        """Redraw the figure for the current sliders, mode and axis."""

        i, g = self._integration(), int(self.group_slider.val)
        mode, axis = self.state['mode'], self.state['axis']
        data, unit = self.image(i, g)
        self.shown.set_data(data)
        self.shown.set_clim(*self._limits(data))
        self.shown.set_cmap('magma' if mode in ('counts', 'increment') else 'RdBu_r')
        self.bar.set_label(unit)

        none = np.zeros((0, 2))
        if self.state['jumps'] and self.groupdq is not None:
            flags = (np.asarray(self.groupdq[i]) & JUMP_DET) != 0
            now = np.argwhere(flags[g])
            before = np.argwhere(flags[:g].any(0) & ~flags[g]) if g else none
            self.current_jumps.set_offsets(now[:, ::-1] if len(now) else none)
            self.earlier_jumps.set_offsets(before[:, ::-1] if len(before) else none)
        else:
            self.current_jumps.set_offsets(none)
            self.earlier_jumps.set_offsets(none)

        median, sd = _robust(data[self.aperture])
        number = self.first_integration + i
        per_integration = mode in RATE_MODES
        jumps = '' if self.new_jumps is None else \
            f'; new jumps in the aperture: {self.new_jumps[i].sum()}'
        self.ax_image.set_title(
            f'{mode}: integration {number}' + ('' if per_integration else f', group {g}')
            + f'   |   aperture median {median:+.4g} {unit}, robust sd {sd:.3g} {unit}{jumps}'
            f'; colour +-{SCALES[self.state["scale"]]} sd', fontsize=9)

        aperture, background = self.profile()
        if axis == 'integrations':
            x = self.first_integration + np.arange(self.nints)
            position = number
            jumps = None if self.new_jumps is None else self.new_jumps.sum(1)
            what = 'integration' + ('' if per_integration else f' (group {g})')
        else:
            x = np.arange(self.ngroups)
            position = g
            jumps = None if self.new_jumps is None else self.new_jumps[i]
            what = f'group (integration {number})'
        self.aperture_line.set_data(x, aperture)
        self.background_line.set_data(x, background)
        if jumps is not None:
            self.jump_line.set_data(x, jumps)
            self.ax_jumps.set_ylim(0, max(1, jumps.max())*1.1)
        self.profile_marker.set_xdata([position, position])
        self.ax_profile.set_xlim(x[0] - 0.5, x[-1] + 0.5)
        self.ax_profile.relim()
        self.ax_profile.autoscale_view(scalex=False)
        self.ax_profile.set_xlabel(what)
        self.ax_profile.set_ylabel(unit)
        self.ax_profile.set_title(f'{mode}: mean of the aperture, median of the background, '
                                  f'against the {axis}', fontsize=9)
        if self.curve_marker is not None:
            self.curve_marker.set_xdata([number, number])
        self.figure.canvas.draw_idle()

    def set(self, integration = None, group = None, mode = None, axis = None):
        """
        Show another integration (number as on the slider), group, mode or axis of the
        middle panel (``'integrations'`` or ``'groups'``).
        """

        if mode is not None:
            if mode not in self.modes:
                raise ValueError(f'mode must be one of {self.modes}')
            self.state['mode'] = mode
        if axis is not None:
            if axis not in AXES:
                raise ValueError(f'axis must be one of {AXES}')
            self.state['axis'] = axis
        # The widgets redraw when set; set them quietly, then redraw once:
        for widget, value in ((self.integration_slider, integration),
                              (self.group_slider, group),
                              (self.mode_buttons, None if mode is None else
                               self.modes.index(mode)),
                              (self.axis_buttons, None if axis is None else AXES.index(axis))):
            if value is None:
                continue
            widget.eventson = False
            if widget in (self.mode_buttons, self.axis_buttons):
                if widget.value_selected != widget.labels[value].get_text():
                    widget.set_active(value)
            else:
                widget.set_val(value)
            widget.eventson = True
        self.update()

    def _toggle_jumps(self, _):
        self.state['jumps'] = not self.state['jumps']
        self.update()

    def _key(self, event):
        slider, step = {'right': (self.integration_slider, 1),
                        'left': (self.integration_slider, -1),
                        'up': (self.group_slider, 1),
                        'down': (self.group_slider, -1)}.get(event.key, (None, 0))
        if slider is not None:
            slider.set_val(min(max(slider.val + step, slider.valmin), slider.valmax))
        elif event.key == 'a':
            axis = AXES[1 - AXES.index(self.state['axis'])]
            self.axis_buttons.set_active(AXES.index(axis))
        elif event.key == 'c':
            self.state['scale'] = (self.state['scale'] + 1) % len(SCALES)
            self.update()

    def show(self):
        """Show the figure (``matplotlib.pyplot.show``)."""

        import matplotlib.pyplot as plt
        plt.show()

    def save(self, filename, **kwargs):
        """Save the figure as it is now (``savefig`` keywords)."""

        self.figure.savefig(filename, **kwargs)


def inspect_ramps(ramps, groupdq = None, show = True, **options):
    """
    Open the interactive ramp viewer (`RampViewer`): the image of one integration and group of
    the ramps (top), and the mean of the image within the aperture and its median in the
    background against the integrations or the groups (middle), with sliders for the integration and
    the group; see `RampViewer` for the image modes, the options and the controls.

    Parameters
    ----------

    ramps : array_like
        Ramps (integration x group x row x column).

    groupdq : array_like
        (Optional) group DQ of the ramps, for the jumps. Default is None.

    show : bool
        (Optional) show the figure (``matplotlib.pyplot.show``; it blocks in a script until
        the window is closed). Default is True.

    **options
        Options of `RampViewer` (``rateints``, ``comparison_rateints``, ``curves``, ``trace``,
        ``aperture_radius``, ``mode``, ``axis``, ...).

    Returns
    -------

    RampViewer
        The viewer; keep it referenced for the widgets to respond. ``viewer.save(filename)``
        saves the figure, and ``viewer.set(integration=..., group=..., mode=..., axis=...)``
        changes what it shows.
    """

    viewer = RampViewer(ramps, groupdq, **options)
    if show:
        viewer.show()
    return viewer
