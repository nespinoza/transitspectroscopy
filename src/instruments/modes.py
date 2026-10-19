"""Declarative description of observing modes and their registry.

Every instrument/mode supported by transitspectroscopy is described by one
:class:`Mode`: how to recognise its data, which detector-level steps to run by
default, how to trace and extract it, and its known limitations. The reduction code
looks these values up instead of branching on mode names, so adding a mode means
registering one more :class:`Mode` (plus, optionally, a tracing or Stage 2 plug-in).
"""

from dataclasses import dataclass, field

_REGISTRY = {}
_ALIASES = {}


class UnknownModeError(ValueError):
    """Raised when data do not match any registered mode."""


@dataclass(frozen=True)
class Mode:
    """One observing mode.

    Parameters
    ----------
    name : str
        Registry name, e.g. ``'nirspec/g395h'``. Stored as ``load.mode``.
    observatory, instrument : str
        E.g. ``'JWST'``, ``'nirspec'``.
    label : str
        Human-readable name; ``{filter}`` is replaced by the filter in messages.
    kind : {'slit_spectroscopy', 'slitless_spectroscopy', 'imaging'}
    match : tuple of dict
        Alternative rules to recognise the data. A rule matches when every key it lists
        is known for the data and equal (case-insensitive) to the rule's value. Keys are
        ``instrument``, ``dispersive_element`` (grating, else pupil, else ``'None'``),
        ``filter``, ``exp_type`` and ``detector``.
    detectors : tuple of str
        Detectors on which the mode is supported.
    n_orders : int
        Number of spectral orders extracted.
    dispersion_axis : int
        Detector axis of the dispersion direction (1: along columns).
    trace_source : str or callable
        ``'ccf'`` (cross-correlation tracing) or ``'reference_model'`` (instrument trace
        model, e.g. PASTASOSS, refined on the data).
    wavelength_source : str
        Where wavelengths come from, e.g. ``'wcs'`` or ``'pastasoss'``.
    stage2 : str or callable
        Built-in Stage 2 handler (``'spectral'``, ``'soss'``, ``'imaging'``), or a callable
        ``f(input_dictionary, config, outputfolder, suffix)`` for a plug-in mode.
    detector_steps : dict
        Defaults of ``load.detector_calibration``: ``refpix`` (``'side_pedestal'`` or
        ``'jwst'``), ``refpix_parameters``, ``steps`` (run before linearity is saved),
        ``post_linearity_steps``, ``post_jump_steps``, ``tso_jump`` (``window``,
        ``nsigma``), ``nan_fill_window`` (median-filter window used to fill NaNs before
        tracing) and ``group_difference_1f`` (True if the mode supports the group-difference
        1/f correction of ``load.fit_ramps``).
    tracing : dict
        Per-detector tracing parameters for the ``'spectral'`` Stage 2 (``xstart``,
        ``xend``, ``nknots``, ``outlier_nsigma``, ``outlier_window``, ``ccf_method``,
        ``ccf_function``, ``ccf_parameters``, ``start_columns`` and ``nan_fill_window``).
    extraction_defaults : dict
        Default Stage 2 extraction and 1/f parameters.
    background_strategy : str
        ``'none'``, ``'local'`` (background aperture around the trace) or
        ``'jwst_background_step'`` (the JWST background subtraction step).
    known_limitations : str
        Free text shown in the documentation.
    aliases : tuple of str
        Other names accepted by :func:`get_mode`.
    """

    name: str
    observatory: str
    instrument: str
    label: str
    kind: str
    match: tuple
    detectors: tuple = ()
    n_orders: int = 1
    dispersion_axis: int = 1
    trace_source: object = 'ccf'
    wavelength_source: str = 'wcs'
    stage2: object = 'spectral'
    detector_steps: dict = field(default_factory=dict)
    tracing: dict = field(default_factory=dict)
    extraction_defaults: dict = field(default_factory=dict)
    background_strategy: str = 'none'
    known_limitations: str = ''
    aliases: tuple = ()

    def describe(self, filter=None):
        """Label used in progress messages, e.g. ``'MIRI/Photometry (F1500W)'``."""
        return self.label.format(filter=(filter or '').upper())

    def tracing_for(self, detector):
        """Tracing parameters for ``detector`` (case-insensitive)."""
        key = str(detector).lower()
        if key not in self.tracing and '*' in self.tracing:
            key = '*'  # Detector-independent parameters.
        if key not in self.tracing:
            raise UnknownModeError(
                f'Detector {detector} not yet supported for {self.label}; '
                f'supported: {sorted(self.tracing)}'
            )
        return dict(self.tracing[key])

    def matches(self, properties):
        """Whether data with the given lowercase ``properties`` belong to this mode."""
        for rule in self.match:
            if all(
                properties.get(key) is not None and properties[key] == str(value).lower()
                for key, value in rule.items()
            ):
                return True
        return False


def register(mode, replace=False):
    """Add a :class:`Mode` to the registry (``replace=True`` to redefine one)."""
    if mode.name in _REGISTRY and not replace:
        raise ValueError(f'Mode {mode.name} is already registered')
    _REGISTRY[mode.name] = mode
    for alias in mode.aliases:
        _ALIASES[alias] = mode.name
    return mode


def unregister(name):
    """Remove a mode (and its aliases) from the registry."""
    mode = _REGISTRY.pop(name)
    for alias in mode.aliases:
        _ALIASES.pop(alias, None)
    return mode


def list_modes():
    """Names of all registered modes."""
    return sorted(_REGISTRY)


def get_mode(name):
    """Return a registered :class:`Mode` by name or alias."""
    name = _ALIASES.get(name, name)
    if name not in _REGISTRY:
        raise UnknownModeError(_unknown_message(f'Mode {name!r}'))
    return _REGISTRY[name]


def identify(instrument=None, dispersive_element=None, filter=None, exp_type=None, detector=None):
    """Find the registered mode of data with the given properties."""
    properties = {
        key: (None if value is None else str(value).lower())
        for key, value in dict(
            instrument=instrument,
            dispersive_element=dispersive_element,
            filter=filter,
            exp_type=exp_type,
            detector=detector,
        ).items()
    }
    for mode in _REGISTRY.values():
        if mode.matches(properties):
            return mode
    described = '/'.join(
        str(v) for v in (instrument, dispersive_element, filter, exp_type) if v is not None
    )
    raise UnknownModeError(_unknown_message(f'Instrument/mode {described}'))


def model_properties(model):
    """Properties used by :func:`identify`, read from a JWST datamodel's metadata.

    Missing metadata are returned as ``None`` (they then neither match nor exclude).
    """
    meta = getattr(model, 'meta', None)
    instrument = getattr(meta, 'instrument', None)
    exposure = getattr(meta, 'exposure', None)
    name = getattr(instrument, 'name', None)
    grating = getattr(instrument, 'grating', None)
    pupil = getattr(instrument, 'pupil', None)
    if grating is not None:
        dispersive_element = grating
    elif pupil is not None:
        dispersive_element = pupil
    else:
        dispersive_element = None if name is None else 'None'
    return dict(
        instrument=name,
        dispersive_element=dispersive_element,
        filter=getattr(instrument, 'filter', None),
        exp_type=getattr(exposure, 'type', None),
        detector=getattr(instrument, 'detector', None),
    )


def identify_model(model):
    """Find the registered mode of a JWST datamodel."""
    return identify(**model_properties(model))


def _unknown_message(what):
    return (
        f'{what} not yet supported! Registered modes: {", ".join(list_modes())}. '
        'See the "Adding an observing mode" section of the documentation to register one.'
    )
