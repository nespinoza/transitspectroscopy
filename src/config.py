"""Configuration objects for the reduction stages.

:class:`ExtractionConfig` collects every option of :func:`transitspectroscopy.jwst.stage2`
in one validated object. The mode-specific checks use the observing-mode registry
(:mod:`transitspectroscopy.instruments`), so the same configuration class serves the
spectroscopic, NIRISS/SOSS and MIRI imaging paths.
"""

from dataclasses import dataclass, field, fields, replace
import warnings

# Stage 2 keywords added in 0.5.0 development that ExtractionConfig replaces. They still
# work, with a DeprecationWarning, and their defaults:
DEPRECATED_STAGE2_KEYWORDS = dict(
    extraction_backend='auto',
    profile_method='polynomial',
    gp_options=None,
    extraction_options=None,
    shared_profile=None,
    shared_profile_options=None,
    soss_options=None,
    miri_options=None,
)

# The original stage2 arguments that are also ExtractionConfig fields, with their defaults.
STAGE2_ARGUMENTS = dict(
    zero_nans=True,
    scale_1f=True,
    single_trace_extraction=True,
    optimal_extraction=False,
    aperture_radius=None,
    mask_dq=False,
)

_EXTRACTION_OPTION_KEYS = ('polynomial_spacing', 'polynomial_order', 'nsigma')


@dataclass
class ExtractionConfig:
    """Options of Stage 2 (tracing and extraction, or photometry for imaging modes).

    Parameters
    ----------
    aperture_radius : float or None
        Extraction (or photometric) aperture radius in pixels; ``None`` uses the mode default.
    optimal_extraction : bool
        Marsh (1989) optimal extraction instead of a box aperture.
    single_trace_extraction : bool
        Extract all integrations with the median trace (``True``) or each with its own.
    mask_dq : bool
        Ignore pixels with non-zero DQ flags.
    zero_nans : bool
        Replace missing pixels by zeros (``True``) or by a median-filtered rate image.
    scale_1f : bool
        Remove 1/f noise with the median-scaling technique.
    backend : {'auto', 'c', 'python'}
        Optimal-extraction backend.
    profile_method : {'polynomial', 'gp'}
        Spatial-profile model of optimal extraction (GP is experimental).
    gp_options : dict or None
        Fixed GP kernel options (``kernel``, ``length_scale``, ``amplitude``, ``n_inducing``);
        for MIRI imaging, the 2-D PSF GP options.
    polynomial_spacing, polynomial_order, nsigma : float, int, float or None
        Optimal-extraction settings; ``None`` uses the mode defaults.
    shared_profile : None, 'train' or SharedProfile
        Use one intrinsic profile for all integrations (see ``fit_shared_profile``).
    shared_profile_options : dict or None
        Extra options of ``fit_shared_profile`` (e.g. ``training_indices``).
    soss : dict or None
        NIRISS/SOSS options (keywords of ``soss.extract_soss``, e.g. ``background_reference``,
        ``optimal_methods``, ``optimal_options``, ``trace_max_shift``).
    miri : dict or None
        MIRI imaging options (keywords of ``miri.extract_photometry``, e.g.
        ``preparation_options``, ``image_options``, ``training_integrations``).
    """

    aperture_radius: object = None
    optimal_extraction: bool = False
    single_trace_extraction: bool = True
    mask_dq: bool = False
    zero_nans: bool = True
    scale_1f: bool = True
    backend: str = 'auto'
    profile_method: str = 'polynomial'
    gp_options: dict = None
    polynomial_spacing: float = None
    polynomial_order: int = None
    nsigma: float = None
    shared_profile: object = None
    shared_profile_options: dict = None
    soss: dict = None
    miri: dict = None
    _deprecated: tuple = field(default=(), repr=False, compare=False)

    def extraction_options(self):
        """Explicit optimal-extraction overrides, as the legacy ``extraction_options`` dict."""
        return {
            key: getattr(self, key)
            for key in _EXTRACTION_OPTION_KEYS
            if getattr(self, key) is not None
        }

    def as_dict(self):
        """Plain dictionary of the settings (e.g. for provenance records)."""
        return {f.name: getattr(self, f.name) for f in fields(self) if not f.name.startswith('_')}

    @classmethod
    def from_stage2_arguments(cls, **arguments):
        """Build a configuration from ``stage2`` keyword arguments (old and deprecated)."""
        unknown = set(arguments) - set(STAGE2_ARGUMENTS) - set(DEPRECATED_STAGE2_KEYWORDS)
        if unknown:
            raise TypeError(f'Unknown stage2 arguments: {sorted(unknown)}')
        values = dict(STAGE2_ARGUMENTS)
        values.update(DEPRECATED_STAGE2_KEYWORDS)
        values.update(arguments)
        extraction_options = dict(values['extraction_options'] or {})
        if set(extraction_options) - set(_EXTRACTION_OPTION_KEYS):
            raise ValueError('Unknown extraction_options key')
        deprecated = tuple(
            name
            for name, default in DEPRECATED_STAGE2_KEYWORDS.items()
            if name in arguments and arguments[name] is not default and arguments[name] != default
        )
        return cls(
            aperture_radius=values['aperture_radius'],
            optimal_extraction=values['optimal_extraction'],
            single_trace_extraction=values['single_trace_extraction'],
            mask_dq=values['mask_dq'],
            zero_nans=values['zero_nans'],
            scale_1f=values['scale_1f'],
            backend=values['extraction_backend'],
            profile_method=values['profile_method'],
            gp_options=values['gp_options'],
            shared_profile=values['shared_profile'],
            shared_profile_options=values['shared_profile_options'],
            soss=values['soss_options'],
            miri=values['miri_options'],
            _deprecated=deprecated,
            **extraction_options,
        )

    def warn_deprecated(self):
        """Emit one DeprecationWarning naming any deprecated stage2 keywords that were used."""
        if self._deprecated:
            warnings.warn(
                'stage2 keywords ' + ', '.join(self._deprecated) + ' are deprecated; '
                'pass config=ExtractionConfig(...) instead',
                DeprecationWarning,
                stacklevel=3,
            )

    def validate(self, mode):
        """Check the options against an observing mode (see ``transitspectroscopy.instruments``).

        Returns the configuration, with mode-independent defaults resolved.
        """
        handler = mode.stage2
        extraction_options = self.extraction_options()
        if handler == 'imaging':
            if (
                self.soss is not None
                or self.shared_profile is not None
                or self.shared_profile_options is not None
                or extraction_options
            ):
                raise ValueError('MIRI imaging uses miri_options, not spectral extraction options')
            return self
        if self.miri is not None:
            raise ValueError('miri_options applies only to MIRI imaging exposures')
        if handler == 'soss':
            if self.shared_profile is not None or self.shared_profile_options is not None:
                raise ValueError(
                    'SOSS trains separate order profiles; use soss_options optimal_options'
                )
            if self.optimal_extraction:
                if self.backend not in ('auto', 'python'):
                    raise ValueError('SOSS shared optimal extraction requires the Python backend')
                if self.profile_method not in ('polynomial', 'gp'):
                    raise ValueError('profile_method must be polynomial or gp')
                if self.gp_options is not None and self.profile_method != 'gp':
                    raise ValueError('gp_options requires profile_method="gp"')
            elif self.gp_options is not None or extraction_options:
                raise ValueError(
                    'SOSS gp_options/extraction_options require optimal_extraction=True'
                )
            return self
        if self.soss is not None:
            raise ValueError('soss_options applies only to NIS_SOSS exposures')
        if self.backend not in ('auto', 'c', 'python') or self.profile_method not in (
            'polynomial',
            'gp',
        ):
            raise ValueError('Invalid extraction_backend or profile_method')
        if self.gp_options is not None and self.profile_method != 'gp':
            raise ValueError('gp_options requires profile_method="gp"')
        if self.profile_method == 'gp' and (not self.optimal_extraction or self.backend == 'c'):
            raise ValueError('GP requires optimal extraction with the Python backend')
        if self.shared_profile is not None:
            from .core.shared_profile import SharedProfile

            if not isinstance(self.shared_profile, SharedProfile) and not (
                isinstance(self.shared_profile, str) and self.shared_profile == 'train'
            ):
                raise ValueError('shared_profile must be "train" or a SharedProfile instance')
            if not self.optimal_extraction or self.backend == 'c':
                raise ValueError('shared profiles require optimal extraction with Python')
            if isinstance(self.shared_profile, SharedProfile):
                if self.gp_options is not None:
                    raise ValueError('supplied SharedProfile already contains frozen GP options')
                model = self.shared_profile
                return replace(
                    self,
                    profile_method=model.profile_method,
                    aperture_radius=(
                        model.aperture_radius
                        if self.aperture_radius is None
                        else self.aperture_radius
                    ),
                    polynomial_spacing=(
                        model.spacing
                        if self.polynomial_spacing is None
                        else self.polynomial_spacing
                    ),
                    polynomial_order=(
                        model.diagnostics.get('polynomial_order', 3)
                        if self.polynomial_order is None
                        else self.polynomial_order
                    ),
                )
        elif self.shared_profile_options is not None:
            raise ValueError('shared_profile_options requires shared_profile')
        return self
