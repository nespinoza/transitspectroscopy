"""Built-in JWST time-series modes.

The numbers below are the defaults that ``transitspectroscopy.jwst`` has always used
for each mode; they were previously spread over ``if mode == ...`` branches.
"""

from .modes import Mode, register

# calwebb_detector1 steps run (before saving the linearity products) on near-IR
# detectors, and the steps run after linearity.
_NIR_STEPS = ('dq_init', 'saturation', 'superbias', 'refpix', 'group_1f', 'linearity')
_NIR_POST_LINEARITY = ('dark_sub',)
# MIRI has no superbias; its reference-pixel step runs after the dark.
_MIRI_STEPS = ('dq_init', 'emicorr', 'saturation', 'linearity')
_MIRI_POST_LINEARITY = ('dark_sub', 'refpix')

_NAN_FILL = dict(row_window=1, column_window=7)


def _gaussian_tracing(**values):
    tracing = dict(
        outlier_nsigma=5,
        outlier_window=10,
        ccf_method='convolve',
        ccf_function='gaussian',
        ccf_parameters=[0.0, 1.7],
        nan_fill_window=_NAN_FILL,
    )
    tracing.update(values)
    return tracing


_G395_EXTRACTION = dict(
    # Rows, columns and window of the median-scaled 1/f correction:
    scale_1f_rows=[0, 32],
    scale_1f_columns=[5, 2044],
    scale_1f_window=5,
    # Optimal extraction:
    polynomial_spacing=0.1,
    nsigma=10,
    polynomial_order=3,
    # Aperture, 1/f and background apertures (pixels from the trace):
    aperture_radius=2,
    onef_inner_radius=3,
    onef_outer_radius=8,
    background_subtraction=True,
    background_inner_radius=10,
    background_outer_radius=None,
)

NIRSPEC_PRISM = register(
    Mode(
        name='nirspec/prism',
        observatory='JWST',
        instrument='nirspec',
        label='NIRSpec/PRISM',
        kind='slit_spectroscopy',
        match=({'instrument': 'nirspec', 'dispersive_element': 'prism'},),
        detectors=('nrs1',),
        detector_steps=dict(
            refpix='side_pedestal',
            refpix_parameters=dict(left_pixels=25, right_pixels=25),
            steps=_NIR_STEPS,
            post_linearity_steps=_NIR_POST_LINEARITY,
            tso_jump=dict(window=200, nsigma=10),
            nan_fill_window=_NAN_FILL,
            group_difference_1f=True,
        ),
        tracing={'*': _gaussian_tracing(xstart=50, xend=490, nknots=8, start_columns=(50, 100))},
        extraction_defaults=dict(
            scale_1f_rows=[5, 25],
            scale_1f_columns=[200, 450],
            scale_1f_window=200,
            polynomial_spacing=0.1,
            nsigma=10,
            polynomial_order=3,
            aperture_radius=10,
            onef_inner_radius=4,
            onef_outer_radius=15,
            background_subtraction=False,
            background_inner_radius=14,
            background_outer_radius=None,
        ),
        background_strategy='none',
        known_limitations=(
            'The standard reference-pixel step is replaced by an edge-column pedestal.'
        ),
    )
)

NIRSPEC_G395H = register(
    Mode(
        name='nirspec/g395h',
        observatory='JWST',
        instrument='nirspec',
        label='NIRSpec/G395H',
        kind='slit_spectroscopy',
        match=({'instrument': 'nirspec', 'dispersive_element': 'g395h'},),
        detectors=('nrs1', 'nrs2'),
        detector_steps=dict(
            refpix='jwst',
            refpix_parameters={},
            steps=_NIR_STEPS,
            post_linearity_steps=_NIR_POST_LINEARITY,
            tso_jump=dict(window=10, nsigma=10),
            nan_fill_window=_NAN_FILL,
            group_difference_1f=True,
        ),
        tracing={
            'nrs1': _gaussian_tracing(xstart=2042, xend=500, nknots=60, start_columns=(1842, 2042)),
            'nrs2': _gaussian_tracing(xstart=5, xend=2043, nknots=60, start_columns=(5, 205)),
        },
        extraction_defaults=_G395_EXTRACTION,
        background_strategy='local',
    )
)

NIRSPEC_G395M = register(
    Mode(
        name='nirspec/g395m',
        observatory='JWST',
        instrument='nirspec',
        label='NIRSpec/G395M',
        kind='slit_spectroscopy',
        match=({'instrument': 'nirspec', 'dispersive_element': 'g395m'},),
        detectors=('nrs1', 'nrs2'),
        detector_steps=dict(
            refpix='jwst',
            refpix_parameters={},
            steps=_NIR_STEPS,
            post_linearity_steps=_NIR_POST_LINEARITY,
            tso_jump=dict(window=10, nsigma=10),
            nan_fill_window=_NAN_FILL,
            group_difference_1f=True,
        ),
        tracing={
            'nrs1': _gaussian_tracing(xstart=2042, xend=700, nknots=60, start_columns=(1842, 2042)),
            'nrs2': _gaussian_tracing(xstart=5, xend=500, nknots=60, start_columns=(5, 205)),
        },
        extraction_defaults=_G395_EXTRACTION,
        background_strategy='local',
    )
)

NIRISS_SOSS = register(
    Mode(
        name='niriss/soss',
        observatory='JWST',
        instrument='niriss',
        label='NIRISS/SOSS',
        kind='slitless_spectroscopy',
        match=({'exp_type': 'nis_soss'}, {'instrument': 'niriss', 'dispersive_element': 'gr700xd'}),
        detectors=('nis',),
        n_orders=3,
        trace_source='reference_model',
        wavelength_source='pastasoss',
        stage2='soss',
        detector_steps=dict(
            refpix='jwst',
            refpix_parameters={},
            steps=_NIR_STEPS,
            post_linearity_steps=_NIR_POST_LINEARITY,
            # Run after the jump step and before ramp fitting, as in calwebb_detector1, where
            # CRDS enables clean_flicker_noise for SOSS. It removes 1/f noise and group-level
            # column offsets (e.g., PID 1541 observation 1, integration 60):
            post_jump_steps=('clean_flicker_noise',),
            tso_jump=dict(window=10, nsigma=10),
            nan_fill_window=None,
            # The group-difference 1/f correction (after a first ramp fit; the default) replaces
            # clean_flicker_noise, which then only runs with group_difference_1f=False:
            group_difference_1f=True,
        ),
        extraction_defaults=dict(aperture_radius=15.0),
        background_strategy='jwst_background_step',
        known_limitations=(
            'CLEAR/GR700XD only (SUBSTRIP256 and SUBSTRIP96). The zodiacal background '
            'is removed with the JWST background step on a median rate image after ramp '
            'fitting, before tracing and extraction. Orders overlap at long wavelengths; '
            'optimal extraction fits them jointly, box spectra are blended there.'
        ),
    )
)

MIRI_IMAGING = register(
    Mode(
        name='miri/photometry',
        observatory='JWST',
        instrument='miri',
        label='MIRI/Photometry ({filter})',
        kind='imaging',
        match=(
            {'instrument': 'miri', 'exp_type': 'mir_image'},
            {'instrument': 'miri', 'dispersive_element': 'None'},
        ),
        detectors=('mirimage',),
        n_orders=0,
        trace_source=None,
        wavelength_source='filter',
        stage2='imaging',
        detector_steps=dict(
            refpix='jwst',
            refpix_parameters={},
            steps=_MIRI_STEPS,
            post_linearity_steps=_MIRI_POST_LINEARITY,
            tso_jump=dict(window=10, nsigma=10),
            nan_fill_window=None,
        ),
        extraction_defaults=dict(aperture_radius=5.0),
        background_strategy='annulus',
        known_limitations=(
            'GP PSF photometry recovered injected 300 ppm eclipses about 30% low in '
            'validation; compare with aperture sums for shallow events.'
        ),
        aliases=('miri/imaging',),
    )
)
