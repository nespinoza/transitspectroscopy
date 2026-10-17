"""Shared native-pixel profile initialization and isolated SOSS extraction.

The positive polynomial/GP log-intensity models are those in shared_profile,
not the legacy unregularized Marsh fit. No empirical incomplete profile is
renormalized. Missing pixels contribute no likelihood, while each column's
flux remains a nuisance parameter during training.
"""
import numpy as np

from .core.extraction import frozen_profile
from .core.geometry import aperture_weights
from .core.shared_profile import fit_shared_profile


def settings(methods, options, radius):
    methods = tuple(methods)
    if len(set(methods)) != len(methods) or set(methods)-{'polynomial', 'gp'}:
        raise ValueError('optimal_methods must contain unique polynomial/gp names')
    defaults = dict(training_integrations=4, spacing=1.5, polynomial_order=5,
                    initial_width=8., optimize_hyperparameters=False,
                    gp_options=dict(n_inducing=12), maxiter=500, nsigma=0.,
                    minimum_profile_fraction=.5, overlap_strategy='joint',
                    profile_radius=None, condition_limit=1e6)
    if set(options or {})-set(defaults):
        raise ValueError('Unknown SOSS optimal_options key')
    defaults.update(options or {})
    if defaults['profile_radius'] is None:
        defaults['profile_radius'] = radius+5 if defaults['overlap_strategy']=='joint' else radius
    if methods:
        n = defaults['training_integrations']
        if not isinstance(n, (int, np.integer)) or n < 2:
            raise ValueError('training_integrations must be an integer >= 2')
        if not np.isfinite(radius) or int(radius) != radius or radius < 1:
            raise ValueError('SOSS optimal extraction requires an integer aperture radius')
        if not 0 < defaults['minimum_profile_fraction'] <= 1:
            raise ValueError('minimum_profile_fraction must be in (0, 1]')
        if not np.isfinite(defaults['nsigma']) or defaults['nsigma'] < 0:
            raise ValueError('nsigma must be finite and nonnegative')
        if defaults['overlap_strategy'] not in ('joint', 'isolated'):
            raise ValueError('overlap_strategy must be joint or isolated')
        r = defaults['profile_radius']
        if not np.isfinite(r) or int(r) != r or r < radius:
            raise ValueError('profile_radius must be an integer >= aperture radius')
        if not np.isfinite(defaults['condition_limit']) or defaults['condition_limit'] <= 1:
            raise ValueError('condition_limit must be finite and greater than one')
        if defaults['overlap_strategy']=='joint' and defaults['optimize_hyperparameters']:
            raise ValueError('Joint SOSS currently requires fixed profile hyperparameters')
    return methods, defaults


def train(data, variance, centers, columns, radius, method, options, integration_ids):
    """Train once on independent ramp-fitted integrations, retaining pixel masks."""
    return fit_shared_profile(data, variance, centers, aperture_radius=radius,
        spacing=options['spacing'], profile_method=method,
        polynomial_order=options['polynomial_order'],
        gp_options=options['gp_options'] if method == 'gp' else None,
        initial_width=options['initial_width'],
        optimize_hyperparameters=options['optimize_hyperparameters'],
        dispersion_coordinates=columns, integration_ids=integration_ids,
        maxiter=options['maxiter'], store_covariance=False, column_batch_size=128)


def extract(model, frame, variance, valid, columns, centers, radius, options):
    """Extract masked native pixels and convert to the fractional box flux scale.

    A pixel's full measured flux and variance enter the native likelihood. The
    estimated total is then scaled by sum(P * fractional_aperture_weights),
    matching the box aperture convention without rescaling its pixel errors.
    Reject inadequately observed profiles instead of extrapolating their flux
    from arbitrarily little remaining light. Coefficients are never refitted.
    """
    y, v = frame[:, columns], variance[:, columns]
    profile = model.evaluate(centers, dispersion_coordinates=columns)
    aperture_fraction = np.sum(profile*aperture_weights(len(frame), centers, radius), axis=0)
    return frozen_profile(profile, y, v, valid[:, columns], aperture_fraction,
                          nsigma=options['nsigma'],
                          minimum_profile_fraction=options['minimum_profile_fraction'])
