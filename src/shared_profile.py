"""Shared intrinsic profiles (compatibility module).

The implementation lives in :mod:`transitspectroscopy.core.shared_profile`.
"""
from .core.shared_profile import SharedProfile, fit_shared_profile  # noqa: F401

__all__ = ['SharedProfile', 'fit_shared_profile']
