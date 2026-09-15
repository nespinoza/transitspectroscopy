from ._version import __version__
__all__ = ['spectroscopy', 'utils', 'transitfitting', 'jwst']

from .spectroscopy import *
from .transitfitting import *
from .utils import *
from .jwst import *
from .shared_profile import SharedProfile, fit_shared_profile


def __getattr__(name):
    # This historically star-exported value needs JWST only when requested.
    if name == 'jwstversion':
        return jwst.jwstversion
    raise AttributeError(name)
