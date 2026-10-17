"""Registry of observing modes (see :class:`~transitspectroscopy.instruments.modes.Mode`).

Importing this package registers the built-in JWST modes::

    from transitspectroscopy import instruments
    instruments.list_modes()
    mode = instruments.identify(instrument='nirspec', dispersive_element='g395h')
    mode.tracing_for('nrs1')['xstart']
"""

from .modes import (  # noqa: F401
    Mode,
    UnknownModeError,
    get_mode,
    identify,
    identify_model,
    list_modes,
    model_properties,
    register,
    unregister,
)
from . import jwst_modes  # noqa: F401  (registers the built-in modes)
