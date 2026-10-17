"""Selection between the optional C extensions and their NumPy implementations."""

from importlib import import_module


def select_backend(module, backend):
    """Return the C extension ``module`` or ``None`` for the NumPy implementation.

    Parameters
    ----------
    module : str
        Name of the C extension (``'Marsh'`` or ``'CCF'``).
    backend : {'auto', 'c', 'python'}
        ``'auto'`` uses the extension if it can be imported, ``'c'`` requires it and
        ``'python'`` never uses it.
    """
    if backend not in ('auto', 'c', 'python'):
        raise ValueError('backend must be auto, c or python')
    if backend == 'python':
        return None
    try:
        return import_module(module)
    except ImportError as exc:
        if backend == 'c':
            raise ImportError(
                f'{module} C backend is unavailable; build it or use backend="python"'
            ) from exc
        return None
