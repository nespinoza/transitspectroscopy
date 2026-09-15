"""Deferred imports for capabilities outside the numerical core."""
from importlib import import_module
from inspect import signature


class OptionalModule:
    """Preserve module-style access, importing only on first use."""

    def __init__(self, name, install=None):
        self._name = name
        self._install = install or name.split('.')[0]

    def __getattr__(self, name):
        try:
            module = import_module(self._name)
        except ImportError as exc:
            raise ImportError(
                f"This capability requires {self._name}; install {self._install}."
            ) from exc
        return getattr(module, name)


class LazyRemote:
    """Keep Ray's .remote/.options interface without importing Ray eagerly."""

    def __init__(self, function):
        self._function = function
        self._remote = None
        self.__doc__ = function.__doc__
        self.__wrapped__ = function
        self.__name__ = function.__name__
        self.__signature__ = signature(function)

    def __call__(self, *args, **kwargs):
        raise TypeError('Ray functions must be called with .remote(...)')

    def __getattr__(self, name):
        if self._remote is None:
            self._remote = OptionalModule('ray').remote(self._function)
        return getattr(self._remote, name)
