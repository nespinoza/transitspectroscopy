"""Build the optional CCF C extension.

All package metadata lives in pyproject.toml. The extension only speeds up
cross-correlations; transitspectroscopy falls back to NumPy when it is absent.
"""
import numpy
from setuptools import Extension, setup

ccf_module = Extension('CCF',
                       sources=['src/c-code/Utilities/CCF.c'],
                       libraries=['m'],
                       include_dirs=[numpy.get_include()],
                       optional=True)

setup(ext_modules=[ccf_module])
