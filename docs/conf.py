"""Sphinx configuration for the transitspectroscopy documentation."""
import os
import sys
import warnings
from pathlib import Path

# Importing the JWST pipeline for autodoc triggers this upstream photutils deprecation.
warnings.filterwarnings('ignore', message='DEFAULT_COLUMNS is deprecated')

# Document the checkout, installed under its package name.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tests'))
import _support  # noqa: E402,F401  (loads ../src as `transitspectroscopy`)
import transitspectroscopy  # noqa: E402

project = 'transitspectroscopy'
author = 'Nestor Espinoza'
copyright = '2022-2026, Nestor Espinoza'
release = transitspectroscopy.__version__
version = release

extensions = [
    'myst_nb',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    'sphinx.ext.mathjax',
    'sphinx.ext.viewcode',
    'sphinx_copybutton',
    'sphinxcontrib.mermaid',
]

exclude_patterns = ['_build', 'dev-notes', 'jupyter_execute', '**.ipynb_checkpoints']

# Markdown (MyST) everywhere; notebooks rendered with myst-nb.
myst_enable_extensions = ['dollarmath', 'amsmath', 'colon_fence', 'deflist']
myst_heading_anchors = 3
myst_fence_as_directive = ['mermaid']

# Synthetic tutorials execute on every build (CI keeps them fast). Real-data
# tutorials need downloads and CRDS, so they are shown without execution.
nb_execution_mode = os.environ.get('TS_DOCS_EXECUTE', 'cache')
nb_execution_timeout = 300
nb_execution_raise_on_error = True
nb_execution_excludepatterns = ['tutorials/realdata-*.ipynb']

autodoc_default_options = {'members': True, 'undoc-members': False}
autodoc_member_order = 'bysource'
autodoc_typehints = 'none'
napoleon_numpy_docstring = True
napoleon_google_docstring = False
autosummary_generate = False

html_theme = 'pydata_sphinx_theme'
html_title = f'transitspectroscopy {release}'
html_theme_options = {
    'github_url': 'https://github.com/nespinoza/transitspectroscopy',
    'navigation_with_keys': False,
}
