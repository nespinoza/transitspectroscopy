"""Use the source checkout without downloading the optional JWST pipeline."""
import importlib.util
from pathlib import Path
import sys


def load_local_package():
    if 'transitspectroscopy' not in sys.modules:
        source = Path(__file__).resolve().parents[1] / 'src'
        spec = importlib.util.spec_from_file_location(
            'transitspectroscopy', source / '__init__.py',
            submodule_search_locations=[str(source)])
        package = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = package
        spec.loader.exec_module(package)


load_local_package()
