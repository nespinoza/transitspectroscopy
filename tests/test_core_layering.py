"""Layering rules: core is observatory-agnostic and modules use only public names."""
import ast
from pathlib import Path
import subprocess
import sys

SRC = Path(__file__).resolve().parents[1] / 'src'


def test_core_imports_no_jwst_stack():
    code = '''
import sys, importlib.util
from pathlib import Path
src = Path(sys.argv[1])
# Load only the core subpackage, not the package __init__ (which also exposes ts.jwst).
spec = importlib.util.spec_from_file_location('transitspectroscopy_core', src / 'core' / '__init__.py',
                                              submodule_search_locations=[str(src / 'core')])
core = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = core
spec.loader.exec_module(core)
import pkgutil, importlib
for info in pkgutil.iter_modules([str(src / 'core')]):
    importlib.import_module(spec.name + '.' + info.name)
loaded = sorted(m for m in sys.modules if m.split('.')[0] in {'jwst', 'stdatamodels', 'gwcs'})
assert not loaded, loaded
'''
    result = subprocess.run([sys.executable, '-c', code, str(SRC)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_core_source_has_no_jwst_imports():
    for path in (SRC / 'core').glob('*.py'):
        for node in ast.walk(ast.parse(path.read_text())):
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0:
                names = [node.module or '']
            for name in names:
                assert name.split('.')[0] not in {'jwst', 'stdatamodels', 'gwcs'}, (path, name)


def test_no_module_imports_private_names_of_another_module():
    for path in SRC.rglob('*.py'):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom) and node.level > 0:
                module = node.module or ''
                # Private helper modules of this package (e.g. _optional, _version) are allowed.
                for alias in node.names:
                    assert not alias.name.startswith('_') or alias.name == '__version__', (
                        path, module, alias.name)
