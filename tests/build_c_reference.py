"""Build unchanged C sources separately: python tests/build_c_reference.py --help.

Requires NumPy < 2 (the original sources use its legacy struct layout), a C
compiler, setuptools and GSL. Does not install or alter either source file.
"""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys

import numpy as np
from setuptools import Extension, setup


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gsl-prefix', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if int(np.__version__.split('.')[0]) >= 2:
        parser.error('Use a separate NumPy < 2 environment for unchanged legacy C sources')
    root = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    gsl = args.gsl_prefix.resolve()
    sources = {'Marsh': root / 'src/c-code/OptimalExtraction/Marsh.c',
               'CCF': root / 'src/c-code/Utilities/CCF.c'}
    extensions = [Extension(name, [str(path)],
                  include_dirs=[np.get_include(), str(gsl / 'include')],
                  library_dirs=[str(gsl / 'lib')],
                  libraries=['gsl', 'gslcblas', 'm'] if name == 'Marsh' else ['m'])
                  for name, path in sources.items()]
    setup(name='transitspectroscopy-c-reference', version='0', packages=[],
          ext_modules=extensions,
          script_args=['build_ext', '--build-lib', str(output),
                       '--build-temp', str(output / 'build')])
    metadata = {'python': sys.version, 'numpy': np.__version__,
                'platform': platform.platform(),
                'gsl': subprocess.check_output([str(gsl / 'bin/gsl-config'), '--version'], text=True).strip(),
                'source_sha256': {k: hashlib.sha256(v.read_bytes()).hexdigest() for k, v in sources.items()}}
    (output / 'build_manifest.json').write_text(json.dumps(metadata, indent=2) + '\n')


if __name__ == '__main__':
    main()
