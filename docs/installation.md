# Installation details

`transitspectroscopy` requires **Python ≥ 3.13** (and < 3.15, the range supported by the
JWST pipeline) and the [JWST calibration pipeline](https://github.com/spacetelescope/jwst),
`jwst>=3.0.0`. Both are enforced by `pyproject.toml`: pip refuses to install the package
on older Pythons, and always installs `jwst`, `stdatamodels`, `crds` and `gwcs`.

## Full pipeline environment (recommended)

Use [environment.yml](../environment.yml) from the checkout on Linux or macOS. Run these
commands from the repository root:

```sh
conda env create -f environment.yml
conda activate transitspectroscopy

# Keep an existing CRDS_PATH; otherwise use a cache in your home directory.
export CRDS_PATH="${CRDS_PATH:-$HOME/crds_cache}"
if [ ! -d "$CRDS_PATH" ]; then
    mkdir -p "$CRDS_PATH"
fi
# Restore this path automatically on future environment activations.
conda env config vars set -n transitspectroscopy CRDS_PATH="$CRDS_PATH"

python -m pip install --no-build-isolation --no-deps .
python -m pip check
```

The final step installs this checkout; repeat it after updating the source. A recent
Conda or Miniforge installation is recommended; `--solver libmamba` speeds up solving.

The environment pins **Python 3.13**, **JWST 3.0.0**, stdatamodels 6.0, CRDS 14,
NumPy 2.2.6, SciPy 1.17.1, Astropy 7.2.2, juliet 2.2.10 and Ray 2.58.0. The pins make
environment creation repeatable; they do not track future releases. See the
[JWST installation guidance](https://jwst-pipeline.readthedocs.io/en/stable/getting_started/install.html).

To upgrade an existing environment, run the following, then repeat the local
installation and `pip check` commands above. Restart Python/Jupyter processes afterwards.

```sh
conda env update -n transitspectroscopy -f environment.yml
```

If your existing environment uses Python 3.12 or older, create a new one instead:
`conda env update` cannot change the Python minor version reliably.

### Samplers and MultiNest

Conda supplies GSL, a C compiler, and the native MultiNest library **and** its Python
wrapper. This matters because the fitting functions default to `sampler='multinest'`.
Dynesty 2, emcee and juliet's UltraNest dependency provide alternative samplers. Dynesty
is held at major version 2 to retain the API used by juliet 2.2.

Use short output-folder paths with MultiNest: its native 3.10 library has a
100-character filename limit, including the filenames added by juliet.

### Checking the installation

```sh
python -m pytest tests -q
python tests/check_environment.py
```

The normal tests use local synthetic references and do not download observations.
See [tests/README.md](../tests/README.md) for the optional checks.

## Calibration reference files (CRDS)

The commands above configure the CRDS cache, preserving any non-empty `CRDS_PATH`
already set in your shell or Conda environment; otherwise they use `$HOME/crds_cache`.
An existing cache and its contents are left untouched. The YAML sets
`CRDS_SERVER_URL=https://jwst-crds.stsci.edu` on activation.

Reference files are downloaded as needed. Choose a cache location with enough disk space
for your instrument modes. For repeatable science reductions, record the JWST version and
the `CRDS_CONTEXT` used alongside your outputs; the environment file alone does not freeze
calibration references. See the
[CRDS setup instructions](https://jwst-pipeline.readthedocs.io/en/stable/getting_started/install.html#calibration-references-data-system-crds-setup).
MAST access to proprietary observations additionally requires your own MAST token.

## Ramp errors in JWST 3 products

The code accepts modern ramp models without `ERR` as well as older ramps with errors.
`load.ramps_err` remains an array for compatibility, with zero placeholders where ramp
errors are absent. Check `load.ramps_err_available` (one boolean per integration) before
using it: missing errors are **not** zero measurement noise. Fitted rate products retain
their own `err` arrays.

## Notebooks

Start notebooks with `jupyter lab` after activating the environment. To expose its kernel
to a Jupyter installation elsewhere:

```sh
python -m ipykernel install --user --name transitspectroscopy --display-name "Python (transitspectroscopy)"
```

## pip installation and optional dependencies

```sh
python -m pip install .
python -m pip install ".[fitting,parallel,download,plots,test]"   # or ".[all]"
```

| Extra | Installs | Needed for |
|---|---|---|
| `fitting` | juliet, dynesty 2 | `ts.transitfitting` light-curve fits |
| `parallel` | ray | parallel tracing, extraction and fitting |
| `download` | astroquery, pandas | `ts.jwst.download` |
| `plots` | matplotlib | plotting helpers and validation figures |
| `test` | pytest, matplotlib | the test suite |

These optional packages are imported only when the corresponding feature is used, so
`import transitspectroscopy` works without them. A pip installation does not provide the
native MultiNest library; use the Conda environment, or another sampler, for fitting.

## Optional C extensions and the legacy C reference

The Python extraction and correlation backends need only NumPy/SciPy. The `CCF` C
extension is built when a compiler is available, and NumPy is used otherwise. An existing
installation with a separately built `Marsh` extension continues to use it with
`backend='auto'`.

The unchanged Marsh C reference requires NumPy 1 headers, so it lives in a separate,
comparison-only environment, [environment-c-reference.yml](../environment-c-reference.yml).
It is not needed for normal use:

```sh
conda env create -f environment-c-reference.yml
conda activate transitspectroscopy-c-reference
python tests/build_c_reference.py --gsl-prefix "$CONDA_PREFIX" --output /tmp/transitspectroscopy-c-reference
PYTHONPATH=/tmp/transitspectroscopy-c-reference python -m pytest tests -q
conda activate transitspectroscopy
```

Keep that reference directory off `PYTHONPATH` in the pipeline environment.
