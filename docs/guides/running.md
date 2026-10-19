# Configuration-driven runs

A whole reduction can be written down in one TOML file and run with one command:

```sh
transitspectroscopy run my-reduction.toml
```

or, from Python, `ts.run('my-reduction.toml')` (a dictionary works too). The same
configuration always gives the same result, and every product is written with a provenance
manifest, so a reduction can be shared and reproduced from its configuration alone.

## A configuration

```toml
[run]
name = "hatp14b-g395h"
output = "runs/hatp14b-g395h"

[data]
source = "mast"                      # or "local", with glob = "data/*_uncal.fits"
program = 1118
observation = 5
exposure = "jw01118005001_04101_00001"
segments = [1, 2, 3]
detectors = ["nrs1"]
product = "uncal"                    # or "rateints"
folder = "JWSTdata"

[mode]
name = "auto"                        # or a registered mode, see `transitspectroscopy modes`

[calibration]
crds_context = "jwst_1584.pmap"      # optional: pin the calibration references
use_tso_jump = true
group_1f = false
group_difference_1f = true           # group-level 1/f with group-difference templates (default)
ramp_fit_method = "jwst"             # or "sparta" (correlated-read fitter)

[extraction.box]                     # one table per extraction variant
aperture_radius = 2.0

[extraction.gp]
aperture_radius = 2.0
optimal_extraction = true
profile_method = "gp"
gp_options = { kernel = "matern32", length_scale = 100.0, n_inducing = 24 }

[lightcurves]
white_light = true
resolving_power = 100                # spectroscopic light curves at R = 100
```

Each `[extraction.<variant>]` table holds the fields of
{class}`~transitspectroscopy.config.ExtractionConfig`. NIRISS/SOSS and MIRI options go in
nested `soss` and `miri` tables (see `validation/configs/` for complete examples).

## Steps, caching and reruns

`run` executes `data → calibrate → extract → lightcurves [→ fit]`. Each step is written to
`<output>/<step>-<hash>/`, where the hash covers the step's settings and all upstream
steps:

- rerunning an unchanged configuration reuses every completed step;
- changing, say, an extraction setting recomputes the extraction and light curves, but
  reuses the downloaded data and the detector calibration;
- `--force` (`force=True`) recomputes everything, and `--from-step extract`
  (`from_step='extract'`) recomputes that step and the following ones.

Files of several exposures (for example, a MIRI visit made of three exposures) are calibrated
exposure by exposure and reduced together in Stage 2.

## Products and provenance

Every extraction variant writes standard products (see {mod}`transitspectroscopy.core.products`):
spectroscopic time series (FITS) for each detector or spectral order, photometric time series
for imaging, and light curves (ECSV). Each file has a `<file>.provenance.json` manifest with
the package version and git commit, the Python, NumPy, JWST pipeline and CRDS versions, the
CRDS context, the input file names and SHA-256 hashes, and the full resolved configuration
with its hash.

```python
from transitspectroscopy.core import products

spectra = products.read_fits('runs/hatp14b-g395h/extract-<hash>/gp/products/nrs1_corrected.fits')
spectra.flux.shape, spectra.wavelength, spectra.times
```

Reading a product made by a different version of the code emits a warning (it is not an
error: a refactor or a documentation change does not alter results). Use `strict=True` to
make it an error.

```python
result = ts.run('my-reduction.toml')
result.products['gp']['nrs1_corrected']          # SpectroscopicTimeSeries
result.lightcurves['gp']['nrs1_corrected_R100']  # LightCurveSet
```
