# TransitSpectroscopy 🪐 --- a package for all your transit spectroscopy needs
-------------------------------------------------------------------------------

`transitspectroscopy` is a package containing algorithms and wrappers useful for performing transit (transmission and emission) spectroscopy of exoplanetary systems.

**Author**: Nestor Espinoza (nespinoza@stsci.edu)

## Statement of need
I might fill this in later, but basically: I needed a package to make available all the tools useful for transit spectroscopy: from tracing, spectral extraction to actual lightcurve fitting of wavelength-dependant transits.

It is built around the [JWST calibration pipeline](https://github.com/spacetelescope/jwst) and currently supports JWST NIRSpec (PRISM, G395H, G395M), NIRISS/SOSS and MIRI imaging time series.

## Installation

Requires Python 3.13 or 3.14; the JWST pipeline (`jwst>=3.0.0`) is installed as a dependency. From the repository root:

```sh
conda env create -f environment.yml          # recommended: includes juliet, MultiNest, Ray
conda activate transitspectroscopy
python -m pip install --no-build-isolation --no-deps .
```

or `python -m pip install ".[all]"`. Calibrating JWST data needs CRDS reference files; see the [installation guide](docs/installation.md).

## Example

```python
import glob
import transitspectroscopy as ts

ts.jwst.download(pid=1118, obs_num='5')                 # HAT-P-14 b, NIRSpec/G395H
files = glob.glob('JWSTdata/jw01118005001_04101*nrs1_uncal.fits')

dataset = ts.jwst.load(files, outputfolder='JWSTdata')
dataset.detector_calibration()                          # includes the TSO jump step
dataset.fit_ramps()

inputs = {'rampstep': dataset.rateints_per_segment, 'times': dataset.times,
          'ints_per_segment': [s.data.shape[0] for s in dataset.rateints_per_segment]}
outputs = ts.jwst.stage2(inputs, aperture_radius=2, outputfolder='JWSTdata')
white_light = outputs['whitelight']
```

## Documentation

The [documentation](docs/index.md) has a quickstart, user guides for each instrument mode, optimal extraction (polynomial and Gaussian-process profiles), shared spatial profiles, correlated-read ramp fitting and light-curve fitting, tutorials, the API reference and validation reports on real JWST data. The original worked notebooks are in [`tutorials/`](tutorials/).

## Licence and attribution

Read the `LICENCE` file for licencing details on how to use the code.

## Citation

You can cite this software via its DOI: 10.5281/zenodo.6960923 (see https://zenodo.org/record/6960924#.YutMaezMLUI). Here's the citation snippet:

    @software{espinoza_nestor_2022_6960924,
      author       = {Espinoza, Nestor},
      title        = {TransitSpectroscopy},
      month        = aug,
      year         = 2022,
      publisher    = {Zenodo},
      version      = {0.3.11},
      doi          = {10.5281/zenodo.6960924},
      url          = {https://doi.org/10.5281/zenodo.6960924}
    }
