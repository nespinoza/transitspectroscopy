# Quickstart

This page reduces a JWST NIRSpec/G395H transit observation from raw (`uncal`) files to a
white-light light curve. The same steps work for NIRSpec/PRISM and G395M. NIRISS/SOSS and
MIRI imaging have their own guides ({doc}`guides/soss`, {doc}`guides/miri`).

## 1. Download the data

`ts.jwst.download` queries MAST for a program and observation and saves the products in
`JWSTdata/` (this needs the `download` extra: `pip install "transitspectroscopy[download]"`):

```python
import glob
import transitspectroscopy as ts

ts.jwst.download(pid=1118, obs_num='5')     # HAT-P-14 b, NIRSpec/G395H commissioning
nrs1_filenames = glob.glob('JWSTdata/jw01118005001_04101*nrs1_uncal.fits')
```

Calibration needs JWST reference files from CRDS; see {doc}`installation` for the
`CRDS_PATH` and `CRDS_SERVER_URL` settings.

## 2. Detector calibration and ramp fitting

`ts.jwst.load` reads all segments of an exposure into one time series. Detector calibration
follows the JWST pipeline except where `transitspectroscopy` uses its own time-series
methods (for example, jump detection across integrations, `tso_jumpstep`):

```python
dataset = ts.jwst.load(nrs1_filenames, outputfolder='JWSTdata')
dataset.detector_calibration()
dataset.fit_ramps()

rates = dataset.rateints          # (integration, row, column), DN/s
errors = dataset.rateints_err
times = dataset.times             # BJD_TDB at mid-integration
```

## 3. Tracing and spectral extraction

`ts.jwst.stage2` traces the spectrum in every integration, corrects 1/f noise and extracts
spectra and a white-light light curve:

```python
inputs = {'rampstep': dataset.rateints_per_segment,
          'times': dataset.times,
          'ints_per_segment': [s.data.shape[0] for s in dataset.rateints_per_segment]}

outputs = ts.jwst.stage2(inputs, aperture_radius=2, outputfolder='JWSTdata', suffix='2pix')
white_light = outputs['whitelight']
```

Add `optimal_extraction=True` for Marsh (1989) optimal extraction, optionally with a
Gaussian-process spatial profile (`profile_method='gp'`); see {doc}`guides/optimal-extraction`.

## 4. Light-curve fitting

`ts.transitfitting.fit_lightcurves` fits one or many light curves with
[juliet](https://juliet.readthedocs.io) (needs the `fitting` extra); see
{doc}`guides/lightcurve-fitting`.

## Next steps

- Work through the {doc}`tutorials/index`; the synthetic ones run in a couple of minutes.
- Read {doc}`concepts` for the array conventions used throughout the package.
