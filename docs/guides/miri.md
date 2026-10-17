# MIRI imaging time series

`ts.miri.reduce_miri` reduces a MIRI imaging time-series visit to aperture and optimal
(PSF-weighted) photometry:

```python
import glob
import transitspectroscopy as ts

files = sorted(glob.glob('data/jw09234001001_03101_00001-seg00?_mirimage_rateints.fits'))
result = ts.miri.reduce_miri(files, outputfolder='miri_products', input_product='rateints')
```

## Processing steps

1. **Calibration:** official JWST `calwebb_detector1` (for `uncal` inputs) and
   `calwebb_image2` (flat field, photometric calibration), never resampling the time series.
2. **Preparation:** a cutout around the target with a WCS-initialized centroid and an annular
   background, with sky uncertainty propagated.
3. **Photometry:** a fractional circular aperture sum, and optimal photometry with a 2-D
   Gaussian-process PSF model that is shared by the whole visit and evaluated at each
   integration's centroid. Fluxes are fitted independently per integration.

Each call reduces one visit; separate exposures keep their identities and time gaps.

## Validation and limitations

The LTT 1445 A F1500W reduction (PID 9234) is documented in {doc}`../validation/miri` and
{doc}`../validation/miri-report`. Injection tests recover deep events, but two injected
300 ppm eclipses were recovered about 30% low (about 2.6 sigma each). Treat GP photometry of
shallow eclipses with caution until this is understood, and compare it with the aperture sums.
