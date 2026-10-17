# MIRI imaging time series

MIRI imaging time series go through the same stages as every other mode: detector
calibration with `ts.jwst.load` (the MIRI step order, with the time-series jump detector),
ramp fitting, and `ts.jwst.stage2`, which for imaging measures aperture and optimal
(PSF-weighted) photometry:

```python
import glob
import transitspectroscopy as ts
from transitspectroscopy.config import ExtractionConfig

files = sorted(glob.glob('data/jw09234001001_03101_00001-seg00?_mirimage_uncal.fits'))
dataset = ts.jwst.load(files, outputfolder='miri_reduction')
dataset.detector_calibration()
dataset.fit_ramps()

inputs = {'rampstep': dataset.rateints_per_segment, 'times': dataset.times,
          'ints_per_segment': [s.data.shape[0] for s in dataset.rateints_per_segment]}
result = ts.jwst.stage2(inputs, outputfolder='miri_reduction',
                        config=ExtractionConfig(aperture_radius=5.))
```

To start from archive `rateints` files (or run the official calwebb_detector1 on `uncal`
files), call the imaging Stage 2 directly:
`ts.miri.extract_photometry(files, outputfolder='miri_products', input_product='rateints')`.
MIRI-specific options go in `ExtractionConfig(miri={...})`.

## Processing steps

1. **Calibration:** `calwebb_image2` (flat field, photometric calibration) on the ramp-fitted
   integrations, never resampling the time series.
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
