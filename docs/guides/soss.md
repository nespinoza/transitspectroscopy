# NIRISS/SOSS

NIRISS single-object slitless spectroscopy (`CLEAR/GR700XD`) disperses three orders on the
detector. SOSS data go through the same stages as every other mode: detector calibration
with `ts.jwst.load`, ramp fitting, and `ts.jwst.stage2`, which for SOSS extracts orders 1,
2 and 3:

```python
import glob
import transitspectroscopy as ts
from transitspectroscopy.config import ExtractionConfig

files = sorted(glob.glob('data/jw01541001001_04101_00001-seg00?_nis_uncal.fits'))
dataset = ts.jwst.load(files, outputfolder='soss_reduction')
dataset.detector_calibration()
dataset.fit_ramps()

inputs = {'rampstep': dataset.rateints_per_segment, 'times': dataset.times,
          'ints_per_segment': [s.data.shape[0] for s in dataset.rateints_per_segment]}
result = ts.jwst.stage2(inputs, outputfolder='soss_reduction',
                        config=ExtractionConfig(aperture_radius=15, optimal_extraction=True))

order1 = result['orders'][1]               # box extraction
wavelength, flux, error = order1['wavelength'], order1['flux'], order1['error']
optimal = result['optimal']['polynomial']['orders'][1]
```

To start from archive `rateints` files instead, call the SOSS Stage 2 directly:
`ts.soss.extract_soss(rateints_files, aperture_radius=15, outputfolder='soss_products',
optimal_methods=('polynomial',))`. SOSS-specific options (e.g. `background_reference`,
`pastasoss_reference`, `trace_max_shift`, `optimal_options`) go in `ExtractionConfig(soss={...})`.

## Processing steps

0. **Group-level 1/f (default):** `load.fit_ramps()` fits the ramps, runs a box-only
   `extract_soss` on those rates (traces, background, white-light factors), and corrects every
   group with median group-difference templates, masking orders 1, 2 and 3
   (`soss.correct_ramps_1f`); the ramps are then fitted again. This replaces the
   `clean_flicker_noise` step, which runs instead with
   `detector_calibration(group_difference_1f=False)`. On PID 1541 (four segments) it gives
   a lower white-light scatter than group 1/f before linearity, and removes the
   integration-60 column offset. Stage 2 does not remove 1/f noise from these rates again
   (`scale_1f` only applies to rates without the group-level correction). The levels can also
   be measured on the group differences, as for NIRSpec (`domain='differences'`).
1. **Background:** the JWST background subtraction step (`BackgroundStep`, with the CRDS SOSS
   background template) is run on a median rate image after ramp fitting, and the fitted
   static background (zodiacal light) is subtracted from every integration **before** the
   traces are refined and the spectra extracted.
2. **Traces and wavelengths:** traces and wavelength solutions for orders 1-3 come from the
   CRDS PASTASOSS reference at the measured pupil-wheel position, then are refined with a
   double-Gaussian cross-correlation.
3. **Extraction:** a fractional box aperture for each order, with propagated variances.
   Missing apertures and geometric overlap between orders are flagged.
4. **Optional optimal extraction** (`optimal_methods=('polynomial', 'gp')`): one intrinsic
   spatial profile per order and method is trained on a few integrations and reused through
   every integration's trace. By default, overlapping orders are fitted jointly
   (`overlap_strategy='joint'`), recovering the long-wavelength ends of orders 1 and 2 and
   returning the inter-order flux covariance.

Products (FITS, NPZ and CSV for each order) are written to `outputfolder`.

## Validation and limitations

The HAT-P-14 b reduction (PID 1541) and its known limitations are documented in
{doc}`../validation/soss`, {doc}`../validation/soss-report`,
{doc}`../validation/soss-optimal-report` and {doc}`../validation/soss-overlap-report`.
In short: white-light depths agree with published values, but the faint red end of order 2
remains sensitive to the profile model, and order 3 is weak.
