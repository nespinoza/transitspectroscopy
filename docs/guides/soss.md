# NIRISS/SOSS

NIRISS single-object slitless spectroscopy (`CLEAR/GR700XD`) disperses three orders on the
detector. `ts.soss.reduce_soss` reduces ramp-fitted `rateints` segments of one exposure
into spectra and white-light curves for orders 1, 2 and 3:

```python
import glob
import transitspectroscopy as ts

files = sorted(glob.glob('data/jw01541001001_04101_00001-seg00?_nis_rateints.fits'))
result = ts.soss.reduce_soss(files, aperture_radius=15, outputfolder='soss_products',
                             optimal_methods=('polynomial',))

order1 = result['orders'][1]               # box extraction
wavelength, flux, error = order1['wavelength'], order1['flux'], order1['error']
optimal = result['optimal']['polynomial']['orders'][1]
```

The same reduction runs through `ts.jwst.stage2` when it is given SOSS data.

## Processing steps

1. **Background:** the JWST SOSS background step (`BackgroundStep`) is run on a median rate
   image after ramp fitting, and the fitted static background (zodiacal light) is subtracted
   from every integration.
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
