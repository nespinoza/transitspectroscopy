# API reference

The most common functions are also available at the top level (`ts.getOptimalSpectrum`,
`ts.load`, `ts.fit_shared_profile`, ...).

## JWST data access, calibration and stages

```{eval-rst}
.. automodule:: transitspectroscopy.jwst
   :members: download, load, stage1, stage2, side_refpix_correction, group_1f_correction,
             tso_jumpstep, get_loom, get_roeba, get_cds, get_last_minus_first,
             get_uniluminated_mask, cc_uniluminated_outliers, spill_filter, correct_1f
```

## Spectral tracing and extraction

```{eval-rst}
.. automodule:: transitspectroscopy.spectroscopy
   :members: getP, getOptimalSpectrum, getFastSimpleSpectrum, getSimpleSpectrum,
             trace_spectrum, trace_fwhm, get_fwhm, get_ccf, get_ccf_convolve, get_pm_ccf,
             gaussian, double_gaussian, get_mad_sigma
```

## Shared intrinsic profiles

```{eval-rst}
.. automodule:: transitspectroscopy.shared_profile
   :members: fit_shared_profile, SharedProfile
```

## NIRISS/SOSS

```{eval-rst}
.. automodule:: transitspectroscopy.soss
   :members: reduce_soss, reference_traces, refine_trace, extract_box, save_products, load_products
```

## MIRI imaging and photometry

```{eval-rst}
.. automodule:: transitspectroscopy.miri
   :members: reduce_miri, calibrate, prepare_segment, extract_visit

.. automodule:: transitspectroscopy.photometry
   :members: circular_weights, background, centroid, optimal_flux, GPPSF, fit_gp_psf
```

## Ramp fitting

```{eval-rst}
.. automodule:: transitspectroscopy.ramp_fitting
   :members: fit_ramps, fit_model
```

## Light-curve fitting

```{eval-rst}
.. automodule:: transitspectroscopy.transitfitting
   :members: fit_lightcurves, notremote_fit_data
```

## Utilities

```{eval-rst}
.. automodule:: transitspectroscopy.utils
   :members:

.. automodule:: transitspectroscopy.timeseries
   :members:
```
