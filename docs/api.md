# API reference

The most common functions are also available at the top level (`ts.getOptimalSpectrum`,
`ts.load`, `ts.fit_shared_profile`, ...). The instrument-independent algorithms are in
`transitspectroscopy.core` (see [Core algorithms](#core-algorithms) below); the modules
above it keep the historical interface and add the JWST-specific layers.

## JWST data access, calibration and stages

```{eval-rst}
.. automodule:: transitspectroscopy.jwst
   :members: download, load, stage1, stage2, side_refpix_correction, group_1f_correction,
             tso_jumpstep, get_cds
```

## Stage 2 configuration and observing modes

```{eval-rst}
.. automodule:: transitspectroscopy.config
   :members: ExtractionConfig

.. automodule:: transitspectroscopy.instruments.modes
   :members: Mode, register, unregister, get_mode, list_modes, identify, identify_model, UnknownModeError
```

## Spectral tracing and extraction

```{eval-rst}
.. automodule:: transitspectroscopy.spectroscopy
   :members: getP, getOptimalSpectrum, getFastSimpleSpectrum, getSimpleSpectrum
```

## NIRISS/SOSS

```{eval-rst}
.. automodule:: transitspectroscopy.soss
   :members: extract_soss, reduce_soss, reference_traces, extract_box, save_products, load_products
```

## MIRI imaging and photometry

```{eval-rst}
.. automodule:: transitspectroscopy.miri
   :members: extract_photometry, reduce_miri, calibrate, prepare_segment, extract_visit

```

## Ramp fitting

```{eval-rst}
.. automodule:: transitspectroscopy.ramp_fitting
   :members: fit_model, detector1_class
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

```

## Core algorithms

```{eval-rst}
.. automodule:: transitspectroscopy.core

.. automodule:: transitspectroscopy.core.extraction
   :members:

.. automodule:: transitspectroscopy.core.profiles
   :members:

.. automodule:: transitspectroscopy.core.shared_profile
   :members: SharedProfile, fit_shared_profile, fit_coefficients, profile_basis, pixel_profile

.. automodule:: transitspectroscopy.core.overlap
   :members: train_joint, extract_blended, unmodeled_pixels, JointProfileFit

.. automodule:: transitspectroscopy.core.tracing
   :members:

.. automodule:: transitspectroscopy.core.geometry
   :members:

.. automodule:: transitspectroscopy.core.kernels
   :members:

.. automodule:: transitspectroscopy.core.photometry
   :members:

.. automodule:: transitspectroscopy.core.ramps
   :members: fit_ramps, difference_weights, validate_selection

.. automodule:: transitspectroscopy.core.detector
   :members:

.. automodule:: transitspectroscopy.core.timeseries
   :members:

.. automodule:: transitspectroscopy.core.binning
   :members:

.. automodule:: transitspectroscopy.core.provenance
   :members:
```
