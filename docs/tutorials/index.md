# Tutorials

## Synthetic data (run in a few seconds)

These notebooks simulate their own data and are executed every time the documentation is
built.

```{toctree}
:maxdepth: 1

optimal-extraction
shared-profile
ramp-fitting
```

## Real JWST data

These notebooks download public JWST data and need CRDS reference files, so they are shown
without being executed.

```{toctree}
:maxdepth: 1

realdata-soss-hatp14b
realdata-miri-ltt1445a
```

## Original notebooks

The original, fully worked notebooks are in the repository's
[`tutorials/`](https://github.com/nespinoza/transitspectroscopy/tree/master/tutorials) folder:

- [Tutorial-HAT-P-14b-JWST-NIRSPEC](https://github.com/nespinoza/transitspectroscopy/blob/master/tutorials/Tutorial-HAT-P-14b-JWST-NIRSPEC.ipynb):
  a step-by-step NIRSpec/G395H reduction, from correlated double sampling to spectra.
- [Quicklook-HAT-P-14b-JWST-NIRSPEC-CURRENT](https://github.com/nespinoza/transitspectroscopy/blob/master/tutorials/Quicklook-HAT-P-14b-JWST-NIRSPEC-CURRENT.ipynb):
  a quick white-light curve with `ts.jwst.load` and `ts.jwst.stage2`.
- [WhiteLightFitting-HAT-P-14b-JWST-NIRSPEC-CURRENT](https://github.com/nespinoza/transitspectroscopy/blob/master/tutorials/WhiteLightFitting-HAT-P-14b-JWST-NIRSPEC-CURRENT.ipynb):
  fitting the white-light curve with juliet.
- [Tutorial-TRAPPIST-1b-JWST-MIRI](https://github.com/nespinoza/transitspectroscopy/blob/master/tutorials/Tutorial-TRAPPIST-1b-JWST-MIRI.ipynb):
  MIRI imaging of TRAPPIST-1 b.
