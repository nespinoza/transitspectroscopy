# transitspectroscopy

`transitspectroscopy` is a package containing algorithms and wrappers useful for performing
transit (transmission and emission) spectroscopy of exoplanetary systems: from detector-level
calibration of JWST time-series observations, through tracing and spectral extraction, to
fitting wavelength-dependent light curves.

It is built around the [JWST calibration pipeline](https://github.com/spacetelescope/jwst),
which it uses for standard calibration steps, and adds detector-level methods developed for
time-series observations: a time-series jump detector, group-level 1/f corrections,
reference-pixel pedestals, background models, correlated-read ramp fitting, and optimal
extraction with polynomial or Gaussian-process spatial profiles.

**Author:** Nestor Espinoza (nespinoza@stsci.edu)

```{toctree}
:maxdepth: 2
:caption: Getting started

installation
quickstart
concepts
```

```{toctree}
:maxdepth: 2
:caption: User guides

guides/index
tutorials/index
```

```{toctree}
:maxdepth: 2
:caption: Reference

api
validation/index
changelog
```

## Citation

You can cite this software via its DOI,
[10.5281/zenodo.6960924](https://doi.org/10.5281/zenodo.6960924).
