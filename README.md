# TransitSpectroscopy 🪐 --- a package for all your transit spectroscopy needs
-------------------------------------------------------------------------------

`transitspectroscopy` is a package containing algorithms and wrappers useful for performing transit (transmission and emission) spectroscopy of exoplanetary systems.

**Author**: Nestor Espinoza (nespinoza@stsci.edu)

## Statement of need
I might fill this in later, but basically: I needed a package to make available all the tools useful for transit spectroscopy: from tracing, spectral extraction to actual lightcurve fitting of wavelength-dependant transits.

## Installation
Installation is as simple as:

        python setup.py install

Or via PyPi:

        pip install transitspectroscopy

Note you need to have the [GNU Scientific Library (GSL)](https://www.gnu.org/software/gsl/) installed for these scripts to successfully install `transitspectroscopy` on your machine. In MacOS, installing the GSL library is as easy as doing `brew install gsl`.

## Licence and attribution

Read the `LICENCE` file for licencing details on how to use the code.

## Citation

You can cite the code via it's DOI: 10.5281/zenodo.6960923 (see https://zenodo.org/record/6960924#.YutMaezMLUI). Here's the citation snippet:

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
