"""Standard data products: spectroscopic and photometric time series, and light curves.

Every reduction path of transitspectroscopy can express its results with three classes:

- :class:`SpectroscopicTimeSeries`: one extracted spectrum per integration (one spectral
  order or one detector), with errors, data-quality flags, wavelengths and traces;
- :class:`PhotometricTimeSeries`: one flux per integration (imaging);
- :class:`LightCurveSet`: white-light or wavelength-binned light curves.

They are written to and read from multi-extension FITS files (time series) and ECSV
tables (light curves), each with a JSON provenance manifest next to it (see
:mod:`transitspectroscopy.core.provenance`). Times are BJD_TDB at mid-integration.
"""

from dataclasses import dataclass, field, fields
import json
from pathlib import Path

import numpy as np

from . import provenance

# Data-quality bits of the extracted products.
DQ_MISSING = 1  # flux or error missing (incomplete aperture, rejected, unsupported)
DQ_BLENDED = 2  # geometric overlap with another trace / detector edge
DQ_FLAGGED = 4  # flagged by the reduction (e.g. photometry quality flags)

_FORMAT_VERSION = 1


def _array(value, dtype=float):
    return None if value is None else np.asarray(value, dtype=dtype)


def _json_ready(value):
    """Recursively convert NumPy values so ``json.dumps`` accepts them."""
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist() if value.size <= 64 else provenance.array_fingerprint(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, 'fingerprint'):  # e.g. a frozen SharedProfile
        return dict(type=type(value).__name__, fingerprint=value.fingerprint())
    return str(value)


@dataclass
class SpectroscopicTimeSeries:
    """Spectra of one trace (order/detector) for every integration.

    Parameters
    ----------
    times : numpy.ndarray
        ``(time,)`` mid-integration BJD_TDB.
    wavelength : numpy.ndarray
        ``(column,)`` wavelength of each extracted column, in microns.
    flux, flux_err : numpy.ndarray
        ``(time, column)`` flux and 1-sigma error.
    dq : numpy.ndarray, optional
        ``(time, column)`` data-quality bits (``DQ_*``). By default, missing values are flagged.
    columns : numpy.ndarray, optional
        ``(column,)`` detector column of each extracted column.
    trace : numpy.ndarray, optional
        Cross-dispersion trace position, ``(column,)`` or ``(time, column)``.
    trace_width : numpy.ndarray, optional
        Profile width, ``(time,)`` or ``(time, column)``.
    covariance : numpy.ndarray, optional
        Extra covariance information (e.g. inter-order covariance), any shape.
    order : int, optional
        Spectral order.
    name : str
        Short name of the product, e.g. ``'order1_box'`` or ``'nrs1_corrected'``.
    flux_unit : str
    metadata : dict
        Free-form, JSON-serializable description (mode, detector, extraction settings...).
    """

    times: np.ndarray
    wavelength: np.ndarray
    flux: np.ndarray
    flux_err: np.ndarray
    dq: np.ndarray = None
    columns: np.ndarray = None
    trace: np.ndarray = None
    trace_width: np.ndarray = None
    covariance: np.ndarray = None
    order: int = None
    name: str = 'spectra'
    flux_unit: str = 'DN/s'
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        self.times = _array(self.times)
        self.flux, self.flux_err = _array(self.flux), _array(self.flux_err)
        self.wavelength = _array(self.wavelength)
        if self.flux.ndim != 2 or self.flux_err.shape != self.flux.shape:
            raise ValueError('flux and flux_err must be matching (time, column) arrays')
        ntimes, ncolumns = self.flux.shape
        if self.times.shape != (ntimes,):
            raise ValueError('times must have one value per integration')
        if self.wavelength.shape not in ((ncolumns,), (ntimes, ncolumns)):
            raise ValueError('wavelength must have shape (column,) or (time, column)')
        self.columns = np.arange(ncolumns) if self.columns is None else np.asarray(self.columns)
        if self.columns.shape != (ncolumns,):
            raise ValueError('columns must have one value per extracted column')
        if self.dq is None:
            missing = ~np.isfinite(self.flux) | ~np.isfinite(self.flux_err)
            self.dq = np.where(missing, DQ_MISSING, 0).astype(np.uint32)
        self.dq = np.asarray(self.dq, dtype=np.uint32)
        if self.dq.shape != self.flux.shape:
            raise ValueError('dq must match flux')
        self.trace, self.trace_width = _array(self.trace), _array(self.trace_width)
        self.covariance = _array(self.covariance)

    @property
    def good(self):
        """Boolean ``(time, column)`` mask of finite, unflagged measurements."""
        return (self.dq & DQ_MISSING) == 0

    def white_light(self, columns=None, name=None):
        """Normalized white-light curve summed over columns valid in every integration.

        ``columns`` (indices into the extracted columns) restricts the band.
        """
        common = np.all(self.good & (self.flux_err > 0), axis=0)
        if columns is not None:
            selected = np.zeros_like(common)
            selected[np.asarray(columns)] = True
            common &= selected
        if not common.any():
            raise ValueError('no column is valid in every integration')
        flux = np.sum(self.flux[:, common], axis=1)
        error = np.sqrt(np.sum(self.flux_err[:, common] ** 2, axis=1))
        normalization = np.median(flux)
        wavelength = (
            self.wavelength if self.wavelength.ndim == 1 else np.median(self.wavelength, axis=0)
        )
        return LightCurveSet(
            times=self.times,
            flux=(flux / normalization)[None],
            flux_err=(error / normalization)[None],
            wavelength_low=[np.min(wavelength[common])],
            wavelength_high=[np.max(wavelength[common])],
            labels=['white'],
            name=name or f'{self.name}_white',
            metadata=dict(
                self.metadata,
                source=self.name,
                n_columns=int(common.sum()),
                normalization=float(normalization),
            ),
        )

    def bin_wavelengths(self, edges, name=None, minimum_columns=1):
        """Normalized spectroscopic light curves in the wavelength bins ``edges``.

        Columns are summed when valid in every integration; bins with fewer than
        ``minimum_columns`` such columns are omitted.
        """
        edges = np.asarray(edges, dtype=float)
        wavelength = (
            self.wavelength if self.wavelength.ndim == 1 else np.median(self.wavelength, axis=0)
        )
        common = np.all(self.good & (self.flux_err > 0), axis=0)
        flux, error, low, high = [], [], [], []
        for left, right in zip(edges[:-1], edges[1:]):
            inside = common & (wavelength >= left) & (wavelength < right)
            if inside.sum() < minimum_columns:
                continue
            total = np.sum(self.flux[:, inside], axis=1)
            normalization = np.median(total)
            flux.append(total / normalization)
            error.append(np.sqrt(np.sum(self.flux_err[:, inside] ** 2, axis=1)) / normalization)
            low.append(left)
            high.append(right)
        if not flux:
            raise ValueError('no wavelength bin contains valid columns')
        return LightCurveSet(
            times=self.times,
            flux=np.array(flux),
            flux_err=np.array(error),
            wavelength_low=low,
            wavelength_high=high,
            labels=[f'{a:.4f}-{b:.4f}' for a, b in zip(low, high)],
            name=name or f'{self.name}_binned',
            metadata=dict(self.metadata, source=self.name),
        )


@dataclass
class PhotometricTimeSeries:
    """One flux per integration (imaging photometry).

    Parameters
    ----------
    times, flux, flux_err : numpy.ndarray
        ``(time,)`` BJD_TDB, flux and 1-sigma error.
    dq : numpy.ndarray, optional
        ``(time,)`` data-quality bits.
    centroid : numpy.ndarray, optional
        ``(time, 2)`` source position (x, y) in detector pixels.
    background : numpy.ndarray, optional
        ``(time,)`` background level per pixel.
    name, flux_unit, metadata
        As for :class:`SpectroscopicTimeSeries`.
    """

    times: np.ndarray
    flux: np.ndarray
    flux_err: np.ndarray
    dq: np.ndarray = None
    centroid: np.ndarray = None
    background: np.ndarray = None
    name: str = 'photometry'
    flux_unit: str = ''
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        self.times, self.flux, self.flux_err = (
            _array(self.times),
            _array(self.flux),
            _array(self.flux_err),
        )
        if not self.times.shape == self.flux.shape == self.flux_err.shape or self.times.ndim != 1:
            raise ValueError('times, flux and flux_err must be matching one-dimensional arrays')
        if self.dq is None:
            missing = ~np.isfinite(self.flux) | ~np.isfinite(self.flux_err)
            self.dq = np.where(missing, DQ_MISSING, 0).astype(np.uint32)
        self.dq = np.asarray(self.dq, dtype=np.uint32)
        self.centroid, self.background = _array(self.centroid), _array(self.background)

    def light_curve(self, name=None):
        """Normalized light curve as a one-curve :class:`LightCurveSet`."""
        normalization = np.nanmedian(self.flux)
        return LightCurveSet(
            times=self.times,
            flux=(self.flux / normalization)[None],
            flux_err=(self.flux_err / normalization)[None],
            labels=[self.name],
            name=name or f'{self.name}_lightcurve',
            metadata=dict(self.metadata, source=self.name, normalization=float(normalization)),
        )


@dataclass
class LightCurveSet:
    """Normalized light curves sharing their times (white light or wavelength bins).

    Parameters
    ----------
    times : numpy.ndarray
        ``(time,)`` BJD_TDB.
    flux, flux_err : numpy.ndarray
        ``(curve, time)`` relative flux and error.
    wavelength_low, wavelength_high : sequence, optional
        Bin edges in microns, one per curve.
    labels : sequence of str, optional
        Names of the curves.
    name : str
    metadata : dict
    """

    times: np.ndarray
    flux: np.ndarray
    flux_err: np.ndarray
    wavelength_low: list = None
    wavelength_high: list = None
    labels: list = None
    name: str = 'lightcurves'
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        self.times = _array(self.times)
        self.flux, self.flux_err = (
            np.atleast_2d(_array(self.flux)),
            np.atleast_2d(_array(self.flux_err)),
        )
        if self.flux.shape != self.flux_err.shape or self.flux.shape[1] != len(self.times):
            raise ValueError('flux and flux_err must be (curve, time) arrays matching times')
        ncurves = len(self.flux)
        self.labels = [str(i) for i in range(ncurves)] if self.labels is None else list(self.labels)
        for name in ('wavelength_low', 'wavelength_high'):
            value = getattr(self, name)
            setattr(self, name, [np.nan] * ncurves if value is None else [float(v) for v in value])
        if not len(self.labels) == len(self.wavelength_low) == len(self.wavelength_high) == ncurves:
            raise ValueError('labels and wavelength edges need one value per curve')

    @property
    def wavelength(self):
        """Bin centres in microns."""
        return (np.asarray(self.wavelength_low) + np.asarray(self.wavelength_high)) / 2

    def as_dictionaries(self):
        """Inputs for ``transitfitting.fit_lightcurves``: ``{label: {times, flux, error}}``."""
        return {
            label: dict(times=self.times, flux=self.flux[i], error=self.flux_err[i])
            for i, label in enumerate(self.labels)
        }


# ---------------------------------------------------------------------------------------
# Input/output


def _header_provenance(header, manifest):
    if manifest is None:
        return
    software = manifest.get('software', {})
    git = manifest.get('git') or {}
    header['TS_VER'] = (str(software.get('transitspectroscopy', '')), 'package version')
    header['TS_GIT'] = (str(git.get('commit', ''))[:40], 'git commit')
    header['TS_DIRTY'] = (bool(git.get('dirty', False)), 'uncommitted changes')
    header['TS_CFG'] = (str(manifest.get('config_sha256', ''))[:16], 'config SHA-256 prefix')
    if manifest.get('crds_context'):
        header['CRDS_CTX'] = (str(manifest['crds_context']), 'CRDS context')


def _metadata_hdu(metadata):
    from astropy.io import fits

    text = json.dumps(_json_ready(metadata), sort_keys=True)
    column = fits.Column(name='JSON', format=f'{max(1, len(text))}A', array=np.array([text]))
    return fits.BinTableHDU.from_columns([column], name='METADATA')


def _time_hdus(times):
    from astropy.io import fits

    hdu = fits.ImageHDU(np.asarray(times, float), name='TIME')
    hdu.header['BUNIT'] = 'd'
    hdu.header['TIMESYS'] = 'TDB'
    hdu.header['TREFPOS'] = 'BARYCENTER'
    hdu.header['TIMEDEL'] = 'BJD_TDB at mid-integration'
    return [hdu]


def write_fits(product, path, manifest=None):
    """Write a :class:`SpectroscopicTimeSeries` or :class:`PhotometricTimeSeries` to FITS.

    The provenance ``manifest`` (if given) is summarized in the primary header and
    written in full to ``<path>.provenance.json``.
    """
    from astropy.io import fits

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    primary = fits.PrimaryHDU()
    header = primary.header
    header['TS_FMT'] = (_FORMAT_VERSION, 'transitspectroscopy product format version')
    header['TS_NAME'] = product.name
    header['BUNIT'] = product.flux_unit
    _header_provenance(header, manifest)
    hdus = [primary]
    if isinstance(product, SpectroscopicTimeSeries):
        header['TS_KIND'] = 'spectroscopic'
        if product.order is not None:
            header['SPORDER'] = int(product.order)
        arrays = [
            ('FLUX', product.flux, product.flux_unit),
            ('FLUX_ERR', product.flux_err, product.flux_unit),
            ('DQ', product.dq, None),
            ('WAVELENGTH', product.wavelength, 'um'),
            ('COLUMNS', product.columns, None),
            ('TRACE', product.trace, 'pixel'),
            ('TRACE_WIDTH', product.trace_width, 'pixel'),
            ('COVARIANCE', product.covariance, None),
        ]
    elif isinstance(product, PhotometricTimeSeries):
        header['TS_KIND'] = 'photometric'
        arrays = [
            ('FLUX', product.flux, product.flux_unit),
            ('FLUX_ERR', product.flux_err, product.flux_unit),
            ('DQ', product.dq, None),
            ('CENTROID', product.centroid, 'pixel'),
            ('BACKGROUND', product.background, product.flux_unit),
        ]
    else:
        raise TypeError('write_fits takes a SpectroscopicTimeSeries or PhotometricTimeSeries')
    hdus += _time_hdus(product.times)
    for name, array, unit in arrays:
        if array is None:
            continue
        hdu = fits.ImageHDU(np.asarray(array), name=name)
        if unit:
            hdu.header['BUNIT'] = unit
        hdus.append(hdu)
    hdus.append(_metadata_hdu(product.metadata))
    fits.HDUList(hdus).writeto(path, overwrite=True, checksum=True)
    if manifest is not None:
        provenance.write_manifest(manifest, provenance.manifest_path(path))
    return path


def read_fits(path, strict=False):
    """Read a product written by :func:`write_fits`.

    If a provenance manifest is next to the file, it is compared with the running
    software: a difference only warns, unless ``strict=True``.
    """
    from astropy.io import fits

    path = Path(path)
    manifest_file = provenance.manifest_path(path)
    if manifest_file.exists():
        provenance.check_manifest(
            provenance.read_manifest(manifest_file), strict=strict, what=str(path)
        )
    with fits.open(path) as hdus:
        header = hdus[0].header
        if header.get('TS_FMT', 0) > _FORMAT_VERSION:
            raise ValueError(f'{path} was written by a newer transitspectroscopy format')
        get = {hdu.name: hdu.data.copy() for hdu in hdus[1:] if hdu.name != 'METADATA'}
        metadata = json.loads(str(hdus['METADATA'].data['JSON'][0]))
        kind = header['TS_KIND']
        common = dict(
            times=get['TIME'],
            flux=get['FLUX'],
            flux_err=get['FLUX_ERR'],
            dq=get['DQ'],
            name=header['TS_NAME'],
            flux_unit=header.get('BUNIT', ''),
            metadata=metadata,
        )
        if kind == 'spectroscopic':
            return SpectroscopicTimeSeries(
                wavelength=get['WAVELENGTH'],
                columns=get.get('COLUMNS'),
                trace=get.get('TRACE'),
                trace_width=get.get('TRACE_WIDTH'),
                covariance=get.get('COVARIANCE'),
                order=header.get('SPORDER'),
                **common,
            )
        return PhotometricTimeSeries(
            centroid=get.get('CENTROID'), background=get.get('BACKGROUND'), **common
        )


def write_ecsv(lightcurves, path, manifest=None):
    """Write a :class:`LightCurveSet` as an ECSV table (one row per integration)."""
    from astropy.table import Table

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    table = Table()
    table['time_bjd_tdb'] = lightcurves.times
    for i, label in enumerate(lightcurves.labels):
        table[f'flux_{i}'] = lightcurves.flux[i]
        table[f'flux_err_{i}'] = lightcurves.flux_err[i]
    table.meta = _json_ready(
        dict(
            name=lightcurves.name,
            labels=lightcurves.labels,
            wavelength_low=lightcurves.wavelength_low,
            wavelength_high=lightcurves.wavelength_high,
            metadata=lightcurves.metadata,
            format_version=_FORMAT_VERSION,
        )
    )
    table.write(path, format='ascii.ecsv', overwrite=True)
    if manifest is not None:
        provenance.write_manifest(manifest, provenance.manifest_path(path))
    return path


def read_ecsv(path, strict=False):
    """Read a :class:`LightCurveSet` written by :func:`write_ecsv`."""
    from astropy.table import Table

    path = Path(path)
    manifest_file = provenance.manifest_path(path)
    if manifest_file.exists():
        provenance.check_manifest(
            provenance.read_manifest(manifest_file), strict=strict, what=str(path)
        )
    table = Table.read(path, format='ascii.ecsv')
    meta = table.meta
    ncurves = len(meta['labels'])
    return LightCurveSet(
        times=np.asarray(table['time_bjd_tdb']),
        flux=np.array([table[f'flux_{i}'] for i in range(ncurves)]),
        flux_err=np.array([table[f'flux_err_{i}'] for i in range(ncurves)]),
        wavelength_low=[np.nan if v is None else v for v in meta['wavelength_low']],
        wavelength_high=[np.nan if v is None else v for v in meta['wavelength_high']],
        labels=meta['labels'],
        name=meta['name'],
        metadata=meta.get('metadata', {}),
    )


def write_products(products, folder, manifest=None):
    """Write a ``{name: product}`` dictionary to ``folder`` (FITS or ECSV per product).

    Returns ``{name: path}``.
    """
    folder = Path(folder)
    paths = {}
    for name, product in products.items():
        if isinstance(product, LightCurveSet):
            paths[name] = write_ecsv(product, folder / f'{name}.ecsv', manifest)
        else:
            paths[name] = write_fits(product, folder / f'{name}.fits', manifest)
    return paths


def read_products(folder, strict=False):
    """Read every product written by :func:`write_products` in ``folder``."""
    folder = Path(folder)
    products = {}
    for path in sorted(folder.glob('*.fits')):
        products[path.stem] = read_fits(path, strict=strict)
    for path in sorted(folder.glob('*.ecsv')):
        products[path.stem] = read_ecsv(path, strict=strict)
    return products


def resolution_edges(wavelength_min, wavelength_max, resolving_power):
    """Wavelength bin edges of constant resolving power ``R = lambda/delta_lambda``."""
    if not 0 < wavelength_min < wavelength_max or resolving_power <= 0:
        raise ValueError('need 0 < wavelength_min < wavelength_max and a positive resolving power')
    edges = [float(wavelength_min)]
    while edges[-1] < wavelength_max:
        edges.append(edges[-1] * (2 * resolving_power + 1) / (2 * resolving_power - 1))
    return np.array(edges)


def product_fields(product):
    """Names of the dataclass fields of a product (for documentation and tests)."""
    return [f.name for f in fields(product)]
