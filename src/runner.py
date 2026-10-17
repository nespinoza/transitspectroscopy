"""Configuration-driven reductions: ``ts.run(config)`` and ``transitspectroscopy run config.toml``.

A run configuration (a TOML file or a dictionary) describes a whole reduction: where
the data are, how to calibrate them, one or more extraction variants, and which light
curves to make. ``run`` executes the steps

    data -> calibrate -> extract -> lightcurves [-> fit]

and writes every step to its own folder ``<output>/<step>-<hash>``, where the hash
covers the step's configuration and all upstream steps. Rerunning an unchanged
configuration reuses completed steps; changing a setting recomputes that step and the
ones after it. ``force=True`` recomputes everything, ``from_step='extract'`` recomputes
from that step on. Every product is written with a provenance manifest.

Configuration sections (all optional except ``data``)::

    [run]          name, output (folder; default 'runs/<name>')
    [data]         source = 'mast' (program, observation, exposure, segments, detectors,
                   product, folder, mast_api_token) or 'local' (glob, product)
    [mode]         name = 'auto' or a registered mode name (see ts.instruments.list_modes())
    [calibration]  parameters, use_tso_jump, group_1f, ramp_fit_method, ramp_fit_options,
                   ramp_fit_parameters, crds_context (pins the CRDS context, e.g. 'jwst_1584.pmap')
    [extraction.<variant>]  ExtractionConfig fields (one table per extraction variant)
    [execution]    nthreads
    [lightcurves]  white_light = true, resolving_power (binned spectroscopic light curves)
    [fit]          sampler, priors (juliet priors applied to every light curve), options
    [validation]   free-form settings read by the validation scripts (ignored by ``run``)

Files of several exposures (e.g. a MIRI visit made of three exposures) are loaded
exposure by exposure and passed together to Stage 2 in chronological order.
"""

from dataclasses import dataclass, field
import glob
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np

from .core import products as _products
from .core import provenance

STEPS = ('data', 'calibrate', 'extract', 'lightcurves', 'fit')
SECTIONS = (
    'run',
    'data',
    'mode',
    'calibration',
    'extraction',
    'execution',
    'lightcurves',
    'fit',
    'validation',
)


def load_config(config):
    """Read a run configuration from a TOML file, or validate a dictionary.

    Returns a new dictionary with the defaults filled in.
    """
    if isinstance(config, (str, Path)):
        import tomllib

        path = Path(config)
        with path.open('rb') as handle:
            config = tomllib.load(handle)
        config.setdefault('run', {}).setdefault('config_file', str(path.resolve()))
    config = json.loads(json.dumps(config))  # Deep copy with JSON types only.
    unknown = set(config) - set(SECTIONS)
    if unknown:
        raise ValueError(f'Unknown configuration sections: {sorted(unknown)}')
    if 'data' not in config:
        raise ValueError('A run configuration needs a [data] section')
    run = config.setdefault('run', {})
    run.setdefault('name', 'reduction')
    run.setdefault('output', str(Path('runs') / run['name']))
    data = config['data']
    data.setdefault('source', 'local')
    data.setdefault('product', 'uncal')
    if data['source'] not in ('mast', 'local'):
        raise ValueError("data.source must be 'mast' or 'local'")
    if data['product'] not in ('uncal', 'rateints'):
        raise ValueError("data.product must be 'uncal' or 'rateints'")
    if data['source'] == 'local' and 'glob' not in data:
        raise ValueError('data.source = "local" needs data.glob')
    config.setdefault('mode', {}).setdefault('name', 'auto')
    config.setdefault('calibration', {})
    extraction = config.setdefault('extraction', {'default': {}})
    if not extraction or not all(isinstance(value, dict) for value in extraction.values()):
        raise ValueError('[extraction] must contain named tables, e.g. [extraction.box]')
    config.setdefault('execution', {})
    config.setdefault('lightcurves', {}).setdefault('white_light', True)
    return config


@dataclass
class RunResult:
    """Outcome of :func:`run`."""

    config: dict
    output: Path
    files: list = field(default_factory=list)
    products: dict = field(default_factory=dict)  # {variant: {name: product}}
    lightcurves: dict = field(default_factory=dict)  # {variant: {name: LightCurveSet}}
    fits: dict = field(default_factory=dict)  # {variant: fit results}
    steps: dict = field(default_factory=dict)  # {step: folder}
    reused: list = field(default_factory=list)  # steps reused from earlier runs
    datasets: list = field(default_factory=list)  # one ts.jwst.load per exposure
    stage2: dict = field(default_factory=dict)  # {variant: raw stage2 result}, if kept

    @property
    def dataset(self):
        """The (first) loaded exposure."""
        return self.datasets[0] if self.datasets else None


def _section_hash(config, step, upstream):
    sections = dict(
        data=('data',),
        calibrate=('mode', 'calibration'),
        extract=('extraction',),
        lightcurves=('lightcurves',),
        fit=('fit',),
    )[step]
    content = {name: config.get(name) for name in sections}
    if 'data' in content:
        # Credentials never enter hashes or manifests.
        content['data'] = {k: v for k, v in content['data'].items() if k != 'mast_api_token'}
    return provenance.config_sha256(dict(step=step, content=content, upstream=upstream))[:12]


class _Step:
    """Folder, completion marker and manifest of one step."""

    def __init__(self, output, name, digest, recompute):
        self.name, self.digest = name, digest
        self.folder = Path(output) / f'{name}-{digest}'
        self.marker = self.folder / 'step.json'
        if recompute and self.folder.exists():
            shutil.rmtree(self.folder)
        self.cached = self.marker.exists()
        self.folder.mkdir(parents=True, exist_ok=True)

    def complete(self, manifest, **record):
        provenance.write_manifest(manifest, self.folder / 'manifest.json')
        self.marker.write_text(
            json.dumps(
                dict(
                    step=self.name,
                    hash=self.digest,
                    status='complete',
                    completed_utc=manifest['created_utc'],
                    **record,
                ),
                indent=2,
                default=str,
            )
            + '\n'
        )


def _resolve_files(config):
    data = config['data']
    if data['source'] == 'local':
        files = sorted(glob.glob(data['glob']))
        if not files:
            raise FileNotFoundError(f"No files match {data['glob']}")
        return files
    from . import jwst

    files = jwst.download(
        data['program'],
        str(data['observation']),
        mast_api_token=data.get('mast_api_token'),
        outputfolder=data.get('folder', 'JWSTdata'),
        data_product=data['product'],
        exposure=data.get('exposure'),
        segments=data.get('segments'),
        detectors=data.get('detectors'),
    )
    return sorted(files)


def _exposures(files):
    """Group segment files by exposure (the file-name part before ``-seg``)."""
    groups = {}
    for name in files:
        groups.setdefault(Path(name).name.split('-seg')[0], []).append(name)
    return list(groups.values())


def _stage2_inputs(datasets):
    """Stage 2 inputs from one or more loaded exposures, in chronological order."""
    datasets = sorted(datasets, key=lambda dataset: dataset.times[0])
    rates = [segment for dataset in datasets for segment in dataset.rateints_per_segment]
    return {
        'rampstep': rates,
        'times': np.concatenate([dataset.times for dataset in datasets]),
        'ints_per_segment': [segment.data.shape[0] for segment in rates],
    }


def _lightcurves(products, settings):
    curves = {}
    for name, product in products.items():
        if isinstance(product, _products.SpectroscopicTimeSeries):
            if settings.get('white_light', True):
                curves[f'{name}_white'] = product.white_light()
            if settings.get('resolving_power'):
                wavelength = (
                    product.wavelength if product.wavelength.ndim == 1 else product.wavelength[0]
                )
                valid = wavelength[product.good.all(axis=0) & (wavelength > 0)]
                if len(valid) > 1:
                    edges = _products.resolution_edges(
                        valid.min(), valid.max(), settings['resolving_power']
                    )
                    curves[f'{name}_R{settings["resolving_power"]:g}'] = product.bin_wavelengths(
                        edges
                    )
        elif isinstance(product, _products.PhotometricTimeSeries):
            curves[f'{name}_lightcurve'] = product.light_curve()
    return curves


def run(config, force=False, from_step=None, hash_inputs=True, keep_stage2=False):
    """Run a reduction described by a configuration (TOML path or dictionary).

    Parameters
    ----------
    config : str, pathlib.Path or dict
    force : bool
        Recompute every step.
    from_step : {'data', 'calibrate', 'extract', 'lightcurves', 'fit'}, optional
        Recompute this step and the following ones.
    hash_inputs : bool
        Record the SHA-256 of every input file in the manifests (slow for large data).
    keep_stage2 : bool
        Keep the raw ``stage2`` results in ``RunResult.stage2`` (for diagnostics). A cached
        extraction step is then recomputed, since raw results are not stored on disk.

    Returns
    -------
    RunResult
    """
    from . import instruments, jwst
    from .config import ExtractionConfig

    config = load_config(config)
    if from_step is not None and from_step not in STEPS:
        raise ValueError(f'from_step must be one of {STEPS}')
    first_recomputed = 0 if force else (STEPS.index(from_step) if from_step else len(STEPS))
    if keep_stage2:
        first_recomputed = min(first_recomputed, STEPS.index('extract'))
    output = Path(config['run']['output'])
    output.mkdir(parents=True, exist_ok=True)
    provenance.write_manifest(dict(config=config), output / 'config.resolved.json')
    result = RunResult(config=config, output=output)
    started = time.time()

    def step(name, upstream):
        digest = _section_hash(config, name, upstream)
        current = _Step(output, name, digest, STEPS.index(name) >= first_recomputed)
        result.steps[name] = current.folder
        if current.cached:
            result.reused.append(name)
            print(f'\t >> Step {name}: reusing {current.folder}')
        else:
            print(f'\t >> Step {name}: running in {current.folder}')
        return current

    # 1. Data: input files, downloaded if needed. Always resolved, so later cached steps
    # still have their inputs.
    data_step = step('data', None)
    result.files = _resolve_files(config)
    if not data_step.cached:
        data = {k: v for k, v in config['data'].items() if k != 'mast_api_token'}
        data_step.complete(
            provenance.build_manifest(
                config=dict(data=data), inputs=result.files, hash_inputs=hash_inputs, step='data'
            ),
            files=result.files,
        )
    input_records = provenance.read_manifest(data_step.folder / 'manifest.json')['inputs']

    # 2. Calibration: detector-level steps and ramp fitting (ts.jwst.load), one exposure at
    # a time. Rerunning in the same folder reuses jwst.load's saved step products.
    calibrate = step('calibrate', data_step.digest)
    calibration = config['calibration']
    if calibration.get('crds_context'):
        os.environ['CRDS_CONTEXT'] = calibration['crds_context']
    mode_name = config['mode']['name']
    mode = None if mode_name == 'auto' else instruments.get_mode(mode_name)
    for number, files in enumerate(_exposures(result.files)):
        exposure_folder = calibrate.folder / f'exposure{number + 1}'
        exposure_folder.mkdir(parents=True, exist_ok=True)
        dataset = jwst.load(
            files,
            outputfolder=str(calibrate.folder / f'exposure{number + 1}'),
            ramp_fit_method=calibration.get('ramp_fit_method', 'jwst'),
            ramp_fit_options=calibration.get('ramp_fit_options') or None,
        )
        if mode is not None and dataset.mode != mode.name:
            dataset.mode_definition, dataset.mode = mode, mode.name
        if config['data']['product'] == 'uncal':
            dataset.detector_calibration(
                parameters=calibration.get('parameters', {}),
                use_tso_jump=calibration.get('use_tso_jump', True),
                group_1f=calibration.get('group_1f', False),
            )
            dataset.fit_ramps(parameters=calibration.get('ramp_fit_parameters', {}))
        result.datasets.append(dataset)
    reference = result.dataset.rateints_per_segment[-1]
    crds_context = calibration.get('crds_context') or getattr(
        getattr(getattr(reference.meta, 'ref_file', None), 'crds', None), 'context_used', None
    )
    if not calibrate.cached:
        calibrate.complete(
            provenance.build_manifest(
                config={k: config[k] for k in ('mode', 'calibration')},
                crds_context=crds_context,
                step='calibrate',
                mode=result.dataset.mode,
                input_files=input_records,
            ),
            mode=result.dataset.mode,
            exposures=len(result.datasets),
        )

    # 3. Extraction: one Stage 2 per variant; products are written with their manifests.
    extract = step('extract', calibrate.digest)
    for variant, settings in config['extraction'].items():
        folder = extract.folder / variant
        if extract.cached:
            result.products[variant] = _products.read_products(folder / 'products')
            continue
        folder.mkdir(parents=True, exist_ok=True)
        extraction = ExtractionConfig(**settings)
        stage2_result = jwst.stage2(
            _stage2_inputs(result.datasets),
            outputfolder=str(folder),
            nthreads=config['execution'].get('nthreads'),
            config=extraction,
            mode=mode,
        )
        manifest = provenance.build_manifest(
            config=dict(
                variant=variant,
                extraction=extraction.as_dict(),
                calibration=config['calibration'],
                mode=config['mode'],
            ),
            crds_context=crds_context,
            step='extract',
            mode=result.dataset.mode,
            input_files=input_records,
        )
        result.products[variant] = stage2_result['products']
        if keep_stage2:
            result.stage2[variant] = stage2_result
        _products.write_products(stage2_result['products'], folder / 'products', manifest)
    if not extract.cached:
        extract.complete(
            provenance.build_manifest(config=config['extraction'], step='extract'),
            variants=list(config['extraction']),
        )

    # 4. Light curves: white light and (optionally) binned at a resolving power.
    lightcurves = step('lightcurves', extract.digest)
    settings = config['lightcurves']
    for variant, products in result.products.items():
        folder = lightcurves.folder / variant
        if lightcurves.cached:
            result.lightcurves[variant] = _products.read_products(folder)
            continue
        curves = _lightcurves(products, settings)
        _products.write_products(
            curves,
            folder,
            provenance.build_manifest(
                config=dict(lightcurves=settings, variant=variant), step='lightcurves'
            ),
        )
        result.lightcurves[variant] = curves
    if not lightcurves.cached:
        lightcurves.complete(provenance.build_manifest(config=settings, step='lightcurves'))

    # 5. Optional light-curve fits (juliet).
    if config.get('fit'):
        fit = step('fit', lightcurves.digest)
        if not fit.cached:
            from . import transitfitting

            options = dict(config['fit'])
            priors, sampler = options.pop('priors'), options.pop('sampler', 'dynesty')
            for variant, curves in result.lightcurves.items():
                data = {}
                for name, curve in curves.items():
                    for label, values in curve.as_dictionaries().items():
                        data[str(fit.folder / variant / name / label)] = values
                result.fits[variant] = transitfitting.fit_lightcurves(
                    data, {key: priors for key in data}, sampler=sampler, **options
                )
            fit.complete(provenance.build_manifest(config=config['fit'], step='fit'))
    provenance.write_manifest(
        dict(
            steps={k: str(v) for k, v in result.steps.items()},
            reused=result.reused,
            seconds=time.time() - started,
        ),
        output / 'run.json',
    )
    return result
