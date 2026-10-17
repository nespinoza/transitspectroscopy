"""NIRISS/SOSS extraction from ramp-fitted integrations.

PASTASOSS supplies detector coordinates and wavelengths; bounded double-Gaussian
correlation refines the trace. Box apertures retain geometric overlap flags;
optimal extraction jointly models overlapping order contributions.
"""
from pathlib import Path
import json
import warnings

import numpy as np

from .core.extraction import fractional_box
from .core.geometry import aperture_weights  # noqa: F401  (public re-export)
from .core.provenance import file_sha256
from .core.tracing import refine_trace, trace_motion  # noqa: F401  (public re-exports)
from .jwst_models import integration_times, open_model


def _reference(model, kind, override):
    if override is not None:
        path = str(Path(override).resolve())
    else:
        from jwst.extract_1d import Extract1dStep
        from jwst.background import BackgroundStep
        step = Extract1dStep() if kind == 'pastasoss' else BackgroundStep()
        path = step.get_reference_file(model, kind)
    if path == 'N/A' or not Path(path).is_file():
        raise ValueError(f'No usable {kind} reference for this exposure')
    return path


def reference_traces(model, reference=None, orders=(1, 2, 3)):
    """Evaluate a CRDS-selected PASTASOSS model at the science PWCPOS.

    Coordinates are zero-based in the science SUBSTRIP256 or SUBSTRIP96 image;
    wavelengths are microns. Unsupported/missing orders fail explicitly.
    """
    from stdatamodels.jwst.datamodels import PastasossModel
    from jwst.extract_1d.soss_extract.pastasoss import get_soss_traces
    if model.meta.exposure.type != 'NIS_SOSS':
        raise ValueError('Expected NIS_SOSS data')
    subarray = model.meta.subarray.name
    if subarray not in ('SUBSTRIP256', 'SUBSTRIP96'):
        raise ValueError('SOSS tracing supports SUBSTRIP256 and SUBSTRIP96')
    angle = model.meta.instrument.pupil_position
    if angle is None or not np.isfinite(angle):
        raise ValueError('A finite science PWCPOS is required')
    path = _reference(model, 'pastasoss', reference)
    result = {}
    with PastasossModel(path) as ref:
        available = {int(t.spectral_order) for t in ref.traces}
        if not set(orders) <= available:
            raise ValueError(f'PASTASOSS lacks requested orders: {set(orders)-available}')
        for order in orders:
            _, x, y, wave = get_soss_traces(angle, order, subarray=subarray, refmodel=ref)
            good = (np.isfinite(x) & np.isfinite(y) & np.isfinite(wave) & (wave > 0)
                    & (x >= 0) & (x < model.data.shape[-1])
                    & (y >= 0) & (y < model.data.shape[-2]))
            x, y, wave = x[good], y[good], wave[good]
            if len(x) < 5 or not np.allclose(x, np.rint(x)) or np.any(np.diff(x) <= 0):
                raise ValueError(f'Order {order} has insufficient or invalid detector support')
            result[int(order)] = dict(columns=x.astype(int), reference_trace=y,
                                     wavelength=wave)
    return result, dict(path=path, sha256=file_sha256(path), pupil_position=float(angle),
                       subarray=subarray)


def extract_box(data, variance, valid, columns, centers, radius):
    """Return flux/error; incomplete or masked apertures remain missing.

    See :func:`transitspectroscopy.core.extraction.fractional_box`.
    """
    return fractional_box(data, variance, valid, columns, centers, radius)


def extract_soss(inputs, *, pastasoss_reference=None, background_reference=None,
                background_options=None, aperture_radius=15., trace_max_shift=4.,
                sample_integrations=32, mask_dq=False, scale_1f=True,
                single_trace_extraction=True, outputfolder=None,
                optimal_methods=(), optimal_options=None):
    """Stage 2 of one SOSS exposure (all supplied rateints segments), orders 1/2/3.

    This is what ``ts.jwst.stage2`` runs for NIRISS/SOSS data, after
    ``ts.jwst.load(...).detector_calibration()`` and ``fit_ramps()``.

    ``inputs`` is a sequence of CubeModels/paths or a stage1 output dictionary.
    BackgroundStep is mandatory and runs on a representative median rate image
    AFTER ramp fitting; its fitted static background is subtracted from every
    integration. No in-place edits of inputs, temporal flux replacement, absolute
    flux calibration are performed. Pixels retain DQ. Box white-light curves
    use a fixed, complete, nonoverlapping wavelength band.
    DO_NOT_USE and SATURATED pixels are always excluded. ``mask_dq=True`` also
    excludes informational flags (e.g. a successfully fitted jump), and can
    leave no complete wavelength band in long series.

    ``optimal_methods=('polynomial', 'gp')`` additionally trains a frozen shared
    intrinsic profile for each order and method. Results are in
    ``result['optimal'][method]['orders']``; the original box products remain
    in ``result['orders']``. Profiles reuse their coefficients through each
    integration's trace geometry. The default joint fit separates overlapping
    native-column contributions and retains inter-order covariance. The older
    exclusion policy is available with overlap_strategy='isolated'.
    """
    from stdatamodels.jwst import datamodels
    from jwst.background import BackgroundStep
    from . import soss_optimal
    source_hashes = {name: file_sha256(Path(__file__).parent/name) for name in
        ('soss.py', 'soss_optimal.py', 'soss_overlap.py', 'core/shared_profile.py',
         'core/overlap.py', 'core/extraction.py', 'core/geometry.py', 'core/profiles.py',
         'core/kernels.py', 'core/tracing.py')}
    optimal_methods, optimal_settings = soss_optimal.settings(
        optimal_methods, optimal_options, aperture_radius)
    joint = optimal_settings['overlap_strategy']=='joint'
    profile_radius = optimal_settings['profile_radius']
    values = inputs['rampstep'] if isinstance(inputs, dict) else inputs
    if not len(values) or sample_integrations < 2 or int(sample_integrations) != sample_integrations:
        raise ValueError('Provide rateints and at least two background/trace samples')
    samples, sample_variances, sample_ids, times, records = [], [], [], [], []
    reference_identity = None
    for value in values:
        with open_model(value) as m:
            if m.data.ndim != 3 or m.meta.exposure.type != 'NIS_SOSS':
                raise ValueError('SOSS stage2 requires ramp-fitted NIS_SOSS CubeModels')
            if m.meta.cal_step.ramp_fit != 'COMPLETE':
                raise ValueError('Run ramp fitting before SOSS stage2')
            if (m.meta.instrument.name, m.meta.instrument.filter, m.meta.instrument.pupil) != ('NIRISS','CLEAR','GR700XD'):
                raise ValueError('Three-order SOSS extraction requires NIRISS CLEAR/GR700XD')
            if m.meta.bunit_data not in ('DN/s', 'DN / s'):
                raise ValueError('SOSS extraction requires rateints in DN/s, before photometric calibration')
            if m.meta.cal_step.bkg_subtract == 'COMPLETE':
                raise ValueError('Background already subtracted; provide rateints to avoid double subtraction')
            identity = (m.meta.exposure.start_time, m.meta.subarray.name, m.meta.instrument.pupil_position)
            if reference_identity is not None and identity != reference_identity:
                raise ValueError('Segments must belong to one exposure/subarray/PWCPOS')
            reference_identity = identity
            if not records:
                traces, reference_info = reference_traces(m, pastasoss_reference)
                bkg_path = _reference(m, 'bkg', background_reference)
                template_model = datamodels.ImageModel()
                template_model.update(m)
            indices = np.unique(np.linspace(0, len(m.data)-1, max(2, sample_integrations//len(values))).astype(int))
            for i in indices:
                valid = np.isfinite(m.data[i]) & np.isfinite(m.err[i]) & (m.err[i] > 0)
                valid &= (m.dq[i] == 0) if mask_dq else ((m.dq[i] & 3) == 0)
                samples.append(np.where(valid, m.data[i], np.nan))
                if optimal_methods:
                    sample_variances.append(np.where(valid, np.asarray(m.err[i], float)**2, np.nan))
                    sample_ids.append(int(m.meta.exposure.integration_start)+int(i))
            times.extend(integration_times(m))
            records.append(dict(filename=m.meta.filename, nints=len(m.data),
                integration_start=m.meta.exposure.integration_start,
                integration_end=m.meta.exposure.integration_end,
                calibration_version=m.meta.calibration_software_version,
                crds_context=m.meta.ref_file.crds.context_used))
    times = np.asarray(times)
    if np.any(np.diff(times) <= 0):
        raise ValueError('SOSS segments must be chronological without overlapping integrations')
    if isinstance(inputs, dict) and 'times' in inputs:
        if not np.allclose(inputs['times'], times, rtol=0, atol=1e-7):
            raise ValueError('Supplied times disagree with rateints INT_TIMES')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        median = np.nanmedian(samples, axis=0)
    template_model.data = median.copy()
    template_model.err = np.zeros_like(median)
    template_model.dq = (~np.isfinite(median)).astype(np.uint32)
    options = dict(background_options or {})
    if set(options)-{'soss_source_percentile', 'soss_bkg_percentile'}:
        raise ValueError('Unsupported SOSS background option')
    corrected = BackgroundStep(override_bkg=bkg_path, **options).run(template_model.copy())
    if corrected.meta.cal_step.bkg_subtract != 'COMPLETE':
        raise RuntimeError('JWST SOSS background subtraction did not complete')
    background = median-corrected.data
    # Missing reference pixels are masked, not evidence for undefined backgrounds.
    background = np.where(np.isfinite(background), background, 0.)
    median_subtracted = median-background
    valid_median = np.isfinite(median_subtracted)
    for order, trace in traces.items():
        y, fit = refine_trace(median_subtracted, trace['columns'], trace['reference_trace'],
                              valid=valid_median, max_shift=trace_max_shift)
        trace.update(trace=y, fit=fit)
    nrows, ncols = median.shape
    source_mask = np.zeros(median.shape, bool)
    for trace in traces.values():
        source_mask[:, trace['columns']] |= aperture_weights(nrows, trace['trace'], aperture_radius+12) > 0
    source_mask[:, :4] = True
    source_mask[:, -4:] = True
    for order, trace in traces.items():
        x, y = trace['columns'], trace['trace']
        isolated = (y-aperture_radius >= -.5) & (y+aperture_radius <= nrows-.5)
        for other_order, other in traces.items():
            if other_order == order:
                continue
            other_y = np.interp(x, other['columns'], other['trace'], left=np.nan, right=np.nan)
            isolated &= ~np.isfinite(other_y) | (np.abs(other_y-y) > 2*aperture_radius+10)
        if isolated.sum() < 10:
            raise ValueError(f'Order {order} has insufficient isolated aperture support')
        trace['isolated'] = isolated
        _, _, trace['motion_template'] = trace_motion(median_subtracted, valid_median,
                                                     x[isolated], y[isolated])
        trace['flux'], trace['error'], trace['shift'], trace['width'] = [], [], [], []
    optimal = {}
    if optimal_methods:
        selected = np.unique(np.linspace(0, len(samples)-1,
            min(len(samples), optimal_settings['training_integrations'])).astype(int))
        training_data, training_variance = [], []
        for index in selected:
            frame = np.asarray(samples[index], float)-background
            if scale_1f:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', RuntimeWarning)
                    level = np.nanmedian(np.where(~source_mask, frame, np.nan), axis=0)
                frame -= np.where(np.isfinite(level), level, 0.)[None, :]
            training_data.append(frame)
            training_variance.append(sample_variances[index])
        for method in optimal_methods:
            optimal[method] = dict(orders={})
            for order, trace in traces.items():
                x, y = trace['columns'], trace['trace']
                keep = np.ones(len(x), bool) if joint else trace['isolated'].copy()
                # Every trained column needs at least two observed pixels in
                # one sample. Others remain explicitly unavailable, not filled.
                active = aperture_weights(nrows, y, profile_radius) > 0
                observed = np.any([np.sum(np.isfinite(frame[:, x]) & active, axis=0) >= 2
                                   for frame in training_data], axis=0)
                keep &= observed
                # Native geometry can extend to whole boundary pixels.
                extent = optimal_settings['spacing']*int(profile_radius/optimal_settings['spacing']+.5)
                keep &= (y-extent >= .5) & (y+extent < nrows-1.5)
                if keep.sum() < 10:
                    raise ValueError(f'Order {order}: insufficient support for optimal training')
                training_traces = []
                for frame in training_data:
                    motion = trace['isolated']
                    shift, _, _ = trace_motion(frame, np.isfinite(frame), x[motion], y[motion], trace['motion_template'])
                    training_traces.append(y[keep] if single_trace_extraction else y[keep]+shift)
                print(f'Training SOSS order {order} {method} profile on {len(selected)} integrations, {keep.sum()} columns', flush=True)
                seed_data = [f[:, x[keep]].copy() for f in training_data]
                if joint:
                    # Initialize from uncontaminated parts of each profile;
                    # the subsequent joint likelihood uses blended pixels too.
                    for other_order, other in traces.items():
                        if other_order == order:
                            continue
                        neighbor = np.interp(x[keep],other['columns'],other['trace'],left=np.nan,right=np.nan)
                        contaminated = np.abs(np.arange(nrows)[:,None]-neighbor) <= aperture_radius+3
                        for frame in seed_data:
                            frame[contaminated] = np.nan
                model = soss_optimal.train(seed_data,
                    [v[:, x[keep]] for v in training_variance], np.asarray(training_traces),
                    x[keep], profile_radius, method, optimal_settings,
                    [sample_ids[i] for i in selected])
                optimal[method]['orders'][order] = dict(columns=x, wavelength=trace['wavelength'],
                    reference_trace=trace['reference_trace'], trace=y, isolated=trace['isolated'].copy(),
                    model_columns=keep, profile_model=model, training_traces=np.asarray(training_traces),
                    flux=[], error=[], profile_fraction=[])
            if joint:
                from .soss_overlap import train_joint, unmodeled_pixels
                outputs=list(optimal[method]['orders'].values())
                training_for_joint=[]
                for i,frame in enumerate(training_data):
                    # Trace shifts used here agree with those used for the
                    # modeled orders; missing orders must not become zero light.
                    complete_centers={o: t['trace']+(0. if single_trace_extraction else
                        np.median(t['training_traces'][i]-t['trace'][t['model_columns']]))
                        for o,t in optimal[method]['orders'].items()}
                    unknown=unmodeled_pixels(optimal[method]['orders'],complete_centers,frame.shape,profile_radius)
                    training_for_joint.append(np.where(unknown,np.nan,frame))
                print(f'Jointly refining all SOSS {method} profiles including overlapping pixels',flush=True)
                models,diagnostic=train_joint([t['profile_model'] for t in outputs],
                    [t['training_traces'] for t in outputs],
                    [t['columns'][t['model_columns']] for t in outputs],
                    training_for_joint,training_variance,maxiter=optimal_settings['maxiter'])
                for t,model in zip(outputs,models):
                    t['profile_model']=model
                optimal[method].update(joint_fit=diagnostic,joint_covariance=[],joint_condition=[])
        del training_data, training_variance
    del samples, sample_variances
    first_image, last_image = None, None
    for value in values:
        with open_model(value) as m:
            for i in range(len(m.data)):
                frame = np.asarray(m.data[i], float)-background
                variance = np.asarray(m.err[i], float)**2
                valid = np.isfinite(frame) & np.isfinite(variance) & (variance > 0)
                valid &= (m.dq[i] == 0) if mask_dq else ((m.dq[i] & 3) == 0)
                if scale_1f:
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore', RuntimeWarning)
                        level = np.nanmedian(np.where(valid & ~source_mask, frame, np.nan), axis=0)
                    frame -= np.where(np.isfinite(level), level, 0.)[None, :]
                if first_image is None:
                    first_image = frame.copy()
                last_image = frame.copy()
                centers = {}
                for order, trace in traces.items():
                    x, y, keep = trace['columns'], trace['trace'], trace['isolated']
                    shift, width, _ = trace_motion(frame, valid, x[keep], y[keep], trace['motion_template'])
                    center = y if single_trace_extraction else y+shift
                    centers[order] = center
                    flux, error = extract_box(frame, variance, valid, x, center, aperture_radius)
                    trace['flux'].append(flux)
                    trace['error'].append(error)
                    trace['shift'].append(shift)
                    trace['width'].append(width)
                    for method in optimal_methods:
                        if joint:
                            continue
                        out = optimal[method]['orders'][order]
                        keep = out['model_columns']
                        f, e, fraction = soss_optimal.extract(out['profile_model'], frame, variance,
                            valid, x[keep], center[keep], aperture_radius, optimal_settings)
                        for key, values in [('flux', f), ('error', e), ('profile_fraction', fraction)]:
                            full = np.full(len(x), np.nan)
                            full[keep] = values
                            out[key].append(full)
                if joint:
                    from .soss_overlap import extract as extract_joint, unmodeled_pixels
                    for method in optimal_methods:
                        product=optimal[method]
                        outputs=list(product['orders'].values())
                        unknown=unmodeled_pixels(product['orders'],centers,frame.shape,profile_radius)
                        f,e,fraction,covariance,condition=extract_joint(
                            [t['profile_model'] for t in outputs],
                            [t['columns'][t['model_columns']] for t in outputs],
                            [centers[o][t['model_columns']] for o,t in product['orders'].items()],
                            frame,variance,valid & ~unknown,aperture_radius,optimal_settings)
                        product['joint_covariance'].append(covariance)
                        product['joint_condition'].append(condition)
                        for k,t in enumerate(outputs):
                            for name,values in [('flux',f),('error',e),('profile_fraction',fraction)]:
                                t[name].append(values[t['columns'],k])
    for order, trace in traces.items():
        for key in ('flux', 'error', 'shift', 'width'):
            trace[key] = np.asarray(trace[key])
        common = trace['isolated'] & np.all(np.isfinite(trace['flux']) & (trace['error'] > 0), axis=0)
        if common.sum() < 10:
            raise ValueError(f'Order {order}: fewer than ten complete wavelengths across all integrations')
        trace['white_light_columns'] = common
        white = trace['flux'][:, common].sum(axis=1)
        normalization = np.median(white)
        if normalization <= 0:
            raise ValueError(f'Order {order}: nonpositive white-light flux')
        trace['white_light'] = white/normalization
        trace['white_light_error'] = np.sqrt((trace['error'][:, common]**2).sum(axis=1))/normalization
        trace['white_light_normalization'] = float(normalization)
    result = dict(orders=traces, times=times, median_rate=median, background=background,
        median_background_subtracted=median_subtracted, first_extracted_image=first_image,
        last_extracted_image=last_image, metadata=dict(mode='niriss/soss', input_segments=records,
        pastasoss=reference_info, background=dict(path=bkg_path, sha256=file_sha256(bkg_path),
        status='COMPLETE', fit='static template fitted by BackgroundStep to sampled median rate',
        options=options), settings=dict(aperture_radius=aperture_radius, trace_max_shift=trace_max_shift,
        sample_integrations=sample_integrations, scale_1f=scale_1f, mask_dq=mask_dq,
        single_trace_extraction=single_trace_extraction),
        extraction='fractional box; incomplete apertures missing; no order decontamination',
        soss_source_sha256=source_hashes['soss.py'], units='DN/s; no absolute photometric calibration',
        uncertainty='pixel variances only; excludes fitted background/trace/1f covariance'))
    result['optimal'] = optimal
    result['metadata']['optimal_methods'] = list(optimal_methods)
    result['metadata']['optimal_settings'] = optimal_settings if optimal_methods else None
    result['metadata']['extraction_source_hashes'] = source_hashes
    for method, products in optimal.items():
        products.update({key: result[key] for key in ('times', 'median_rate', 'background',
            'median_background_subtracted', 'first_extracted_image', 'last_extracted_image')})
        products['metadata'] = dict(result['metadata'], optimal_methods=[],
            extraction=f'shared intrinsic {method} profile; {optimal_settings["overlap_strategy"]} orders; fractional aperture flux convention',
            method=method, profile_training={})
        if joint:
            products['metadata']['joint_fit']=products['joint_fit']
            products['joint_covariance']=np.asarray(products['joint_covariance'])
            products['joint_condition']=np.asarray(products['joint_condition'])
        for order, trace in products['orders'].items():
            for key in ('flux', 'error', 'profile_fraction'):
                trace[key] = np.asarray(trace[key])
            trace.update({key: traces[order][key] for key in ('shift', 'width')})
            common = trace['model_columns'] & np.all(np.isfinite(trace['flux']) &
                np.isfinite(trace['error']) & (trace['error'] > 0), axis=0)
            if common.sum() < 10:
                raise ValueError(f'Order {order} {method}: fewer than ten common wavelengths')
            white = trace['flux'][:, common].sum(axis=1)
            normalization = np.median(white)
            if not np.isfinite(normalization) or normalization <= 0:
                raise ValueError(f'Order {order} {method}: nonpositive white-light flux')
            trace.update(white_light_columns=common, white_light=white/normalization,
                white_light_error=np.sqrt((trace['error'][:, common]**2).sum(axis=1))/normalization,
                white_light_normalization=float(normalization))
            products['metadata']['profile_training'][str(order)] = trace['profile_model'].diagnostics
    corrected.close()
    template_model.close()
    result['products'] = to_products(result)
    if outputfolder is not None:
        save_products(result, outputfolder)
    return result



def reduce_soss(inputs, **options):
    """Deprecated: reduce SOSS rateints; use ``ts.jwst.load`` + ``ts.jwst.stage2``.

    Kept for compatibility; it runs :func:`extract_soss`, the SOSS Stage 2 used by
    ``ts.jwst.stage2``, with the same keywords.
    """
    warnings.warn('soss.reduce_soss is deprecated; use ts.jwst.stage2 (or soss.extract_soss)',
                  DeprecationWarning, stacklevel=2)
    return extract_soss(inputs, **options)

def save_products(result, folder):
    """Write per-order FITS spectra, NPZ arrays, light curves and trace tables."""
    from astropy.io import fits
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    (folder/'soss_metadata.json').write_text(json.dumps(result['metadata'], indent=2, allow_nan=False)+'\n')
    if 'joint_covariance' in result:
        np.savez_compressed(folder/'order_flux_covariance.npz',covariance=result['joint_covariance'],
            condition=result['joint_condition'],orders=np.asarray(list(result['orders'])),
            time_bjd_tdb=result['times'],columns=np.arange(result['joint_covariance'].shape[1]))
    np.savez_compressed(folder/'soss_diagnostics.npz', **{key: result[key] for key in
        ('median_rate', 'background', 'median_background_subtracted', 'first_extracted_image',
         'last_extracted_image')})
    for order, trace in result['orders'].items():
        prefix = folder/f'order{order}'
        arrays = {k: trace[k] for k in ('columns', 'wavelength', 'reference_trace', 'trace',
            'flux', 'error', 'shift', 'width', 'isolated', 'white_light_columns',
            'white_light', 'white_light_error')}
        arrays['time_bjd_tdb'] = result['times']
        np.savez_compressed(str(prefix)+'_spectra.npz', **arrays)
        hdus = [fits.PrimaryHDU()]
        hdus[0].header['INSTRUME'] = 'NIRISS'
        hdus[0].header['EXP_TYPE'] = 'NIS_SOSS'
        hdus[0].header['SPORDER'] = order
        hdus[0].header['S_BKDSUB'] = 'COMPLETE'
        hdus[0].header['EXTRACT'] = result['metadata'].get('method', 'box').upper()
        for name, key, unit in [('SCI', 'flux', 'DN/s'), ('ERR', 'error', 'DN/s'),
                                ('WAVELENGTH', 'wavelength', 'um'), ('TIME', 'time_bjd_tdb', 'd')]:
            hdu = fits.ImageHDU(arrays[key], name=name)
            hdu.header['BUNIT'] = unit
            if name == 'TIME':
                hdu.header['TIMESYS'] = 'TDB'
                hdu.header['TREFPOS'] = 'BARYCENTER'
                hdu.header['JDREFI'] = 0
                hdu.header['JDREFF'] = 0.
            hdus.append(hdu)
        dq = (~np.isfinite(trace['flux'])).astype(np.uint16)
        dq |= np.broadcast_to(~trace['isolated'], dq.shape).astype(np.uint16)*2
        hdus.append(fits.ImageHDU(dq, name='DQ'))
        hdus[-1].header['DQBIT0'] = 'Missing/incomplete aperture'
        hdus[-1].header['DQBIT1'] = 'Geometric overlap/edge; joint fit may recover'
        hdus[0].header['DEBLEND'] = result['metadata'].get('optimal_settings',{}).get('overlap_strategy')=='joint' if result['metadata'].get('method') else False
        hdus.append(fits.ImageHDU(trace['white_light_columns'].astype(np.uint8), name='WHITE_MASK'))
        fits.HDUList(hdus).writeto(str(prefix)+'_spectra.fits', overwrite=True, checksum=True)
        np.savetxt(str(prefix)+'_white_light.csv', np.column_stack([result['times'],
            trace['white_light'], trace['white_light_error'], trace['shift'], trace['width']]),
            delimiter=',', header='time_bjd_tdb,relative_flux,relative_error,trace_shift_pixels,profile_sigma_pixels', comments='')
        np.savetxt(str(prefix)+'_trace.csv', np.column_stack([trace['columns'], trace['reference_trace'],
            trace['trace'], trace['wavelength'], trace['isolated'], trace['white_light_columns']]),
            delimiter=',', header='column,reference_row,fitted_row,wavelength_microns,isolated,white_light_column', comments='')
        if 'profile_model' in trace:
            trace['profile_model'].save(str(prefix)+'_profile.npz')
            np.savez_compressed(str(prefix)+'_profile_support.npz',
                model_columns=trace['model_columns'], profile_fraction=trace['profile_fraction'])
    for method, products in result.get('optimal', {}).items():
        save_products(products, folder/'optimal'/method)
    if 'products' in result:
        from .core import products as standard, provenance
        manifest = provenance.build_manifest(config=dict(metadata=result['metadata']), step='soss stage2',
                                             mode='niriss/soss')
        standard.write_products(result['products'], folder/'products', manifest)


def load_products(folder, strict=False):
    """Read this module's saved products for additional validation diagnostics.

    If the extraction code changed since the products were written, a warning is
    issued (``strict=True`` raises instead): a changed file, e.g. a docstring edit or a
    refactor, does not necessarily change the results.
    """
    folder = Path(folder)
    result = dict(metadata=json.loads((folder/'soss_metadata.json').read_text()), orders={})
    changed = []
    if result['metadata']['soss_source_sha256'] != file_sha256(__file__):
        changed.append('soss.py')
    for name, expected in result['metadata'].get('extraction_source_hashes', {}).items():
        current = Path(__file__).parent/name
        if not current.exists() or file_sha256(current) != expected:
            changed.append(name)
    if changed:
        message = 'SOSS extraction code changed since these products were written: ' + ', '.join(sorted(set(changed)))
        if strict:
            raise ValueError(message + '; regenerate products before reuse')
        warnings.warn(message)
    with np.load(folder/'soss_diagnostics.npz', allow_pickle=False) as saved:
        result.update({key: saved[key].copy() for key in saved.files})
    if (folder/'order_flux_covariance.npz').exists():
        with np.load(folder/'order_flux_covariance.npz',allow_pickle=False) as saved:
            result['joint_covariance']=saved['covariance'].copy()
            result['joint_condition']=saved['condition'].copy()
    for order in (1, 2, 3):
        with np.load(folder/f'order{order}_spectra.npz', allow_pickle=False) as saved:
            result['orders'][order] = {key: saved[key].copy() for key in saved.files}
            if order == 1:
                result['times'] = saved['time_bjd_tdb'].copy()
            elif not np.array_equal(result['times'], saved['time_bjd_tdb']):
                raise ValueError('Saved orders have different timestamps')
        if result['metadata'].get('method'):
            from .shared_profile import SharedProfile
            result['orders'][order]['profile_model'] = SharedProfile.load(folder/f'order{order}_profile.npz')
            with np.load(folder/f'order{order}_profile_support.npz', allow_pickle=False) as saved:
                result['orders'][order].update({key: saved[key].copy() for key in saved.files})
    result['optimal'] = {method: load_products(folder/'optimal'/method, strict=strict)
                         for method in result['metadata'].get('optimal_methods', [])}
    if 'method' not in result['metadata']:  # The top-level (box) result.
        result['products'] = to_products(result)
    return result


def to_products(result):
    """Standard products of a SOSS result: one spectroscopic time series per order and method.

    Names are ``order<N>_<method>`` (``box``, ``polynomial``, ``gp``), plus the SOSS
    fixed-band white-light curves ``order<N>_<method>_whitelight``.
    """
    from .core.products import DQ_BLENDED, DQ_MISSING, LightCurveSet, SpectroscopicTimeSeries
    products = {}

    def add(result, method):
        for order, trace in result['orders'].items():
            flux, error = np.asarray(trace['flux']), np.asarray(trace['error'])
            dq = np.where(np.isfinite(flux) & np.isfinite(error), 0, DQ_MISSING).astype(np.uint32)
            dq |= (np.broadcast_to(~np.asarray(trace['isolated'], bool), dq.shape)*DQ_BLENDED).astype(np.uint32)
            name = f'order{order}_{method}'
            metadata = dict(mode='niriss/soss', method=method, order=int(order),
                            extraction=result['metadata'].get('extraction', method),
                            aperture_radius=result['metadata'].get('aperture_radius'),
                            calibration=result['metadata'].get('inputs', result['metadata'].get('segments')))
            products[name] = SpectroscopicTimeSeries(
                times=result['times'], wavelength=trace['wavelength'], flux=flux, flux_err=error, dq=dq,
                columns=trace['columns'], trace=trace['trace'], trace_width=trace.get('width'),
                order=int(order), name=name, flux_unit='DN/s', metadata=metadata)
            band = np.asarray(trace['white_light_columns'], bool)
            wavelength = np.asarray(trace['wavelength'])[band]
            products[name+'_whitelight'] = LightCurveSet(
                times=result['times'], flux=np.asarray(trace['white_light'])[None],
                flux_err=np.asarray(trace['white_light_error'])[None],
                wavelength_low=[wavelength.min()], wavelength_high=[wavelength.max()],
                labels=['white'], name=name+'_whitelight',
                metadata=dict(metadata, band='fixed SOSS white-light columns', n_columns=int(band.sum())))

    add(result, 'box')
    for method, optimal in result.get('optimal', {}).items():
        add(optimal, method)
    return products
