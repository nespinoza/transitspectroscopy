"""MIRI imaging TSO calibration and aperture/2-D GP optimal photometry.

Uses official Detector1/Image2 stages and local extraction only. Each call
reduces one visit; separate exposures retain their identities and time gaps.
"""
from pathlib import Path
from copy import deepcopy
import json
import warnings
import numpy as np
from astropy.io import fits
from . import photometry as ph
from .soss import _model, _times, _hash


def calibrate(path, outputfolder, *, input_product='rateints', detector_options=None,
              image_options=None, save_calints=True, ramp_fit_method='jwst',
              ramp_fit_options=None):
    """Return calibrated CubeModel; caller owns/closes it. No input mutation.

    Defaults keep official MIRI Detector1 ordering and Image2 flat/photom.
    Configurable JWST ``steps`` dictionaries are passed to the official pipelines.
    """
    from jwst.pipeline import Detector1Pipeline, Image2Pipeline
    from stdatamodels.jwst import datamodels
    from .ramp_fitting import validate_selection, detector1_class
    validate_selection(ramp_fit_method, ramp_fit_options)
    if ramp_fit_method != 'jwst' and input_product != 'uncal':
        raise ValueError('Selecting an alternative ramp fitter requires uncal inputs')
    folder = Path(outputfolder); folder.mkdir(parents=True,exist_ok=True)
    if input_product == 'calints':
        return datamodels.open(str(path))
    if input_product not in ('uncal','rateints'):
        raise ValueError('input_product must be uncal, rateints or calints')
    if input_product == 'uncal':
        pipeline = Detector1Pipeline if ramp_fit_method == 'jwst' else detector1_class(ramp_fit_options)
        if ramp_fit_method != 'jwst' and (detector_options or {}).get('ramp_fit', {}).get('skip', False):
            raise ValueError('Cannot skip the selected SPARTA ramp-fitting step')
        pipeline.call(str(path),output_dir=str(folder),save_results=True,
            steps=detector_options or {})
        path = folder/Path(path).name.removesuffix('.gz').replace('_uncal.fits','_rateints.fits')
        if not path.is_file():
            raise RuntimeError('Detector1 did not produce rateints')
    options = deepcopy(image_options or {})
    # Never resample or combine a TSO: native integrations are the measurements.
    options.setdefault('resample',{})['skip'] = True
    compressed=isinstance(path,(str,Path)) and str(path).endswith('.gz')
    source = datamodels.open(str(path)) if compressed else (str(path) if isinstance(path,(str,Path)) else path.copy())
    try:
        products = Image2Pipeline.call(source,output_dir=str(folder),
            save_results=save_calints,steps=options)
    finally:
        if not isinstance(source,str):
            source.close()
    if len(products) != 1:
        raise ValueError('Expected one imaging cube per input segment')
    return products[0]


def prepare_segment(model, *, center=None, stamp_radius=10, annulus=(12.,20.),
                    aperture_radius=5., exclude_sources=()):
    """Reduce a calibrated segment to small source stamps and sky statistics.

    MJy/sr is converted to Jy/pixel using the calibrated pixel area. DN/s is
    retained when photom is explicitly skipped. No gain is guessed or refitted.
    Excluded sources are (x,y,radius) in zero-based input coordinates.
    """
    if model.meta.instrument.name != 'MIRI' or model.meta.exposure.type != 'MIR_IMAGE' or not model.meta.visit.tsovisit:
        raise ValueError('Expected MIRI MIR_IMAGE TSO science integrations')
    if model.data.ndim != 3 or model.err.shape != model.data.shape:
        raise ValueError('Expected SCI/ERR integration cubes')
    if model.meta.cal_step.flat_field != 'COMPLETE':
        raise ValueError('MIRI photometry requires Image2 flat-field calibration')
    shape = model.data.shape[1:]
    if center is None:
        ra,dec = model.meta.target.ra,model.meta.target.dec
        if model.meta.wcs is None or ra is None or dec is None:
            raise ValueError('Supply target center=(x,y) or calibrated target WCS')
        center = np.array(model.meta.wcs.invert(ra,dec)[:2],dtype=float)
    center = np.asarray(center,dtype=float)
    if center.shape != (2,) or not np.isfinite(center).all():
        raise ValueError('Invalid target center')
    unit = str(model.meta.bunit_data).lower().replace(' ','')
    factor = np.ones(shape)
    if unit == 'mjy/sr':
        area = model.meta.photometry.pixelarea_steradians
        if area is None or not np.isfinite(area) or area <= 0:
            raise ValueError('MJy/sr requires calibrated pixel solid angle')
        relative = getattr(model,'area',None)
        if relative is not None and np.size(relative):
            if np.shape(relative) != shape:
                raise ValueError('Pixel area map shape mismatch')
            factor *= relative
        factor *= float(area)*1e6
        output_unit = 'Jy'
    elif unit in ('dn/s','dn/sec'):
        output_unit = 'DN/s'
    else:
        raise ValueError('Unsupported photometry units: '+unit)
    exclude = np.zeros(shape,bool)
    yy,xx = np.indices(shape)
    for cx,cy,r in exclude_sources:
        exclude |= np.hypot(xx-cx,yy-cy) <= r
    sample = np.unique(np.linspace(0,len(model.data)-1,min(12,len(model.data))).astype(int))
    sample_data=np.asarray(model.data[sample],float)*factor
    sample_var=np.asarray(model.err[sample],float)**2*factor**2
    sample_dq=model.dq[sample] if model.dq.ndim==3 else model.dq
    sample_good=ph.valid_pixels(sample_data,sample_var,sample_dq)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning)  # intentionally masked reference pixels
        median = np.nanmedian(np.where(sample_good,sample_data,np.nan),axis=0)
        vmedian = np.nanmedian(np.where(sample_good,sample_var,np.nan),axis=0)
    refined,_ = ph.centroid(median,vmedian,center,max_shift=4.)
    x0,y0 = np.rint(refined).astype(int)-stamp_radius
    size = 2*stamp_radius+1
    if x0<0 or y0<0 or x0+size>shape[1] or y0+size>shape[0]:
        raise ValueError('Target PSF stamp crosses detector edge')
    if not 0 < aperture_radius < stamp_radius < annulus[0] < annulus[1]:
        raise ValueError('Require aperture < stamp radius < sky inner < sky outer')
    sl = np.s_[y0:y0+size,x0:x0+size]
    stamps,variances,centers,widths,skies,skyvars,skycounts,flags = [],[],[],[],[],[],[],[]
    for i in range(len(model.data)):
        image = np.asarray(model.data[i],float)*factor
        var = np.asarray(model.err[i],float)**2*factor**2
        dq = model.dq[i] if model.dq.ndim == 3 else model.dq
        good = ph.valid_pixels(image,var,dq) & ~exclude & np.isfinite(factor) & (factor>0)
        image = np.where(good,image,np.nan)
        failed = False
        try:
            c,w = ph.centroid(image,var,refined)
            b,bv,n = ph.background(image,var,c,annulus,good,exclude)
        except ValueError:
            c,w,b,bv,n = refined,np.full(2,np.nan),np.nan,np.nan,0
            failed = True
        stamps.append((image[sl]-b).astype('float32'))
        variances.append(np.where(good[sl],var[sl],np.nan).astype('float32'))
        centers.append(c-[x0,y0]); widths.append(w)
        skies.append(b);skyvars.append(bv);skycounts.append(n);flags.append(int(failed))
    return dict(data=np.asarray(stamps),variance=np.asarray(variances),centers=np.asarray(centers),
        widths=np.asarray(widths),sky=np.asarray(skies),sky_variance=np.asarray(skyvars),
        sky_pixels=np.asarray(skycounts),flags=np.asarray(flags),times=_times(model),
        origin=np.array([x0,y0]),median_image=median,unit=output_unit,
        exposure=str(model.meta.filename).split('-seg')[0],
        metadata=dict(filter=model.meta.instrument.filter,program=model.meta.observation.program_number,
            observation=model.meta.observation.observation_number,visit=model.meta.observation.visit_number,
            input_unit=model.meta.bunit_data,unit=output_unit,calibration_version=model.meta.calibration_software_version,
            crds_context=model.meta.ref_file.crds.context_used,cal_step=model.meta.cal_step.instance,
            annulus=list(annulus),stamp_radius=stamp_radius,aperture_radius=aperture_radius,
            exclude_sources=list(exclude_sources)))


def save_prepared(segment, path):
    """Portable calibrated cutouts; retain geometry, masking and original times."""
    values = dict(segment); values['metadata'] = json.dumps(values['metadata'])
    np.savez_compressed(path,**values)


def load_prepared(path):
    with np.load(path,allow_pickle=False) as a:
        result = {k:a[k] for k in a.files}
    result['metadata'] = json.loads(str(result['metadata']))
    for key in ('unit','exposure'):
        result[key] = str(result[key])
    return result


def extract_visit(segments, *, aperture_radius=5., training_integrations=12,
                  gp_options=None, minimum_fraction=.8, moving_aperture=False,
                  outputfolder=None):
    """One shared 2-D GP per visit, independent integration photometry.

    Permanently invalid detector pixels are excluded by a fixed sum mask. Other
    incomplete simple apertures are flagged/missing, never filled or renormalized.
    Optimal weights use full observed PSF plus the correlated sky uncertainty.
    Sum flux refers to the observed fixed aperture. GP flux is corrected to a
    fixed, unmasked reference aperture, avoiding pointing-dependent throughput
    changes caused by dead pixels. No infinite-aperture correction is implied.
    """
    if not segments:
        raise ValueError('No prepared segments')
    if not 0<minimum_fraction<=1 or training_integrations<2 or int(training_integrations)!=training_integrations:
        raise ValueError('Invalid support fraction or training count')
    if not np.isfinite(aperture_radius) or aperture_radius<=0:
        raise ValueError('Positive finite aperture radius required')
    segments = sorted(segments,key=lambda s:s['times'][0])
    for s in segments:
        if s['unit'] != segments[0]['unit'] or s['data'].shape[1:] != segments[0]['data'].shape[1:]:
            raise ValueError('Incompatible segment units/stamp shapes')
        for key in ('filter','program','observation','visit'):
            if s['metadata'].get(key) != segments[0]['metadata'].get(key):
                raise ValueError('Select one visit and filter')
    keys = ('data','variance','centers','widths','sky','sky_variance','sky_pixels','flags','times')
    result = {key:np.concatenate([s[key] for s in segments]) for key in keys}
    if not np.all(np.diff(result['times'])>0):
        raise ValueError('Duplicate, overlapping or unordered integrations')
    result['segment'] = np.concatenate([np.full(len(s['times']),i) for i,s in enumerate(segments)])
    result['detector_centers'] = np.concatenate([s['centers']+s['origin'] for s in segments])
    candidates = np.flatnonzero((result['flags']==0)&np.isfinite(result['sky_variance']))
    if len(candidates)<2:
        raise ValueError('Insufficient usable integrations')
    train = candidates[np.unique(np.linspace(0,len(candidates)-1,min(training_integrations,len(candidates))).astype(int))]
    print(f'Fitting shared 2-D GP PSF on {len(train)} of {len(result["times"])} integrations',flush=True)
    model = ph.fit_gp_psf(result['data'][train],result['variance'][train],result['centers'][train],
        sky_variances=result['sky_variance'][train],**(gp_options or {}))
    shape = result['data'].shape[1:]
    canonical_center=(np.array(shape[::-1])-1)/2
    evaluate = model.evaluator(shape,np.vstack((result['centers'][candidates],canonical_center)))
    approximation = max(np.max(abs(evaluate(c)-model.evaluate(shape,c))) for c in result['centers'][train[:3]])
    if approximation > 1e-6:
        raise ValueError('Cached GP evaluator exceeds profile tolerance')
    reference_center=np.nanmedian(result['detector_centers'][candidates],axis=0)
    fixed = {i:reference_center-s['origin'] for i,s in enumerate(segments)}
    fixed_weights = {i:ph.circular_weights(shape,c,aperture_radius) for i,c in fixed.items()}
    reference_fraction=float(np.sum(evaluate(canonical_center)*
                                    ph.circular_weights(shape,canonical_center,aperture_radius)))
    # A permanent bad detector pixel must not invalidate an entire visit's sum
    # photometry. Freeze that exclusion across integrations and use the same
    # masked aperture for both flux conventions. Transient losses still flag NaN.
    never_observed=set()
    for s in segments:
        if np.all(s['flags']&1):
            continue
        good=ph.valid_pixels(s['data'],s['variance'])
        ys,xs=np.where(~np.any(good,axis=0))
        never_observed.update(zip((xs+s['origin'][0]).astype(int),(ys+s['origin'][1]).astype(int)))
    sum_masks={}
    for i,s in enumerate(segments):
        y,x=np.indices(shape)
        mask=np.array([(int(xx+s['origin'][0]),int(yy+s['origin'][1])) not in never_observed
                      for xx,yy in zip(x.ravel(),y.ravel())]).reshape(shape)
        sum_masks[i]=mask
        fixed_weights[i]*=mask
        if fixed_weights[i].sum()<.5*np.pi*aperture_radius**2:
            raise ValueError('Fixed bad-pixel mask removes more than half the sum aperture')
    flux = np.full((len(result['times']),2),np.nan); error=flux.copy()
    fractions=np.full(len(flux),np.nan); chi2=fractions.copy()
    first_profile=first_weights=first_image=first_model=first_residual=None
    for i,(d,v,c,sv) in enumerate(zip(result['data'],result['variance'],result['centers'],result['sky_variance'])):
        if result['flags'][i]:
            continue
        seg=result['segment'][i]
        a = ph.circular_weights(shape,c,aperture_radius)*sum_masks[seg] if moving_aperture else fixed_weights[seg]
        good = ph.valid_pixels(d,v)
        if np.all(good[a>0]):
            flux[i,0] = np.sum(a*np.where(good,d,0.))
            error[i,0] = np.sqrt(np.sum(a*a*np.where(good,v,0.))+a.sum()**2*sv)
        else:
            result['flags'][i] |= 2
        p = evaluate(c)
        f,e,observed,w = ph.optimal_flux(d,v,p,sky_variance=sv,minimum_fraction=minimum_fraction)
        fraction = reference_fraction
        flux[i,1],error[i,1] = f*fraction,e*fraction
        fractions[i]=observed
        if not np.isfinite(f):
            result['flags'][i] |= 4
        else:
            residual=np.where(good,d-f*p,0.)
            chi2[i]=np.sum(residual*ph._precision(residual,v,good,sv))/max(1,good.sum()-1)
        if first_profile is None:
            first_profile,first_weights = p,w*fraction
            first_image,first_model = d,f*p
            first_residual = np.divide(d-f*p,np.sqrt(v),out=np.full_like(d,np.nan),where=good)
    result.update(flux=flux,error=error,profile_fraction=fractions,profile_chi2=chi2,
        profile_model=model,profile=first_profile,weights=first_weights,
        first_image=first_image,first_model=first_model,first_residual=first_residual,
        median_image=segments[0]['median_image'],unit=segments[0]['unit'],
        metadata=dict(segments=[s['metadata'] for s in segments],training_indices=train.tolist(),
            gp=model.diagnostics,aperture_radius=aperture_radius,moving_aperture=moving_aperture,
            fixed_aperture_detector_center=reference_center.tolist(),
            gp_reference_aperture_fraction=reference_fraction,
            flux_convention='Sum: observed fixed masked aperture; GP: fixed unmasked reference aperture',
            fixed_excluded_detector_pixels=[[int(x),int(y)] for x,y in sorted(never_observed)],
            evaluator_max_absolute_error=float(approximation),minimum_profile_fraction=minimum_fraction,
            error_scope='Conditional on PSF/centroid; includes correlated annular sky variance; no absolute aperture correction'))
    if outputfolder is not None:
        save_products(result,outputfolder)
    return result


def save_products(result, folder):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    arrays={k:result[k] for k in ('times','flux','error','centers','detector_centers','widths','sky',
        'sky_variance','sky_pixels','flags','segment','profile_fraction','profile_chi2','profile','weights',
        'first_image','first_model','first_residual')}
    np.savez_compressed(folder/'photometry.npz',**arrays)
    result['profile_model'].save(folder/'gp_psf.npz')
    (folder/'photometry_metadata.json').write_text(json.dumps(result['metadata'],indent=2))
    for j,method in enumerate(('sum','gp')):
        f,e=result['flux'][:,j],result['error'][:,j]
        norm=np.nanmedian(f)
        if not np.isfinite(norm) or norm<=0:
            raise ValueError('Nonpositive photometric baseline')
        values=np.column_stack((result['times'],f,e,f/norm,e/norm,result['flags'],result['segment']))
        np.savetxt(folder/f'{method}_lightcurve.csv',values,delimiter=',',
            header='time_bjd_tdb,flux,flux_error,relative_flux,relative_error,flags,segment',comments='')
    header=fits.Header();header['INSTRUME']='MIRI';header['EXP_TYPE']='MIR_IMAGE'
    header['BUNIT']=result['unit'];header['TIMESYS']='TDB';header['TIMEUNIT']='d'
    header['METHOD0']='sum';header['METHOD1']='2d_gp';header['DQBIT0']='centroid/background failure'
    header['DQBIT1']='incomplete sum aperture';header['DQBIT2']='insufficient optimal support'
    hdus=[fits.PrimaryHDU(header=header)]
    for name,key in [('SCI','flux'),('ERR','error'),('TIME','times'),('DQ','flags'),
                     ('PSF','profile'),('WEIGHTS','weights')]:
        hdus.append(fits.ImageHDU(result[key],name=name))
    fits.HDUList(hdus).writeto(folder/'photometry.fits',overwrite=True)
    from stdatamodels.jwst.datamodels import ImageModel
    with ImageModel(data=result['median_image'].astype('float32')) as model:
        model.meta.instrument.name='MIRI';model.meta.exposure.type='MIR_IMAGE'
        model.meta.bunit_data=result['unit'];model.save(folder/'median_image.fits')


def reduce_miri(inputs, *, outputfolder, input_product='rateints', detector_options=None,
                image_options=None, preparation_options=None, ramp_fit_method='jwst',
                ramp_fit_options=None, **extraction_options):
    """Calibrate segments sequentially, then extract both methods for one visit."""
    from .ramp_fitting import validate_selection
    validate_selection(ramp_fit_method, ramp_fit_options)
    paths=inputs['rampstep'] if isinstance(inputs,dict) else inputs
    preparation_options=dict(preparation_options or {})
    preparation_options.setdefault('aperture_radius',extraction_options.get('aperture_radius',5.))
    prepared=[]
    for path in paths:
        if ramp_fit_method != 'jwst' and not isinstance(path, (str, Path)):
            raise ValueError('Alternative ramp fitting requires uncal file inputs, not already fitted models')
        if isinstance(path,(str,Path)):
            model=calibrate(path,Path(outputfolder)/'calibration',input_product=input_product,
                detector_options=detector_options,image_options=image_options,
                ramp_fit_method=ramp_fit_method,ramp_fit_options=ramp_fit_options)
            try:
                prepared.append(prepare_segment(model,**(preparation_options or {})))
            finally:
                model.close()
        else:
            if input_product=='calints':
                prepared.append(prepare_segment(path,**(preparation_options or {})))
            elif input_product=='rateints':
                model=calibrate(path,Path(outputfolder)/'calibration',input_product=input_product,
                    image_options=image_options)
                try:
                    prepared.append(prepare_segment(model,**(preparation_options or {})))
                finally:
                    model.close()
            else:
                raise ValueError('In-memory uncal inputs are not supported; pass filenames')
    return extract_visit(prepared,outputfolder=outputfolder,**extraction_options)
