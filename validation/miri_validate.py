"""MIRI imaging validation with JWST calibration and local sum/2-D GP photometry.

The reduction of each visit is the run configuration
``validation/configs/miri-ltt1445a.toml`` (``ts.run``); this module adds the diagnostics.
Run it through ``validation/validate.py``.
"""
from copy import deepcopy
import gzip
from pathlib import Path
import sys

import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from diagnostics import smooth_series, write_json
from validate import sha256


def download(name, folder, compress=False):
    """Direct MAST download of one file (used by miri_ramp_compare.py; ts.run uses ts.jwst.download)."""
    import requests
    folder.mkdir(parents=True,exist_ok=True)
    path=folder/name
    if path.exists():
        return path,False
    compressed=folder/(name+'.gz')
    if compressed.exists():
        return compressed,False
    if compress:
        path=compressed
    temporary=path.with_suffix('.fits.part')
    print('Downloading',name,flush=True)
    with requests.get('https://mast.stsci.edu/api/v0.1/Download/file',
                     params={'uri':'mast:JWST/product/'+name},stream=True,timeout=120) as response:
        response.raise_for_status()
        received=0
        with (gzip.open(temporary,'wb',compresslevel=3) if compress else temporary.open('wb')) as handle:
            for chunk in response.iter_content(4*1024**2):
                handle.write(chunk)
                received+=len(chunk)
        expected=response.headers.get('Content-Length')
        if expected is not None and received!=int(expected):
            raise IOError('Incomplete MAST download')
    with fits.open(temporary) as hdus:
        hdus.verify('exception')
    temporary.replace(path)
    return path,True


def inspect_input(path):
    with fits.open(path) as h:
        header=h[0].header
        if header.get('INSTRUME')!='MIRI' or header.get('EXP_TYPE')!='MIR_IMAGE' or not header.get('TSOVISIT'):
            raise ValueError('Select MIRI imaging TSO science, excluding acquisition data')
        return dict(path=str(path.resolve()),sha256=sha256(path),bytes=path.stat().st_size,
            shape=list(h['SCI'].shape),program=int(header['PROGRAM']),observation=int(header['OBSERVTN']),
            filter=header['FILTER'],integration_start=header['INTSTART'],integration_end=header['INTEND'],
            calibration_version=header.get('CAL_VER'),crds_context=header.get('CRDS_CTX'))


def bin_series(times, flux, error, seconds):
    """Weighted fixed-duration bins, preserving empty gaps and bin counts."""
    index=np.floor((times-times[0])*86400/seconds).astype(int)
    rows=[]
    for j in np.unique(index):
        keep=(index==j)&np.isfinite(flux)&np.isfinite(error)&(error>0)
        if not keep.any():
            continue
        w=1/error[keep]**2
        rows.append([np.sum(times[keep]*w)/w.sum(),np.sum(flux[keep]*w)/w.sum(),
                     np.sqrt(1/w.sum()),keep.sum()])
    return np.asarray(rows).reshape(-1,4)


def plots(result, config, folder, visit):
    times=result['times'];hours=(times-times[0])*24
    report=dict(integrations=len(times),time_range_bjd_tdb=[float(times[0]),float(times[-1])],
        profile_fit=result['metadata']['gp'],unit=result['unit'],methods={},
        failed_centroids_or_backgrounds=int(np.sum((result['flags']&1)!=0)),
        median_profile_chi2_per_pixel=float(np.nanmedian(result['profile_chi2'])),
        status='diagnostics_complete_not_certified',event_description=config['visits'][visit]['description'])
    fig,axes=plt.subplots(1,3,figsize=(14,4),layout='constrained')
    for ax,key,title in zip(axes,['median_image','profile','weights'],
                            ['Median calibrated image','Normalized shared 2-D GP PSF','Optimal extraction weights']):
        arr=result[key];im=ax.imshow(arr,origin='lower',cmap='magma')
        ax.set_title(title);fig.colorbar(im,ax=ax)
    fig.savefig(folder/'01_miri_image_psf_weights.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(13,4),layout='constrained')
    for ax,key,title in zip(axes,['first_image','first_model','first_residual'],
                            ['First background-subtracted stamp','GP PSF × fitted flux','Residual / pixel error']):
        im=ax.imshow(result[key],origin='lower',cmap='RdBu_r' if key=='first_residual' else 'magma')
        ax.set_title(title);fig.colorbar(im,ax=ax)
    fig.savefig(folder/'05_miri_psf_residuals.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(3,1,figsize=(12,8),sharex=True,layout='constrained')
    for j,label in enumerate(('x','y')):
        axes[0].plot(hours,result['detector_centers'][:,j]-np.nanmedian(result['detector_centers'][:,j]),'.',ms=1,label=label)
        axes[1].plot(hours,2.35482*result['widths'][:,j],'.',ms=1,label=label)
    axes[0].set_ylabel('Centroid shift (pixel)');axes[0].legend()
    axes[1].set_ylabel('Gaussian core FWHM (pixel)')
    axes[2].plot(hours,result['sky'],'.',ms=1);axes[2].set(ylabel='Sky per pixel ('+result['unit']+')',xlabel='Hours')
    fig.savefig(folder/'02_miri_motion_background.png',dpi=150);plt.close(fig)
    np.savetxt(folder/'centroid_width_background.csv',np.column_stack((times,result['detector_centers'],
        2.35482*result['widths'],result['sky'],np.sqrt(result['sky_variance']),result['segment'])),delimiter=',',
        header='time_bjd_tdb,x,y,fwhm_x,fwhm_y,sky,sky_error,segment',comments='')
    fig,axes=plt.subplots(2,2,figsize=(14,8),sharex='col',layout='constrained')
    figbin,axbin=plt.subplots(2,1,figsize=(13,7),sharex=True,layout='constrained')
    # Smooth each exposure separately: do not bridge gaps or instrumental resets.
    for j,method in enumerate(('sum','gp')):
        f,e=result['flux'][:,j],result['error'][:,j]
        norm=np.nanmedian(f);f,e=f/norm,e/norm
        smooth=np.full(len(f),np.nan)
        for exposure in np.unique(result['segment']):
            sel=result['segment']==exposure
            # Running median is diagnostic only, never divided out of photometry.
            smooth[sel]=smooth_series(f[sel],config['white_light_filter_points'])
        residual=(f-smooth)*1e6
        valid=np.isfinite(residual)
        rms=float(np.sqrt(np.mean(residual[valid]**2)))
        report['methods'][method]=dict(valid_integrations=int(valid.sum()),
            median_flux=float(norm),median_filter_residual_rms_ppm=rms,
            median_formal_error_ppm=float(np.nanmedian(e)*1e6))
        axes[0,j].plot(hours,f,'.',ms=1,alpha=.3);axes[0,j].plot(hours,smooth,lw=1)
        axes[0,j].set(title=method+' photometry',ylabel='Relative flux')
        axes[1,j].plot(hours,residual,'.',ms=1);axes[1,j].set(xlabel='Hours',ylabel='Residual (ppm)',title=f'RMS {rms:.1f} ppm')
        np.savetxt(folder/f'{method}_filtered_lightcurve.csv',np.column_stack((times,f,e,smooth,residual)),
            delimiter=',',header='time_bjd_tdb,relative_flux,relative_error,running_median,residual_ppm',comments='')
        b=bin_series(times,f,e,config['bin_seconds'])
        axbin[j].errorbar((b[:,0]-times[0])*24,b[:,1],yerr=b[:,2],fmt='.',ms=4)
        axbin[j].set(ylabel='Relative flux',title=f'{method}: {config["bin_seconds"]:g}-second bins')
        np.savetxt(folder/f'{method}_binned_lightcurve.csv',b,delimiter=',',
            header='time_bjd_tdb,relative_flux,relative_error,count',comments='')
        for event in config.get('events',{}).get(visit,[]):
            t=(event['mid_bjd_tdb']-times[0])*24
            axbin[j].axvline(t,color='gray',ls='--');axbin[j].text(t,1,event['label'],transform=axbin[j].get_xaxis_transform(),rotation=90,va='top')
    axbin[-1].set_xlabel('Hours since BJD_TDB '+str(times[0]))
    fig.savefig(folder/'03_miri_lightcurves_residuals.png',dpi=150);plt.close(fig)
    figbin.savefig(folder/'04_miri_binned_events.png',dpi=150);plt.close(figbin)
    # Use exactly the same integrations to compare the extraction methods.
    a,b=result['flux'].T;common=np.isfinite(a)&np.isfinite(b)
    delta=(a[common]/np.median(a[common])-b[common]/np.median(b[common]))*1e6
    report['comparison']=dict(common_integrations=int(common.sum()),median_difference_ppm=float(np.median(delta)),
        rms_difference_ppm=float(np.sqrt(np.mean(delta**2))))
    np.savetxt(folder/'method_comparison.csv',np.column_stack((times[common],delta)),delimiter=',',
               header='time_bjd_tdb,sum_minus_gp_ppm',comments='')
    return report


def miri_settings(config):
    """Settings of the plotting functions (formerly config.miri-ltt1445.json)."""
    settings = deepcopy(config['validation'])
    settings['visits'] = {str(k): v for k, v in settings['visits'].items()}
    settings['events'] = {str(k): v for k, v in settings.get('events', {}).items()}
    return settings


def validate_miri(config, folder, report, ts, force=False, from_step=None, *, visits=None):
    """One ts.run per visit (its exposures and segments), then the MIRI diagnostics."""
    settings = miri_settings(config)
    report['settings'] = settings
    report.update(inputs=[], visits={}, runs={})
    for visit in visits or sorted(settings['visits']):
        description = settings['visits'][visit]
        visit_config = deepcopy(config)
        visit_config['data'].update(observation=description['observation'],
                                    exposure=description['exposures'], segments=description['segments'])
        visit_config['run']['output'] = str(Path(config['run']['output'])/f'visit{visit}')
        run = ts.run(visit_config, force=force, from_step=from_step, keep_stage2=True)
        report['runs'][visit] = dict(output=str(run.output), steps={k: str(v) for k, v in run.steps.items()},
                                     reused=run.reused, inputs=run.files)
        records = [inspect_input(Path(path)) for path in run.files]
        for record in records:
            if (record['program'] != config['data']['program'] or record['observation'] != int(visit)
                    or record['filter'] != settings['filter']):
                raise ValueError('Input does not match validation target/filter/visit')
        report['inputs'] += records
        result = run.stage2['photometry']
        out = folder/f'visit{visit}'
        out.mkdir(parents=True, exist_ok=True)
        ts.miri.save_products(result, out)
        report['visits'][visit] = plots(result, settings, out, visit)
        stems = sorted(Path(path).name.rsplit('_', 1)[0] for path in run.files)
        expected = {f'{e}-seg{s:03d}_mirimage' for e in description['exposures'] for s in description['segments']}
        report['visits'][visit]['all_configured_segments_selected'] = set(stems) == expected
        report['visits'][visit]['selected_segments'] = stems
        write_json(folder/'reduction.json', report)
        del result, run
    report['calibration_scope'] = ('Detector1 by ts.jwst.load for uncal input; Image2 for rateints; local '
                                   'extraction from calibrated native integrations')
    report['science_validation'] = ('diagnostics_complete_not_certified; no inferred event identities or '
                                    'eclipse depths without ephemerides/systematics model')


if __name__ == '__main__':
    from validate import main
    main(['--mode', 'miri', *sys.argv[1:]])
