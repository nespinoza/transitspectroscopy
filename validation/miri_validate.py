"""MIRI imaging validation with JWST calibration and local sum/2-D GP photometry."""
import argparse
from datetime import datetime, timezone
import glob
import gzip
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time
import traceback
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from diagnostics import smooth_series, write_json
from validate import ROOT, checkout_package, sha256


def download(name, folder, compress=False):
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


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'validation/config.miri-ltt1445.json')
    parser.add_argument('--visits',nargs='+',default=['1','2'])
    parser.add_argument('--input-product',choices=['uncal','rateints','calints'],default='rateints')
    parser.add_argument('--input-glob')
    parser.add_argument('--download',action='store_true')
    parser.add_argument('--compress-downloads',action='store_true',help='Retain losslessly compressed FITS inputs to reduce disk use')
    parser.add_argument('--data-dir',type=Path,default=ROOT/'validation/data/miri')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--reuse-prepared',type=Path,help='Directory with calibrated *_prepared.npz and provenance JSON')
    parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--temporary-downloads',action='store_true',help='Remove only newly downloaded files after verified cutouts are saved')
    parser.add_argument('--save-calints',action='store_true',help='Retain full calibrated cubes in addition to cutouts')
    args=parser.parse_args(argv)
    config=json.loads(args.config.read_text())
    if config.get('ramp_fit_method', 'jwst') != 'jwst' and args.input_product != 'uncal' and not args.reuse_prepared:
        raise ValueError('An alternative ramp fitter requires --input-product uncal')
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    folder=(args.output or ROOT/'validation/runs'/f'{stamp}_miri').resolve();folder.mkdir(parents=True,exist_ok=False)
    os.environ.setdefault('CRDS_PATH',str(Path.home()/'crds_cache'))
    os.environ.setdefault('CRDS_SERVER_URL','https://jwst-crds.stsci.edu')
    ts=checkout_package();start=time.monotonic()
    report=dict(status='running',mode='miri/imaging',config=config,command=sys.argv,
        started_utc=stamp,inputs=[],visits={},source_hashes={str(p.relative_to(ROOT)):sha256(p)
            for p in [ROOT/'src/miri.py',ROOT/'src/photometry.py',Path(__file__),ROOT/'validation/diagnostics.py']},
        software={k:importlib.metadata.version(k) for k in ('jwst','stdatamodels','numpy','scipy','astropy')})
    write_json(folder/'reduction.json',report)
    cache=folder/'prepared';cache.mkdir()
    try:
        for visit in args.visits:
            vconfig=config['visits'][visit]
            stems=[f'{e}-seg{s:03d}_mirimage' for e in vconfig['exposures'] for s in vconfig['segments']]
            if args.input_glob:
                paths=[Path(p) for p in sorted(glob.glob(args.input_glob)) if int(fits.getheader(p)['OBSERVTN'])==int(visit)]
                stems=[p.name.rsplit('_',1)[0] for p in paths]
            prepared=[]
            for stem in stems:
                cached=args.reuse_prepared/(stem+'_prepared.npz') if args.reuse_prepared else None
                if cached is not None:
                    meta=json.loads(cached.with_suffix('.json').read_text())
                    if sha256(cached)!=meta['prepared_sha256'] or meta['preparation']!=config['preparation']:
                        raise ValueError('Prepared data hash/settings mismatch')
                    if meta['image_options']!=config['image_options'] or meta['detector_options']!=config['detector_options']:
                        raise ValueError('Prepared calibration settings mismatch')
                    for key, default in (('ramp_fit_method', 'jwst'), ('ramp_fit_options', {})):
                        if meta.get(key, default) != config.get(key, default):
                            raise ValueError(f'Prepared calibration {key} mismatch')
                    record=meta['input'];s=ts.miri.load_prepared(cached)
                    record=dict(record,prepared_path=str(cached.resolve()),prepared_sha256=meta['prepared_sha256'],
                        preparation_manifest_path=str(cached.with_suffix('.json').resolve()),
                        preparation_manifest_sha256=sha256(cached.with_suffix('.json')),
                        preparation_source_hashes=meta['source_hashes'],preparation_software=meta['software'])
                else:
                    name=stem+'_'+args.input_product+'.fits'
                    path=next((p for p in paths if p.name in (name,name+'.gz')),None) if args.input_glob else args.data_dir/name
                    if path is None:
                        raise FileNotFoundError(name)
                    downloaded=False
                    if not path.exists() and path.with_suffix('.fits.gz').exists():
                        path=path.with_suffix('.fits.gz')
                    if not path.exists() and args.download:
                        path,downloaded=download(name,args.data_dir,args.compress_downloads)
                    record=inspect_input(path)
                    if record['program']!=config['program'] or record['observation']!=int(visit) or record['filter']!=config['filter']:
                        raise ValueError('Input does not match validation target/filter/visit')
                    print('Calibrating/preparing',name,flush=True)
                    model=ts.miri.calibrate(path,folder/'calibration',input_product=args.input_product,
                        detector_options=config['detector_options'],image_options=config['image_options'],save_calints=args.save_calints,
                        ramp_fit_method=config.get('ramp_fit_method', 'jwst'),ramp_fit_options=config.get('ramp_fit_options'))
                    try:
                        s=ts.miri.prepare_segment(model,**config['preparation'])
                    finally:
                        model.close()
                    saved=cache/(stem+'_prepared.npz');ts.miri.save_prepared(s,saved)
                    checksum=sha256(saved)
                    write_json(saved.with_suffix('.json'),dict(input=record,preparation=config['preparation'],
                        image_options=config['image_options'],detector_options=config['detector_options'],
                        ramp_fit_method=config.get('ramp_fit_method', 'jwst'),ramp_fit_options=config.get('ramp_fit_options', {}),
                        prepared_sha256=checksum,source_hashes=report['source_hashes'],software=report['software']))
                    if args.temporary_downloads and downloaded:
                        # Only files created by this invocation; never user-provided inputs.
                        ts.miri.load_prepared(saved)  # verify readable before discarding temporary input
                        path.unlink();record['temporary_download_removed']=True
                prepared.append(s);report['inputs'].append(record)
                write_json(folder/'reduction.json',report)
            if not args.prepare_only:
                out=folder/f'visit{visit}'
                result=ts.miri.extract_visit(prepared,outputfolder=out,**config['extraction'])
                report['visits'][visit]=plots(result,config,out,visit)
                expected={f'{e}-seg{s:03d}_mirimage' for e in vconfig['exposures'] for s in vconfig['segments']}
                report['visits'][visit]['all_configured_segments_selected']=set(stems)==expected
                report['visits'][visit]['selected_segments']=stems
                del result
            del prepared
        report['status']='complete'
        report['calibration_scope']='JWST Detector1 only for uncal input; Image2 for rateints; local extraction from calibrated native integrations'
        report['science_validation']='diagnostics_complete_not_certified; no inferred event identities or eclipse depths without ephemerides/systematics model'
    except BaseException as error:
        report['status']='failed';report['error']=dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc())
        raise
    finally:
        report['elapsed_seconds']=time.monotonic()-start
        report['products']=[dict(path=str(p.relative_to(folder)),bytes=p.stat().st_size,sha256=sha256(p))
                            for p in sorted(folder.rglob('*')) if p.is_file() and p.name!='reduction.json']
        write_json(folder/'reduction.json',report)
    print('MIRI validation report:',folder/'reduction.json',flush=True)


if __name__=='__main__':
    main()
