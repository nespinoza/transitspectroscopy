"""Three-order NIRISS/SOSS validation, including the PID 1541/1 transit.

The reduction is the run configuration ``validation/configs/soss-hatp14b.toml``
(``ts.run``); this module adds the diagnostics. Run it through ``validation/validate.py``.
"""
from pathlib import Path
import sys

import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from astropy.visualization import ImageNormalize, AsinhStretch

from diagnostics import smooth_series, write_json
from validate import sha256
from soss_transit import fit_transit, transmission_spectrum


def input_records(paths):
    records = []
    for path in paths:
        with fits.open(path) as hdus:
            h = hdus[0].header
            if (h.get('INSTRUME'),h.get('EXP_TYPE'),h.get('FILTER')) != ('NIRISS','NIS_SOSS','CLEAR'):
                raise ValueError('Select NIRISS/SOSS CLEAR science integrations')
            records.append(dict(path=str(path.resolve()), sha256=sha256(path),
                bytes=path.stat().st_size, shape=list(hdus['SCI'].shape),
                integration_start=h['INTSTART'],integration_end=h['INTEND'],
                exposure_start_mjd=h['EXPSTART'], exposure_nints=h['NINTS'],
                program=h.get('PROGRAM'), observation=h.get('OBSERVTN'),
                calibration_version=h.get('CAL_VER'),crds_context=h.get('CRDS_CTX')))
    records.sort(key=lambda r:r['integration_start'])
    for a,b in zip(records,records[1:]):
        if a['integration_end'] >= b['integration_start'] or a['exposure_start_mjd'] != b['exposure_start_mjd']:
            raise ValueError('Select nonoverlapping chronological segments of one exposure')
    return records


def plots(result, config, folder, report, *, transit=True, reference_spectrum=None):
    colors = {1:'tab:orange',2:'tab:blue',3:'tab:green'}
    from transitspectroscopy.soss import aperture_weights
    background_region = np.ones(result['median_rate'].shape, bool)
    for trace in result['orders'].values():
        background_region[:,trace['columns']] &= aperture_weights(background_region.shape[0],
            trace['trace'],config['extraction']['aperture_radius']+12)==0
    background_region[:,:4]=False
    background_region[:,-4:]=False
    background_region &= np.isfinite(result['median_rate'])
    report['background_diagnostics']={}
    for key in ('median_rate','background','median_background_subtracted'):
        values=result[key][background_region & np.isfinite(result[key])]
        report['background_diagnostics'][key]=dict(pixels=len(values),median_dn_per_second=float(np.median(values)),
            rms_about_median_dn_per_second=float(np.sqrt(np.mean((values-np.median(values))**2))))
    vmin,vmax = config['image_limits_dn_per_second']
    norm = ImageNormalize(vmin=vmin,vmax=vmax,stretch=AsinhStretch(.1))
    fig,axes = plt.subplots(4,1,figsize=(13,10),layout='constrained')
    for ax,key,title in zip(axes,['median_rate','background','median_background_subtracted','first_extracted_image'],
            ['Sampled median rate','Fitted static Stage 2 background','Background-subtracted median','First integration after background and 1/f correction']):
        im=ax.imshow(result[key],origin='lower',aspect='auto',norm=norm,cmap='magma')
        ax.set(title=title,ylabel='Detector row')
        fig.colorbar(im,ax=ax,label='DN/s')
    axes[-1].set_xlabel('Detector column')
    fig.savefig(folder/'01_soss_background.png',dpi=150);plt.close(fig)
    fig,ax=plt.subplots(figsize=(13,5),layout='constrained')
    ax.imshow(result['median_background_subtracted'],origin='lower',aspect='auto',norm=norm,cmap='gray')
    for order,t in result['orders'].items():
        ax.plot(t['columns'],t['reference_trace'],'--',color=colors[order],lw=.8)
        ax.plot(t['columns'],t['trace'],color=colors[order],lw=1.1,label=f'Order {order}')
    ax.set(xlabel='Detector column',ylabel='Detector row',title='PASTASOSS reference (dashed) and refined traces (solid)')
    ax.legend();fig.savefig(folder/'02_soss_traces.png',dpi=150);plt.close(fig)
    hours=(result['times']-result['times'][0])*24
    fig,axes=plt.subplots(2,1,figsize=(12,7),sharex=True,layout='constrained')
    for order,t in result['orders'].items():
        axes[0].plot(hours,t['shift'],'.',ms=2,color=colors[order],label=f'Order {order}')
        axes[1].plot(hours,t['width'],'.',ms=2,color=colors[order])
    axes[0].set_ylabel('Cross-dispersion shift (pixels)');axes[0].legend()
    axes[1].set(xlabel='Hours',ylabel='Profile second-moment width (pixels)')
    fig.savefig(folder/'03_soss_motion.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(3,1,figsize=(12,9),layout='constrained')
    for (order,t),ax in zip(result['orders'].items(),axes):
        import warnings
        methods = {result['metadata'].get('method', 'box'): result,
                   **result.get('optimal', {})}
        for method, product in methods.items():
            with warnings.catch_warnings():
                warnings.simplefilter('ignore',RuntimeWarning)
                trace = product['orders'][order]
                spectrum=np.nanmedian(trace['flux'],axis=0)
            ax.plot(trace['wavelength'],spectrum,lw=.8,
                label='Box' if method == 'box' else f'Optimal {method} (shared profile)')
            keep=trace['white_light_columns']
            np.savetxt(folder/f'order{order}_{method}_median_spectrum.csv',
                np.column_stack([trace['columns'],trace['wavelength'],spectrum,keep]),
                delimiter=',',header='column,wavelength_microns,median_flux_dn_per_second,white_light_column',comments='')
        ax.set(title=f'Order {order}',ylabel='DN/s',xlabel='Wavelength (µm)');ax.legend(fontsize=8)
    fig.savefig(folder/'04_soss_spectra.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(16,8),sharex=True,layout='constrained')
    report['orders']={}
    for column,(order,t) in enumerate(result['orders'].items()):
        filtered=smooth_series(t['white_light'],config['white_light_filter_points'])
        residual=(t['white_light']-filtered)*1e6
        rms=float(np.sqrt(np.mean(residual**2)))
        axes[0,column].errorbar(hours,t['white_light'],yerr=t['white_light_error'],fmt='.',ms=2,alpha=.5,color=colors[order])
        axes[0,column].plot(hours,filtered,'k-',lw=1)
        axes[0,column].set(title=f'Order {order}',ylabel='Relative flux')
        axes[1,column].plot(hours,residual,'.',ms=2,color=colors[order])
        axes[1,column].set(title=f'100-point filter RMS: {rms:.1f} ppm',xlabel='Hours',ylabel='Residual (ppm)')
        np.savetxt(folder/f'order{order}_filtered_lightcurve.csv',np.column_stack([result['times'],t['white_light'],
            t['white_light_error'],filtered,residual]),delimiter=',',
            header='time_bjd_tdb,relative_flux,relative_error,median_filter,residual_ppm',comments='')
        keep=t['white_light_columns']
        report['orders'][str(order)]=dict(fixed_columns=int(keep.sum()),
            wavelength_range_microns=[float(t['wavelength'][keep].min()),float(t['wavelength'][keep].max())],
            median_filter_residual_rms_ppm=rms,
            trace_shift_rms_pixels=float(np.nanstd(t['shift'])),
            profile_width_rms_pixels=float(np.nanstd(t['width'])),
            trace_reference_offset_median_pixels=float(np.median(t['trace']-t['reference_trace'])))
        if 'joint_covariance' in result:
            report['orders'][str(order)]['jointly_recovered_overlap_columns']=int(np.sum(keep & ~t['isolated']))
    fig.savefig(folder/'05_soss_white_light.png',dpi=150);plt.close(fig)
    if not transit:
        report['science_validation']=dict(status='not_run',reason='Transit fitting explicitly skipped')
        return
    settings=config['transit']
    trim=settings['trim_first_integrations']
    fig,axes=plt.subplots(2,3,figsize=(16,8),sharex=True,layout='constrained')
    transmission={}
    for column,(order,t) in enumerate(result['orders'].items()):
        print('Fitting white light and transmission spectrum for order',order,flush=True)
        m,arrays=fit_transit(result['times'][trim:],t['white_light'][trim:],t['white_light_error'][trim:],settings,fit_time=True)
        report['orders'][str(order)]['transit_fit']=m
        reference=settings['reference_white_depth_ppm'].get(str(order))
        if reference:
            m['reference_depth_ppm'],m['reference_error_ppm']=reference
            m['reference_difference_sigma']=(m['depth_ppm']-reference[0])/np.hypot(m['depth_error_ppm'],reference[1])
        np.savetxt(folder/f'order{order}_transit_fit.csv',np.column_stack(list(arrays.values())),
            delimiter=',',header=','.join(arrays),comments='')
        x=hours[trim:]
        axes[0,column].plot(x,arrays['normalized_flux'],'.',ms=2,color=colors[order])
        axes[0,column].plot(x,arrays['model'],'k-',lw=1)
        axes[0,column].set(title=f'Order {order}: Rp²/Rs² = {m["depth_ppm"]:.0f} ± {m["depth_error_ppm"]:.0f} ppm',ylabel='Relative flux')
        axes[1,column].plot(x,arrays['residual_ppm'],'.',ms=2,color=colors[order])
        axes[1,column].set(title=f'Fit residual RMS: {m["residual_rms_ppm"]:.1f} ppm',xlabel='Hours',ylabel='Residual (ppm)')
        table,details=transmission_spectrum(result['times'],t,settings,t0=m['t0_bjd_tdb'])
        transmission[order]=table
        np.savetxt(folder/f'order{order}_transmission.csv',table,delimiter=',',header=
            'wavelength_microns,lower_microns,upper_microns,columns,depth_ppm,depth_error_ppm,residual_rms_ppm,beta,boundary_parameters,fit_failed',comments='')
        write_json(folder/f'order{order}_transmission_fits.json',details)
        finite=np.isfinite(table[:,5])&(table[:,5]>0)
        weights=1/table[finite,5]**2
        mean=np.sum(table[finite,4]*weights)/weights.sum() if weights.size else np.nan
        report['orders'][str(order)]['transmission']=dict(bins=len(table),
            failed_bins=int(np.sum(table[:,9]>0)),
            unconstrained_bins=int(np.sum(~np.isfinite(table[:,5]) | (table[:,5]>=table[:,4]))),
            finite_uncertainty_bins=int(finite.sum()),weighted_depth_ppm=float(mean),
            rms_about_weighted_depth_ppm=float(np.sqrt(np.mean((table[finite,4]-mean)**2))),
            boundary_bins=int(np.sum(table[:,8]>0)),
            interpretation='conditional diagnostic: free quadratic limb darkening per bin; fixed geometry; no common-mode correction')
    fig.savefig(folder/'06_soss_transit_fits.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(2,1,figsize=(12,9),sharex=True,layout='constrained')
    mean,unc=settings['reference_soss_weighted_depth_ppm']
    for ax in axes:
        for order,table in transmission.items():
            ax.errorbar(table[:,0],table[:,4],yerr=table[:,5],fmt='o',ms=3,color=colors[order],label=f'Order {order}')
        ax.axhline(mean,color='gray',ls='--',label='Published SOSS aggregate depth (context only)')
        ax.axhspan(mean-unc,mean+unc,color='gray',alpha=.15)
    if reference_spectrum is not None:
        ref=np.genfromtxt(reference_spectrum,delimiter=',',names=True)
        required={'wavelength_microns','depth_ppm','depth_error_ppm'}
        if not required <= set(ref.dtype.names):
            raise ValueError('Reference CSV requires wavelength_microns, depth_ppm, depth_error_ppm')
        for ax in axes:
            ax.errorbar(ref['wavelength_microns'],ref['depth_ppm'],yerr=ref['depth_error_ppm'],fmt='.',color='black',alpha=.5,label='Supplied reference spectrum')
        report['reference_spectrum']=dict(path=str(Path(reference_spectrum).resolve()),sha256=sha256(reference_spectrum))
    axes[0].set(ylabel='Rp²/Rs² (ppm)',title='Full uncertainty range — weak order 3 bins are poorly constrained')
    axes[0].legend(fontsize=8)
    axes[1].set(xlabel='Wavelength (µm)',ylabel='Rp²/Rs² (ppm)',ylim=config.get('transmission_plot_limits_ppm',[5000,9000]),
                title='Zoom around the planetary depth (all bins retained in the upper panel and tables)')
    fig.savefig(folder/'07_soss_transmission.png',dpi=150);plt.close(fig)
    report['science_validation']=dict(status='diagnostics_complete_not_certified',
        reason='Transmission fits require review of saturation, order contamination, limb-darkening degeneracy and correlated noise; aggregate literature depth is not a wavelength-by-wavelength benchmark',
        reference=settings['reference'])


def compare_extractions(result, config, folder, report, *, transit=True):
    """Use identical, time-independent wavelengths for precision/depth comparisons."""
    methods = {'box': result, **result.get('optimal', {})}
    if len(methods) == 1:
        return
    hours=(result['times']-result['times'][0])*24
    fig, axes=plt.subplots(2,3,figsize=(17,9),sharex=True,layout='constrained')
    report['extraction_comparison'] = {}
    for column, order in enumerate(result['orders']):
        traces = {method: product['orders'][order] for method,product in methods.items()}
        common=np.logical_and.reduce([t['white_light_columns'] for t in traces.values()])
        if common.sum() < 10:
            raise ValueError(f'Order {order}: insufficient identical wavelengths for method comparison')
        reference=np.median(traces['box']['flux'][:,common],axis=0)
        metrics=dict(common_columns=int(common.sum()),methods={})
        report['extraction_comparison'][str(order)]=metrics
        np.savetxt(folder/f'order{order}_comparison_columns.csv',
            np.column_stack([traces['box']['columns'],traces['box']['wavelength'],common]),
            delimiter=',',header='column,wavelength_microns,comparison_column',comments='')
        for method,t in traces.items():
            flux=t['flux'][:,common].sum(axis=1)
            error=np.sqrt((t['error'][:,common]**2).sum(axis=1))
            norm=np.median(flux)
            flux,error=flux/norm,error/norm
            filtered=smooth_series(flux,config['white_light_filter_points'])
            residual=(flux-filtered)*1e6
            rms=float(np.sqrt(np.mean(residual**2)))
            spectrum=np.median(t['flux'][:,common],axis=0)
            m=dict(available_fixed_columns=int(t['white_light_columns'].sum()),
                median_filter_residual_rms_ppm=rms,
                median_spectrum_difference_from_box_dn_per_second=float(np.median(spectrum-reference)),
                median_relative_spectrum_difference_from_box=float(np.median((spectrum-reference)/reference)))
            metrics['methods'][method]=m
            axes[0,column].plot(hours,flux,'.',ms=1.5,alpha=.5,label=method)
            axes[0,column].plot(hours,filtered,lw=.8)
            axes[1,column].plot(hours,residual,'.',ms=1.5,alpha=.5,label=f'{method}: {rms:.1f} ppm')
            np.savetxt(folder/f'order{order}_{method}_matched_lightcurve.csv',
                np.column_stack([result['times'],flux,error,filtered,residual]),delimiter=',',
                header='time_bjd_tdb,relative_flux,relative_error,median_filter,residual_ppm',comments='')
            if transit:
                trim=config['transit']['trim_first_integrations']
                m['transit_fit'], arrays=fit_transit(result['times'][trim:],flux[trim:],error[trim:],
                    config['transit'],fit_time=True)
                np.savetxt(folder/f'order{order}_{method}_matched_transit_fit.csv',
                    np.column_stack(list(arrays.values())),delimiter=',',header=','.join(arrays),comments='')
        axes[0,column].set(title=f'Order {order}: same {common.sum()} columns',ylabel='Relative flux')
        axes[1,column].set(xlabel='Hours',ylabel='Filter residual (ppm)')
        axes[0,column].legend(fontsize=8)
        axes[1,column].legend(fontsize=8)
    fig.savefig(folder/'08_soss_extraction_comparison.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(15,4),layout='constrained')
    for ax,order in zip(axes,result['orders']):
        for method,product in result.get('optimal',{}).items():
            diagnostic=product['metadata']['profile_training'][str(order)]
            training=diagnostic['training']
            values=[t['chi2']/t['valid_pixels'] for t in training]
            ax.plot(np.arange(len(values)),values,'o-',label=method)
            report['extraction_comparison'][str(order)]['methods'][method]['profile_training'] = dict(
                chi2_per_valid_pixel=values, integration_ids=diagnostic['integration_ids'],
                training_seconds=diagnostic['seconds'],
                interpretation='Conditional profile residual diagnostic, not degrees-of-freedom corrected; large values indicate model mismatch or underestimated pixel variance')
        ax.axhline(1,color='gray',ls='--')
        ax.set(title=f'Order {order}',xlabel='Training sample',ylabel='Profile chi² / valid pixel',yscale='log')
        ax.legend(fontsize=8)
    fig.savefig(folder/'09_soss_profile_fit_quality.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(3,1,figsize=(12,9),layout='constrained')
    for ax,order in zip(axes,result['orders']):
        box=result['orders'][order]
        common=np.logical_and.reduce([p['orders'][order]['white_light_columns'] for p in methods.values()])
        reference=np.median(box['flux'][:,common],axis=0)
        for method,product in result.get('optimal',{}).items():
            spectrum=np.median(product['orders'][order]['flux'][:,common],axis=0)
            relative=(spectrum-reference)/reference
            ax.plot(box['wavelength'][common],100*relative,'.',ms=3,label=method)
            np.savetxt(folder/f'order{order}_{method}_matched_spectrum_difference.csv',
                np.column_stack([box['columns'][common],box['wavelength'][common],
                    reference,spectrum,spectrum-reference,relative]),delimiter=',',
                header='column,wavelength_microns,box_median_dn_per_second,optimal_median_dn_per_second,difference_dn_per_second,relative_difference',comments='')
        ax.axhline(0,color='gray',ls='--')
        ax.set(title=f'Order {order}: identical fixed wavelengths',
            xlabel='Wavelength (µm)',ylabel='(Optimal − box) / box (%)')
        ax.legend(fontsize=8)
    fig.savefig(folder/'10_soss_spectral_differences.png',dpi=150);plt.close(fig)
    joint={method:p for method,p in result.get('optimal',{}).items() if 'joint_covariance' in p}
    if joint:
        fig,axes=plt.subplots(2,1,figsize=(12,7),sharex=True,layout='constrained')
        import warnings
        report['order_overlap']={}
        for method,p in joint.items():
            covariance=p['joint_covariance']
            denominator=np.sqrt(covariance[:,:,0,0]*covariance[:,:,1,1])
            correlation=np.divide(covariance[:,:,0,1],denominator,
                out=np.full_like(denominator,np.nan),where=denominator>0)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore',RuntimeWarning)
                correlation=np.nanmedian(correlation,axis=0)
                condition=np.nanmedian(p['joint_condition'],axis=0)
            axes[0].plot(correlation,label=method)
            axes[1].plot(condition,label=method)
            report['order_overlap'][method]=dict(
                minimum_order12_flux_correlation=float(np.nanmin(correlation)),
                maximum_finite_condition=float(np.max(condition[np.isfinite(condition)])),
                recovered_columns={str(o):int(np.sum(t['white_light_columns'] & ~t['isolated']))
                                   for o,t in p['orders'].items()},
                joint_profile_fit=p['metadata']['joint_fit'])
            np.savetxt(folder/f'{method}_order_overlap.csv',
                np.column_stack([np.arange(len(correlation)),correlation,condition]),delimiter=',',
                header='detector_column,median_order12_flux_correlation,median_scaled_normal_condition',comments='')
        axes[0].set(ylabel='Order 1–2 flux correlation',title='Conditional covariance from simultaneous extraction')
        axes[1].set(xlabel='Detector column',ylabel='Scaled normal condition',yscale='log')
        for ax in axes:
            ax.legend()
        fig.savefig(folder/'11_soss_order_overlap.png',dpi=150);plt.close(fig)
        reference=result.get('metadata',{}).get('pastasoss',{}).get('path')
        if reference and Path(reference).is_file():
            from stdatamodels.jwst.datamodels import PastasossModel
            with PastasossModel(reference) as ref:
                throughputs={int(t.spectral_order):(np.asarray(t.wavelength).copy(),
                    np.asarray(t.throughput).copy()) for t in ref.throughputs}
            fig,axes=plt.subplots(2,1,figsize=(12,7),sharex=True,layout='constrained')
            report['throughput_consistency']={}
            for method,p in joint.items():
                a,b=p['orders'][1],p['orders'][2]
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore',RuntimeWarning)
                    f1=np.nanmedian(a['flux'],axis=0)
                    f2=np.nanmedian(b['flux'],axis=0)
                w1,w2=a['wavelength'],b['wavelength']
                good=np.isfinite(f1)&a['white_light_columns']
                increasing=np.argsort(w1[good])
                density=f1[good]/np.abs(np.gradient(w1))[good]
                expected=np.interp(w2,w1[good][increasing],density[increasing],left=np.nan,right=np.nan)
                t1=np.interp(w2,*throughputs[1]);t2=np.interp(w2,*throughputs[2])
                expected*=t2/t1*np.abs(np.gradient(w2))
                keep=np.isfinite(expected)&(expected>0)&b['white_light_columns']
                ratio=np.divide(f2,expected,out=np.full_like(f2,np.nan),where=keep)
                axes[0].plot(w2[keep],f2[keep],label=f'{method}: order 2')
                axes[0].plot(w2[keep],expected[keep],'--',label=f'{method}: from order 1 and throughput')
                axes[1].plot(w2[keep],ratio[keep],label=method)
                red=keep&(w2>=1.25)
                blue=keep&(w2>=.9)&(w2<=1.1)
                report['throughput_consistency'][method]=dict(
                    ratio_median_09_to_11_microns=float(np.nanmedian(ratio[blue])),
                    ratio_median_above_125_microns=float(np.nanmedian(ratio[red])),
                    interpretation='Diagnostic only: reference relative throughput, dispersion width and order 1 predict order 2; ignores wavelength-dependent aperture losses and response uncertainties; no constraint applied to extraction')
                np.savetxt(folder/f'{method}_throughput_consistency.csv',
                    np.column_stack([w2,f2,expected,ratio]),delimiter=',',
                    header='wavelength_microns,order2_flux_dn_per_second,order2_predicted_dn_per_second,observed_to_predicted_ratio',comments='')
            axes[0].set(ylabel='DN/s',title='Order 2 consistency with reference throughput (diagnostic only)')
            axes[1].axhline(1,color='gray',ls='--')
            axes[1].set(xlabel='Wavelength (µm)',ylabel='Observed / predicted',ylim=(0,10))
            for ax in axes:
                ax.legend(fontsize=8)
            fig.savefig(folder/'12_soss_throughput_consistency.png',dpi=150);plt.close(fig)


def soss_settings(config):
    """Settings of the plotting/comparison functions (formerly config.soss-hatp14-optimal.json)."""
    extraction = config['extraction']['soss']
    return dict(config['validation'], extraction=dict(aperture_radius=extraction['aperture_radius']))


def validate_soss(config, folder, report, ts, force=False, from_step=None, *, transit=True,
                  reference_spectrum=None):
    """Reduce with ts.run, then add the SOSS validation diagnostics to ``report``."""
    settings = soss_settings(config)
    report['settings'] = settings
    run = ts.run(config, force=force, from_step=from_step)
    report['run'] = run_summary = dict(output=str(run.output), steps={k: str(v) for k, v in run.steps.items()},
                                       reused=run.reused, inputs=run.files)
    report['inputs'] = input_records([Path(path) for path in run.files])
    report['calibration'] = dict(input_product=config['data']['product'], interpretation=(
        'Reused public/local Stage 1 rateints; no fresh detector calibration or group-level validation'
        if config['data']['product'] == 'rateints' else 'Local detector calibration and ramp fitting'))
    write_json(folder/'reduction.json', report)
    # The SOSS Stage 2 saved its complete result; read it back (also when the step was reused).
    result = ts.soss.load_products(run.steps['extract']/'soss'/'pipeline_outputs'/'soss')
    ts.soss.save_products(result, folder)
    report['stages'] = result['metadata']
    report['loaded_integrations'] = len(result['times'])
    from stdatamodels.jwst.datamodels import ImageModel
    for key in ('median_rate', 'background', 'median_background_subtracted'):
        with ImageModel(data=result[key].astype('float32')) as image:
            image.meta.instrument.name = 'NIRISS'
            image.meta.instrument.detector = 'NIS'
            image.meta.instrument.filter = 'CLEAR'
            image.meta.instrument.pupil = 'GR700XD'
            image.meta.exposure.type = 'NIS_SOSS'
            image.meta.bunit_data = 'DN/s'
            if key == 'median_background_subtracted':
                image.meta.cal_step.bkg_subtract = 'COMPLETE'
            image.save(folder/f'{key}.fits')
    if transit:
        if any(str(r['program']).lstrip('0') != '1541' or str(r['observation']).lstrip('0') != '1'
               for r in report['inputs']):
            raise ValueError('The transit preset is only for PID 1541 observation 1; use --skip-transit')
    plots(result, settings, folder, report, transit=transit, reference_spectrum=reference_spectrum)
    report['optimal'] = {}
    for method, product in result.get('optimal', {}).items():
        report['optimal'][method] = {}
        plots(product, settings, folder/'optimal'/method, report['optimal'][method], transit=transit,
              reference_spectrum=reference_spectrum)
    compare_extractions(result, settings, folder, report, transit=transit)
    return run_summary


if __name__ == '__main__':
    from validate import main
    main(['--mode', 'soss', *sys.argv[1:]])
