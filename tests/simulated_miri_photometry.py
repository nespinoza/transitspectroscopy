"""Save noisy MIRI-like two-visit injection/recovery figures and metrics.

python tests/simulated_miri_photometry.py --output validation/runs/miri_injection
Uses a pixel-integrated Airy PSF, not the fitted GP, as the generative model.
This is an algorithm test, not simulated JWST detector calibration or ephemerides.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.special import j1
import matplotlib.pyplot as plt
import _support  # noqa: F401
from transitspectroscopy import miri


def simulate(visit,n=1200):
    rng=np.random.default_rng(92340+visit)
    t=np.linspace(0,8,n)
    centers=np.column_stack((10+.08*np.sin(t*3.1),10+.06*np.cos(t*2.7)))
    def event(mid,duration):
        return np.clip((duration/2-abs(t-mid))/.12,0,1)
    templates=np.column_stack((event(2.,1.),event(5.8,1.1)))
    depths=np.array([.008,.0003]) if visit==1 else np.array([.0006,.0003])
    signal=1-templates@depths
    data=[];variances=[];y,x=np.indices((21,21),dtype=float)
    offsets=(np.arange(5)+.5)/5-.5
    for i,c in enumerate(centers):
        profile=np.zeros_like(x)
        for dy in offsets:
            for dx in offsets:
                r=np.hypot((x+dx-c[0])/1.03,(y+dy-c[1])/.97)*.78
                airy=np.divide(2*j1(r),r,out=np.ones_like(r),where=r!=0)**2
                profile+=airy
        profile/=profile.sum()
        expected=1e5*signal[i]*profile
        variance=expected/100.+4.  # photon term in rate units + read/background noise
        data.append(expected+rng.normal(0,np.sqrt(variance)))
        variances.append(variance)
    data=np.asarray(data);var=np.asarray(variances)
    data[:,7,12]=np.nan  # persistent bad pixel
    data[90,10,10]=np.nan  # transient missing PSF core
    var[400,12,12]=0
    return dict(data=data,variance=var,centers=centers,widths=np.full((n,2),1.8),
        times=2460000.+visit+t/24,sky=np.zeros(n),sky_variance=np.zeros(n),sky_pixels=np.full(n,500),
        flags=np.zeros(n,int),origin=np.zeros(2),median_image=np.nanmedian(data,axis=0),
        unit='DN/s',exposure='synthetic',metadata=dict(filter='F1500W',program='simulation',
        observation=str(visit),visit='1')),templates,depths


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    report={}
    for visit in (1,2):
        s,templates,depths=simulate(visit)
        out=args.output/f'visit{visit}'
        result=miri.extract_visit([s],aperture_radius=6,training_integrations=12,
            gp_options=dict(n_inducing=21,length_scale=1.,quadrature=3,maxiter=600),outputfolder=out)
        fig,axes=plt.subplots(2,1,figsize=(12,7),sharex=True,layout='constrained')
        report[str(visit)]={}
        hours=(s['times']-s['times'][0])*24
        truth=1-templates@depths
        for j,method in enumerate(('sum','gp')):
            f,e=result['flux'][:,j],result['error'][:,j]
            norm=np.nanmedian(f[(templates==0).all(axis=1)]);f,e=f/norm,e/norm
            good=np.isfinite(f)&np.isfinite(e)&(e>0)
            design=np.column_stack((np.ones(len(f)), -templates))
            w=1/e[good]**2
            cov=np.linalg.inv(design[good].T@(w[:,None]*design[good]))
            fit=cov@(design[good].T@(w*f[good]))
            recovered=fit[1:]/fit[0]
            jac=np.zeros((2,3));jac[:,0]=-fit[1:]/fit[0]**2;jac[:,1:]=np.eye(2)/fit[0]
            errors=np.sqrt(np.diag(jac@cov@jac.T))
            metrics=dict(injected_depth_ppm=(depths*1e6).tolist(),recovered_depth_ppm=(recovered*1e6).tolist(),
                depth_error_ppm=(errors*1e6).tolist(),difference_sigma=((recovered-depths)/errors).tolist(),
                residual_rms_ppm=float(np.nanstd((f-truth)*1e6)),valid_integrations=int(good.sum()))
            report[str(visit)][method]=metrics
            axes[j].plot(hours,f,'.',ms=1,alpha=.3,label=method)
            axes[j].plot(hours,truth,'k-',lw=1,label='Injected')
            axes[j].set(ylabel='Relative flux',title=f'{method}: recovered depths '+', '.join(f'{v:.1f} ppm' for v in recovered*1e6))
            axes[j].legend()
            np.savetxt(out/f'{method}_injection.csv',np.column_stack((s['times'],truth,f,e)),delimiter=',',
                header='time_bjd_tdb,injected,measured,error',comments='')
        axes[-1].set_xlabel('Hours')
        fig.savefig(out/'injection_recovery.png',dpi=150);plt.close(fig)
    (args.output/'injection_metrics.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
