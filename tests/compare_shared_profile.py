"""Noisy joint training and held-out extraction with moving detector traces."""
import argparse
import json
from pathlib import Path

import _support  # noqa: F401
import numpy as np
from transitspectroscopy import fit_shared_profile
from simulated_spectra import simulate_sequence, simulate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path(__file__).parent/'shared_profile_comparison')
    args = parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    v = simulate_sequence()
    training = np.array([0,1,2,12,13,14,15])
    output, summaries, models = dict(v,training_indices=training), {}, {}
    for method in ['polynomial','gp']:
        model = fit_shared_profile(v['data'],v['variance'],v['centers'],training_indices=training,
                                   profile_method=method,aperture_radius=7,polynomial_order=3,
                                   gp_options={'n_inducing':12} if method == 'gp' else None,
                                   n_starts=2,hyper_maxiter=40)
        results = np.stack([model.extract(d,t,var) for d,t,var in
                            zip(v['data'],v['centers'],v['variance'])])
        profiles = np.stack([model.evaluate(t) for t in v['centers']])
        # The fitted flux is normalized within the selected whole-pixel aperture.
        # Keep full-source truth too, so aperture losses are not mistaken for
        # either model bias or a correction inferred by the extraction algorithm.
        aperture_truth = v['truth']*np.sum(v['profile']*(profiles > 0),axis=1)
        reference = aperture_truth/(1-v['depth']*v['transit'][:,None])
        residual = (results[:,1]-aperture_truth)*np.sqrt(results[:,2])
        lightcurve = np.mean(results[:,1]/reference,axis=1)
        error = np.sqrt(np.sum(1/results[:,2]/reference**2,axis=1)/reference.shape[1]**2)
        held = np.ones(len(results),dtype=bool)
        held[training] = False
        recovered = 1-np.mean(lightcurve[v['transit']])/np.mean(lightcurve[~v['transit']])
        summaries[method] = dict(options=model.options,training_indices=training.tolist(),
                                 normalized_residual_mean=float(np.mean(residual[held])),
                                 normalized_residual_std=float(np.std(residual[held])),
                                 full_source_normalized_residual_mean=float(np.mean(
                                     ((results[:,1]-v['truth'])*np.sqrt(results[:,2]))[held])),
                                 recovered_depth=float(recovered),injected_depth=v['depth'],
                                 fit_seconds=model.diagnostics['seconds'],
                                 boundary_parameters=model.diagnostics['boundary_parameters'])
        output.update({method+'_spectra':results,method+'_profiles':profiles,
                       method+'_aperture_truth':aperture_truth,
                       method+'_lightcurve':lightcurve,method+'_error':error})
        model.save(args.output/(method+'_model.npz'))
        models[method] = model
    np.savez_compressed(args.output/'simulation.npz',**output)
    np.savetxt(args.output/'lightcurves.csv',np.column_stack([np.arange(16),1-v['depth']*v['transit'],
                  output['polynomial_lightcurve'],output['polynomial_error'],
                  output['gp_lightcurve'],output['gp_error']]),delimiter=',',
               header='integration,truth,polynomial,polynomial_error,gp,gp_error',comments='')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes = plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    for method in ['polynomial','gp']:
        axes[0,0].errorbar(np.arange(16),output[method+'_lightcurve'],yerr=output[method+'_error'],
                          label=method,marker='.',capsize=2)
    axes[0,0].plot(1-v['depth']*v['transit'],'k--',label='Injected')
    axes[0,0].scatter(training,np.ones(len(training)),s=60,facecolors='none',edgecolors='k',label='Training')
    axes[0,0].set(title='Shared intrinsic profile; independent spectra',xlabel='Integration',
                  ylabel='Flux / injected out-of-transit aperture flux')
    axes[0,0].legend()
    for i in [0,1,2]:
        axes[0,1].plot(output['gp_profiles'][i,:,48],label=f'Integration {i}')
    axes[0,1].set(title='One GP model evaluated at three trace positions',xlabel='Detector row',ylabel='P at column 48')
    axes[0,1].legend()
    for method in ['polynomial','gp']:
        res = (output[method+'_spectra'][:,1]-output[method+'_aperture_truth'])*np.sqrt(output[method+'_spectra'][:,2])
        axes[1,0].plot(np.mean(res,axis=0),label=method)
    axes[1,0].axhline(0,color='k',lw=.5)
    axes[1,0].set(title='Recovery against independently simulated truth',xlabel='Column',ylabel='Mean residual / conditional error')
    axes[1,0].legend()
    diagnostics = [models['gp'].diagnose(d,t,var)['reduced_chi2'] for d,t,var in
                   zip(v['data'],v['centers'],v['variance'])]
    axes[1,1].plot(diagnostics,'o-',label='Stable intrinsic shape')
    # Deliberate violation: a time-variable broadened profile, via a detector-row
    # convolution of the noiseless expectation followed by independent noise.
    from scipy.ndimage import gaussian_filter1d
    changed = []
    for i in range(16):
        case = simulate(seed=900+i,ncolumns=96,evolving=True,noise=False)
        mean = gaussian_filter1d(case['data'],.7,axis=0)
        rng = np.random.default_rng(1200+i)
        observed = rng.poisson(mean)+rng.normal(0,5,mean.shape)
        changed.append(models['gp'].diagnose(observed,case['centers'],mean+25)['reduced_chi2'])
    axes[1,1].plot(changed,'x--',label='Broadened held-out shape')
    axes[1,1].set(title='Residual diagnostics expose profile changes',xlabel='Integration',ylabel='Conditional reduced chi-square')
    axes[1,1].legend()
    fig.savefig(args.output/'comparison.png',dpi=150)
    summaries['diagnostics'] = dict(stable=diagnostics,broadened=changed,
                                    note='New positive normalized models; errors condition on P. Recovery uses '
                                    'injected aperture flux; the comparison does not claim an inferred aperture correction.')
    (args.output/'summary.json').write_text(json.dumps(summaries,indent=2)+'\n')
    print(json.dumps(summaries,indent=2))


if __name__ == '__main__': main()
