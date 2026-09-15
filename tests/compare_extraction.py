"""Generate noisy C/Python/GP comparison data, figures and numerical summaries.

Run after build_c_reference.py, with its output directory on PYTHONPATH.
Requires the REAL C backend; never substitutes Python for a missing reference.
"""
import argparse
import json
from pathlib import Path
import platform
import time

import _support  # noqa: F401
import numpy as np
import scipy
from simulated_spectra import simulate
from transitspectroscopy import spectroscopy as s


def compare(name, output, seed, evolving=False, cosmic=False):
    v = simulate(seed=seed, evolving=evolving, cosmic=cosmic)
    results, elapsed = {}, {}
    for key, backend, method in [('c', 'c', 'polynomial'), ('python', 'python', 'polynomial'),
                                  ('gp', 'python', 'gp')]:
        start = time.perf_counter()
        results[key] = s.getOptimalSpectrum(v['data'], v['centers'], 7, 5., 1., 8., 1., 3,
                          data_variance=v['variance'], return_P=True,
                          backend=backend, profile_method=method)
        elapsed[key] = time.perf_counter()-start
    simple = s.getFastSimpleSpectrum(v['data'], v['centers'], 7, backend='python')
    payload = dict(v, simple=simple)
    for key, (spectrum, profile) in results.items():
        payload[key+'_spectrum'], payload[key+'_profile'] = spectrum, profile
    np.savez_compressed(output / (name+'.npz'), **payload)
    c, cp = results['c']
    p, pp = results['python']
    difference = p[1]-c[1]
    summary = {'seed': seed, 'evolving': evolving, 'cosmic': cosmic,
               'max_abs_flux_difference': float(np.max(abs(difference))),
               'max_flux_difference_in_c_sigma': float(np.max(abs(difference)*np.sqrt(c[2]))),
               'max_abs_profile_difference': float(np.max(abs(pp-cp))),
               'max_abs_inverse_variance_difference': float(np.max(abs(p[2]-c[2]))),
               'seconds': elapsed,
               'normalized_residuals': {key: {'mean': float(np.mean((spec[1]-v['truth'])*np.sqrt(spec[2]))),
                                             'std': float(np.std((spec[1]-v['truth'])*np.sqrt(spec[2])))}
                                        for key, (spec, _) in results.items()}}
    np.savetxt(output / (name+'.csv'), np.column_stack([v['x'], v['truth'], simple,
                 c[1], p[1], results['gp'][0][1], difference, np.sqrt(1./c[2]), np.sqrt(1./p[2])]),
               delimiter=',', header='column,truth,simple,c_flux,python_flux,gp_flux,python_minus_c,c_error,python_error', comments='')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 2, figsize=(13, 10), constrained_layout=True)
    colors = {'c':'C1', 'python':'C2', 'gp':'C3'}
    axes[0, 0].imshow(v['data'], origin='lower', aspect='auto')
    axes[0, 0].plot(v['x'], v['centers'], color='white', lw=.8)
    axes[0, 0].set(title='Noisy detector data and true trace', ylabel='Detector row')
    axes[0, 1].plot(v['x'], v['truth'], 'k', label='Injected flux')
    axes[0, 1].plot(v['x'], simple, alpha=.4, label='Simple')
    for key, (spec, _) in results.items():
        axes[0, 1].plot(v['x'], spec[1], label=key, color=colors[key], lw=1, linestyle='--' if key=='python' else '-')
    axes[0, 1].set(title='Recovered spectra', ylabel='Flux')
    axes[0, 1].legend()
    axes[1, 0].plot(v['x'], difference)
    axes[1, 0].set(title='Python polynomial minus C (absolute)', ylabel='Flux difference')
    axes[1, 0].ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    axes[1, 1].plot(v['x'], np.sqrt(1./p[2])-np.sqrt(1./c[2]))
    axes[1, 1].set(title='Python polynomial minus C uncertainty', ylabel='Error difference')
    im = axes[2, 0].imshow(pp-cp, origin='lower', aspect='auto', cmap='coolwarm')
    axes[2, 0].set(title='Python polynomial minus C profile', ylabel='Detector row')
    fig.colorbar(im, ax=axes[2, 0])
    for key, (spec, _) in results.items():
        axes[2, 1].plot(v['x'], (spec[1]-v['truth'])*np.sqrt(spec[2]), label=key, color=colors[key], alpha=.7)
    axes[2, 1].axhline(0., color='k', lw=.5)
    axes[2, 1].set(title='Recovery against injected truth', ylabel='Residual / conditional error')
    axes[2, 1].legend()
    for ax in axes.flat:
        ax.set_xlabel('Detector column')
    fig.suptitle(name.replace('_', ' ') + f' — seed {seed}; Poisson + read noise')
    fig.savefig(output / (name+'.png'), dpi=150)
    plt.close(fig)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path(__file__).parent / 'extraction_comparison')
    args = parser.parse_args()
    import Marsh  # require C before writing any artifacts
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    summaries = {name: compare(name, output, seed, evolving, cosmic)
                 for name, seed, evolving, cosmic in [
                     ('smooth_noisy', 123, False, False),
                     ('varying_profile_cosmics', 456, True, True)]}
    build_manifest = Path(Marsh.__file__).with_name('build_manifest.json')
    metadata = {'numpy': np.__version__, 'scipy': scipy.__version__, 'platform': platform.platform(),
                'c_build': json.loads(build_manifest.read_text()) if build_manifest.exists() else
                           {'module_path': Marsh.__file__, 'provenance': 'build manifest unavailable'},
                'parameters': {'aperture_radius':7, 'nsigma':8, 'polynomial_spacing':1., 'polynomial_order':3,
                               'gp': {'kernel':'matern32', 'length_scale':19.2, 'amplitude':1., 'n_inducing':16}},
                'cases': summaries}
    (output / 'summary.json').write_text(json.dumps(metadata, indent=2)+'\n')
    print(json.dumps(summaries, indent=2))


if __name__ == '__main__':
    main()
