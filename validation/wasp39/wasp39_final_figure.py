"""WASP-39 b transmission spectrum: SOSS (self correction, joint optimal extraction; order 2
below 0.84 um, order 1 above) and PRISM (self correction, saturated core and the rows next to
it masked: 'self_grow1'), with PRISM - SOSS on their common 0.04 um bins.
  python wasp39_final_figure.py <results folder> <out.png>
"""
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

results, png = Path(sys.argv[1]), sys.argv[2]
SATURATION = 2.06
prism = np.load(results/'prism_spectrum_self_grow1.npz')
orders = {o: np.load(results/f'soss_spectrum_self_order{o}.npz') for o in (1, 2)}
soss_edges = np.concatenate([orders[2]['edges'][:-1], orders[1]['edges'][:-1]])
soss_centres = np.concatenate([orders[2]['centres'], orders[1]['centres']])
soss_depth = np.concatenate([orders[2]['depth'], orders[1]['depth']])
soss_error = np.concatenate([orders[2]['error'], orders[1]['error']])
p_edges = prism['edges'][:-1]
pairs = [(i, j) for i, a in enumerate(p_edges) for j, b in enumerate(soss_edges)
         if abs(a - b) < 1e-6 and np.isfinite(prism['depth'][i]) and np.isfinite(soss_depth[j])]
i, j = np.array(pairs).T
diff = prism['depth'][i] - soss_depth[j]
err = np.hypot(prism['error'][i], soss_error[j])
centre = soss_centres[j]

fig, (top, bottom) = plt.subplots(2, 1, figsize=(11, 7.5), sharex=True,
                                  gridspec_kw=dict(height_ratios=(2.2, 1)))
ok = np.isfinite(prism['depth'])
top.errorbar(prism['centres'][ok], 1e6*prism['depth'][ok], 1e6*prism['error'][ok], fmt='o',
             ms=3.5, lw=0.8, color='tab:blue', label='NIRSpec PRISM (self linearity, saturated '
             'core $\\pm$1 row masked)')
for o, marker, color in ((1, 'D', 'k'), (2, 's', 'tab:grey')):
    d = orders[o]
    use = np.isfinite(d['depth'])
    use &= (d['centres'] < 0.86) if o == 2 else (d['centres'] > 0.86)
    top.errorbar(d['centres'][use], 1e6*d['depth'][use], 1e6*d['error'][use], fmt=marker, ms=3.5,
                 lw=0.8, color=color, label=f'NIRISS/SOSS order {o} (self linearity)')
top.set_ylabel('Transit depth (ppm)')
top.legend(fontsize=9, loc='upper left')
top.set_title('WASP-39 b (orbit of Carter, May et al. 2024); shaded: PRISM saturated range')
w = 1/err**2
for lo, hi, label in ((0, SATURATION, '< 2.06 µm'), (SATURATION, 9, '2.06-2.84 µm')):
    m = (centre >= lo) & (centre < hi)
    mean = np.sum(w[m]*diff[m])/np.sum(w[m])
    chi2 = np.sum(((diff[m] - mean)/err[m])**2)
    bottom.plot([], [], ' ', label=f'{label}: mean {1e6*mean:+.0f} $\\pm$ '
                f'{1e6/np.sqrt(np.sum(w[m])):.0f} ppm, $\\chi^2$ {chi2:.0f} / {m.sum() - 1} dof')
bottom.errorbar(centre, 1e6*diff, 1e6*err, fmt='o', ms=3.5, lw=0.8, color='tab:blue')
bottom.axhline(0, color='k', lw=0.6)
bottom.set_ylabel('PRISM - SOSS (ppm)')
bottom.set_xlabel('Wavelength (µm)')
bottom.legend(fontsize=9, loc='upper right', handlelength=0)
for ax in (top, bottom):
    ax.axvspan(0.6, SATURATION, color='orange', alpha=0.1, lw=0)
    ax.set_xlim(0.55, 5.4)
fig.tight_layout()
fig.savefig(png, dpi=140)
print(png)
