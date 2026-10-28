"""PRISM vs SOSS transmission spectra of WASP-39 b (outputs of wasp39_spectrum.py).

Common 0.04 um bins: SOSS order 2 for 0.60-0.84 um, order 1 for 0.88-2.84 um. Per pair
(PRISM variant, SOSS variant): PRISM - SOSS per bin (errors added in quadrature), its
weighted mean and the chi2 about it, in the PRISM saturated range (bin centre < 2.06 um,
Carter, May et al. 2024: ~0.63-2.06 um) and above it (2.06-2.84 um).
  python wasp39_compare.py <run folder> <out.png> <out.json>
"""
import sys, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

run, png, out_json = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
SATURATION = 2.06


def soss(variant):
    parts = [np.load(run/'soss'/f'spectrum_{variant}_order{o}.npz') for o in (2, 1)]
    edges = np.concatenate([p['edges'][:-1] for p in parts])
    return (np.concatenate([p['centres'] for p in parts]), edges,
            np.concatenate([p['depth'] for p in parts]), np.concatenate([p['error'] for p in parts]))


def prism(variant):
    p = np.load(run/'prism'/f'spectrum_{variant}.npz')
    return p['centres'], p['edges'][:-1], p['depth'], p['error']


def stats(d, e):
    w = 1/e**2
    mean = np.sum(w*d)/np.sum(w)
    return dict(mean_ppm=1e6*mean, mean_error_ppm=1e6/np.sqrt(np.sum(w)),
                chi2=float(np.sum(((d - mean)/e)**2)), bins=int(len(d)),
                rms_ppm=1e6*float(np.std(d - mean)))


results, curves = {}, {}
for pv in ('crds', 'self'):
    pc, pe, pd, perr = prism(pv)
    for sv in ('crds', 'self'):
        sc, se, sd, serr = soss(sv)
        common = [(i, j) for i, a in enumerate(pe) for j, b in enumerate(se) if abs(a - b) < 1e-6]
        i, j = np.array(common).T
        ok = np.isfinite(pd[i]) & np.isfinite(sd[j])
        i, j = i[ok], j[ok]
        centre = sc[j]
        diff, err = pd[i] - sd[j], np.hypot(perr[i], serr[j])
        key = f'PRISM {pv} - SOSS {sv}'
        results[key] = {'saturated (<2.06 um)': stats(diff[centre < SATURATION], err[centre < SATURATION]),
                        'unsaturated (2.06-2.84 um)': stats(diff[centre >= SATURATION],
                                                            err[centre >= SATURATION]),
                        'all': stats(diff, err)}
        curves[key] = (centre, diff, err)
        print(key, json.dumps({k: {kk: round(vv, 1) for kk, vv in v.items()}
                               for k, v in results[key].items()}))
Path(out_json).write_text(json.dumps(results, indent=1))

fig, axes = plt.subplots(2, 1, figsize=(11, 9), sharex=True)
styles = {'crds': dict(color='tab:red', label='CRDS linearity'),
          'self': dict(color='tab:blue', label='self linearity')}
for pv in ('crds', 'self'):
    c, _, d, e = prism(pv)
    m = (c < 2.9)
    axes[0].errorbar(c[m], 1e6*d[m], 1e6*e[m], fmt='o', ms=3, lw=0.7, mfc='none',
                     color=styles[pv]['color'], label=f"PRISM, {styles[pv]['label']}")
for sv, marker in (('crds', 's'), ('self', 'D')):
    c, _, d, e = soss(sv)
    axes[0].errorbar(c + 0.008, 1e6*d, 1e6*e, fmt=marker, ms=3, lw=0.7,
                     color='k' if sv == 'self' else 'grey',
                     label=f"SOSS, {styles[sv]['label']}")
for pv in ('crds', 'self'):
    c, d, e = curves[f'PRISM {pv} - SOSS self']
    axes[1].errorbar(c + (0.006 if pv == 'self' else -0.006), 1e6*d, 1e6*e, fmt='o', ms=3,
                     lw=0.7, color=styles[pv]['color'],
                     label=f"PRISM {styles[pv]['label']} - SOSS self linearity")
for ax in axes:
    ax.axvspan(0.6, SATURATION, color='orange', alpha=0.1, lw=0)
axes[1].axhline(0, color='k', lw=0.5)
axes[0].set_ylabel('Transit depth (ppm)')
axes[1].set_ylabel('PRISM - SOSS (ppm)')
axes[1].set_xlabel('Wavelength (µm)')
axes[0].set_title('WASP-39 b: NIRSpec PRISM vs NIRISS/SOSS (shaded: PRISM saturated range)')
axes[0].legend(fontsize=8, ncol=2)
axes[1].legend(fontsize=8)
fig.tight_layout()
fig.savefig(png, dpi=130)
print(png)
