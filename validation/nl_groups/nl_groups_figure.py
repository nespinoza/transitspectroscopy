"""Figure of the group-number study: spectra, N-group differences, and self - CRDS (measured
and predicted) per detector, from nl_groups_spectrum.py and nl_groups_predict.py outputs.
  python nl_groups_figure.py <out.png> <detector>:<spectrum.npz>:<predicted.npz> ...
"""
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

out, entries = sys.argv[1], [a.split(':') for a in sys.argv[2:]]
colors = {3: 'tab:red', 10: 'tab:orange', 20: 'tab:blue'}
fig, axes = plt.subplots(3, 1, figsize=(10, 11), sharex=True)
for detector, spectrum_file, predicted_file in entries:
    s, p = np.load(spectrum_file), np.load(predicted_file)
    w = s[f'{detector}__wavelength']
    for n, color in colors.items():
        for corr, marker, shift in (('crds', 'o', -0.004), ('self', 's', 0.004)):
            d, e = s[f'{detector}__depth_{corr}_n{n}'], s[f'{detector}__error_{corr}_n{n}']
            label = f'{corr.upper() if corr == "crds" else "self"}, {n} groups' \
                if detector == entries[0][0] else None
            axes[0].errorbar(w + shift, 1e6*d, 1e6*e, fmt=marker, color=color, ms=3, lw=0.6,
                             mfc='none' if corr == 'self' else color, label=label)
        if n != 20:
            for corr, marker, shift in (('crds', 'o', -0.004), ('self', 's', 0.004)):
                key = f'{corr}_n{n}-{corr}_n20'
                d, e = s[f'{detector}__diff_{key}'], s[f'{detector}__differr_{key}']
                axes[1].errorbar(w + shift, 1e6*d, 1e6*e, fmt=marker, color=color, ms=3, lw=0.6,
                                 mfc='none' if corr == 'self' else color,
                                 label=(f'{corr} {n} - {corr} 20 groups'
                                        if detector == entries[0][0] else None))
        key = f'self_n{n}-crds_n{n}'
        d, e = s[f'{detector}__diff_{key}'], s[f'{detector}__differr_{key}']
        axes[2].errorbar(w, 1e6*d, 1e6*e, fmt='o', color=color, ms=3, lw=0.6,
                         label=f'{n} groups, measured' if detector == entries[0][0] else None)
        factor = p[f'{detector}__crds_n{n}'][:, :-1].mean(0)
        predicted = s[f'{detector}__depth_self_n{n}']*(1 - factor)
        axes[2].plot(w, 1e6*predicted, '-', color=color, lw=1,
                     label=f'{n} groups, predicted' if detector == entries[0][0] else None)
axes[0].set_ylabel('Transit depth (ppm)')
axes[0].set_ylim(4500, 10500)
axes[1].set_ylim(-2500, 2500)
axes[1].set_ylabel('Depth difference (ppm)')
axes[2].set_ylabel('Self - CRDS depth (ppm)')
axes[1].axhline(0, color='k', lw=0.5)
axes[2].axhline(0, color='k', lw=0.5)
axes[2].set_xlabel('Wavelength (µm)')
axes[0].set_title('HAT-P-14 b, NIRSpec/G395H: number of groups fitted, CRDS and self linearity')
for ax in axes:
    ax.legend(fontsize=7, ncol=3)
fig.tight_layout()
fig.savefig(out, dpi=130)
print(out)
