"""Self vs CRDS linearity correction of the PRISM pixels in and around the saturated region.

From the exposure's mean raw ramps (self_linearity.npz, groups unsaturated in >= 99.5% of the
integrations), the CRDS coefficients (jwst_nirspec_linearity_0024, NRS1, SUB512 corner
x 1025, y 959) and the per-pixel self fit; and the rates of the pipeline (median over the
first chunk, seg001, of crds_00000.fits and self_00000.fits). Pixels saturating in > 0.1% of
the integrations (the masked core) keep CRDS in the self variant. Panels: (a) highest mean
count; (b) self/CRDS rate - 1; (c) local gain dh/dx of example pixels, over the CRDS linear
coefficient (the h'(0) of both); (d) group increments of their mean ramps, raw and corrected,
over the first one; (e) linearity of the corrected mean ramps, (h4 - h3)/(h1 - h0) (1 if
linear), against the counts.
  python wasp39_pixels_figure.py <prism folder> <crds linearity.fits> <out.png>
"""
import sys
from pathlib import Path
import numpy as np
from astropy.io import fits
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO/'tests'))
import _support  # noqa: F401
from transitspectroscopy.core.linearity import SelfLinearity

folder, reference, png = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
self_fit = SelfLinearity.load(folder/'self_linearity.npz')
corner = self_fit.metadata
with fits.open(reference) as f:
    y0, x0 = corner['ystart'] - f[0].header['SUBSTRT2'], corner['xstart'] - f[0].header['SUBSTRT1']
    coefficients = f['COEFFS'].data[:, y0:y0 + 32, x0:x0 + 512].astype(float)
mean, counts = self_fit.mean_ramps[0], self_fit.counts[0]
full = np.cumprod((counts >= 0.995*counts[0]) & np.isfinite(mean), axis=0).astype(bool)
highest = np.where(full, mean, -np.inf).max(0)
core = np.load(folder/'saturation_mask_core.npy')
grow = np.load(folder/'saturation_mask_grow1.npy')
neighbour = grow & ~core


def crds(x):
    return sum(c*x**k for k, c in enumerate(coefficients))


rates = {}
for v in ('crds', 'self'):
    with fits.open(folder/'seg001'/f'{v}_00000.fits') as f:
        rates[v] = np.nanmedian(f['SCI'].data, axis=0)
ratio = rates['self']/rates['crds'] - 1

c0, c1 = 55, 150
rows = np.arange(32)
fig = plt.figure(figsize=(15, 12))
grid = fig.add_gridspec(3, 3, height_ratios=(1, 1, 1.2))
ax = fig.add_subplot(grid[0, :])
image = ax.imshow(np.log10(np.clip(highest[:, c0:c1], 10, None)), origin='lower', aspect='auto',
                  extent=(c0 - 0.5, c1 - 0.5, -0.5, 31.5), cmap='viridis')
ax.contour(np.arange(c0, c1), rows, core[:, c0:c1].astype(float), [0.5], colors='r',
           linewidths=0.8)
ax.contour(np.arange(c0, c1), rows, grow[:, c0:c1].astype(float), [0.5], colors='w',
           linewidths=0.6, linestyles='--')
fig.colorbar(image, ax=ax, label='log10 highest unsaturated mean count (DN)', pad=0.01)
ax.set_title('(a) mean count; red: saturating core (masked, keeps CRDS); white dashed: core '
             '$\\pm$1 row')
ax.set_ylabel('row')
ax.set_ylim(8, 24)
ax = fig.add_subplot(grid[1, :])
image = ax.imshow(100*ratio[:, c0:c1], origin='lower', aspect='auto', cmap='RdBu_r',
                  vmin=-4, vmax=4, extent=(c0 - 0.5, c1 - 0.5, -0.5, 31.5))
ax.contour(np.arange(c0, c1), rows, core[:, c0:c1].astype(float), [0.5], colors='k',
           linewidths=0.8)
fig.colorbar(image, ax=ax, label='self / CRDS rate - 1 (%)', pad=0.01)
ax.set_title('(b) pipeline rates (median of 1800 integrations): self over CRDS')
ax.set_xlabel('column')
ax.set_ylabel('row')
ax.set_ylim(8, 24)

# Examples (5-group self-fitted pixels in the saturated columns): the brightest vertical
# neighbour of the core, the brightest pixel more than 2 pixels from it, and a halo pixel
# (above or below the core, > 2 rows from it, the most negative self/CRDS rate - 1).
fitted = self_fit.pixels & (full.sum(0) == 5)
near = core.copy()
for dr in range(-2, 3):
    for dc in range(-2, 3):
        near |= np.roll(np.roll(core, dr, 0), dc, 1)
cols = np.arange(512)[None, :]
region = (cols >= c0) & (cols < c1)
examples = []
for label, score, sel in (
        ('core neighbour', highest, fitted & neighbour & region),
        ('> 2 px from core', highest, fitted & ~near & region),
        ('halo', -ratio, fitted & ~near & (core.any(0)[None, :] & region)
         & (highest > 1000))):
    r, c = np.unravel_index(np.argmax(np.where(sel, score, -np.inf)), highest.shape)
    examples.append((label, r, c))
colors = ('tab:red', 'tab:blue', 'tab:green')
ax = fig.add_subplot(grid[2, 0])
for (label, r, c), color in zip(examples, colors):
    x = np.linspace(100, highest[r, c], 300)
    index = dict(rows=np.full(len(x), r), columns=np.full(len(x), c))
    gain = coefficients[1, r, c]
    h_self = self_fit.correct(x, **index)
    h_crds = sum(coefficients[k, r, c]*x**k for k in range(len(coefficients)))
    ax.plot(x/1e3, np.gradient(h_self, x)/gain, color=color,
            label=f'{label} ({r}, {c}), {highest[r, c]/1e3:.1f} kDN')
    ax.plot(x/1e3, np.gradient(h_crds, x)/gain, color=color, ls='--')
ax.set_xlabel('raw counts (kDN)')
ax.set_ylabel("dh/dx / CRDS linear coefficient")
ax.set_title('(c) local gain (solid: self, dashed: CRDS)', fontsize=10)
ax.legend(fontsize=7)
ax = fig.add_subplot(grid[2, 1])
for (label, r, c), color in zip(examples, colors):
    ramp = {'raw': mean[:, r, c],
            'CRDS': sum(coefficients[k, r, c]*mean[:, r, c]**k for k in range(5)),
            'self': self_fit.correct(mean[:, r, c], rows=np.full(5, r), columns=np.full(5, c))}
    for name, style in (('raw', ':'), ('CRDS', '--'), ('self', '-')):
        step = np.diff(ramp[name])
        ax.plot(np.arange(1, 5), step/step[0], ls=style, marker='o', ms=3, color=color,
                label=name if color == 'tab:red' else None)
ax.axhline(1, color='k', lw=0.6)
ax.set_xticks(range(1, 5))
ax.set_xlabel('group increment k (group k - group k-1)')
ax.set_ylabel('increment / first increment')
ax.set_title('(d) mean-ramp increments, same pixels\n(dotted raw, dashed CRDS, solid self)',
             fontsize=10)
ax.legend(fontsize=7)
ax = fig.add_subplot(grid[2, 2])
h = {'CRDS': crds(mean), 'self': np.where(self_fit.pixels, self_fit.correct(mean), np.nan)}
for name, marker in (('CRDS', 'x'), ('self', 'o')):
    q = (h[name][4] - h[name][3])/(h[name][1] - h[name][0])
    for sel, color, label in ((fitted & neighbour, 'tab:red', 'core neighbours'),
                              (fitted & ~grow & (highest > 1000), 'tab:blue', 'other pixels')):
        ax.scatter(highest[sel]/1e3, q[sel], s=8 if name == 'self' else 12, marker=marker,
                   color=color, alpha=0.6, lw=0.8, facecolors='none' if marker == 'o' else None,
                   label=f'{name}, {label} ({sel.sum()}): median {np.nanmedian(q[sel]):.3f}')
ax.text(0.02, 0.02, 'core neighbours: $\\pm$1 row of the core', transform=ax.transAxes,
        fontsize=7)
ax.axhline(1, color='k', lw=0.6)
ax.set_ylim(0.8, 1.7)
ax.set_xscale('log')
ax.set_xlabel('highest mean count (kDN)')
ax.set_ylabel('(h4 - h3) / (h1 - h0) of the corrected mean ramp')
ax.set_title('(e) linearity of the corrected mean ramps\n(5-group pixels)', fontsize=10)
ax.legend(fontsize=7)
fig.tight_layout()
fig.savefig(png, dpi=130)
for label, r, c in examples:
    print(label, r, c, f'{highest[r, c]:.0f} DN, rate self/CRDS - 1 {100*ratio[r, c]:+.2f}%')
print(png)
