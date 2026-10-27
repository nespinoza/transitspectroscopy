"""Summary of onef_homog.py outputs of all segments of one exposure.

rms: white light about a 51-integration running median, 5-sigma clipped, per segment and
all; difference to the package (med/pkg/pkg) with bootstrap errors (same integrations).
depth: the corrected minus the raw white light, normalised, regressed on (s - 1), s the
package white light normalised and median filtered over <depth window> integrations: the
fractional change of the transit depth by the correction (block bootstrap errors).
  python onef_homog_summary.py <depth window> <seg1.npz> <seg2.npz> ...
"""
import sys
import numpy as np
from scipy.ndimage import median_filter

window = int(sys.argv[1])
files = [np.load(f) for f in sys.argv[2:]]
names = list(files[0]['names'])
apertures = files[0]['apertures']
white = np.concatenate([f['white'] for f in files], axis=1)  # (variant, int, aperture)
segment = np.concatenate([np.full(f['white'].shape[1], j) for j, f in enumerate(files)])
rng = np.random.default_rng(1)
ref = names.index('med/pkg/pkg')
print('1/f pixels per column (package, new): '
      + ', '.join(f"{int(f['npix_pkg'])}/{int(f['npix_new'])}" for f in files)
      + f"; factor window {int(files[0]['window'])}")
for f in files:
    p, n, nr = f['factors_pkg'], f['factors_new'], f['factors_new_raw']
    sc = lambda x: 1e6*np.std(x - median_filter(x, 51))
    print(f'  factor scatter ppm: package {sc(p):.0f}, new unfiltered {sc(nr):.0f}, '
          f'new {sc(n):.0f}')

for a, radius in enumerate(apertures):
    print(f'\naperture {radius}:')
    norm = white[:, :, a]/np.nanmedian(white[:, :, a], axis=1, keepdims=True)
    detrended = norm - np.array([median_filter(x, 51, mode='nearest') for x in norm])
    keep = np.ones(norm.shape[1], bool)
    for x in detrended:
        mad = 1.4826*np.median(np.abs(x - np.median(x)))
        keep &= np.abs(x - np.median(x)) < 5*mad
    s = median_filter(norm[ref], window, mode='nearest')
    header = f"{'variant':12s} " + ' '.join(f'rms s{j+1}' for j in range(len(files))) + \
        '  rms all   d(rms) vs pkg   depth change %'
    print(header)
    for v, name in enumerate(names):
        rms = [1e6*np.std(detrended[v][keep & (segment == j)]) for j in range(len(files))]
        total = 1e6*np.std(detrended[v][keep])
        if v != ref:
            idx = np.flatnonzero(keep)
            boot = [1e6*(np.std(detrended[v][b]) - np.std(detrended[ref][b]))
                    for b in (rng.choice(idx, len(idx)) for _ in range(200))]
            diff = f'{total - 1e6*np.std(detrended[ref][keep]):+6.1f} +- {np.std(boot):4.1f}'
        else:
            diff = ' ' * 14
        if name != 'raw':
            d = norm[v] - norm[names.index('raw')]
            x = s - 1
            use = keep & np.isfinite(d)
            c = np.polyfit(x[use], d[use], 1)[0]
            # Block bootstrap (blocks of 50 integrations):
            blocks = np.array_split(np.flatnonzero(use), max(2, use.sum()//50))
            boots = []
            for _ in range(200):
                b = np.concatenate([blocks[i] for i in rng.integers(0, len(blocks), len(blocks))])
                boots.append(np.polyfit(x[b], d[b], 1)[0])
            depth = f'{100*c:+6.2f} +- {100*np.std(boots):4.2f}'
        else:
            depth = ''
        print(f'{name:12s} ' + ' '.join(f'{r:6.1f}' for r in rms)
              + f'  {total:6.1f}   {diff}   {depth}')
    # Paired depth differences (same bootstrap blocks), against two/new/new:
    x = s - 1
    use = keep & np.all(np.isfinite(norm), 0)
    blocks = np.array_split(np.flatnonzero(use), max(2, use.sum()//50))
    picks = [np.concatenate([blocks[i] for i in rng.integers(0, len(blocks), len(blocks))])
             for _ in range(300)]
    base = names.index('two/new/new')
    coefficient = lambda v, b: np.polyfit(x[b], norm[v][b] - norm[names.index('raw')][b], 1)[0]
    for v, name in enumerate(names):
        if name in ('raw', 'two/new/new'):
            continue
        d0 = 100*(coefficient(v, use) - coefficient(base, use))
        db = [100*(coefficient(v, b) - coefficient(base, b)) for b in picks]
        print(f'  depth {name} - two/new/new: {d0:+.2f} +- {np.std(db):.2f} %')
    print(f'transit depth in s: {1e6*(np.nanmax(s) - np.nanmin(s)):.0f} ppm (max - min)')
