"""Homogeneous group 1/f correction: templates, 1/f pixels and white-light factors compared on
the inputs saved by onef_extract.py (chunks of one segment, joined here).

Variants (template / 1/f pixels / factors):
  raw          no correction
  med/pkg/pkg  the package: median of x_i/s_i, the mode's pixels and factors
  two/pkg/pkg  two-term template x = a + s_i b (robust per-pixel fit), package pixels, factors
  med/new/pkg  median template, pixels from the trace centre, package factors
  med/new/new  ... and factors from the aperture white light of the first-pass rates,
               median filtered over a window in time
  two/new/new  two-term template, new pixels, new factors
For each, the white light (aperture sums of per-pixel OLS slopes over the groups, before
linearity) of every integration is saved, for the apertures of the mode.

  python onef_homog.py <segment folder> <effinttm> <new inner> <new outer|none> <window s>
                       <apertures, comma separated> <out.npz>
"""
import sys, json, time, glob, resource, warnings
from pathlib import Path
import numpy as np
from scipy.ndimage import median_filter

warnings.simplefilter('ignore', RuntimeWarning)
folder = Path(sys.argv[1])
effinttm = float(sys.argv[2])
new_inner = float(sys.argv[3])
new_outer = None if sys.argv[4] == 'none' else float(sys.argv[4])
window_s = float(sys.argv[5])
apertures = [float(a) for a in sys.argv[6].split(',')]
out = sys.argv[7]
start = time.monotonic()

tags = sorted(Path(p).stem.split('_')[1] for p in glob.glob(str(folder/'geometry_*.npz')))
deltas = [np.load(folder/f'delta_{t}.npy', mmap_mode='r') for t in tags]
masks = [np.load(folder/f'masks_{t}.npy', mmap_mode='r') for t in tags]
references = [np.load(folder/f'reference_{t}.npy') for t in tags]
rates = np.concatenate([np.load(folder/f'rates_{t}.npy').astype(np.float32) for t in tags])
rates_bad = np.concatenate([np.load(folder/f'rates_bad_{t}.npy') for t in tags])
geometry = np.load(folder/f'geometry_{tags[0]}.npz')
settings = json.loads(str(geometry['settings']))
traces = {int(o): (geometry[f'columns{o}'], geometry[f'centers{o}']) for o in geometry['orders']}
median, background = geometry['median'], geometry['background']
nints = sum(len(d) for d in deltas)
ngroups, nrows, ncols = deltas[0].shape[1:]
chunk_of = np.concatenate([np.full(len(d), j) for j, d in enumerate(deltas)])
index_in = np.concatenate([np.arange(len(d)) for d in deltas])
print(f'{folder}: {nints} integrations, {ngroups} groups, {nrows} x {ncols}', flush=True)


def pixels(s):
    """Background pixels and aperture, as group_difference_1f_from_traces."""
    rows = np.arange(nrows)[:, None]
    distance = np.full((nrows, ncols), np.inf)
    for columns, centers in traces.values():
        columns, centers = np.asarray(columns, int), np.asarray(centers, float)
        if s['extend_traces']:
            order = np.argsort(columns)
            centers = np.interp(np.arange(ncols), columns[order], centers[order])
            columns = np.arange(ncols)
        if s['trace_reference'] == 'floor':
            centers = np.floor(centers) - 0.5
        distance[:, columns] = np.minimum(distance[:, columns], np.abs(rows - centers))
    outer = np.inf if s['outer_radius'] is None else s['outer_radius']
    background = (distance >= s['inner_radius']) & (distance <= outer)
    if s['edge_columns'] > 0:
        background[:, :s['edge_columns']] = False
        background[:, -s['edge_columns']:] = False
    return background


def aperture(radius):
    columns, centers = traces[min(traces)]
    rows = np.arange(nrows)[:, None]
    mask = np.zeros((nrows, ncols), bool)
    mask[:, columns] = np.abs(rows - np.asarray(centers, float)) <= radius
    return mask


def package_factors(s):
    if isinstance(s['scale_pixels'], str):
        use = aperture(s['aperture_radius'])
    else:
        (r0, r1), (c0, c1) = s['scale_pixels']['rows'], s['scale_pixels']['columns']
        use = np.zeros((nrows, ncols), bool)
        use[r0:r1, c0:c1] = True
    if s['scale_normalization'] == 'timeseries':
        r = np.where(np.isnan(rates), 0., rates) - background
        f = np.sum(np.where(use, r, 0.), axis=(1, 2), dtype=float)
        f = f/np.nanmedian(f)
    else:
        f = median_rate_factors(use)
    if s['scale_window'] > 1:
        f = median_filter(f, s['scale_window'])
    return f


def median_rate_factors(use):
    f = np.empty(nints)
    for i in range(nints):
        frame = rates[i] - background
        good = np.isfinite(frame) & ~rates_bad[i] & use & np.isfinite(median)
        f[i] = np.sum(frame[good])/np.sum(median[good])
    return f


package_settings = dict(settings)
new_settings = dict(settings, trace_reference='center', inner_radius=new_inner,
                    outer_radius=new_outer)
background_pixels = {'pkg': pixels(package_settings), 'new': pixels(new_settings)}
window = max(1, int(round(window_s/effinttm)) | 1)
factors = {'pkg': package_factors(package_settings),
           'new': median_rate_factors(aperture(settings['aperture_radius']))}
factors['new_raw'] = factors['new'].copy()
if window > 1:
    factors['new'] = median_filter(factors['new'], window)
print('1/f pixels per column: package %.1f, new %.1f; window %d; factor scatter (ppm about '
      'a 51 median): package %.0f, new %.0f' % (
          background_pixels['pkg'].sum()/ncols, background_pixels['new'].sum()/ncols, window,
          *[1e6*np.std(f - median_filter(f, 51)) for f in (factors['pkg'], factors['new'])]),
      flush=True)

# Templates, per group, in blocks of rows (median of x/s for each factor set; two-term fit):
templates = {}
for name in ('pkg', 'new'):
    templates[('med', name)] = np.zeros((ngroups, nrows, ncols), np.float32)
    templates[('two', name)] = np.zeros((2, ngroups, nrows, ncols), np.float32)
block = max(1, int(2e8/(nints*ncols*4)))
for k in range(ngroups):
    for r0 in range(0, nrows, block):
        r1 = min(nrows, r0 + block)
        x = np.concatenate([references[j][k, r0:r1] + np.asarray(d[:, k, r0:r1], np.float32)
                            for j, d in enumerate(deltas)])
        good = np.concatenate([(np.asarray(m[:, k, r0:r1]) & 2) > 0 for m in masks])
        for name in ('pkg', 'new'):
            s = factors[name].astype(np.float32)[:, None, None]
            templates[('med', name)][k, r0:r1] = np.nanmedian(np.where(good, x/s, np.nan), 0)
            # Robust least squares of x on (1, s) over the integrations, 3 iterations of 5-sigma
            # clipping (sigma from the MAD of the residuals):
            w = good.copy()
            for _ in range(3):
                n, sx = w.sum(0), np.sum(w*s, 0)
                sxx, sy = np.sum(w*s*s, 0), np.sum(np.where(w, x, 0.), 0)
                sxy = np.sum(np.where(w, x*s, 0.), 0)
                det = n*sxx - sx*sx
                b = np.where(det > 0, (n*sxy - sx*sy)/np.where(det > 0, det, 1), 0.)
                a = np.where(n > 0, (sy - b*sx)/np.maximum(n, 1), np.nan)
                residual = x - (a + b*s)
                sigma = 1.4826*np.nanmedian(np.where(good, np.abs(residual), np.nan), 0)
                w = good & (np.abs(residual) < 5*sigma)
            templates[('two', name)][0, k, r0:r1] = a
            templates[('two', name)][1, k, r0:r1] = b
        del x, good
    print(f'group {k} templates: {time.monotonic() - start:.0f} s', flush=True)

variants = {'med/pkg/pkg': ('med', 'pkg', 'pkg'), 'two/pkg/pkg': ('two', 'pkg', 'pkg'),
            'med/new/pkg': ('med', 'new', 'pkg'), 'med/new/new': ('med', 'new', 'new'),
            'two/new/new': ('two', 'new', 'new')}
apertures_masks = [aperture(a) for a in apertures]
t = np.arange(ngroups, dtype=np.float32)[:, None, None]


def slopes(x, m):
    m = m.astype(np.float32)
    s0, s1, s2 = m.sum(0), (m*t).sum(0), (m*t*t).sum(0)
    sx, stx = (m*x).sum(0), (m*t*x).sum(0)
    det = s0*s2 - s1*s1
    return np.where((s0 >= 2) & (det > 0), (s0*stx - s1*sx)/np.where(det > 0, det, 1), np.nan)


def white(slope, fill):
    slope = np.where(np.isfinite(slope), slope, fill)
    return [np.nansum(slope[a]) for a in apertures_masks]


# Per-pixel fill for pixels without a slope: median raw slope over 100 integrations.
sample = []
for i in np.linspace(0, nints - 1, 100).astype(int):
    j, ii = chunk_of[i], index_in[i]
    x = references[j] + np.asarray(deltas[j][ii], np.float32)
    sample.append(slopes(x, (np.asarray(masks[j][ii]) & 1) > 0))
fill = np.nanmedian(sample, 0)
del sample

names = ['raw'] + list(variants)
result = {name: np.zeros((nints, len(apertures))) for name in names}
levels_rms = {name: np.zeros(nints) for name in variants}
for i in range(nints):
    j, ii = chunk_of[i], index_in[i]
    x = references[j] + np.asarray(deltas[j][ii], np.float32)
    m = np.asarray(masks[j][ii])
    valid, clean = (m & 1) > 0, (m & 2) > 0
    fit_mask = np.where(clean.sum(0) >= 2, clean, valid)
    result['raw'][i] = white(slopes(x, fit_mask), fill)
    for name, (template, pix, fac) in variants.items():
        s = factors[fac][i]
        T = templates[(template, fac)]
        model = s*T if template == 'med' else T[0] + s*T[1]
        residual = x - model
        use = valid & background_pixels[pix][None]
        unflagged = use & clean
        shared = (use.sum((1, 2)) - unflagged.sum((1, 2))) > 0.5*np.maximum(use.sum((1, 2)), 1)
        use = np.where(shared[:, None, None], use, unflagged)
        level = np.nanmedian(np.where(use, residual, np.nan), axis=1)
        level = np.where(np.isfinite(level), level, 0.)
        levels_rms[name][i] = np.std(level)
        result[name][i] = white(slopes(x - level[:, None, :], fit_mask), fill)
    if i % 500 == 0:
        print(f'integration {i}: {time.monotonic() - start:.0f} s, peak '
              f'{resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**30:.1f} GB', flush=True)

np.savez(out, names=np.array(names), apertures=np.array(apertures),
         white=np.array([result[n] for n in names]),
         levels_rms=np.array([levels_rms[n] for n in variants]),
         factors_pkg=factors['pkg'], factors_new=factors['new'],
         factors_new_raw=factors['new_raw'], window=window,
         npix_pkg=background_pixels['pkg'].sum(), npix_new=background_pixels['new'].sum(),
         settings=json.dumps(settings))
print(f'done: {time.monotonic() - start:.0f} s, peak '
      f'{resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**30:.1f} GB', flush=True)
