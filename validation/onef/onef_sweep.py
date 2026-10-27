"""1/f pixel sweep of the group 1/f correction on the inputs of onef_extract.py (one segment).

Default factors (aperture rates / median rate, median filtered over 300 s) and median
templates (median_i of x_i/s_i), as the package; the 1/f pixels are the bands
inner <= d < inner + width (d: distance to the nearest trace centre, rows), for every pair.
Per configuration: the white light (apertures given) and 32 spectroscopic bins (sums of
per-pixel OLS slopes over the groups, before linearity), and a chromatic injection: every
pixel's source counts of the columns of bin b are scaled by a_b (s_i - 1), a_b = +-0.3
alternating (zero white-light change), and the injected ramps corrected separately (their
own templates); dL - dL_true measures how much of the injection the correction removes.
Also the first-difference variance of the white light and the sum of those of its pixels.

  python onef_sweep.py <segment folder> <effinttm> <inners> <widths> <apertures> <out.npz>
     (lists comma separated; width 'all' = no limit)
"""
import sys, json, time, glob, resource, warnings
from pathlib import Path
import numpy as np
from scipy.ndimage import median_filter

warnings.simplefilter('ignore', RuntimeWarning)
folder = Path(sys.argv[1])
effinttm = float(sys.argv[2])
inners = [float(a) for a in sys.argv[3].split(',')]
widths = [np.inf if a == 'all' else float(a) for a in sys.argv[4].split(',')]
apertures = [float(a) for a in sys.argv[5].split(',')]
out = sys.argv[6]
NBINS, AMPLITUDE, WINDOW_TIME = 32, 0.3, 300.
start = time.monotonic()

tags = sorted(Path(p).stem.split('_')[1] for p in glob.glob(str(folder/'geometry_*.npz')))
deltas = [np.load(folder/f'delta_{t}.npy', mmap_mode='r') for t in tags]
masks = [np.load(folder/f'masks_{t}.npy', mmap_mode='r') for t in tags]
references = [np.load(folder/f'reference_{t}.npy') for t in tags]
geometry = np.load(folder/f'geometry_{tags[0]}.npz')
settings = json.loads(str(geometry['settings']))
traces = {int(o): (geometry[f'columns{o}'], geometry[f'centers{o}']) for o in geometry['orders']}
median, background = geometry['median'], geometry['background']
nints = sum(len(d) for d in deltas)
ngroups, nrows, ncols = deltas[0].shape[1:]
chunk_of = np.concatenate([np.full(len(d), j) for j, d in enumerate(deltas)])
index_in = np.concatenate([np.arange(len(d)) for d in deltas])
print(f'{folder}: {nints} integrations, {ngroups} groups, {nrows} x {ncols}', flush=True)


def frame(i):
    j, ii = chunk_of[i], index_in[i]
    return references[j] + np.asarray(deltas[j][ii], np.float32), np.asarray(masks[j][ii])


# Distances to the nearest trace centre (traces extended to all columns for NIRSpec):
rows = np.arange(nrows)[:, None]
distance = np.full((nrows, ncols), np.inf)
for columns, centers in traces.values():
    columns, centers = np.asarray(columns, int), np.asarray(centers, float)
    if settings['extend_traces']:
        order = np.argsort(columns)
        centers = np.interp(np.arange(ncols), columns[order], centers[order])
        columns = np.arange(ncols)
    distance[:, columns] = np.minimum(distance[:, columns], np.abs(rows - centers))
edge = settings['edge_columns']
usable = np.ones((nrows, ncols), bool)
if edge > 0:
    usable[:, :edge] = usable[:, -edge:] = False
configs = [(a, w) for a in inners for w in widths]
config_pixels = np.array([usable & (distance >= a) & (distance < a + w) for a, w in configs])
columns1, centers1 = traces[min(traces)]
columns1 = np.asarray(columns1, int)


def aperture(radius):
    mask = np.zeros((nrows, ncols), bool)
    mask[:, columns1] = np.abs(rows - np.asarray(centers1, float)) <= radius
    return mask


apertures_masks = np.array([aperture(a) for a in apertures])
bin_of_column = np.full(ncols, -1)
for b, cols in enumerate(np.array_split(np.sort(columns1), NBINS)):
    bin_of_column[cols] = b
sign = np.where(np.arange(NBINS) % 2 == 0, AMPLITUDE, -AMPLITUDE)
column_amplitude = np.where(bin_of_column >= 0, sign[np.maximum(bin_of_column, 0)], 0.)

# Default white-light factors (as the package now):
rates = np.concatenate([np.load(folder/f'rates_{t}.npy').astype(np.float32) for t in tags])
rates_bad = np.concatenate([np.load(folder/f'rates_bad_{t}.npy') for t in tags])
use = aperture(settings['aperture_radius'])
factors = np.empty(nints)
for i in range(nints):
    f = rates[i] - background
    good = np.isfinite(f) & ~rates_bad[i] & use & np.isfinite(median)
    factors[i] = np.sum(f[good])/np.sum(median[good])
del rates, rates_bad
window = min(int(round(WINDOW_TIME/effinttm)) | 1, (nints - 1) | 1)
if window > 1:
    factors = median_filter(factors, window)
injection = factors - 1  # the transit shape of the injection
print(f'{len(configs)} configurations; window {window}; factors {factors.min():.4f}-'
      f'{factors.max():.4f}', flush=True)

# Median templates, plain and injected (source counts: s_i (T_k - T_0) of the plain ones):
s32 = factors.astype(np.float32)[:, None, None]
templates = np.zeros((ngroups, nrows, ncols), np.float32)
block = max(1, int(1.5e8/(nints*ncols*4)))


def load_group(k, r0, r1):
    x = np.concatenate([references[j][k, r0:r1] + np.asarray(d[:, k, r0:r1], np.float32)
                        for j, d in enumerate(deltas)])
    good = np.concatenate([(np.asarray(m[:, k, r0:r1]) & 2) > 0 for m in masks])
    return x, good


for k in range(ngroups):
    for r0 in range(0, nrows, block):
        x, good = load_group(k, r0, min(nrows, r0 + block))
        templates[k, r0:r0 + len(x[0])] = np.nanmedian(np.where(good, x/s32, np.nan), 0)
# Only the star's share of every pixel transits (not the sky, e.g. the SOSS zodiacal light):
star_fraction = np.clip(np.nan_to_num(median/(median + background)), 0, 1).astype(np.float32)
source = np.nan_to_num(templates - templates[0])*star_fraction[None]
gain = (injection[:, None] * column_amplitude[None, :]).astype(np.float32)  # (int, column)
templates_injected = np.zeros_like(templates)
for k in range(ngroups):
    for r0 in range(0, nrows, block):
        r1 = min(nrows, r0 + block)
        x, good = load_group(k, r0, r1)
        x = x + gain[:, None, :]*s32*source[k, r0:r1][None]
        templates_injected[k, r0:r1] = np.nanmedian(np.where(good, x/s32, np.nan), 0)
    del x, good
print(f'templates: {time.monotonic() - start:.0f} s', flush=True)

t = np.arange(ngroups, dtype=np.float32)[:, None, None]
fill_sample = []


def weights(m):
    """OLS slope weights over the groups, (group, row, column); zero without two groups."""
    m = m.astype(np.float32)
    s0, s1, s2 = m.sum(0), (m*t).sum(0), (m*t*t).sum(0)
    det = s0*s2 - s1*s1
    ok = (s0 >= 2) & (det > 0)
    return np.where(ok, m*(s0*t - s1)/np.where(ok, det, 1), 0.), ok


def masked_median(r, use):
    use = use & np.isfinite(r)
    v = np.sort(np.where(use, r, np.inf), axis=1)
    n = use.sum(1)
    lo = np.take_along_axis(v, np.maximum((n - 1)//2, 0)[:, None], 1)[:, 0]
    hi = np.take_along_axis(v, np.minimum(n//2, nrows - 1)[:, None], 1)[:, 0]
    return np.where(n > 0, (lo + hi)/2, 0.)


for i in np.linspace(0, nints - 1, 100).astype(int):
    x, m = frame(i)
    w, ok = weights((m & 1) > 0)
    fill_sample.append(np.where(ok, (w*x).sum(0), np.nan))
fill = np.nan_to_num(np.nanmedian(fill_sample, 0))
del fill_sample

ncfg, nap = len(configs), len(apertures)
aperture_any = apertures_masks.any(0)
bins_main = [(apertures_masks[0] & (bin_of_column[None, :] == b)) for b in range(NBINS)]
column_bins = np.zeros((NBINS, ncols), np.float32)
for b in range(NBINS):
    column_bins[b, bin_of_column == b] = 1
W = np.zeros((ncfg + 1, nints, nap), np.float32)  # last: raw
L = np.zeros((ncfg + 1, nints, NBINS), np.float32)
dL = np.zeros((ncfg + 1, nints, NBINS), np.float32)
dL_true = np.zeros((nints, NBINS), np.float32)
fd_pixels = np.zeros((ncfg + 1, nap))
fd_pixels_bins = np.zeros((ncfg + 1, NBINS))
previous = None
for i in range(nints):
    x, m = frame(i)
    valid, clean = (m & 1) > 0, (m & 2) > 0
    w, ok = weights(np.where(clean.sum(0) >= 2, clean, valid))
    slope = np.where(ok, (w*x).sum(0), fill)
    x_injected = x + gain[i][None, None, :]*factors[i]*source
    slope_injected = np.where(ok, (w*x_injected).sum(0), fill)
    truth = (slope_injected - slope)
    dL_true[i] = column_bins @ np.where(apertures_masks[0], truth, 0.).sum(0)
    # Per column: the slope weights of the aperture pixels, (aperture, group, column), and of
    # the main aperture's pixels per bin through column_bins.
    w_ap = np.einsum('akrc->akc', w[None]*apertures_masks[:, None]).astype(np.float32)
    residual = x - factors[i]*templates
    residual_injected = x_injected - factors[i]*templates_injected
    current = np.zeros((ncfg + 1, nrows, ncols), np.float32)
    sums = np.array([np.sum(slope[a]) for a in apertures_masks])
    column_slope = np.where(apertures_masks[0], slope, 0.).sum(0)
    column_slope_injected = np.where(apertures_masks[0], slope_injected, 0.).sum(0)
    for c in range(ncfg + 1):
        if c < ncfg:
            use = valid & config_pixels[c][None]
            unflagged = use & clean
            n_use = use.sum((1, 2))
            shared = (n_use - unflagged.sum((1, 2))) > 0.5*np.maximum(n_use, 1)
            use = np.where(shared[:, None, None], use, unflagged)
            level = masked_median(residual, use)
            level_injected = masked_median(residual_injected, use)
        else:
            level = level_injected = np.zeros((ngroups, ncols), np.float32)
        correction = (w_ap*level[None]).sum(1)              # (aperture, column)
        correction_injected = (w_ap*level_injected[None]).sum(1)
        W[c, i] = sums - correction.sum(1)
        column_main = column_slope - correction[0]
        column_injected = column_slope_injected - correction_injected[0]
        L[c, i] = column_bins @ column_main
        dL[c, i] = column_bins @ (column_injected - column_main)
        current[c] = np.where(aperture_any, slope - (w*level[:, None, :]).sum(0), 0.)
    if previous is not None:
        squared = (current - previous)**2
        for a in range(nap):
            fd_pixels[:, a] += squared[:, apertures_masks[a]].sum(1)
        for b in range(NBINS):
            fd_pixels_bins[:, b] += squared[:, bins_main[b]].sum(1)
    previous = current
    if i % 500 == 0:
        print(f'integration {i}: {time.monotonic() - start:.0f} s, peak '
              f'{resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**30:.1f} GB', flush=True)

np.savez(out, configs=np.array(configs), apertures=np.array(apertures), W=W, L=L, dL=dL,
         dL_true=dL_true, fd_pixels=fd_pixels, fd_pixels_bins=fd_pixels_bins,
         npix=config_pixels.sum((1, 2)), factors=factors, window=window,
         settings=json.dumps(settings))
print(f'done: {time.monotonic() - start:.0f} s, peak '
      f'{resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**30:.1f} GB', flush=True)
