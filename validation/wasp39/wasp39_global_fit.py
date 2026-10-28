"""One self non-linearity correction shared by all pixels, a function of the counts alone.

The per-pixel cubic (wasp39_combine_fit.py) needs >= 5 groups, so the PRISM pixels that
saturate early keep the CRDS correction. Here a2 and a3 are common to all pixels, with one
top T for all, and every pixel has its own zero point and slope:

    x_p(t_k) + (T/2) [a2 T2(u_pk) + a3 T3(u_pk)] = z_p + b_p t_k,    u_pk = 2 x_p(t_k)/T - 1,

fitted by generalized least squares (the covariance of fit_self_linearity; z_p, b_p profiled
out) to the mean ramps of the exposure fit (self_linearity.npz). Only groups unsaturated in
all integrations are used (>= 99.5% of the integrations of group 0; the mean of a group that
saturates in some integrations is biased low), in pixels above 500 DN with >= 3 such groups.
Writes the shared correction (map 'global') and a 'hybrid' one: the per-pixel coefficients
where they exist, the shared ones for the other pixels above 500 DN (the saturating core).
  python wasp39_global_fit.py <self_linearity.npz> <global.npz> <hybrid.npz>
      [read noise DN] [gain e/DN]
"""
import sys
from pathlib import Path
import numpy as np
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO/'tests'))
import _support  # noqa: F401
from transitspectroscopy.core.linearity import SelfLinearity

source, out_global, out_hybrid = sys.argv[1:4]
read_noise = float(sys.argv[4]) if len(sys.argv) > 4 else 7.0
gain = float(sys.argv[5]) if len(sys.argv) > 5 else 1.4
pixel_fit = SelfLinearity.load(source)
mean, counts = pixel_fit.mean_ramps[0], pixel_fit.counts[0]
ngroups = len(mean)
full = (counts >= 0.995*counts[0]) & np.isfinite(mean)
full = np.cumprod(full, axis=0).astype(bool)  # contiguous from group 0
highest = np.where(full, mean, -np.inf).max(0)
fit = (highest > 500) & (full.sum(0) >= 3)
rows, cols = np.nonzero(fit)
top = 1.02*highest[fit].max()
x = np.where(full[:, rows, cols], mean[:, rows, cols], 0.).T          # (pixel, group)
use = full[:, rows, cols].T
n = counts[:, rows, cols].T.astype(float)
t = np.arange(ngroups, dtype=float)


def chebyshev_design(x, top):
    u = 2*x/top - 1
    return np.stack([-top/2*(2*u**2 - 1), -top/2*(4*u**3 - 3*u)], -1)


charge = np.maximum(x, 0.)
covariance = (read_noise**2*np.eye(ngroups)[None]
              + np.minimum(charge[:, :, None], charge[:, None, :])/gain)
covariance /= np.maximum(np.maximum(n[:, :, None], n[:, None, :]), 1.)
covariance = np.where(use[:, :, None] & use[:, None, :], covariance, 0.)
np.einsum('pgg->pg', covariance)[...] = np.where(use, np.einsum('pgg->pg', covariance), 1e30)
weight = np.linalg.inv(covariance)
line = np.stack([np.ones_like(x), np.broadcast_to(t, x.shape)], -1)*use[..., None]
# Weight with the pixel's zero point and slope profiled out:
wa = weight @ line
projected = weight - wa @ np.linalg.inv(np.swapaxes(line, 1, 2) @ wa) @ np.swapaxes(wa, 1, 2)
design = chebyshev_design(x, top)*use[..., None]
normal = np.einsum('pgi,pgh,phj->ij', design, projected, design)
vector = np.einsum('pgi,pgh,ph->i', design, projected, x)
a2, a3 = np.linalg.solve(normal, vector)
residual = x - design @ np.array([a2, a3])
chi2 = np.einsum('pg,pgh,ph->p', residual, projected, residual)
dof = use.sum(1) - 2
# Per-pixel fits of the same pixels (5 groups): their chi2, for comparison.
own = pixel_fit.pixels[rows, cols] & (use.sum(1) == ngroups)
own_a = np.stack([pixel_fit.a2[rows, cols], pixel_fit.a3[rows, cols]], -1)
own_top = pixel_fit.top[rows, cols]
own_design = chebyshev_design(x, own_top[:, None])*use[..., None]
own_residual = x - np.einsum('pgi,pi->pg', own_design, np.nan_to_num(own_a))
own_chi2 = np.einsum('pg,pgh,ph->p', own_residual, projected, own_residual)
scale = np.sum(chi2)/np.sum(dof)
errors = np.sqrt(np.diag(np.linalg.inv(normal))*scale)
print(f'{len(rows)} pixels ({(use.sum(1) < ngroups).sum()} with 3-4 full groups), top '
      f'{top:.0f} DN: a2 {a2:.6f} +- {errors[0]:.6f}, a3 {a3:.6f} +- {errors[1]:.6f}; '
      f'chi2/dof {scale:.2f}')
for lo, hi in ((500, 5000), (5000, 20000), (20000, 40000), (40000, 60000)):
    band = own & (highest[rows, cols] >= lo) & (highest[rows, cols] < hi)
    if band.any():
        print(f'  5-group pixels {lo}-{hi} DN ({band.sum()}): chi2/dof shared '
              f'{np.sum(chi2[band])/np.sum(dof[band]):.2f}, per pixel (dof 1) '
              f'{np.sum(own_chi2[band])/band.sum():.2f}')
shared = SelfLinearity(np.full(fit.shape, a2), np.full(fit.shape, a3), np.full(fit.shape, top),
                       dict(pixel_fit.metadata, shared=True, pixels=len(rows), chi2_dof=scale,
                            a2_error=errors[0], a3_error=errors[1]))
# Shape of the shared correction against the per-pixel ones (5-group pixels above 20 kDN):
# h(0.8 R)/h(0.2 R), R the highest mean count; its normalisation h'(0) is an extrapolation.
bright = own & (highest[rows, cols] > 20000)
r_b, c_b, top_b = rows[bright], cols[bright], highest[rows, cols][bright]
shape = {name: correction.correct(0.8*top_b, rows=r_b, columns=c_b)
         / correction.correct(0.2*top_b, rows=r_b, columns=c_b)
         for name, correction in (('shared', shared), ('pixel', pixel_fit))}
ratio = shape['shared']/shape['pixel'] - 1
print(f'  h(0.8R)/h(0.2R), shared over per-pixel - 1 ({bright.sum()} pixels above 20 kDN): '
      f'median {np.median(ratio):+.4f}, rms {np.std(ratio):.4f} (per-pixel h(0.8R)/h(0.2R)/4 '
      f'- 1: median {np.median(shape["pixel"]/4 - 1):+.4f})')
shared.save(out_global)
hybrid = ~pixel_fit.pixels & (np.where(np.isfinite(mean), mean, -np.inf).max(0) > 500)
maps = {k: np.where(hybrid, getattr(shared, k), getattr(pixel_fit, k)) for k in ('a2', 'a3', 'top')}
SelfLinearity(maps['a2'], maps['a3'], maps['top'],
              dict(pixel_fit.metadata, hybrid_shared_pixels=int(hybrid.sum()))).save(out_hybrid)
print(f'hybrid: {pixel_fit.pixels.sum()} per-pixel, {hybrid.sum()} shared')
