# Self non-linearity correction

The CRDS linearity correction of a detector is a polynomial per pixel calibrated on the
ground. In time series, the mean ramp of every pixel is measured from hundreds of
integrations, so its curvature can be fitted precisely on the science data themselves. On
the NIRSpec/G395H HAT-P-14 b data (`validation/nl_groups/`), the CRDS correction leaves the
bright pixels super-linear by about 0.5%, which makes the transit 0.2-0.9% too deep, by an
amount that depends on the wavelength and on the number of groups fitted. Spectra built
with different numbers of groups at different wavelengths (e.g., to avoid saturation) then
change shape. The self correction makes the mean ramps linear and removes that dependence.

## Using it

```python
import transitspectroscopy as ts

dataset = ts.jwst.load(uncal_files)
dataset.detector_calibration(self_linearity=True)
dataset.fit_ramps()
```

`detector_calibration(self_linearity=True)` runs the steps up to the linearity step on all
segments, fits the correction to the mean ramps of all of them, and then applies it in place
of the CRDS correction:

- The mean ramp of every pixel is taken over the integrations without flags (from the
  first flagged group on, e.g. saturated, the groups of an integration are not used) and
  without outliers: an integration whose groups deviate from the median over the
  integrations by more than `nsigma` robust standard deviations (e.g., a cosmic ray) is
  excluded.
- Pixels whose mean ramp ends above `min_counts` (500 DN) are fitted. The CRDS linearity step
  still runs: its data-quality flags are kept, and so is its correction of the pixels that
  are not fitted (faint pixels, where the non-linearity is small).
- The group-level 1/f correction of `fit_ramps`, applied before linearity, uses the same
  correction when it calibrates the ramps again from the linearity step on.
- The products carry `selflin` in their suffix (`<name>_selflin_...`), so that they are never
  mixed with CRDS-corrected ones, and the correction is saved (with the mean ramps it was
  fitted to) as `ts_outputs/<first segment>_selflin_self_linearity.npz`. Later runs with the
  same suffix reuse it.

The fit options go in `parameters['self_linearity']`:

| Option | Default | Meaning |
|---|---|---|
| `min_counts` | 500 | Minimum mean count (DN) of the last group of a fitted pixel |
| `min_ramps` | 20 | Minimum number of integrations of a group to use it |
| `min_groups` | 5 | Minimum number of groups of a fitted pixel |
| `top_factor` | 1.02 | End of the fitted range over the highest mean count |
| `nsigma` | 5 | Outlier threshold of an integration |
| `read_noise`, `gain` | 7 DN, 1.4 e/DN | Noise model of the fit (they only weight it) |
| `stop_after_fit` | False | Return from `detector_calibration` once the fit is saved |

## The correction

For every pixel, with raw counts `x` (DN, after the bias, reference-pixel and other steps
before linearity):

```
g(x) = x + (top/2) [a2 T2(u) + a3 T3(u)],    u = 2x/top - 1,
h(x) = c1 (g(x) - g(0)) / g'(0)
```

where `T2(u) = 2u² - 1` and `T3(u) = 4u³ - 3u` are Chebyshev polynomials, `top` is the end of
the fitted range (1.02 times the pixel's highest mean count), and `c1` is the linear
coefficient of the CRDS correction, so that `h(0) = 0` and `h'(0) = c1` (corrected pixels keep
the relative gains of the CRDS correction). Outside `[0, top]`, `h` continues as straight lines
with its value and slope at the ends. `a2` and `a3` are fitted so that `g` of the mean ramp is
a straight line in the group number, `g(x(t_k)) = z + b t_k`; the fit is linear and weighted
by the covariance of the mean ramp (read noise in every group, Poisson noise accumulated along
the ramp). A pixel is kept if `g` increases on `[0, top]`. Chebyshev polynomials give the same
fit as a power series, with nearly uncorrelated coefficients that can be compared from pixel
to pixel.

The fit's statistical scatter is per pixel and fixed in time (it acts like a small gain
error): with a few hundred integrations it is about 1% for pixels near `min_counts` and a few
0.1% for bright ones, and it averages out over an aperture.

## Your own coefficients

`self_linearity_coefficients` applies given coefficients instead of fitting them (it implies
`self_linearity=True`):

```python
dataset.detector_calibration(self_linearity_coefficients='my_correction.npz')
```

It takes a `transitspectroscopy.core.linearity.SelfLinearity` or the path to a file written by
`SelfLinearity.save`, or to any `.npz` file with three `(row, column)` maps of the subarray:
`a2`, `a3` and `top` (NaN where the CRDS correction is to be kept).

## Combining several visits

Visits of the same star with the same subarray can be fitted together: they share `a2` and
`a3`, and each mean ramp keeps its own zero point and slope (they may have different numbers
of groups, and the groups a visit lacks, e.g. saturated ones, are simply not used). First save
the mean ramps of every visit, stopping each calibration right after its fit:

```python
from transitspectroscopy.core.linearity import SelfLinearity, fit_self_linearity

files = []
for visit, uncal_files in visits.items():
    dataset = ts.jwst.load(uncal_files, outputfolder=visit)
    dataset.detector_calibration(self_linearity=True,
                                 parameters={'self_linearity': {'stop_after_fit': True}})
    files.append(dataset.self_linearity_file)

fits = [SelfLinearity.load(name) for name in files]
combined = fit_self_linearity([mean for fit in fits for mean in fit.mean_ramps],
                              [count for fit in fits for count in fit.counts])
combined.save('combined_self_linearity.npz')
```

and then calibrate every visit with the combined correction:

```python
dataset.detector_calibration(self_linearity_coefficients='combined_self_linearity.npz')
```

The same functions work on any `(group, row, column)` mean ramps: `mean_ramp_sums` gives the
sums and numbers of integrations of a set of raw ramps, which can be added over exposures
before dividing.

## Validation

`validation/nl_groups/` holds the study on the HAT-P-14 b G395H data (NRS1 and NRS2, three
segments each): transmission spectra from 3, 10 and 20 groups with both corrections, their
paired differences, and the depth factors predicted from the mean ramps. With the CRDS
correction the depths are too deep by 0.17/0.37/0.50% (NRS2) and 0.69/0.88/0.53% (NRS1) for
3/10/20 groups; with the self correction the predicted factors are 1.000 to 0.02%.
