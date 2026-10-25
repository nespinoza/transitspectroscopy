# Inspecting ramps

The ramp viewer is an interactive figure for finding what causes a feature in a light curve:
a jump, a frame-wide offset, a moving artefact, a pixel that misbehaves at one calibration
stage. It shows

- **top**: the detector image of one integration and one group, in one of several modes
  (below), with the aperture around the trace (dashed) and the pixels flagged as jumps;
- **middle**: the mean of that image within the aperture and its median in the background
  (rows far from the trace), **against the integrations** (at the group shown) **or against
  the groups** (of the integration shown), with the number of aperture pixels with a new
  jump (green);
- **bottom** (optional): your own curves against the integration, e.g. light curves or the
  residuals of a transit fit.

Sliders select the integration and the group; a button (or the `a` key) swaps the middle panel
between integrations and groups. matplotlib is needed (`pip install
transitspectroscopy[plots]`).

## From a calibrated dataset

After `detector_calibration`, open the viewer on the ramps of any calibration stage:

```python
import transitspectroscopy as ts

dataset = ts.jwst.load(filenames, outputfolder='reduction')
dataset.detector_calibration()
dataset.fit_ramps()

viewer = dataset.inspect_ramps()                         # the calibrated ramps
viewer = dataset.inspect_ramps('input', segments=0)      # the uncal ramps
viewer = dataset.inspect_ramps('linearity', segments=0)  # after the linearity step
viewer = dataset.inspect_ramps('jump', segments=0)       # after the (TSO) jump step
```

`stage` is `'calibrated'` (default; the current `dataset.ramps`, i.e. after the jump step,
or the ramps corrected for group-difference 1/f noise once `fit_ramps` applied it),
`'input'`, or the name of a detector step of the mode (`'dq_init'`, `'saturation'`,
`'superbias'`, `'refpix'`, `'group_1f'`, `'linearity'`, `'dark_sub'`, `'jump'`, ...): the
ramps right after it. For those, the input ramps of the selected segments are calibrated
again (nothing is saved) with the steps and parameters `detector_calibration` used; this needs
memory for another copy of the ramps, so select one segment (`segments=0`) for large
datasets. These ramps do not include the group-difference 1/f correction of `fit_ramps`.

The rates per integration of the dataset (`dataset.rateints`) are added automatically; the
integration numbers shown are those of the whole dataset.

## From arrays

`ts.utils.inspect_ramps` takes any ramps (integration x group x row x column; numpy arrays,
memory maps, the `data` of a JWST `RampModel`):

```python
import numpy as np
import transitspectroscopy as ts

ramps = np.load('raw.npy', mmap_mode='r')        # read one group or integration at a time
groupdq = np.load('groupdq.npy', mmap_mode='r')
viewer = ts.utils.inspect_ramps(
    ramps, groupdq,
    rateints=rates_self, comparison_rateints=rates_crds,   # 'rate ratio' mode
    curves={'CRDS residuals (ppm)': residual_crds, 'self residuals (ppm)': residual_self},
    integration=44,
)
```

`show=False` returns the viewer without showing it; `viewer.set(integration=..., group=...,
mode=..., axis=...)` changes what it shows and `viewer.save('figure.png')` saves it, which
also works without a display (`MPLBACKEND=Agg`). Keep the returned viewer referenced: its
widgets stop responding once it is garbage collected. In a notebook, use an interactive
backend (`%matplotlib widget`).

## Image modes

| mode | shows |
|---|---|
| `counts` | the counts of the group (DN) |
| `counts - median` | the counts minus their median over the integrations, same group (DN) |
| `counts - median (sigma)` | that, in units of its noise |
| `increment` | the group minus the previous group (DN) |
| `increment - median` | the increment minus its median over the integrations (DN) |
| `increment - median (sigma)` | that, in units of its noise |
| `rate / median - 1` | with `rateints`: the rate over the median rate (%) |
| `rate ratio / median - 1` | with `rateints` and `comparison_rateints`: their ratio, relative to its median (%) |

The noise is the robust scatter over the integrations of every pixel and group, or, with
`read_noise` (DN) and `gain` (e/DN), `sqrt(read_noise**2 + counts/gain)` (for ramps with bias
and dark subtracted). The medians and the scatter use up to `reference_integrations` (100)
evenly spaced integrations, and are computed once per group, when first needed.

How features look:

- **light** (a change in the illumination) grows over the groups: in the increments it is in
  every group;
- **an additive offset** (bias, a moving artefact) is the same in every group of the counts,
  and only in group 0 of the increments: the middle panel against the groups is flat in
  `counts - median` and has one point off zero in `increment - median`;
- **a jump** is in the increments only in the group where it happens (red), and in the counts
  from that group on.

## Aperture and trace

The trace is estimated from the median counts of the last group minus those of the first
(the centroid of every column, smoothed by a polynomial; `ts.inspection.estimate_trace`),
which suits traces along the columns (NIRSpec, NIRCam). Give `trace=` (the centre row of every
column) for other geometries, or `trace=False` to use the whole frame as the aperture.
`aperture_radius` (2.5 rows) sets the aperture; rows farther than `background_distance`
(three times the aperture radius) from the trace are the background.

## Controls

| control | action |
|---|---|
| integration / group sliders, ← → / ↑ ↓ | change the integration / group |
| mode buttons | change the image mode |
| `integrations` / `groups` buttons, `a` | swap the axis of the middle panel |
| `jumps` box | show or hide the jump flags |
| `c` | cycle the colour range (3, 5, 10, 30 robust standard deviations of the aperture) |
