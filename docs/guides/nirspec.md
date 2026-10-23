# NIRSpec time series (PRISM, G395H, G395M)

NIRSpec bright-object time series are reduced with the object-oriented `ts.jwst.load`
workflow for detector calibration, followed by `ts.jwst.stage2` for tracing and extraction.
The {doc}`../quickstart` walks through a complete G395H example; this page summarizes the
choices `transitspectroscopy` makes for each mode.

## Detector calibration

`load.detector_calibration()` runs the JWST `calwebb_detector1` steps on all segments of an
exposure, with these time-series-specific changes:

- **PRISM:** the standard reference-pixel step is replaced by a pedestal estimated from the
  leftmost and rightmost 25 columns of every group (`side_refpix_correction`).
- **Jump detection:** by default, jumps are found with `tso_jumpstep`, which looks for
  outliers in the time series of each pixel's group differences across *all* integrations.
  Defaults are a 200-integration window for PRISM and 10 for the gratings, at 10 sigma.
  Pass `use_tso_jump=False` to use the JWST jump step instead.
- **Group-level 1/f (default):** after a first ramp fit, `fit_ramps` traces the median rate
  and, for every group `k`, subtracts the median over all integrations of group `k` (each
  divided by its white-light factor, the same as the Stage 2 1/f correction), times the
  white-light factor of the integration; the column-by-column median of the remaining pixels
  between `onef_inner_radius` and `onef_outer_radius` from the trace is removed from that
  group, and the ramps are fitted again (`group_difference_1f_correction`;
  `group_difference_1f=False` turns it off). The correction is applied before the linearity
  correction: the input ramps are calibrated again up to the linearity step, corrected (with
  the jump flags of the first pass), and calibrated from the linearity step on. Removed after
  it, the per-integration pedestal and 1/f offsets go through its curvature, larger for
  brighter pixels, and add about 10 ppm to the white light of G395H/NRS1. The
  corrected rates are marked, and Stage 2 does not remove 1/f noise from them again. With
  `parameters={'group_difference_1f': {'domain': 'differences'}}`, the same levels are
  measured on the group differences (CDS frames: the median difference over integrations,
  scaled by the white light, is subtracted), where offsets that change from integration to
  integration cancel, and their cumulative sums are removed from the groups. The same is
  selected with `detector_calibration(group_difference_1f_domain='differences')`, and
  `detector_calibration(ramp_fit_method='sparta')` fits the corrected group differences with
  the correlated ramp fit (see {doc}`ramp-fitting`).
  `parameters={'group_difference_1f': {'template': 'difference'}}` restores the earlier
  correction: templates from the median group differences, scaled to every group in the
  aperture, applied after the linearity correction (on raw counts those factors are biased
  by the non-linear aperture pixels).
- **Group-level 1/f before linearity:** `group_1f=True` subtracts column-by-column medians of
  unilluminated pixels from every group (`group_1f_correction`). Off by default.

Step parameters can be overridden with the `parameters` dictionary, keyed by step name. Every
step's output is saved in the output folder and reloaded if it already exists.

## Ramp fitting

`load.fit_ramps()` uses the JWST ramp-fitting step by default. A correlated adjacent-read
fitter can be selected when the dataset is loaded; see {doc}`ramp-fitting`. With the
group-difference 1/f correction (the default), the ramps are fitted twice: the first rates
give the trace and white-light factors of the correction, and the corrected ramps are fitted
again (saved as `*_groupdiff1f_rateints.fits`).

## Tracing and extraction

`ts.jwst.stage2(inputs, ...)` traces every integration (correlation with a Gaussian kernel,
computed by FFT convolution), smooths the traces with splines, corrects rate-level 1/f noise (`scale_1f=True`;
skipped for rates already corrected at the group level),
and extracts spectra with a fractional-pixel box aperture of `aperture_radius` pixels.
Useful options:

| Option | Meaning |
|---|---|
| `aperture_radius` | Box (or optimal) extraction half-width in pixels |
| `single_trace_extraction` | Use the median trace for all integrations (default) or each integration's trace |
| `optimal_extraction` | Marsh (1989) optimal extraction instead of a box; see {doc}`optimal-extraction` |
| `mask_dq` | Ignore pixels with non-zero DQ flags, except `JUMP_DET` alone |
| `nthreads` | Parallelize tracing and extraction with Ray |
| `outputfolder`, `suffix` | Where and under which name products are cached |

The returned dictionary holds the traces, FWHMs, spectra (`outputs['spectra']`), wavelength
map and the normalized white-light curve (`outputs['whitelight']`).

## Validation

A complete NRS1 reduction of the HAT-P-14 b commissioning data, with diagnostic plots, is
documented in {doc}`../validation/nirspec-nrs1`.
