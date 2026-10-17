# SOSS long-wavelength overlap correction

## Diagnosis

The apparent extraction failures were caused by a hard selection rule in
`src/soss.py`: optimal profile training and extraction used only columns whose
traces were separated by more than `2*aperture_radius + 10` pixels. With the
15-pixel aperture this was a **40-pixel cutoff**, reached around **2.14 µm in
order 1** and **1.08 µm in order 2**. All later optimal columns were deliberately
left missing. They were not optimizer failures or missing PASTASOSS wavelengths.

Farther along the detector the profiles genuinely overlap. At approximately
2.80 µm/order 1 and 1.40 µm/order 2, the centers are only about 20.6 pixels apart.
The corresponding box spectra include light from the other order; simply
removing the mask and applying independent one-profile extraction would not
correct that contamination.

## Implemented correction

The default optimal path now uses `overlap_strategy='joint'`. At each native
detector column it models

`D_i = F_1 P_1i + F_2 P_2i + F_3 P_3i + noise_i`.

Only orders present at a column enter its fit. Order fluxes remain independent
parameters; different orders are not forced to have identical spectra, transit
depths or baselines. For fixed profiles the weighted linear solve returns all
fluxes and the complete conditional covariance matrix. Singular or poorly
conditioned blends are flagged rather than solved with a pseudoinverse.

Profile training also treats the mixture. It initializes each order from pixels
clear of neighboring orders, then jointly refines all intrinsic polynomial/GP
coefficients using the blended detector data. A variable-projection
Gauss–Newton optimizer eliminates the per-column flux increments from the
coefficient system. The fitted coefficients are reused across integrations and
evaluated through each integration's trace geometry. The numerical GP kernel
defaults and the legacy NIRSpec/Marsh extraction paths are unchanged.

The profile support extends five pixels beyond the extraction radius by default
(20-pixel profile support for the standard 15-pixel aperture), allowing the model
to account for contaminating wings. Recovered fluxes and covariances are then
scaled to the original fractional-aperture convention. Invalid pixels remain
excluded; at least half the modeled profile must remain observed. Pixels of an
unmodeled order are masked rather than assumed to contain zero contaminating
light. Detector-edge limitations remain explicit.

Box extraction and its conservative white-light band remain unchanged. Its
overlap flag remains meaningful. For optimal FITS products `DEBLEND=True` marks
joint extraction: geometric overlap (DQ bit 1) is informational, while missing
flux/error and `WHITE_MASK` determine usable measurements.

`order_flux_covariance.npz` in each optimal method directory preserves covariance
in `(integration, detector_column, order, order)` order, together with order
labels, detector columns, times and condition numbers. It includes the
anticorrelation between overlapping order fluxes, which independent extraction
would omit. Unavailable covariances are NaN; each diagonal matches the saved
squared spectral error.

The original behavior is available with
`optimal_options={'overlap_strategy': 'isolated'}`. Other existing options and
box-result keys remain supported. Joint hyperparameter optimization is explicitly
unsupported in this version; kernel values can be supplied, while the older
isolated path retains its optimizer.

This is a **native-column multi-profile optimal extraction**, not a reproduction
of ATOCA's wavelength/line-spread-function coupling across columns. The
[official JWST SOSS description](https://jwst-pipeline.readthedocs.io/en/1.20.2/jwst/extract_1d/description.html)
describes that separate algorithm. The correction removes the hard wavelength
cutoff and models spatial cross-contamination; it does not certify all other
calibration/model assumptions at ppm precision.

## Verification

The new noisy overlap test has a roughly 15-times-brighter contaminating order,
trace motion, a masked pixel, and different injected transit depths of 0.7% and
2%. Both polynomial and GP joint fits recover the separate depths within four
propagated standard errors on genuinely overlapping columns. The blended box
depth of the faint order fails that comparison by more than ten standard errors.
The profile coefficient fingerprints remain fixed throughout extraction.

Additional tests check full covariance against the linear weighted-fit result,
negative inter-order covariance, completely missing and degenerate columns,
masking of unavailable order support, analytic profile gradients against finite
differences, positive-semidefinite projected curvature, three-order product
round trips and agreement between covariance diagonals and spectral errors.

Repository suite: **114 passed, 13 skipped**, with two existing warnings about
shared-profile hyperparameters approaching their bounds. The skipped checks
remain unverified in this environment; they are not counted as passes.

## Executed HAT-P-14 b validation

The final reduction used all **572 integrations** in the four local PID 1541,
observation 1 `rateints` segments. It starts from archive ramp-fitted products,
then applies the post-ramp SOSS background/tracing/extraction workflow; detector
calibration was not rerun from raw ramps for this comparison. Existing validation
configuration files were preserved.

The full run is `validation/runs/soss_overlap_20260925_final`.
Its saved extraction was reused with the latest plotting code in
`validation/runs/soss_overlap_20260925_reviewed`, which is
the final diagnostic set. Its JSON manifest (`validation/runs/soss_overlap_20260925_reviewed/reduction.json`)
records settings, metrics, software, input/source hashes, product hashes and the
parent manifest hash. Reuse regenerated the plots and transit fits without
retraining profiles or re-extracting detector data.

The configuration fits four sampled integrations (IDs 1, 185, 370 and 572),
uses spacing 1.5 pixels, polynomial order 5 or 12 GP inducing points, and keeps
hyperparameters fixed. One intrinsic coefficient set per order/method is reused
for the visit. This run uses the configured fixed trace for extraction;
trace-dependent evaluation is supported and exercised by the simulation tests.
Extraction radius is 15 pixels and profile radius is 20 pixels.

| Order | Polynomial fixed columns | GP fixed columns | Optimal wavelength range (µm) | Previously excluded overlap columns recovered, each method |
| --- | ---: | ---: | --- | ---: |
| 1 | 2012 | 2013 | 0.8493–2.8274 | 700 |
| 2 | 1763 | 1763 | 0.5935–1.4108 | 700 |
| 3 | 1065 | 1065 | 0.6472–0.9537 | 0 |

Fixed columns have usable measurements throughout the visit; a wavelength range
is an extent, not a promise that every intervening column is usable. Larger
profile support excludes some detector-edge columns. The saved box spectra
match the earlier box-only validation arrays exactly, including NaNs.

The complete run took **707 seconds (11.8 minutes)** on this workstation with
one BLAS/OpenMP thread, excluding downloads. Joint refinement took 220 seconds
for polynomial profiles (420 coefficients, 82 iterations) and 167 seconds for GP
profiles (1008 coefficients, 57 iterations). These timings exclude independent
seed fits. Both optimizers met the coefficient Newton-decrement tolerance of
`1e-7`. Regenerating the final diagnostics from saved extraction took 18.6 seconds.

Full optimal-band transit-fit residual RMS values are 121.5/121.7 ppm for order
1, 138.1/139.1 ppm for order 2 and 629.9/626.3 ppm for order 3
(polynomial/GP). These use much wider bands than the conservative box light
curves. On identical columns, the residual RMS values are:

| Order | Common columns | Box (ppm) | Polynomial (ppm) | GP (ppm) |
| --- | ---: | ---: | ---: | ---: |
| 1 | 81 | 180.65 | 179.61 | 179.25 |
| 2 | 100 | 232.78 | 227.66 | 226.66 |
| 3 | 120 | 1556.55 | 1534.14 | 1531.77 |

These are fitted-transit residuals after the configured first-80-integration
trim, not the separately saved 100-point median-filter residual statistic.
Equal-band improvements are modest; a full-band RMS comparison alone would
overstate the extraction gain.

The maximum scaled flux-system condition numbers are 1.445 (polynomial) and
2.306 (GP). The most negative order 1–2 conditional flux correlations are
−0.182 and −0.395, respectively. Thus the fixed-profile linear solves are well
conditioned, even where profile-model uncertainty remains substantial.

An independent product audit passed for all **167 manifest products**: source,
input and product hashes; all nine FITS/NPZ/CSV spectral sets; timestamps and
white-light masks/sums; covariance symmetry, positive semidefiniteness and
diagonal agreement with squared spectral errors; normalized profiles and
minimum observed support; restored red coverage; and unchanged box arrays.

Key review products:

- Three-method spectra for all orders (`validation/runs/soss_overlap_20260925_reviewed/04_soss_spectra.png`).
- Matched-band light-curve comparison (`validation/runs/soss_overlap_20260925_reviewed/08_soss_extraction_comparison.png`).
- Profile residual diagnostics (`validation/runs/soss_overlap_20260925_reviewed/09_soss_profile_fit_quality.png`).
- Inter-order covariance and conditioning (`validation/runs/soss_overlap_20260925_reviewed/11_soss_order_overlap.png`).
- Order 2 throughput consistency (`validation/runs/soss_overlap_20260925_reviewed/12_soss_throughput_consistency.png`).

The method directories also contain the spectra, white-light curves,
transmission fits, portable profile models and `order_flux_covariance.npz`.
Numerical series underlying the new overlap and throughput plots are saved as
CSV files in the reviewed run directory.

## Scientific limits

Restoring coverage and fitting the spatial mixture do **not** settle the
absolute flux of the faint order 2 red tail. The new throughput-consistency
diagnostic compares order 2 against order 1 at the same wavelength, scaled by
the PASTASOSS throughput ratio and dispersion widths. It is not applied as a
constraint or an empirical renormalization of the extracted spectra. It ignores
aperture-loss and response-calibration uncertainty, but exposes discrepancies
that are much larger than the conditional pixel errors.

In particular, the polynomial and GP solutions still differ substantially
beyond about 1.25 µm in order 2. That region remains sensitive to profile wings,
background and model assumptions, despite the well-conditioned linear flux
solve. It must not be treated as a certified transmission measurement merely
because the previous NaNs have been replaced by finite estimates. A calibrated
spatial-profile reference or an independently observed order-1-only profile
would provide a stronger constraint there.

In the final run, median observed/predicted order 2 flux over 0.9–1.1 µm is
1.038 (polynomial) and 1.039 (GP), but above 1.25 µm it is **2.082 and 6.098**.
This is material evidence that the faint red tail is not yet reliably calibrated.
The GP's lower detector residual objective does not establish a more accurate
decomposition of that tail. The throughput prediction is a diagnostic and is
not treated as exact ground truth.

The joint profile residual chi-square per valid pixel in order 1 is about
108 (polynomial) or 96 (GP), far above unity. Most transmission bins reach a
limb-darkening boundary, five order 2 bins are unconstrained for each optimal
method, and one order 3 bin fails. Consequently the saved science status remains
`diagnostics_complete_not_certified`. The abrupt cutoff and independent-order
mixture handling are corrected; faint-tail model calibration remains unresolved.

Flux errors are conditional on the fitted profiles and supplied pixel variance.
They include the linear inter-order covariance, but not uncertainty in the
profile coefficients, trace, background, detector response or spectral mapping.
The previous validation already found profile residuals above the assumed noise
and poorly constrained transmission fits. These remain relevant diagnostics;
a finite deblended measurement is not automatically a precise measurement.
Weak order 2 throughput and negative/weak order 3 regions must still be assessed
using their fluxes, uncertainties and failed-bin flags.
