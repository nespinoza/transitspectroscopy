After reading CODEX_REPORT.md, revisiting the extraction code, and reading Marsh’s original paper through ADS, I would prioritize these three changes. The GP extension is mathematically feasible,
  but its scientific advantage needs validation independently of the C-to-Python port. I have not modified any files.

  1. Establish a tested compatibility baseline and repair the failures that prevent existing workflows from running.

  This should precede numerical changes. Currently, missing nominally optional dependencies can prevent package import, parallel fitting discards its results, and some documented argument
  combinations encounter undefined variables.

  I would make this a focused reliability release:

  - Record existing public calls, array layouts, dictionary keys, file conventions, and representative numerical outputs.
  - Make optional dependencies load only when their capabilities are requested, while preserving existing imports and names.
  - Fix direct runtime failures with targeted regression tests.
  - Characterize scientifically consequential behavior—DQ handling, variance propagation, timestamps, and input mutation—before changing it.

  Compatibility requirement: preserve successful existing calls and their outputs. Corrections that change scientific results need explicit documentation and, where necessary, an opt-in corrected
  mode. Reproducing a legacy result and obtaining the scientifically correct result are separate test objectives.

  This gives us an executable definition of backward compatibility, which the current example scripts do not provide.

  2. Port the C algorithms to Python, with Marsh profile estimation designed to support an optional GP model.

  This is the highest-value algorithmic task: the normal installation currently excludes Marsh, leaving optimal extraction unavailable through its advertised wrappers.

  I would deliver it in two independently reviewable stages:

  - Python polynomial implementation: reproduce the current C behavior, including pixel geometry, profile normalization, variance iterations, rejection rules, and return conventions. Keep the C
    implementation available for reference comparisons.

  - Optional GP profile implementation: introduce a different profile model after the polynomial port passes numerical and scientific tests.

  The existing separation between getP() and getOptimalSpectrum() (src/spectroscopy.py:29) already supports this design. Internally, geometry, profile fitting, and weighted extraction should become
  separate components.

  Proposed additive options could distinguish backend='c'|'python' from profile_method='polynomial'|'gp'. Existing calls would retain polynomial behavior; selecting Python must not implicitly select
  a new scientific model. Existing positional arguments, supplied P, return_P, and the inverse-variance output row must remain supported.

  The smaller CCF extension belongs in the conversion effort too, with explicit checks of its lag convention, normalization, numerical differences, and performance.

  3. Make reduction configuration and cached outputs reproducible.

  This becomes particularly important when comparing polynomial and GP extraction: a cached spectrum must not silently substitute for a newly requested method.

  I would introduce resolved configuration records and sidecar manifests containing input identity, extraction settings, package/pipeline versions, CRDS context, and relevant reference files. GP
  products would additionally record kernel, hyperparameters, fitting region, training integrations, and profile-sharing strategy.

  Compatibility requirement: retain current filenames, dictionaries, and legacy cache reading. Introduce explicit validation/recomputation options while preserving legacy behavior for existing
  calls. New GP products should have distinct cache identities from their introduction.

  I would prioritize this over a broad split of jwst.py. Internal modularization is worthwhile, but reproducibility and regression protection provide more immediate scientific value.

  Why the GP proposal is feasible

  Marsh separates determining spatial light fractions from applying extraction weights. His profile model follows the trace, integrates an underlying spatial representation over detector pixels, and
  fits its components jointly. Equations (6)–(8) express the fitted fractions as a sum of geometrical coefficients (Q) multiplied by functions (G_k) that are polynomial along dispersion. Those
  functions provide a replaceable smoothness model. Marsh (1989), §3 (https://adsabs.harvard.edu/pdf/1989PASP..101.1032M).

  Using this repository’s orientation, let (j) denote dispersion column and (i) spatial row. The C reconstruction is:

  [
  \widetilde P_{ij}=\sum_k Q_{kij}G_k(j),
  \qquad
  G_k(j)=\sum_{n=0}^{N-1}a_{kn}j^n.
  ]

  The code subsequently clips negative profile values and normalizes each column. One small specification detail matters: the C parameter described as polynomial “order” actually counts terms,
  giving maximum degree (N-1).

  My proposed extension would replace the polynomial functions with:

  [
  G_k(j)\sim\mathrm{GP}!\left(m_k(j),,\kappa_k(j,j')\right),
  ]

  while retaining the trace-dependent (Q) geometry.

  This GP models spatial-profile evolution along dispersion. It does not require a Gaussian-shaped spatial profile. Trace-position smoothing is a separate modeling decision.

  There is a direct mathematical route. Stack the latent component values into (g), and let (H) map them to detector pixels using (Q). For an unconstrained Gaussian approximation to the empirical
  profile estimates:

  [
  e=Hg+\epsilon,\qquad
  g\sim\mathcal N(m,K),\qquad
  \epsilon\sim\mathcal N(0,\Sigma),
  ]

  so

  [
  \operatorname{Cov}(e)=HKH^\mathsf{T}+\Sigma.
  ]

  This yields a tractable joint Gaussian inference problem at fixed geometry, noise covariance, and hyperparameters. It follows from standard Gaussian conditioning; the application to Marsh’s
  operator is my proposed derivation. Rasmussen & Williams, Chapter 2 (https://gaussianprocess.org/gpml/chapters/RW2.pdf).

  The important qualification is joint inference. Even with independent GP priors for different (k), their posterior estimates are coupled through shared pixels. Fitting unrelated GPs to detector
  rows would not preserve this construction.

  What needs scientific and engineering care

  - Positivity and normalization. Physical profiles require (P_{ij}\geq0) and (\sum_iP_{ij}=1). Clipping and normalizing a GP prediction is possible, but its uncertainty is no longer the original
    Gaussian posterior. A positive latent model, such as exponentiated component amplitudes followed by normalization, offers a principled alternative but requires nonlinear inference.

  - Noise in the training fractions. Dividing pixels by their measured column sum introduces correlated errors. The C implementation uses individual variance estimates. That approximation should
    remain in the compatibility port; the GP design should explicitly assess it against a count-space likelihood.

  - Overfitting and flux bias. A short GP length scale can follow noise, cosmic rays, or trace errors and produce data-dependent extraction weights. I would start with a restricted smooth kernel,
    such as Matérn-3/2 or Matérn-5/2, and assess held-out predictive performance and recovery bias before expanding flexibility.

  - Uncertainty propagation. Existing extraction errors condition on a fitted profile. GP profile uncertainty can introduce covariance between spectral columns—and between integrations when they
    share a profile. Adding a GP variance term independently to each output error would generally be insufficient.

  - Computational cost. A dense covariance over every aperture pixel is unsuitable for long time series. I would first validate small cutouts, then assess inducing-point or state-space
    approximations that preserve the coupled observation model. Reusing trained profiles or hyperparameters may help when measured profile stability supports it.

  With a fixed normalized profile and independent pixel errors, the existing weighted extraction remains:

  [
  \widehat F_j=
  \frac{\sum_iP_{ij}D_{ij}/V_{ij}}
  {\sum_iP_{ij}^{2}/V_{ij}}.
  ]

  Thus, changing profile estimation can leave the extraction interface intact. It does not automatically preserve unbiasedness or calibrated uncertainties.

  The acceptance tests I would require

   Comparison                                                                                 Main question
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   C polynomial versus Python polynomial                                                      Are profiles, fluxes, inverse variances, boundaries, and rejection decisions reproduced within
                                                                                              justified tolerances?
  ─────────────────────────────────────────────────────────────────────────────────────────  ─────────────────────────────────────────────────────────────────────────────────────────────────────────
   Known injected profiles with varying width, asymmetry, curvature, and subpixel position    Does GP fitting improve recovery where polynomial fitting is inadequate?
  ─────────────────────────────────────────────────────────────────────────────────────────  ─────────────────────────────────────────────────────────────────────────────────────────────────────────
   Noise, bad pixels, cosmic rays, and low-flux columns                                       Does flexibility produce bias, false rejection, or underestimated errors?
  ─────────────────────────────────────────────────────────────────────────────────────────  ─────────────────────────────────────────────────────────────────────────────────────────────────────────
   Injected wavelength-dependent transits with pointing and width variations                  Are depths and spectral features preserved, including when profile variations correlate with transit
                                                                                              phase?
  ─────────────────────────────────────────────────────────────────────────────────────────  ─────────────────────────────────────────────────────────────────────────────────────────────────────────
   Null injections and repeated noise realizations                                            Are false features controlled and uncertainty intervals calibrated?

  The injection generator should use pixel-integrated profiles beyond the fitted model family, including empirical profiles where possible. Otherwise, the tests could favor either implementation by
  construction. Simple extraction and extraction using the known true profile should provide controls.

  I would approve a compatibility-first Python port followed by an experimental, explicitly selected GP profile model. Mathematical feasibility is established by the model structure; improved
  precision, unbiased transit recovery, uncertainty calibration, and practical runtime remain the release criteria.
