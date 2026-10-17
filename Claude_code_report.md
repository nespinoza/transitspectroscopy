# Claude Code review of the Codex upgrades to `transitspectroscopy`

*Reviewed 2026-10-16 on branch `dev-codex-upgrades` (HEAD `fd0cc30` plus uncommitted work), compared against `dev` (`8458f6b`).*

This file has two parts:

- **Part A** is the assessment: what Codex changed, what works, what doesn't, and how far the code has moved from your original design.
- **Part B** is a set of **self-contained prompts**. Each can be run on its own in a later session, e.g. *"Implement Prompt 3 from Claude_code_report.md"*. Every prompt ends with a **"Decisions to confirm or edit"** block. Change those lines to steer the work before running the prompt.

---

## Part A — Assessment

### A.1 What I inspected and verified

- Read every Codex document: `CODEX_REPORT.md`, `CODEX_IMPLEMENTATION_RECS.md`, `CODEX_CHANGELOG.md`, `SHARED_PROFILE.md`, `RAMP_FITTING.md`, `tests/README.md`, `validation/README.md`, and the `SOSS_*`, `MIRI_*` and `GP_SWEEP_*` reports.
- Read the diff of every pre-existing module against `dev`. Read the structure and key paths of every new module: `soss.py`, `soss_optimal.py`, `soss_overlap.py`, `miri.py`, `photometry.py`, `ramp_fitting.py`, `shared_profile.py`, `_optimal_extraction.py`, `_ccf.py` and `_optional.py`.
- **Ran the test suite** in the `transitspectroscopy` conda env: **157 passed, 13 skipped, 2 warnings in 26 s**, which matches Codex's claim.
- I did **not** re-run the real-data validations (they need about 15 GB of data that is present locally under `validation/data` and `validation/runs`).

### A.2 Scale of the change

| | `dev` (human) | `dev-codex-upgrades` |
|---|---|---|
| Source modules | 7 (`jwst`, `spectroscopy`, `utils`, `transitfitting`, `timeseries`, `retrievals`, C ext.) | 17 (+ `soss`, `soss_optimal`, `soss_overlap`, `miri`, `photometry`, `ramp_fitting`, `shared_profile`, `_optimal_extraction`, `_ccf`, `_optional`) |
| `jwst.py` | ~3,200 lines | 3,637 lines; `stage2` now has **20 keyword arguments**, including `soss_options` and `miri_options` |
| Tests | 2 example scripts | 18 pytest files, 170 tests, fixtures, benchmark JSONs and comparison figures |
| Prose documentation | README ≈ 206 words | **≈ 39,000 words** across 16 Markdown files |
| Validation | none | 9 scripts, 5 configs, 6 reports; 7.5 GB data and 7.2 GB runs on disk (git-ignored) |
| Version / CHANGELOG | 0.4.1 | still **0.4.1**; none of this work is in `CHANGELOG.md` |
| Tutorials | 4 notebooks | **unchanged**, so none of the new features has a tutorial |

Nearly all new modules (`src/soss*.py`, `src/miri.py`, `src/photometry.py`, `src/ramp_fitting.py`) and most tests and validation scripts are **untracked in git**. A `git clean` or a lost laptop would erase most of this work.

### A.3 What Codex did well (keep these)

1. **It fixed things before adding things.** `CODEX_IMPLEMENTATION_RECS.md` sets a compatibility-first plan, and the code follows it. Several real bugs were fixed:
   - `RAMPS` vs `RAMP` in `download`
   - `is 'None'` → `== 'None'`
   - `interpolate_nans` name errors
   - `nints` miscounted when only some segments are loaded
   - build 11.3 `ERR` removal, handled properly with `ramps_err_available`
   - eager imports of ray, astroquery and pandas that broke `import transitspectroscopy`

   The `_optional.py` lazy-import shim is a clean, minimal solution.
2. **The Marsh algorithm is now pure Python** (`_optimal_extraction.py`). The C/GSL dependency is optional, and the archived C outputs are used as a numerical reference. This was the single most useful upgrade for other astronomers, because optimal extraction used to be impossible to install.
3. **A real test suite now exists**, with legacy-API snapshots (`tests/legacy_api.json`), injection/recovery tests and noisy simulations. Before this, the repo had no executable definition of "still works".
4. **The scientific reporting is honest.** The reports repeatedly separate "the reduction runs" from "the result is unbiased". For example, the GP sweep recommends *not* changing settings, the SOSS report concludes "successful three-order reduction with scientific limitations exposed", and the MIRI report admits the residual PSF structure.
5. **`ramp_fitting.py` is well isolated.** It takes arrays in and returns arrays out, it is detector-agnostic apart from the JWST adapter `fit_model`, its tests use analytic covariance checks, and it is opt-in.
6. **The `SharedProfile` idea is good.** You train once, freeze the model and evaluate it through each trace's geometry. It has `save`/`load`/`fingerprint`, which is the right abstraction for time-series optimal extraction.
7. **Provenance is taken seriously**, with CRDS context, JWST version and input hashes stored in JSON sidecars.

### A.4 Where I disagree, and why

**(1) New instruments were added as separate pipelines, not as modes of your pipeline.** This is the most important architectural issue.

Your design is a single flow: `jwst.load` → custom detector calibration (side refpix, group 1/f, `tso_jumpstep`, LOOM/ROEBA) → `fit_ramps` → `stage2` (trace with your CCF tracer, then extract). Codex added SOSS and MIRI outside that flow:

- `soss.reduce_soss` requires official `rateints`. It uses PASTASOSS traces and the official `BackgroundStep`, has its own tracer (`refine_trace`), box extractor, product writer, product reader and output schema.
- `miri.reduce_miri` runs the official `Detector1Pipeline` and `Image2Pipeline`. It bypasses `jwst.load` and your custom detector steps entirely, even though a MIRI path already existed (`miri/photometry` mode, and the TRAPPIST-1b tutorial).
- `stage2` (`src/jwst.py:2638`) peeks at the first file and **dispatches** to these functions early (`src/jwst.py:2706–2757`). As a result, **`stage2` returns a differently shaped dictionary depending on the instrument**, and it rejects option combinations per instrument.

The result is three pipelines that share a function name. Your detector-level methods, which are the main scientific contribution of the library, do not reach SOSS or MIRI. A user cannot write instrument-agnostic downstream code.

**(2) Instrument knowledge is still hard-coded with `if/elif` on mode strings.** `self.mode == 'nirspec/prism'`-style branches appear about 25 times in `jwst.py`. Codex added more special cases on top instead of introducing a mode description, a central place that says what each mode needs. Adding a new mode (NIRCam grism, MIRI LRS, NIRSpec G140/G235, NRS2) still means editing many scattered branches.

**(3) The GP and profile machinery is duplicated three or four times.** There are:

- Matérn kernels plus inducing points in `_optimal_extraction._gp_basis`
- a second implementation in `shared_profile._basis`
- a third, 2-D version in `photometry.GPPSF`
- `soss_overlap` reusing private `shared_profile` helpers (`_q`, `_geometry`, `_pixel_profile`, `_fit_coefficients`)

`miri.py` imports **private SOSS helpers** (`from .soss import _model, _times, _hash`, `src/miri.py:13`). These cross-module private imports are a sign that a shared core was never factored out.

**(4) Documentation is now hard to navigate, and none of it is a tutorial.** The documentation went from 206 words to about 39,000. The README has become an installation essay about CRDS, NumPy ABIs and MultiNest path limits. To learn how to reduce SOSS you must read `README` → `validation/SOSS_README.md` → three reports. Changelog-style prose ("this upgrade does not infer…", "this milestone…") leaks into user-facing docs. There is no notebook for any new capability, and the existing tutorials were not checked against the new code. For an astronomer picking this up, the barrier to entry has **risen**.

**(5) The code style has moved away from yours and away from what astronomers can easily read.** The new modules use semicolon-chained statements, one-letter names, 110+ character lines, deeply nested dictionary plumbing, and docstrings that list caveats rather than explain parameters. `jwst.py` mixes your verbose, commented style with dense Codex blocks. This matters for an astronomy library because users read the code to understand the method.

**(6) Reproducibility is enforced in ways that make reuse harder.** `soss.load_products` **refuses to load** saved products if any of five source files changed by a single byte (`src/soss.py:571–576`). Any bug fix, `pip install -U` or whitespace edit invalidates every saved reduction. That is unusual: normal practice is to record versions and hashes, then warn, and refuse only on an explicit `strict=True`. Meanwhile there is still no single config-driven way to rerun an analysis. Each validation script has its own argparse/JSON schema, loads the checkout through `importlib` spec hacks rather than the installed package, and downloads data with its own MAST URL code rather than `ts.jwst.download`.

**(7) One questionable scientific default and one under-discussed result.**

- `ramp_fitting.fit_ramps(..., residual_correction=True)` is the **default** (`src/ramp_fitting.py:101`). The empirical median residual template is trained on all integrations of a segment, which includes in-transit ones. The docs warn about variable scenes, and the README example sets it to `False`, but the default should match the safe choice.
- In the MIRI injection test, both shallow 300 ppm eclipses are recovered at about 212 ppm (−30%, about −2.6σ each, `validation/MIRI_VALIDATION_REPORT.md`). Two same-sign misses on independent events point to a systematic bias in the 2-D GP photometry for shallow events, which is exactly the eclipse regime MIRI F1500W is used for. The report notes it, but this should block recommending the GP photometry for eclipse science until it is understood.

**(8) Packaging and repo hygiene did not keep up.**

- Still a `setup.py` with `python_requires='>=3.0'`, which is not a tested range. `jwst` is correctly a hard dependency and must stay one: the library is built around the JWST pipeline. However, the supported Python floor is undeclared, and `environment.yml` pins Python 3.12. Project policy is **Python ≥ 3.13** (see Prompt 1).
- A compiled `CCF.cpython-311-darwin.so`, `build/`, `.README.md.swp`, `__pycache__` folders and an `upgrade.sh` that uploads to PyPI all sit in the working tree.
- There is no CI, so the 170 tests only run when someone remembers to run them.
- `transitfitting` still hard-codes the juliet instrument name `'SOSS'` for every dataset (`src/transitfitting.py:177`).

### A.5 How far from the original design, and do I agree?

**Distance:** substantial. The original was a small, opinionated toolkit: your detector methods plus a thin wrapper around the JWST pipeline, written in a readable, heavily commented style, with notebooks as the documentation. The Codex branch keeps that core intact and backward-compatible, which is to its credit. Around it, Codex built a second, much larger system with a different philosophy: official-pipeline-first, validation-report-driven, a large API surface with many option dictionaries, and its own coding style.

**Verdict: I agree with about half of the direction.**

- **Agree:** compatibility-first bug fixing, the pure-Python Marsh port, the test suite, opt-in GP profiles, the shared/frozen profile model, the isolated alternative ramp fitter, provenance recording, and honest scientific reporting. Keep all of these.
- **Disagree:** adding instruments as parallel pipelines dispatched from `stage2`, per-instrument output schemas, duplicated GP cores, documentation sprawl with no tutorials, the style drift, all-or-nothing source-hash enforcement, the `residual_correction=True` default, and leaving the work uncommitted and unversioned.

The path forward is not to revert. It is to **refactor what Codex built** into the following:

1. an observatory-agnostic **core** (arrays in, arrays out: tracing, extraction, profiles, photometry, ramp fitting, light curves, binning)
2. a declarative **instrument/mode registry**
3. **observatory adapters** (JWST first) that turn files into the core's inputs
4. **one product schema** and **one config-driven runner**
5. short docs plus **one tutorial per mode**

Your custom detector methods should become reusable steps that any mode can turn on. That fixes generalizability (new modes and observatories become plug-ins) and reproducibility (one config file reruns everything) together.

---

## Part B — Implementation prompts

Run them roughly in order. Prompts 0–2 are low-risk groundwork. Prompts 3–5 are the core refactor. Prompts 6–9 extend the library. Each prompt assumes a fresh session that has only this file and the repo.

> **Global constraints for every prompt:**
> - Preserve every public name and signature that existed on `dev` (`tests/legacy_api.json` is the contract). Deprecate; don't delete.
> - Do not change numerical outputs of existing default paths unless the prompt says so. Run `python -m pytest tests -q` in the `transitspectroscopy` conda env before and after; report the counts. (After Prompt 1, that env must be the rebuilt Python 3.13 one.)
> - `jwst` (the STScI pipeline) is a **required** dependency of `transitspectroscopy`, never an optional extra. The supported Python is **≥ 3.13**. No prompt should make `jwst` optional or lower the Python floor.
> - Match the author's readable style in new code: numpy-style docstrings, descriptive names, no semicolon-chained statements, lines ≤ 100 characters, and comments that explain the astronomy.
> - Work on a new branch off `dev-codex-upgrades`; commit in small, reviewable commits. Do not push or open PRs unless asked.

---

### Prompt 0 — Secure the work and clean the repository

```
Context: On branch dev-codex-upgrades, most new modules (src/soss*.py, src/miri.py,
src/photometry.py, src/ramp_fitting.py), tests and validation scripts are untracked.
Version is still 0.4.1 and CHANGELOG.md does not mention any Codex work.

Tasks:
1. Create branch `refactor/consolidation` from dev-codex-upgrades. Commit all
   untracked source, tests, fixtures and validation scripts/configs/reports
   (NOT validation/data or validation/runs). Check sizes first; anything > 5 MB
   must be listed to me before committing.
2. Remove from the working tree and ignore: CCF.*.so, build/, *.egg-info,
   .README.md.swp, __pycache__/, .pytest_cache/, ts_outputs/. Verify .gitignore.
3. Bump version to 0.5.0.dev0. Add a 0.5.0 (unreleased) section to CHANGELOG.md
   that summarizes, in Keep-a-Changelog form, every user-visible change recorded
   in CODEX_CHANGELOG.md (Added / Changed / Fixed / Deprecated), max ~60 lines.
4. Move CODEX_REPORT.md, CODEX_IMPLEMENTATION_RECS.md, CODEX_CHANGELOG.md into
   docs/dev-notes/ (keep them verbatim; they are history, not user docs).
5. Change the ramp_fitting.fit_ramps default to residual_correction=False, and
   make every caller and doc consistent with it. Add a test asserting the default.
   Note it under "Changed" in CHANGELOG.

Acceptance: test suite passes with the same counts; `git status` clean except
ignored data; CHANGELOG reads as a user-facing summary.
```

---

### Prompt 1 — Packaging that installs with the JWST stack, plus CI

```
Context: setup.py declares python_requires='>=3.0' (untested) and has no CI.
`jwst` is, and must remain, a REQUIRED dependency: transitspectroscopy is built
around the STScI JWST pipeline (jwst 3.0.0 is the current release,
https://github.com/spacetelescope/jwst). The project's supported Python is >=3.13.
environment.yml and the existing conda env currently use Python 3.12 and must be
moved to 3.13.

Tasks:
1. Replace setup.py/setup.cfg with pyproject.toml (setuptools backend; keep the
   optional CCF C extension, still optional at build time).
   requires-python = ">=3.13". Also add the same upper bound that the pinned jwst
   release declares (read it from jwst's own metadata; do not guess).
   Required deps: jwst (>=3.0.0), stdatamodels, crds, gwcs, numpy, scipy,
   astropy, tqdm, jdcal. Optional extras (things that are truly optional):
   [fitting] (juliet, dynesty), [parallel] (ray), [download] (astroquery, pandas),
   [plots] (matplotlib), [test] (pytest), [all].
2. Keep the lazy imports of the truly optional packages (ray, astroquery, pandas,
   juliet) from src/_optional.py, so `import transitspectroscopy` works without
   the extras. jwst/stdatamodels may be imported normally since they are always
   installed. Keep every `from transitspectroscopy import *` name from dev
   (tests/legacy_api.json).
3. Update environment.yml to python=3.13. Check that every pinned conda/pip
   package (multinest, pymultinest, celerite, george, dynesty, ray, juliet, jwst,
   crds, stdatamodels, astropy, numpy, scipy) has Python 3.13 builds for
   linux-64 and osx-arm64. Report any that don't, with the closest working pin,
   BEFORE changing them. Rebuild the env and rerun the full suite.
4. Add GitHub Actions on Linux and macOS, Python 3.13 (add 3.14 as a
   non-blocking job if jwst supports it): (a) a fast job, `pip install .[test]`,
   running the unit tests; (b) a full job from environment.yml running all tests.
   Tests that need CRDS reference downloads or network get a pytest marker and are
   skipped by default in CI.
5. Shorten the README install section to: conda full env (recommended), pip
   install (with extras), and a link to docs/installation.md holding the CRDS /
   MultiNest / C-reference details. State Python >=3.13 and jwst as required.

Acceptance: `pip install .` on Python 3.13 pulls in jwst and imports the package;
installing on Python <=3.12 is refused by pip; the rebuilt 3.13 env passes the
whole suite (report counts vs the 157 passed / 13 skipped baseline); CI YAML
validated locally (act or yamllint).
```

---

### Prompt 2 — Documentation users can learn from: short README, docs site, one tutorial per mode

```
Context: README grew from ~200 to ~2,000 words of installation caveats; user
guidance is scattered over ~16 Markdown files totalling ~39k words, and no
tutorial covers the new features (Python optimal extraction, GP profiles,
SharedProfile, SOSS, MIRI photometry, alternative ramp fitting).

Tasks:
1. Create docs/ (Sphinx + myst-nb or mkdocs-material; choose per decision below)
   with: installation, quickstart, concepts (array conventions table from
   docs/dev-notes/CODEX_REPORT.md, the data flow diagram), per-mode user guides,
   API reference generated from docstrings, validation summaries (1 page each,
   linking to full reports), changelog.
2. Rewrite README.md to <= 400 words: what it is, install one-liner, 15-line
   example, links to docs, citation. Keep author voice and citation block.
3. Move validation/*_README.md and *_REPORT.md under docs/validation/ (keep
   content; trim changelog-style phrasing like "this upgrade", "this milestone").
4. Write small, fast tutorials (notebooks executed in CI on synthetic data,
   < 2 min each): optimal extraction (polynomial vs GP), SharedProfile train/
   save/load/extract, alternative ramp fitting on synthetic ramps. Write real-data
   tutorials (not executed in CI): SOSS HAT-P-14b, MIRI LTT 1445A.
5. Re-run the 4 existing tutorials against the current code; fix breakages in the
   notebooks (not in the library unless it is a genuine regression — report those).
6. Configure Read the Docs.

Acceptance: docs build without warnings; README renders; each tutorial runs.
```

---

### Prompt 3 — Factor out an observatory-agnostic core and remove the duplicated GP code

```
Context: GP/Matérn/inducing-point profile code exists in at least three places
(src/_optimal_extraction.py:_gp_basis, src/shared_profile.py:_basis,
src/photometry.py:GPPSF), and modules import each other's private helpers
(src/miri.py:13 imports soss._model/_times/_hash; soss_overlap imports
shared_profile._q/_geometry/_pixel_profile/_fit_coefficients).

Goal: a `transitspectroscopy.core` subpackage operating only on numpy arrays +
small dataclasses. jwst remains a required dependency of the package; this is
purely an internal layering rule: core modules do not import jwst/stdatamodels,
so the same algorithms can be fed by other observatory adapters (Prompt 8) and
unit-tested on plain arrays:
  core/kernels.py        Matérn 3/2, 5/2 (+ squared-exp), whitened inducing bases, 1-D and 2-D
  core/geometry.py       Marsh pixel-overlap geometry (Q), apertures, fractional weights
  core/profiles.py       polynomial & GP profile models (one implementation each),
                         SharedProfile, joint multi-trace (overlap) fitting
  core/extraction.py     box, Marsh optimal, joint multi-order solve
  core/tracing.py        trace_spectrum + CCF utilities (from spectroscopy.py), trace refinement
  core/photometry.py     aperture, background annulus, centroid, GP PSF
  core/ramps.py          correlated-read ramp fitting (array-level part of ramp_fitting.py)
  core/detector.py       your custom detector algorithms that are array-level
                         (side refpix, group 1/f, tso_jumpstep, LOOM, ROEBA, CDS, uniluminated masks)
  core/timeseries.py, core/binning.py (bin_at_resolution, etc.)

Tasks:
1. Move/merge code into these modules; keep thin re-export shims at every old
   import location (spectroscopy.getOptimalSpectrum, ts.fit_shared_profile,
   jwst.get_loom, etc.) so tests/legacy_api.json still holds.
2. One kernel/basis implementation shared by 1-D profiles and 2-D PSF.
3. A small shared I/O/provenance helper (core/provenance.py) replaces the private
   _hash/_model/_times cross-imports.
4. Numerical equivalence: before refactoring, record outputs of every existing
   test fixture (spectra, profiles, PSFs, ramp rates) to a temp npz; after
   refactoring assert np.allclose(rtol=1e-12) or explain any change.
5. Restyle moved code per the global style constraints.

Acceptance: a test asserts that no module under transitspectroscopy/core imports
jwst, stdatamodels or gwcs (e.g. by inspecting sys.modules after importing core in
a subprocess); full suite passes; equivalence check passes; no module imports
another module's underscore-prefixed names.
```

---

### Prompt 4 — Instrument/mode registry and one `stage2`, without per-instrument forks

```
Context: Modes are hard-coded via ~25 `self.mode == '...'` branches in
src/jwst.py, and stage2 (src/jwst.py:2638) dispatches SOSS and MIRI to separate
pipelines (soss.reduce_soss, miri.reduce_miri) that return different dict shapes,
bypass jwst.load and skip the author's custom detector calibration.

Goal: a declarative registry describing each observing mode, used by both the
detector stage and the extraction stage.

Tasks:
1. Create `transitspectroscopy/instruments/` with a `Mode` dataclass, e.g.:
     name, observatory, instrument, detector(s), disperser/filter/pupil match rules,
     kind ('slitless_spectroscopy'|'slit_spectroscopy'|'imaging'),
     dispersion_axis, n_orders, trace_source ('ccf'|'reference_model'|callable),
     wavelength_source, default aperture/background geometry,
     detector_steps (which custom steps are on by default: refpix pedestal,
     group_1f, tso_jump window/nsigma, ...), background_strategy,
     extraction_defaults, known_limitations (str).
   Register existing modes: nirspec/prism, nirspec/g395h (nrs1 and nrs2),
   nirspec/g395m, niriss/soss (orders 1-3, PASTASOSS traces, use the jwst pipeline background 
   correction modelling), miri/imaging.
2. Replace every `self.mode == ...` branch in jwst.py with lookups on the Mode.
   Unknown modes should fail with a message listing registered modes and pointing
   to docs on how to register one.
3. Rebuild SOSS and MIRI as modes inside the main flow:
   jwst.load → detector_calibration (custom steps chosen by the Mode; official
   Detector1 still available as an option) → fit_ramps → stage2 → Mode-specific
   tracing/background → core extraction. Keep soss.reduce_soss and
   miri.reduce_miri as thin, deprecated convenience wrappers calling this flow. The 
   only change here is, in SOSS, to be careful to call and use the background subtraction 
   step to remove zodiacal light before tracing/extraction (see: https://jwst-pipeline.readthedocs.io/en/stable/jwst/background_subtraction/description.html). 
4. Collapse stage2's per-instrument kwargs (soss_options, miri_options,
   shared_profile_options, extraction_options, gp_options) into a single
   `ExtractionConfig` (dataclass) with mode-aware validation; keep old kwargs
   working with a DeprecationWarning that maps them onto the config.
5. Test: each registered mode on small synthetic cubes; and re-run SOSS and MIRI
   validations (see validation/configs) to confirm results match the previous
   runs within stated tolerances (white-light RMS, depths), reporting the diff.

Acceptance: adding a new mode = one registry entry + (optionally) a tracing or
background plug-in; no edits to jwst.py branching. Old calls still work.
```

---

### Prompt 5 — One product schema, one config-driven runner, and provenance that warns instead of refusing

```
Context: Each path writes different products (stage2 dicts, soss order*_spectra.
{npz,fits,csv}, miri photometry.fits ...). soss.load_products refuses to load if
any source file hash changed (src/soss.py:571-576). Validation scripts each have
their own argparse/JSON schema, load the checkout via importlib hacks and
re-implement MAST downloads.

Tasks:
1. Define product dataclasses in core/products.py:
     SpectroscopicTimeSeries(times_bjd_tdb, wavelength[, order], flux, flux_err,
       dq, trace, trace_width, covariance (optional), metadata)
     PhotometricTimeSeries(times, flux, flux_err, dq, centroid, background, metadata)
     LightCurveSet(...) for binned/white-light curves.
   One writer/reader: FITS (multi-extension, x1dints-like where sensible) +
   ECSV for light curves. Every reduction path returns and writes these.
2. Provenance manifest (JSON next to products): package version + git commit +
   dirty flag, Python/numpy/jwst/crds versions, CRDS context, input file names +
   sha256, full resolved config, random seeds, timestamps. Reading products with a
   different code version emits a warning; `strict=True` raises.
3. A runner: `transitspectroscopy run config.toml` (console entry point) and
   `ts.run(config)`. The config holds: data source (MAST pid/obs/exposure/segments
   or local glob), mode (auto-detected, overridable), detector steps, ramp
   fitting method, extraction config, binning, optional light-curve fit. Steps
   cache by config hash; `--force` / `--from-step` to recompute.
4. Port validation/validate.py, soss_validate.py and miri_validate.py to thin
   wrappers over the runner + a diagnostics module; they must import the
   installed package and use ts.jwst.download. Keep their configs as runner configs
   under validation/configs/.
5. Optionally save SharedProfile / PSF models inside the product file.

Acceptance: the three validations rerun from config alone and reproduce their
reported headline numbers; editing a docstring no longer invalidates saved
products (only warns).
```

---

### Prompt 6 — Light-curve fitting that is instrument-agnostic

```
Context: transitfitting hard-codes the juliet instrument name 'SOSS' for every
dataset (src/transitfitting.py:177 and others), duplicates fit_data /
notremote_fit_data, and the SOSS validation has its own transit script
(validation/soss_transit.py).

Tasks:
1. Accept LightCurveSet / SpectroscopicTimeSeries from Prompt 5; derive instrument
   keys from metadata; support joint fits of multiple instruments/visits with
   shared transit parameters and per-instrument systematics.
2. Merge the remote/non-remote duplicates; keep old functions as wrappers.
3. Add a backend interface (juliet default) behind one function, and a chromatic wrapper: fit white light, fix
   chosen geometry, fit each bin (serial or Ray), assemble a transmission/emission
   spectrum product with provenance.
4. Make limb-darkening priors pluggable rather than hard-coded. Limb-darkening values (eg, as means of normal distributions, standard deviations, etc.) 
   should always be passed by the user explicitly (eg as a dictionary). This should be explained in a simple tutorial as in previous prompts. Philosophy should be 
   users using either Kipping+2013 q1,q2, using u1, u2 or non-linear coefficients c1,c2,c3,c4, all with either uniform priors, normal or uniform priors. Default should 
   be uniform, quadratic, u1 and u2 with uniform priors between -3 and 3 each.
5. Replace validation/soss_transit.py with the new API.

Acceptance: a synthetic two-instrument chromatic test recovers injected depths
within 2σ; legacy fit_lightcurves calls still work.
```

Stop here.
---

### Prompt 7 — Add more JWST modes through the registry

```
Context: After Prompt 4, modes are registry entries. Currently supported:
NIRSpec PRISM/G395H/G395M, NIRISS SOSS, MIRI imaging.

Tasks (one commit per mode; each with a synthetic test and a short user guide):
1. NIRSpec NRS2 for G395H/G395M (if not done in Prompt 4), G140H/M, G235H/M
   (F070LP/F100LP/F170LP), including the subarray/readout direction details.
2. NIRCam grism time series (F322W2, F444W; GRISMR, SUBGRISM64/128/256) incl. its
   1/f behavior (row-wise) and the simultaneous short-wave weak-lens photometry
   via core/photometry.
3. MIRI LRS slitless (SLITLESSPRISM), incl. background from adjacent columns and
   the known shadowed region / brighter-fatter related considerations.
4. NIRISS SOSS SUBSTRIP96 and F277W calibration exposures.
For each: wavelength solution from the JWST pipeline (assign_wcs / reference files),
default apertures, default detector steps, documented known limitations.
Optional real-data smoke tests: one public TSO per mode, first segment only,
behind a `--run-realdata` pytest flag.

Acceptance: `ts.instruments.list_modes()` shows them; each synthetic test passes;
real-data smoke tests run end-to-end on the chosen public programs.
```

**Decisions to confirm or edit:**
- Priority order: NIRCam grism → MIRI LRS → NIRSpec G140/G235 → NRS2 → SOSS SUBSTRIP96
- Public programs for the smoke tests: *(list the PIDs you trust, e.g. ERS 1366 targets)*

---

### Prompt 8 — Support for other observatories via adapters

```
Context: After Prompts 3–5, core is observatory-agnostic and products are
standard. The JWST code becomes one adapter. jwst stays a required dependency of
the package (Python >=3.13) even for users of other adapters; the adapter layer is
about code structure, not about making jwst optional.

Tasks:
1. Define an `ObservatoryAdapter` protocol in transitspectroscopy/adapters/base.py:
     discover/download(…) -> list of files
     load(files) -> detector cube(s) (ramps or rates), variance, dq, times (BJD_TDB),
                    header metadata, Mode
     wavelength_solution(mode, trace) -> wavelengths
   Move JWST code behind adapters/jwst.py (keep ts.jwst as the public name).
2. Implement ONE additional adapter end-to-end (see decision below), with a
   tutorial and a validation against a published spectrum.
3. Document "how to add an observatory/instrument" in docs, using the new
   adapter as the worked example.
4. Keep times in BJD_TDB everywhere; adapters are responsible for conversion
   (astropy.time + barycentric correction for ground-based data).

Acceptance: the new adapter produces SpectroscopicTimeSeries products, runs
through the same extraction + light-curve fitting code, and reproduces a
published white-light depth within errors.
```

**Decisions to confirm or edit:**
- First non-JWST adapter, one of:
  - **(a) HST/WFC3 G141/G102 spatial scan.** The most-used legacy transmission data; IMA reads map naturally onto your ramp/CDS tools.
  - **(b) Ground-based multi-object spectroscopy** (Magellan/IMACS or LDSS3, VLT/FORS2, the ACCESS-style data). It exercises comparison-star differential spectroscopy, which would add a "reference star" concept to the core.
  - **(c) Future missions** (Pandora, Ariel) with simulated data.

  *Recommended: (a) first, because it is closest to the current architecture; (b) second, because it forces the useful comparison-star generalization.*

---

### Prompt 9 — Scientific validation that can be reproduced, plus investigating the MIRI GP bias

```
Context: Validations live as local 7.5 GB data + 7.2 GB runs; reports are
excellent but not re-runnable by others without heavy setup. MIRI 2-D GP
photometry recovered both injected 300 ppm eclipses at ~212 ppm (~-30%,
~-2.6σ each; validation/MIRI_VALIDATION_REPORT.md).

Tasks:
1. Investigate the MIRI shallow-event bias: injection ensemble (>= 50 noise
   realizations × depths 100–1000 ppm × centroid jitter levels), comparing GP,
   aperture sum, and GP with profile uncertainty marginalized or with PSF
   retrained excluding in-event integrations. Determine whether the bias comes from
   the PSF training including in-event frames, inducing-grid smoothing, or the
   flux normalization. Fix or document; until resolved, emit a warning when GP
   photometry is used for events shallower than a configurable threshold.
2. Build a standard injection–recovery harness in core (spectroscopy and
   photometry) used by all modes: null tests, depth bias vs SNR, error calibration
   (pull distributions), chromatic injection. Include in CI on small synthetic data.
3. Make the real-data validations shareable: a `validation/` runner config per
   benchmark + a script to download exactly the needed public segments; publish
   reduced products + manifests to Zenodo (not git); docs pages summarize the
   headline metrics with links.
4. Add a benchmark table to docs comparing transitspectroscopy with published
   results (HAT-P-14b NIRSpec/SOSS, LTT 1445A MIRI, ...) with exact config hashes.

Acceptance: MIRI bias explained with figures; pull distributions within
[0.9, 1.1] width for default settings on synthetic data, or documented otherwise;
a new user can reproduce one benchmark with two commands.
```

**Decisions to confirm or edit:**
- Bias threshold for the warning: 500 ppm
- Zenodo community or record to use: *(fill in)*

---

### Optional Prompt 10 — Style pass on Codex-written modules (if you skip Prompt 3)

```
Restyle src/soss.py, soss_optimal.py, soss_overlap.py, miri.py, photometry.py,
ramp_fitting.py, shared_profile.py, _optimal_extraction.py and the Codex-added
parts of jwst.py to the author's style: numpy-style docstrings with Parameters/
Returns, descriptive variable names, no semicolon-chained statements, <= 100-char
lines, comments explaining astronomy rather than listing caveats. Purely
cosmetic: assert identical outputs on the full test suite and on saved fixtures.
Run ruff with a config matching this style and add it to CI.
```

---

### Suggested order and effort

| Prompt | Risk | Effort | Unlocks |
|---|---|---|---|
| 0 Hygiene | very low | small | everything else; protects the untracked work |
| 1 Packaging + CI | low–medium (Python 3.13 env rebuild) | small–medium | a declared, tested Python ≥3.13 + jwst install; regressions caught automatically |
| 2 Docs + tutorials | low | medium | adoption by other astronomers |
| 3 Core extraction | medium | large | Prompts 4, 7, 8 |
| 4 Mode registry | medium–high | large | new JWST modes cheaply |
| 5 Products + runner | medium | medium–large | one-command reproducibility |
| 6 Light-curve fitting | medium | medium | end-to-end spectra for any mode |
| 7 More JWST modes | medium | medium per mode | coverage |
| 8 Other observatories | high | large | generalizability beyond JWST |
| 9 Validation + MIRI bias | medium | medium | scientific trust |
