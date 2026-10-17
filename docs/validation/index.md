# Validation

These pages document reductions of real JWST observations made while developing the
library, with their diagnostics and known limitations. Each validation is run with a script
and a configuration file in the
[`validation/`](https://github.com/nespinoza/transitspectroscopy/tree/master/validation)
folder. Run products (`validation/runs/`) and downloaded data (`validation/data/`) are not
distributed with the repository.

| Validation | Data | Headline result | Pages |
|---|---|---|---|
| NIRSpec/G395H NRS1 | HAT-P-14 b, PID 1118, segment 1 (391 integrations) | White-light RMS after a 100-point filter: box 264.1, polynomial 273.0, GP 265.6 ppm | {doc}`nirspec-nrs1`, {doc}`gp-sweep-report` |
| NIRISS/SOSS | HAT-P-14 b, PID 1541 obs 1 (572 integrations) | Order 1/2 white-light depths consistent with published values; faint order-2 red tail model-sensitive | {doc}`soss`, {doc}`soss-report`, {doc}`soss-optimal-report`, {doc}`soss-overlap-report` |
| MIRI F1500W imaging | LTT 1445 A, PID 9234, two visits (29,689 integrations) | Running-median RMS about 632-638 ppm (GP) per visit; shallow injected eclipses recovered ~30% low | {doc}`miri`, {doc}`miri-report` |
| Ramp fitting | Same MIRI visits, JWST vs correlated-read fitter | JWST more precise in this configuration (GP RMS 639/633 vs 708/706 ppm) | {doc}`miri-ramp-comparison-report` |

```{toctree}
:maxdepth: 1
:hidden:

nirspec-nrs1
gp-sweep-report
soss
soss-report
soss-optimal-report
soss-overlap-report
miri
miri-report
miri-ramp-comparison-report
```
