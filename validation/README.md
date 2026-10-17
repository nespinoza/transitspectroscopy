# Validation scripts

Scripts and configurations used to validate `transitspectroscopy` on real JWST data. The
documentation of each validation, its results and limitations is in
[`docs/validation/`](../docs/validation/index.md).

| Script | Purpose | Configuration |
|---|---|---|
| `validate.py --mode nrs1` | NIRSpec/G395H NRS1 reduction of HAT-P-14 b with diagnostics | `config.json`, `config.segment001-measured.json` |
| `validate.py --mode soss` (`soss_validate.py`, `soss_transit.py`) | NIRISS/SOSS three-order reduction and transit fits | `config.soss-hatp14.json`, `config.soss-hatp14-optimal.json` |
| `validate.py --mode miri` (`miri_validate.py`) | MIRI F1500W photometry of LTT 1445 A | `config.miri-ltt1445.json` |
| `gp_sweep.py` | GP profile hyperparameter sweep from a saved NRS1 run | — |
| `miri_ramp_compare.py`, `miri_ramp_repair_tails.py`, `audit_miri_ramp_comparison.py` | JWST vs correlated-read ramp fitting on MIRI data | — |
| `diagnostics.py` | Shared plotting and metric helpers | — |

Downloaded data go to `validation/data/` and run products to `validation/runs/`; both are
git-ignored. Pinned SOSS reference files are described in `references/soss/README.md`.
