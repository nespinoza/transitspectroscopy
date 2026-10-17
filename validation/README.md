# Validation

Validations of `transitspectroscopy` on real JWST data. Each one is a run configuration in
[`configs/`](configs/): the reduction is done by `ts.run` (the same as
`transitspectroscopy run <config>`), and `validate.py` adds the diagnostic plots and metrics.
The documentation of each validation, its results and limitations is in
[`docs/validation/`](../docs/validation/index.md).

From the repository root, with this checkout installed (`python -m pip install
--no-build-isolation --no-deps .`) in the full environment:

```sh
python validation/validate.py --config validation/configs/nrs1-hatp14b.toml   # or --mode nrs1
python validation/validate.py --config validation/configs/soss-hatp14b.toml   # or --mode soss
python validation/validate.py --config validation/configs/miri-ltt1445a.toml  # or --mode miri
```

Data are downloaded with `ts.jwst.download` into `validation/data/` (existing files are
reused). Outputs go to `validation/runs/<name>/` (`--output DIR` to change it): the
reduction steps in `reduction/`, the diagnostics and `reduction.json` report next to it.
Rerunning reuses completed reduction steps of the same configuration; `--force` recomputes
everything, `--from-step extract` only the extraction. Both folders are git-ignored.

| File | Purpose |
|---|---|
| `validate.py` | Entry point; NRS1 diagnostics |
| `soss_validate.py`, `soss_transit.py` | NIRISS/SOSS diagnostics and transit fits |
| `miri_validate.py` | MIRI photometry diagnostics |
| `diagnostics.py` | Shared plotting and metric helpers |
| `gp_sweep.py` | GP profile hyperparameter sweep from a saved NRS1 run |
| `miri_ramp_compare.py`, `miri_ramp_repair_tails.py`, `audit_miri_ramp_comparison.py` | JWST vs correlated-read ramp fitting on MIRI data |
| `configs/legacy/` | JSON settings of the runs recorded in the reports |

Pinned SOSS reference files are described in `references/soss/README.md`.
