# Group 1/f study

Scripts behind the group 1/f defaults (white-light factors and 1/f pixels of every mode); the
study and its results are in `Claude_progress.md` (October 2026 entries).

| File | Purpose |
|---|---|
| `onef_extract.py` | Stage 1 of a chunk of integrations of one segment; saves the inputs of `jwst.group_difference_1f_from_traces` (float16 ramps, masks, first-pass rates, geometry) |
| `chain.sh` | Runs `onef_extract.py` on each segment in 3 chunks (fits in 8 GB) |
| `onef_homog.py`, `onef_homog_summary.py` | Templates (median, two-term), pixels and factors compared: white-light rms and depth change |
| `onef_sweep.py`, `onef_sweep_summary.py` | 1/f pixel bands (inner distance x width from the trace centre): rms, chromatic injection kept, noise floor |
| `sweep_driver.sh` | Extraction and sweep segment by segment, removing the inputs after each |
| `results/sweep_<mode>.csv` | Sweep summaries of all segments (G395H NRS2, PRISM, SOSS) |

Runs (from the repository root; `PYTHON` selects the interpreter; outputs in
`validation/runs/onef-full/<name>/`):

```sh
bash validation/onef/sweep_driver.sh g395h 18.04 2,3,4,5,6,7,8,9,10 1,2,3,4,6,8,12,all 2 \
    validation/data/jw01118005001_04101_00001-seg00{1,2,3}_nrs2_uncal.fits
bash validation/onef/sweep_driver.sh prism 1.131 2,3,4,5,6,7,8,9,10,11,12 1,2,3,4,6,8,12,all \
    10,2.5 validation/data/prism_3969/jw03969001001_04101_00001-seg00{1,2,3,4,5}_nrs1_uncal.fits
bash validation/onef/sweep_driver.sh soss 32.964 16,20,24,27.5,32,40,50 4,8,16,32,all 15.5 \
    validation/data/soss_uncal/jw01541001001_04101_00001-seg00{1,2,3,4}_nis_uncal.fits
python validation/onef/onef_sweep_summary.py validation/runs/onef-full/g395h/sweep_seg*.npz \
    --csv validation/onef/results/sweep_g395h.csv
```

The extracted inputs take ~10-20 GB per segment on disk; watch memory and disk while running.
