# Pinned SOSS validation references

The executed run uses references selected by the locally cached
`jwst_1584.pmap` mappings for NIRISS/NIS, CLEAR/GR700XD, NIS_SOSS on 2022-06-08:

| Reference | Mapping | SHA-256 |
|---|---|---|
| `jwst_niriss_pastasoss_0003.asdf` | `jwst_niriss_pastasoss_0006.rmap` | `c9b5aa03f76acb80af59b13952ddc88f855ee06fe68765dba7b7651e0931700f` |
| `jwst_niriss_bkg_0033.fits` | `jwst_niriss_bkg_0005.rmap` | `9110222d8c7558031df491673bc20023f315ab043920ad790a9ff496865a47aa` |

They were downloaded directly from CRDS and passed as explicit overrides to
avoid modifying an external cache. The binaries are git-ignored (approximately
258 KB and 103 MB). Download them here if reproducing the pinned run:

```sh
curl -L --fail https://jwst-crds.stsci.edu/unchecked_get/references/jwst/jwst_niriss_pastasoss_0003.asdf \
  -o validation/references/soss/jwst_niriss_pastasoss_0003.asdf
curl -L --fail https://jwst-crds.stsci.edu/unchecked_get/references/jwst/jwst_niriss_bkg_0033.fits \
  -o validation/references/soss/jwst_niriss_bkg_0033.fits
```

These reference choices do not imply that the archive rateints were originally
calibrated with the same context. Their Stage 1 provenance is JWST 2.0.1,
`jwst_1535.pmap`, recorded separately in the validation manifest. Normal library
use selects references from the user's configured CRDS context and the science
metadata rather than hardcoding these filenames.
