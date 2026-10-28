"""Package SOSS stage 2 (ts.soss.extract_soss: background step, PASTASOSS traces, box
extraction with the mode's aperture) of all the chunks of one variant at once, so that the
traces and background are the same for every integration.
  python wasp39_soss_stage2.py <folder with seg*/<variant>_*.fits> <variant> <out prefix>
                                [optimal]
Writes <out prefix>_order1.npz and _order2.npz (times, wavelengths, spectra): box fluxes,
with the columns where the orders overlap set to NaN, or with 'optimal' the joint (deblended)
polynomial optimal extraction of all columns.
"""
import os, sys, time, glob
from pathlib import Path
import numpy as np
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO/'tests'))
import _support  # noqa: F401
os.environ.setdefault('CRDS_PATH', str(Path.home()/'crds_cache'))
os.environ.setdefault('CRDS_SERVER_URL', 'https://jwst-crds.stsci.edu')
import transitspectroscopy as ts

folder, variant, prefix = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
start = time.monotonic()
files = sorted(glob.glob(str(folder/'seg*'/f'{variant}_[0-9]*.fits')))
optimal = len(sys.argv) > 4 and sys.argv[4] == 'optimal'
if optimal:
    # Orders 1 and 2 overlap at the red end of order 1 (traces 20-40 px apart): joint
    # (deblended) polynomial optimal extraction of the overlapping orders.
    result = ts.soss.extract_soss(files, outputfolder=prefix + '_products',
                                  optimal_methods=('polynomial',))
    products = result['optimal']['polynomial']['orders']
else:
    result = ts.soss.extract_soss(files, outputfolder=prefix + '_products')
    products = result['orders']
for order in (1, 2):
    trace = products[order]
    flux = np.array(trace['flux'], float)
    if not optimal:
        flux[:, ~np.asarray(result['orders'][order]['isolated'], bool)] = np.nan
    np.savez(f'{prefix}_order{order}.npz', times=np.asarray(result['times']),
             wavelengths=np.asarray(trace['wavelength'], float), spectra=flux,
             spectra_err=np.array(trace['error'], float))
print(f'{variant}: {len(files)} chunks, {len(result["times"])} integrations, '
      f'{time.monotonic() - start:.0f} s', flush=True)
