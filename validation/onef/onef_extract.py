"""Inputs of the group 1/f correction of one segment, compactly (package default flow: load ->
detector_calibration (TSO jump) -> fit_ramps, with the source checkout).

The call of jwst.group_difference_1f_from_traces inside fit_ramps is intercepted and saved:
  reference.npy   float32 (group, row, column): median of the first 20 integrations
  delta.npy       float16 (integration, group, row, column): ramps (before linearity, as the
                  correction sees them) minus the reference, clipped to +-65000
  masks.npy       uint8: bit 0 valid, bit 1 template_valid
  rates.npy       float16 (integration, row, column): the first-pass rates (NaN kept)
  rates_bad.npy   bool: rates with DO_NOT_USE or SATURATED
  geometry.npz    traces (order, columns, centers), median, background, settings (json),
                  and the package's levels and white-light factors for this segment
The run stops after the correction.
  python onef_extract.py <uncal> <folder> <first> <last>
Only integrations first..last-1 of the segment are calibrated (a chunk uncal is written to the
folder and removed), so that Stage 1 fits in memory; outputs carry the suffix _<first>.
"""
import os, sys, time, json, resource, shutil
from pathlib import Path
import numpy as np
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO/'tests'))
import _support  # noqa: F401
os.environ.setdefault('CRDS_PATH', str(Path.home()/'crds_cache'))
os.environ.setdefault('CRDS_SERVER_URL', 'https://jwst-crds.stsci.edu')
import transitspectroscopy as ts
from transitspectroscopy import jwst as tj
from jwst.stpipe.core import JwstStep

# Steps called alone deep-copy their input (two copies of the ramps during every step); the
# wrapper keeps only the newest ramps, so they can work in place, as inside a pipeline:
_prepare_output = JwstStep.prepare_output


def prepare_output(self, init, make_copy=None, **kwargs):
    return _prepare_output(self, init, make_copy=False, **kwargs)


JwstStep.prepare_output = prepare_output
from transitspectroscopy.jwst_models import open_model, ramp_1f_validity

uncal, folder = sys.argv[1], Path(sys.argv[2])
first, last = int(sys.argv[3]), int(sys.argv[4])
folder.mkdir(parents=True, exist_ok=True)
tag = f'_{first:05d}'


def write_chunk(uncal, path, first, last):
    """The integrations first..last-1 of the uncal file, as an uncal file of their own."""
    from stdatamodels.jwst import datamodels
    with datamodels.Level1bModel(uncal) as model:
        chunk = datamodels.Level1bModel(data=np.asarray(model.data[first:last]))
        chunk.update(model, only=None)
        for name in ('group', 'int_times'):
            table = getattr(model, name, None)
            if table is not None and len(table):
                setattr(chunk, name, table[first:last] if name == 'int_times' else table)
        start = model.meta.exposure.integration_start or 1
        chunk.meta.exposure.integration_start = start + first
        chunk.meta.exposure.integration_end = start + last - 1
        chunk.meta.filename = Path(path).name
        chunk.save(path)
        chunk.close()


class Done(Exception):
    pass


original = tj.group_difference_1f_from_traces


def intercept(ramps, rateints, traces, median, background, settings, **kwargs):
    assert len(ramps) == 1
    data = ramps[0].data
    reference = np.nanmedian(np.asarray(data[:20], np.float32), axis=0)
    np.save(folder/f'reference{tag}.npy', reference)
    delta = np.lib.format.open_memmap(folder/f'delta{tag}.npy', 'w+', np.float16, data.shape)
    for i in range(len(data)):
        delta[i] = np.clip(np.asarray(data[i], np.float32) - reference, -65000, 65000)
    delta.flush()
    del delta
    valid, template_valid = ramp_1f_validity(ramps)
    np.save(folder/f'masks{tag}.npy',
            valid[0].astype(np.uint8) | (template_valid[0].astype(np.uint8) << 1))
    del valid, template_valid
    with open_model(rateints[0]) as model:
        np.save(folder/f'rates{tag}.npy', np.asarray(model.data, np.float16))
        np.save(folder/f'rates_bad{tag}.npy', (np.asarray(model.dq) & 3) > 0)
    diagnostics = original(ramps, rateints, traces, median, background, settings, **kwargs)
    orders = list(traces)
    np.savez(folder/f'geometry{tag}.npz', orders=np.array(orders),
             **{f'columns{o}': np.asarray(traces[o][0]) for o in orders},
             **{f'centers{o}': np.asarray(traces[o][1], float) for o in orders},
             median=np.asarray(median, np.float32), background=np.asarray(background, np.float32),
             settings=json.dumps(settings), levels=diagnostics['levels'].astype(np.float32),
             scale=diagnostics['white_light_scale'])
    raise Done


work = folder/f'work{tag}'
work.mkdir(exist_ok=True)
name = Path(uncal).name
write_chunk(uncal, work/name, first, last)
dataset = ts.jwst.load([str(work/name)], outputfolder=str(work))
tj.group_difference_1f_from_traces = intercept
start = time.monotonic()
try:
    dataset.detector_calibration(use_tso_jump=True, save=False)
    dataset.fit_ramps(save=False)
except Done:
    pass
else:
    raise RuntimeError('the group 1/f correction was not called')
finally:
    tj.group_difference_1f_from_traces = original
with open(folder/f'mode{tag}.txt', 'w') as f:
    f.write(dataset.mode)
shutil.rmtree(work, ignore_errors=True)
print(f'{dataset.mode}: {time.monotonic() - start:.0f} s, peak '
      f'{resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**30:.1f} GB')
