"""Stage 1 of a chunk of integrations of one segment for the group-number linearity study.

Package default flow (load -> detector_calibration (TSO jump) -> fit_ramps, group 1/f
correction before linearity), intercepted where the corrected ramps are calibrated from the
linearity step on (`load._ramps_after_linearity`):
  rawdark_<first>.npy   float32 (integration, group, row, column): 1/f-corrected ramps before
                        linearity, dark subtracted (the input of the self correction)
  groupdq_<first>.npy   uint8: the group DQ the package keeps (saturation, jumps)
  template_<first>.fits one-integration RampModel with the metadata, pixel DQ and int_times
  crds_n<N>_<first>.fits rates per integration of the CRDS-linearized ramps fitted with the
                        first N groups only (later groups DO_NOT_USE), JWST ramp fit as the
                        package; variance arrays dropped (stage2 does not use them)
  linearity_<first>.txt the CRDS linearity reference file
  ramp_fit_<first>.json the package's ramp-fit step parameters
  python nl_groups_stage1.py <uncal> <folder> <first> <last> <N1,N2,...>
"""
import os, sys, json, time, resource, shutil
from pathlib import Path
import numpy as np
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO/'tests'))
import _support  # noqa: F401
os.environ.setdefault('CRDS_PATH', str(Path.home()/'crds_cache'))
os.environ.setdefault('CRDS_SERVER_URL', 'https://jwst-crds.stsci.edu')
import transitspectroscopy as ts
from transitspectroscopy import jwst as tj
from transitspectroscopy.jwst_models import mark_group_1f
from jwst.pipeline import calwebb_detector1
from jwst.stpipe.core import JwstStep
from stdatamodels.jwst import datamodels

# Steps called alone deep-copy their input; keep one copy of the ramps, as inside a pipeline:
_prepare_output = JwstStep.prepare_output


def prepare_output(self, init, make_copy=None, **kwargs):
    return _prepare_output(self, init, make_copy=False, **kwargs)


JwstStep.prepare_output = prepare_output

uncal, folder = sys.argv[1], Path(sys.argv[2])
first, last = int(sys.argv[3]), int(sys.argv[4])
groups = [int(n) for n in sys.argv[5].split(',')]
folder.mkdir(parents=True, exist_ok=True)
tag = f'_{first:05d}'
DO_NOT_USE = 1


def write_chunk(uncal, path, first, last):
    """The integrations first..last-1 of the uncal file, as an uncal file of their own."""
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


original = tj.load._ramps_after_linearity


def intercept(self, groupdq):
    assert len(self.ramps_per_segment) == 1
    ramp = self.ramps_per_segment[0]
    # Dark-subtracted raw ramps (the dark step on a copy, before linearity):
    dark = calwebb_detector1.dark_current_step.DarkCurrentStep.call(ramp.copy())
    np.save(folder/f'rawdark{tag}.npy', np.asarray(dark.data, np.float32))
    del dark
    np.save(folder/f'groupdq{tag}.npy', np.asarray(groupdq[0], np.uint8))
    original(self, groupdq)
    raise Done


work = folder/f'work{tag}'
work.mkdir(exist_ok=True)
name = Path(uncal).name
write_chunk(uncal, work/name, first, last)
dataset = ts.jwst.load([str(work/name)], outputfolder=str(work))
tj.load._ramps_after_linearity = intercept
start = time.monotonic()
try:
    dataset.detector_calibration(use_tso_jump=True, save=False)
    dataset.fit_ramps(save=False)
except Done:
    pass
else:
    raise RuntimeError('the group 1/f correction was not applied before linearity')
finally:
    tj.load._ramps_after_linearity = original
print(f'stage 1: {time.monotonic() - start:.0f} s', flush=True)

ramp = dataset.ramps_per_segment[0]
with open(folder/f'linearity{tag}.txt', 'w') as f:
    f.write(ramp.meta.ref_file.linearity.name)
template = datamodels.RampModel(data=np.asarray(ramp.data[:1]), groupdq=np.asarray(ramp.groupdq[:1]),
                                pixeldq=np.asarray(ramp.pixeldq))
template.update(ramp, only=None)
template.int_times = ramp.int_times
template.save(folder/f'template{tag}.fits')
parameters = dict(dataset.calibration_parameters['ramp_fit'])
for key in ['output_dir', 'save_results', 'suffix']:
    parameters.pop(key, None)
with open(folder/f'ramp_fit{tag}.json', 'w') as f:
    json.dump(parameters, f)
groupdq = np.array(ramp.groupdq)
for n in groups:
    ramp.groupdq = groupdq.copy()
    ramp.groupdq[:, n:] |= DO_NOT_USE
    rateints = calwebb_detector1.ramp_fit_step.RampFitStep.call(ramp, **parameters)[1]
    mark_group_1f(rateints)
    for array in ('var_poisson', 'var_rnoise', 'var_flat'):
        if rateints.hasattr(array):
            delattr(rateints, array)
    rateints.save(folder/f'crds_n{n}{tag}.fits')
    rateints.close()
    print(f'CRDS, {n} groups: {time.monotonic() - start:.0f} s', flush=True)
shutil.rmtree(work, ignore_errors=True)
print(f'done: {time.monotonic() - start:.0f} s, peak '
      f'{resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**30:.1f} GB')
