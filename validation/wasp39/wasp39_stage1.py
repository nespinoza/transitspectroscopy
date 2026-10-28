"""Stage 1 of a chunk of integrations of one segment (WASP-39 b, ERS 1366), package flow.

load -> detector_calibration (TSO jump) -> fit_ramps (group 1/f correction before linearity),
with the CRDS linearity correction or a self correction of given coefficients (variant
'self', or any other name, e.g. 'hybrid'):
  python wasp39_stage1.py <uncal> <folder> <first> <last> crds
  python wasp39_stage1.py <uncal> <folder> <first> <last> <variant> <coefficients.npz>
  python wasp39_stage1.py <uncal> <folder> <first> <last> fit
The first two save the rates per integration as <variant>_<first>.fits (variance arrays
dropped; stage 2 does not use them; one-group ramps kept, suppress_one_group = False); 'fit' runs detector_calibration(self_linearity=True,
stop_after_fit) and keeps the saved fit (mean ramps and counts) as selffit_<first>.npz.

Only integrations first..last-1 are calibrated (memory). The group 1/f template of a chunk
holds the median 1/f pattern of its integrations, which stays in the corrected ramps (times
the white-light factor); so that it is the same in all chunks, each chunk's template is
aligned to the first chunk's (../template_reference_<variant>.npy, shared by the segment
folders): the column median over
the background pixels of their difference is removed (the static scene cancels in it).
"""
import os, sys, time, resource, shutil, warnings
from pathlib import Path
import numpy as np
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO/'tests'))
import _support  # noqa: F401
os.environ.setdefault('CRDS_PATH', str(Path.home()/'crds_cache'))
os.environ.setdefault('CRDS_SERVER_URL', 'https://jwst-crds.stsci.edu')
import transitspectroscopy as ts
from transitspectroscopy.core import detector as core_detector
from stdatamodels.jwst import datamodels

uncal, folder = sys.argv[1], Path(sys.argv[2])
first, last, variant = int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
folder.mkdir(parents=True, exist_ok=True)
tag = f'_{first:05d}'
reference_file = folder.parent/f'template_reference_{variant}.npy'  # all segments


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


def group_difference_1f_aligned(segments, valid, background_pixels, scale, scale_pixels=None,
                                template_valid=None, shared_fraction=0.5, template='median'):
    """core.detector.group_difference_1f (median templates), templates aligned between chunks."""
    assert template == 'median'
    ngroups = segments[0].shape[1]
    template_valid = valid if template_valid is None else template_valid
    factors = np.asarray(scale, dtype=np.float32)[:, None, None]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        templates = np.array([
            np.nanmedian(np.concatenate([np.where(good[:, k], data[:, k], np.nan)
                                         for data, good in zip(segments, template_valid)])
                         .astype(np.float32)/factors, axis=0) for k in range(ngroups)])
        if reference_file.exists():
            reference = np.load(reference_file)
            offset = np.nanmedian(np.where(background_pixels, templates - reference, np.nan),
                                  axis=1)
            templates -= np.where(np.isfinite(offset), offset, 0.)[:, None, :]
            print(f'group 1/f: template aligned (column offsets rms '
                  f'{np.sqrt(np.nanmean(offset**2)):.2f} DN)', flush=True)
        else:
            np.save(reference_file, templates.astype(np.float32))
            print('group 1/f: reference template saved', flush=True)
        levels = np.zeros((len(scale), ngroups, segments[0].shape[3]))
        shared = np.zeros((len(scale), ngroups), bool)
        integration = 0
        for data, good, clean in zip(segments, valid, template_valid):
            for i in range(len(data)):
                for k in range(ngroups):
                    residual = data[i, k] - scale[integration]*templates[k]
                    use = good[i, k] & background_pixels
                    unflagged = use & clean[i, k]
                    excluded = use.sum() - unflagged.sum()
                    shared[integration, k] = excluded > shared_fraction*max(use.sum(), 1)
                    if not shared[integration, k]:
                        use = unflagged
                    level = np.nanmedian(np.where(use, residual, np.nan), axis=0)
                    level = np.where(np.isfinite(level), level, 0.0)
                    data[i, k] -= level[None, :]
                    levels[integration, k] = level
                integration += 1
    return dict(levels=levels, shared=shared)


core_detector.group_difference_1f = group_difference_1f_aligned

work = folder/f'work{tag}'
work.mkdir(exist_ok=True)
name = Path(uncal).name
write_chunk(uncal, work/name, first, last)
start = time.monotonic()
dataset = ts.jwst.load([str(work/name)], outputfolder=str(work))
if variant == 'fit':
    dataset.detector_calibration(self_linearity=True, save=True,
                                 parameters={'self_linearity': {'stop_after_fit': True}})
    shutil.move(dataset.self_linearity_file, folder/f'selffit{tag}.npz')
else:
    # Pixels that saturate after the first group in some integrations keep a rate from that
    # group (the JWST default drops one-group ramps, so their flux would come and go):
    options = {'parameters': {'ramp_fit': {'suppress_one_group': False}}}
    if variant != 'crds':
        options['self_linearity_coefficients'] = sys.argv[6]
    dataset.detector_calibration(save=False, **options)
    dataset.fit_ramps(save=False)
    rateints = dataset.rateints_per_segment[0]
    for array in ('var_poisson', 'var_rnoise', 'var_flat'):
        if rateints.hasattr(array):
            delattr(rateints, array)
    rateints.save(folder/f'{variant}{tag}.fits')
shutil.rmtree(work, ignore_errors=True)
print(f'{variant} {first}-{last}: {time.monotonic() - start:.0f} s, peak '
      f'{resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**30:.1f} GB', flush=True)
