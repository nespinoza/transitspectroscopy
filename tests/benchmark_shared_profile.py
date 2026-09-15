"""Measure real shared-profile fitting, including inner solves and curvature.

One configuration per process makes the reported driver peak RSS interpretable.
Examples in tests/README.md. Ray process memory is not included in driver RSS.
"""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import resource
import signal
import time

import _support  # noqa: F401
import numpy as np
import scipy
from transitspectroscopy import fit_shared_profile
from transitspectroscopy import shared_profile
from simulated_spectra import simulate


class Images:
    def __init__(self, source, key): self.source, self.key = source, key
    def __len__(self): return self.source.count
    def __getitem__(self, index): return self.source.get(index)[self.key]


class Source:
    def __init__(self, count, rows, columns):
        self.count, self.rows, self.columns = count, rows, columns
        self.previous, self.case = None, None

    def get(self, index):
        if index != self.previous:
            self.case = simulate(seed=5000+index,nrows=self.rows,ncolumns=self.columns,
                                 phase=.6*np.sin(index),flux_scale=10000.)
            self.previous = index
        return self.case


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--integrations',type=int,default=10)
    parser.add_argument('--columns',type=int,default=2048)
    parser.add_argument('--rows',type=int,default=256)
    parser.add_argument('--aperture',type=int,default=7)
    parser.add_argument('--spacing',type=float,default=1.)
    parser.add_argument('--method',choices=['gp','polynomial'],default='gp')
    parser.add_argument('--inducing',type=int,default=16)
    parser.add_argument('--workers',type=int,default=0)
    parser.add_argument('--optimize',action='store_true')
    parser.add_argument('--max-seconds',type=float,default=300.,help='Training wall-time limit; timeout is recorded')
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    implementation_hash = hashlib.sha256(Path(shared_profile.__file__).read_bytes()).hexdigest()
    source = Source(args.integrations,args.rows,args.columns)
    # Trace generation needs no full image cube.
    x = np.arange(args.columns)/(args.columns-1)
    base = args.rows/2+2*(x-.5)+.4*np.sin(2*np.pi*x)
    traces = np.array([base+.6*np.sin(i) for i in range(args.integrations)])
    start = time.perf_counter()
    def deadline(signum, frame):
        raise TimeoutError('benchmark training exceeded its wall-time limit')
    previous = signal.signal(signal.SIGALRM,deadline)
    signal.setitimer(signal.ITIMER_REAL,args.max_seconds)
    model = None
    failure = None
    try:
        model = fit_shared_profile(Images(source,'data'), Images(source,'variance'), traces,
                                   aperture_radius=args.aperture,spacing=args.spacing,
                                   profile_method=args.method,
                                   gp_options={'n_inducing':args.inducing} if args.method == 'gp' else None,
                                   optimize_hyperparameters=args.optimize, n_starts=1,
                                   execution='ray' if args.workers else 'serial',n_workers=max(1,args.workers))
    except (TimeoutError, RuntimeError) as exc:
        failure = dict(type=type(exc).__name__,message=str(exc))
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        signal.signal(signal.SIGALRM,previous)
    elapsed = time.perf_counter()-start
    extraction_seconds = None
    if model is not None:
        sample = source.get(0)
        start = time.perf_counter()
        for _ in range(10): model.extract(sample['data'],sample['centers'],sample['variance'])
        extraction_seconds = (time.perf_counter()-start)/10
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rss *= 1 if platform.system() == 'Darwin' else 1024
    payload = dict(configuration={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
                   platform=platform.platform(),numpy=np.__version__,scipy=scipy.__version__,
                   implementation_sha256=implementation_hash,
                   status=('converged' if model is not None else
                           'timeout' if failure['type'] == 'TimeoutError' else 'failed'),
                   failure=failure,
                   fitting_seconds=elapsed if model is not None else None,
                   elapsed_seconds=elapsed,extraction_seconds_per_integration=extraction_seconds,
                   driver_peak_rss_mb=rss/1e6,
                   memory_note='Driver peak RSS; excludes Ray workers. Synthetic images generated lazily during preparation.',
                   diagnostics=model.diagnostics if model is not None else None)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(payload,indent=2)+'\n')
    print(json.dumps({k:v for k,v in payload.items() if k != 'diagnostics'},indent=2))


if __name__ == '__main__': main()
