"""Audit paired MIRI products and add reproducible numerical diagnostics."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
from astropy.io import fits

from validate import ROOT, sha256
from diagnostics import write_json


def audit(folder):
    manifest = json.loads((folder/'comparison_manifest.json').read_text())
    assert manifest['status'] == 'complete'
    original = Path(manifest.get('original_run', folder))
    groups = defaultdict(list)
    hashes = 0
    for row in manifest['chunks']:
        groups[row['stem']].append(row)
        for method in ('jwst', 'sparta'):
            file = folder/method/'prepared'/f'{row["stem"]}_{row["start"]:05d}.npz'
            assert sha256(file) == row['methods'][method]['prepared_sha256'], file
            hashes += 1
            with np.load(file) as cutout:
                assert len(cutout['times']) == row['stop']-row['start']
                assert np.all(np.diff(cutout['times']) > 0)
    for name, rows in groups.items():
        assert rows[0]['start'] == 0
        assert all(a['stop'] == b['start'] for a,b in zip(rows, rows[1:])), name
    # Preserve original driver provenance after its short-tail handling was fixed.
    for name, checksum in manifest['source_hashes'].items():
        file = ROOT/name
        if sha256(file) != checksum:
            archived = original/'source'/Path(name).name
            assert archived.exists() and sha256(archived) == checksum, name
    for name, checksum in manifest.get('repair_source_hashes', {}).items():
        assert sha256(ROOT/name) == checksum, name
    output = dict(status='passed', cutout_hashes_verified=hashes, segments=len(groups),
                  integration_count=sum(r['stop']-r['start'] for r in manifest['chunks']),
                  nonconverged_sparta_ramps=sum(r['methods']['sparta']['nonconverged_ramps'] for r in manifest['chunks']),
                  visits={}, fit_runtime_seconds={})
    for method in ('jwst', 'sparta'):
        times = [r['methods'][method]['fit_seconds'] for r in manifest['chunks']]
        output['fit_runtime_seconds'][method] = dict(sum=float(sum(times)), median_chunk=float(np.median(times)))
    for visit in manifest['config']['visits']:
        expected = sum(rows[-1]['stop'] for stem, rows in groups.items()
                       if any(stem.startswith(e+'-seg') for e in manifest['config']['visits'][visit]['exposures']))
        arrays = {}
        entries = {}
        for method in ('jwst', 'sparta'):
            base = folder/method/f'visit{visit}'
            with np.load(base/'photometry.npz') as data:
                arrays[method] = {k: data[k] for k in ('times','flux','error','flags','segment','profile')}
            a = arrays[method]
            assert len(a['times']) == expected
            assert np.all(np.diff(a['times']) > 0)
            assert np.isclose(a['profile'].sum(), 1., rtol=1.e-6)
            with fits.open(base/'photometry.fits') as h:
                for hdu, key in [('SCI','flux'), ('ERR','error'), ('TIME','times'), ('DQ','flags')]:
                    np.testing.assert_allclose(h[hdu].data, a[key], equal_nan=True)
                assert h[0].header['BUNIT'] == 'Jy'
            metadata = json.loads((base/'photometry_metadata.json').read_text())
            assert metadata['gp']['success']
            entries[method] = dict(integrations=expected, gp_training_success=True, methods={})
            for j, extraction in enumerate(('sum','gp')):
                table = np.loadtxt(base/f'{extraction}_lightcurve.csv', delimiter=',', skiprows=1)
                np.testing.assert_allclose(table[:,0], a['times'], rtol=0, atol=0)
                np.testing.assert_allclose(table[:,1], a['flux'][:,j], equal_nan=True)
                np.testing.assert_allclose(table[:,2], a['error'][:,j], equal_nan=True)
                filtered = np.loadtxt(base/f'{extraction}_filtered_lightcurve.csv', delimiter=',', skiprows=1)
                np.testing.assert_allclose(filtered[:,4], (filtered[:,1]-filtered[:,3])*1.e6, equal_nan=True)
                residual = filtered[:,4]
                info = dict(valid_integrations=int(np.isfinite(residual).sum()),
                    rms_ppm=float(np.sqrt(np.nanmean(residual**2))),
                    robust_sigma_ppm=float(1.4826*np.nanmedian(abs(residual-np.nanmedian(residual)))),
                    median_formal_error_ppm=float(np.nanmedian(filtered[:,2])*1.e6))
                saved = json.loads((base/'metrics.json').read_text())['methods'][extraction]
                np.testing.assert_allclose(info['rms_ppm'], saved['median_filter_residual_rms_ppm'])
                bins = np.loadtxt(base/f'{extraction}_binned_lightcurve.csv', delimiter=',', skiprows=1)
                assert int(bins[:,3].sum()) == info['valid_integrations']
                entries[method]['methods'][extraction] = info
        np.testing.assert_array_equal(arrays['jwst']['times'], arrays['sparta']['times'])
        for j, extraction in enumerate(('sum', 'gp')):
            paired = np.loadtxt(folder/f'visit{visit}_{extraction}_paired.csv', delimiter=',', skiprows=1)
            common = paired[:,1].astype(bool)
            np.testing.assert_array_equal(paired[:,0], arrays['jwst']['times'])
            np.testing.assert_allclose(paired[:,6], (paired[:,3]-paired[:,2])*1.e6, equal_nan=True)
            for name, column in (('jwst',4),('sparta',5)):
                expected_rms = manifest['comparison'][visit][extraction][name]['rms_ppm']
                np.testing.assert_allclose(np.sqrt(np.mean(paired[common,column]**2)), expected_rms)
        output['visits'][visit] = entries
    write_json(folder/'audit.json', output)
    products = [dict(path=str(p.relative_to(folder)), bytes=p.stat().st_size, sha256=sha256(p))
                for p in sorted(folder.rglob('*')) if p.is_file() and p.name != 'product_manifest.json']
    write_json(folder/'product_manifest.json', dict(products=products))
    print(json.dumps(output, indent=2))
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path)
    audit(parser.parse_args().folder.resolve())
