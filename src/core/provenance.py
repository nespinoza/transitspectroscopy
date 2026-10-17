"""Provenance helpers: file hashes and compact records of processing options."""

import hashlib
import os

import numpy as np


def file_sha256(path, block_size=8 * 1024**2):
    """SHA-256 hex digest of a file, read in blocks."""
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(block_size), b''):
            digest.update(block)
    return digest.hexdigest()


def option_record(options):
    """JSON-friendly copy of an options dictionary.

    Arrays are replaced by their shape, dtype and SHA-256, so large arrays are never
    embedded in headers; NumPy scalars and paths become plain Python values.
    """
    record = {}
    for key, value in options.items():
        if isinstance(value, np.ndarray):
            arr = np.ascontiguousarray(value)
            value = dict(
                shape=list(arr.shape),
                dtype=str(arr.dtype),
                sha256=hashlib.sha256(arr.tobytes()).hexdigest(),
            )
        elif isinstance(value, np.generic):
            value = value.item()
        elif isinstance(value, os.PathLike):
            value = os.fspath(value)
        record[key] = value
    return record


# ---------------------------------------------------------------------------------------
# Provenance manifests

# Distributions whose versions are recorded in every manifest.
SOFTWARE = (
    'transitspectroscopy',
    'numpy',
    'scipy',
    'astropy',
    'jwst',
    'stdatamodels',
    'crds',
    'stcal',
    'juliet',
)


def array_fingerprint(array):
    """Shape, dtype and SHA-256 of an array (used instead of embedding large arrays)."""
    arr = np.ascontiguousarray(array)
    return dict(
        shape=list(arr.shape),
        dtype=str(arr.dtype),
        sha256=hashlib.sha256(arr.tobytes()).hexdigest(),
    )


def software_versions():
    """Python and package versions (from installed metadata; nothing is imported)."""
    import platform
    from importlib import metadata

    versions = dict(python=platform.python_version(), platform=platform.platform())
    for name in SOFTWARE:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    try:
        from .._version import __version__

        versions['transitspectroscopy_source'] = __version__
    except ImportError:  # core imported on its own
        pass
    return versions


def git_state(path=None):
    """Commit and dirty flag of the git checkout containing ``path`` (None outside git)."""
    import subprocess
    from pathlib import Path

    folder = Path(path or __file__).resolve()
    folder = folder if folder.is_dir() else folder.parent
    try:
        commit = subprocess.run(
            ['git', 'rev-parse', 'HEAD'], cwd=folder, capture_output=True, text=True, timeout=10
        )
        if commit.returncode != 0:
            return None
        status = subprocess.run(
            ['git', 'status', '--porcelain', '--untracked-files=no'],
            cwd=folder,
            capture_output=True,
            text=True,
            timeout=10,
        )
        return dict(commit=commit.stdout.strip(), dirty=bool(status.stdout.strip()))
    except (OSError, subprocess.SubprocessError):
        return None


def _canonical(value):
    if isinstance(value, dict):
        return {str(k): _canonical(v) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}
    if isinstance(value, (list, tuple)):
        return [_canonical(v) for v in value]
    if isinstance(value, np.ndarray):
        return array_fingerprint(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, os.PathLike):
        return os.fspath(value)
    if hasattr(value, 'fingerprint'):  # e.g. a frozen SharedProfile
        return dict(type=type(value).__name__, fingerprint=value.fingerprint())
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def config_sha256(config):
    """Stable SHA-256 of a (nested) configuration dictionary."""
    import json

    text = json.dumps(_canonical(config), sort_keys=True, allow_nan=True)
    return hashlib.sha256(text.encode()).hexdigest()


def build_manifest(
    config=None, inputs=(), crds_context=None, seeds=None, hash_inputs=True, **extra
):
    """Provenance manifest of a reduction.

    Parameters
    ----------
    config : dict, optional
        Fully resolved configuration; stored, and hashed into ``config_sha256``.
    inputs : sequence of path
        Input files; their names, sizes and (if ``hash_inputs``) SHA-256 are recorded.
    crds_context : str, optional
        CRDS context of the calibration references.
    seeds : dict, optional
        Random seeds used.
    **extra
        Additional JSON-serializable entries (e.g. ``mode``, ``step``).
    """
    from datetime import datetime, timezone
    from pathlib import Path

    records = []
    for path in inputs:
        path = Path(path)
        record = dict(name=path.name, path=str(path.resolve()))
        if path.is_file():
            record['bytes'] = path.stat().st_size
            if hash_inputs:
                record['sha256'] = file_sha256(path)
        records.append(record)
    manifest = dict(
        manifest_version=1,
        created_utc=datetime.now(timezone.utc).isoformat(),
        software=software_versions(),
        git=git_state(),
        config=_canonical(config or {}),
        config_sha256=config_sha256(config or {}),
        inputs=records,
        crds_context=crds_context,
        seeds=seeds or {},
    )
    manifest.update(_canonical(extra))
    return manifest


def manifest_path(product_path):
    """Path of the manifest written next to a product file."""
    from pathlib import Path

    product_path = Path(product_path)
    return product_path.with_name(product_path.name + '.provenance.json')


def write_manifest(manifest, path):
    import json
    from pathlib import Path

    Path(path).write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str) + '\n')
    return path


def read_manifest(path):
    import json
    from pathlib import Path

    return json.loads(Path(path).read_text())


def check_manifest(manifest, strict=False, what='product'):
    """Compare a manifest with the running software.

    A different transitspectroscopy version or git commit is reported with a warning,
    or raised as ``ValueError`` if ``strict=True``. Returns the list of differences.
    """
    import warnings

    differences = []
    current = software_versions()
    recorded = manifest.get('software', {})
    for name in ('transitspectroscopy', 'jwst'):
        if recorded.get(name) != current.get(name):
            differences.append(f'{name} {recorded.get(name)} (now {current.get(name)})')
    old_git, new_git = manifest.get('git') or {}, git_state() or {}
    if old_git.get('commit') and new_git.get('commit') and old_git['commit'] != new_git['commit']:
        differences.append(f"git commit {old_git['commit'][:10]} (now {new_git['commit'][:10]})")
    if differences:
        message = f'{what} was produced by different code: ' + '; '.join(differences)
        if old_git.get('dirty'):
            message += ' (written from a checkout with uncommitted changes)'
        if strict:
            raise ValueError(message)
        warnings.warn(message)
    return differences
