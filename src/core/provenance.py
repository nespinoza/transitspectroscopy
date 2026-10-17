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
