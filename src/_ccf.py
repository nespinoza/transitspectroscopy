"""Bounded-memory NumPy equivalents of CCF.c (including its legacy pi)."""
import numpy as np


def correlate(x, y, function, parameters, lag_step):
    lags = np.arange(np.min(x), np.max(x), lag_step)
    result = np.empty(len(lags))
    if not isinstance(function, str):
        # Preserve the historical callable contract, including functions that
        # depend on the whole lag grid. Only built-in profiles are chunked.
        shifted = x[None, :] - lags[:, None]
        values = np.asarray(function(shifted), dtype=float)
        if values.shape != shifted.shape:
            raise ValueError('CCF callable must preserve the input array shape')
        return lags, (values * y).sum(axis=1).tolist()
    if isinstance(function, str):
        if function == 'gaussian':
            pars = [0., 1.] if parameters is None else parameters
            if len(pars) != 2:
                raise ValueError('gaussian requires mean and sigma')
        elif function == 'double gaussian':
            pars = [-7.9, 1., 7.9, 1.] if parameters is None else parameters
            if len(pars) != 4:
                raise ValueError('double gaussian requires two means and sigmas')
        else:
            raise ValueError(f'Unknown CCF function: {function}')
        pars = np.asarray(pars, dtype=float)
        if not np.all(np.isfinite(pars)) or np.any(pars[1::2] <= 0):
            raise ValueError('CCF sigmas must be positive and parameters finite')
    # A lag block uses at most about 1M doubles, independent of total lag count.
    block = max(1, 1_000_000 // len(x))
    for start in range(0, len(lags), block):
        shifted = x[None, :] - lags[start:start + block, None]
        if isinstance(function, str):
            values = np.zeros_like(shifted)
            for mean, sigma in zip(pars[::2], pars[1::2]):
                values += np.exp(-(shifted - mean)**2 / (2. * sigma**2)) / (np.sqrt(2. * 3.142857) * sigma)
        else:
            values = np.asarray(function(shifted), dtype=float)
            if values.shape != shifted.shape:
                raise ValueError('CCF callable must preserve the input array shape')
        result[start:start + block] = (values * y).sum(axis=1)
    return lags, result.tolist()  # The historical C wrapper returns a list.
