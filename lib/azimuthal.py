import numpy as np
from functools import wraps


__all__ = 'aziavg azistd azimedian azimad'.split()


def azimuthaloperator(func):
    @wraps(func)
    def wrappedoperator(data, center, *args, **kwargs):
        x_p, y_p = center
        ny, nx = data.shape
        x = np.arange(nx) - x_p
        y = np.arange(ny) - y_p

        d = data.ravel()
        r = np.hypot.outer(y, x).astype(int).ravel()
        return func(d, r, *args, **kwargs)

    return wrappedoperator


def docstring(purpose):
    parameters = '''
    Parameters
    ----------
    data: numpy.ndarray
        Two-dimensional data set
    center: (x, y)
        Center of azimuthal average

    Returns
    -------'''

    def _doc(func):
        outcome = func.__doc__
        func.__doc__ = f'{purpose}\n{parameters}{outcome}'
        return func
    return _doc


@azimuthaloperator
@docstring('Azimuthal average')
def aziavg(d, r):
    '''
    avg: ndarray
        Average value of data as a function of distance from center
    '''
    nr = np.bincount(r)
    return np.bincount(r, d) / nr


@azimuthaloperator
@docstring('Azimuthal standard deviation')
def azistd(d, r):
    '''
    avg, std: tuple of numpy.ndarray
        Azimuthal average and
        azimuthal standard deviation
        as functions of distance from center
    '''
    nr = np.bincount(r)
    avg = np.bincount(r, d) / nr
    std = np.sqrt(np.bincount(r, (d - avg[r])**2) / nr)
    return avg, std


@azimuthaloperator
@docstring('Azimuthal median')
def azimedian(d, r):
    '''
    med: numpy.ndarray
        Median value as a function of distance from center
    '''
    med = [np.median(d[np.where(r == n)]) for n in np.arange(r.max())]
    return np.array(med)


@azimuthaloperator
@docstring('Azimuthal median absolute deviation')
def azimad(d, r):
    '''
    med, mad: tuple of numpy.ndarray
        Azimuthal median and
        azimuthal median absolute deviation
        as functions of distance from center
    '''
    radii = np.arange(r.max())
    med = np.empty_like(radii)
    mad = np.empty_like(radii)
    for n in radii:
        dn = d[np.where(r == n)]
        med[n] = np.median(dn)
        mad[n] = np.abs(dn - med[n])
    return med, mad
